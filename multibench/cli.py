"""Command-line interface over the multibench Python API.

Every subcommand is a thin wrapper around one public Python function, with the
same parameter names where a flag exists (``--category``, ``--metrics``,
``--only``, ``--out-dir`` ...). The end-to-end story mirrors the Python one::

    multibench layout vertical                         # how to lay out MY data
    multibench convert my.h5ad data/MYCITE --rna X --adt obsm:protein --labels obs:celltype
    multibench scan MYCITE --category vertical --data-path data
    multibench run --method Matilda --category vertical --input rna=... --out-dir runs/Matilda
    multibench evaluate --output runs/Matilda/embedding.h5 --labels data/MYCITE/cty.csv \\
        --method Matilda --dataset MYCITE --category vertical --out runs/Matilda/long.csv
    multibench plot bubble --category vertical --dataset D11 --out fig.pdf
    multibench cite Matilda MOFA2

Exit codes and streams
----------------------
``0`` success; ``1`` a runtime error raised by the API (the message is printed
as ``error: ...`` on stderr; set ``MULTIBENCH_DEBUG=1`` to get the traceback);
``2`` a usage error (argparse: unknown flag, missing required flag, bad
choice, or a flag combination the subcommand rejects).

DATA goes to stdout (tables, ids, commands, yml, citations, ``wrote ...``
lines); DIAGNOSTICS go to stderr (``error: ...``, ``warning: ...``, progress
such as ``[run_all] ...`` and ``# dry run ...`` notes), so
``multibench scan ... --format tsv > plan.tsv`` captures a clean table and
``2>/dev/null`` silences the chatter. Tables default to a compact column set
(``--columns all`` for everything, ``--format csv|tsv|json`` for scripts).

``multibench <command> --help`` documents every flag of every command.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
import warnings
from pathlib import Path

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_USAGE = 2


# ----------------------------------------------------------------- helpers
def _csv_list(s) -> list[str] | None:
    """Split a comma-separated flag value into a stripped list (``None`` stays ``None``).

    ``"ARI, NMI"`` -> ``['ARI', 'NMI']``; empty items are dropped; a value that is
    already a list is returned unchanged (``action="append"`` flags).
    """
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s if str(x).strip()]
    return [x.strip() for x in str(s).split(",") if x.strip()]


_TRI = {None: None, "true": True, "false": False}


def _tri_state(value) -> bool | None:
    """Map the ``--needs-labels [true|false]`` flag value to the Python tri-state.

    ``None`` (flag absent) -> ``None`` (no filter); ``"true"`` (the value, or
    the bare flag via ``const``) -> ``True``; ``"false"`` -> ``False``.
    """
    return _TRI[value]


#: The columns ``scan`` / ``run-all --dry-run`` print by default in table
#: mode: the verdict, the two gates and the (truncated) reason. The full
#: 17-column frame is ~1450 characters wide - unreadable on a terminal; it is
#: still there via ``--columns all`` or any machine format (csv/tsv/json).
_COMPACT_PLAN_COLUMNS = ["method", "modalities", "runnable", "files_ok", "env_ok",
                         "runtime_tier", "reason"]
#: Free-text columns clipped to this many characters in TABLE mode (never in
#: csv/tsv/json, never with an explicit ``--columns`` list).
_TRUNCATE_WIDTH = 80
_TRUNCATE_COLUMNS = ("reason", "files_reason", "env_reason", "caveat", "command", "error")


def _truncate(text, width: int = _TRUNCATE_WIDTH) -> str:
    """Clip ``text`` to ``width`` characters with a trailing ``...``."""
    t = "" if text is None else str(text)
    return t if len(t) <= width else t[: width - 3] + "..."


def _resolve_columns(df, columns, fmt: str, compact=None) -> list | None:
    """Which columns to print: an explicit list, ``all``, or the default.

    The default is the ``compact`` set in table mode (a readable terminal
    width) and EVERY column for csv/tsv/json (a script wants the data, not a
    pretty page). ``columns=["all"]`` is every column in any format. Compact
    columns absent from the frame are skipped silently (a mocked or older
    frame); an unknown explicit name is an error naming the available ones.
    """
    if columns and [c.lower() for c in columns] == ["all"]:
        return None
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"unknown column(s) {missing}; available: {list(df.columns)} "
                f"(or --columns all)")
        return list(columns)
    if compact and fmt == "table":
        picked = [c for c in compact if c in df.columns]
        return picked or None              # a frame with none of them: print it whole
    return None


def _print_frame(df, columns=None, fmt: str = "table", file=None, *,
                 compact=None, truncate: bool | None = None) -> None:
    """Print a DataFrame as an aligned table (default), CSV, TSV or JSON.

    ``columns`` restricts (and orders) the printed columns (``["all"]`` = every
    column); an unknown name is an error naming the columns the frame actually
    has, so a concurrently added column never needs a CLI change to be
    selectable. ``compact`` is the default column list for TABLE mode (csv/
    tsv/json print everything unless ``columns`` is given). ``truncate``
    clips the long free-text columns (:data:`_TRUNCATE_COLUMNS`) to
    :data:`_TRUNCATE_WIDTH` characters with ``...`` - by default only in table
    mode without an explicit ``columns`` list; machine formats are never
    clipped. JSON is a list of row objects (``orient="records"``).
    """
    file = sys.stdout if file is None else file
    picked = _resolve_columns(df, columns, fmt, compact)
    if picked is not None:
        df = df[picked]
    if fmt == "csv":
        print(df.to_csv(index=False), end="", file=file)
    elif fmt == "tsv":
        print(df.to_csv(index=False, sep="\t"), end="", file=file)
    elif fmt == "json":
        print(df.to_json(orient="records", indent=1, default_handler=str), file=file)
    else:
        if truncate is None:
            truncate = not columns
        if truncate:
            df = df.copy()
            for c in _TRUNCATE_COLUMNS:
                if c in df.columns:
                    df[c] = df[c].map(_truncate)
        if len(df) == 0:
            print(f"(empty table; columns: {list(df.columns)})", file=file)
        else:
            print(df.to_string(index=False), file=file)


def _quiet_stdout():
    """Route a library call's stdout chatter (``[run_all] ...`` progress,
    ``[env] unpacking ...``) to stderr so stdout stays data-only for pipes."""
    return contextlib.redirect_stdout(sys.stderr)


def _parse_scalar(text: str):
    """``--param`` VALUE -> Python scalar: ``5`` -> int, ``0.1``/``1e-3`` ->
    float, ``true``/``false`` -> bool, ``none``/``null`` -> None, a JSON
    list/object (``[1,2]``) -> that object, anything else -> the string."""
    t = text.strip()
    low = t.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    if t[:1] in "[{":
        try:
            return json.loads(t)
        except ValueError:
            pass
    return t


def _parse_params(pairs, args=None, default_method: str | None = None) -> dict:
    """Turn repeated ``--param [METHOD:]KEY=VALUE`` values into
    ``{METHOD: {KEY: value}}`` (the ``params=`` shape of ``run`` / ``run_all``).

    ``METHOD:`` is required for ``run-all`` (several methods) and optional for
    ``run`` (``default_method`` = ``--method``; a METHOD that names another
    method is a usage error). An unknown METHOD raises the did-you-mean
    ``KeyError`` of the registry (exit 1); a value without ``=`` is a usage
    error (exit 2). VALUE is parsed with :func:`_parse_scalar`.
    """
    from .engine import registry
    out: dict = {}
    for p in pairs or []:
        if "=" not in p:
            msg = (f"--param must be [METHOD:]KEY=VALUE, got {p!r} (e.g. "
                   f"--param Matilda:epochs=5)")
            if args is not None:
                _usage_error(args, msg)
            raise SystemExit(msg)
        key, value = p.split("=", 1)
        if ":" in key:
            method, key = key.split(":", 1)
        else:
            method = default_method
        method, key = (method or "").strip(), key.strip()
        if not method or not key:
            msg = (f"--param needs METHOD:KEY=VALUE with non-empty parts, got {p!r}"
                   + ("" if default_method else " (METHOD: is required for run-all)"))
            if args is not None:
                _usage_error(args, msg)
            raise SystemExit(msg)
        if default_method is not None and method != default_method:
            msg = (f"--param names method {method!r} but --method is "
                   f"{default_method!r}; drop the METHOD: prefix or make them agree")
            if args is not None:
                _usage_error(args, msg)
            raise SystemExit(msg)
        registry.check_method(method)          # KeyError with a did-you-mean hint
        out.setdefault(method, {})[key] = _parse_scalar(value)
    return out


def _packed_manifest() -> dict:
    """The ``{env: archive_url}`` map shipped as ``engine/packed_urls.json`` (``{}`` if absent).

    Only the KEYS are used by the CLI (is an archive published for this env?);
    the file carries no sizes, so no download size is ever printed.
    """
    from .engine import envs
    mf = Path(envs.__file__).parent / "packed_urls.json"
    if not mf.is_file():
        return {}
    try:
        data = json.loads(mf.read_text())
    except Exception:  # noqa: BLE001 - a broken manifest means "no archives"
        return {}
    return data if isinstance(data, dict) else {}


def _usage_error(args, message: str) -> None:
    """Report a usage error the way argparse does (usage line + message, exit 2)."""
    parser = getattr(args, "_parser", None)
    if parser is not None:
        parser.error(message)          # prints usage + "error: ..." and exits 2
    raise SystemExit(_EXIT_USAGE)


# ----------------------------------------------------------------- commands
def _cmd_list(args) -> int:
    """``multibench list``: method ids from the registry, one per line."""
    from .engine import registry
    for m in registry.list_methods(category=args.category, task=args.task,
                                   runnable=args.runnable or None):
        print(m)
    return _EXIT_OK


def _cmd_find(args) -> int:
    """``multibench find``: :func:`multibench.find_methods` with the same filters."""
    from . import discover
    for m in discover.find_methods(category=args.category, task=args.task,
                                   needs_labels=_tri_state(args.needs_labels),
                                   atac=args.atac,
                                   modalities=_csv_list(args.modalities),
                                   runnable=args.runnable or None,
                                   tunable=_tri_state(getattr(args, "tunable", None))):
        print(m)
    return _EXIT_OK


def _cmd_scan(args) -> int:
    """``multibench scan``: print :func:`multibench.scan` for one dataset.

    Whatever columns ``scan`` returns are printed (no hard-coded list); use
    ``--columns`` to pick some and ``--format csv`` to pipe into other tools.
    """
    import multibench
    from .engine import registry
    methods = _csv_list(args.methods)
    for m in methods or []:
        registry.check_method(m)               # did-you-mean KeyError before any I/O
    df = multibench.scan(args.dataset, args.category, data_path=args.data_path,
                         modalities=_csv_list(args.modalities))
    if methods:
        unknown = sorted(set(methods) - set(df["method"]))
        if unknown:
            raise ValueError(
                f"method(s) {unknown} are not in the scan table for "
                f"{args.dataset}/{args.category}; methods present: "
                f"{sorted(df['method'])}")
        df = df[df["method"].isin(methods)]
    _print_frame(df, columns=_csv_list(args.columns), fmt=args.format,
                 compact=_COMPACT_PLAN_COLUMNS)
    return _EXIT_OK


def _cmd_layout(args) -> int:
    """``multibench layout``: print :func:`multibench.describe_layout`."""
    import multibench
    print(multibench.describe_layout(args.category))
    return _EXIT_OK


_EXPORT_FLAGS = ("rna", "adt", "atac", "atac_kind", "labels", "batch")


def _cmd_convert(args) -> int:
    """``multibench convert``: one canonical ``.h5`` (``to_canonical``) or a whole
    dataset folder (``export_dataset``), chosen by the flags given.

    * No ``--rna/--adt/--atac/--labels/--batch`` flag: ``SRC`` -> ``OUT`` via
      :func:`multibench.io.to_canonical` (``--modality`` picks the filename when
      ``OUT`` is a directory; ``--layer/--obsm/--mod`` select the matrix).
    * Any of those flags: ``SRC`` is read (``.h5ad``/``.h5mu``) and
      :func:`multibench.io.export_dataset` writes ``OUT/`` as a dataset folder
      (``rna.h5``, ``adt.h5``, ``atac_*.h5``, ``cty.csv`` ...). ``--modality``,
      ``--layer``, ``--obsm`` and ``--mod`` are then usage errors: the selector
      grammar (``X``, ``obsm:<key>``, ``layer:<key>``, ``mod:<name>``) carries
      that information per modality.
    """
    from .engine import ingest
    export_mode = any(getattr(args, f) is not None for f in _EXPORT_FLAGS)
    if export_mode:
        clash = [f"--{f}" for f in ("modality", "layer", "obsm", "mod")
                 if getattr(args, f) is not None]
        if clash:
            _usage_error(args, f"{', '.join(clash)} cannot be combined with the "
                         "dataset-export flags (--rna/--adt/--atac/--labels/"
                         "--batch); put the selector in the flag value instead, "
                         "e.g. --adt obsm:protein")
        if args.rna is None and args.adt is None and args.atac is None \
                and args.labels is None:
            _usage_error(args, "dataset export needs at least one of --rna, "
                         "--adt, --atac, --labels (note: --rna has NO default "
                         "on the command line; pass --rna X to export adata.X)")
        data = ingest._to_anndata(args.src)
        p = ingest.export_dataset(data, args.out, rna=args.rna, adt=args.adt,
                                  atac=args.atac, atac_kind=args.atac_kind,
                                  labels=args.labels, batch=args.batch,
                                  dtype=args.dtype)
        print(f"wrote dataset folder {p} (files: "
              f"{', '.join(sorted(q.name for q in Path(p).iterdir()))})")
        return _EXIT_OK
    if args.layer is not None and args.obsm is not None:
        _usage_error(args, "--layer and --obsm are mutually exclusive")
    p = ingest.to_canonical(args.src, args.out, modality=args.modality,
                            layer=args.layer, obsm=args.obsm, mod=args.mod,
                            dtype=args.dtype)
    print(f"wrote {p}")
    return _EXIT_OK


def _cmd_cite(args) -> int:
    """``multibench cite``: :func:`multibench.cite` for the benchmark + methods."""
    from . import discover
    if args.all and args.methods:
        _usage_error(args, "--all and explicit method ids are mutually exclusive")
    methods = "all" if args.all else (args.methods or None)
    text = discover.cite(methods, fmt=args.format)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return _EXIT_OK


def _load_long_input(path) -> "pd.DataFrame":  # noqa: F821 - pandas imported lazily
    """Read a tidy long results frame from ``path``: a ``long.csv`` (or any CSV
    with ``metric,value,method`` columns) or a ``run_all`` output directory."""
    import pandas as pd
    p = Path(path)
    if p.is_dir():
        from .workflow import load_batch
        return load_batch(p).long()
    if not p.is_file():
        raise FileNotFoundError(
            f"--input {p}: not a file or a run_all output directory")
    df = pd.read_csv(p)
    need = {"metric", "value", "method"}
    if not need <= set(df.columns):
        raise ValueError(
            f"--input {p}: expected a long results table with columns "
            f"{sorted(need)} (what `multibench evaluate --method/--dataset` "
            f"and run_all's long.csv write); got {list(df.columns)}")
    return df


def _cmd_plot(args) -> int:
    """``multibench plot {bubble,bar}``: draw a results table to ``--out``.

    The frame comes from ``--input`` (a long.csv / run_all dir) or from
    :func:`multibench.load_results` (``--category``, ``--dataset``,
    ``--source``); ``--methods`` restricts the rows in both cases.
    """
    from . import plot as plot_ns
    if args.input is None and args.category is None:
        _usage_error(args, "need --category (stored results) or --input LONG_CSV")
    if args.input is not None:
        df = _load_long_input(args.input)
        if args.dataset:
            df = df[df["dataset"].astype(str).isin(_csv_list(args.dataset))]
    else:
        from . import load_results
        df = load_results(category=args.category, dataset=_csv_list(args.dataset),
                          source=args.source)
    methods = _csv_list(args.methods)
    if methods and args.kind == "bar":
        # plot.bar has no methods=; filter here with the same "unknown name"
        # discipline bubble applies
        present = sorted(df["method"].astype(str).unique())
        unknown = [m for m in methods if m not in present]
        if unknown:
            raise ValueError(f"unknown method(s) {unknown}; methods in the "
                             f"table: {present}")
        df = df[df["method"].isin(methods)]
    if len(df) == 0:
        raise ValueError("the results table is empty after filtering - nothing to plot")
    if df["method"].nunique() == 1:
        print("warning: only one method in this table; ranks and Overall bars "
              "are not meaningful with a single method", file=sys.stderr)
    title = args.title
    if title is None:
        title = " ".join(x for x in (args.category, args.dataset) if x) or None
    metrics = _csv_list(args.metrics)
    if args.kind == "bubble":
        kw = dict(metrics=metrics, methods=methods, aggregate=args.aggregate,
                  title=title, save=args.out, require_complete=args.require_complete)
        if args.overall is not None:
            kw["overall"] = args.overall
        with _quiet_stdout():
            plot_ns.bubble(df, **kw)
    else:
        kw = dict(metrics=metrics, group=args.group, top=args.top, title=title,
                  save=args.out)
        if args.overall is not None:
            kw["overall"] = args.overall
        with _quiet_stdout():
            plot_ns.bar(df, **kw)
    print(f"wrote {args.out}")
    return _EXIT_OK


def _parse_inputs(pairs, args=None) -> dict:
    """Turn repeated ``--input role=path`` values into ``{role: path}``.

    A value without ``=`` (or with an empty side) is a usage error (exit 2).
    """
    out = {}
    for p in pairs or []:
        if "=" not in p:
            msg = f"--input must be role=path, got {p!r}"
            if args is not None:
                _usage_error(args, msg)
            raise SystemExit(msg)
        role, path = p.split("=", 1)
        if role.strip() == "" or path == "":
            msg = f"--input must be role=path with non-empty parts, got {p!r}"
            if args is not None:
                _usage_error(args, msg)
            raise SystemExit(msg)
        out[role] = path
    return out


def _cmd_run(args) -> int:
    """``multibench run``: :func:`multibench.run` one method on explicit inputs.

    ``--param KEY=VALUE`` (repeatable; ``METHOD:KEY=VALUE`` is accepted when
    METHOD is ``--method``) becomes ``params={KEY: value}``. ``--dry-run``
    prints the command line ``run`` would execute (via
    :func:`multibench.workflow.command_preview`) and executes nothing.
    """
    import multibench
    from .workflow import command_preview
    params = _parse_params(args.param, args, default_method=args.method) or {}
    inputs = _parse_inputs(args.input, args)
    if args.dry_run:
        argv = command_preview(args.method, args.category, inputs=inputs,
                               out_dir=args.out, params=params.get(args.method),
                               cmd_template=args.runner)
        print("# dry run - nothing was executed; run() would execute:", file=sys.stderr)
        print(shlex.join(argv))
        return _EXIT_OK
    with _quiet_stdout():                     # library progress -> stderr
        res = multibench.run(method=args.method, category=args.category, task=args.task,
                           inputs=inputs, out_dir=args.out,
                           params=params.get(args.method) or None,
                           cmd_template=args.runner)
    print(f"ran {args.method} -> {res.out_dir}")
    return _EXIT_OK


def _cmd_run_all(args) -> int:
    """``multibench run-all``: :func:`multibench.run_all` on a laid-out dataset.

    ``--dry-run`` prints the plan (one row per method variant: runnable,
    files_ok, env_ok, reason; compact columns - ``--columns all`` for every
    column) and, in table mode, the command line each variant would run (via
    :func:`multibench.workflow.plan_commands`; in csv/tsv/json it is the
    ``command`` column); nothing is executed or created. Otherwise the
    summary table is printed and everything is saved under ``--out-dir``
    (reload with ``multibench plot bubble --input OUT_DIR``). Progress lines
    go to stderr; tables to stdout. ``--param METHOD:KEY=VALUE`` (repeatable)
    becomes ``params={METHOD: {KEY: value}}``.
    """
    import multibench
    from .workflow import plan_commands
    params = _parse_params(args.param, args) or None
    columns = _csv_list(args.columns)
    if args.dry_run:
        with _quiet_stdout():
            df = plan_commands(args.dataset, args.category, out_dir=args.out,
                               methods=_csv_list(args.methods),
                               modalities=_csv_list(args.modalities),
                               data_path=args.data_path, params=params, verbose=False)
        k, n = int(df["runnable"].sum()), len(df)
        print(f"# dry run - nothing was executed; {k} of {n} variant(s) runnable on "
              f"{args.dataset} ({args.category}); commands below are what run() "
              f"would execute (rows without files_ok have none)", file=sys.stderr)
        _print_frame(df, columns=columns, fmt=args.format, compact=_COMPACT_PLAN_COLUMNS)
        if args.format == "table" and not columns:
            have = df[df["command"].astype(str).str.len() > 0]
            print()
            print(f"# commands ({len(have)} variant(s) with resolvable inputs; "
                  f"'[env missing]' = blocked by env_ok only)")
            for _, r in have.iterrows():
                tag = "" if r["env_ok"] else " [env missing]"
                print(f"{r['method']} ({r['modalities']}){tag}: {r['command']}")
        return _EXIT_OK
    with _quiet_stdout():                     # [run_all] progress -> stderr
        res = multibench.run_all(args.dataset, args.category, out_dir=args.out,
                               methods=_csv_list(args.methods),
                               modalities=_csv_list(args.modalities),
                               data_path=args.data_path, params=params,
                               evaluate=not args.no_evaluate, dry_run=False,
                               timeout=args.timeout, skip_existing=args.skip_existing)
    _print_frame(res.summary, columns=columns, fmt=args.format)
    print(f"saved under {args.out}", file=sys.stderr)
    return _EXIT_OK


def _cmd_evaluate(args) -> int:
    """``multibench evaluate``: :func:`multibench.evaluate` on an embedding file.

    Default output is the wide ``metric.csv`` shape (index = metric, one
    ``Value`` column). With ``--method`` AND ``--dataset`` (and ``--category``)
    the frame is reshaped with :func:`multibench.to_long` to the tidy
    ``metric,value,method,dataset,category`` table that ``plot --input`` and
    ``load_results`` speak.
    """
    import multibench
    long_mode = args.method is not None or args.dataset is not None
    if long_mode:
        missing = [f for f, v in (("--method", args.method), ("--dataset", args.dataset),
                                  ("--category", args.category)) if v is None]
        if missing:
            _usage_error(args, f"--method/--dataset write a long table and need "
                         f"all of --method, --dataset, --category; missing "
                         f"{', '.join(missing)}")
    kw = dict(output=args.output, category=args.category, task=args.task,
              labels=args.labels, clustering=args.cluster,
              batch=getattr(args, "batch", None), only=_csv_list(args.only))
    if args.obsm is not None:
        kw["obsm"] = args.obsm
    if args.column is not None:
        kw["column"] = args.column
    with _quiet_stdout():                     # library progress -> stderr
        df = multibench.evaluate(**kw)
    if long_mode:
        df = multibench.to_long(df, args.method, args.dataset, args.category)
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"wrote {args.out}")
        else:
            print(df.to_string(index=False))
        return _EXIT_OK
    if args.out:
        df.to_csv(args.out)
        print(f"wrote {args.out}")
    else:
        print(df.to_string())
    return _EXIT_OK


def _cmd_env(args) -> int:
    """``multibench env ...``: environment recipes, preflight and installation."""
    from .engine import envs, registry
    cmd = args.env_cmd
    if cmd == "status":
        _mlist = _csv_list(getattr(args, "methods", None))
        _cat = getattr(args, "category", None)
        keep = None
        if _mlist:
            keep = set(registry.check_method(m) for m in _mlist)
        elif _cat:
            keep = set(registry.list_methods(category=_cat))
        for r in envs.status():
            if keep is not None and r["method"] not in keep:
                continue
            mark = "x" if r["exists"] else " "
            tag = r["difficulty"] + ("*" if r["verified_working"] else "")
            print(f"[{mark}] {r['method']:16} {r['env']:16} {tag}")
        return _EXIT_OK
    if cmd == "groups":
        for name, spec in envs.groups().items():
            if spec.get("shared"):
                print(f"{name:16} ({len(spec['members']):2}): {', '.join(spec['members'])}")
        return _EXIT_OK
    if cmd == "plan":
        _mlist = _csv_list(getattr(args, "methods", None))
        for p in envs.plan(category=getattr(args, "category", None), methods=_mlist):
            tag = "shared" if p["shared"] else "own"
            print(f"{p['env']:16} [{tag:6}] <- {', '.join(p['methods'])}")
        return _EXIT_OK
    if cmd == "doctor":
        _mlist = _csv_list(getattr(args, "methods", None))
        rows = envs.doctor(category=getattr(args, "category", None), methods=_mlist)
        for r in rows:
            mark = "x" if r["exists"] else ("L" if r["has_lock"] else "!")
            print(f"[{mark}] {r['env']:18} ({len(r['methods']):2}) <- {', '.join(r['methods'])}")
        missing = [r for r in rows if not r["exists"]]
        nolock = [r["env"] for r in missing if not r["has_lock"]]
        print(f"# {len(rows)} envs needed, {len(missing)} missing"
              + (f"; NO lockfile for: {', '.join(nolock)}" if nolock else ""))
        print("# legend: [x]=installed  [L]=missing, lockfile ready (run `multibench env install --run`)  [!]=missing, no lockfile")
        if missing:
            miss_methods = sorted({m for r in missing for m in r["methods"]})
            print(f"# next: multibench env install --methods {','.join(miss_methods)} "
                  "--packed --run")
        if getattr(args, "strict", False) and missing:
            return _EXIT_ERROR
        return _EXIT_OK
    if cmd == "install":
        _mlist = _csv_list(getattr(args, "methods", None))
        do_run = getattr(args, "run", False)
        packed = getattr(args, "packed", False)
        if packed and do_run:
            for r in envs.doctor(category=getattr(args, "category", None),
                                 methods=_mlist):
                with _quiet_stdout():           # "[env] unpacking ..." -> stderr
                    got = (not r["exists"]) and envs.install_packed(r["env"])
                if got:
                    print(f"{r['env']:18} [PACKED        ] <- {', '.join(r['methods'])}")
        # a RuntimeError (no conda here, a failed build) propagates to main():
        # "error: ..." on stderr, exit 1 - never an error line on stdout
        with _quiet_stdout():
            rows = envs.create_all(category=getattr(args, "category", None),
                                   methods=_mlist, dry_run=not do_run)
        manifest = _packed_manifest() if (packed and not do_run) else {}
        states = []
        for r in rows:
            if r["exists"]:
                state = "have"
            elif do_run:
                state = "BUILD" if r["has_lock"] else "NO-LOCK"
            elif packed:
                # packed_urls.json carries URLs only (no sizes), so the dry run
                # can only say whether an archive is published, never how big
                state = ("packed archive published" if r["env"] in manifest
                         else ("no archive - lockfile build" if r["has_lock"]
                               else "no archive - NO-LOCK"))
            else:
                state = "build(dry-run)" if r["has_lock"] else "NO-LOCK"
            states.append(state)
        width = max([14] + [len(s) for s in states])
        for r, state in zip(rows, states):
            print(f"{r['env']:18} [{state:{width}}] <- {', '.join(r['methods'])}")
        if not do_run:
            print("# dry-run - add --run to create the missing envs"
                  + (" (packed archives first, lockfile build otherwise)"
                     if packed else " from their lockfiles"), file=sys.stderr)
        return _EXIT_OK
    if cmd == "freeze":
        if getattr(args, "all", False):
            for env in envs.required_envs(category=getattr(args, "category", None)):
                try:
                    print(f"froze {env} -> {envs.freeze(env)}")
                except Exception as e:  # noqa: BLE001 - report per-env, keep going
                    print(f"SKIP {env}: {str(e)[:120]}")
        else:
            if not getattr(args, "env", None):
                _usage_error(args, "name an env to freeze, or pass --all")
            print(f"froze {args.env} -> {envs.freeze(args.env)}")
        return _EXIT_OK
    if cmd == "create-group":
        with _quiet_stdout():
            cmds = envs.create_group(args.group, dry_run=not getattr(args, "run", False))
        if not getattr(args, "run", False):
            print("# dry-run - add --run to execute:", file=sys.stderr)
            for c in cmds:
                print(shlex.join(c))
        else:
            print(f"created group env {args.group}")
        return _EXIT_OK
    method = getattr(args, "method", None)
    # ONE resolver for every per-method env command: the name scan()['env'],
    # run(), env doctor/plan/install/create expect. A recipe that built
    # scmb_matilda while everything else looked for matilda was the bug.
    expected = envs.default_env_name(method)      # KeyError (did-you-mean) on a typo
    name = getattr(args, "name", None) or expected
    if name == expected:
        banner = (f"# env {name!r} is the name scan/run/env doctor expect for {method} "
                  f"(mtb.env.default_env_name({method!r})); `multibench env create "
                  f"{method}` builds the same env from its lockfile")
    else:
        banner = (f"# env {name!r} is a custom --name: scan/run/env doctor expect "
                  f"{expected!r} for {method} and will not find this one")
    if cmd == "recipe":
        print(banner)
        for c in envs.create_commands(method, env_name=name):
            print(shlex.join(c))
        return _EXIT_OK
    if cmd == "yml":
        y = banner + "\n" + envs.environment_yml(method, env_name=name)
        out = getattr(args, "out", None)
        if out:
            with open(out, "w") as f:
                f.write(y)
            print(f"wrote {out}")
        else:
            print(y, end="")
        return _EXIT_OK
    if cmd == "create":
        with _quiet_stdout():
            cmds = envs.create(method, env_name=name, dry_run=not getattr(args, "run", False))
        if not getattr(args, "run", False):
            print("# dry-run - add --run to execute:", file=sys.stderr)
            for c in cmds:
                print(shlex.join(c))
        else:
            print(f"created environment for {method}")
        return _EXIT_OK
    raise SystemExit(
        "usage: multibench env {status|groups|plan|doctor|install|freeze|recipe|yml|create|create-group}")


# ----------------------------------------------------------------- parser
_CATEGORY_HELP = ("integration category: vertical (several modalities measured in the "
                  "SAME cells, e.g. CITE-seq), diagonal (modalities measured in DIFFERENT "
                  "cells, no pairing), mosaic (several batches, only some share a "
                  "modality) or cross (several batches with ALL modalities; batch-effect "
                  "removal, incl. spatial slice registration)")
_TASK_HELP = ("task within the category: clustering (default), batch, "
              "dimension_reduction, classification, imputation, registration "
              "(mtb.list_tasks())")
_METHODS_HELP = "comma-separated method ids (as printed by `multibench list`)"


def build_parser() -> argparse.ArgumentParser:
    """Build the ``multibench`` argument parser (every command and flag has help text)."""
    p = argparse.ArgumentParser(
        prog="multibench",
        description="Run, evaluate and plot single-cell multimodal integration "
                    "methods from the scMultiBench benchmark. Each command wraps "
                    "one function of the Python API (import multibench as mtb).",
        epilog="Exit codes: 0 ok, 1 runtime error (error: ... on stderr; "
               "MULTIBENCH_DEBUG=1 shows the traceback), 2 usage error. "
               "Run `multibench <command> --help` for the flags of a command.")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {_version()}",
                   help="print the multibench version and exit")
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>",
                           title="commands")

    # ---- list
    pl = sub.add_parser("list", help="list method ids (mtb.list_methods)",
                        description="Print registry method ids, one per line.")
    pl.add_argument("--category", help=_CATEGORY_HELP)
    pl.add_argument("--task", help=_TASK_HELP)
    pl.add_argument("--runnable", action="store_true",
                    help="only methods with a declared variant (usable by run)")
    pl.set_defaults(func=_cmd_list, _parser=pl)

    # ---- find
    pf = sub.add_parser(
        "find", help="find methods by category/modalities/labels/ATAC (mtb.find_methods)",
        description="Print method ids matching ALL given filters, one per line. "
                    "Every filter is optional; with none, every method is listed.")
    pf.add_argument("--category", help=_CATEGORY_HELP)
    pf.add_argument("--task", help=_TASK_HELP)
    pf.add_argument("--modalities",
                    help="comma-separated modalities the method must consume, e.g. "
                         "rna,adt or rna,atac ('protein' is accepted for adt)")
    pf.add_argument("--needs-labels", nargs="?", const="true", choices=["true", "false"],
                    metavar="{true,false}",
                    help="true (or the bare flag): only methods that consume cell-type "
                         "labels (supervised); false: only label-free methods; absent: "
                         "no filter. NOTE: the optional value must come right after "
                         "the flag")
    pf.add_argument("--atac", choices=["peak", "gene_activity"],
                    help="filter by the ATAC representation the method consumes: "
                         "peak (chr:start-end matrix) or gene_activity (gene scores)")
    pf.add_argument("--runnable", action="store_true",
                    help="only methods with a declared variant (usable by run)")
    pf.add_argument("--tunable", nargs="?", const="true", choices=["true", "false"],
                    metavar="{true,false}",
                    help="true (or the bare flag): only methods exposing hyperparameters "
                         "to --param / mtb.params_for; false: only methods that hardcode "
                         "them; absent: no filter")
    pf.set_defaults(func=_cmd_find, _parser=pf)

    # ---- scan
    ps = sub.add_parser(
        "scan", help="which methods can run on a dataset, and why not the rest (mtb.scan)",
        description="Print the preflight table of mtb.scan: one row per method variant "
                    "of the category with the two gates (files_ok, env_ok), the "
                    "runnable verdict and the reason. The table shows a compact column "
                    "set (" + ", ".join(_COMPACT_PLAN_COLUMNS) + "; long text clipped to "
                    f"{_TRUNCATE_WIDTH} chars); --columns all (or any of --format "
                    "csv/tsv/json) gives every column mtb.scan returns: category, env, "
                    "output_kind, n_tunable, observed_worst_sec, caveat, files_reason, "
                    "env_reason, needs_labels, atac ...")
    ps.add_argument("dataset", help="dataset id = the folder name under --data-path "
                                   "(e.g. D11, or MYCITE for your own data)")
    ps.add_argument("--category", required=True, help=_CATEGORY_HELP)
    ps.add_argument("--data-path", dest="data_path",
                    help="folder that CONTAINS the dataset folder (default: the "
                         "package data path, see mtb.config)")
    ps.add_argument("--methods", help=_METHODS_HELP + "; only those rows (unknown "
                                                      "id -> did-you-mean error)")
    ps.add_argument("--modalities", help="comma-separated modality roles to restrict the "
                                         "variants to, e.g. rna,adt ('protein' is "
                                         "accepted for adt)")
    ps.add_argument("--columns", help="comma-separated columns to print, in this order, "
                                      "or 'all' for every column (default: the compact "
                                      "set in table mode, all columns for csv/tsv/json; "
                                      "an unknown name lists the available ones)")
    ps.add_argument("--format", choices=["table", "csv", "tsv", "json"], default="table",
                    help="output format (default table = aligned text, compact and "
                         "clipped; csv/tsv/json = every column, never clipped, for "
                         "scripts; json = a list of row objects)")
    ps.set_defaults(func=_cmd_scan, _parser=ps)

    # ---- layout
    play = sub.add_parser(
        "layout", help="how to lay out YOUR dataset on disk (mtb.describe_layout)",
        description="Print the directory layout and role -> filename contract the "
                    "package expects, optionally for one category.")
    play.add_argument("category", nargs="?", help=_CATEGORY_HELP + " (optional: "
                      "without it the general layout for every category is printed)")
    play.set_defaults(func=_cmd_layout, _parser=play)

    # ---- convert
    pc = sub.add_parser(
        "convert", help="convert .h5ad/.h5mu/.csv to the canonical .h5 files "
                        "(mtb.io.to_canonical / mtb.io.export_dataset)",
        description="Two modes. (1) One file: `convert SRC OUT [--modality M]` writes "
                    "one canonical .h5 (features x cells, matrix/data + features + "
                    "barcodes); OUT may be a directory when --modality is given "
                    "(the canonical filename rna.h5 / adt.h5 / atac_peak.h5 / "
                    "atac_gas.h5 is appended). (2) Whole dataset: any of --rna/--adt/"
                    "--atac/--labels/--batch switches to export_dataset, which "
                    "reads SRC (.h5ad or .h5mu) and writes OUT/ as a dataset folder "
                    "ready for `multibench scan OUT_NAME --data-path <parent>`.")
    pc.add_argument("src", help="input: .h5ad, .h5mu (then --mod or mod: selectors), "
                               ".csv/.tsv (cells x features), .loom, or an already "
                               "canonical .h5 (passed through)")
    pc.add_argument("out", help="output .h5 file (mode 1; or a directory with --modality) "
                               "or the dataset folder to create (mode 2)")
    pc.add_argument("--modality", help="mode 1: rna | adt | atac | atac_peak | atac_gas "
                                       "(aliases protein, peak, gas/gene_activity); "
                                       "validated, picks the filename when OUT is a "
                                       "directory and checks ATAC feature names")
    pc.add_argument("--layer", help="mode 1: take the matrix from adata.layers[LAYER] "
                                    "instead of .X")
    pc.add_argument("--obsm", help="mode 1: take the matrix from adata.obsm[OBSM] "
                                   "(e.g. protein for CITE-seq ADT)")
    pc.add_argument("--mod", help="mode 1: for .h5mu input, the modality to export")
    pc.add_argument("--rna", help="mode 2: where the RNA matrix lives: X, obsm:<key>, "
                                  "layer:<key>, mod:<name> (no default - omit to skip RNA)")
    pc.add_argument("--adt", help="mode 2: where the ADT/protein matrix lives "
                                  "(same grammar, e.g. obsm:protein)")
    pc.add_argument("--atac", help="mode 2: where the ATAC matrix lives (same grammar); "
                                   "requires --atac-kind")
    pc.add_argument("--atac-kind", dest="atac_kind", choices=["peak", "gene_activity"],
                    help="mode 2: peak -> atac_peak.h5 (+ atac.h5); gene_activity -> "
                         "atac_gas.h5")
    pc.add_argument("--labels", help="mode 2: cell-type column, obs:<col> (or "
                                     "mod:<name>.obs:<col>) -> cty.csv")
    pc.add_argument("--batch", help="mode 2: batch column (same grammar); cells are "
                                    "split per batch into numbered files rna1.h5, "
                                    "rna2.h5, cty1.csv ...")
    pc.add_argument("--dtype", default="float64",
                    help="stored dtype of matrix/data (default float64 like the "
                         "shipped files; float32 halves the size)")
    pc.set_defaults(func=_cmd_convert, _parser=pc)

    # ---- cite
    pci = sub.add_parser(
        "cite", help="citation entries for the benchmark and the methods you used (mtb.cite)",
        description="Print the scMultiBench citation, then one entry per method id "
                    "given (in that order). Methods without a verified DOI are "
                    "emitted as a comment naming their repository.")
    pci.add_argument("methods", nargs="*", metavar="METHOD",
                     help="method ids to cite (none: the benchmark entry only)")
    pci.add_argument("--all", action="store_true",
                     help="cite every registry method")
    pci.add_argument("--format", choices=["bibtex", "text"], default="bibtex",
                     help="bibtex (default; one @article per entry) or text (one line "
                          "per entry)")
    pci.add_argument("--out", help="write to this file instead of stdout")
    pci.set_defaults(func=_cmd_cite, _parser=pci)

    # ---- plot
    pp = sub.add_parser(
        "plot", help="draw the bubble table or summary bars of a results frame "
                     "(mtb.plot.bubble / mtb.plot.bar)",
        description="Plot stored benchmark results (--category [--dataset] [--source]) "
                    "or your own long table (--input long.csv or a run_all output "
                    "dir, as written by `multibench evaluate --method/--dataset` "
                    "and `multibench run-all`). Concatenate the two in Python "
                    "(pd.concat) to compare your method with the benchmark's.")
    pp.add_argument("kind", choices=["bubble", "bar"],
                    help="bubble: paper-style bubble table (methods x metrics, best "
                         "first, Overall bars per family) | bar: one bar per method "
                         "with its across-dataset Overall score")
    pp.add_argument("--category", help=_CATEGORY_HELP + " (stored results; required "
                                                        "unless --input is given)")
    pp.add_argument("--dataset", help="dataset id(s), comma-separated; with --category "
                                      "selects the stored table(s), with --input "
                                      "filters the frame")
    pp.add_argument("--source", choices=["published", "rerun"], default="published",
                    help="stored table to load: published (the paper's numbers, "
                         "default) or rerun (the package's own re-execution)")
    pp.add_argument("--input", help="a long results CSV (metric,value,method,dataset,"
                                    "category) or a run_all output directory; replaces "
                                    "--category/--source")
    pp.add_argument("--methods", help=_METHODS_HELP + "; only those rows (unknown name "
                                                      "-> error)")
    pp.add_argument("--metrics", help="comma-separated metric codes to draw, in this "
                                      "order (e.g. ARI,NMI,ASW; default: all present)")
    pp.add_argument("--aggregate", choices=["dataset", "summary"], default="dataset",
                    help="bubble only: dataset = one dataset's raw values (default); "
                         "summary = across-dataset rank panel")
    pp.add_argument("--overall", choices=["rank", "mean_overall"],
                    help="how the Overall score is formed (default: rank for bubble, "
                         "mean_overall for bar; pass the same value to both to get "
                         "the same ordering)")
    pp.add_argument("--require-complete", dest="require_complete", action="store_true",
                    help="bubble --aggregate summary: keep only methods present in "
                         "EVERY dataset instead of warning")
    pp.add_argument("--group", choices=["clustering", "batch"],
                    help="bar only: metric family shorthand (overrides --metrics)")
    pp.add_argument("--top", type=int, help="bar only: keep the N best methods")
    pp.add_argument("--title", help="figure title (default: '<category> <dataset>')")
    pp.add_argument("--out", required=True,
                    help="figure file to write; the suffix picks the format "
                         "(.pdf, .png, .svg)")
    pp.set_defaults(func=_cmd_plot, _parser=pp)

    # ---- run
    pr = sub.add_parser(
        "run", help="run ONE method on explicit input files (mtb.run)",
        description="Run one method in its conda env on the given role=path inputs "
                    "and write its outputs under --out-dir. Use `multibench scan` "
                    "first to see which roles a method needs and whether its env is "
                    "installed; `multibench run-all` runs every method of a category "
                    "on a laid-out dataset folder.")
    pr.add_argument("--method", required=True, help="method id (see `multibench list`)")
    pr.add_argument("--category", required=True, help=_CATEGORY_HELP)
    pr.add_argument("--task", default="clustering", help=_TASK_HELP)
    pr.add_argument("--input", action="append", metavar="ROLE=PATH",
                    help="one input as role=path, repeatable, e.g. --input rna=rna.h5 "
                         "--input adt=adt.h5 --input cty=cty.csv (roles: see "
                         "`multibench layout`); non-canonical files are converted")
    pr.add_argument("--out-dir", "--out", dest="out", required=True,
                    help="output directory for this run (embedding.h5, metric.csv, "
                         "log); --out is an alias")
    pr.add_argument("--runner", help="custom command template instead of `conda run "
                                     "-n <env> ...` (advanced; skips the env preflight)")
    pr.add_argument("--param", "-p", action="append", metavar="KEY=VALUE",
                    help="one hyperparameter override, repeatable: --param epochs=5 "
                         "--param lr=0.001 (METHOD:KEY=VALUE is accepted when METHOD "
                         "is --method). VALUE is parsed as a scalar: 5 -> int, 0.1 -> "
                         "float, true/false -> bool, else string. Keys a method does "
                         "not accept are rejected naming the accepted ones; see "
                         "mtb.params_for(METHOD) or `multibench scan` n_tunable")
    pr.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="print the exact command line run() would execute (conda run "
                         "-n <env> python <script> ...; inputs as given, params merged) "
                         "and execute nothing")
    pr.set_defaults(func=_cmd_run, _parser=pr)

    # ---- run-all
    pra = sub.add_parser(
        "run-all", help="run every runnable method of a category on a dataset folder "
                        "(mtb.run_all)",
        description="Run (or with --dry-run, only plan) every method of the category "
                    "on <data-path>/<DATASET>/, evaluate each output and save the "
                    "summary, long.csv and figure under --out-dir.")
    pra.add_argument("dataset", help="dataset id = the folder name under --data-path")
    pra.add_argument("--category", required=True, help=_CATEGORY_HELP)
    pra.add_argument("--out-dir", "--out", dest="out", required=True,
                    help="directory for all outputs (one sub-folder per method); "
                         "--out is an alias")
    pra.add_argument("--methods", help=_METHODS_HELP + "; only those (default: every "
                                                       "runnable method)")
    pra.add_argument("--modalities", help="comma-separated modalities to restrict the "
                                          "method variants to, e.g. rna,adt")
    pra.add_argument("--data-path", dest="data_path",
                     help="folder that CONTAINS the dataset folder (default: the "
                          "package data path)")
    pra.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="print the plan (one row per method variant: runnable, "
                          "files_ok, env_ok, reason; --columns all for every column) "
                          "and, per variant whose inputs resolve, the exact command "
                          "line run() would execute (a 'command' column in csv/tsv/"
                          "json); execute and create nothing")
    pra.add_argument("--param", "-p", action="append", metavar="METHOD:KEY=VALUE",
                     help="one hyperparameter override, repeatable: --param "
                          "Matilda:epochs=5 --param Matilda:lr=0.001 -> params="
                          "{'Matilda': {'epochs': 5, 'lr': 0.001}}. VALUE is parsed "
                          "as a scalar (5 -> int, 0.1 -> float, true/false -> bool, "
                          "else string). Unknown METHOD -> did-you-mean error; a key "
                          "the method does not accept is rejected naming the accepted "
                          "ones (checked in --dry-run too); see mtb.params_for(METHOD). "
                          "Not allowed with --skip-existing")
    pra.add_argument("--columns", help="comma-separated columns of the printed table "
                                       "(plan with --dry-run, summary otherwise), or "
                                       "'all' (default: compact plan columns in table "
                                       "mode; every column for csv/tsv/json)")
    pra.add_argument("--format", choices=["table", "csv", "tsv", "json"], default="table",
                     help="output format of the printed table (default table; csv/tsv/"
                          "json never clip text and, for --dry-run, carry the command "
                          "column)")
    pra.add_argument("--timeout", type=float,
                     help="per-method wall-clock limit in seconds (default: none)")
    pra.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                     help="reuse a method's existing output under --out-dir instead "
                          "of re-running it")
    pra.add_argument("--no-evaluate", dest="no_evaluate", action="store_true",
                     help="run only; do not compute metrics on the outputs")
    pra.set_defaults(func=_cmd_run_all, _parser=pra)

    # ---- evaluate
    pe = sub.add_parser(
        "evaluate", help="scIB metrics for an embedding against labels (mtb.evaluate)",
        description="Compute the benchmark's metrics for one embedding file. Prints "
                    "the metric table (or writes it with --out). With --method, "
                    "--dataset and --category the table is written in the long "
                    "format (metric,value,method,dataset,category) that `multibench "
                    "plot --input` reads, so your method can be plotted next to the "
                    "benchmark's.")
    pe.add_argument("--output", required=True,
                    help="the embedding, cells x dims: .h5 (dataset 'data', the "
                         "benchmark's embedding.h5), .h5ad (uses --obsm), .npy, "
                         ".csv/.tsv (a leading barcode column is dropped)")
    pe.add_argument("--category", help=_CATEGORY_HELP + " (validated when given; "
                                                        "required with --method/--dataset)")
    pe.add_argument("--task", default="clustering", choices=["clustering", "batch", "all",
                                                             "dimension_reduction"],
                    help="metric group: clustering (default; ARI, NMI, ASW, ...), batch "
                         "(ASW_batch, GC, iLISI, ... needs --batch), all")
    pe.add_argument("--labels", help="cell-type labels CSV, one row per cell in the "
                                     "embedding's order (header row; column 'x', the "
                                     "only column, or see --column)")
    pe.add_argument("--batch", help="per-cell batch labels CSV (required for --task "
                                    "batch/all)")
    pe.add_argument("--clustering", "--cluster", dest="cluster", metavar="PATH",
                    help="precomputed cluster assignment (CSV, or an .h5 read from "
                         "/obs/cluster_leiden); skips the Leiden resolution sweep. "
                         "--cluster is an alias")
    pe.add_argument("--only", help="comma-separated metric names to compute (e.g. "
                                   "ARI,NMI); everything else, including the Leiden "
                                   "sweep when not needed, is skipped")
    pe.add_argument("--obsm", help="for .h5ad input: the .obsm key holding the "
                                   "embedding (default X_emb; 'X' = .X)")
    pe.add_argument("--column", help="column of the labels CSV to use when it has "
                                     "several")
    pe.add_argument("--method", help="label the rows with this method name and write "
                                     "the long format (needs --dataset and --category)")
    pe.add_argument("--dataset", help="label the rows with this dataset id (needs "
                                      "--method and --category)")
    pe.add_argument("--out", help="CSV to write (default: print the table)")
    pe.set_defaults(func=_cmd_evaluate, _parser=pe)

    # ---- env
    pv = sub.add_parser(
        "env", help="per-method conda environments: status, plan, doctor, install, "
                    "recipes (mtb.env)",
        description="Every method runs in its own (or a shared) conda env built from "
                    "a committed lockfile or a published packed archive. Typical "
                    "flow: `env doctor --category C` then `env install --category C "
                    "--packed --run`.")
    ev = pv.add_subparsers(dest="env_cmd", required=True, metavar="<env-command>",
                           title="env commands")
    es = ev.add_parser("status", help="per method: env installed? env name, difficulty",
                       description="One line per method: [x] installed / [ ] not, the "
                                   "env the package uses for it (the same name scan/run/"
                                   "doctor/recipe use) and a difficulty tag (* = verified "
                                   "working).")
    es.add_argument("--category", help=_CATEGORY_HELP + " (only its methods)")
    es.add_argument("--methods", help=_METHODS_HELP + "; only those")
    es.set_defaults(func=_cmd_env, _parser=es)
    egr = ev.add_parser("groups", help="list the shared env groups and their members",
                        description="Shared conda envs (one env serving several methods) "
                                    "and the methods in each.")
    egr.set_defaults(func=_cmd_env, _parser=egr)
    _NAME_HELP = ("environment name (default: the env the package uses for METHOD - "
                  "mtb.env.default_env_name(METHOD), the same name `multibench scan` "
                  "shows in its env column and run/env doctor/env create use; e.g. "
                  "`multibench env recipe Matilda` -> matilda, "
                  "`multibench env recipe UINMF` -> scmb_r)")
    er = ev.add_parser("recipe", help="print the conda/pip commands that build a method's env",
                       description="Print, without running, the hand-written recipe "
                                   "commands that create the environment for METHOD, "
                                   "named as scan/run expect it (first line: a comment "
                                   "naming that env). `env create METHOD` builds the same "
                                   "env from its committed lockfile - the reproducible "
                                   "path; the recipe is the transparent one.")
    er.add_argument("method", help="method id")
    er.add_argument("--name", help=_NAME_HELP)
    er.set_defaults(func=_cmd_env, _parser=er)
    ey = ev.add_parser("yml", help="print/write a conda environment.yml for a method",
                       description="Emit an environment.yml for METHOD (stdout, or --out) "
                                   "whose name: is the env scan/run expect "
                                   "(mtb.env.default_env_name).")
    ey.add_argument("method", help="method id")
    ey.add_argument("--name", help=_NAME_HELP)
    ey.add_argument("--out", help="write the yml here instead of stdout")
    ey.set_defaults(func=_cmd_env, _parser=ey)
    ec = ev.add_parser("create", help="create ONE method's env (dry run unless --run)",
                       description="Create the conda environment for METHOD from its "
                                   "committed lockfile (falling back to the recipe when "
                                   "none is captured), under the name scan/run expect; "
                                   "without --run only the commands are printed.")
    ec.add_argument("method", help="method id")
    ec.add_argument("--name", help=_NAME_HELP)
    ec.add_argument("--run", action="store_true",
                    help="actually create the env; without it the command is a dry run")
    ec.set_defaults(func=_cmd_env, _parser=ec)
    ep = ev.add_parser("plan", help="which envs a set of methods needs (collapsed per env)",
                       description="Collapse methods into the conda envs they need, "
                                   "marking shared vs own envs.")
    ep.add_argument("--category", help=_CATEGORY_HELP)
    ep.add_argument("--methods", help=_METHODS_HELP + "; only their envs")
    ep.set_defaults(func=_cmd_env, _parser=ep)
    eg = ev.add_parser("create-group", help="create one shared group env (dry run unless --run)",
                       description="Create the shared conda env GROUP; without --run only "
                                   "the commands are printed.")
    eg.add_argument("group", help="group name (see `multibench env groups`)")
    eg.add_argument("--run", action="store_true",
                    help="actually create the env; without it the command is a dry run")
    eg.set_defaults(func=_cmd_env, _parser=eg)
    edoc = ev.add_parser(
        "doctor", help="preflight: which envs are present / need building",
        description="One line per env needed by the selected methods: [x] installed, "
                    "[L] missing but a lockfile is ready, [!] missing and no "
                    "lockfile; then the install command for the missing ones.")
    edoc.add_argument("--category", help=_CATEGORY_HELP)
    edoc.add_argument("--methods", help=_METHODS_HELP + "; only their envs")
    edoc.add_argument("--strict", action="store_true",
                      help="exit 1 when any env is missing (for scripts: "
                           "`env doctor --strict || env install ...`)")
    edoc.set_defaults(func=_cmd_env, _parser=edoc)
    ei = ev.add_parser(
        "install", help="build every needed env from its lockfile or packed archive",
        description="Install the envs the selected methods need. Dry run by default: "
                    "prints per env 'have' / 'build(dry-run)' / 'NO-LOCK', or with "
                    "--packed 'packed archive published' / 'no archive - lockfile "
                    "build'. Add --run to do it.")
    ei.add_argument("--category", help=_CATEGORY_HELP)
    ei.add_argument("--methods", help=_METHODS_HELP + "; only their envs")
    ei.add_argument("--packed", action="store_true",
                    help="use prebuilt archives when published; fall back to the "
                         "lockfile build")
    ei.add_argument("--run", action="store_true",
                    help="actually create the envs; without it the command is a dry run")
    ei.set_defaults(func=_cmd_env, _parser=ei)
    ef = ev.add_parser("freeze", help="capture an env (or --all) to a committed lockfile",
                       description="Write the lockfile of an installed env (maintainers).")
    ef.add_argument("env", nargs="?", help="env name to freeze")
    ef.add_argument("--all", action="store_true", help="freeze every required env")
    ef.add_argument("--category", help=_CATEGORY_HELP + " (with --all: only its envs)")
    ef.set_defaults(func=_cmd_env, _parser=ef)
    return p


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:  # noqa: BLE001 - version is cosmetic here
        return "unknown"


def main(argv=None) -> int:
    """Entry point: parse ``argv`` (default ``sys.argv[1:]``) and run the command.

    Returns the exit code (0 ok, 1 runtime error, 2 usage error - argparse
    raises ``SystemExit(2)`` for the latter). This is the CLI's single error
    boundary: any exception from the API is printed as ``error: <message>`` on
    stderr and mapped to 1, unless ``MULTIBENCH_DEBUG=1`` is set, in which case
    it is re-raised with its traceback. Python warnings raised while a command
    runs are printed as ``warning: <message>`` on stderr (raw
    ``<file>:<line>: UserWarning`` form only under ``MULTIBENCH_DEBUG``).
    """
    args = build_parser().parse_args(argv)
    prev_show = warnings.showwarning

    def _show(message, category, filename, lineno, file=None, line=None):
        # library UserWarnings reach the terminal as 'warning: <text>' on
        # stderr, like the CLI's own notes - not as '<path>:<line>: UserWarning'
        print(f"warning: {message}", file=sys.stderr)

    if not os.environ.get("MULTIBENCH_DEBUG"):
        warnings.showwarning = _show
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 - the CLI's single error boundary
        if os.environ.get("MULTIBENCH_DEBUG"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {msg}", file=sys.stderr)
        print("(set MULTIBENCH_DEBUG=1 for the full traceback)", file=sys.stderr)
        return _EXIT_ERROR
    finally:
        warnings.showwarning = prev_show


if __name__ == "__main__":
    raise SystemExit(main())
