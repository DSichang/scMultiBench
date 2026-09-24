"""Command-line interface over the multibench Python API.

Every subcommand is a thin wrapper around one public Python function, with the
same parameter names where a flag exists (``--category``, ``--metrics``,
``--out-dir`` ...). The end-to-end story mirrors the Python one::

    multibench config                                  # where data, envs and scripts live
    multibench fetch D11                               # a demo dataset
    multibench layout vertical                         # how to lay out my data
    multibench convert my.h5ad data/MYCITE --rna X --adt obsm:protein --labels obs:celltype
    multibench scan MYCITE --category vertical --data-path data
    multibench info Matilda                            # env, GPU, labels, ATAC input
    multibench params Matilda                          # what --param accepts
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

Data goes to stdout (tables, ids, commands, yml, citations, ``wrote ...``
lines); diagnostics go to stderr (``error: ...``, ``warning: ...``, progress
such as ``[run_all] ...`` and ``# dry run ...`` notes), so
``multibench scan ... --format tsv > plan.tsv`` captures a clean table and
``2>/dev/null`` silences the rest. Tables default to a compact column set
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
#: mode. The full frame is too wide for a terminal; ``--columns all`` or a
#: machine format (csv/tsv/json) prints every column.
#: :func:`_compact_plan_columns` adds ``atac`` and ``caveat`` when they apply.
_COMPACT_PLAN_COLUMNS = ["method", "modalities", "runnable", "files_ok", "env_ok",
                         "runtime_tier", "reason"]
#: Free-text columns clipped to this many characters in table mode (never in
#: csv/tsv/json, never with an explicit ``--columns`` list).
_TRUNCATE_WIDTH = 80
_TRUNCATE_COLUMNS = ("reason", "files_reason", "env_reason", "caveat", "command", "error")
#: The ``caveat`` column of the compact plan view is clipped shorter: it is a
#: pointer to the full text (``--columns all`` or ``--format csv``).
_CAVEAT_WIDTH = 40


def _truncate(text, width: int = _TRUNCATE_WIDTH) -> str:
    """Clip ``text`` to at most ``width`` characters, ending in ``...``.

    The cut falls on the last space before the limit when there is one in
    the second half of the kept text, so a word or path is not split.
    """
    t = "" if text is None else str(text)
    if len(t) <= width:
        return t
    head = t[: width - 3]
    cut = head.rfind(" ")
    if cut >= (width - 3) // 2:
        head = head[:cut]
    return head.rstrip(" ,;:") + "..."


def _blank(value) -> bool:
    """``None``, NaN or an empty / whitespace-only string."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return str(value).strip() in ("", "None", "nan")


def _compact_plan_columns(df) -> list[str]:
    """The compact column set of the ``scan`` / ``run-all --dry-run`` table for ``df``.

    :data:`_COMPACT_PLAN_COLUMNS`, plus ``atac`` after ``modalities`` when a
    row reads an ATAC file (the peak / gene-activity choice decides whether
    the method's embedding is valid), plus ``caveat`` after ``reason`` when
    any row has one.
    """
    cols = list(_COMPACT_PLAN_COLUMNS)
    if "atac" in df.columns and "modalities" in df.columns and any(
            "atac" in str(m) for m in df["modalities"]):
        cols.insert(cols.index("modalities") + 1, "atac")
    if "caveat" in df.columns and not all(_blank(c) for c in df["caveat"]):
        cols.insert(cols.index("reason") + 1, "caveat")
    return cols


def _resolve_columns(df, columns, fmt: str, compact=None) -> list | None:
    """Which columns to print: an explicit list, ``all``, or the default.

    The default is the ``compact`` set in table mode (a readable terminal
    width) and every column for csv/tsv/json. ``columns=["all"]`` is every
    column in any format. Compact columns absent from the frame are skipped
    silently (a mocked or older frame); an unknown explicit name is an error
    naming the available ones.
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
    has, so a column added to the frame is selectable without a CLI change.
    ``compact`` is the default column list for table mode (csv/tsv/json print
    everything unless ``columns`` is given). ``truncate``
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
        # to_json converts the values (NaN -> null, numpy scalars); json.dumps
        # writes them without escaping '/' as '\/'
        rows = json.loads(df.to_json(orient="records", default_handler=str))
        print(json.dumps(rows, indent=1), file=file)
    else:
        if truncate is None:
            truncate = not columns
        if truncate:
            df = df.copy()
            for c in _TRUNCATE_COLUMNS:
                if c in df.columns:
                    width = _CAVEAT_WIDTH if c == "caveat" else _TRUNCATE_WIDTH
                    df[c] = df[c].map(lambda t, w=width: "" if _blank(t) else _truncate(t, w))
            if "atac" in df.columns:
                df["atac"] = df["atac"].map(lambda v: "" if _blank(v) else v)
        if len(df) == 0:
            print(f"(empty table; columns: {list(df.columns)})", file=file)
        else:
            print(df.to_string(index=False), file=file)


def _quiet_stdout():
    """Route a library call's stdout progress lines (``[run_all] ...``,
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


def _usage_error(args, message: str) -> None:
    """Report a usage error the way argparse does (usage line + message, exit 2)."""
    parser = getattr(args, "_parser", None)
    if parser is not None:
        parser.error(message)          # prints usage + "error: ..." and exits 2
    raise SystemExit(_EXIT_USAGE)


def _platform_note() -> str | None:
    """Print ``warning: ...`` on stderr when method envs cannot be built here.

    The method envs are linux-64 conda envs; on macOS/Windows ``env install
    --run`` refuses (``--force`` overrides). Printed once, first, on stderr,
    so stdout stays the table and the user reads it before a multi-GB
    download starts. Returns the problem text (or ``None``).
    """
    from .engine import envs
    problem = envs.host_platform_problem()
    if problem:
        print(f"warning: {envs.linux_only_text(problem)} `multibench env install "
              f"--run` refuses here; --force tries anyway. The registry, stored "
              f"results, scan's file check, evaluate and plot work on this computer.",
              file=sys.stderr)
    return problem


# ----------------------------------------------------------------- commands
def _cmd_list(args) -> int:
    """``multibench list``: method ids from the registry, one per line.

    Plain ``list [--category C]`` is :func:`multibench.list_methods`; the
    ``--task`` / ``--runnable`` filters are :func:`multibench.find_methods`
    filters (same order, same set).
    """
    from . import discover
    for m in discover.find_methods(category=args.category, task=args.task,
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
    ``--category`` is optional, exactly as in ``mtb.scan(dataset)``: without
    it every category is scanned.
    """
    import multibench
    from .engine import registry
    methods = _csv_list(args.methods)
    for m in methods or []:
        registry.check_method(m)               # did-you-mean KeyError before any I/O
    kw = dict(data_path=args.data_path, modalities=_csv_list(args.modalities),
              verbose=False, assume_gpu=getattr(args, "assume_gpu", False))
    df = None
    try:
        # named methods keep a wrong-ATAC-kind row runnable, as in mtb.scan
        df = multibench.scan(args.dataset, args.category, methods=methods, **kw)
    except ValueError as e:
        # a representation token that dropped a named method: the message says why
        if not methods or getattr(e, "representation", False):
            raise
    if methods:
        unknown = sorted(set(methods) - set([] if df is None else df["method"]))
        if unknown:
            where = f"{args.dataset}/{args.category}" if args.category else (
                f"{args.dataset} (all categories)")
            # what the category holds, not only the named methods it holds
            present = multibench.scan(args.dataset, args.category, **kw)["method"]
            raise ValueError(
                f"method(s) {unknown} are not in the scan table for "
                f"{where}; methods present: "
                f"{sorted(set(present))}")
        df = df[df["method"].isin(methods)]
    _print_frame(df, columns=_csv_list(args.columns), fmt=args.format,
                 compact=_compact_plan_columns(df))
    if getattr(args, "strict", False):
        problem = _strict_problem(df, methods)
        if problem:
            print(f"error: --strict: {problem}", file=sys.stderr)
            return _EXIT_ERROR
    return _EXIT_OK


def _strict_problem(df, methods) -> str | None:
    """Why ``scan --strict`` fails for the printed rows ``df``, or ``None``.

    It fails when no row is runnable, or when a method named in
    ``--methods`` has no runnable row. The text counts the rows each check
    blocks and, for named methods, gives the reason of each one's first row.
    A GPU-only method on a host without a GPU is counted apart from a
    missing env, with a pointer to ``--assume-gpu``; so are a row given the
    wrong ATAC kind, a row whose peak names the method cannot read, and
    scripts at another commit than ``MULTIBENCH_SCRIPTS_REF``. Method
    scripts not fetched block every row, although ``runnable`` stays true:
    a job on a node without network cannot fetch them.
    """
    from . import config
    fetched = config.scripts_present()
    ok = df["runnable"].astype(bool) & fetched
    runnable = df[ok]
    if methods:
        blocked = [m for m in methods if m not in set(runnable["method"])]
    else:
        blocked = [] if len(runnable) else list(dict.fromkeys(df["method"]))
    if not blocked and (methods or len(runnable)):
        return None
    head = f"{len(runnable)} of {len(df)} row(s) runnable"
    rest = df[~ok]
    counts = []
    if "files_ok" in rest and (~rest["files_ok"].astype(bool)).any():
        counts.append(f"input files missing in {int((~rest['files_ok'].astype(bool)).sum())}")
    from .workflow import PEAK_NAMES_REASON, _is_wrong_atac, _is_wrong_ref
    if "reason" in rest:
        wrong = _is_wrong_atac(rest["reason"])
        names = wrong & rest["reason"].astype(str).str.contains(PEAK_NAMES_REASON,
                                                                regex=False)
        if (wrong & ~names).any():
            counts.append(f"wrong ATAC kind in {int((wrong & ~names).sum())}")
        if names.any():
            counts.append(f"unreadable peak names in {int(names.sum())}")
        if _is_wrong_ref(rest["reason"]).any():
            counts.append(f"scripts not at MULTIBENCH_SCRIPTS_REF in "
                          f"{int(_is_wrong_ref(rest['reason']).sum())}")
    if not fetched:
        counts.append(f"method scripts not fetched in {len(df)}")
    n_env, n_gpu, gpu_only = _env_and_gpu_counts(rest)
    if n_env:
        counts.append(f"env not ready in {n_env}")
    if n_gpu:
        counts.append(f"needs a GPU on this host in {n_gpu}")
    gpu_note = ("\nFor a job that runs on a GPU node, add --assume-gpu."
                if gpu_only else "")
    if not fetched:                     # the fix for the scripts, before the GPU note
        fix = config.scripts_folder_problem() or "Run multibench fetch --scripts first"
        gpu_note = f"\n{fix}.{gpu_note}"
    if counts:
        head += f" ({'; '.join(counts)})"
    if not methods:
        return head + "; the reason column says why" + gpu_note
    lines = []
    for m in blocked:
        rows = df[df["method"] == m]
        reason = rows["reason"].iloc[0] if len(rows) else ""
        if len(rows) and _blank(reason):            # runnable but for the scripts
            reason = "method scripts not fetched"
        lines.append(f"  {m}: {'no row' if _blank(reason) else _truncate(reason, 120)}")
    return (head + f"; no runnable row for {', '.join(blocked)}:\n"
            + "\n".join(lines) + gpu_note)


def _env_and_gpu_counts(rest) -> tuple[int, int, int]:
    """Count the rows of ``rest`` blocked by a missing env and by this host's GPU test.

    ``env_reason`` joins the env sentence and the GPU sentence
    (``spec.requires_gpu_reason``) with ``"; "``; a row counts under each
    part it carries. The third count is the rows the GPU test alone blocks
    (files and env ready): the ones ``--assume-gpu`` makes runnable.
    """
    from .engine import registry
    if "env_ok" not in rest:
        return 0, 0, 0
    n_env = n_gpu = gpu_only = 0
    for _, r in rest[~rest["env_ok"].astype(bool)].iterrows():
        text = "" if _blank(r.get("env_reason")) else str(r["env_reason"])
        spec = registry.get(r["method"])
        gpu = spec.requires_gpu_reason if spec.requires_gpu else ""
        blocked_by_gpu = bool(gpu) and gpu in text
        env_missing = bool(text.replace(gpu, "").strip("; ") if gpu else text.strip("; "))
        n_gpu += blocked_by_gpu
        n_env += env_missing
        gpu_only += blocked_by_gpu and not env_missing and bool(r.get("files_ok", True))
    return n_env, n_gpu, gpu_only


def _cmd_layout(args) -> int:
    """``multibench layout``: print :func:`multibench.describe_layout`."""
    import multibench
    print(multibench.describe_layout(args.category))
    return _EXIT_OK


_EXPORT_FLAGS = ("rna", "adt", "atac", "atac_kind", "labels", "batch", "batch_index",
                 "atac_from")

#: The recipes of ``multibench convert --help``: a feature filter, unpaired
#: RNA and ATAC, and a D46-shaped mosaic folder (batch 1 RNA+ADT, batch 2
#: RNA+ATAC, batch 3 RNA).
_CONVERT_EPILOG = """\
Genes and peaks in one X (10x Multiome read with gex_only=False):
  multibench convert B.h5ad data/multiome --rna "X[feature_types=Gene Expression]" --atac "X[feature_types=Peaks]" --atac-kind peak --labels obs:cell_type --category vertical

RNA and ATAC from different cells (diagonal):
  multibench convert rna.h5ad data/lung --rna X --atac-from atac.h5ad --atac-kind peak --labels obs:cell_type --category diagonal

One file per batch (mosaic or cross): write each file as its batch number.
  multibench convert A.h5ad data/LAB --rna X --adt obsm:protein --labels obs:cell_type --category mosaic --batch-index 1
  multibench convert B.h5mu data/LAB --rna mod:rna --atac mod:atac --atac-kind peak --labels rna:cell_type --category mosaic --batch-index 2
  multibench convert C.h5ad data/LAB --rna X --labels obs:cell_type --category mosaic --batch-index 3
  multibench scan LAB --category mosaic --data-path data
"""


def _rewrite_canonical(src, out, dtype: str) -> bool:
    """Copy the canonical ``.h5`` ``src`` to ``out``, storing ``matrix/data`` as ``dtype``.

    A byte copy when the stored dtype already matches (fast, exact), else a
    block-wise rewrite that casts ``matrix/data`` (features x cells stays as
    is) and copies ``matrix/features`` / ``matrix/barcodes`` verbatim.

    Returns ``True`` when the matrix was re-encoded, ``False`` for a byte copy.
    """
    import shutil

    import h5py
    import numpy as np
    src, out = Path(src), Path(out)
    with h5py.File(src, "r") as f:
        same = f["matrix/data"].dtype == np.dtype(dtype)
    if same:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f, h5py.File(out, "w") as g:
        d = f["matrix/data"]
        comp = d.compression
        o = g.create_dataset("matrix/data", shape=d.shape, dtype=dtype,
                             chunks=True if comp else None, compression=comp)
        step = max(1, 1024)
        for i in range(0, d.shape[0], step):
            o[i:i + step] = np.asarray(d[i:i + step]).astype(dtype, copy=False)
        for k in ("matrix/features", "matrix/barcodes"):
            if k in f:
                f.copy(k, g["matrix"], name=k.split("/")[-1])
    return True


def _folder_state(folder) -> dict:
    """``{name: (inode, mtime_ns, size)}`` of the files in ``folder`` ({} if absent)."""
    d = Path(folder)
    if not d.is_dir():
        return {}
    return {q.name: (st.st_ino, st.st_mtime_ns, st.st_size)
            for q in d.iterdir() if q.is_file() for st in (q.stat(),)}


def _cmd_convert(args) -> int:
    """``multibench convert``: one canonical ``.h5`` (``to_canonical``) or a whole
    dataset folder (``export_dataset``), chosen by the flags given.

    * No ``--rna/--adt/--atac/--labels/--batch`` flag: ``SRC`` -> ``OUT`` via
      :func:`multibench.io.to_canonical` (``--modality`` picks the filename when
      ``OUT`` is a directory; ``--layer/--obsm/--mod`` select the matrix). A
      ``SRC`` that is already canonical is copied to ``OUT`` (``--dtype``
      honoured by re-encoding ``matrix/data``) and reported as ``copied``;
      when ``OUT`` is ``SRC`` itself nothing is written and the line says so.
      The word ``wrote`` is printed only for a file this command produced.
    * Any of those flags: ``SRC`` is read (``.h5ad``/``.h5mu``) and
      :func:`multibench.io.export_dataset` writes ``OUT/`` as a dataset folder
      (``rna.h5``, ``adt.h5``, ``atac_*.h5``, ``cty.csv`` ...). ``--modality``,
      ``--layer``, ``--obsm`` and ``--mod`` are then usage errors: the selector
      grammar (``X``, ``obsm:<key>``, ``layer:<key>``, ``mod:<name>``) carries
      that information per modality.

    ``--category`` is forwarded to both (``category=``): ``vertical`` writes
    the paired-multiome ``atac.h5`` instead of ``atac_gas.h5``/``atac_peak.h5``.
    """
    from .engine import ingest
    export_mode = any(getattr(args, f) is not None for f in _EXPORT_FLAGS)
    category = getattr(args, "category", None)
    if export_mode:
        clash = [f"--{f}" for f in ("modality", "layer", "obsm", "mod")
                 if getattr(args, f) is not None]
        if clash:
            _usage_error(args, f"{', '.join(clash)} cannot be combined with the "
                         "dataset-export flags (--rna/--adt/--atac/--labels/"
                         "--batch/--batch-index/--atac-from); put the selector in "
                         "the flag value instead, e.g. --adt obsm:protein")
        if args.rna is None and args.adt is None and args.atac is None \
                and args.labels is None and args.atac_from is None:
            _usage_error(args, "dataset export needs at least one of --rna, "
                         "--adt, --atac, --labels (note: --rna has no default "
                         "on the command line; pass --rna X to export adata.X)")
        batch_index = getattr(args, "batch_index", None)
        extra = {}
        if batch_index is not None:
            if args.batch is not None:
                _usage_error(args, "--batch and --batch-index are mutually exclusive: "
                             "--batch splits one file by a column, --batch-index "
                             "writes the whole file as one batch")
            if category not in ("mosaic", "cross"):
                _usage_error(args, "--batch-index needs --category mosaic or --category "
                             "cross (the categories that read numbered batch files)")
            if batch_index < 1:
                _usage_error(args, f"--batch-index counts from 1, got {batch_index}")
            extra["batch_index"] = batch_index
        atac = args.atac
        if args.atac_from is not None:
            if category != "diagonal":
                _usage_error(args, "--atac-from is for --category diagonal (RNA and "
                             "ATAC from different cells)")
            atac = ingest._select_object(ingest._to_anndata(args.atac_from),
                                         args.atac or "X", what="atac")
        data = ingest._to_anndata(args.src)
        before = _folder_state(args.out)
        p = ingest.export_dataset(data, args.out, rna=args.rna, adt=args.adt,
                                  atac=atac, atac_kind=args.atac_kind,
                                  labels=args.labels, batch=args.batch,
                                  dtype=args.dtype, category=category,
                                  overwrite=args.overwrite, **extra)
        after = _folder_state(p)
        wrote = sorted(n for n in after if before.get(n) != after[n])
        kept = sorted(n for n in after if n not in wrote)
        print(f"wrote dataset folder {p} (files: {', '.join(wrote)})")
        if kept:
            print(f"# already in the folder: {', '.join(kept)} (not written by this call)",
                  file=sys.stderr)
        return _EXIT_OK
    if args.layer is not None and args.obsm is not None:
        _usage_error(args, "--layer and --obsm are mutually exclusive")
    src = Path(args.src)
    if src.is_file() and ingest._is_canonical_h5(src):
        # to_canonical() passes a canonical file through and returns SRC
        # (enough for run(), which only needs a readable path); the CLI must
        # put a file at OUT, so copy (re-encoding for --dtype) and never say
        # 'wrote' for a path this command did not write.
        out = Path(args.out)
        dir_like = out.is_dir() or str(args.out).endswith(("/", os.sep))
        if dir_like:
            if args.modality is None:
                _usage_error(args, f"OUT {args.out} is a directory; pass --modality "
                             "so the canonical filename (rna.h5, adt.h5, ...) can be "
                             "chosen, or give OUT as a file path")
            out = out / ingest._atac_filename(ingest._norm_modality(args.modality),
                                              ingest._check_category(category))
        if out.exists() and out.resolve() == src.resolve():
            print(f"already canonical - nothing written; use {src} in place")
            return _EXIT_OK
        recoded = _rewrite_canonical(src, out, args.dtype)
        how = (f"matrix/data re-encoded as {args.dtype}" if recoded
               else "byte copy, dtype unchanged")
        print(f"copied {src} -> {out} (already canonical; {how})")
        return _EXIT_OK
    p = ingest.to_canonical(args.src, args.out, modality=args.modality,
                            layer=args.layer, obsm=args.obsm, mod=args.mod,
                            dtype=args.dtype, category=category)
    print(f"wrote {p}")
    return _EXIT_OK


def _cmd_fetch(args) -> int:
    """``multibench fetch``: :func:`multibench.data.fetch` (or ``fetch_outputs``)
    for the datasets named, and the method scripts with ``--scripts``.

    ``--scripts`` runs the fetch the first ``mtb.run`` would run
    (``PYangLab/scMultiBench`` into ``repo_path``, with ``git``; ``--ref`` or
    ``MULTIBENCH_SCRIPTS_REF`` picks a commit or tag), or reports the scripts
    present, with the commit either way; a compute node without network then
    needs no download. One ``<id>: <folder>`` line per dataset goes to
    stdout, download progress to stderr.
    """
    from . import config
    from .data.fetch import AVAILABLE, fetch, fetch_outputs
    ids = [d for tok in (args.datasets or []) for d in _csv_list(tok)]
    if not ids and not args.scripts:
        _usage_error(args, "name the dataset ids to download (e.g. multibench fetch D11 "
                     f"D46; available: {', '.join(sorted(AVAILABLE))}), or pass "
                     "--scripts for the method scripts")
    if args.ref and not args.scripts:
        _usage_error(args, "--ref picks the commit or tag of the method scripts; use it "
                     "with --scripts")
    if args.scripts:
        present = config.scripts_present()
        with _quiet_stdout():                 # the clone's progress -> stderr
            repo = config.ensure_repo(ref=args.ref)
        state = "present" if present else "fetched"
        print(config.scripts_line(repo).replace("method scripts:",
                                                f"method scripts {state}:", 1))
    for ds in ids:
        with _quiet_stdout():
            if args.outputs:
                where = fetch_outputs(ds, data_path=args.data_path)
            else:
                where = fetch(ds, data_path=args.data_path) / ds
        print(f"{ds}: {where}")
    return _EXIT_OK


def _cmd_config(args) -> int:
    """``multibench config``: the paths and settings of ``mtb.config.DEFAULT``.

    One line per setting: name, resolved value and where the value came
    from (an environment variable, the conda probe or the default).
    ``--get NAME`` prints the bare value, for ``$(multibench config --get
    data_path)`` in a shell script.
    """
    from . import config
    rows = config._sources()
    if args.get:
        print(next(r["value"] for r in rows if r["name"] == args.get))
        return _EXIT_OK
    if args.format == "json":
        print(json.dumps([{**r, "value": str(r["value"])} for r in rows], indent=1))
        return _EXIT_OK
    width = max(len(r["name"]) for r in rows)
    for r in rows:
        print(f"{r['name']:<{width}}  {r['value']}")
        print(f"{'':<{width}}  ({r['source']})")
    return _EXIT_OK


def _yes_no(value) -> str:
    return "yes" if value else "no"


def _cmd_info(args) -> int:
    """``multibench info METHOD``: the facts of :func:`multibench.method_info`
    a user checks before running the method.

    Prints the env, how the method uses a GPU (``method_info(m)['gpu']``),
    whether labels are needed, which ATAC representation it reads, its
    variants (category and input roles), each observed runtime with its cell
    count, the host the times come from, and the setup hint. ``--format
    json`` prints the whole ``method_info`` dict.
    """
    import textwrap

    from . import discover
    info = discover.method_info(args.method)          # KeyError (did-you-mean)
    if args.format == "json":
        print(json.dumps(info, indent=1, default=str))
        return _EXIT_OK
    print(f"{info['id']} ({info.get('language') or '?'}), env {info['env']}")
    print(f"  GPU:          {info.get('gpu') or 'unknown'}")
    print(f"  needs_labels: {_yes_no(info.get('needs_labels'))}")
    print(f"  atac:         {info.get('atac') or 'none (reads no ATAC)'}")
    print("  variants:")
    for v in info.get("supports") or []:
        mods = "+".join(v.get("modalities") or []) or "(data_dir)"
        labels = f"  labels: {', '.join(v['labels'])}" if v.get("labels") else ""
        print(f"    {v.get('category'):9} {mods}{labels}")
    rt = info.get("runtime") or {}
    if rt:
        observed = sorted(rt.get("observed") or [], key=lambda o: -(o.get("sec") or 0))
        pad = " " * 16
        print(f"  runtime:      {rt.get('tier') or '?'}"
              + ("" if observed else " (never measured)"))
        for o in observed:
            cells = o.get("cells")
            size = f"{cells:,} cells" if isinstance(cells, int) else "cells not recorded"
            print(f"{pad}{o.get('sec')} s on {o.get('dataset')} ({size})")
        if observed and rt.get("note"):
            print(textwrap.fill(rt["note"], width=88, initial_indent=pad,
                                subsequent_indent=pad))
    hint = (info.get("setup_hint") or "").strip()
    print("  setup_hint:" + ("   none" if not hint else ""))
    if hint:
        print(textwrap.fill(hint, width=88, initial_indent="    ", subsequent_indent="    "))
    if info.get("scripts_url"):
        print(f"  scripts:      {info['scripts_url']}")
    print(f"  more:         multibench params {info['id']}; multibench cite {info['id']}")
    return _EXIT_OK


def _params_rows(method: str, category: str | None, modalities) -> list[dict]:
    """The :func:`multibench.params_for` dict of every variant of ``method``
    that matches ``category`` / ``modalities`` (both optional).

    The Python function needs the variant pinned down and raises ``KeyError``
    for a multi-variant method called without ``category``; on the command
    line an unspecified selector means "every matching variant", one block
    each, so one call is enough.
    """
    from . import discover
    from .engine import registry, upstream
    spec = registry.get(method)                       # KeyError (did-you-mean)
    registry.check_category(category)                 # ValueError listing the four
    want = registry.normalize_modalities(modalities) if modalities else None
    picked = [v for v in spec.variants
              if (category is None or v.when.get("category") == category)
              and (want is None or set(v.when.get("modalities", [])) == set(want))]
    if not picked:
        avail = [f"{v.when.get('category')}:{'+'.join(v.when.get('modalities', [])) or '(data_dir)'}"
                 for v in spec.variants]
        raise ValueError(f"{method}: no variant for category={category!r} "
                         f"modalities={list(want) if want else None}; available: {avail}")
    rows = []
    for v in picked:
        cat, mods = v.when.get("category"), list(v.when.get("modalities", []))
        try:
            rows.append(discover.params_for(method, cat, mods or None))
        except KeyError:
            # a category holding both a data_dir and a modality variant: build
            # the same dict straight from the variant
            up = upstream.knobs_for(spec.id)
            eff = {k: t.get("default") for k, t in v.tunable.items()} | dict(v.params)
            rows.append({"method": spec.id,
                         "variant": f"{cat}:{'+'.join(mods) or '(data_dir)'}",
                         "defaults": dict(v.params), "tunable": dict(v.tunable),
                         "effective": eff, "fixed_in_script": up["fixed_in_script"],
                         "upstream_knobs": up["upstream_knobs"],
                         "upstream_url": up["upstream_url"]})
    return rows


def _cmd_params(args) -> int:
    """``multibench params METHOD``: the hyperparameters of a method, per variant.

    Prints, for every variant matching ``--category`` / ``--modalities``
    (all variants when neither is given), the ``tunable`` table of
    :func:`multibench.params_for` - ``key``, ``type``, upstream ``default`` and
    the ``effective`` value a wrapper run uses - then the ``fixed_in_script``
    values (pinned upstream, with ``file:line``) and the names of the
    ``upstream_knobs`` the script does not expose. ``--format json`` dumps the
    full ``params_for`` dicts.
    """
    import pandas as pd
    rows = _params_rows(args.method, args.category, _csv_list(args.modalities))
    fmt = args.format
    if fmt == "json":
        print(json.dumps(rows, indent=1, default=str))
        return _EXIT_OK
    frames = []
    for p in rows:
        tun = p["tunable"] or {}
        eff = p["effective"] or {}
        tbl = pd.DataFrame([{"variant": p["variant"], "key": k,
                             "type": (t or {}).get("type"),
                             "default": (t or {}).get("default"),
                             "effective": eff.get(k)}
                            for k, t in sorted(tun.items())],
                           columns=["variant", "key", "type", "default", "effective"])
        if fmt in ("csv", "tsv"):
            frames.append(tbl)
            continue
        print(f"# {p['method']} {p['variant']}: {len(tun)} tunable parameter(s) "
              f"(set with --param KEY=VALUE)")
        if len(tun):
            _print_frame(tbl.drop(columns=["variant"]), fmt="table")
        else:
            print("  (none: the upstream script exposes no hyperparameter on its "
                  "command line)")
        fixed = p.get("fixed_in_script") or []
        if fixed:
            print("# fixed in the script (not tunable through the wrapper):")
            for f in fixed:
                print(f"  {f.get('name')} = {f.get('value')}   ({f.get('source')})")
        knobs = p.get("upstream_knobs") or []
        if knobs:
            print("# upstream library knobs the script does not expose: "
                  + ", ".join(str(k.get("name")) for k in knobs)
                  + (f"  (see {p['upstream_url']})" if p.get("upstream_url") else ""))
    if frames:
        _print_frame(pd.concat(frames, ignore_index=True), fmt=fmt)
    return _EXIT_OK


def _cmd_cite(args) -> int:
    """``multibench cite``: :func:`multibench.cite` for the benchmark + methods."""
    from . import discover
    if args.all and args.methods:
        _usage_error(args, "--all and explicit method ids are mutually exclusive")
    # `cite A B` and `cite A,B` both work: every other multi-id flag of the
    # CLI is comma-separated, so the positional list splits on commas too
    ids = [m for tok in (args.methods or []) for m in _csv_list(tok)]
    methods = "all" if args.all else (ids or None)
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
        # BatchResult.long is a property, not a method
        return load_batch(p).long
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


#: ``load_results`` names its keywords in its messages; on the command line
#: the same options are flags. Ordered: longest pattern first.
_API_TO_FLAGS = (
    (r"result_path= / mtb\.config\.DEFAULT\.result_path", "--result-path"),
    (r"result_path=<file>", "--result-path <file>"),
    (r"result_path=", "--result-path"),
    (r"\b(source|category)=['\"](\w+)['\"]", r"--\1 \2"),
    (r"\(or '(\w+)'\)", r"(or \1)"),
)


def _cli_spelling(message: str) -> str:
    """A ``load_results`` message with its keywords spelled as ``multibench
    plot`` flags (``result_path=`` -> ``--result-path``, ``source='rerun'``
    -> ``--source rerun``)."""
    import re
    for pattern, flag in _API_TO_FLAGS:
        message = re.sub(pattern, flag, message)
    return message


def _names(values) -> str:
    """``'MYCITE'`` / ``'D1, D2'``: the distinct values, in first-seen order."""
    return ", ".join(dict.fromkeys(str(v) for v in values))


def _filter_own_rows(frames: list, args) -> list:
    """Apply ``--dataset`` / ``--methods`` to the ``--input`` frames of ``plot``.

    Rows the filters drop are reported: all of them is an error naming the
    datasets / methods the rows are for, some of them a ``warning:`` naming
    what was dropped. ``--methods`` is checked here only when ``--category``
    also loads a stored table (the one case where it can silently remove
    every row of your own).
    """
    if not frames:
        return frames
    import pandas as pd
    whole = pd.concat(frames, ignore_index=True, sort=False)
    n = len(whole)
    datasets = _csv_list(args.dataset)
    methods = _csv_list(args.methods) if args.category is not None else None
    keep = pd.Series(True, index=whole.index)
    if datasets and "dataset" in whole.columns:
        keep &= whole["dataset"].astype(str).isin(datasets)
        if n and not keep.any():
            fix = ("plot them without --category, or score your method on "
                   f"{', '.join(datasets)}" if args.category is not None else
                   f"drop --dataset, or pass --dataset {_names(whole['dataset'])}")
            raise ValueError(
                f"your {n} rows are for dataset {_names(whole['dataset'])}; "
                f"--dataset {args.dataset} removed all of them ({fix})")
    if methods:
        by_method = whole["method"].astype(str).isin(methods)
        if (keep & ~by_method).any() and not (keep & by_method).any():
            raise ValueError(
                f"your rows are for method {_names(whole.loc[keep, 'method'])}; "
                f"--methods {args.methods} removed all of them (add "
                f"{_names(whole.loc[keep, 'method'])} to --methods)")
        keep &= by_method
    dropped = whole[~keep]
    if len(dropped):
        what = [f"method {_names(dropped['method'])}"]
        if "dataset" in dropped.columns:
            what.insert(0, f"dataset {_names(dropped['dataset'])}")
        flags = " and ".join(f for f, on in (("--dataset", datasets), ("--methods", methods))
                             if on)
        print(f"warning: {flags} dropped {len(dropped)} of your {n} rows "
              f"({'; '.join(what)})", file=sys.stderr)
    out, start = [], 0
    for f in frames:
        stop = start + len(f)
        out.append(f[keep.iloc[start:stop].to_numpy()])
        start = stop
    return out


def _cmd_plot(args) -> int:
    """``multibench plot {bubble,bar}``: draw a results table to ``--out``.

    The frame comes from ``--input`` (a long.csv / run_all dir; repeatable)
    and/or from :func:`multibench.load_results` (``--category``,
    ``--dataset``, ``--source``). Given both, the ``--input`` rows are
    concatenated onto the stored table - the shell equivalent of
    ``pd.concat`` - so a method evaluated with ``multibench evaluate
    --method/--dataset`` is drawn next to the stored table; a
    ``# overlay: ...`` note on stderr says how many rows came from where. ``--methods`` restricts the rows in every
    case; ``--dataset`` selects the stored table(s) and filters the inputs.
    ``--result-path`` is ``load_results(result_path=...)``: another results
    root, named in that spelling when a stored table is missing.
    """
    import pandas as pd
    from . import plot as plot_ns
    inputs = _csv_list(args.input) if isinstance(args.input, (list, tuple)) else (
        [args.input] if args.input else [])
    if not inputs and args.category is None:
        _usage_error(args, "need --category (stored results) or --input LONG_CSV")
    frames = []
    if args.category is not None:
        from . import load_results
        kw = dict(category=args.category, dataset=_csv_list(args.dataset),
                  source=args.source)
        if args.result_path:
            kw["result_path"] = args.result_path
        try:
            frames.append(load_results(**kw))
        except FileNotFoundError as e:
            # respell the API keywords as this command's flags
            raise FileNotFoundError(_cli_spelling(str(e))) from e
    own_rows = 0
    loaded = [_load_long_input(path) for path in inputs]
    for own in _filter_own_rows(loaded, args):
        own_rows += len(own)
        frames.append(own)
    if len(frames) == 1:
        df = frames[0]
    else:
        df = pd.concat(frames, ignore_index=True, sort=False)
        if args.category is not None and inputs:
            own_methods = sorted(set().union(*[set(f["method"].astype(str)) for f in frames[1:]]))
            print(f"# overlay: {own_rows} row(s) from --input (methods: "
                  f"{', '.join(own_methods)}) concatenated onto the stored "
                  f"{args.category} table ({len(frames[0])} rows, source={args.source})",
                  file=sys.stderr)
    methods = _csv_list(args.methods)
    if methods and args.kind == "bar":
        # plot.bar has no methods=; filter here with the same "unknown name"
        # error bubble raises
        present = sorted(df["method"].astype(str).unique())
        unknown = [m for m in methods if m not in present]
        if unknown:
            raise ValueError(f"unknown method(s) {unknown}; methods in the "
                             f"table: {present}")
        df = df[df["method"].isin(methods)]
    if len(df) == 0:
        raise ValueError("the results table is empty after filtering - nothing to plot")
    if df["method"].nunique() == 1 and args.category is None and args.kind == "bar":
        # --category frames come through load_results, which already warns
        # (and names the source that holds more methods); --input-only
        # frames never pass through it. plot.bubble warns about one method
        # itself, so only bar needs this line
        print("warning: only one method in this table; ranks and Overall bars "
              "are not meaningful with a single method", file=sys.stderr)
    title = args.title
    if title is None:
        title = " ".join(x for x in (args.category, args.dataset) if x) or None
    metrics = _csv_list(args.metrics)
    if args.kind == "bubble":
        kw = dict(metrics=metrics, methods=methods, aggregate=args.aggregate,
                  title=title, save=args.out, require_complete=args.require_complete,
                  na=getattr(args, "na", "warn"))
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
    prints the command line ``run`` would execute (``mtb.run(...,
    dry_run=True)``) and executes nothing.
    """
    import multibench
    params = _parse_params(args.param, args, default_method=args.method) or {}
    inputs = _parse_inputs(args.input, args)
    if args.dry_run:
        argv = multibench.run(args.method, args.category, inputs=inputs,
                              out_dir=args.out, params=params.get(args.method),
                              cmd_template=args.runner, dry_run=True)
        print("# dry run - nothing was executed; multibench run would execute:",
              file=sys.stderr)
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

    ``--dry-run`` prints the plan - the :func:`multibench.scan` frame
    ``run_all(dry_run=True)`` returns (one row per method variant: runnable,
    files_ok, env_ok, reason; compact columns - ``--columns all`` for every
    column) and, in table mode, the command line each variant would run (in
    csv/tsv/json it is the ``command`` column); nothing is executed or
    created. Otherwise the summary table is printed and everything is saved
    under ``--out-dir`` (reload with ``multibench plot bubble --input
    OUT_DIR``). Progress lines go to stderr; tables to stdout. ``--param
    METHOD:KEY=VALUE`` (repeatable) becomes ``params={METHOD: {KEY: value}}``.
    ``--out-dir`` is optional with ``--dry-run`` only: the commands then show
    the ``<out_dir>`` placeholder, as ``run_all(dry_run=True)`` does.
    ``--batch CSV`` becomes ``run_all(batch=CSV)``; a missing file exits 1
    before any method runs, and ``--dry-run`` reads the file as a check.
    """
    import multibench
    if args.out is None and not args.dry_run:
        # the text argparse prints for a missing required option
        _usage_error(args, "the following arguments are required: --out-dir/--out")
    params = _parse_params(args.param, args) or None
    columns = _csv_list(args.columns)
    if getattr(args, "assume_gpu", False) and not args.dry_run:
        _usage_error(args, "--assume-gpu applies to --dry-run only; a real run checks "
                           "this host's GPU")
    batch = getattr(args, "batch", None)
    batch_vec = None
    if batch is not None:
        from .eval import io as eio
        eio._require_file(batch, "--batch file")   # exit 1 before any method runs
        if args.dry_run:
            batch_vec = eio.as_vector(batch, what="batch")   # a file it cannot read fails
    if args.dry_run:
        from .workflow import _batch_length_problem, _dry_run_notes
        with _quiet_stdout():
            df = multibench.run_all(args.dataset, args.category, out_dir=args.out,
                                    methods=_csv_list(args.methods),
                                    modalities=_csv_list(args.modalities),
                                    data_path=args.data_path, params=params,
                                    dry_run=True, verbose=False,
                                    assume_gpu=getattr(args, "assume_gpu", False))
        k, n = int(df["runnable"].sum()), len(df)
        print(f"# dry run - nothing was executed; {k} of {n} variant(s) runnable on "
              f"{args.dataset} ({args.category}); commands below are what multibench "
              f"run would execute (rows with files_ok False have none)", file=sys.stderr)
        # the caveats of the rows the sweep would run: the compact table clips them
        scripts, lines = _dry_run_notes(df)
        if scripts:
            print(f"# {scripts}", file=sys.stderr)
        for m, cav in lines:
            print(f"# {m} {cav}", file=sys.stderr)
        if batch_vec is not None:
            bad = _batch_length_problem(df, batch_vec, args.dataset, args.category,
                                        args.data_path)
            if bad:          # the scoring of the real run would fail: exit 1 now
                raise ValueError(bad)
        _print_frame(df, columns=columns, fmt=args.format, compact=_compact_plan_columns(df))
        if args.format == "table" and not columns:
            have = df[df["command"].astype(str).str.len() > 0]
            print()
            from .engine.runner import _PREPARED_PREFIX
            print(f"# commands ({len(have)} variant(s) with resolvable inputs; "
                  f"'[env missing]' = blocked by env_ok only; '[use multibench run]' = "
                  f"reads a file under inputs/ that `multibench run` writes first)")
            for _, r in have.iterrows():
                tag = "" if r["env_ok"] else " [env missing]"
                if _PREPARED_PREFIX in str(r.get("caveat", "")):
                    tag += " [use multibench run]"
                print(f"{r['method']} ({r['modalities']}){tag}: {r['command']}")
        return _EXIT_OK
    with _quiet_stdout(), _leiden_flavor(getattr(args, "leiden_flavor", None)):
        res = multibench.run_all(args.dataset, args.category, out_dir=args.out,
                               methods=_csv_list(args.methods),
                               modalities=_csv_list(args.modalities),
                               data_path=args.data_path, params=params,
                               evaluate=not args.no_evaluate, dry_run=False,
                               timeout=args.timeout, skip_existing=args.skip_existing,
                               batch=batch)
    _print_frame(res.summary, columns=columns, fmt=args.format)
    print(f"saved under {args.out}", file=sys.stderr)
    return _EXIT_OK


@contextlib.contextmanager
def _leiden_flavor(flavor):
    """Set ``mtb.config.DEFAULT.leiden_flavor`` for one command (``None`` = unchanged)."""
    from . import config
    if flavor is None:
        yield
        return
    before = config.DEFAULT.leiden_flavor
    config.DEFAULT.leiden_flavor = flavor
    try:
        yield
    finally:
        config.DEFAULT.leiden_flavor = before


def _evaluate_labels(args, stack):
    """The ``labels=`` value of ``multibench evaluate``.

    One ``--labels`` file is passed as a path; several are passed as a list,
    stacked in the given order with each file one batch (as
    ``mtb.evaluate(labels=[...])``). ``--column`` picks the column of every
    file. Without ``--labels``, ``--dataset/--method/--category`` read the
    dataset's label files in the method's cell order (``mtb.labels_for``).
    """
    import multibench
    files = list(args.labels or [])
    if not files:
        if args.dataset is None or args.method is None or args.category is None:
            _usage_error(args, "need --labels CSV (repeatable), or --dataset, --method "
                         "and --category to read the dataset's label files")
        from .engine import registry
        try:
            registry.check_method(args.method)
        except KeyError as e:
            # the label order comes from a registry method; the user's own
            # method passes its label files instead
            raise KeyError(f"{e.args[0]}; for your own method, pass the label files "
                           f"with --labels, once per file, in your embedding's cell "
                           f"order") from None
        labels = multibench.labels_for(args.dataset, args.category, args.method,
                                       data_path=args.data_path)
        print(f"# labels: {', '.join(Path(v).name for v in labels.values())} from "
              f"{args.dataset}, in {args.method}'s cell order", file=sys.stderr)
        if args.column is None:
            return labels
        files = list(labels.values())
    elif args.data_path is not None:
        _usage_error(args, "--data-path reads the labels from the dataset folder; "
                     "drop it when --labels names the files")
    if args.column is not None:
        # one column out of each file, written as the one-column files the
        # stacking reader expects, so several files still count as batches
        import tempfile

        import pandas as pd
        from .eval import io as eio
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        picked = []
        for i, f in enumerate(files, 1):
            # the file keeps its name: evaluate's errors name the files typed
            out = tmp / str(i) / Path(f).name
            out.parent.mkdir()
            pd.DataFrame({"x": eio.read_labels(f, args.column)}).to_csv(out, index=False)
            picked.append(str(out))
        files = picked
    return files[0] if len(files) == 1 else files


def _cmd_evaluate(args) -> int:
    """``multibench evaluate``: :func:`multibench.evaluate` on an embedding file.

    Default output is the wide ``metric.csv`` shape (index = metric, one
    ``Value`` column). With ``--method`` and ``--dataset`` (and ``--category``)
    the frame is reshaped with :func:`multibench.to_long` to the long
    ``metric,value,method,dataset,category,clustering,source,scored_with`` table that
    ``plot --input`` reads and ``load_results`` returns. Without
    ``--metrics`` the metric set is Python's default: clustering, plus the
    batch family when ``--batch`` or several ``--labels`` are given.
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
    # one metric-selection knob: --metrics (a family token or a comma list of
    # codes); --only and --task are older spellings of it
    metrics = _csv_list(args.metrics)
    if metrics is not None and len(metrics) == 1 and metrics[0] in _METRIC_FAMILIES:
        metrics = metrics[0]
    if getattr(args, "only", None) is not None:
        print("warning: --only is deprecated; use --metrics", file=sys.stderr)
        if metrics is None:
            metrics = _csv_list(args.only)
    if args.task is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(f"--task is deprecated; use --metrics {args.task}",
                          DeprecationWarning, stacklevel=2)
        if metrics is None:
            metrics = args.task
    from .eval import pipeline
    with contextlib.ExitStack() as stack:
        labels = _evaluate_labels(args, stack)
        kw = dict(category=args.category, labels=labels, clustering=args.cluster,
                  batch=getattr(args, "batch", None))
        if args.obsm is not None:
            kw["obsm"] = args.obsm
        # labels read from --dataset: a count error names that dataset's files
        token = pipeline._CLI_LABELS_FROM.set(None if args.labels else args.dataset)
        try:
            with _quiet_stdout(), _leiden_flavor(args.leiden_flavor):   # progress -> stderr
                df = multibench.evaluate(output=args.output, metrics=metrics, **kw)
        finally:
            pipeline._CLI_LABELS_FROM.reset(token)
    if long_mode:
        df = multibench.to_long(df, method=args.method, dataset=args.dataset,
                                category=args.category)
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


def _size_total_line(rows, sizes: dict, what: str = "download", *,
                     flavor: str | None = None) -> str:
    """The ``# total`` lines for ``rows``: download and on-disk sums.

    Every size known: one line, ``# total for N envs (GPU builds): X GB
    download, Y GB on disk``. A column with an unknown size is not summed as
    if complete: the disk figure is left out, a download figure reads ``at
    least X GB``, and a second line counts the unknowns per column (a row
    printing ``? disk`` is one disk unknown even when its download size is
    known), with the disk sum of the others as a lower bound. An unknown
    disk size adds the advice to check with ``du`` after the first install.
    No line holds more than two semicolons.

    Parameters
    ----------
    rows : list of dict
        ``env`` per row; a ``flavor`` of ``'cpu'`` keys the row's size on
        the ``'<env>-cpu'`` archive (``multibench.env.archive_key``), any
        other value on ``'<env>'``.
    sizes : dict
        :func:`multibench.env.packed_sizes` (or a replacement in tests).
    what : str
        The word after the download figure (``'download'`` / ``'to download'``).
    flavor : str, keyword-only, optional
        The flavour the caller asked for (``'auto'``, ``'cpu'``, ``'gpu'``):
        when given, the first line names the builds it summed and how many
        envs fell back to the GPU build; ``'auto'`` adds why this host got
        that flavour. ``None`` omits the parenthesis.

    Returns
    -------
    str
        One ``#`` line, or two joined by a newline (stderr on the CLI).
    """
    from .engine import envs
    dl = disk = 0
    n = n_dl = n_disk = 0
    gpu_rows = 0
    for r in rows:
        n += 1
        key = envs.archive_key(r["env"], r.get("flavor"))
        sz = sizes.get(key) or {}
        a, u = sz.get("archive_bytes"), sz.get("unpacked_bytes")
        if a is not None:
            n_dl += 1
            dl += a
        if u is not None:
            n_disk += 1
            disk += u
        gpu_rows += r.get("flavor") == "gpu"

    def _envs(k: int) -> str:
        return f"{k} env{'s' if k != 1 else ''}"

    def _which(k: int) -> str:
        if n == 1:
            return "this env"
        if k == n:
            return "both envs" if n == 2 else f"all {n} envs"
        return f"{k} of {n} envs"

    line = f"# total for {_envs(n)}"
    if flavor is not None:
        eff = envs.resolve_flavor(flavor)
        s = "s" if n != 1 else ""
        if eff == "cpu" and n and gpu_rows == n:
            builds = (f"GPU build{s}; no CPU build is published for "
                      + ("it" if n == 1 else "these envs"))
        elif eff == "cpu":
            builds = f"CPU build{s}"
            if flavor == "auto":
                builds += ", as this host has no NVIDIA GPU"
            if gpu_rows:
                builds += (f"; {_envs(gpu_rows)} "
                           f"{'has' if gpu_rows == 1 else 'have'} only a GPU build")
        else:
            builds = f"GPU build{s}"
            if flavor == "auto":
                builds += ", as this host has an NVIDIA GPU"
        line += f" ({builds})"
    if n_dl == n:
        line += f": {envs._gb(dl)} {what}"
    elif n_dl:
        line += f": at least {envs._gb(dl)} {what}"
    else:
        line += ": download size not recorded"
    if n_disk == n:
        line += f", {envs._gb(disk)} on disk"
    unknown = []
    if 0 < n_dl < n:            # none known: the first line already says so
        unknown.append(f"download size not recorded for {_which(n - n_dl)}")
    if n_disk < n:
        # the known part is a lower bound, e.g. for a storage-quota request
        known = f" ({envs._gb(disk)} for the other {n_disk})" if n_disk else ""
        unknown.append(("size on disk for " if unknown else "size on disk not recorded for ")
                       + _which(n - n_disk) + known)
    if not unknown:
        return line
    second = "# " + ", ".join(unknown)
    if n_disk < n:
        second += ("; unpacked envs are larger than the download, so check with du "
                   "after the first install")
    return line + "\n" + second


def _flavor_token(flavor) -> str:
    """`` flavor=cpu`` for a row whose installed flavour is recorded, else ``''``."""
    return f" flavor={flavor}" if flavor in ("cpu", "gpu") else ""


def _cmd_env(args) -> int:
    """``multibench env ...``: environment recipes, preflight and installation.

    ``status`` / ``plan`` / ``doctor`` / ``install`` first print a
    ``warning:`` on stderr when this host cannot build method envs
    (non-Linux; see :func:`multibench.env.host_platform_problem`); ``install
    --run`` then refuses unless ``--force``.
    """
    from .engine import envs, registry
    cmd = args.env_cmd
    refusing = (cmd == "install" and getattr(args, "run", False)
                and not getattr(args, "force", False))
    if cmd in ("status", "plan", "doctor", "install") and not refusing:
        _platform_note()                     # once, first, on stderr
    if cmd == "status":
        _mlist = _csv_list(getattr(args, "methods", None))
        _cat = getattr(args, "category", None)
        keep = None
        if _mlist:
            keep = set(registry.check_method(m) for m in _mlist)
        elif _cat:
            keep = set(registry.list_methods(category=_cat))
        seen_tags = []
        for r in envs.status():
            if keep is not None and r["method"] not in keep:
                continue
            mark = envs.env_mark(r["exists"], r["has_lock"])   # same marks as doctor
            tag = r["difficulty"] + ("*" if r["verified_working"] else "")
            if r["difficulty"] not in seen_tags:
                seen_tags.append(r["difficulty"])
            print(f"[{mark}] {r['method']:16} {r['env']:18} {tag}"
                  f"{_flavor_token(r.get('flavor'))}")
        # the legend is a note, so stderr: stdout stays one line per method
        print(f"# legend: {envs.MARK_LEGEND};  tag = difficulty of building "
              "the env: " + "; ".join(f"{t} = {envs.DIFFICULTY.get(t, '?')}"
                                       for t in seen_tags)
              + f";  {envs.VERIFIED_STAR}", file=sys.stderr)
        return _EXIT_OK
    if cmd == "groups":
        for name, spec in envs.groups().items():
            if spec.get("shared"):
                print(f"{name:16} ({len(spec['members']):2}): {', '.join(spec['members'])}")
        return _EXIT_OK
    if cmd == "plan":
        _mlist = _csv_list(getattr(args, "methods", None))
        flavor = getattr(args, "flavor", "auto")
        sizes = envs.packed_sizes()
        manifest = envs.packed_manifest()
        rows = envs.plan(category=getattr(args, "category", None), methods=_mlist)
        # sizes follow the archive --flavor selects per env (its fallback to
        # the GPU build included); the total line says which flavour it summed
        summed = []
        for p in rows:
            tag = "shared" if p["shared"] else "own"
            key, eff = envs.archive_for(p["env"], flavor, manifest=manifest, sizes=sizes)
            summed.append({**p, "flavor": eff})
            sz = sizes.get(key) or {}
            print(f"{p['env']:18} [{tag:6}] {envs._gb(sz.get('archive_bytes')):>8} dl "
                  f"{envs._gb(sz.get('unpacked_bytes')):>8} disk <- {', '.join(p['methods'])}"
                  f"{_flavor_token(p.get('flavor'))}")
        print(_size_total_line(summed, sizes, flavor=flavor), file=sys.stderr)
        note = envs.auto_flavor_note(flavor, planning=True)
        if note and any(p["flavor"] == "cpu" for p in summed):
            print(note, file=sys.stderr)
        return _EXIT_OK
    if cmd == "doctor":
        _mlist = _csv_list(getattr(args, "methods", None))
        rows = envs.doctor(category=getattr(args, "category", None), methods=_mlist)
        for r in rows:
            mark = envs.env_mark(r["exists"], r["has_lock"])
            print(f"[{mark}] {r['env']:18} ({len(r['methods']):2}) <- {', '.join(r['methods'])}"
                  f"{_flavor_token(r.get('flavor'))}")
        missing = [r for r in rows if not r["exists"]]
        nolock = [r["env"] for r in missing if not r["has_lock"]]
        print(f"# {len(rows)} envs needed, {len(missing)} missing"
              + (f"; no lockfile for: {', '.join(nolock)}" if nolock else ""))
        print(f"# legend: {envs.MARK_LEGEND}")
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
        force = getattr(args, "force", False)
        flavor = getattr(args, "flavor", "auto")
        # mtb.env.install(): a RuntimeError (no conda here, a failed build, a
        # non-Linux host) propagates to main(): "error: ..." on stderr, exit 1
        # - never an error line on stdout; "[env] unpacking ..." -> stderr
        with _quiet_stdout():
            rows = envs.install(_mlist, category=getattr(args, "category", None),
                                packed=packed, dry_run=not do_run, force=force,
                                flavor=flavor)
        width = max([14] + [len(r["state"]) for r in rows])
        for r in rows:
            extra = ""
            if r["state"] == "packed archive published":
                extra = (f" {envs._gb(r.get('archive_bytes')):>8} dl "
                         f"{envs._gb(r.get('unpacked_bytes')):>8} disk  {r['packed_url']}")
            elif r["state"] in ("have", "PACKED"):
                extra = _flavor_token(r.get("flavor"))
            print(f"{r['env']:18} [{r['state']:{width}}] <- {', '.join(r['methods'])}{extra}")
        if not do_run:
            print("# dry-run - add --run to create the missing envs"
                  + (" (packed archives first, lockfile build otherwise)"
                     if packed else " from their lockfiles"), file=sys.stderr)
            if packed:
                todo = [r for r in rows if not r["exists"] and r.get("packed_url")]
                # mtb.env.install printed why 'auto' took the CPU builds; the
                # total then names the builds without repeating the reason
                said = envs.auto_flavor_note(flavor) is not None
                print(_size_total_line(todo, envs.packed_sizes(), what="to download",
                                       flavor=envs.resolve_flavor(flavor) if said
                                       else flavor), file=sys.stderr)
        return _EXIT_OK
    if cmd == "freeze":
        if getattr(args, "all", False):
            for env in envs.required_envs(category=getattr(args, "category", None)):
                try:
                    print(f"froze {env} -> {envs.freeze(env)}")
                except Exception as e:  # noqa: BLE001 - report per-env, keep going
                    print(f"skipped {env}: {str(e)[:120]}")
        else:
            if not getattr(args, "env", None):
                _usage_error(args, "name an env to freeze, or pass --all")
            print(f"froze {args.env} -> {envs.freeze(args.env)}")
        return _EXIT_OK
    if cmd == "create-group":
        with _quiet_stdout():
            cmds = envs.create_group(args.group, dry_run=not getattr(args, "run", False),
                                     force=getattr(args, "force", False))
        if not getattr(args, "run", False):
            print("# dry-run - add --run to execute:", file=sys.stderr)
            for c in cmds:
                print(shlex.join(c))
        else:
            print(f"created group env {args.group}")
        return _EXIT_OK
    method = getattr(args, "method", None)
    # one resolver for every per-method env command, so recipe/yml/create build
    # the env name that scan()['env'], run() and env doctor/plan/install expect
    expected = envs.default_env_name(method)      # KeyError (did-you-mean) on a typo
    name = getattr(args, "name", None) or expected
    if name == expected:
        banner = (f"# env {name!r} is the name scan/run/env doctor expect for {method} "
                  f"(`multibench info {method}` shows it); `multibench env create "
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
            cmds = envs.create(method, env_name=name, dry_run=not getattr(args, "run", False),
                               force=getattr(args, "force", False))
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
class _HelpFormatter(argparse.HelpFormatter):
    """argparse's formatter, except that a text holding line breaks (a recipe
    epilog) is printed line by line instead of being re-wrapped."""

    def _fill_text(self, text, width, indent):
        if "\n" in text:
            return "".join(indent + line for line in text.splitlines(keepends=True))
        return super()._fill_text(text, width, indent)


def _status_epilog() -> str:
    """The ``env status`` help epilog: one difficulty tag per line."""
    import textwrap

    from .engine.envs import DIFFICULTY, MARK_LEGEND, VERIFIED_STAR
    rows = [(tag, text) for tag, text in DIFFICULTY.items()]
    star, _, star_text = VERIFIED_STAR.partition(" = ")
    field, _, what = star_text.partition(": ")
    rows.append((star, f"after a tag: {what} ({field})"))
    width = max(len(tag) for tag, _ in rows) + 2
    lines = ["difficulty tags (how hard the environment is to build from its recipe):"]
    for tag, text in rows:
        lines += textwrap.wrap(text, width=78, initial_indent=f"  {tag:<{width}}",
                               subsequent_indent=" " * (width + 2),
                               break_on_hyphens=False)
    lines += [""] + textwrap.wrap("marks: " + MARK_LEGEND, width=78,
                                  subsequent_indent="  ", break_on_hyphens=False)
    lines.append("The same legend is printed on stderr.")
    return "\n".join(lines)


_CATEGORY_HELP = ("integration category: vertical (several modalities measured in the "
                  "same cells, e.g. CITE-seq), diagonal (modalities measured in different "
                  "cells, no pairing), mosaic (several batches, only some share a "
                  "modality) or cross (several batches, each with RNA and ADT).")
_TASK_HELP = ("task within the category: clustering, batch or dimension_reduction. "
              "mtb.list_tasks() lists them.")
_METHODS_HELP = "comma-separated method ids (as printed by `multibench list`)"
_FLAVORS = ("auto", "cpu", "gpu")        # mtb.env.FLAVORS (module imported lazily)
_FLAVOR_HELP = ("which packed archive to take per env: 'cpu' = the '<env>-cpu' archive "
                "(the same env without the CUDA libraries, 3-4x smaller; the GPU build "
                "with a warning where no CPU archive is published yet), 'gpu' = the "
                "full CUDA build, 'auto' (default) = 'cpu' when no NVIDIA GPU is "
                "visible on this host (mtb.env.host_has_gpu), 'gpu' otherwise. The env "
                "name is the same whatever the flavour; env status/doctor show which "
                "flavour is installed. Installing on a login node for jobs on GPU "
                "nodes: pass gpu")
_FORCE_HELP = ("try anyway on a computer that is not Linux (method environments run "
               "only on Linux; without --force the command refuses there before any "
               "download)")
#: the family tokens ``--metrics`` accepts besides a comma list of codes
_METRIC_FAMILIES = ("clustering", "batch", "all")


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
    pl = sub.add_parser("list", help="list method ids (mtb.list_methods; --task/--runnable "
                                     "are mtb.find_methods filters)",
                        description="Print registry method ids, one per line.")
    pl.add_argument("--category", help=_CATEGORY_HELP)
    pl.add_argument("--task", help=_TASK_HELP + " The same filter as "
                                               "mtb.find_methods(task=).")
    pl.add_argument("--runnable", action="store_true",
                    help="only methods with a declared variant (usable by run; "
                         "mtb.find_methods(runnable=True))")
    pl.set_defaults(func=_cmd_list, _parser=pl)

    # ---- find
    pf = sub.add_parser(
        "find", help="find methods by category/modalities/labels/ATAC (mtb.find_methods)",
        description="Print method ids matching all given filters, one per line. "
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
                         "no filter. Note: the optional value must come right after "
                         "the flag")
    pf.add_argument("--atac", choices=["peak", "gene_activity"],
                    help="filter by the ATAC representation the method consumes: "
                         "peak (chr:start-end matrix) or gene_activity (gene scores)")
    pf.add_argument("--runnable", action="store_true",
                    help="only methods with a declared variant (usable by run)")
    pf.add_argument("--tunable", nargs="?", const="true", choices=["true", "false"],
                    metavar="{true,false}",
                    help="true (or the bare flag): only methods exposing hyperparameters "
                         "to --param (`multibench params METHOD` lists them); false: "
                         "only methods that hardcode them; absent: no filter")
    pf.set_defaults(func=_cmd_find, _parser=pf)

    # ---- params
    ppa = sub.add_parser(
        "params", help="the hyperparameters a method accepts (mtb.params_for)",
        description="Print, per variant of METHOD, the tunable hyperparameters "
                    "(key, type, upstream default, effective value in a wrapper run) "
                    "that --param / run(params=) accept, then the values the script "
                    "pins (fixed_in_script, with file:line) and the upstream library "
                    "knobs it does not expose. Without --category/--modalities every "
                    "variant is printed. An empty tunable table means the script "
                    "hardcodes its hyperparameters.")
    ppa.add_argument("method", help="method id (see `multibench list`; unknown id -> "
                                    "did-you-mean error)")
    ppa.add_argument("--category", help=_CATEGORY_HELP + " Only that category's variants.")
    ppa.add_argument("--modalities", help="comma-separated modality roles selecting one "
                                          "variant, e.g. rna,adt ('protein' is accepted "
                                          "for adt)")
    ppa.add_argument("--format", choices=["table", "csv", "tsv", "json"], default="table",
                     help="table (default: one block per variant) | csv/tsv (one row per "
                          "tunable key with a variant column) | json (the full "
                          "mtb.params_for dicts, one per variant)")
    ppa.set_defaults(func=_cmd_params, _parser=ppa)

    # ---- info
    pin = sub.add_parser(
        "info", help="one method's env, GPU and label needs, ATAC input, variants and "
                     "runtime (mtb.method_info)",
        description="Print what to check before running METHOD: its env, how it uses "
                    "a GPU (required, used when present, not used or unknown), whether "
                    "it needs labels, which ATAC representation it reads, its variants "
                    "(category and input roles), each observed runtime with its cell "
                    "count and host, and the setup hint.")
    pin.add_argument("method", help="method id (see `multibench list`)")
    pin.add_argument("--format", choices=["text", "json"], default="text",
                     help="text (default) or json (the whole mtb.method_info dict)")
    pin.set_defaults(func=_cmd_info, _parser=pin)

    # ---- fetch
    pfe = sub.add_parser(
        "fetch", help="download demo datasets, stored outputs or the method scripts "
                      "(mtb.data.fetch / mtb.data.fetch_outputs)",
        description="Download the demo datasets named into the data root (skipping "
                    "those present), or with --outputs their stored run-all outputs. "
                    "--scripts fetches the method scripts the first run would fetch, so "
                    "compute nodes without network need no download.")
    pfe.add_argument("datasets", nargs="*", metavar="DATASET",
                     help="dataset ids, space- or comma-separated (mtb.data.fetchable() "
                          "lists them)")
    pfe.add_argument("--outputs", action="store_true",
                     help="download the stored run-all outputs of each dataset instead "
                          "(D11, D28, D46, D52; mtb.data.fetch_outputs), under "
                          "<data root>/outputs/")
    pfe.add_argument("--scripts", action="store_true",
                     help="fetch the method scripts (PYangLab/scMultiBench, with git) "
                          "into repo_path, or report them present; prints the commit")
    pfe.add_argument("--ref", metavar="REF",
                     help="with --scripts: fetch this commit or tag instead of the "
                          "default branch (default: $MULTIBENCH_SCRIPTS_REF). Scripts "
                          "already present must be at that commit or tag.")
    pfe.add_argument("--data-path", dest="data_path",
                     help="data root to download into (default: the configured "
                          "data_path, see `multibench config`)")
    pfe.set_defaults(func=_cmd_fetch, _parser=pfe)

    # ---- config
    pco = sub.add_parser(
        "config", help="the resolved paths and where each came from (mtb.config.DEFAULT)",
        description="Print data_path, envs_dir, repo_path, result_path and "
                    "leiden_flavor with the source of each value: an environment "
                    "variable (MULTIBENCH_DATA_PATH, MULTIBENCH_ENVS_DIR, "
                    "MULTIBENCH_REPO_PATH), the conda probe or the default. "
                    "scripts_commit is the commit of the method scripts in "
                    "repo_path.")
    pco.add_argument("--get", choices=["data_path", "envs_dir", "repo_path", "result_path",
                                       "leiden_flavor"],
                     help="print only this value, e.g. DATA=$(multibench config --get "
                          "data_path)")
    pco.add_argument("--format", choices=["text", "json"], default="text",
                     help="text (default) or json (name, value, source per setting)")
    pco.set_defaults(func=_cmd_config, _parser=pco)

    # ---- scan
    ps = sub.add_parser(
        "scan", help="which methods can run on a dataset, and why not the rest (mtb.scan)",
        description="Print the preflight table of mtb.scan: one row per method variant "
                    "of the category with its two checks (files_ok, env_ok), the "
                    "runnable verdict and the reason. The table shows a compact column "
                    "set (" + ", ".join(_COMPACT_PLAN_COLUMNS) + "; plus atac when a "
                    "variant reads ATAC and caveat when a row has one; long text "
                    f"clipped to {_TRUNCATE_WIDTH} chars); --columns all (or any of "
                    "--format csv/tsv/json) gives every column mtb.scan returns: "
                    "category, env, output_kind, n_tunable, observed_worst_sec, caveat, "
                    "files_reason, env_reason, needs_labels, atac ...")
    ps.add_argument("dataset", help="dataset id = the folder name under --data-path "
                                   "(e.g. D11, or MYCITE for your own data)")
    ps.add_argument("--category", help=_CATEGORY_HELP + " Default: every category, "
                                                        "as mtb.scan(dataset) does.")
    ps.add_argument("--data-path", dest="data_path",
                    help="folder that contains the dataset folder (default: see "
                         "`multibench config`)")
    ps.add_argument("--methods", help=_METHODS_HELP + "; only those rows (unknown "
                                                      "id -> did-you-mean error)")
    ps.add_argument("--modalities", help="comma-separated modality roles to restrict the "
                                         "variants to, e.g. rna,adt ('protein' is "
                                         "accepted for adt); atac_peak and atac_gas "
                                         "select by what the method reads")
    ps.add_argument("--columns", help="comma-separated columns to print, in this order, "
                                      "or 'all' for every column (default: the compact "
                                      "set in table mode, all columns for csv/tsv/json; "
                                      "an unknown name lists the available ones)")
    ps.add_argument("--format", choices=["table", "csv", "tsv", "json"], default="table",
                    help="output format (default table = aligned text, compact and "
                         "clipped; csv/tsv/json = every column, never clipped, for "
                         "scripts; json = a list of row objects)")
    ps.add_argument("--strict", action="store_true",
                    help="exit 1 when no requested row is runnable; with --methods, "
                         "when any named method has none; also when the method "
                         "scripts are not fetched (for scripts: multibench "
                         "scan DS --category C --strict && sbatch ...)")
    ps.add_argument("--assume-gpu", dest="assume_gpu", action="store_true",
                    help="skip this host's GPU test. Use it on a login node without a "
                         "GPU to check a job for a GPU node (mtb.scan(assume_gpu=True))")
    ps.set_defaults(func=_cmd_scan, _parser=ps)

    # ---- layout
    play = sub.add_parser(
        "layout", help="how to lay out your dataset on disk (mtb.describe_layout)",
        description="Print the directory layout and role -> filename contract the "
                    "package expects, optionally for one category.")
    play.add_argument("category", nargs="?", help=_CATEGORY_HELP + " Optional: "
                      "without it, the layout of every category is printed.")
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
                    "--atac/--labels/--batch/--batch-index switches to export_dataset, which "
                    "reads SRC (.h5ad or .h5mu) and writes OUT/ as a dataset folder "
                    "ready for `multibench scan OUT_NAME --data-path <parent>`. "
                    "Give raw counts: the methods normalise the data themselves.",
        epilog=_CONVERT_EPILOG, formatter_class=_HelpFormatter)
    pc.add_argument("src", help="input: .h5ad, .h5mu (then --mod or mod: selectors), "
                               ".csv/.tsv (cells x features), .loom, or an already "
                               "canonical .h5 (copied to OUT, --dtype honoured; "
                               "'already canonical - nothing written' when OUT is SRC)")
    pc.add_argument("out", help="output .h5 file (mode 1; or an existing directory - or a "
                               "path ending in / - with --modality) or the dataset "
                               "folder to create (mode 2)")
    pc.add_argument("--category", help=_CATEGORY_HELP + " Sets the file names. "
                    "vertical writes atac.h5. diagonal writes atac_peak.h5 or "
                    "atac_gas.h5, and the labels as rna_cty.csv and atac_cty.csv. "
                    "mosaic writes atac<i>.h5.")
    pc.add_argument("--modality", help="mode 1: rna | adt | atac | atac_peak | atac_gas "
                                       "(aliases protein, peak, gas/gene_activity); "
                                       "validated, picks the filename when OUT is a "
                                       "directory and checks ATAC feature names")
    pc.add_argument("--layer", help="mode 1: take the matrix from adata.layers[LAYER] "
                                    "instead of .X")
    pc.add_argument("--obsm", help="mode 1: take the matrix from adata.obsm[OBSM] "
                                   "(e.g. protein for CITE-seq ADT)")
    pc.add_argument("--mod", help="mode 1: for .h5mu input, the modality to export")
    pc.add_argument("--rna", help="mode 2: where the raw-count RNA matrix lives: X, "
                                  "obsm:<key>, layer:<key>, mod:<name> (no default - omit "
                                  "to skip RNA); append [<var column>=<value>] to keep "
                                  "some features, e.g. \"X[feature_types=Gene Expression]\"")
    pc.add_argument("--adt", help="mode 2: where the ADT/protein matrix lives "
                                  "(same grammar, e.g. obsm:protein)")
    pc.add_argument("--atac", help="mode 2: where the ATAC matrix lives (same grammar, "
                                   "e.g. \"X[feature_types=Peaks]\"); requires --atac-kind")
    pc.add_argument("--atac-from", dest="atac_from", metavar="PATH",
                    help="mode 2, --category diagonal: a second .h5ad/.h5mu with the "
                         "ATAC cells; --atac then selects from it (default X)")
    pc.add_argument("--atac-kind", dest="atac_kind", choices=["peak", "gene_activity"],
                    help="mode 2: peak -> atac_peak.h5 (+ atac.h5 without --category); "
                         "gene_activity -> atac_gas.h5; --category vertical writes atac.h5 "
                         "and mosaic atac<i>.h5 for both")
    pc.add_argument("--labels", help="mode 2: cell-type column, obs:<col> (the MuData's "
                                     "global obs) or <mod>:<col> -> cty.csv (diagonal: "
                                     "rna_cty.csv / atac_cty.csv)")
    pc.add_argument("--batch", help="mode 2: batch column (same grammar); cells are "
                                    "split per batch into numbered files rna1.h5, "
                                    "rna2.h5, cty1.csv ..., read by mosaic and cross methods")
    pc.add_argument("--batch-index", dest="batch_index", type=int, metavar="N",
                    help="mode 2, --category mosaic or cross: write the whole file "
                         "as batch N (rna<N>.h5, adt<N>.h5, atac<N>.h5, cty<N>.csv); "
                         "one call per batch file, see the recipe below "
                         "(mtb.io.export_dataset(batch_index=N))")
    pc.add_argument("--overwrite", action="store_true",
                    help="mode 2: replace files already in OUT (default: refuse and "
                         "list them)")
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
                     help="method ids to cite, space- or comma-separated (`cite A B` "
                          "and `cite A,B` are the same; none: the benchmark entry only)")
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
        description="Plot stored result tables (--category [--dataset] [--source]), "
                    "your own long table (--input long.csv or a run_all output "
                    "dir, as written by `multibench evaluate --method/--dataset` "
                    "and `multibench run-all`), or both: --input together with "
                    "--category adds your rows to the stored table.")
    pp.add_argument("kind", choices=["bubble", "bar"],
                    help="bubble: paper-style bubble table (methods x metrics, best "
                         "first, Overall bars per family) | bar: one bar per method "
                         "with its across-dataset Overall score")
    pp.add_argument("--category", help=_CATEGORY_HELP + " Selects the stored results. "
                                                        "Required unless --input is given.")
    pp.add_argument("--dataset", help="dataset id(s), comma-separated; with --category "
                                      "selects the stored table(s), with --input "
                                      "filters the frame")
    pp.add_argument("--source", choices=["published", "rerun"], default="published",
                    help="stored table to load: published (the scIB tables shipped "
                         "with the package, default) or rerun (the package's re-run "
                         "sweeps)")
    pp.add_argument("--result-path", dest="result_path", metavar="DIR",
                    help="with --category: another results root to read the stored "
                         "tables from (holding scib_metric/ and/or rerun/; default: "
                         "the tables shipped with the package)")
    pp.add_argument("--input", action="append", metavar="LONG_CSV_OR_DIR",
                    help="a long results CSV (metric,value,method,dataset,category[,clustering,source]) or "
                         "a run_all output directory; repeatable. Alone: the table to "
                         "plot. With --category: concatenated onto the stored table "
                         "(a '# overlay: ...' note on stderr says how many rows came "
                         "from where); --dataset then filters these rows too")
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
                         "every dataset instead of warning")
    pp.add_argument("--na", choices=["warn", "skip", "raise"], default="warn",
                    help="bubble only: a metric not computed for a method is drawn as "
                         "a dash; warn = name those cells (default), skip = draw "
                         "without the warning, raise = refuse an incomplete table")
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
        "run", help="run one method on explicit input files (mtb.run)",
        description="Run one method in its conda env on the given role=path inputs "
                    "and write its outputs under --out-dir. Use `multibench scan` "
                    "first to see which roles a method needs and whether its env is "
                    "installed; `multibench run-all` runs every method of a category "
                    "on a laid-out dataset folder.")
    pr.add_argument("--method", required=True, help="method id (see `multibench list`)")
    pr.add_argument("--category", required=True, help=_CATEGORY_HELP)
    pr.add_argument("--task", default="clustering", help=_TASK_HELP + " Default: clustering.")
    pr.add_argument("--input", action="append", metavar="ROLE=PATH",
                    help="one input as role=path, repeatable, e.g. --input rna=rna.h5 "
                         "--input adt=adt.h5 --input cty=cty.csv (roles: see "
                         "`multibench layout`); non-canonical files are converted")
    pr.add_argument("--out-dir", "--out", dest="out", required=True,
                    help="output directory for this run (embedding.h5, metric.csv, "
                         "log); --out is an alias")
    pr.add_argument("--runner", metavar="TEMPLATE",
                    help="template for launching the command: {cmd} = the bare command "
                         "(no env activation), {env_cmd} = the command inside the "
                         "method env, e.g. \"srun --gres=gpu:1 {env_cmd}\" "
                         "(mtb.run(cmd_template=))")
    pr.add_argument("--param", "-p", action="append", metavar="KEY=VALUE",
                    help="one hyperparameter override, repeatable: --param epochs=5 "
                         "--param lr=0.001 (METHOD:KEY=VALUE is accepted when METHOD "
                         "is --method). VALUE is parsed as a scalar: 5 -> int, 0.1 -> "
                         "float, true/false -> bool, else string. Keys a method does "
                         "not accept are rejected naming the accepted ones; `multibench "
                         "params METHOD` lists them (mtb.params_for(METHOD, category, "
                         "modalities))")
    pr.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="print the exact command line the run would execute, "
                         "environment activation included, and execute nothing")
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
    pra.add_argument("--out-dir", "--out", dest="out",
                    help="directory for all outputs (one sub-folder per method); "
                         "required unless --dry-run, whose commands then show "
                         "<out_dir>; --out is an alias")
    pra.add_argument("--methods", help=_METHODS_HELP + "; only those (default: every "
                                                       "runnable method)")
    pra.add_argument("--modalities", help="comma-separated modalities to restrict the "
                                          "method variants to, e.g. rna,adt; atac_peak "
                                          "and atac_gas select by what the method reads")
    pra.add_argument("--data-path", dest="data_path",
                     help="folder that contains the dataset folder (default: the "
                          "package data path)")
    pra.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="print the plan - the mtb.scan frame run_all(dry_run=True) "
                          "returns (one row per method variant: runnable, files_ok, "
                          "env_ok, reason; --columns all for every column) and, per "
                          "variant whose inputs resolve, the exact command line "
                          "multibench run would execute (the 'command' column in "
                          "csv/tsv/json); "
                          "execute and create nothing")
    pra.add_argument("--assume-gpu", dest="assume_gpu", action="store_true",
                     help="with --dry-run: skip this host's GPU test, as scan "
                          "--assume-gpu does")
    pra.add_argument("--param", "-p", action="append", metavar="METHOD:KEY=VALUE",
                     help="one hyperparameter override, repeatable: --param "
                          "Matilda:epochs=5 --param Matilda:lr=0.001 -> params="
                          "{'Matilda': {'epochs': 5, 'lr': 0.001}}. VALUE is parsed "
                          "as a scalar (5 -> int, 0.1 -> float, true/false -> bool, "
                          "else string). Unknown METHOD -> did-you-mean error; a key "
                          "the method does not accept is rejected naming the accepted "
                          "ones (checked in --dry-run too); `multibench params METHOD` "
                          "lists them (mtb.params_for). Not allowed with --skip-existing")
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
    pra.add_argument("--batch", metavar="CSV",
                     help="CSV as for evaluate --batch, cells in the order of the label "
                          "files (default: one batch per label file)")
    pra.add_argument("--leiden-flavor", dest="leiden_flavor", choices=["igraph", "leidenalg"],
                     help="Leiden backend of the clustering sweep when scoring (default: "
                          "see `multibench config`); leidenalg matches both stored "
                          "tables (published and re-run)")
    pra.set_defaults(func=_cmd_run_all, _parser=pra)

    # ---- evaluate
    pe = sub.add_parser(
        "evaluate", help="scIB metrics for an embedding against labels (mtb.evaluate)",
        description="Compute the benchmark's metrics for one embedding file. Prints "
                    "the metric table (or writes it with --out). With --method, "
                    "--dataset and --category the table is written in the long "
                    "format (metric,value,method,dataset,category,clustering,source,"
                    "scored_with) that `multibench plot --input` reads.")
    pe.add_argument("--output", required=True,
                    help="the embedding, cells x dims: .h5 (dataset 'data', the "
                         "benchmark's embedding.h5), .h5ad (uses --obsm), .npy, "
                         ".csv/.tsv (a leading barcode column is dropped)")
    pe.add_argument("--category", help=_CATEGORY_HELP + " Required with --method or "
                                                        "--dataset.")
    pe.add_argument("--task", choices=["clustering", "batch", "all"],
                    help="deprecated: use --metrics with the same word")
    pe.add_argument("--labels", action="append", metavar="CSV",
                    help="cell-type labels CSV, one row per cell in the embedding's "
                         "order (header row; column 'x', the only column, or see "
                         "--column). Repeat it for several files: they are stacked in "
                         "the order given and each file counts as one batch. Without "
                         "it, --dataset/--method/--category read the dataset's label "
                         "files (mtb.labels_for)")
    pe.add_argument("--data-path", dest="data_path",
                    help="with --dataset/--method/--category and no --labels: the "
                         "folder that holds the dataset folder (default: the "
                         "configured data_path, see `multibench config`)")
    pe.add_argument("--batch", help="per-cell batch labels CSV; without --metrics the "
                                    "batch metrics are then computed too")
    pe.add_argument("--clustering", "--cluster", dest="cluster", metavar="PATH",
                    help="precomputed clusters (CSV, or an .h5 read from "
                         "/obs/cluster_leiden); they replace the sweep for ARI and NMI "
                         "(iF1 still sweeps unless --metrics leaves it out). "
                         "--cluster is an alias")
    pe.add_argument("--metrics", help="what to compute (mtb.evaluate(metrics=)): a family "
                                      "(clustering | batch | all) or a comma-separated "
                                      "list of metric codes (e.g. ARI,NMI); a list "
                                      "skips everything else, including the Leiden "
                                      "sweep when no listed metric needs it. Default: "
                                      "clustering, plus batch when --batch or several "
                                      "--labels are given")
    pe.add_argument("--leiden-flavor", dest="leiden_flavor", choices=["igraph", "leidenalg"],
                    help="Leiden backend of the clustering sweep (default: see "
                         "`multibench config`); leidenalg matches both stored tables "
                         "(published and re-run)")
    pe.add_argument("--only", help=argparse.SUPPRESS)     # deprecated spelling of --metrics
    pe.add_argument("--obsm", help="for .h5ad input: the .obsm key holding the "
                                   "embedding (default X_emb; 'X' = .X)")
    pe.add_argument("--column", metavar="NAME",
                    help="the column to read in each --labels CSV when a file has "
                         "several columns")
    pe.add_argument("--method", help="method name for the rows of the long format "
                                     "(needs --dataset and --category); without "
                                     "--labels it must be a registry method, which "
                                     "sets the label order")
    pe.add_argument("--dataset", help="label the rows with this dataset id (needs "
                                      "--method and --category); without --labels its "
                                      "label files are read")
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
                       description="One line per method: installed or not, the "
                                   "environment name and a difficulty tag.",
                       epilog=_status_epilog(), formatter_class=_HelpFormatter)
    es.add_argument("--category", help=_CATEGORY_HELP + " Only that category's methods.")
    es.add_argument("--methods", help=_METHODS_HELP + "; only those")
    es.set_defaults(func=_cmd_env, _parser=es)
    egr = ev.add_parser("groups", help="list the shared env groups and their members",
                        description="Shared conda envs (one env serving several methods) "
                                    "and the methods in each.")
    egr.set_defaults(func=_cmd_env, _parser=egr)
    _NAME_HELP = ("environment name (default: the env the package uses for METHOD - "
                  "the name `multibench scan` shows in its env column and "
                  "run/env doctor/env create use; e.g. "
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
                                   "whose name: is the env scan/run expect (the env "
                                   "column of `multibench scan`).")
    ey.add_argument("method", help="method id")
    ey.add_argument("--name", help=_NAME_HELP)
    ey.add_argument("--out", help="write the yml here instead of stdout")
    ey.set_defaults(func=_cmd_env, _parser=ey)
    ec = ev.add_parser("create", help="create one method's env (dry run unless --run)",
                       description="Create the conda environment for METHOD from its "
                                   "committed lockfile (falling back to the recipe when "
                                   "none is captured), under the name scan/run expect; "
                                   "without --run only the commands are printed.")
    ec.add_argument("method", help="method id")
    ec.add_argument("--name", help=_NAME_HELP)
    ec.add_argument("--run", action="store_true",
                    help="actually create the env; without it the command is a dry run")
    ec.add_argument("--force", action="store_true", help=_FORCE_HELP)
    ec.set_defaults(func=_cmd_env, _parser=ec)
    ep = ev.add_parser("plan", help="which envs a set of methods needs (collapsed per env)",
                       description="Collapse methods into the conda envs they need, "
                                   "marking shared vs own envs, with the packed-archive "
                                   "download size ('dl') and unpacked size on disk "
                                   "('disk') recorded for this release ('?' = not "
                                   "measured); a '# total' line on stderr sums them.")
    ep.add_argument("--category", help=_CATEGORY_HELP)
    ep.add_argument("--methods", help=_METHODS_HELP + "; only their envs")
    ep.add_argument("--flavor", choices=_FLAVORS, default="auto", help=_FLAVOR_HELP)
    ep.set_defaults(func=_cmd_env, _parser=ep)
    eg = ev.add_parser("create-group", help="create one shared group env (dry run unless --run)",
                       description="Create the shared conda env GROUP; without --run only "
                                   "the commands are printed.")
    eg.add_argument("group", help="group name (see `multibench env groups`)")
    eg.add_argument("--run", action="store_true",
                    help="actually create the env; without it the command is a dry run")
    eg.add_argument("--force", action="store_true", help=_FORCE_HELP)
    eg.set_defaults(func=_cmd_env, _parser=eg)
    edoc = ev.add_parser(
        "doctor", help="preflight: which envs are present / need building",
        description="One line per env needed by the selected methods: [x] installed, "
                    "[L] missing but a lockfile is ready, [!] missing and no "
                    "lockfile; then the install command for the missing ones. Method "
                    "envs are Linux-only: on another host a warning says so first.")
    edoc.add_argument("--category", help=_CATEGORY_HELP)
    edoc.add_argument("--methods", help=_METHODS_HELP + "; only their envs")
    edoc.add_argument("--strict", action="store_true",
                      help="exit 1 when any env is missing (for scripts: "
                           "`env doctor --strict || env install ...`)")
    edoc.set_defaults(func=_cmd_env, _parser=edoc)
    ei = ev.add_parser(
        "install", help="build every needed env from its lockfile or packed archive "
                        "(mtb.env.install)",
        description="Install the envs the selected methods need. Dry run by default: "
                    "prints per env 'have' / 'build(dry-run)' / 'NO-LOCK', or with "
                    "--packed 'packed archive published' plus the archive's download "
                    "size, unpacked size and URL recorded for this release ('?' = not "
                    "measured) / 'no archive - lockfile build', and a '# total' line "
                    "on stderr. Add --run to do it. "
                    "--flavor picks the CPU-only or the CUDA archive (auto = by "
                    "whether this host has an NVIDIA GPU); the env name is the same "
                    "either way. Method envs are linux-64 conda envs: on "
                    "macOS/Windows a warning is printed first and --run refuses "
                    "(--force overrides).")
    ei.add_argument("--category", help=_CATEGORY_HELP)
    ei.add_argument("--methods", help=_METHODS_HELP + "; only their envs")
    ei.add_argument("--packed", action="store_true",
                    help="use prebuilt archives when published (the URLs recorded "
                         "for this release); fall back to the lockfile build")
    ei.add_argument("--flavor", choices=_FLAVORS, default="auto", help=_FLAVOR_HELP)
    ei.add_argument("--run", action="store_true",
                    help="actually create the envs; without it the command is a dry run")
    ei.add_argument("--force", action="store_true", help=_FORCE_HELP)
    ei.set_defaults(func=_cmd_env, _parser=ei)
    ef = ev.add_parser("freeze", help="capture an env (or --all) to a committed lockfile",
                       description="Write the lockfile of an installed env (maintainers).")
    ef.add_argument("env", nargs="?", help="env name to freeze")
    ef.add_argument("--all", action="store_true", help="freeze every required env")
    ef.add_argument("--category", help=_CATEGORY_HELP + " With --all: only that "
                                                        "category's envs.")
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
    While the command runs, ``config._CLI`` is ``True``: messages built with
    ``config.hint`` then name ``multibench ...`` commands instead of Python
    calls.
    """
    args = build_parser().parse_args(argv)
    from . import config
    prev_show, prev_cli = warnings.showwarning, config._CLI
    # messages built with config.hint() name `multibench ...` commands
    config._CLI = True

    def _show(message, category, filename, lineno, file=None, line=None):
        # library UserWarnings reach the terminal as 'warning: <text>' on
        # stderr, like the CLI's own notes - not as '<path>:<line>: UserWarning'
        print(f"warning: {message}", file=sys.stderr)

    if not os.environ.get("MULTIBENCH_DEBUG"):
        warnings.showwarning = _show
    try:
        with warnings.catch_warnings():
            # a deprecated argument the command passes on (a renamed --input
            # key) warns at this module's call: show it, as a script would
            warnings.filterwarnings("default", category=DeprecationWarning,
                                    module=r"multibench\.cli")
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
        config._CLI = prev_cli


if __name__ == "__main__":
    raise SystemExit(main())
