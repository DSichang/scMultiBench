"""Command-line interface over the multibench Python API.

Every subcommand is a thin wrapper around one public Python function, with the
same parameter names where a flag exists (``--category``, ``--metrics``,
``--out-dir`` ...). The end-to-end story mirrors the Python one::

    multibench config                                  # where data, envs and scripts live
    multibench fetch D11                               # a demo dataset
    multibench layout vertical                         # how to lay out my data
    multibench export my.h5ad data/MYCITE --adt obsm:protein --labels obs:celltype
    multibench scan data/MYCITE --category vertical
    multibench info Matilda                            # env, GPU, labels, ATAC input
    multibench params Matilda                          # what --param accepts
    multibench run-all data/MYCITE --category vertical --methods Matilda --out-dir runs
    multibench evaluate runs/Matilda_MYCITE/embedding.h5 --dataset data/MYCITE \\
        --category vertical --method Matilda --out runs/Matilda_long.csv
    multibench plot bubble --category vertical --input runs --out fig.pdf
    multibench cite Matilda MOFA2

Exit codes and streams
----------------------
``0`` means success. ``1`` is a runtime error raised by the API. The message
is printed as ``error: ...`` on stderr, and ``MULTIBENCH_DEBUG=1`` adds the
traceback. ``2`` is a usage error: an unknown flag, a missing required flag,
a bad choice, or a flag combination the subcommand rejects. ``3`` means
``run-all`` finished and saved its files, but a method failed or a method
named in ``--methods`` was skipped. Without ``--methods``, a skipped method
is only logged and marked ``skipped`` in ``summary.csv``. A line on stderr
names the methods.

Data goes to stdout (tables, ids, commands, yml, citations, ``wrote ...``
lines); diagnostics go to stderr (``error: ...``, ``warning: ...``, progress
such as ``[run_all] ...`` and ``# Dry run ...`` notes), so
``multibench run-all ... --dry-run --format tsv > plan.tsv`` captures a clean table and
``2>/dev/null`` silences the rest. ``run-all --dry-run --format csv`` prints the
whole plan for scripts.

``multibench <command> --help`` documents every flag of every command.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import textwrap
import warnings
from io import StringIO
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import multibench
from . import __version__, _compat, config, discover, workflow
from . import plot as plot_ns
from .config import _and_list
from .data.results import _unpublished_msg
from .engine import envs, ingest, registry, resolve, schema
from .engine.runner import _prepared_at, run as _run
from .eval import io as eio
from .eval import pipeline
from .plot.bubble import THIS_RUN
from .workflow import (_column_name_error, _dry_run_notes, _is_wrong_atac, _is_wrong_ref,
                       _rows_word, _scan_note, _scripts_ref_note)

_EXIT_OK = 0
_EXIT_ERROR = 1
#: ``run-all`` finished and saved, but ``failures`` is not empty
_EXIT_METHOD_FAILED = 3


# ----------------------------------------------------------------- helpers
def _csv_list(s) -> list[str] | None:
    """``"ARI, NMI"`` -> ``['ARI', 'NMI']``, empty items dropped; ``None`` stays ``None``.

    A list (an ``action="append"`` flag) has its items stripped, not split."""
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s if str(x).strip()]
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _modalities_arg(value):
    """``--modalities`` as ``mtb.scan`` and ``mtb.run_all`` take it: the list
    of tokens, or ``"all"`` (every combination) for the one token ``all``."""
    tokens = _csv_list(value)
    return "all" if tokens == ["all"] else tokens


_TRI = {None: None, "true": True, "false": False}


def _tri_state(value) -> bool | None:
    """An optional ``true|false`` flag value as ``True`` / ``False``; absent -> ``None``."""
    return _TRI[value]


#: Free-text columns are clipped to this many characters in the default table view.
_TRUNCATE_WIDTH = 80
_TRUNCATE_COLUMNS = ("reason", "files_reason", "env_reason", "caveat", "command", "error")
#: ``caveat`` is clipped shorter: it points to the full text (``--columns all``).
_CAVEAT_WIDTH = 40


def _truncate(text, width: int = _TRUNCATE_WIDTH) -> str:
    """Clip ``text`` to at most ``width`` characters, ending in ``...``; the cut
    falls on a space in the second half of the kept text, if there is one."""
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


def _resolve_columns(df, columns, fmt: str, compact=None) -> list | None:
    """The columns to print, ``None`` for all: an explicit list, ``all``, or by default
    the ``compact`` columns the frame has in table mode and all otherwise."""
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
    """Print a DataFrame as an aligned table (default), CSV, TSV or JSON (a list of rows).

    ``truncate`` clips :data:`_TRUNCATE_COLUMNS`; by default in table mode without ``columns``."""
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
            print(f"The table has no rows. Its columns are {', '.join(map(str, df.columns))}.",
                  file=file)
        else:
            print(df.to_string(index=False), file=file)


def _quiet_stdout():
    """Send a library call's progress lines to stderr, so stdout stays data only."""
    return contextlib.redirect_stdout(sys.stderr)


def _parse_scalar(text: str):
    """``--param`` VALUE as an int, float, bool (true/yes, false/no), None
    (none/null), a JSON list or object, or else the string."""
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
    """Repeated ``--param [METHOD:]KEY=VALUE`` as ``{METHOD: {KEY: value}}``.

    ``METHOD:`` may be left out only for ``run`` (``default_method`` = ``--method``)."""
    out: dict = {}
    for p in pairs or []:
        if "=" not in p:
            _usage_error(args, f"--param must be [METHOD:]KEY=VALUE, got {p!r} (e.g. "
                               f"--param Matilda:epochs=5)")
        key, value = p.split("=", 1)
        if ":" in key:
            method, key = key.split(":", 1)
        else:
            method = default_method
        method, key = (method or "").strip(), key.strip()
        if not method or not key:
            _usage_error(args, f"--param needs METHOD:KEY=VALUE with non-empty parts, got {p!r}"
                         + ("" if default_method else " (METHOD: is required for run-all)"))
        if default_method is not None and registry.check_method(method) \
                != registry.check_method(default_method):
            _usage_error(args, f"--param names method {method!r} but --method is "
                               f"{default_method!r}; drop the METHOD: prefix or make them agree")
        method = registry.check_method(method)
        out.setdefault(method, {})[key] = _parse_scalar(value)
    return out


def _usage_error(args, message: str) -> None:
    """Exit 2 with the command's usage line and ``message``, as argparse does.

    Without ``args`` (a helper called directly) it raises ``SystemExit(message)``."""
    if args is not None:
        args._parser.error(message)
    raise SystemExit(message)


def _platform_note() -> str | None:
    """Print ``warning: ...`` on stderr when this host cannot build the method envs
    (they are linux-64); return the problem, or ``None``."""
    problem = envs.host_platform_problem()
    if problem:
        print(f"warning: {envs.linux_only_text(problem)} multibench env install "
              f"refuses here. The method list, stored results, scan's file check, "
              f"evaluate and plot work on this computer.", file=sys.stderr)
    return problem


# ----------------------------------------------------------------- deprecations
class _Retired(argparse.Action):
    """A deprecated flag: it parses, ``--help`` leaves it out, and :func:`main`
    warns with ``tail``. Its value goes to ``dest``: the new flag's for a renamed
    one."""

    def __init__(self, option_strings, dest, *, tail=_compat.NO_EFFECT, **kwargs):
        super().__init__(option_strings, dest, help=argparse.SUPPRESS, **kwargs)
        self.tail = tail

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True if self.nargs == 0 else values)
        tail = self.tail.replace("{value}", str(values))       # "Use --metrics {value}."
        namespace._retired = [*getattr(namespace, "_retired", ()), (option_string, tail)]


def _retired_command(name: str, new: str) -> dict:
    """``set_defaults`` of a deprecated command: :func:`main` warns that ``new``
    replaces ``name``."""
    return {"_retired_command": (f"multibench {name}", _compat.use(f"multibench {new}"))}


def _warn_retired(args) -> None:
    """One ``FutureWarning`` per deprecated command or flag on the command line;
    a flag is named with its command (``scan --format``)."""
    if getattr(args, "_retired_command", None):
        _compat.deprecated(*args._retired_command)
    command = " ".join(c for c in (getattr(args, "command", None),
                                   getattr(args, "env_cmd", None)) if c)
    for flag, tail in getattr(args, "_retired", ()):
        _compat.deprecated(f"{command} {flag}".strip(), tail)


# ----------------------------------------------------------------- commands
def _cmd_list(args) -> int:
    """``multibench list`` (deprecated): ``multibench find --category``."""
    for m in discover._find_methods(category=args.category, task=args.task,
                                    runnable=args.runnable or None):
        print(m)
    return _EXIT_OK


def _cmd_find(args) -> int:
    """``multibench find``: :func:`multibench.find_methods` with the same filters."""
    ids = discover._find_methods(category=args.category, task=args.task,
                                 needs_labels=_tri_state(args.needs_labels),
                                 atac=args.atac, modalities=_csv_list(args.modalities),
                                 runnable=args.runnable or None,
                                 tunable=_tri_state(args.tunable))
    for m in ids:
        print(m)
    return _EXIT_OK


def _cmd_scan(args) -> int:
    """``multibench scan``: :func:`multibench.scan`'s table on stdout, its lines on
    stderr. The deprecated ``--columns``, ``--format`` and ``--strict`` print the
    plan of ``run-all --dry-run`` instead."""
    methods = _csv_list(args.methods)
    for m in methods or []:
        registry.check_method(m)               # did-you-mean KeyError before any I/O
    category = args.category or workflow.known_category(args.dataset)
    if args.columns is None and args.format == "table" and not args.strict:
        with contextlib.redirect_stdout(sys.stderr):
            table = workflow._scan_methods(
                args.dataset, category, methods=methods, data_path=args.data_path,
                modalities=_modalities_arg(args.modalities), verbose=True,
                assume_gpu=args.assume_gpu, allow_atac_mismatch=args.allow_atac_mismatch)
        _print_frame(table)
        return _EXIT_OK
    df, _ = workflow._scan(args.dataset, category, methods=methods,
                           data_path=args.data_path,
                           modalities=_modalities_arg(args.modalities), verbose=False,
                           assume_gpu=args.assume_gpu,
                           allow_atac_mismatch=args.allow_atac_mismatch)
    shown = _strict_table(df) if args.strict else df
    _print_frame(shown, columns=_csv_list(args.columns), fmt=args.format,
                 compact=workflow.PLAN_COLUMNS[:4])
    note = _scan_note(df, args.dataset, args.data_path, category)
    code = _EXIT_OK
    if args.strict:
        problem = _strict_problem(df, methods)
        if problem:
            print(f"error: --strict: {problem}", file=sys.stderr)
            code = _EXIT_ERROR
    if note:
        print(f"# {note}", file=sys.stderr)
    fit = workflow._category_hint(df, category)
    if fit:
        print(f"# {fit}", file=sys.stderr)
    return code


#: The reason ``scan --strict`` prints for a row blocked only by the method scripts.
_SCRIPTS_REASON = "The method scripts are not fetched. Run multibench fetch --scripts."


def _strict_table(df):
    """The ``scan --strict`` table: while the scripts are not fetched, a runnable row
    shows ``runnable`` False and :data:`_SCRIPTS_REASON`, as the error under it says."""
    if config.scripts_present() or "runnable" not in df:
        return df
    df = df.copy()
    blocked = df["runnable"].astype(bool)
    df.loc[blocked, "runnable"] = False
    if "reason" in df:
        df.loc[blocked, "reason"] = _SCRIPTS_REASON
    return df


def _strict_problem(df, methods) -> str | None:
    """Why ``scan --strict`` fails, or ``None``: no runnable row, or a ``--methods``
    method without one. Unfetched scripts block every row (a node may lack network)."""
    fetched = config.scripts_present()
    ok = df["runnable"].astype(bool) & fetched
    runnable = df[ok]
    if methods:
        blocked = [m for m in methods if m not in set(runnable["method"])]
    else:
        blocked = [] if len(runnable) else list(dict.fromkeys(df["method"]))
    if not blocked and (methods or len(runnable)):
        return None
    rows_word = "row is" if len(df) == 1 else "rows are"
    head = f"{len(runnable)} of {len(df)} {rows_word} runnable."
    rest = df[~ok]
    counts = []
    if "files_ok" in rest and (~rest["files_ok"].astype(bool)).any():
        counts.append(f"Rows with missing input files: "
                      f"{int((~rest['files_ok'].astype(bool)).sum())}")
    if "reason" in rest:
        wrong = _is_wrong_atac(rest["reason"])
        names = wrong & rest["reason"].astype(str).str.contains(
            workflow.PEAK_NAMES_REASON, regex=False)
        if (wrong & ~names).any():
            counts.append(f"Rows with the wrong ATAC kind: {int((wrong & ~names).sum())}")
        if names.any():
            counts.append(f"Rows whose peak names the method cannot read: "
                          f"{int(names.sum())}")
        if _is_wrong_ref(rest["reason"]).any():
            counts.append(f"Rows whose scripts are not at MULTIBENCH_SCRIPTS_REF: "
                          f"{int(_is_wrong_ref(rest['reason']).sum())}")
    if not fetched:
        counts.append(f"Rows whose method scripts are not fetched: {len(df)}")
    n_env, n_gpu, gpu_only = _env_and_gpu_counts(rest)
    if n_env:
        counts.append(f"Rows whose environment is not ready: {n_env}")
    if n_gpu:
        counts.append(f"Rows that need a GPU this host lacks: {n_gpu}")
    gpu_note = ("\nFor a job that runs on a GPU node, add --assume-gpu."
                if gpu_only else "")
    if not fetched:                     # the fix for the scripts, before the GPU note
        fix = config.scripts_folder_problem() or "Run multibench fetch --scripts first"
        gpu_note = f"\n{fix}.{gpu_note}"
    head += "".join(f" {c}." for c in counts)
    if not methods:
        # rows blocked only by the scripts have an empty reason
        why = ("" if not fetched and df["runnable"].astype(bool).any()
               else " The reason column says why.")
        # the scripts-ref reason is the same on every row, and the table clips it
        ref = (config.scripts_ref_problem()
               if "reason" in rest and _is_wrong_ref(rest["reason"]).any() else None)
        return head + why + (f"\n{ref}" if ref else "") + gpu_note
    lines = []
    for m in blocked:
        rows = df[df["method"] == m]
        if len(rows) and "files_ok" in rows:
            # the row whose files are on disk says what blocks the method
            rows = rows.sort_values("files_ok", ascending=False, kind="stable")
        reason = rows["reason"].iloc[0] if len(rows) else ""
        if len(rows) and _blank(reason):            # runnable but for the scripts
            reason = "The method scripts are not fetched."   # the fix follows once
        # an ATAC or scripts-ref reason ends with its fix: never clipped
        text = ("no row" if _blank(reason) else str(reason)
                if _is_wrong_atac(reason) or _is_wrong_ref(reason)
                else _truncate(reason, 120))
        lines.append(f"  {m}: {text}")
    return (head + f" No runnable row for {', '.join(blocked)}:\n"
            + "\n".join(lines) + gpu_note)


def _env_and_gpu_counts(rest) -> tuple[int, int, int]:
    """Rows of ``rest`` blocked by a missing env, by this host's GPU test, and by
    the GPU test alone (the ones ``--assume-gpu`` makes runnable)."""
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
    print(multibench.describe_layout(args.category))
    return _EXIT_OK


_EXPORT_FLAGS = ("rna", "adt", "atac", "atac_kind", "labels", "batch", "batch_index",
                 "atac_from", "n_batches")

#: The recipes of ``multibench export --help``.
_CONVERT_EPILOG = """\
Genes and peaks in one X (10x Multiome read with gex_only=False):
  multibench export B.h5ad data/multiome --rna "X[feature_types=Gene Expression]" --atac "X[feature_types=Peaks]" --atac-kind peak --labels obs:cell_type --category vertical

RNA and ATAC from different cells (diagonal):
  multibench export rna.h5ad data/lung --atac-from atac.h5ad --atac-kind peak --labels obs:cell_type --category diagonal

One file per batch (mosaic or cross): write each file as its batch number.
  multibench export A.h5ad data/LAB --adt obsm:protein --labels obs:cell_type --category mosaic --batch-index 1
  multibench export B.h5mu data/LAB --rna mod:rna --atac mod:atac --atac-kind peak --labels rna:cell_type --category mosaic --batch-index 2
  multibench export C.h5ad data/LAB --labels obs:cell_type --category mosaic --batch-index 3
  multibench scan data/LAB --category mosaic
"""


def _rewrite_canonical(src, out, dtype: str) -> bool:
    """Copy the canonical ``.h5`` ``src`` to ``out`` with ``matrix/data`` as ``dtype``:
    a byte copy (returns ``False``) when it already is, else a re-encoding (``True``)."""
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
        step = 1024
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


def _refuse_canonical(args, path, flag, role, category) -> None:
    """Dataset mode reads an .h5ad or .h5mu: raise for a canonical ``.h5`` at
    ``path``. With a ``role`` (the caller found the copy is the whole export)
    and an empty ``OUT``, the message gives the command that copies the file
    there under the name the export would write; else it asks for the source."""
    p = Path(path)
    if not (p.suffix.lower() == ".h5" and p.is_file() and ingest._is_canonical_h5(p)):
        return
    out = Path(args.out)
    action = "Pass the .h5ad or .h5mu it came from."
    # a folder with files: the export would order the rows by them, a copy does not
    if role is not None and not (out.is_dir() and any(out.iterdir())):
        name = ingest._export_filename(role, str(args.batch_index or ""),
                                       ingest._check_category(category))
        copy = shlex.join(["cp", str(path), str(out / name)])
        if not out.is_dir():
            copy = shlex.join(["mkdir", "-p", str(out)]) + " && " + copy
        action = "Copy it into the folder with: " + copy
    raise ValueError(f"{path} is already a canonical .h5 file, and {flag} reads an .h5ad "
                     f"or .h5mu. {action}")


def _reads_x(spec) -> bool:
    """Whether a matrix flag value selects ``X`` (``X`` or ``X[...]``)."""
    return spec is not None and (spec == "X" or str(spec).startswith("X["))


def _cmd_convert(args) -> int:
    """``multibench export``: a dataset folder (``export_dataset``), the RNA from X
    unless another flag reads X. Its deprecated name ``convert`` writes one
    canonical ``.h5`` (``to_canonical``) without any :data:`_EXPORT_FLAGS` flag."""
    mode1 = any(getattr(args, f) is not None for f in ("modality", "layer", "obsm", "mod"))
    if isinstance(args.rna, str) and args.rna.lower() == "none":   # export_dataset(rna=None)
        args.rna = None
        args._no_rna = True
    if getattr(args, "_export", False) and args.rna is None and not mode1 \
            and not getattr(args, "_no_rna", False) \
            and not any(map(_reads_x, (args.adt, args.atac))) \
            and Path(args.src).suffix.lower() != ".h5mu":
        args.rna = "X"                      # as mtb.io.export_dataset(rna="X")
    export_mode = any(getattr(args, f) is not None for f in _EXPORT_FLAGS)
    category = args.category
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
        batch_index = args.batch_index
        extra = {}
        if batch_index is not None:
            if args.batch is not None:
                _usage_error(args, "--batch and --batch-index are mutually exclusive: "
                             "--batch splits one file by a column, --batch-index "
                             "writes the whole file as one batch")
            if category not in ("mosaic", "cross", "diagonal"):
                _usage_error(args, "--batch-index needs --category mosaic or --category "
                             "cross (the categories that read numbered batch files), or "
                             "--category diagonal for data in several batches")
            if batch_index < 1:
                _usage_error(args, f"--batch-index counts from 1, got {batch_index}")
            extra["batch_index"] = batch_index
        if args.n_batches is not None:
            extra["n_batches"] = args.n_batches
        atac = args.atac
        # a canonical .h5 given here: the copy command only when the copy is the
        # whole export (one matrix as it is, no labels or batch split) and its
        # file name is certain (peak without a category writes two files)
        atac_role = {"peak": "atac_peak", "gene_activity": "atac_gas"}.get(args.atac_kind)
        if args.atac_kind == "peak" and category is None:
            atac_role = None
        alone = args.labels is None and args.batch is None
        if args.atac_from is not None:
            whole = alone and args.rna is None and args.adt is None \
                and args.atac in (None, "X")
            _refuse_canonical(args, args.atac_from, "--atac-from",
                              atac_role if whole else None, category)
            atac = ingest._select_object(ingest._to_anndata(args.atac_from),
                                         args.atac or "X", what="atac")
        reads = [f for f in ("rna", "adt", "atac", "labels", "batch")
                 if getattr(args, f) is not None
                 and not (f == "atac" and args.atac_from is not None)]
        matrices = [f for f in reads if f in ("rna", "adt", "atac")]
        role = None
        if alone and args.atac_from is None and len(matrices) == 1 \
                and getattr(args, matrices[0]) == "X":
            role = {"rna": "rna", "adt": "adt", "atac": atac_role}[matrices[0]]
        _refuse_canonical(args, args.src, f"--{reads[0]}" if reads else "the dataset mode",
                          role, category)
        data = ingest._to_anndata(args.src)
        if args.rna is None and args.adt is None and args.atac is None \
                and args.atac_from is None and not any(Path(args.out).glob("*.h5")):
            # a folder of labels alone; adding them to a folder with matrices is fine
            mods = getattr(data, "mod", None)
            sel = ("X" if not isinstance(mods, dict)
                   else "mod:rna" if "rna" in mods else "mod:<name>")
            _usage_error(args, f"--labels alone writes only the label file. Add --rna "
                         f"{sel} to export the matrix too.")
        before = _folder_state(args.out)
        p = ingest._export_dataset(data, args.out, rna=args.rna, adt=args.adt,
                                  atac=atac, atac_kind=args.atac_kind,
                                  labels=args.labels, batch=args.batch,
                                  dtype=args.dtype, category=category,
                                  overwrite=args.overwrite, **extra)
        after = _folder_state(p)
        wrote = sorted(n for n in after if before.get(n) != after[n])
        kept = sorted(n for n in after if n not in wrote)
        print(f"wrote dataset folder {p} (files: {', '.join(wrote)})")
        # a batch of a mosaic or cross folder may hold no RNA by design
        one_batch = category in ("mosaic", "cross") and args.batch_index is not None
        has_rna = any(re.fullmatch(r"rna\d*\.h5", n) for n in after)
        if args.rna is None and not one_batch and not has_rna \
                and not getattr(args, "_no_rna", False) and (
                args.adt is not None or args.atac is not None or args.atac_from is not None):
            print("# No RNA written: pass --rna X (or a layer) to write it.",
                  file=sys.stderr)
        if kept:
            print(f"# already in the folder: {', '.join(kept)} (not written by this call)",
                  file=sys.stderr)
        return _EXIT_OK
    if args.layer is not None and args.obsm is not None:
        _usage_error(args, "--layer and --obsm are mutually exclusive")
    src = Path(args.src)
    if src.is_file() and ingest._is_canonical_h5(src):
        # to_canonical() returns a canonical SRC as it is; the CLI must put a file at OUT
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
    p = ingest._to_canonical(args.src, args.out, modality=args.modality,
                             layer=args.layer, obsm=args.obsm, mod=args.mod,
                             dtype=args.dtype, category=category)
    print(f"wrote {p}")
    return _EXIT_OK


def _cmd_fetch(args) -> int:
    """``multibench fetch``: :func:`multibench.data.fetch` per dataset named (the
    deprecated ``--outputs``: ``fetch_outputs``), and with ``--scripts`` the method
    scripts (``config.ensure_repo``)."""
    from .data.fetch import AVAILABLE, _fetch, fetch_outputs
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
        with _quiet_stdout():
            repo = config.ensure_repo(ref=args.ref)
        state = "present" if present else "fetched"
        print(config.scripts_line(repo).replace("method scripts:",
                                                f"method scripts {state}:", 1))
    for ds in ids:
        with _quiet_stdout():
            if args.outputs:
                where = fetch_outputs(ds, data_path=args.data_path)
            else:
                where = _fetch(ds, data_path=args.data_path)
        print(f"{ds}: {where}")
    return _EXIT_OK


def _cmd_config(args) -> int:
    """``multibench config``: each setting of ``mtb.config.DEFAULT`` and where it came
    from; ``--get NAME`` prints the bare value."""
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
    """``multibench info METHOD``: the :func:`multibench.method_info` facts to check
    before a run; ``--format json`` prints the whole dict."""
    info = discover._method_info(args.method, verbose=True)   # KeyError (did-you-mean)
    if args.format == "json":
        print(json.dumps(info, indent=1, default=str))
        return _EXIT_OK
    print(f"{info['id']} ({info['language'] or '?'}), env {info['env']}")
    print(f"  GPU:          {info['gpu'] or 'unknown'}")
    print(f"  needs_labels: {_yes_no(info['needs_labels'])}")
    print(f"  atac:         {info['atac'] or 'none (reads no ATAC)'}")
    print("  variants:")
    for v in info["supports"]:
        mods = "+".join(v["modalities"]) or "(data_dir)"
        labels = f"  labels: {', '.join(v['labels'])}" if v["labels"] else ""
        print(f"    {v['category']:9} {mods}{labels}")
    rt = info["runtime"]
    observed = sorted(rt["observed"], key=lambda o: -o["sec"])
    pad = " " * 16
    print(f"  runtime:      {rt['tier']}"
          + ("" if observed else " (never measured)"))
    for o in observed:
        cells = o["cells"]
        size = f"{cells:,} cells" if isinstance(cells, int) else "cells not recorded"
        print(f"{pad}{o['sec']} s on {o['dataset']} ({size})")
    if observed:
        print(textwrap.fill(rt["note"], width=88, initial_indent=pad, subsequent_indent=pad))
    hint = (info["setup_hint"] or "").strip()
    if hint:
        print("  setup_hint:")
        print(textwrap.fill(hint, width=88, initial_indent="    ", subsequent_indent="    "))
    print(f"  scripts:      {info['scripts_url']}")
    print(f"  more:         multibench params {info['id']}; multibench cite {info['id']}")
    return _EXIT_OK


def _params_rows(method: str, category: str | None, modalities) -> list[dict]:
    """The :func:`multibench.params_for` dict of every variant of ``method`` that
    matches ``category`` / ``modalities``; a selector left out matches all."""
    spec = registry.get(method)                       # KeyError (did-you-mean)
    registry.check_category(category)
    want = registry.normalize_modalities(modalities) if modalities else None
    picked = [v for v in spec.variants
              if (category is None or v.when.get("category") == category)
              and (want is None or set(v.when.get("modalities", [])) == set(want))]
    if not picked and want is not None:
        # the tokens params_for accepts (rna,adt for a three-batch cross variant)
        try:
            return [discover._params_for(method, category, list(modalities))]
        except (KeyError, ValueError):
            pass
    if not picked:
        if category is not None and category not in spec.wired_categories:
            raise ValueError(schema.no_category_message(method, spec.wired_categories, category))
        avail = [f"{v.when.get('category')}:{'+'.join(v.when.get('modalities', [])) or '(data_dir)'}"
                 for v in spec.variants]
        raise ValueError(f"{method}: no variant for category={category!r} "
                         f"modalities={list(want) if want else None}; available: {avail}")
    return [discover._params_for(method, v.when.get("category"),
                                 list(v.when.get("modalities", [])) or None)
            for v in picked]


def _cmd_params(args) -> int:
    """``multibench params METHOD``: the :func:`multibench.params_for` tables, one
    block per matching variant (``--format json``: the dicts)."""
    rows = _params_rows(args.method, args.category, _csv_list(args.modalities))
    fmt = args.format
    if fmt == "json":
        print(json.dumps(rows, indent=1, default=str))
        return _EXIT_OK
    frames = []
    for p in rows:
        tun = p["tunable"]
        # object columns: an int default prints as 256, not 256.000
        tbl = pd.DataFrame([{"variant": p["variant"], "key": k,
                             "type": t.get("type"),
                             "default": t.get("default")}
                            for k, t in sorted(tun.items())],
                           columns=["variant", "key", "type", "default"],
                           dtype=object)
        if fmt in ("csv", "tsv"):
            frames.append(tbl)
            continue
        print(f"# {p['method']} {p['variant']}: {len(tun)} tunable parameter(s) "
              f"(set with --param {p['method']}:KEY=VALUE)")
        if len(tun):
            _print_frame(tbl.drop(columns=["variant"]), fmt="table")
        else:
            print("  (none: the upstream script exposes no hyperparameter on its "
                  "command line)")
        fixed = p["fixed_in_script"]
        if fixed:
            print("# fixed in the script (not tunable through the wrapper):")
            for f in fixed:
                print(f"  {f.get('name')} = {f.get('value')}   ({f.get('source')})")
        knobs = p["upstream_knobs"]
        if knobs:
            print("# upstream library knobs the script does not expose: "
                  + ", ".join(str(k.get("name")) for k in knobs)
                  + (f"  (see {p['upstream_url']})" if p["upstream_url"] else ""))
    if frames:
        _print_frame(pd.concat(frames, ignore_index=True), fmt=fmt)
    return _EXIT_OK


def _cmd_cite(args) -> int:
    """``multibench cite``: :func:`multibench.cite` for the benchmark + methods."""
    if args.all and args.methods:
        _usage_error(args, "--all and explicit method ids are mutually exclusive")
    # cite A,B as well: every other multi-id flag of the CLI is comma-separated
    ids = [m for tok in (args.methods or []) for m in _csv_list(tok)]
    methods = "all" if args.all else (ids or None)
    text = discover.cite(methods, format=args.format)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return _EXIT_OK


def _load_long_input(path) -> pd.DataFrame:
    """A long results frame from a CSV with ``metric,value,method`` columns or a
    ``run_all`` output directory."""
    p = Path(path)
    if p.is_dir():
        from .workflow import load_batch
        return load_batch(p).long
    if not p.is_file():
        raise FileNotFoundError(
            f"--input {p}: not a file or a run_all output directory")
    df = pd.read_csv(p)
    need = {"metric", "value", "method"}
    if not need <= set(df.columns):
        raise ValueError(
            f"--input {p}: expected a long results table with columns "
            f"{sorted(need)} (what multibench evaluate --method/--dataset "
            f"and run_all's long.csv write); got {list(df.columns)}")
    return df


#: ``load_results`` keywords in its messages -> ``plot`` flags; longest pattern first.
_API_TO_FLAGS = (
    (r"\b(source|category)=['\"](\w+)['\"]", r"--\1 \2"),
    (r"\bsource=", "--source"),
    (r"\(or '(\w+)'\)", r"(or \1)"),
)


def _cli_spelling(message: str) -> str:
    """A ``load_results`` message with its keywords spelled as ``plot`` flags."""
    for pattern, flag in _API_TO_FLAGS:
        message = re.sub(pattern, flag, message)
    return message


def _unpublished_plot_msg(category: str, inputs) -> str:
    """``plot``'s error for a category without published tables (mosaic)."""
    if inputs:
        return (f"{category} has no published tables. Leave out --source to plot your "
                f"rows with the stored results, or drop --category to plot "
                f"your rows alone.")
    return (f"{category} has no published tables. Leave out --source to plot the "
            f"stored results.")


def _names(values) -> str:
    """``'MYCITE'`` / ``'D1, D2'``: the distinct values, in first-seen order."""
    return ", ".join(dict.fromkeys(str(v) for v in values))


def _filter_own_rows(frames: list, args) -> list:
    """Apply ``--dataset`` (and with ``--category``, ``--methods``) to the ``--input``
    frames of ``plot``: dropping every row is an error, some rows a warning."""
    if not frames:
        return frames
    whole = pd.concat(frames, ignore_index=True, sort=False)
    n = len(whole)
    datasets = _csv_list(args.dataset)
    methods = _csv_list(args.methods) if args.category is not None else None
    keep = pd.Series(True, index=whole.index)
    if datasets and "dataset" in whole.columns:
        keep &= whole["dataset"].astype(str).isin(datasets)
        if n and not keep.any():
            fix = ("Plot them without --category, or score your method on "
                   f"{', '.join(datasets)}." if args.category is not None else
                   f"Drop --dataset, or pass --dataset {_names(whole['dataset'])}.")
            raise ValueError(
                f"Your {n} rows are for dataset {_names(whole['dataset'])}, and "
                f"--dataset {args.dataset} removed all of them. {fix}")
    if methods:
        by_method = whole["method"].astype(str).isin(methods)
        if (keep & ~by_method).any() and not (keep & by_method).any():
            own = list(dict.fromkeys(whole.loc[keep, "method"].astype(str)))
            noun = "method" if len(own) == 1 else "methods"
            raise ValueError(
                f"Your rows are for {noun} {_and_list(own)}, and --methods {args.methods} "
                f"removed all of them. Add {_and_list(own)} to --methods.")
        keep &= by_method
    dropped = whole[~keep]
    if len(dropped):
        what = [f"method {_names(dropped['method'])}"]
        if "dataset" in dropped.columns:
            what.insert(0, f"dataset {_names(dropped['dataset'])}")
        flags = " and ".join(f for f, on in (("--dataset", datasets), ("--methods", methods))
                             if on)
        print(f"warning: {flags} dropped {len(dropped)} of your {n} rows "
              f"({', '.join(what)})", file=sys.stderr)
    out, start = [], 0
    for f in frames:
        stop = start + len(f)
        out.append(f[keep.iloc[start:stop].to_numpy()])
        start = stop
    return out


def _rename_clashes(stored: pd.DataFrame, own: list) -> list:
    """Your --input rows whose method and dataset the stored results also have,
    renamed ``<method> (this run)`` (the command line cannot rename them), with
    one line on stderr."""
    keys = ["method", "dataset"]
    have = set(map(tuple, stored[keys].astype(str).drop_duplicates().to_numpy()))
    out, renamed = [], []
    for f in own:
        if not set(keys) <= set(f.columns):
            out.append(f)
            continue
        hit = [tuple(k) in have for k in f[keys].astype(str).to_numpy()]
        if any(hit):
            f = f.copy()
            renamed += sorted(set(f.loc[hit, "method"].astype(str)))
            f.loc[hit, "method"] = f.loc[hit, "method"].astype(str) + THIS_RUN
        out.append(f)
    if renamed:
        names = list(dict.fromkeys(renamed))
        print(f"# Your rows of {_and_list(names)} are drawn as "
              f"{_and_list([m + THIS_RUN for m in names])}: the stored results have "
              f"the same method and dataset.", file=sys.stderr)
    return out


def _rename_repeats(own: list, inputs: list) -> list:
    """The rows of ``--input`` files that share a method and dataset, each renamed
    ``<method> (<its folder or file name>)``."""
    keys = ["method", "dataset"]
    pairs = [set(map(tuple, f[keys].astype(str).to_numpy())) if set(keys) <= set(f.columns)
             else set() for f in own]
    shared = {k for i, p in enumerate(pairs) for k in p
              if any(k in q for q in pairs[:i] + pairs[i + 1:])}
    out = []
    for f, path in zip(own, inputs):
        if shared and set(keys) <= set(f.columns):
            hit = [tuple(k) in shared for k in f[keys].astype(str).to_numpy()]
            f = f.copy()
            # "<method> (<folder>)": --methods <method> keeps it (registry.method_of)
            tag = Path(path).parent.name if Path(path).name == "long.csv" else Path(path).stem
            f.loc[hit, "method"] = f.loc[hit, "method"].astype(str) + f" ({tag})"
        out.append(f)
    return out


def _cmd_plot(args) -> int:
    """``multibench plot {bubble,bar}``: draw the ``--input`` rows, the stored results
    of ``--category`` (:func:`multibench.load_results`), or both concatenated."""
    inputs = _csv_list(args.input) or []
    if not inputs and args.category is None:
        _usage_error(args, "need --category (stored results) or --input LONG_CSV")
    frames = []
    if args.category is not None:
        from .data.results import _load_results
        try:
            frames.append(_load_results(args.category, dataset=_csv_list(args.dataset),
                                        source=args.source, result_path=args.result_path))
        except FileNotFoundError as e:
            if str(e) == _unpublished_msg(args.category):
                raise FileNotFoundError(_unpublished_plot_msg(args.category, inputs)) from e
            raise FileNotFoundError(_cli_spelling(str(e))) from e
    loaded = [_load_long_input(path) for path in inputs]
    frames += _rename_repeats(_filter_own_rows(loaded, args), inputs)
    if args.category is not None and len(frames) > 1:
        frames[1:] = _rename_clashes(frames[0], frames[1:])
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True, sort=False)
    methods = _csv_list(args.methods)
    if methods:
        # a method id keeps its renamed runs too, as "scJoint (ep40)"
        present = list(dict.fromkeys(df["method"].astype(str)))
        ids = {registry.method_of(m) for m in methods} - {None}
        runs = [h for h in present if h not in methods and registry.method_of(h) in ids]
        renamed = {registry.method_of(h) for h in runs}
        methods = [m for m in methods if m in present
                   or registry.method_of(m) not in renamed] + runs
    if len(df) == 0:
        raise ValueError("Nothing is left to plot after the filters.")
    if df["method"].nunique() == 1 and args.category is None and args.kind == "bar":
        # load_results and plot.bubble warn themselves; --input frames meet neither
        print("warning: The table has one method. Every rank is the same.",
              file=sys.stderr)
    title = args.title
    if title is None:
        title = " ".join(x for x in (args.category, args.dataset) if x) or None
    metrics = _csv_list(args.metrics)
    if metrics is not None and len(metrics) == 1 and metrics[0].lower() in _METRIC_FAMILIES:
        metrics = metrics[0].lower()
    # the deprecated flags (warned once) keep their effect until 0.5
    from .plot.bar import _bar
    from .plot.bubble import _bubble
    with _quiet_stdout():
        if args.kind == "bubble":
            _bubble(df, metrics=metrics, methods=methods, aggregate=args.aggregate,
                    title=title, save=args.out, overall=args.overall or "rank",
                    require_complete=args.require_complete, na=args.na or "warn")
        else:
            _bar(df, metrics=metrics, methods=methods, top=args.top, title=title,
                 save=args.out, overall=args.overall)
    print(f"wrote {args.out}")
    return _EXIT_OK


def _parse_inputs(pairs, args) -> dict:
    """Repeated ``--input role=path`` as ``{role: path}``."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            _usage_error(args, f"--input must be role=path, got {p!r}")
        role, path = p.split("=", 1)
        if role.strip() == "" or path == "":
            _usage_error(args, f"--input must be role=path with non-empty parts, got {p!r}")
        out[role] = path
    return out


def _cmd_run(args) -> int:
    """``multibench run``: :func:`multibench.run` one method on explicit inputs."""
    if not args.method or not args.category:
        args._parser.error("")
    args.method = registry.check_method(args.method)     # any case, or a display name
    params = _parse_params(args.param, args, default_method=args.method)
    inputs = _parse_inputs(args.input, args)
    if args.dry_run:
        # the command is built first: an error prints no dry-run header
        notes = StringIO()
        with contextlib.redirect_stderr(notes):
            argv = _run(args.method, args.category, inputs=inputs,
                                  out_dir=args.out or workflow.OUT_DIR_PLACEHOLDER,
                                  params=params.get(args.method),
                                  cmd_template=args.cmd_template, dry_run=True)
        # the header first, then run's notes, then the command
        print("# Dry run. Nothing was executed.", file=sys.stderr, flush=True)
        print(notes.getvalue(), end="", file=sys.stderr, flush=True)
        print("# multibench run would execute:", file=sys.stderr, flush=True)
        print(shlex.join(argv))
        return _EXIT_OK
    if not args.out:
        _usage_error(args, "the following arguments are required: --out-dir/--out")
    with _quiet_stdout():
        res = _run(method=args.method, category=args.category,
                           inputs=inputs, out_dir=args.out,
                           params=params.get(args.method) or None,
                           cmd_template=args.cmd_template)
    print(f"ran {args.method} -> {res.out_dir}")
    return _EXIT_OK


def _cmd_run_all(args) -> int:
    """``multibench run-all``: :func:`multibench.run_all`; with ``--dry-run`` the plan
    and its commands. Exit 3 when a run saved a non-empty ``failures``."""
    with contextlib.ExitStack() as stack:
        return _run_all_command(args, stack)


def _run_all_command(args, stack) -> int:
    """The body of :func:`_cmd_run_all`; ``stack`` removes the ``--batch-column`` copy."""
    if args.out is None and not args.dry_run:
        # the text argparse prints for a missing required option
        _usage_error(args, "the following arguments are required: --out-dir/--out")
    params = _parse_params(args.param, args) or None
    columns = _csv_list(args.columns)
    if args.assume_gpu and not args.dry_run:
        _usage_error(args, "--assume-gpu applies to --dry-run only; a real run checks "
                           "this host's GPU")
    if eio._names_no_file(args.batch):
        # the name of an obs column: a dataset folder holds none
        raise FileNotFoundError(_column_name_error("batch", args.batch))
    batch = _batch_arg(args, stack)
    if batch is not None:
        eio._require_file(batch, "--batch file")   # exit 1 before any method runs
    if args.dry_run:
        modalities = _modalities_arg(args.modalities)
        with _quiet_stdout():
            df = workflow._run_all(args.dataset, args.category, out_dir=args.out,
                                   methods=_csv_list(args.methods),
                                   modalities=modalities,
                                   data_path=args.data_path, params=params,
                                   dry_run=True, verbose=False, batch=batch,
                                   assume_gpu=args.assume_gpu,
                                   allow_atac_mismatch=args.allow_atac_mismatch)
        k, n = int(df["runnable"].sum()), len(df)
        # as the Python dry run says it
        print(f"# Dry run: {k} of {n} {_rows_word(df, n)} can run on {args.dataset} "
              f"({args.category})" + ("; the reason column says why." if n > k else "."),
              file=sys.stderr)
        note = _scan_note(df, args.dataset, args.data_path, args.category)
        if note:                            # the rows the plan does not list
            print(f"# {note}", file=sys.stderr)
        wrong_ref = _scripts_ref_note()     # every row also carries it as its reason
        if wrong_ref:
            print(f"# {wrong_ref}", file=sys.stderr)
        # the caveats of the rows the sweep would run: the compact table clips them
        scripts, lines = _dry_run_notes(df, None if modalities == "all" else modalities)
        if scripts:
            print(f"# {scripts}", file=sys.stderr)
        for _m, cav in lines:              # each caveat starts with its method
            print(f"# {cav}", file=sys.stderr)
        plan = df if columns else workflow._shown_modalities(df)[workflow.PLAN_COLUMNS]
        _print_frame(plan, columns=columns, fmt=args.format,
                     compact=workflow.PLAN_COLUMNS[:4])
        if args.format == "table" and not columns:
            have = df[df["command"].astype(str).str.len() > 0]
            missing = not have["env_ok"].astype(bool).all()
            prepared = any(_prepared_at(c or "") >= 0 for c in have.get("caveat", []))
            print()
            # each mark is explained only when a row carries it
            print(f"# Commands of the {len(have)} {_rows_word(df, len(have))} "
                  f"whose input files are in place."
                  + (f" [env missing] marks a {_rows_word(df, 1)} whose environment is "
                     f"not installed." if missing else "")
                  + (" [use multibench run-all] marks a command that reads a file "
                     "multibench run-all writes first." if prepared else ""))
            for _, r in have.iterrows():
                tag = "" if r["env_ok"] else " [env missing]"
                if _prepared_at(r.get("caveat", "")) >= 0:
                    tag += " [use multibench run-all]"
                # a data_dir method's modalities already read "(data_dir)"
                mods = str(r["modalities"])
                mods = mods if mods.startswith("(") else f"({mods})"
                print(f"{r['method']} {mods}{tag}: {r['command']}")
        return _EXIT_OK
    with _quiet_stdout(), _setting("leiden_flavor", args.leiden_flavor):
        res = workflow._run_all(args.dataset, args.category, out_dir=args.out,
                                methods=_csv_list(args.methods),
                                modalities=_modalities_arg(args.modalities),
                                data_path=args.data_path, params=params,
                                evaluate=not args.no_evaluate, dry_run=False,
                                timeout=args.timeout, skip_existing=args.skip_existing,
                                batch=batch,
                                allow_atac_mismatch=args.allow_atac_mismatch)
    _print_frame(res.summary, columns=columns, fmt=args.format)
    print(f"saved under {args.out}", file=sys.stderr)
    not_run = _not_run_line(res, Path(args.out) / "summary.csv")
    if not_run:
        print(not_run, file=sys.stderr)
    failed = _failed_line(res, Path(args.out) / "failures.csv")
    if failed:
        print(failed, file=sys.stderr)
        return _EXIT_METHOD_FAILED
    return _EXIT_OK


#: how the stderr line of ``run-all`` counts the statuses of ``failures``, in its order
_FAILED_WORDS = ("failed", "timed out", "not scored", "skipped")


def _failed_word(status) -> str:
    """The word :func:`_failed_line` counts a ``failures`` status under."""
    return {"skipped": "skipped", "timeout": "timed out", "ran": "not scored"}.get(
        workflow._status(status), "failed")


def _saved_methods(where, status=None) -> set | None:
    """The methods of a saved CSV (with ``status`` when given); ``None`` when unreadable."""
    try:
        df = pd.read_csv(where)
    except Exception:  # noqa: BLE001 - no saved file to compare with
        return None
    if status is not None:
        # a 0.4.0 summary.csv holds the record codes (SKIPPED)
        df = df[df["status"].map(workflow._status) == status]
    return set(df["method"].astype(str))


def _kept_sentence(methods) -> str:
    """The sentence naming methods whose ``SKIPPED`` record did not replace an earlier one."""
    return (f" The folder keeps the earlier record of {', '.join(methods)}."
            if methods else "")


def _failed_line(res, where) -> str | None:
    """``# 1 failed (Seurat_WNN), 1 skipped (totalVI). See runs/failures.csv.``, or
    ``None``: the stderr line of a ``run-all`` whose ``failures`` is not empty."""
    bad = res.failures
    if bad.empty:
        return None
    saved = _saved_methods(where)
    groups: dict = {w: [] for w in _FAILED_WORDS}
    for m, s in zip(bad["method"], bad["status"]):
        if str(m) not in groups[_failed_word(s)]:
            groups[_failed_word(s)].append(str(m))
    counts = ", ".join(f"{len(ms)} {w} ({', '.join(ms)})" for w, ms in groups.items()
                       if ms)
    kept = [str(m) for m in dict.fromkeys(bad["method"])
            if saved is not None and str(m) not in saved]
    see = (f" See {where}." if saved is None or saved & set(map(str, bad["method"]))
           else "")
    return f"# {counts}.{see}{_kept_sentence(kept)}"


def _not_run_line(res, where) -> str | None:
    """``# Not run: MIRA, moETM (skipped). ...``, or ``None``: the stderr line naming
    the ``SKIPPED`` methods ``--methods`` did not name (they do not set exit 3)."""
    skipped = list(dict.fromkeys(str(r.get("method"))
                                 for r in res.records
                                 if workflow._status(r.get("status")) == "skipped"
                                 and not r.get("requested")))
    if not skipped:
        return None
    saved = _saved_methods(where, status="skipped")
    kept = [m for m in skipped if saved is not None and m not in saved]
    why = (" The reason column of summary.csv says why."
           if saved is None or set(skipped) - set(kept) else "")
    return f"# Not run: {', '.join(skipped)} (skipped).{why}{_kept_sentence(kept)}"


@contextlib.contextmanager
def _setting(name: str, value):
    """Set ``mtb.config.DEFAULT.<name>`` for one command (``None`` = unchanged)."""
    if value is None:
        yield
        return
    before = getattr(config.DEFAULT, name)
    setattr(config.DEFAULT, name, value)
    try:
        yield
    finally:
        setattr(config.DEFAULT, name, before)


def _evaluate_labels(args, stack, method):
    """The ``labels=`` value of ``multibench evaluate``: the ``--labels`` file(s), or
    ``None`` for the label files of ``--dataset``, which evaluate reads in
    ``method``'s cell order when it is a package method."""
    files = list(args.labels or [])
    if not files:
        if args.dataset is None:
            _usage_error(args, "need --labels CSV (repeatable), or --dataset to read the "
                         "dataset's label files")
        dataset, root = resolve.split_dataset(args.dataset, args.data_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # evaluate reads them again and says it
            labels = pipeline._dataset_labels(dataset, args.category, method, root)
        own = registry.method_of(method)
        order = f"in {own}'s cell order" if own else "in that order"
        print(f"# labels: {', '.join(Path(v).name for v in labels.values())} from "
              f"{dataset}, {order}", file=sys.stderr)
        if args.column is None:
            return None
        files = list(labels.values())
    elif args.data_path is not None:
        _usage_error(args, "--data-path reads the labels from the dataset folder; "
                     "drop it when --labels names the files")
    else:
        bad = _label_order_problem(files, method, args.dataset, args.category)
        if bad:
            raise ValueError(bad)
    if args.column is not None:
        # one column out of each file, written as column x after the file's
        # first column, which may hold the cell ids evaluate aligns by; several
        # files still count as batches
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        picked = []
        for i, f in enumerate(files, 1):
            # the file keeps its name: evaluate's errors name the files typed
            out = tmp / str(i) / Path(f).name
            out.parent.mkdir()
            # as the file writes them: a message names the word for a missing label
            vals, first = eio.read_labels_ids(f, args.column, as_written=True)
            cols = {"x": vals} if first is None else {"id": first.to_numpy(), "x": vals}
            pd.DataFrame(cols).to_csv(out, index=False, sep=eio._sep_for(out))
            picked.append(str(out))
        files = picked
    return files[0] if len(files) == 1 else files


def _batch_arg(args, stack):
    """``--batch``, or with ``--batch-column`` a copy of that file with only the cell
    ids and that column, which ``stack`` removes."""
    batch, column = args.batch, args.batch_column
    if column is None:
        return batch
    if batch is None:
        _usage_error(args, "--batch-column needs --batch")
    return pipeline._one_column_file(batch, column, stack)


def _label_order_problem(files, method, dataset, category) -> str | None:
    """Why the ``--labels`` files contradict ``--method``'s cell order, or ``None``;
    checked only for two or more of the dataset's label files of a package method."""
    names = [Path(f).name for f in files]
    if (None in (method, dataset, category) or len(names) < 2
            or len(set(names)) != len(names)):
        return None
    try:
        registry.check_method(method)
        registry.check_category(category)
    except (KeyError, ValueError):
        # a row name such as SCALEX_rerun without --name: nothing to check against
        return None
    here = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            labels = resolve.labels_for(dataset, category, method)
        except FileNotFoundError:
            labels = None
        except Exception:  # noqa: BLE001 - no variant for category etc.: not checked
            return None
        if labels is None:
            # no such folder here: order the typed names as that folder would
            if not all("cty" in n and n.endswith(".csv") for n in names):
                return None
            here = False
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / dataset).mkdir()
                for n in names:
                    (Path(tmp) / dataset / n).touch()
                try:
                    labels = resolve.labels_for(dataset, category, method, data_path=tmp)
                except Exception:  # noqa: BLE001 - not checked
                    return None
    expected = [Path(p).name for p in labels.values()]
    if not set(names) <= set(expected):
        return None
    want = [n for n in expected if n in names]
    if names == want:
        return None
    stems = [Path(n).stem for n in want]
    order = (f"{stems[0]} before {stems[1]}" if len(stems) == 2
             else ", then ".join(stems))
    return (f"{method} puts the cells of {order}. Repeat --labels in that order"
            + (", or drop --labels to read them in that order." if here else "."))


def _own_method_flags(name, files) -> str:
    """``--name "RNA+ADT PCA" --labels cty.csv``: the flags for your own method."""
    def q(v):
        v = str(v)
        return v if shlex.quote(v) == v else '"' + v.replace('"', '\\"') + '"'
    labels = " ".join(f"--labels {q(f)}" for f in files) or "--labels CSV"
    return f"--name {q(name)} {labels}"


def _unknown_method_message(method, name=None, files=()) -> str:
    """The error of ``multibench evaluate`` for a ``--method`` the package does not know."""
    msg = f"{method} is not a package method."
    ids = [s.id for s in registry.load()]
    base = max((i for i in ids if len(str(method)) > len(i)
                and str(method).lower().startswith(i.lower())), key=len, default=None)
    near = registry.closest_method(method)
    if name is not None:
        if near:
            msg += f" Did you mean {near}?"
        return (msg + " For your own embedding, leave out --method: "
                + _own_method_flags(name, files) + ".")
    if base:
        msg += f" To name your rows {method}, pass --method {base} --name {method}."
    elif near:
        msg += f" Did you mean {near}?"
    return (msg + " For your own method, pass --name and the label files with "
            "--labels, once per file, in your embedding's cell order.")


def _cmd_evaluate(args) -> int:
    """``multibench evaluate``: :func:`multibench.evaluate` on an embedding file,
    printed or written as the long table."""
    if args.embedding is not None and args.embedding_arg is not None:
        _usage_error(args, "give the embedding once: EMBEDDING or --embedding")
    embedding = args.embedding_arg if args.embedding is None else args.embedding
    if embedding is None:
        _usage_error(args, "the following arguments are required: EMBEDDING")
    method = args.method or "embedding"
    if args.name is not None and args.method is not None:
        try:
            registry.check_method(args.method)
        except KeyError:
            # with --name, --method only sets the label order: it must be known
            raise KeyError(_unknown_method_message(args.method, args.name,
                                                   args.labels or ())) from None
    metrics = _csv_list(args.metrics)
    if metrics is not None and len(metrics) == 1:
        metrics = metrics[0].lower() if metrics[0].lower() in _METRIC_FAMILIES else metrics[0]
    if args.task is not None and metrics is None:     # an older spelling of --metrics
        metrics = args.task
    if args.dataset is not None and args.category is None:
        # a benchmark dataset has one
        args.category = workflow.known_category(
            resolve.split_dataset(args.dataset, args.data_path)[0])
    with contextlib.ExitStack() as stack:
        batch = _batch_arg(args, stack)
        if isinstance(batch, str) and not Path(batch).exists() and not Path(batch).suffix \
                and Path(embedding).suffix.lower() not in (".h5ad", ".h5mu"):
            raise ValueError(f"--batch {batch} is not a file. With a "
                             f"{Path(embedding).suffix or 'matrix'} embedding, --batch "
                             f"takes a CSV of one batch per cell.")
        labels = _evaluate_labels(args, stack, method)
        with _quiet_stdout(), _setting("leiden_flavor", args.leiden_flavor):
            df = pipeline._evaluate(
                embedding, labels, dataset=args.dataset, method=method,
                category=args.category, batch=batch, metrics=metrics, obsm=args.obsm,
                clustering=args.clusters, missing_labels=args.missing_labels,
                data_path=args.data_path, from_dataset=not args.labels)
    df["method"] = args.name or method
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    else:
        # no category is an empty cell, as in the CSV
        print(df.assign(category=df["category"].fillna("")).to_string(index=False))
    return _EXIT_OK


def _flavor_token(flavor, env: str | None = None) -> str:
    """`` flavor=cpu`` for a row whose installed flavour is recorded and whose env has
    two builds, else ``''``."""
    if flavor not in ("cpu", "gpu"):
        return ""
    if env is not None:
        if envs._single_build(env):
            return ""
    return f" flavor={flavor}"


def _cmd_env(args) -> int:
    """``multibench env ...``: environment recipes, preflight and installation."""
    cmd = args.env_cmd
    # an install off Linux stops with the error alone
    if cmd in ("status", "plan", "doctor"):
        _platform_note()
    if cmd == "status":
        methods = _csv_list(args.methods)
        keep = None
        if methods:
            keep = set(registry.check_method(m) for m in methods)
        elif args.category:
            keep = set(registry.list_methods(category=args.category))
        seen_tags = []
        for r in envs._status():
            if keep is not None and r["method"] not in keep:
                continue
            mark = envs.env_mark(r["exists"], r["has_lock"])
            tag = r["difficulty"] + ("*" if r["verified_working"] else "")
            if r["difficulty"] not in seen_tags:
                seen_tags.append(r["difficulty"])
            print(f"[{mark}] {r['method']:16} {r['env']:18} {tag}"
                  f"{_flavor_token(r.get('flavor'), r['env'])}")
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
        flavor = args.flavor
        sizes = envs.packed_sizes()
        manifest = envs.packed_manifest()
        rows = envs._plan(category=args.category, methods=_csv_list(args.methods))
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
                  f"{_flavor_token(p.get('flavor'), p['env'])}")
        plan_layers = envs.layers_for(
            [envs.archive_key(p["env"], p["flavor"]) for p in summed], missing_only=False)
        print("# " + envs.plan_totals(summed, sizes, layers=plan_layers)[1], file=sys.stderr)
        note = envs.auto_flavor_note(flavor, planning=True)
        if note and any(p["flavor"] == "cpu" for p in summed):
            print(note, file=sys.stderr)
        return _EXIT_OK
    if cmd == "doctor":
        rows = envs._doctor_rows(category=args.category, methods=_csv_list(args.methods))
        for r in rows:
            mark = envs.env_mark(r["exists"], r["has_lock"])
            print(f"[{mark}] {r['env']:18} ({len(r['methods']):2}) <- {', '.join(r['methods'])}"
                  f"{_flavor_token(r.get('flavor'), r['env'])}")
            if r.get("other_env"):
                print(f"#   {envs.other_env_text(r['env'], r['other_env'], 'install.')}")
        missing = [r for r in rows if not r["exists"]]
        nolock = [r["env"] for r in missing if not r["has_lock"]]
        print(f"# {len(rows)} envs needed, {len(missing)} missing"
              + (f"; no lockfile for: {', '.join(nolock)}" if nolock else ""))
        print(f"# legend: {envs.MARK_LEGEND}")
        if missing:
            miss_methods = sorted({m for r in missing for m in r["methods"]})
            where = " (on the Linux machine)" if envs.host_platform_problem() else ""
            print(f"# next{where}: multibench env install --methods "
                  f"{','.join(miss_methods)}")
        if args.strict and missing:
            return _EXIT_ERROR
        return _EXIT_OK
    if cmd == "install":
        flavor, packed = args.flavor, not args.build
        with _quiet_stdout():
            rows = envs._install_rows(_csv_list(args.methods), category=args.category,
                                      packed=packed, dry_run=args.dry_run,
                                      force=args.force, flavor=flavor)
        width = max([14] + [len(r["state"]) for r in rows])
        for r in rows:
            extra = ""
            if r["state"] == "packed archive published":
                extra = (f" {envs._gb(r.get('archive_bytes')):>8} dl "
                         f"{envs._gb(r.get('unpacked_bytes')):>8} disk  {r['packed_url']}")
            elif r["state"] in ("have", "PACKED"):
                extra = _flavor_token(r.get("flavor"), r["env"])
            print(f"{r['env']:18} [{r['state']:{width}}] <- {', '.join(r['methods'])}{extra}")
            if r.get("other_env"):
                print(f"#   {envs.other_env_text(r['env'], r['other_env'], 'install.')}")
        taken = [r["env"] for r in rows if r["state"] == "name taken"]
        if taken:
            print(f"# name taken: {', '.join(taken)} - rename or remove "
                  f"{'that env' if len(taken) == 1 else 'those envs'} first",
                  file=sys.stderr)
        if args.dry_run and any(not r["exists"] and r["state"] != "name taken"
                                for r in rows):
            todo = envs.to_download(rows) if packed else []
            if todo:
                print("# " + envs.plan_totals(todo)[1], file=sys.stderr)
            print("# dry run - run it without --dry-run on a Linux machine"
                  if envs.host_platform_problem() else
                  "# dry run - leave out --dry-run to create the missing envs"
                  + (" (packed archives first, lockfile build otherwise)"
                     if packed else " from their lockfiles"), file=sys.stderr)
        return _EXIT_OK
    if cmd == "freeze":
        if args.all:
            for env in envs.required_envs(category=args.category):
                try:
                    print(f"froze {env} -> {envs.freeze(env)}")
                except Exception as e:  # noqa: BLE001 - report per-env, keep going
                    print(f"skipped {env}: {str(e)[:120]}")
        else:
            if not args.env:
                _usage_error(args, "name an env to freeze, or pass --all")
            print(f"froze {args.env} -> {envs.freeze(args.env)}")
        return _EXIT_OK
    if cmd == "create-group":
        with _quiet_stdout():
            cmds = envs.create_group(args.group, dry_run=not args.run, force=args.force)
        if not args.run:
            print("# dry-run - add --run to execute:", file=sys.stderr)
            for c in cmds:
                print(shlex.join(c))
        else:
            print(f"created group env {args.group}")
        return _EXIT_OK
    # recipe, yml, create: the env name scan, run and env doctor expect
    method = args.method
    expected = envs.default_env_name(method)      # KeyError (did-you-mean) on a typo
    name = args.name or expected
    if name == expected:
        banner = (f"# env {name!r} is the name scan/run-all/env install expect for {method} "
                  f"(multibench info {method} shows it). multibench env create "
                  f"{method} builds the same env from its lockfile.")
    else:
        banner = (f"# env {name!r} is a custom --name: scan/run-all/env install expect "
                  f"{expected!r} for {method} and will not find this one")
    if cmd == "recipe":
        print(banner)
        for c in envs.create_commands(method, env_name=name):
            print(shlex.join(c))
        return _EXIT_OK
    if cmd == "yml":
        y = banner + "\n" + envs.environment_yml(method, env_name=name)
        if args.out:
            with open(args.out, "w") as f:
                f.write(y)
            print(f"wrote {args.out}")
        else:
            print(y, end="")
        return _EXIT_OK
    with _quiet_stdout():                         # create
        cmds = envs.create(method, env_name=name, dry_run=not args.run, force=args.force)
    if not args.run:
        print("# dry-run - add --run to execute:", file=sys.stderr)
        for c in cmds:
            print(shlex.join(c))
    else:
        print(f"created environment for {method}")
    return _EXIT_OK


# ----------------------------------------------------------------- parser
class _Parser(argparse.ArgumentParser):
    """An unknown command's error lists only the commands ``--help`` lists."""

    def _check_value(self, action, value):
        if isinstance(action, argparse._SubParsersAction) and value not in action.choices:
            shown = ", ".join(a.dest for a in action._choices_actions)
            raise argparse.ArgumentError(action, f"invalid choice: {value!r} "
                                                 f"(choose from {shown})")
        super()._check_value(action, value)


class _HelpFormatter(argparse.HelpFormatter):
    """argparse's formatter, except that a text holding line breaks (a recipe
    epilog) is printed line by line instead of being re-wrapped."""

    def _fill_text(self, text, width, indent):
        if "\n" in text:
            return "".join(indent + line for line in text.splitlines(keepends=True))
        return super()._fill_text(text, width, indent)


def _status_epilog() -> str:
    """The ``env status`` help epilog: one difficulty tag per line."""
    rows = [(tag, text) for tag, text in envs.DIFFICULTY.items()]
    star, _, star_text = envs.VERIFIED_STAR.partition(" = ")
    field, _, what = star_text.partition(": ")
    rows.append((star, f"after a tag: {what} ({field})"))
    width = max(len(tag) for tag, _ in rows) + 2
    lines = ["difficulty tags (how hard the environment is to build from its recipe):"]
    for tag, text in rows:
        lines += textwrap.wrap(text, width=78, initial_indent=f"  {tag:<{width}}",
                               subsequent_indent=" " * (width + 2),
                               break_on_hyphens=False)
    lines += [""] + textwrap.wrap("marks: " + envs.MARK_LEGEND, width=78,
                                  subsequent_indent="  ", break_on_hyphens=False)
    lines.append("The same legend is printed on stderr.")
    return "\n".join(lines)


_CATEGORY_HELP = ("vertical, diagonal, mosaic or cross (multibench layout explains "
                  "them).")
_METHODS_HELP = "comma-separated method ids, as printed by multibench find"
#: the tail of the deprecated ``--leiden-flavor`` of run-all and evaluate
_LEIDEN_TAIL = _compat.use("the default (the article's leidenalg)")
_FLAVOR_HELP = ("cpu or gpu build of each env; auto (default) = gpu when this computer "
                "has an NVIDIA GPU (on a login node for GPU jobs, pass gpu). In the "
                "plan, any = one build serves both")
#: the tail of the deprecated ``--allow-atac-mismatch`` of scan and run-all
_ATAC_MISMATCH_TAIL = _compat.use("an ATAC file in the form the method reads")
#: the tail of the deprecated ``--data-path`` of scan, run-all and evaluate
_DATA_PATH_TAIL = _compat.use("the dataset folder path")
_FORCE_HELP = ("try anyway on a computer that is not Linux (method environments run "
               "only on Linux; without --force the command refuses there before any "
               "download)")
#: the family tokens ``--metrics`` accepts besides a comma list of codes
_METRIC_FAMILIES = ("clustering", "batch", "classification", "all")


def _export_arguments(pc) -> None:
    """The arguments of ``export`` and of its deprecated name ``convert``."""
    pc.add_argument("src", help="the .h5ad or .h5mu file. A .h5mu file needs mod: "
                               "selectors, as --rna mod:rna")
    pc.add_argument("out", metavar="DATASET_DIR", help="the dataset folder to write")
    pc.add_argument("--category", help=_CATEGORY_HELP + " Sets the file names. "
                    "vertical writes atac.h5. diagonal writes atac_peak.h5 or "
                    "atac_gas.h5, and the labels as rna_cty.csv and atac_cty.csv. "
                    "mosaic writes atac<i>.h5.")
    pc.add_argument("--rna", help="where the raw-count RNA matrix lives: X (default), "
                                  "obsm:<key>, layer:<key>, mod:<name>, or none. Append "
                                  "[<var column>=<value>] to keep some features, e.g. "
                                  "\"X[feature_types=Gene Expression]\"")
    pc.add_argument("--adt", help="the ADT matrix, as for --rna, e.g. obsm:protein")
    pc.add_argument("--atac", help="the ATAC matrix, as for --rna, e.g. "
                                   "\"X[feature_types=Peaks]\". Peaks or gene activity is "
                                   "read from the feature names, or set by --atac-kind")
    pc.add_argument("--atac-from", dest="atac_from", metavar="PATH",
                    help="the file that holds the ATAC matrix, another .h5ad or .h5mu: "
                         "the ATAC cells for --category diagonal, else the same cells, "
                         "matched by barcode. --atac selects from it, default X")
    pc.add_argument("--atac-kind", dest="atac_kind", choices=["peak", "gene_activity"],
                    help="peak writes atac_peak.h5, and atac.h5 without --category. "
                         "gene_activity writes atac_gas.h5. --category vertical writes "
                         "atac.h5 and mosaic atac<i>.h5 for both")
    pc.add_argument("--labels", help="cell-type column: obs:<col>, or <mod>:<col> for a "
                                     "modality of a .h5mu file. Written to cty.csv, for "
                                     "diagonal to rna_cty.csv and atac_cty.csv")
    pc.add_argument("--batch", help="split SRC into numbered batches by this column, "
                                    "as for --labels: rna1.h5, rna2.h5, cty1.csv ..., read "
                                    "by mosaic and cross methods")
    pc.add_argument("--batch-index", dest="batch_index", type=int, metavar="N",
                    help="write all of SRC as batch N, one call per batch. --category "
                         "mosaic writes rna<N>.h5, adt<N>.h5, atac<N>.h5 and cty<N>.csv, "
                         "cross the same with atac_peak<N>.h5, and diagonal rna<N>.h5, "
                         "atac_gas<N>.h5 or atac_peak<N>.h5, rna_cty<N>.csv and "
                         "atac_cty<N>.csv")
    pc.add_argument("--overwrite", action="store_true",
                    help="replace files already in DATASET_DIR (default: refuse and "
                         "list them). Files of other data that the call does not write "
                         "are refused either way")
    # the one-file mode of convert and the 0.3 flags: they parse, with the warning
    pc.add_argument("--modality", action=_Retired,
                    tail=_compat.use("multibench export --rna/--adt/--atac"))
    pc.add_argument("--layer", action=_Retired, tail=_compat.use("--rna layer:<key>"))
    pc.add_argument("--obsm", action=_Retired, tail=_compat.use("--adt obsm:<key>"))
    pc.add_argument("--mod", action=_Retired, tail=_compat.use("--rna, --adt or --atac "
                                                               "mod:<name>"))
    pc.add_argument("--n-batches", dest="n_batches", type=int, action=_Retired,
                    tail=_compat.use("a new dataset folder"))
    pc.add_argument("--dtype", default="float64", action=_Retired,
                    tail=_compat.use("the default float64"))


def build_parser() -> argparse.ArgumentParser:
    """Build the ``multibench`` argument parser (every command and flag has help text)."""
    p = _Parser(
        prog="multibench",
        description="Run, evaluate and plot single-cell multimodal integration "
                    "methods from the scMultiBench benchmark. Each command wraps "
                    "one function of the Python API (import multibench as mtb).",
        epilog="Exit codes: 0 means success. 1 is a runtime error, printed as "
               "error: ... on stderr. MULTIBENCH_DEBUG=1 shows the traceback. 2 is a "
               "usage error. 3 means run-all finished, but a method failed or a "
               "method named in --methods was skipped. Without --methods, a skipped "
               "method is only logged and marked skipped in summary.csv. "
               "Run multibench <command> --help for the flags of a command.")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}",
                   help="print the multibench version and exit")
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>",
                           title="commands")

    # ---- list: deprecated, not listed (no help=): multibench find
    pl = sub.add_parser("list", description="Deprecated: use multibench find.")
    pl.add_argument("--category", help=_CATEGORY_HELP)
    pl.add_argument("--task", action=_Retired, tail=_compat.note("it has no replacement"))
    pl.add_argument("--runnable", action=_Retired, nargs=0, default=False,
                    tail=_compat.use("multibench scan"))
    pl.set_defaults(func=_cmd_list, _parser=pl, **_retired_command("list", "find"))

    # ---- find
    pf = sub.add_parser(
        "find", help="find methods by category/modalities/labels (mtb.find_methods)",
        description="Print method ids matching all given filters, one per line. "
                    "Every filter is optional; with none, every method is listed.")
    pf.add_argument("--category", help=_CATEGORY_HELP)
    pf.add_argument("--task", action=_Retired, tail=_compat.note("it has no replacement"))
    pf.add_argument("--modalities",
                    help="comma-separated modalities the method must consume, e.g. "
                         "rna,adt or rna,atac_peak ('protein' is accepted for adt; "
                         "atac_peak and atac_gas select the ATAC form)")
    pf.add_argument("--needs-labels", nargs="?", const="true", choices=["true", "false"],
                    metavar="{true,false}",
                    help="true (or the bare flag): only methods that consume cell-type "
                         "labels (supervised); false: only label-free methods; absent: "
                         "no filter. Note: the optional value must come right after "
                         "the flag")
    pf.add_argument("--atac", choices=["peak", "gene_activity"], action=_Retired,
                    tail=_compat.use("--modalities atac_peak or atac_gas"))
    pf.add_argument("--runnable", action=_Retired, nargs=0, default=False,
                    tail=_compat.use("multibench scan"))
    pf.add_argument("--tunable", nargs="?", const="true", choices=["true", "false"],
                    action=_Retired, tail=_compat.use("multibench params METHOD"))
    pf.set_defaults(func=_cmd_find, _parser=pf)

    # ---- params
    ppa = sub.add_parser(
        "params", help="the hyperparameters a method accepts (mtb.params_for)",
        description="Print, per variant of METHOD, the tunable hyperparameters "
                    "(key, type, the default a run uses) that run-all --param "
                    "accepts, then the values the script "
                    "pins (fixed_in_script, with file:line) and the upstream library "
                    "knobs it does not expose. Without --category/--modalities every "
                    "variant is printed. An empty tunable table means the script "
                    "hardcodes its hyperparameters.")
    ppa.add_argument("method", help="method id, as printed by multibench find. An "
                                    "unknown id gets a did-you-mean error")
    ppa.add_argument("--category", help=_CATEGORY_HELP + " Only that category's variants.")
    ppa.add_argument("--modalities", help="comma-separated modality roles selecting one "
                                          "variant, e.g. rna,adt ('protein' is accepted "
                                          "for adt)")
    ppa.add_argument("--format", choices=["table", "csv", "tsv", "json"], default="table",
                     action=_Retired, tail=_compat.use("mtb.params_for"))
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
    pin.add_argument("method", help="method id, as printed by multibench find")
    pin.add_argument("--format", choices=["text", "json"], default="text",
                     action=_Retired, tail=_compat.use("mtb.method_info"))
    pin.set_defaults(func=_cmd_info, _parser=pin)

    # ---- fetch
    pfe = sub.add_parser(
        "fetch", help="download demo datasets (mtb.data.fetch)",
        description="Download the demo datasets named into the data root, skipping "
                    "those present.")
    pfe.add_argument("datasets", nargs="*", metavar="DATASET",
                     help="dataset ids, space- or comma-separated (the fetchable column "
                          "of mtb.catalog.datasets())")
    pfe.add_argument("--outputs", action=_Retired, nargs=0, default=False,
                     tail=_compat.note("use multibench plot for the stored results"))
    # the cluster page: compute nodes without network fetch the scripts first
    pfe.add_argument("--scripts", action="store_true",
                     help="also fetch the method scripts, so compute nodes need no download")
    # a maintainer switch, as MULTIBENCH_SCRIPTS_REF: not in --help
    pfe.add_argument("--ref", metavar="REF", help=argparse.SUPPRESS)
    pfe.add_argument("--data-path", dest="data_path", action=_Retired,
                     tail=_compat.use("MULTIBENCH_DATA_PATH"))
    pfe.set_defaults(func=_cmd_fetch, _parser=pfe)

    # ---- config
    pco = sub.add_parser(
        "config", help="the resolved paths and where each came from (mtb.config.DEFAULT)",
        description="Print data_path, envs_dir, repo_path and leiden_flavor "
                    "with the source of each value: an environment "
                    "variable (MULTIBENCH_DATA_PATH, MULTIBENCH_ENVS_DIR, "
                    "MULTIBENCH_REPO_PATH), the conda probe or the default. "
                    "scripts_commit is the commit of the method scripts in "
                    "repo_path.")
    pco.add_argument("--get", choices=["data_path", "envs_dir", "repo_path", "leiden_flavor"],
                     action=_Retired, tail=_compat.use("multibench config"))
    pco.add_argument("--format", choices=["text", "json"], default="text",
                     action=_Retired, tail=_compat.use("mtb.config.DEFAULT"))
    pco.set_defaults(func=_cmd_config, _parser=pco)

    # ---- scan
    ps = sub.add_parser(
        "scan", help="which methods can run on a dataset, and why not the rest (mtb.scan)",
        description="Print one row per method: the modalities of the variant run-all "
                    "would run, whether it can run here and why not, as mtb.scan "
                    "returns it. Nothing runs.")
    ps.add_argument("dataset", help="the dataset folder: its path, or its name under the "
                                   "data root (multibench config)")
    ps.add_argument("--category", help=_CATEGORY_HELP + " Default: the benchmark's "
                                                        "category for a dataset it scored, "
                                                        "such as vertical for D11, else "
                                                        "every category.")
    ps.add_argument("--data-path", dest="data_path", action=_Retired,
                    tail=_compat.use("the dataset folder path"))
    ps.add_argument("--methods", help=_METHODS_HELP + ". Only those rows. An unknown "
                                                      "id gets a did-you-mean error")
    ps.add_argument("--modalities", help="comma-separated modality roles to restrict the "
                                         "variants to, e.g. rna,adt. protein is accepted "
                                         "for adt. atac_peak and atac_gas select by what "
                                         "the method reads. Default: the combinations "
                                         "whose modalities the folder holds. all lists "
                                         "every combination")
    ps.add_argument("--columns", action=_Retired,
                    tail=_compat.use("multibench run-all --dry-run --format csv"))
    ps.add_argument("--format", choices=["table", "csv", "tsv", "json"], default="table",
                    action=_Retired, tail=_compat.use("multibench run-all --dry-run --format"))
    ps.add_argument("--strict", action=_Retired, nargs=0, default=False,
                    tail=_compat.use("the exit code of multibench run-all"))
    # the cluster page: check a job for a GPU node from a login node without one
    ps.add_argument("--assume-gpu", dest="assume_gpu", action="store_true",
                    help="skip this host's GPU test, to check a job for a GPU node from "
                         "a login node")
    ps.add_argument("--allow-atac-mismatch", dest="allow_atac_mismatch", action=_Retired,
                    nargs=0, default=False, tail=_ATAC_MISMATCH_TAIL)
    ps.set_defaults(func=_cmd_scan, _parser=ps)

    # ---- layout
    play = sub.add_parser(
        "layout", help="how to lay out your dataset on disk (mtb.describe_layout)",
        description="Print the directory layout and role -> filename contract the "
                    "package expects, optionally for one category.")
    play.add_argument("category", nargs="?", help="vertical, diagonal, mosaic or cross. "
                      "Optional: without it, the layout of every category is printed.")
    play.set_defaults(func=_cmd_layout, _parser=play)

    # ---- export (convert: its deprecated name)
    pc = sub.add_parser(
        "export", help="write an .h5ad or .h5mu as a dataset folder (mtb.io.export_dataset)",
        description="Write the dataset folder that scan and run-all read: rna.h5, "
                    "adt.h5, the ATAC files and the label files. The RNA comes from X "
                    "unless --rna says where it is. Give raw counts: the methods "
                    "normalise the data themselves.",
        epilog=_CONVERT_EPILOG, formatter_class=_HelpFormatter)
    _export_arguments(pc)
    pc.set_defaults(func=_cmd_convert, _parser=pc, _export=True)
    pcv = sub.add_parser("convert", description="Deprecated: use multibench export.")
    _export_arguments(pcv)
    pcv.set_defaults(func=_cmd_convert, _parser=pcv, **_retired_command("convert", "export"))

    # ---- cite
    pci = sub.add_parser(
        "cite", help="citation entries for the benchmark and the methods you used (mtb.cite)",
        description="Print the scMultiBench citation, then one entry per method id "
                    "given (in that order). Methods without a verified DOI are "
                    "emitted as a comment naming their repository.")
    pci.add_argument("methods", nargs="*", metavar="METHOD",
                     help="method ids to cite, space- or comma-separated: cite A B "
                          "and cite A,B are the same. Without ids, only the benchmark "
                          "entry is printed")
    pci.add_argument("--all", action=_Retired, nargs=0, default=False,
                     tail=_compat.use("multibench cite $(multibench find)"))
    pci.add_argument("--format", choices=["text", "bibtex"], default="text",
                     help="text (default; one line per entry) or bibtex (one @article "
                          "per entry)")
    pci.add_argument("--out", action=_Retired, tail=_compat.use("> FILE"))
    pci.set_defaults(func=_cmd_cite, _parser=pci)

    # ---- plot
    pp = sub.add_parser(
        "plot", help="draw the bubble table or summary bars of a results frame "
                     "(mtb.plot.bubble / mtb.plot.bar)",
        description="Plot the stored results (--category [--dataset]), your own "
                    "long table (--input long.csv or a run-all output folder, as "
                    "multibench evaluate and multibench run-all write them), or both: "
                    "--input together with --category adds your rows to the stored "
                    "results.")
    pp.add_argument("kind", choices=["bubble", "bar"], metavar="{bubble,bar}",
                    help="bubble: paper-style bubble table (methods x metrics, best "
                         "first, Overall bars per family) | bar: one bar per method "
                         "with its across-dataset Overall score")
    pp.add_argument("--category", help=_CATEGORY_HELP + " Selects the stored results. "
                                                        "Required unless --input is given.")
    pp.add_argument("--dataset", help="dataset id(s), comma-separated; with --category "
                                      "selects the stored results, with --input "
                                      "filters the rows")
    pp.add_argument("--source", choices=["published", "rerun"], action=_Retired,
                    tail=_compat.note("leave it out to plot the stored results"))
    pp.add_argument("--result-path", dest="result_path", action=_Retired,
                    tail=_compat.use("--input"))
    pp.add_argument("--input", action="append", metavar="LONG_CSV_OR_DIR",
                    help="a long results CSV (metric,value,method,dataset,category[,source]) or "
                         "a run-all output folder; repeatable. Alone: the table to "
                         "plot. With --category: added to the stored results; "
                         "--dataset then filters these rows too")
    pp.add_argument("--methods", help=_METHODS_HELP + "; only those rows (unknown name "
                                                      "-> error)")
    pp.add_argument("--metrics", help="comma-separated metric codes to draw, in this "
                                      "order (e.g. ARI,NMI,ASW), or a family: clustering, "
                                      "batch, classification, all (default: all present)")
    pp.add_argument("--aggregate", choices=["dataset", "summary"], default="dataset",
                    action=_Retired, tail=_compat.use("multibench plot bar"))
    # only a frame of several datasets can tell the two apart: not in --help
    pp.add_argument("--overall", choices=["rank", "mean_overall"], default=None,
                    help=argparse.SUPPRESS)
    pp.add_argument("--require-complete", dest="require_complete", action=_Retired,
                    nargs=0, default=False, tail=_compat.use("--methods"))
    pp.add_argument("--na", choices=["warn", "skip", "raise"], action=_Retired,
                    tail=_compat.note("leave it out for a warning that names the n/a cells"))
    pp.add_argument("--group", choices=["clustering", "batch", "classification"],
                    dest="metrics", action=_Retired, tail=_compat.use("--metrics"))
    pp.add_argument("--top", type=int, action=_Retired, tail=_compat.use("--methods"))
    pp.add_argument("--title", help="figure title (default: the category and dataset "
                                    "given, else none)")
    pp.add_argument("--out", required=True,
                    help="figure file to write; the suffix picks the format "
                         "(.pdf, .png, .svg)")
    pp.set_defaults(func=_cmd_plot, _parser=pp)

    # ---- run
    pr = sub.add_parser(
        "run",
        description="Deprecated: use multibench run-all DATASET --methods M.")
    # any wrong call gets the deprecated line, not a list of hidden flags
    pr.error = lambda message: pr.exit(2, (f"multibench run: error: {message}\n" if message
                                           else "") + f"multibench run: {pr.description}\n")
    pr.add_argument("--method", help=argparse.SUPPRESS)
    pr.add_argument("--category", help=argparse.SUPPRESS)
    pr.add_argument("--task", action=_Retired)
    pr.add_argument("--input", action="append", metavar="ROLE=PATH",
                    help=argparse.SUPPRESS)
    pr.add_argument("--out-dir", "--out", dest="out",
                    help=argparse.SUPPRESS)
    pr.add_argument("--runner", dest="cmd_template", metavar="TEMPLATE", action=_Retired,
                    tail=_compat.use("a job script that calls multibench run-all"))
    pr.add_argument("--param", "-p", action="append", metavar="KEY=VALUE",
                    help=argparse.SUPPRESS)
    pr.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help=argparse.SUPPRESS)
    pr.set_defaults(func=_cmd_run, _parser=pr, cmd_template=None,
                    **_retired_command("run", "run-all DATASET --methods M"))

    # ---- run-all
    pra = sub.add_parser(
        "run-all", help="run the runnable methods of a category on a dataset folder "
                        "(mtb.run_all)",
        description="Run (or with --dry-run, only plan) the methods of the category "
                    "on the dataset folder, score each output and save summary.csv, "
                    "long.csv, failures.csv and batch_result.json under --out-dir. "
                    "multibench plot bubble --input OUT draws the figure.",
        epilog="Exit code 1 means no method could start; 3 means the run finished, "
               "but a method failed or a method named in --methods was skipped. "
               "Without --methods, a skipped method is only logged and marked "
               "skipped in summary.csv.")
    pra.add_argument("dataset", help="the dataset folder: its path, or its name under the "
                                     "data root (multibench config)")
    pra.add_argument("--category", required=True, help=_CATEGORY_HELP)
    pra.add_argument("--out-dir", dest="out",
                     help="directory for all outputs (one sub-folder per method); "
                          "required unless --dry-run")
    # 0.3's spelling; --out names a file in plot and evaluate
    pra.add_argument("--out", dest="out", help=argparse.SUPPRESS)
    pra.add_argument("--methods", help=_METHODS_HELP + "; only those (default: every "
                                                       "runnable method)")
    pra.add_argument("--modalities", help="comma-separated modalities to restrict the "
                                          "method variants to, e.g. rna,adt; atac_peak "
                                          "and atac_gas select by what the method reads. "
                                          "Default: the combinations whose modalities "
                                          "the folder holds. all lists every combination")
    pra.add_argument("--data-path", dest="data_path", action=_Retired, tail=_DATA_PATH_TAIL)
    pra.add_argument("--dry-run", dest="dry_run", action="store_true",
                     help="print the plan (one row per method variant) and the commands; "
                          "nothing runs")
    pra.add_argument("--assume-gpu", dest="assume_gpu", action=_Retired, nargs=0,
                     default=False, tail=_compat.use("multibench scan --assume-gpu"))
    pra.add_argument("--allow-atac-mismatch", dest="allow_atac_mismatch", action=_Retired,
                     nargs=0, default=False, tail=_ATAC_MISMATCH_TAIL)
    pra.add_argument("--param", "-p", action="append", metavar="METHOD:KEY=VALUE",
                     help="one hyperparameter override, repeatable, e.g. --param "
                          "Matilda:epochs=5 --param Matilda:lr=0.001. This is "
                          "params={'Matilda': {'epochs': 5, 'lr': 0.001}} in Python. "
                          "VALUE is read as an int (5), a float (0.1), true/false or "
                          "text. A key the method does not accept is rejected, also "
                          "with --dry-run. multibench params METHOD lists the keys. Not "
                          "allowed with --skip-existing")
    pra.add_argument("--columns", action=_Retired, tail=_compat.use("--format csv"))
    pra.add_argument("--format", choices=["table", "csv", "tsv", "json"], default="table",
                     help="output format of the printed table (default table; csv/tsv/"
                          "json never clip text and, for --dry-run, carry the command "
                          "column)")
    pra.add_argument("--timeout", type=float,
                     help="per-method wall-clock limit in seconds (default: none)")
    pra.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                     help="reuse a method's existing output under --out-dir instead "
                          "of re-running it, and score it again")
    pra.add_argument("--no-evaluate", dest="no_evaluate", action="store_true",
                     help="run only; do not compute metrics on the outputs")
    pra.add_argument("--batch", metavar="CSV",
                     help="one batch per cell, in the order of the label files or by a "
                          "first column of barcodes (default: one batch per label file)")
    pra.add_argument("--batch-column", dest="batch_column", metavar="NAME",
                     action=_Retired, tail=_compat.use("a one-column --batch CSV"))
    pra.add_argument("--leiden-flavor", dest="leiden_flavor", choices=["igraph", "leidenalg"],
                     action=_Retired, tail=_LEIDEN_TAIL)
    pra.set_defaults(func=_cmd_run_all, _parser=pra)

    # ---- evaluate
    pe = sub.add_parser(
        "evaluate", help="the benchmark's metrics of an embedding (mtb.evaluate)",
        description="Score one embedding file and print the long table (metric, value, "
                    "method, dataset, category, source, scored_with) that multibench "
                    "plot --input reads, or write it with --out.")
    pe.add_argument("embedding_arg", nargs="?", metavar="EMBEDDING",
                    help="the embedding, cells x dims: .h5 (dataset 'data', "
                         "the benchmark's embedding.h5), .h5ad (uses --obsm), .npy, "
                         ".csv/.tsv (a first column of barcodes gives the cell ids: "
                         "labels and batches with barcodes are matched by them)")
    # the 0.4.0 spelling, kept as a hidden alias of EMBEDDING (both work)
    pe.add_argument("--embedding", help=argparse.SUPPRESS)
    pe.add_argument("--output", dest="embedding", action=_Retired,
                    tail=". Pass the embedding as the first argument.")
    pe.add_argument("--labels", action="append", metavar="CSV",
                    help="cell-type labels CSV with a header row, one row per cell "
                         "in the embedding's order; a first column of barcodes is "
                         "aligned to the embedding's cells. With an .h5ad EMBEDDING it "
                         "can name a column of its obs, as celltype. Repeat it for "
                         "several files: they are stacked in the order given, and each "
                         "counts as one batch. Without --labels, --dataset gives them")
    pe.add_argument("--dataset", help="the dataset folder, a path or a name under the "
                                      "data root: its name fills the dataset column, and "
                                      "without --labels its label files are read")
    pe.add_argument("--category", help=_CATEGORY_HELP + " Fills the category column.")
    pe.add_argument("--method", help="fills the method column: a package method id, or "
                                     "your own name. A package method, also as 'StabMap "
                                     "(this run)', reads --dataset's label files in its "
                                     "cell order, with --category")
    pe.add_argument("--task", choices=["clustering", "batch", "all"], action=_Retired,
                    tail=_compat.use("--metrics {value}"))
    pe.add_argument("--data-path", dest="data_path", action=_Retired, tail=_compat.use(
        "--dataset as the folder path"))
    pe.add_argument("--batch", help="per-cell batch labels CSV; a first column of "
                                    "barcodes is aligned as for --labels. With an .h5ad "
                                    "EMBEDDING it can name a column of its obs. Without "
                                    "--metrics the batch metrics are then computed too")
    pe.add_argument("--batch-column", dest="batch_column", metavar="NAME",
                    action=_Retired, tail=_compat.use("a one-column --batch CSV"))
    pe.add_argument("--clustering", "--cluster", dest="clusters", action=_Retired,
                    tail=_compat.use("the default, the article's Leiden clusters"))
    pe.add_argument("--metrics", help="a family (clustering, batch, classification, all), "
                                      "a metric code, or a comma-separated list of codes "
                                      "and families (e.g. ARI,NMI). Default: clustering, "
                                      "plus batch with --batch or several --labels, plus "
                                      "classification with two or more batches outside "
                                      "vertical")
    pe.add_argument("--leiden-flavor", dest="leiden_flavor", choices=["igraph", "leidenalg"],
                    action=_Retired, tail=_LEIDEN_TAIL)
    pe.add_argument("--missing-labels", dest="missing_labels", choices=["error", "drop"],
                    default="error", action=_Retired,
                    tail=_compat.use("label files without missing labels"))
    pe.add_argument("--obsm", help="for .h5ad input: the .obsm key holding the "
                                   "embedding (default: X_emb; X = .X)")
    pe.add_argument("--column", metavar="NAME", action=_Retired,
                    tail=_compat.use("a one-column --labels CSV"))
    pe.add_argument("--name", action=_Retired, tail=_compat.use("--method"))
    pe.add_argument("--out", help="CSV to write (default: print the table)")
    pe.set_defaults(func=_cmd_evaluate, _parser=pe)
    # EMBEDDING is required; nargs="?" only lets the hidden --embedding stand in for it
    pe.usage = pe.format_usage().replace("[EMBEDDING]", "EMBEDDING").split(" ", 1)[1].strip()

    # ---- env
    pv = sub.add_parser(
        "env", help="the methods' conda environments (mtb.env.install)",
        description="Every method runs in its own (or a shared) conda env, unpacked "
                    "from a published archive or built from a committed lockfile. "
                    "env install --dry-run shows each env's state and size.")
    ev = pv.add_subparsers(dest="env_cmd", required=True, metavar="<env-command>",
                           title="env commands")
    es = ev.add_parser("status",
                       description="Deprecated: use multibench env install --dry-run. One line "
                                   "per method: installed or not, the "
                                   "environment name and a difficulty tag.",
                       epilog=_status_epilog(), formatter_class=_HelpFormatter)
    es.add_argument("--category", help=_CATEGORY_HELP + " Only that category's methods.")
    es.add_argument("--methods", help=_METHODS_HELP + "; only those")
    es.set_defaults(func=_cmd_env, _parser=es,
                    **_retired_command("env status", "env install --dry-run"))
    egr = ev.add_parser("groups", description="Deprecated: use multibench env install "
                                               "--dry-run.")
    egr.set_defaults(func=_cmd_env, _parser=egr,
                     **_retired_command("env groups", "env install --dry-run"))
    _NAME_HELP = ("environment name. The default is the environment the package "
                  "uses for METHOD, shown in the env column of multibench scan and "
                  "used by run-all, env install and env create. For example, Matilda uses "
                  "matilda and UINMF uses scmb_r")
    # maintainer commands: no help=, so env --help does not list them
    er = ev.add_parser("recipe",
                       description="(Maintainers.) Print, without running, the "
                                   "hand-written recipe "
                                   "commands that create the environment for METHOD, "
                                   "named as scan/run expect it (first line: a comment "
                                   "naming that env). env create METHOD builds the same "
                                   "env from its committed lockfile - the reproducible "
                                   "path; the recipe is the transparent one.")
    er.add_argument("method", help="method id")
    er.add_argument("--name", help=_NAME_HELP)
    er.set_defaults(func=_cmd_env, _parser=er)
    ey = ev.add_parser("yml",
                       description="(Maintainers.) Emit an environment.yml for METHOD "
                                   "(stdout, or --out) "
                                   "whose name: is the env scan/run expect (the env "
                                   "column of multibench scan).")
    ey.add_argument("method", help="method id")
    ey.add_argument("--name", help=_NAME_HELP)
    ey.add_argument("--out", help="write the yml here instead of stdout")
    ey.set_defaults(func=_cmd_env, _parser=ey)
    ec = ev.add_parser("create",
                       description="(Maintainers.) Create the conda environment for "
                                   "METHOD from its "
                                   "committed lockfile (falling back to the recipe when "
                                   "none is captured), under the name scan/run expect; "
                                   "without --run only the commands are printed.")
    ec.add_argument("method", help="method id")
    ec.add_argument("--name", help=_NAME_HELP)
    ec.add_argument("--run", action="store_true",
                    help="actually create the env; without it the command is a dry run")
    ec.add_argument("--force", action="store_true", help=_FORCE_HELP)
    ec.set_defaults(func=_cmd_env, _parser=ec)
    ep = ev.add_parser("plan",
                       description="Deprecated: use multibench env install (its dry run "
                                   "shows the same rows). Group methods by the conda "
                                   "environment they need, "
                                   "and mark shared and own environments. 'dl' is the "
                                   "packed-archive download size and 'disk' the unpacked "
                                   "size, recorded for this release. '?' means not "
                                   "measured. A '# total' line on stderr sums them.")
    ep.add_argument("--category", help=_CATEGORY_HELP)
    ep.add_argument("--methods", help=_METHODS_HELP + "; only their envs")
    ep.add_argument("--flavor", choices=envs.FLAVORS, default="auto", help=_FLAVOR_HELP)
    ep.set_defaults(func=_cmd_env, _parser=ep, **_retired_command("env plan", "env install"))
    eg = ev.add_parser("create-group",
                       description="(Maintainers.) Create the shared conda env GROUP; "
                                   "without --run only the commands are printed.")
    eg.add_argument("group", help="group name: an env that serves several methods")
    eg.add_argument("--run", action="store_true",
                    help="actually create the env; without it the command is a dry run")
    eg.add_argument("--force", action="store_true", help=_FORCE_HELP)
    eg.set_defaults(func=_cmd_env, _parser=eg)
    edoc = ev.add_parser(
        "doctor",
        description="Deprecated: use multibench env install --dry-run. "
                    "One line per env needed by the selected methods: [x] installed, "
                    "[L] missing but a lockfile is ready, [!] missing and no "
                    "lockfile; then the install command for the missing ones. Method "
                    "envs are Linux-only: on another host a warning says so first.")
    edoc.add_argument("--category", help=_CATEGORY_HELP)
    edoc.add_argument("--methods", help=_METHODS_HELP + "; only their envs")
    edoc.add_argument("--strict", action="store_true",
                      help="exit 1 when any env is missing. For scripts: env doctor "
                           "--strict || env install ...")
    edoc.set_defaults(func=_cmd_env, _parser=edoc,
                      **_retired_command("env doctor", "env install --dry-run"))
    ei = ev.add_parser(
        "install", help="unpack or build every needed env (mtb.env.install)",
        description="Install the environments the selected methods need, from the "
                    "published archives (a lockfile build where none is published). "
                    "Installed ones are skipped. With --dry-run it prints one line per "
                    "environment instead: 'have', 'packed archive published' with the "
                    "archive's download size, unpacked size and URL, or 'no archive - "
                    "lockfile build'. '?' means not measured. A '# total' line goes "
                    "to stderr. They are linux-64 conda environments: on macOS and "
                    "Windows the install refuses.")
    ei.add_argument("--category", help=_CATEGORY_HELP)
    ei.add_argument("--methods", help=_METHODS_HELP + "; only their envs")
    # the maintainer page: build every env from its lockfile instead
    ei.add_argument("--build", action="store_true",
                    help="build every env from its lockfile instead of unpacking the archive")
    ei.add_argument("--packed", action="store_true", help=argparse.SUPPRESS)
    ei.add_argument("--flavor", choices=envs.FLAVORS, default="auto", help=_FLAVOR_HELP)
    ei.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="print the plan (state, sizes, URLs) and install nothing")
    # 0.4.0: installing is the default; --run still parses and changes nothing
    ei.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    ei.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    ei.set_defaults(func=_cmd_env, _parser=ei)
    ef = ev.add_parser("freeze",
                       description="(Maintainers.) Write the lockfile of an installed env.")
    ef.add_argument("env", nargs="?", help="env name to freeze")
    ef.add_argument("--all", action="store_true", help="freeze every required env")
    ef.add_argument("--category", help=_CATEGORY_HELP + " With --all: only that "
                                                        "category's envs.")
    ef.set_defaults(func=_cmd_env, _parser=ef)
    return p


def _refusal(e: Exception) -> bool:
    """``e`` is a ValueError, KeyError or FileNotFoundError the package raised:
    its message names the fix, and a traceback adds nothing."""
    tb = e.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    return (isinstance(e, (ValueError, KeyError, FileNotFoundError)) and tb is not None
            and Path(tb.tb_frame.f_code.co_filename).resolve().is_relative_to(
                Path(__file__).resolve().parent))


def main(argv=None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), run the command, return the exit code.

    The CLI's single error boundary: an exception prints ``error: <message>`` on
    stderr and returns 1 (``MULTIBENCH_DEBUG=1`` re-raises it); warnings print as
    ``warning: <message>``. While it runs, ``config.hint`` names shell commands.
    """
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra:
        # the usage of the command that was given, not the top-level one
        (getattr(args, "_parser", None) or parser).error(
            f"unrecognized arguments: {' '.join(extra)}")
    prev_show, prev_cli = warnings.showwarning, config._CLI
    config._CLI = True

    def _show(message, category, filename, lineno, file=None, line=None):
        print(f"warning: {message}", file=sys.stderr)

    if not os.environ.get("MULTIBENCH_DEBUG"):
        warnings.showwarning = _show
    try:
        _warn_retired(args)
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 - the CLI's single error boundary
        if os.environ.get("MULTIBENCH_DEBUG"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"error: {msg}", file=sys.stderr)
        if not _refusal(e):
            print("(set MULTIBENCH_DEBUG=1 for the full traceback)", file=sys.stderr)
        return _EXIT_ERROR
    finally:
        warnings.showwarning = prev_show
        config._CLI = prev_cli


if __name__ == "__main__":
    raise SystemExit(main())
