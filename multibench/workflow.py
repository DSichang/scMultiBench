"""High-level workflow: point at a dataset, run everything that applies, get metrics and a figure.

The low-level API (``inputs_for`` -> ``run`` -> ``evaluate`` -> ``plot``) requires
a method's name, its integration category and its exact modality combination.
This module works from the dataset instead:

    mtb.scan("D11")                     # what can I run on this data?
    res = mtb.run_all("D11", "vertical", out_dir="out/")  # run all of it, with metrics
    res.plot()                            # one figure

It also handles two traps that otherwise yield wrong numbers without an error:

* **output kind** - not every method returns an embedding. Methods emitting a
  graph are recorded as such instead of being scored with embedding metrics
  (scoring a KNN index matrix gives ARI ~ 0).
* **label order** - ``evaluate`` needs labels in the embedding's cell order, and
  matching by length cannot distinguish orders because every permutation has the
  same length. Candidate orders are scored and the best kept, with the full
  spread recorded so the choice stays auditable.
"""
from __future__ import annotations

import functools
import glob
import itertools
import json
import os
import re
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .discover import _runtimes, _runtime_hint
from .engine import envs, registry, resolve as _resolve
from .engine import runner as _runner
from .engine.runner import run as _run
from .eval import io as _eio, scib as _escib
from .eval.pipeline import evaluate as _evaluate, to_long as _to_long

__all__ = ["scan", "run_all", "BatchResult", "list_categories", "describe_layout",
           "load_batch", "sweep"]



def load_batch(out_dir, *, methods=None) -> "BatchResult":
    """Reload a ``BatchResult`` from the folder ``BatchResult.save`` wrote.

    ``run_all`` saves every sweep, so a finished one can be re-plotted,
    re-scored or inspected later without re-running any method.

    Parameters
    ----------
    out_dir : path-like
        Folder holding ``batch_result.json``: a ``run_all`` ``out_dir`` or a
        ``mtb.data.fetch_outputs`` tree.
    methods : list[str] | None
        Methods whose records to keep; ``None`` = every record.

    Returns
    -------
    BatchResult
        The reloaded sweep. It remembers ``out_dir``, so ``save()`` with no
        argument writes back to the same folder.

    Raises
    ------
    FileNotFoundError
        ``out_dir`` holds no ``batch_result.json``.
    KeyError
        A name in ``methods`` has no record; the message lists the methods that do.

    Examples
    --------
    >>> import multibench as mtb
    >>> res = mtb.load_batch("out/")
    >>> res.summary
    >>> res.plot().savefig("compare.png")
    >>> mtb.load_batch(mtb.data.fetch_outputs("D11"), methods=["Matilda", "scMM"])

    Notes
    -----
    **Files read.** ``batch_result.json`` holds the per-method records.
    ``long.csv``, written when the run produced metrics, restores each
    method's unrounded tidy frame; without it ``BatchResult.long`` is rebuilt
    from the records' rounded ``metrics``.

    **Record order.** ``methods=`` only filters: the kept records stay in the
    order the tree ran them, not the order of ``methods``.

    See Also
    --------
    mtb.BatchResult : the object returned.

    mtb.run_all : writes the folder this function reads.

    mtb.data.fetch_outputs : downloads recorded run outputs in the same layout.
    """
    d = Path(out_dir)
    with open(d / "batch_result.json") as fh:
        blob = json.load(fh)
    recs = blob["records"]
    if methods is not None:
        have = [r.get("method") for r in recs]
        unknown = [m for m in methods if m not in have]
        if unknown:
            raise KeyError(f"no record for {unknown} in {d}; methods in the "
                           f"tree: {have}")
        recs = [r for r in recs if r.get("method") in set(methods)]
    lp = d / "long.csv"
    if lp.exists():
        lng = pd.read_csv(lp)
        for r in recs:
            sub = lng[lng["method"] == r.get("method")]
            r["_long"] = sub if len(sub) else None
    else:
        for r in recs:
            r["_long"] = None
    return BatchResult(recs, blob["dataset"], blob["category"], out_dir=d)

#: The four integration scenarios, and what each one's data looks like.
CATEGORIES = {
    "vertical": "Several modalities measured in the same cells (e.g. CITE-seq "
                "RNA+ADT, or 10x multiome RNA+ATAC). Cells are already matched.",
    "diagonal": "Modalities measured in different cells, with no pairing "
                "(e.g. an RNA experiment and a separate ATAC experiment).",
    "mosaic":   "Several batches where only some share a modality; a paired batch "
                "bridges the others.",
    "cross":    "Several batches in which all modalities are present; the task is "
                "removing batch effects.",
}

#: Modality role -> the file the loader looks for in <data_path>/<dataset>/.
ROLES = {
    "rna":       "rna.h5      - gene expression",
    "adt":       "adt.h5      - surface protein (CITE-seq antibody-derived tags)",
    "atac":      "atac.h5     - chromatin accessibility",
    "atac_gas":  "atac.h5     - ATAC as gene-activity scores  <-- note: plain atac.h5",
    "atac_peak": "peak.h5     - ATAC as peaks                 <-- note: peak.h5, not atac.h5",
    "rna1/rna2/...": "rna1.h5, rna2.h5, ... - one file per batch (mosaic/cross)",
    "adt1/adt2/...": "adt1.h5, adt2.h5, ... - one file per batch (mosaic/cross)",
    "cty":       "cty.csv     - cell-type labels, one label set (vertical)",
    "rna_cty / atac_cty":
                 "rna_cty.csv, atac_cty.csv - one label file per modality, used when "
                 "RNA and ATAC come from different cells (diagonal)",
    "cty1/cty2/...":
                 "cty1.csv, cty2.csv, ... - one label file per batch (mosaic/cross)",
}


def list_categories() -> dict:
    """Return the valid ``category`` values with a plain-language description of each.

    Returns
    -------
    dict
        ``{category: description}`` for ``vertical``, ``diagonal``, ``mosaic``
        and ``cross``: the values ``mtb.run_all`` requires as ``category``.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.list_categories()["vertical"]
    'Several modalities measured in the same cells ...'

    See Also
    --------
    mtb.describe_layout : the file layout each category expects.
    """
    return dict(CATEGORIES)


def describe_layout(category: str | None = None) -> str:
    """Return the directory layout the package expects for your own dataset.

    Start here when bringing your own data, then confirm the folder with
    ``mtb.scan``.

    Parameters
    ----------
    category : str | None
        Integration category whose layout block to include; ``None`` = all
        four.

    Returns
    -------
    str
        The layout description, ready to ``print``.

    Raises
    ------
    ValueError
        Unknown ``category``.

    Examples
    --------
    >>> import multibench as mtb
    >>> print(mtb.describe_layout("vertical"))    # CITE-seq / multiome, cells already matched
    >>> print(mtb.describe_layout("cross"))       # numbered batches
    >>> print(mtb.describe_layout())              # everything

    Notes
    -----
    **What the text covers.** The role -> filename mapping, the numbered
    per-batch files, the ATAC representation trap, the ``.h5`` file format,
    the label CSV and the conda envs.

    **Roles.** A "role" is the name of one input a method takes. For
    CITE-seq the roles are ``rna`` (``rna.h5``) and ``adt`` (``adt.h5``,
    surface protein / antibody-derived tags), plus ``cty.csv`` for cell-type
    labels:

    ```text
    <data_path>/MYCITE/
        rna.h5
        adt.h5
        cty.csv
    ```

    **Several batches.** Mosaic and cross integration use one numbered file
    per batch, in the same flat directory - not sub-folders, and not one
    pre-concatenated matrix. Batch membership is carried by the file
    numbering; there is no batch column. Three batches of CITE-seq:

    ```text
    <data_path>/COREBATCH/
        rna1.h5   adt1.h5   cty1.csv     # batch 1
        rna2.h5   adt2.h5   cty2.csv     # batch 2
        rna3.h5   adt3.h5   cty3.csv     # batch 3
    ```

    **ATAC method lists.** The methods that need gene-activity vs peak ATAC
    matrices are listed from the method registry at call time
    (``find_methods(atac=...)``), so they always agree with ``method_info``
    and the ``atac`` column of ``mtb.scan``.

    **Errors.** An unknown ``category`` raises a ``ValueError`` that lists the
    four.

    See Also
    --------
    mtb.list_categories : the four categories with a description of each.

    mtb.scan : confirms a laid-out folder (``files_ok`` / ``files_reason`` per method).

    mtb.io.export_dataset : writes a whole dataset in this layout from an AnnData.
    """
    registry.check_category(category)       # None passes; typo -> ValueError
    # ATAC representation lists come from the registry, so they cannot go stale.
    from .discover import find_methods as _find_methods
    gas_methods = _find_methods(atac="gene_activity")
    peak_methods = _find_methods(atac="peak")
    lines = ["Put your files in  <data_path>/<DATASET_NAME>/ , e.g. ./data/MYDATA/",
             "  (dataset = the folder name; data_path = the folder that contains it)",
             ""]
    LAYOUTS = {
        "vertical": ["  rna.h5 + adt.h5 (CITE-seq)  or  rna.h5 + atac.h5 (multiome)",
                     "  cty.csv        <- one label file; the cells are already matched"],
        "diagonal": ["  rna.h5         <- the RNA cells",
                     "  atac.h5        <- the ATAC cells (gene activity); peak.h5 for peaks",
                     "  rna_cty.csv and atac_cty.csv",
                     "                 <- one label file per modality. The two cell sets are",
                     "                    disjoint, so they cannot share a single cty.csv."],
        "mosaic":   ["  rna1.h5 rna2.h5 atac2.h5 atac3.h5   <- numbered, one per batch",
                     "  cty1.csv cty2.csv cty3.csv          <- one per batch"],
        "cross":    ["  rna1.h5 rna2.h5 rna3.h5 + adt1.h5 adt2.h5 adt3.h5",
                     "  cty1.csv cty2.csv cty3.csv          <- one per batch"],
    }
    if category in LAYOUTS:
        lines += [f"LAYOUT FOR {category.upper()}:"] + LAYOUTS[category] + [""]
    else:
        for _c, _ls in LAYOUTS.items():
            lines += [f"{_c}:"] + _ls
        lines += [""]
    lines += ["  (numbered files live in the same flat dir; there is no batch column)",
             "", "Modality roles and the filenames they resolve to:"]
    lines += [f"    {k:16s} {v}" for k, v in ROLES.items()]
    lines += ["",
              "!! ATAC: the role name does not guarantee the representation.",
              "   atac_gas resolves to atac_gas.h5 if present, otherwise FALLS BACK",
              "   to atac.h5 - and a multiome atac.h5 usually holds peaks, not gene",
              "   activity. Check the feature names: chr1:3094772-3095489 is a peak,",
              "   a gene symbol is gene activity.",
              "   atac_peak resolves to atac_peak.h5, else peak.h5.",
              "   This matters because methods disagree (lists from the registry;",
              "   see mtb.find_methods(atac=...) / mtb.method_info(m)['atac']):",
              f"     need GENE ACTIVITY: {'/'.join(gas_methods)}",
              f"     need PEAKS:         {'/'.join(peak_methods)}",
              "   Feeding the wrong one runs to completion and returns a plausible",
              "   but WRONG embedding - no error.",
              "   Check what you actually have before trusting a cross-dataset result.",
              "   scan() flags the commonest trap in its `caveat` column: an atac_gas",
              "   role that fell back to an atac.h5 whose features are chr:start-end.",
              "",
              "MODALITY FILE FORMAT (.h5) - easiest route first:",
              "  mtb.io.to_canonical(src, dst)   converts an .h5ad and writes",
              "  everything below correctly. Prefer it over building the file by hand.",
              "  For a whole dataset in one call (every modality + labels, numbered",
              "  per batch when batch= is given):",
              "    mtb.io.export_dataset(adata, '<data_path>/MYDATA', rna='X',",
              "                          adt='obsm:protein', labels='obs:cell_type')",
              "  Both store matrix/data as float64, gzip-compressed and chunked, like",
              "  the shipped files (a 3000x2000 8%-dense matrix is ~1.5 MB on disk).",
              "",
              "  If you do build it yourself, all three datasets are required:",
              "    matrix/data      the matrix, stored FEATURES x CELLS",
              "    matrix/features  one entry per feature (row of matrix/data)",
              "    matrix/barcodes  one entry per cell    (column of matrix/data)",
              "  e.g. 2,000 genes x 5,000 cells -> matrix/data has shape (2000, 5000),",
              "  matrix/features has 2000 entries and matrix/barcodes has 5000.",
              "  Note: this is the TRANSPOSE of the scanpy/AnnData convention",
              "  (AnnData.X is cells x genes). scan() rejects a transposed file, and",
              "  a file with only matrix/data fails with a KeyError about 'features'.",
              "Labels are a single-column CSV: one header line (typically 'x'),",
              "then one cell-type label per cell; the evaluator reads the first",
              "column and skips the header.", ""]
    if category:
        lines += [f"{category}: {CATEGORIES.get(category, '(unknown category)')}", ""]
    lines += ["", "ENVIRONMENTS",
              "  Every method runs in its own conda env (they need mutually",
              "  incompatible framework versions). scan() checks two gates per row -",
              "  files_ok (the inputs are on disk, oriented and labelled) and env_ok",
              "  (that conda env exists) - and marks a method runnable only when both",
              "  pass, so a sweep never starts one that cannot finish.",
              "      multibench env doctor          # what is needed / what is missing",
              "      multibench env install --run   # build them all from lockfiles",
              "      multibench env install --methods X --packed --run   # just one",
              ""]
    lines += ["Then:  mtb.scan('MYDATA')  ->  mtb.run_all('MYDATA', '<category>', out_dir=...)"]
    return "\n".join(lines)



# --------------------------------------------------------------------------- scan
def _data_dir_usable(variant, ds_dir) -> tuple[bool, str]:
    """Whether the dataset holds what a ``data_dir`` method needs, with the reason.

    ``data_dir`` resolves to the dataset directory itself when there is no
    ``processed/`` subdir, so the path always exists; without a content check
    a ``data_dir`` method (scBridge) would look runnable on every dataset.
    Delegates to :func:`multibench.engine.resolve._check_data_dir`, which
    ``inputs_for(check=True)`` - and through it ``scan`` - calls directly;
    this wrapper is kept only as an importable alias.
    """
    return _resolve._check_data_dir(variant, ds_dir)



@functools.lru_cache(maxsize=1)
def _installed_envs() -> frozenset:
    """Conda envs present on this machine (cached; see mtb.env.doctor())."""
    try:
        return frozenset(envs.installed_envs())
    except Exception:      # never let an env probe break discovery
        return frozenset()


#: Known method x dataset incompatibilities that file/env checks cannot see:
#: the files exist and the env is installed, but the content stops the method.
#: Surfaced by scan() so a sweep does not discover them hours in.
_CAVEATS = {
    ("GLUE", "D28"): ("GLUE parses coordinates out of peak names and needs them "
                      "colon-delimited (chr1:1-200); D28's are underscore-delimited "
                      "and it IndexErrors. Use D27, or rename the peaks."),
}


def _missing_script(variant, *, method: str | None = None) -> str:
    """Why this variant's script is unreachable here, or "" when it is fine.

    A method whose script is absent must not be reported runnable: the run
    would fail minutes later with a shell error instead of here, instantly.
    Two cases are checked:

    * an entrypoint missing from a reference checkout that is present. When
      no checkout exists at all, nothing is reported: `run` and `run_all`
      fetch it on first use, and flagging every method as broken before that
      first fetch would be wrong;
    * a local helper module the entrypoint imports from its own directory
      (``variant.helpers``, e.g. MIRA's ``logger.py``) that the public
      repository does not ship - the script would ``ImportError`` at start.

    Parameters
    ----------
    variant : Variant
        The variant whose ``entrypoint`` / ``helpers`` are checked.
    method : str, keyword-only, optional
        The method id, named in the helper message.
    """
    from pathlib import Path as _P

    ep = _P(variant.entrypoint)
    repo = _P(config.DEFAULT.repo_path)
    for root in (repo, _P(config.__file__).resolve().parent.parent):
        if (root / "tools_scripts").is_dir():
            if not (root / ep).exists():
                return (f"method script {ep} is missing from the reference checkout at "
                        f"{root} - update it (git pull) or delete it and let the next "
                        f"run fetch a fresh copy")
            gone = [h for h in (getattr(variant, "helpers", None) or [])
                    if not (root / ep).parent.joinpath(h).exists()]
            if gone:
                who = f"mtb.method_info({method!r})" if method else "method_info(m)"
                return (f"method script {ep.name} imports the local module(s) {gone} from "
                        f"its own directory, which the public scMultiBench repository "
                        f"does not ship (none next to it in the checkout at {root}); "
                        f"the benchmark host runs it with a local shim - supply the "
                        f"file(s) beside {ep.name}, see {who}['setup_hint']")
            return ""
    return ""            # no checkout yet: run()/run_all() fetch one


def _caveat(method: str, dataset: str) -> str:
    return _CAVEATS.get((method, dataset), "")


def _variant_rows(category=None):
    for spec in registry.load():
        for v in spec.variants:
            cat = v.when.get("category")
            if category and cat != category:
                continue
            yield spec, v, cat, list(v.when.get("modalities", []))


#: scan() columns, in order. New columns are appended, never inserted or
#: removed, so positional readers keep working.
SCAN_COLUMNS = ["method", "category", "modalities", "env", "output_kind", "n_tunable",
                "runtime_tier", "observed_worst_sec", "caveat", "runnable", "reason",
                "files_ok", "files_reason", "env_ok", "env_reason", "needs_labels", "atac",
                "command"]

#: Default ``out_dir`` of :func:`scan`'s ``command`` column - a literal
#: placeholder so a preview needs no real directory; the lines then read
#: ``--save_path <out_dir>/Matilda_D11/``.
OUT_DIR_PLACEHOLDER = "<out_dir>"


def _env_hint(env: str, method: str, category: str | None) -> str:
    """The env_reason text: names the env, the method-specific install command
    and the category-wide alternative, and where to look."""
    alt = f" (or --category {category})" if category else ""
    return (f"conda env {env!r} is not installed - run "
            f"`multibench env install --methods {method} --packed --run`{alt}; "
            f"see mtb.env.doctor()")


def _truncate_tail(msg: str, limit: int = 500) -> str:
    # cut the middle, keep the tail: the filename sits at the end of the message
    return msg if len(msg) <= limit else msg[:100] + " ... " + msg[-(limit - 120):]


_ABS_PATH_RE = re.compile(r"(?<![\w./-])/(?:[^\s'\"\[\]{}(),:;]+/)+[^\s'\"\[\]{}(),:;]*")
_EXC_PREFIX_RE = re.compile(r"^[A-Z]\w*(?:Error|Exception|Warning): ")


def _short_reason(text: str, method: str, dataset: str, category: str | None) -> str:
    """The ``reason`` column form of a ``files_reason``: what is missing, no noise.

    ``files_reason`` keeps the verbatim exception text (``FileNotFoundError:
    UnitedNet/D11/vertical: input files not found on disk: {'atac_gas':
    '/path/to/data/D11/atac_gas.h5', ...}``) because the full path is what a
    user greps for. ``reason`` is what the scan frame, the CLI table
    and the "nothing is runnable" error show, so it drops what every row
    repeats: the exception class, the ``method/dataset/category:`` prefix and
    the absolute directory (each path becomes its basename). The env half of
    ``reason`` is untouched - it carries the copy-pasteable install command.
    """
    if not text:
        return text
    parts = []
    for part in text.split("; "):
        part = _EXC_PREFIX_RE.sub("", part)
        prefix = f"{method}/{dataset}/{category}: "
        if part.startswith(prefix):
            part = part[len(prefix):]
        part = _ABS_PATH_RE.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], part)
        parts.append(part)
    return "; ".join(parts)


# Reject a bare string where a list of ids is expected (shared with mtb.env.*).
_list_of_ids = registry.check_id_list


def _variant_consumes_atac(variant) -> bool:
    """Whether this variant takes an ATAC input (role or const filename)."""
    if "atac" in variant.modality_types:
        return True
    return any(a.const and "atac" in str(a.const) for a in variant.args)


def _command_line(method: str, category: str, inputs: dict, *, out_dir, dataset: str,
                  params: dict | None) -> str:
    """The shell line ``run`` would execute for one scan row (``shlex``-joined).

    ``(no preview: ...)`` when building it failed - a preview must never
    abort the scan.
    """
    import shlex
    try:
        # the runner itself, not the module-level ``_run`` hook the dispatch
        # tests replace: a preview must never count as a dispatch
        argv = _runner.run(method, category, inputs=inputs,
                           out_dir=Path(out_dir) / f"{method}_{dataset}",
                           params=params, dry_run=True)
        return shlex.join(argv)
    except Exception as e:  # noqa: BLE001 - a preview must never abort the scan
        return f"(no preview: {type(e).__name__}: {e})"


def scan(dataset: str, category: str | None = None, *,
         methods: list[str] | None = None,
         modalities: list[str] | None = None,
         data_path: Path | str | None = None,
         out_dir=OUT_DIR_PLACEHOLDER,
         params: dict | None = None,
         verbose: bool = True) -> pd.DataFrame:
    """Report what can run on a dataset, why the rest cannot, and each command.

    Nothing is executed. Call it first on a new dataset;
    ``run_all(dry_run=True)`` returns the same frame.

    Parameters
    ----------
    dataset : str
        Dataset folder name under ``data_path`` (not a path).
    category : str | None
        Integration category to scan; ``None`` = all four.
    methods : list[str] | None
        Method ids to include, as a list; ``None`` = every method.
    modalities : list[str] | None
        Modality tokens of one combination, e.g. ``["rna", "adt"]``; ``None`` =
        every combination.
    data_path : Path | str | None
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.DEFAULT.data_path``.
    out_dir : path | str
        Root the ``command`` lines write under; default the literal
        placeholder ``'<out_dir>'``. Pass the real one for ready-to-run lines.
    params : dict | None
        ``{method: {key: value}}`` hyperparameter overrides, rendered into
        ``command`` and checked against the keys each method accepts.
    verbose : bool
        Print one line ``[scan] files OK for k/n method rows; e/n envs
        installed``.

    Returns
    -------
    pandas.DataFrame
        One row per (method, category, modalities) variant, runnable rows
        first. Read ``df[["method", "modalities", "runnable", "reason"]]``;
        all 18 columns are listed in Notes.

    Raises
    ------
    FileNotFoundError
        ``<data_path>/<dataset>`` does not exist; the message lists the folders present.
    ValueError
        Unknown ``category``, or no variant of ``methods`` exists under ``category``.
    KeyError
        Unknown id in ``methods`` or ``params``, or a ``params`` key no variant accepts.
    TypeError
        ``methods`` or ``modalities`` given as a bare string.

    Warns
    -----
    UserWarning
        ``dataset`` matches a folder only up to letter case, or ``modalities``
        drops directory-input methods.

    Examples
    --------
    >>> import multibench as mtb
    >>> df = mtb.scan("D11", "vertical")
    >>> df[["method", "modalities", "runnable", "reason"]]
    >>> df.loc[~df.runnable, ["method", "files_reason", "env_reason"]]   # what blocks the rest
    >>> print(df.loc[df.files_ok, "command"].iloc[0])                  # a ready-to-run shell line
    >>> mtb.scan("MYCITE", "vertical", data_path="/path/to/data", out_dir="out/")

    Notes
    -----
    **Column reference.** The full frame is 18 columns wide
    (``SCAN_COLUMNS``):

    ```text
    method              registry id
    category            integration category of the variant
    modalities          '+'-joined string ("rna+adt"); "(data_dir)" for a
                        directory-fed variant
    env                 the conda env the method runs in
    output_kind         embedding / graph
    n_tunable           number of command-line hyperparameters
    runtime_tier        fast / medium / slow / very_slow / unknown
    observed_worst_sec  the slowest observed run, seconds (None = unmeasured)
    caveat              known content trap for this method x dataset, or ""
    runnable            files_ok & env_ok
    reason              short form of the non-empty reasons, "; "-joined
    files_ok            the inputs resolve, are oriented and labelled
    files_reason        verbatim file-gate text, full paths
    env_ok              the env exists (and a GPU, when the script needs one)
    env_reason          verbatim env-gate text with the install command
    needs_labels        this variant demands a label file as an input
    atac                ATAC representation the method expects: 'peak' /
                        'gene_activity'; None when the variant takes no ATAC
    command             the shell line the variant would run; "" if the
                        inputs do not resolve
    ```

    **Two gates.** Every row carries two independent gates, each a flag plus
    a reason, and ``runnable = files_ok & env_ok``:

    - ``files_ok`` / ``files_reason`` - the method's script is present, the
      input files resolve on disk and are oriented features x cells, every
      label CSV has one row per cell of the modality it labels, and a
      ``data_dir`` method (scBridge) finds the files it names.
    - ``env_ok`` / ``env_reason`` - the method's conda env exists on this
      machine; the reason names the env and the one-method install command
      (``multibench env install --methods X --packed --run``).
    - ``env_ok`` on a GPU-only method - when the upstream script calls CUDA
      unconditionally (``method_info(m)['requires_gpu']``), ``env_ok`` also
      needs an NVIDIA GPU (``mtb.env.host_has_gpu()``); without one,
      ``env_reason`` carries the sentence ``run`` would raise (``"<method>
      needs an NVIDIA GPU: the upstream script calls CUDA unconditionally
      (<file>:<line>) ..."``).

    The file gate always runs, whether or not any conda env is installed, so
    a laptop without envs still tells you whether your layout is right.

    **Reason columns.** ``reason`` joins the non-empty reasons with ``"; "``
    and is empty iff the row is runnable. It is the short form: the file half
    drops the exception class, the ``method/dataset/category:`` prefix and
    the absolute directory (``input files not found on disk: {'atac':
    'atac.h5'}. Available files in D11: [...]``). ``files_reason`` /
    ``env_reason`` keep the verbatim text with full paths; read them for a
    row you are debugging.

    **The command column.**

    - ``command`` is ``run(..., dry_run=True)``, ``shlex``-joined, writing
      under ``<out_dir>/<method>_<dataset>/`` exactly like ``run_all`` - the
      literal ``'<out_dir>'`` placeholder unless ``out_dir`` is given.
    - Paths are absolute: a relative ``out_dir``, the placeholder included,
      is resolved against the working directory.
    - ``params`` are merged in the way ``run_all(params=)`` merges them.
    - On a GPU-less host it already carries each method's ``cpu_params``
      (the flags that turn CUDA off where a switch exists).
    - A row blocked only by ``env_ok`` still shows its command - the line to
      put in a job script once the env is built.

    **The modalities column.** ``modalities`` is a ``+``-joined string here
    (``"rna+adt"``); ``run_all`` / ``inputs_for`` take a list
    (``["rna", "adt"]``), so split on ``"+"``. The sentinel ``"(data_dir)"``
    marks a method that consumes a whole directory rather than named modality
    files (scBridge); for it, pass no ``modalities`` at all.

    **Sizing a sweep.** ``runtime_tier`` / ``observed_worst_sec`` (see
    ``method_info(m)['runtime']``) let you size a sweep before launching it;
    ``caveat`` carries known content traps (e.g. an ``atac_gas`` role that
    fell back to a peak matrix).

    **Selection and input checks.**

    - ``category`` - a typo raises ``ValueError`` listing the four valid
      values.
    - ``methods`` - an unknown id raises ``KeyError`` with a did-you-mean
      hint; blocked rows of the selected methods are kept, with their reason.
      A selection with no variant under ``category`` (a known id with no
      diagonal variant, say) raises ``ValueError`` - never a silently empty
      frame.
    - ``modalities`` - an exact selector: ``protein`` is accepted for
      ``adt``, ``modalities=[]`` selects exactly the directory-input
      variants, and any non-empty list excludes them (a ``UserWarning`` names
      them and says ``modalities=[]`` selects them).
    - ``params`` - a key no variant of that method accepts raises
      ``KeyError`` naming the accepted keys, so a typo is caught here rather
      than hours into a sweep.
    - ``dataset`` - a spelling that differs from the folder only in case
      (``'d52'`` on macOS) is replaced by the on-disk spelling, with a
      ``UserWarning``.

    **Choosing a category.** A CITE-seq folder (``rna.h5`` + ``adt.h5`` +
    ``cty.csv``) is ``vertical`` with modalities ``["rna", "adt"]``; RNA and
    ATAC from different cells is ``diagonal``. See ``mtb.list_categories``
    and ``mtb.describe_layout``.

    **Environments and CLI.** Each method runs in its own conda environment
    (they need mutually incompatible framework versions). List them with
    ``multibench env doctor``; build them with
    ``multibench env install --run``. ``multibench scan`` prints a compact
    view by default; ``--columns all`` adds the rest, including ``command``.

    See Also
    --------
    mtb.run_all : run the runnable rows, with metrics; ``dry_run=True`` returns this frame.

    mtb.describe_layout : how to lay out a dataset folder so ``files_ok`` passes.

    mtb.env.doctor : the env gate on its own, per env.

    mtb.inputs_for : the ``{role: path}`` resolution behind ``files_ok``.
    """
    registry.check_category(category)       # raises with the valid list on a typo
    _list_of_ids(methods, "methods")        # TypeError before iterating characters
    _list_of_ids(modalities, "modalities")
    if methods is not None:
        methods = [registry.check_method(m) for m in methods]   # did-you-mean KeyError
    params = params or {}
    for _m in params:                       # KeyError (did-you-mean) before any I/O
        registry.check_method(_m)
    want_mods = None
    if modalities is not None:
        want_mods = "+".join(registry.normalize_modalities(modalities)) or "(data_dir)"
    base = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    dataset = _resolve.canonical_dataset(base, dataset)
    ds_dir = base / dataset
    if not ds_dir.is_dir():
        dirs = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
        raise FileNotFoundError(
            f"dataset folder '{ds_dir}' does not exist; folders present under {base}: "
            f"{dirs}. dataset= is the folder name and data_path= the folder that "
            f"contains it (see mtb.describe_layout())")
    installed = _installed_envs()
    rows = []
    dropped_dirs: list[str] = []
    for spec, v, cat, mods in _variant_rows(category):
        if methods is not None and spec.id not in methods:
            continue
        mod_str = "+".join(mods) or "(data_dir)"
        if want_mods is not None and mod_str != want_mods:
            if mod_str == "(data_dir)" and spec.id not in dropped_dirs:
                dropped_dirs.append(spec.id)
            continue
        rt = _runtimes().get(spec.id, {})
        rec = {"method": spec.id, "category": cat, "modalities": mod_str,
               "env": envs.group_for(spec.id), "output_kind": v.output.kind,
               "n_tunable": len(v.tunable),
               "runtime_tier": rt.get("tier", "unknown"),
               "observed_worst_sec": rt.get("worst_sec"),
               "caveat": _caveat(spec.id, dataset), "runnable": False, "reason": "",
               "files_ok": True, "files_reason": "", "env_ok": True, "env_reason": "",
               "needs_labels": bool(v.needs_labels),
               "atac": spec.atac if _variant_consumes_atac(v) else None,
               "command": ""}
        # --- gate 1: files. Runs whether or not any env is installed. -------
        # Both halves are checked (the method's script and the dataset's files)
        # so a missing script does not hide a layout problem or vice versa.
        file_problems = []
        why_script = _missing_script(v, method=spec.id)
        if why_script:
            file_problems.append(why_script)
        got = None
        try:
            got = _resolve.inputs_for(dataset, cat, spec.id, modalities=mods or None,
                                      data_path=data_path, check=True)
            extra = _resolve._preflight_caveats(got, atac=spec.atac)
            if extra:
                rec["caveat"] = "; ".join(x for x in [rec["caveat"], *extra] if x)
        except Exception as e:  # missing files / no variant / bad layout
            file_problems.append(_truncate_tail(f"{type(e).__name__}: {e}"))
        if file_problems:
            rec["files_ok"], rec["files_reason"] = False, "; ".join(file_problems)
        # --- gate 2: env. -----------------------------------------------------
        if rec["env"] and rec["env"] not in installed:
            rec["env_ok"] = False
            rec["env_reason"] = _env_hint(rec["env"], spec.id, cat)
        # ... and the host: a script that calls CUDA unconditionally cannot
        # finish without an NVIDIA GPU, however complete the env - the same
        # sentence run() raises as OSError, so the sweep never starts it.
        if spec.requires_gpu and not envs.host_has_gpu():
            rec["env_ok"] = False
            rec["env_reason"] = "; ".join(
                r for r in (rec["env_reason"], spec.requires_gpu_reason) if r)
        rec["runnable"] = bool(rec["files_ok"] and rec["env_ok"])
        rec["reason"] = "; ".join(
            r for r in (_short_reason(rec["files_reason"], spec.id, dataset, cat),
                        rec["env_reason"]) if r)
        # --- the command line: only when the files resolved (something to
        # hand the script); an env-blocked row still gets one
        if rec["files_ok"] and got is not None:
            rec["command"] = _command_line(spec.id, cat, got, out_dir=out_dir,
                                           dataset=dataset, params=params.get(spec.id))
        rows.append(rec)
    df = pd.DataFrame(rows, columns=SCAN_COLUMNS)
    df = df.sort_values(["runnable", "category", "method"],
                        ascending=[False, True, True]).reset_index(drop=True)
    if df.empty and methods is not None:
        # no variant of the requested methods exists under this category: a
        # request problem (Matilda is not a cross method), reported as such
        # rather than as a silently empty frame
        raise ValueError(
            f"no {category!r} variant matches dataset={dataset!r} methods={methods} "
            f"modalities={modalities}; see mtb.method_info(m)['supports'] and "
            f"mtb.scan({dataset!r})")
    if params:
        _check_param_keys(df, params)       # a typo'd key must not start a sweep
    if dropped_dirs:
        # the selector is exact by design (a sweep must not silently grow);
        # say what it excluded rather than dropping the rows in silence
        warnings.warn(
            f"scan: modalities={list(modalities)} excludes {len(dropped_dirs)} "
            f"directory-input method(s) ({', '.join(dropped_dirs)}: they take a "
            f"data_dir, shown as '(data_dir)', not modality files); pass "
            f"modalities=[] to select them, or no modalities for every variant",
            UserWarning, stacklevel=2)
    if verbose:
        n = len(df)
        print(f"[scan] files OK for {int(df['files_ok'].sum())}/{n} method rows; "
              f"{int(df['env_ok'].sum())}/{n} envs installed", flush=True)
    return df


# ------------------------------------------------------------------- label order
def _read_cty(path):
    """Read a label CSV with the package's single reader (``eval.io.read_labels``)."""
    return _eio.read_labels(path)


_NOT_ARMED = object()


def _arm_deadline(seconds):
    """Start a SIGALRM deadline covering an entire per-method step.

    The step includes reading the output back and computing the metrics, which
    can take far longer than the method itself; an alarm around the dispatch
    call alone would leave them unbounded.

    Returns the previous handler, or ``_NOT_ARMED`` when no deadline was asked
    for. ``None`` is not usable as that sentinel: ``signal.signal`` returns
    None when the previous handler was not installed from Python.
    """
    if not seconds:
        return _NOT_ARMED
    import math
    import signal
    import threading
    import warnings

    if threading.current_thread() is not threading.main_thread():
        # signal.signal raises off the main thread; warn rather than drop the
        # deadline silently
        warnings.warn("timeout= is unavailable off the main thread; "
                      "running without a deadline")
        return _NOT_ARMED

    def _fire(signum, frame):
        raise TimeoutError(f"exceeded timeout of {seconds}s")

    prev = signal.signal(signal.SIGALRM, _fire)
    # ceil, not int(): truncating timeout=0.5 gives alarm(0), which sets no alarm
    signal.alarm(max(1, math.ceil(seconds)))
    return prev


def _disarm_deadline(prev):
    if prev is _NOT_ARMED:
        return
    import signal
    signal.alarm(0)
    signal.signal(signal.SIGALRM, prev)



def _label_candidates(dataset, n, data_path=None):
    """Every label ordering whose length matches ``n`` (length alone cannot pick one)."""
    base = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    ctys = {}
    for p in sorted((base / dataset).glob("*cty*.csv")):
        if "scjoint" in p.name.lower():
            continue
        try:
            ctys[p.name] = _read_cty(p)
        except Exception:
            pass
    cands = [([k], v) for k, v in ctys.items()]
    rna = [k for k in ctys if k.startswith("rna_cty")]
    ata = [k for k in ctys if k.startswith(("atac_cty", "peak_cty"))]
    for a, b in itertools.product(rna, ata):
        cands += [([a, b], np.concatenate([ctys[a], ctys[b]])),
                  ([b, a], np.concatenate([ctys[b], ctys[a]]))]
    nums = sorted(k for k in ctys if k.startswith("cty") and any(c.isdigit() for c in k))
    if len(nums) > 1:
        # Only orderings whose total length equals n survive the filter below,
        # and length is order-independent: pick length-matching subsets first
        # (combinations) and permute only those, rather than concatenating every
        # permutation of every subset (factorial in files).
        sizes = {k: len(ctys[k]) for k in nums}
        for r in range(len(nums), 1, -1):
            for combo in itertools.combinations(nums, r):
                if sum(sizes[k] for k in combo) != n:
                    continue
                for perm in itertools.permutations(combo):
                    cands.append((list(perm),
                                  np.concatenate([ctys[k] for k in perm])))
    out, seen = [], set()
    for names, lab in cands:
        if len(lab) != n or tuple(names) in seen:
            continue
        seen.add(tuple(names))
        # batch = which label file each cell came from. For multi-batch designs
        # (mosaic / cross) that is the batch, so batch-correction metrics are
        # computable without asking the caller for anything extra.
        bat = np.concatenate([np.full(len(ctys[k]), i + 1) for i, k in enumerate(names)])
        out.append((list(names), lab, bat))
    return out


#: Below this ARI the winning ordering is itself at chance, so the confidence
#: ratio would compare two noise values and mean nothing.
_CHANCE_ARI = 0.05


def _order_confidence(cands) -> float | None:
    """How clearly the winning label order beat the alternatives, on a 0-1 scale.

    ``(best - runner_up) / best``. A ratio, not a difference: the runner-up
    sits near chance (ARI ~ 0), so a difference is bounded above by the ARI itself
    and a method scoring 0.3 could never look clearly separated. Dividing by the
    winner makes an unambiguous order read ~1.0 whether the method scored 0.9 or
    0.2.
    """
    if not cands or len(cands) < 2:
        return None
    best, second = cands[0]["ARI"], cands[1]["ARI"]
    # A ratio of two chance-level values is noise (0.0004 vs 0.0002 would read
    # 0.5): report None so the column is not misread when no ordering worked.
    if best < _CHANCE_ARI:
        return None
    return round(max(0.0, (best - second) / best), 4)


def _evaluate_best_order(emb, category, cands, *, batch=None, metrics=None):
    """Score each candidate label order, keep the best, return the full spread.

    ``batch`` (optional, one entry per cell in embedding order) replaces the
    file-of-origin batch vector carried by each candidate - ``run_all(batch=)``
    / ``BatchResult.rescore(batch=)``. ``metrics`` (a family token or a list
    of metric codes, ``evaluate(metrics=)``) restricts the set the winner is
    scored on; ``None`` = the family the batch structure implies (screening
    still needs ARI only).
    """
    def _full(lab, bat, clustering=None):
        # several distinct source files (or a user batch with >1 level) => a
        # real batch structure, so ask for both metric families; else clustering.
        if batch is not None:
            bat = np.asarray(batch)
        grp = "all" if len(set(np.asarray(bat).tolist())) > 1 else "clustering"
        return _evaluate(emb, category=category, labels=lab,
                                verbose=False, batch=(bat if grp == "all" else None),
                                clustering=clustering,
                                metrics=(grp if metrics is None else metrics))

    if len(cands) == 1:
        # nothing to disambiguate - do not pay for a screening pass
        names, lab, bat = cands[0]
        try:
            val = _full(lab, bat)
        except Exception as e:  # noqa: BLE001 - surfaced on the record, not swallowed
            # return the error: an empty result would make a bad metrics=/batch=
            # combination look like 'no label file matched'
            return names, None, [{"order": names,
                                  "error": f"{type(e).__name__}: {str(e)[:300]}"}]
        return names, val, [{"order": names,
                             "ARI": round(float(val["Value"]["ARI"]), 4)}]

    # Ranking orderings needs only ARI, and the Leiden sweep behind ARI depends
    # on the embedding alone, not on the label vector. So sweep once, reuse it
    # for every candidate, and compute the full metric set once, on the winner;
    # doing either per candidate multiplies the cost by the number of orderings.
    import scib.metrics as _me

    try:
        sweep_adata, sweep_keys = _escib.leiden_sweep(emb)
    except Exception:
        sweep_adata = None

    scored = []
    for names, lab, bat in cands:
        try:
            if sweep_adata is None:      # fall back to a self-contained screen
                val = _evaluate(emb, category=category, verbose=False,
                                       labels=lab, metrics=["ARI"])
                scored.append((float(val["Value"]["ARI"]), names, lab, bat, None))
                continue
            sweep_adata.obs["celltype"] = pd.Categorical(
                np.asarray(lab).astype(str))
            best_key, best_nmi = None, -1.0
            for k in sweep_keys:
                s = float(_me.nmi(sweep_adata, cluster_key=k, label_key="celltype"))
                if s > best_nmi:
                    best_nmi, best_key = s, k
            ari = float(_me.ari(sweep_adata, cluster_key=best_key,
                                label_key="celltype"))
            scored.append((ari, names, lab, bat,
                           np.asarray(sweep_adata.obs[best_key].values)))
        except Exception as e:  # noqa: BLE001 - keep screening other orders
            _last_err = f"{type(e).__name__}: {e}"
            continue
    if not scored:
        raise RuntimeError(
            "no label ordering could be screened"
            + (f"; last error: {_last_err}" if '_last_err' in dir() else ""))
    scored.sort(key=lambda r: -r[0])
    ari, names, lab, bat, clus = scored[0]
    try:
        # hand the winning clustering to the full evaluation so it does not
        # repeat the sweep
        val = _full(lab, bat, clustering=clus)
        if clus is not None and val.attrs.get("clustering") == "user":
            # the clusters came from the screening sweep, not from the user
            val.attrs.update(clustering="sweep",
                             leiden_flavor=sweep_adata.uns.get("leiden_flavor"))
    except Exception as e:
        raise RuntimeError(
            f"evaluation failed for the winning label order {names}: "
            f"{type(e).__name__}: {e}") from e
    spread = [{"order": n, "ARI": round(a, 4)} for a, n, _, _, _ in scored]
    return names, val, spread


# ------------------------------------------------------------------------ results
def _with_label_order_note(sm: "pd.DataFrame") -> "pd.DataFrame":
    """Attach ``label_order_note``, which says why ``label_order_confidence`` is
    blank (used by ``BatchResult.summary`` and ``save()``)."""
    if "label_order_confidence" in sm:
        scored = sm["status"].astype(str).str.startswith("CHAIN_OK")
        blank = scored & sm["label_order_confidence"].isna()
        ari = pd.to_numeric(sm.get("ARI"), errors="coerce")
        why = pd.Series([None] * len(sm), index=sm.index, dtype=object)
        why[blank & (ari < 0.05)] = "winner at chance"
        why[blank & ~(ari < 0.05)] = "single ordering"
        why[~scored] = "not scored"
        sm["label_order_note"] = why
    return sm


class BatchResult:
    """Outcome of ``mtb.run_all`` - a summary table, a tidy frame and a figure.

    Built by ``mtb.run_all`` and ``mtb.load_batch``, not by hand. The
    per-method records are kept, so a finished sweep can be re-scored or
    re-plotted without re-running any method.

    Parameters
    ----------
    records : list[dict]
        One record per method run, as ``run_all`` builds them.
    dataset : str
        Dataset folder name the sweep ran on.
    category : str
        Integration category the sweep ran under.
    out_dir : path | None
        Where the sweep wrote its outputs; ``save()`` defaults to it.

    Attributes
    ----------
    records : list[dict]
        The raw per-method records (the same list ``results`` returns).
    dataset : str
        Dataset folder name.
    category : str
        Integration category.
    out_dir : path | None
        The sweep's output root, or ``None`` for an in-memory result.
    summary : pandas.DataFrame
        One row per method with its status, timing and metrics (property).
    long : pandas.DataFrame
        Tidy ``metric, value, method, dataset, category, ...`` frame for
        plotting (property).
    results : list[dict]
        The raw records, including every label ordering tried (property).
    failures : pandas.DataFrame
        Methods that failed, timed out or could not be scored:
        ``method, status, error`` (property).

    Examples
    --------
    >>> import multibench as mtb
    >>> res = mtb.run_all("D11", "vertical", out_dir="out/", timeout=3600)
    >>> res.summary[["method", "status", "ARI", "NMI"]]
    >>> res.failures                              # empty frame when all went well
    >>> fig = res.plot(metrics=["ARI", "NMI", "ASW"])
    >>> res.rescore(metrics="clustering").save("out/rescored")   # nothing re-run

    Notes
    -----
    **Size and repr.** ``len(res)`` is the number of method records. The
    repr counts the methods with metrics, those that ran but could not be
    scored, and the failures.

    See Also
    --------
    mtb.run_all : produces one.

    mtb.load_batch : reloads one from ``save()``'s folder.

    mtb.plot.bubble : the figure ``plot`` draws from ``long``.
    """

    def __init__(self, records, dataset, category, out_dir=None):
        self.records = records
        self.dataset = dataset
        self.category = category
        self.out_dir = out_dir

    @property
    def summary(self) -> pd.DataFrame:
        """One row per method: status, timing, shape, label matching and every metric.

        Returns
        -------
        pandas.DataFrame
            One row per method, sorted by ``method``. Read ``method``,
            ``status`` and the metric columns (``ARI``, ``NMI`` ...); all
            columns are listed in Notes.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> res.summary[["method", "status", "ARI", "label_order_confidence"]]
        >>> res.summary.query("status == 'CHAIN_OK'").sort_values("ARI", ascending=False)

        Notes
        -----
        **Column reference.** When nothing ran the frame is empty, with the
        first ten columns below:

        ```text
        method                  registry id
        status                  outcome; see "Status values"
        run_sec                 wall-clock seconds of the method run
        output_kind             embedding / graph
        emb_shape               [cells, dims] of the embedding; None without one
        n_tunable               number of command-line hyperparameters
        label_order             label file(s) the metrics used, in order
        label_order_confidence  how clearly that ordering won, 0-1
        batch_source            'file_of_origin' / 'user' / None
        n_batches               distinct batch values used (1 = none)
        ARI, NMI, ASW, ...      one column per metric
        label_order_note        why label_order_confidence is blank
        ```

        **Status values.**

        - ``CHAIN_OK`` - ran and scored.
        - ``CHAIN_OK_GRAPH_METHOD`` - a graph method, scored via a secondary
          embedding.
        - ``RUN_OK_NO_EMBEDDING`` - ran, but the method emits only a graph,
          so clustering metrics do not apply.
        - ``RUN_OK_EVAL_FAILED`` - the method ran and produced an embedding,
          but scoring it failed; see ``error`` in ``failures``.
        - ``RUN_OK_NO_LABEL_MATCH`` - ran, but no label file matches the
          embedding's cell count.
        - ``RUN_OK`` - ran with ``evaluate=False``.
        - ``TIMEOUT`` - exceeded ``run_all(timeout=...)``.
        - ``FAIL`` - the method itself errored; see ``error`` in ``failures``.

        ``FAIL``, ``TIMEOUT``, ``RUN_OK_EVAL_FAILED`` and
        ``RUN_OK_NO_LABEL_MATCH`` also appear in ``failures``.

        **Graph methods.** Two methods can both be ``output_kind=graph`` and
        still end differently: scMoMaT also writes a UMAP embedding among its
        ``extra_outputs``, so it is scored through that
        (``CHAIN_OK_GRAPH_METHOD``); Seurat_WNN writes only a neighbour graph,
        so there is nothing to score (``RUN_OK_NO_EMBEDDING``) and its
        ``emb_shape`` is ``None``.

        **Batch columns.** ``batch_source`` / ``n_batches`` say which batch
        vector the batch metrics (ASW_batch, GC, iLISI ...) were computed
        against, and how many distinct values it has (1 = none):

        - ``'file_of_origin'`` - each cell's label file (``cty1.csv`` -> 1,
          ``cty2.csv`` -> 2 ...), the rule for multi-batch datasets.
        - ``'user'`` - the vector passed as ``run_all(batch=)`` /
          ``rescore(batch=)``.
        - ``None`` - a single label file, so no batch structure and
          clustering metrics only.

        **Label order.** ``label_order`` is which label file(s), in which
        order, the metrics were computed against (e.g.
        ``rna_cty.csv+atac_cty.csv``). For unpaired/diagonal data the
        embedding holds two disjoint cell sets stacked in a method-specific
        order, so this is the difference between a meaningful ARI and a
        meaningless one.

        **Label-order confidence.** ``label_order_confidence`` is
        ``(best - runner_up) / best`` over the candidate orderings' ARI, on a
        0-1 scale. Near 1.0 - every alternative ordering scored near chance,
        so the correspondence is unambiguous and the metrics can be read
        normally. Below ~0.5 - two orderings explained the embedding
        comparably well, which should not happen for a correct one; treat
        that row with suspicion.

        **Why a ratio.** The runner-up sits near chance, so a difference is
        bounded above by the ARI itself and a method scoring 0.3 could never
        look well-separated.

        **Optimistic bias.** When more than one ordering is possible the
        reported metrics are those of the ordering with the highest ARI, so
        they carry a small optimistic bias; ``label_order_confidence`` shows
        whether the choice was clear-cut.

        **Blank confidence.** The column stays numeric, so ``> 0.5`` and
        ``.isna()`` behave. It is ``None`` in three cases, named by
        ``label_order_note``:

        - ``"single ordering"`` - only one ordering was possible (normal for a
          paired/vertical dataset with a single ``cty.csv``).
        - ``"winner at chance"`` - the winning ordering itself scored
          ARI < 0.05, so the ratio would compare two noise values.
        - ``"not scored"`` - the row has no metrics.

        See Also
        --------
        BatchResult.failures : the rows whose status means something went wrong.

        BatchResult.results : the raw records with every ordering tried.
        """
        rows = []
        for r in self.records:
            cands = r.get("label_order_candidates") or []
            rows.append({k: r.get(k) for k in
                         ("method", "status", "run_sec", "output_kind", "emb_shape", "n_tunable")}
                        | {"label_order": "+".join(r.get("labels_used") or []) or None,
                           "label_order_confidence": _order_confidence(cands),
                           "batch_source": r.get("batch_source"),
                           "n_batches": r.get("n_batches")}
                        | {m: v for m, v in (r.get("metrics") or {}).items()})
        if not rows:      # nothing ran (e.g. no method was runnable on this dataset)
            return pd.DataFrame(columns=["method", "status", "run_sec", "output_kind",
                                         "emb_shape", "n_tunable", "label_order",
                                         "label_order_confidence", "batch_source",
                                         "n_batches"])
        return _with_label_order_note(
            pd.DataFrame(rows).sort_values("method").reset_index(drop=True))

    @property
    def long(self) -> pd.DataFrame:
        """Tidy frame (``metric, value, method, dataset, category``) for plotting.

        This is what ``plot`` and ``mtb.plot.bubble`` consume.

        Returns
        -------
        pandas.DataFrame
            Columns ``metric, value, method, dataset, category, clustering,
            source``; empty, with those columns, when no method produced
            metrics.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> mtb.plot.bubble(res.long, metrics=["ARI", "NMI"])
        >>> res.long.pivot_table(index="method", columns="metric", values="value")

        Notes
        -----
        **Source of the rows.** Each record contributes the unrounded frame
        ``run_all`` attached (or ``long.csv`` via ``mtb.load_batch``) when
        present, otherwise its ``metrics`` dict - so a result built or
        reloaded without ``long.csv`` still plots.

        See Also
        --------
        BatchResult.plot : draws the bubble figure from this frame.

        mtb.to_long : the wide -> tidy conversion used for the metrics dict.
        """
        cols = ["metric", "value", "method", "dataset", "category", "clustering", "source"]
        frames = []
        for r in self.records:
            if r.get("_long") is not None:
                frames.append(r["_long"])
            elif r.get("metrics"):
                # rebuild through to_long so the derived frame carries the same
                # seven columns (clustering/source) as an attached one
                wide = pd.DataFrame({"Value": list(r["metrics"].values())},
                                    index=list(r["metrics"]))
                frames.append(_to_long(wide, method=r.get("method"),
                                       dataset=r.get("dataset", self.dataset),
                                       category=r.get("category", self.category)))
        if not frames:
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True)

    @property
    def results(self) -> list:
        """The raw per-method records: status, out_dir, metrics and the orderings tried.

        Keeps a long sweep's outputs addressable, so you can re-score or
        re-plot without re-running the methods.

        Returns
        -------
        list[dict]
            One dict per method. Read ``method``, ``status``, ``out_dir`` and
            ``metrics`` first; the other keys are listed in Notes.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> [r["out_dir"] for r in res.results]
        >>> res.results[0].get("label_order_candidates")    # None with a single ordering

        Notes
        -----
        **Record keys.** A key is present only when it applies:

        ```text
        method, category, dataset   what ran, and on what
        modalities                  the variant's modality roles
        status                      outcome (see BatchResult.summary)
        out_dir                     the method's output folder
        metrics                     {metric: value}, rounded to 4 places
        params_used                 the hyperparameter overrides passed
        run_sec, emb_shape          timing and embedding shape
        labels_used                 the label file(s) behind the metrics
        label_order_candidates      every label ordering tried, with its ARI
        batch_source, n_batches     the batch vector the batch metrics used
        error, traceback, note      why a method failed or was not scored
        reused                      True when skip_existing reused the output
        env, output_kind, n_tunable the scan row the method ran from
        data_path, multibench_version, started_at   provenance of the run
        _long                       internal tidy frame; read BatchResult.long instead
        ```

        **Label-order evidence.** ``label_order_candidates`` holds every
        ordering tried and the ARI each achieved - the evidence behind
        ``summary``'s ``label_order_confidence``. It is present only when more
        than one ordering was possible.

        See Also
        --------
        BatchResult.summary : the same records as a table.
        """
        return self.records

    @property
    def failures(self) -> pd.DataFrame:
        """Methods that failed, timed out or could not be scored.

        ``run_all`` records failures instead of raising, so always check this -
        a sweep can finish with several methods having failed.

        Returns
        -------
        pandas.DataFrame
            Columns ``method, status, error``; empty when nothing failed.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> res.failures
        >>> assert res.failures.empty, res.failures.to_string()

        Notes
        -----
        **Statuses listed.**

        - ``FAIL`` and ``TIMEOUT``.
        - ``RUN_OK_EVAL_FAILED`` - the embedding exists, scoring it failed.
        - ``RUN_OK_NO_LABEL_MATCH`` - ran, but no label file matched the
          output's cell count, so nothing could be scored; usually a
          data-layout problem worth fixing.

        **Not listed.** ``RUN_OK_NO_EMBEDDING``: those methods ran correctly and
        emit a graph instead of an embedding, so there is nothing for
        clustering metrics to score. See ``summary`` for them.

        See Also
        --------
        BatchResult.summary : every method, including the ones that ran but could not be scored.
        """
        # RUN_OK_NO_EMBEDDING is excluded: the method ran; listing it here would
        # send people hunting for a bug that does not exist.
        bad = [r for r in self.records
               if str(r.get("status", "")).startswith(("FAIL", "TIMEOUT"))
               or r.get("status") in ("RUN_OK_EVAL_FAILED",
                                      "RUN_OK_NO_LABEL_MATCH")]
        if not bad:
            return pd.DataFrame(columns=["method", "status", "error"])
        return pd.DataFrame([{k: r.get(k) for k in ("method", "status", "error")} for r in bad])

    def plot(self, **kw):
        """Bubble figure of every method that produced metrics.

        Rows are methods (best first) and columns metrics; bubble size encodes
        the method's rank (rank 1 is the largest), colour the value (darker is
        higher).

        Parameters
        ----------
        **kw
            Keyword arguments of ``mtb.plot.bubble``: ``metrics=``,
            ``methods=``, ``order=``, ``title=``, ``cmap=``, ``save=`` ...

        Returns
        -------
        matplotlib.figure.Figure
            Save it with ``fig.savefig("out.png")``.

        Raises
        ------
        ValueError
            Nothing was scored (see ``failures``), or an unknown name in ``order=`` / ``methods=``.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> fig = res.plot()
        >>> fig = res.plot(metrics=["ARI", "NMI", "ASW"], title="D11 vertical")
        >>> fig.savefig("D11_vertical.png", dpi=200)

        Notes
        -----
        **Keywords.** ``metrics=`` sets the column order, ``methods=`` a
        subset, ``order=`` the row order (unlisted methods follow best-first;
        unknown names raise ``ValueError``). There is no default title: pass
        ``title=`` when one is wanted.

        **Reading the figure.** Size and colour are both relative to the
        methods in this figure. Read it next to ``summary`` - with few methods
        a small absolute gap still spans the whole colour scale.

        See Also
        --------
        mtb.plot.bubble : the underlying function and its full keyword list.

        BatchResult.long : the frame handed to it.
        """
        from . import plot as _plot
        lng = self.long
        if lng.empty:
            raise ValueError("no method produced metrics; see .failures / .summary")
        # No default title: a dataset-name banner adds nothing a caption cannot
        # say and crowds the page layout.
        return _plot.bubble(lng, **kw)

    def rescore(self, *, batch=None, labels=None, metrics=None,
                verbose: bool = False) -> "BatchResult":
        """Re-evaluate the stored outputs with different labels / batch / metrics.

        Nothing is re-run: each record's embedding is read back from its
        ``out_dir`` and scored again, so an overnight sweep can be re-scored
        in minutes.

        Parameters
        ----------
        batch : array-like | Series | path | None
            One batch id per cell, in embedding row order (array, Series or CSV
            path); ``None`` = each cell's source label file, or none with
            ``labels=``.
        labels : array-like | Series | path | None
            One cell-type label per cell, in embedding row order (same forms);
            ``None`` = search the dataset's label files again.
        metrics : str | list[str] | None
            Metric family (``"clustering"``, ``"batch"``, ``"all"``) or metric
            codes; ``None`` = every metric the batch structure allows.
        verbose : bool
            Print one line per method.

        Returns
        -------
        BatchResult
            A new result; this one is untouched. Call ``.save(out_dir)`` on it
            to persist it.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> res.rescore(metrics=["ARI", "NMI"]).summary
        >>> res.rescore(batch="data/D11/donor.csv").summary[["method", "batch_source", "iLISI"]]
        >>> res.rescore(labels=my_labels).save("out/rescored")

        Notes
        -----
        **Typical uses.** Re-score with the batch vector the dataset really
        has instead of the file-of-origin rule, with your own labels, or with
        a different metric selection.

        **Arguments.** A ``batch`` CSV path is read like a label file, and the
        vector is recorded as ``batch_source='user'``. With ``labels=None``
        the label-order search runs again (``label_order`` /
        ``label_order_confidence`` are refilled); given labels read
        ``(user labels)`` in ``label_order``. ``metrics`` is handed to
        ``evaluate(metrics=)``.

        **Labels without batch.** Given ``labels`` and no ``batch``, every
        cell is in one batch (``batch_source`` ``None``, ``n_batches`` 1), so
        only clustering metrics are computed; pass ``batch`` as well to get
        the batch metrics.

        **Record status.** A method that emits no embedding (graph-only) is
        marked ``RUN_OK_NO_EMBEDDING`` with a ``note``. A record
        whose output file is gone (a deleted ``out_dir``) or whose new scoring
        fails (wrong ``batch`` length, say - ``batch has N entries, embedding
        has M cells``) becomes ``RUN_OK_EVAL_FAILED`` with the reason in
        ``error``.

        **Other hosts.** Records keep ``out_dir`` and ``data_path`` as
        ``run_all`` received them - relative if you passed a relative path.
        A tree fetched or copied from another host, or re-scored from another
        working directory, re-scores only where those folders exist: a
        missing output folder gives ``RUN_OK_EVAL_FAILED``, a missing
        ``data_path`` (without ``labels=``) ``RUN_OK_NO_LABEL_MATCH``.

        **Persisting.** ``mtb.load_batch`` keeps returning the original result
        until the new one is saved.

        See Also
        --------
        mtb.evaluate : the scoring function applied per record.

        BatchResult.save : persist the re-scored result.
        """
        import copy
        new_records = []
        lab_vec = None if labels is None else _eio.as_vector(labels, what="labels")
        bat_vec = None if batch is None else _eio.as_vector(batch, what="batch")
        for r in self.records:
            rec = copy.deepcopy({k: v for k, v in r.items() if k != "_long"})
            rec["_long"] = None
            m = rec.get("method")
            try:
                mods = rec.get("modalities") or []
                v = registry.get(m).select(self.category, set(mods))
                emb = _load_embedding(Path(rec["out_dir"]), v)
                if emb is None:
                    rec["status"] = "RUN_OK_NO_EMBEDDING"
                    rec["note"] = (f"output kind={v.output.kind}; this method does not "
                                   "produce an embedding, so embedding-based metrics do not apply")
                else:
                    _score_record(rec, emb, self.dataset, self.category,
                                  rec.get("data_path"), v,
                                  batch=bat_vec, labels=lab_vec, metrics=metrics)
            except Exception as e:  # noqa: BLE001 - one bad record must not abort the rest
                rec["status"] = "RUN_OK_EVAL_FAILED"
                em = f"{type(e).__name__}: {e}"
                rec["error"] = em if len(em) <= 600 else "... " + em[-596:]
            if verbose:
                print(f"[rescore] {m} -> {rec['status']} "
                      f"{(rec.get('metrics') or {}).get('ARI', '')}", flush=True)
            new_records.append(rec)
        return BatchResult(new_records, self.dataset, self.category, out_dir=self.out_dir)

    def save(self, out_dir=None) -> "Path":
        """Write this result to disk so it outlives the process.

        Parameters
        ----------
        out_dir : path | None
            Target folder, created if missing; ``None`` = the result's own
            ``out_dir``, else the current directory.

        Returns
        -------
        Path
            The folder written.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> res.rescore(metrics="clustering").save("out/clustering_only")
        >>> mtb.load_batch("out/clustering_only").summary

        Notes
        -----
        **Files written.**

        - ``summary.csv`` - the ``summary`` frame.
        - ``long.csv`` - the ``long`` frame, only when some method produced
          metrics.
        - ``failures.csv`` - the ``failures`` frame.
        - ``batch_result.json`` - dataset, category and the per-method records.

        Reload with ``mtb.load_batch`` to re-score or re-plot later without
        re-running any method.

        **Blank confidence on disk.** In ``summary.csv`` the
        ``label_order_note`` column says why ``label_order_confidence`` is
        empty on a row, so that a "single ordering" result is not confused
        with a run that never scored.

        See Also
        --------
        mtb.load_batch : reads the folder back.
        """
        d = Path(out_dir or self.out_dir or ".")
        d.mkdir(parents=True, exist_ok=True)
        sm = self.summary.copy()
        # "single ordering" is a result, "never ran" an absence, and a bare NaN
        # cannot tell them apart on disk. The note has its own column: a sentinel
        # string in the numeric one breaks `> 0.5` and `.isna()` and trips a
        # pandas incompatible-dtype FutureWarning.
        sm = _with_label_order_note(sm)
        sm.to_csv(d / "summary.csv", index=False)
        if not self.long.empty:
            self.long.to_csv(d / "long.csv", index=False)
        self.failures.to_csv(d / "failures.csv", index=False)
        slim = [{k: v for k, v in r.items() if k != "_long"} for r in self.records]
        with open(d / "batch_result.json", "w") as fh:
            json.dump({"dataset": self.dataset, "category": self.category,
                       "records": slim}, fh, indent=1, default=str)
        return d

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        ok = sum(1 for r in self.records if str(r.get("status", "")).startswith("CHAIN_OK"))
        noemb = sum(1 for r in self.records if r.get("status") == "RUN_OK_NO_EMBEDDING")
        nolab = sum(1 for r in self.records if r.get("status") == "RUN_OK_NO_LABEL_MATCH")
        bad = len(self.failures)
        extra = (f", {noemb} ran but not scorable" if noemb else "") + (
            f", {nolab} ran but no labels matched" if nolab else "")
        return (f"<BatchResult {self.category}/{self.dataset}: "
                f"{ok}/{len(self.records)} with metrics{extra}, {bad} failed>")


# ---------------------------------------------------------------------- run_all
def _nothing_runnable_message(dataset: str, category: str, blocked: pd.DataFrame,
                              methods) -> str:
    """The ``ValueError`` text for "not one requested variant can start".

    Scoped to what the caller asked for: with ``methods=`` every requested
    variant is listed with its own reason (one per line); without it the
    first three blocked variants are shown and the message says how many
    there are in total. Reasons of methods the caller did not request are
    never listed: they would point at the wrong fix.
    """
    def _line(r):
        return f"  {r['method']} ({r['modalities']}): {r['reason']}"
    head = f"nothing is runnable for dataset={dataset!r} category={category!r}"
    if methods:
        lines = [_line(r) for _, r in blocked.iterrows()]
        return (f"{head} (methods={list(methods)}). Blocked - one line per requested "
                f"variant:\n" + "\n".join(lines) +
                f"\nfiles_ok / env_ok in mtb.scan({dataset!r}, {category!r}, "
                f"methods={list(methods)}) say which gate failed; mtb.env.doctor() "
                f"for envs." + _platform_suffix(blocked))
    n, k = len(blocked), min(3, len(blocked))
    lines = [_line(r) for _, r in blocked.head(k).iterrows()]
    return (f"{head}. First {k} of {n} blocked variants:\n" + "\n".join(lines) +
            f"\nInspect mtb.scan({dataset!r}, {category!r}) for the full table "
            f"(files_ok / env_ok say which gate failed; mtb.env.doctor() for envs)."
            + _platform_suffix(blocked))


def _platform_suffix(blocked: pd.DataFrame) -> str:
    """One extra sentence for the "nothing is runnable" error on a non-Linux
    host whose rows are blocked by the env gate: the install command every
    reason quotes will refuse here, so say where to run instead of sending
    the user to a download that fails."""
    problem = envs.host_platform_problem()
    if not problem or "env_ok" not in blocked or blocked["env_ok"].all():
        return ""
    return (f"\nNote: {problem} - the `multibench env install` commands above "
            f"refuse on this host; run methods on a Linux host (plan / scan / "
            f"evaluate / plot work here).")


def _check_param_keys(plan_df: pd.DataFrame, params: dict) -> None:
    """Raise ``KeyError`` when ``params`` names a key no scanned variant of that
    method accepts - the same check the run loop applies per method, pulled
    forward into :func:`scan` so a dry run catches the typo before the sweep
    starts.

    A method in ``params`` that has no row in the plan is left alone (it is
    simply not run); unknown method names are caught earlier by
    ``registry.check_method``.
    """
    for m, overrides in (params or {}).items():
        rows = plan_df[plan_df["method"] == m]
        if rows.empty or not overrides:
            continue
        allowed: set = set()
        for _, r in rows.iterrows():
            mods = [] if r["modalities"] == "(data_dir)" else r["modalities"].split("+")
            v = registry.get(m).select(r["category"], set(mods))
            allowed |= set(v.tunable) | set(v.params)
        unknown = [k for k in overrides if k not in allowed]
        if unknown:
            raise KeyError(
                f"{m} does not accept {unknown}; it accepts {sorted(allowed)}. "
                "An empty set means it hardcodes its hyperparameters upstream.")


def _load_embedding(mdir: Path, variant):
    """Read the embedding a finished run left in ``mdir``, or ``None``.

    The primary output when ``output.kind == 'embedding'``; otherwise the first
    ``extra_outputs`` entry of kind embedding that exists (scMoMaT writes a UMAP
    next to its KNN graph). Oriented cells x dims (the larger axis is cells).
    """
    import h5py
    emb = None
    if variant.output.kind == "embedding":
        p = mdir / variant.output.file
        if not p.exists():
            raise FileNotFoundError(f"no output {p} to score - was the run deleted?")
        with h5py.File(p) as h:
            k = variant.output.dataset or ("data" if "data" in h else list(h.keys())[0])
            emb = np.array(h[k])
    else:
        for o in variant.extra_outputs:            # a graph method may still ship an embedding
            if o.kind == "embedding":
                p = mdir / o.file
                if p.exists():
                    with h5py.File(p) as h:
                        k = "data" if "data" in h else list(h.keys())[0]
                        emb = np.array(h[k])
                    break
    if emb is not None and emb.ndim == 2 and emb.shape[0] < emb.shape[1]:
        emb = emb.T
    return emb


def _score_record(rec, emb, dataset, category, data_path, variant, *,
                  batch=None, labels=None, metrics=None):
    """Fill ``rec`` with metrics for ``emb`` (shared by run_all and rescore).

    Sets ``status`` (``CHAIN_OK`` / ``CHAIN_OK_GRAPH_METHOD`` /
    ``RUN_OK_NO_LABEL_MATCH`` / ``RUN_OK_EVAL_FAILED``), ``metrics``,
    ``labels_used``, ``label_order_candidates``, ``batch_source``,
    ``n_batches``, ``emb_shape`` and the tidy ``_long`` frame. ``labels``
    (one per cell) bypasses the label-order search; ``batch`` (one per cell)
    replaces the file-of-origin batch; ``metrics`` restricts the metric set
    (``evaluate(metrics=)``).
    """
    rec["emb_shape"] = list(emb.shape)
    n = emb.shape[0]
    if batch is not None:
        batch = np.asarray(batch)
        if len(batch) != n:
            raise ValueError(f"batch has {len(batch)} entries, embedding has {n} cells")
    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != n:
            raise ValueError(f"labels has {len(labels)} entries, embedding has {n} cells")
        cands = [(["(user labels)"], labels, np.ones(n, dtype=int))]
    else:
        cands = _label_candidates(dataset, n, data_path)
    for k in ("metrics", "labels_used", "label_order_candidates", "_long"):
        rec.pop(k, None)
    if not cands:
        rec["status"] = "RUN_OK_NO_LABEL_MATCH"
        return rec
    names, val, spread = _evaluate_best_order(emb, category, cands, batch=batch,
                                              metrics=metrics)
    if val is None:
        rec["status"] = "RUN_OK_EVAL_FAILED"
        errs = [s["error"] for s in spread if isinstance(s, dict) and s.get("error")]
        if errs:
            rec["error"] = errs[0]
        return rec
    rec["metrics"] = {k: (None if pd.isna(x) else round(float(x), 4))
                      for k, x in val["Value"].items()}
    rec["labels_used"] = names
    if len(spread) > 1:
        rec["label_order_candidates"] = spread
    # which batch vector the batch metrics saw (summary columns batch_source/n_batches)
    if batch is not None:
        rec["batch_source"], rec["n_batches"] = "user", int(len(set(batch.tolist())))
    else:
        bat = next(b for nm, _, b in cands if nm == names)
        nb = int(len(set(np.asarray(bat).tolist())))
        rec["batch_source"], rec["n_batches"] = ("file_of_origin" if nb > 1 else None), nb
    rec["_long"] = _to_long(val, method=rec["method"], dataset=dataset, category=category)
    rec["status"] = ("CHAIN_OK" if variant.output.kind == "embedding"
                     else "CHAIN_OK_GRAPH_METHOD")
    return rec


def run_all(dataset: str, category: str, out_dir=None, *, methods=None, modalities=None,
            params: dict | None = None, data_path=None, evaluate: bool = True,
            dry_run: bool = False, verbose: bool = True,
            timeout: float | None = None,
            skip_existing: bool = False,
            batch=None) -> "BatchResult | pd.DataFrame":
    """Run every runnable method on a dataset under one category and score it.

    Only rows ``mtb.scan`` marks runnable are attempted. A method's failure is
    recorded, never raised, and the sweep is saved under ``out_dir``.

    Parameters
    ----------
    dataset : str
        Dataset folder name under ``data_path``, e.g. ``"MYCITE"`` (not a path).
    category : str
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or ``cross``.
    out_dir : path | None
        Output root, one ``<out_dir>/<method>_<dataset>/`` per method; required
        unless ``dry_run=True``.
    methods : list[str] | None
        Method ids to include, as a list; ``None`` = every runnable method.
    modalities : list[str] | None
        Modality tokens of one combination, e.g. ``["rna", "adt"]``; ``None`` =
        every combination.
    params : dict | None
        Per-method hyperparameters, ``{"Cobolt": {"lr": 1e-3}}``; see
        ``mtb.params_for`` for the accepted keys.
    data_path : path | None
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.DEFAULT.data_path``.
    evaluate : bool
        Score each embedding; ``False`` only runs (status ``RUN_OK``).
    dry_run : bool
        ``True`` = return the ``mtb.scan`` frame for this selection and run
        nothing.
    verbose : bool
        Print ``[run_all] ...`` progress lines.
    timeout : float | None
        Per-method wall-clock cap in seconds; ``None`` = no cap.
    skip_existing : bool
        Reuse an output file already in ``out_dir`` instead of re-running the
        method, to resume an interrupted sweep.
    batch : array-like | None
        One batch id per cell, in embedding row order (array, Series or CSV
        path); ``None`` = batch by the label file each cell came from.

    Returns
    -------
    BatchResult or pandas.DataFrame
        The sweep's ``BatchResult``; read ``summary`` and ``failures`` first.
        With ``dry_run=True``, the ``mtb.scan`` frame.

    Raises
    ------
    FileNotFoundError
        ``<data_path>/<dataset>`` does not exist; the message lists the folders present.
    ValueError
        Unknown ``category``, no matching variant, nothing runnable, or
        ``skip_existing`` with ``params``.
    KeyError
        Unknown id in ``methods`` or ``params``; on a dry run, a rejected ``params`` key.
    TypeError
        A real run without ``out_dir``; ``methods`` or ``modalities`` given as
        a bare string.

    Warns
    -----
    UserWarning
        ``dataset`` matches a folder only up to letter case, or ``modalities``
        drops directory-input methods.

    Examples
    --------
    >>> import multibench as mtb
    >>> plan = mtb.run_all("D11", "vertical", dry_run=True)       # free: what would run?
    >>> plan[["method", "modalities", "runnable", "reason"]]
    >>> res = mtb.run_all("D11", "vertical", out_dir="out/", timeout=3600)
    >>> res.summary                                # one row per method, metrics as columns
    >>> res.failures                               # always check: failures are recorded, not raised

    Notes
    -----
    **Dry run.** ``dry_run=True`` is free; do it first. It returns the
    ``mtb.scan`` frame for the same selection: blocked rows are kept with
    their ``reason``, and ``command`` is rendered for ``out_dir`` (or the
    literal ``'<out_dir>'`` placeholder). Filter ``plan[plan.runnable]`` for
    what will be attempted - ``len(plan)`` is not the sweep size.
    ``multibench run-all --dry-run --format csv`` writes the same frame.

    **Before the sweep.** Every attempted row passed both ``mtb.scan`` gates
    (input files and conda env, plus a GPU where the script needs one), so a
    missing env is reported up front rather than hours in
    (``multibench env doctor`` / ``env install --run``). Methods take minutes
    to hours each.

    **Failures are recorded.** In a real run a method that raises is
    recorded as ``FAIL`` (with its ``error``), one that exceeds ``timeout``
    as ``TIMEOUT``, and the sweep moves on; a ``params`` key the variant does
    not accept is a ``FAIL`` too. Check ``res.failures``.

    **Timeout.** Strongly recommended for unattended runs: without a cap a
    single hanging method blocks everything. Size it from the
    ``runtime_tier`` / ``observed_worst_sec`` columns of ``mtb.scan`` (or
    ``method_info(m)['runtime']``); the slowest methods take more than 4 h.
    The cap covers the run and its scoring; off the main thread it is
    unavailable, with a warning.

    **Saved files.** The result is saved automatically under ``out_dir``
    (``summary.csv``, ``failures.csv``, ``batch_result.json``, and
    ``long.csv`` when some method produced metrics); reload it with
    ``mtb.load_batch``.

    **Resuming.** ``skip_existing=True`` skips the hours an interrupted
    sweep already did. Reuse only checks that the output file exists, not
    that it is complete: a method killed mid-write leaves a truncated file
    that would be reused as if it had succeeded. After a hard kill, delete
    that method's sub-directory before resuming.

    **Tuning.** ``skip_existing=True`` together with ``params=...`` raises
    ``ValueError`` on a real run: reuse is keyed on the output file, not on
    ``params``, so it would return results computed with the old parameters.
    Give each setting a fresh ``out_dir`` (or leave ``skip_existing`` False),
    as ``mtb.sweep`` does:

    ```python
    mtb.run_all("D11", "vertical", out_dir="out/lr", methods=["Multigrate"],
                params={"Multigrate": {"lr": 1e-3}})
    ```

    **Batch vector.** By default the batch metrics use the label file each
    cell came from (``cty1.csv`` -> 1 ...); ``batch=`` replaces that rule and
    is recorded as ``batch_source='user'``. A vector of the wrong length marks
    that method ``RUN_OK_EVAL_FAILED`` (``batch has N entries, embedding has M
    cells``). Re-score a finished sweep with ``BatchResult.rescore``.

    **ATAC representation.** Besides ``["rna", "adt"]``, ``modalities``
    takes ``["rna", "atac_gas"]`` (RNA + ATAC gene activity) and
    ``["rna", "atac_peak"]`` (RNA + ATAC peaks); ``mtb.describe_layout``
    lists every role name. The two ATAC representations do not map to the
    obvious filenames: gene activity is ``atac.h5`` but peaks are
    ``peak.h5``. A peak matrix in ``atac.h5`` runs every method on the
    wrong representation without an error; the numbers are plausible and
    wrong.

    **Errors raised.**

    - An unknown ``category`` - ``ValueError`` listing the four.
    - An unknown id in ``methods`` or ``params`` - ``KeyError`` with a
      did-you-mean hint, before anything runs.
    - A selection that matches no variant - ``ValueError`` ("no 'cross'
      variant matches ..."); a dry run is never empty.
    - A dry run with a ``params`` key no planned variant of that method
      accepts - ``KeyError`` naming the accepted keys, instead of the typo
      being discovered hours in.
    - Variants exist but not one passes both gates - the "nothing is
      runnable ..." ``ValueError``. Its message lists the reason of every
      requested variant (or the first 3 of N when ``methods`` was not
      given), never the reasons of methods you did not ask for. On a
      non-Linux host with env-blocked rows it adds that the install commands
      refuse there.

    **Dataset spelling.** A ``dataset`` that differs from the folder only in
    case (``'d52'``) is replaced by the on-disk spelling, with a
    ``UserWarning``, before anything is named after it.

    See Also
    --------
    mtb.scan : the preflight frame this function runs from.

    mtb.BatchResult : what is returned - ``summary``, ``long``, ``failures``, ``plot``, ``rescore``.

    mtb.sweep : one method over a range of one hyperparameter.

    mtb.load_batch : reload a saved sweep without re-running anything.

    mtb.run : one method, one variant, with explicit inputs.
    """
    registry.check_category(category)      # raises with the valid list on a typo
    _list_of_ids(methods, "methods")       # 'StabMap' is not ['S','t',...]
    _list_of_ids(modalities, "modalities")
    # validate the argument combination before anything is resolved or touched,
    # so a bad combination is reported as such instead of surfacing as an
    # unrelated I/O or "nothing is runnable" error
    params = params or {}
    for _m in params:                      # KeyError (did-you-mean) before any I/O
        registry.check_method(_m)
    if not dry_run and skip_existing and params:
        raise ValueError(
            "skip_existing=True with params=... would silently return results computed "
            "with the OLD parameters (reuse is keyed on the output file, not on params). "
            "Use a fresh out_dir per parameter setting, or skip_existing=False.")
    # the on-disk spelling, decided once here so out_dir names, records and
    # every downstream call agree (and warn once, not per method)
    dataset = _resolve.canonical_dataset(
        Path(data_path) if data_path is not None else config.DEFAULT.data_path, dataset)
    if not dry_run and out_dir is None:
        raise TypeError("run_all() needs out_dir= for a real run (dry_run=True "
                        "returns the scan frame without one)")
    # KeyError (did-you-mean) on an unknown method id, FileNotFoundError on a
    # missing dataset folder, ValueError when no variant of the requested
    # methods exists under this category and KeyError on a params key no
    # variant accepts all come from scan(); blocked rows are kept.
    plan_df = scan(dataset, category, data_path=data_path, methods=methods,
                   modalities=modalities, verbose=False,
                   # the dry run renders (and validates) params in the frame; a real
                   # run validates per method and records a bad override as FAIL
                   params=params if dry_run else None,
                   out_dir=OUT_DIR_PLACEHOLDER if out_dir is None else out_dir)
    if plan_df.empty:
        # only reachable through a modalities= selector that matches nothing:
        # a request problem, reported as such rather than as "nothing is
        # runnable" with other methods' reasons attached
        raise ValueError(
            f"no {category!r} variant matches dataset={dataset!r} methods={methods} "
            f"modalities={modalities}; see mtb.method_info(m)['supports'] and "
            f"mtb.scan({dataset!r})")
    if dry_run:
        if verbose:
            k, n = int(plan_df["runnable"].sum()), len(plan_df)
            msg = (f"[run_all] dry run: {k} of {n} requested variant(s) runnable on "
                   f"{dataset} ({category})")
            if n > k:
                msg += (f"; {n - k} blocked - see the reason column "
                        f"(files_ok / env_ok say which gate; mtb.env.doctor() for envs)")
            print(msg, flush=True)
        return plan_df                     # = scan(): runnable rows first, blocked rows keep `reason`
    blocked = plan_df[~plan_df["runnable"]]
    plan_df = plan_df[plan_df["runnable"]]
    if plan_df.empty:
        # A per-method failure is recorded, never raised - but "not one method
        # could start" means the request is wrong (bad dataset name, wrong
        # category, missing files, missing env). An empty result would report
        # "0 failed", which reads as success and hides a typo.
        raise ValueError(_nothing_runnable_message(dataset, category, blocked, methods))

    batch_vec = None if batch is None else _eio.as_vector(batch, what="batch")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for _, row in plan_df.iterrows():
        m, mods = row["method"], row["modalities"]
        mod_list = [] if mods == "(data_dir)" else mods.split("+")
        from . import __version__ as _pkg_version
        rec = {"method": m, "category": category, "dataset": dataset,
               "modalities": mod_list, "output_kind": row["output_kind"],
               "env": row["env"], "n_tunable": row["n_tunable"], "status": "?", "_long": None,
               # provenance: what ran, where, with what
               "params_used": dict(params.get(m) or {}),
               "out_dir": str(out_dir / f"{m}_{dataset}"),
               "data_path": str(data_path) if data_path else None,
               "multibench_version": _pkg_version,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        t0 = time.time()
        if verbose:
            print(f"[run_all] {m} ({category}/{dataset}) ...", flush=True)
        _deadline_prev = _NOT_ARMED
        try:
            _deadline_prev = _arm_deadline(timeout)
            inp = _resolve.inputs_for(dataset, category, m, modalities=mod_list or None,
                                      data_path=data_path, check=True)
            mdir = out_dir / f"{m}_{dataset}"
            v0 = registry.get(m).select(category, set(mod_list))
            reused = skip_existing and (mdir / v0.output.file).exists()

            # Validate overrides before dispatching, on every path: an unknown key
            # must fail loudly instead of being dropped from the command line.
            mp = params.get(m)
            if mp:
                allowed = set(v0.tunable) | set(v0.params)
                # An empty allowed-set rejects everything: a method that hardcodes
                # its hyperparameters accepts no overrides.
                unknown = [k for k in mp if k not in allowed]
                if unknown:
                    raise KeyError(
                        f"{m} does not accept {unknown}; it accepts {sorted(allowed)}. "
                        "An empty set means it hardcodes its hyperparameters upstream.")

            if reused:
                if verbose:
                    print(f"[run_all]   reusing existing output in {mdir}", flush=True)
                res = None                       # read back from disk below
            else:
                res = _run(method=m, category=category, inputs=inp,
                           out_dir=str(mdir), params=mp)
            rec["reused"] = bool(reused)
            rec["run_sec"] = round(time.time() - t0, 1)
            v = v0
            if v.output.kind == "embedding" and res is not None:
                emb = np.asarray(res.output)
                if emb.ndim == 2 and emb.shape[0] < emb.shape[1]:
                    emb = emb.T
            else:
                emb = _load_embedding(mdir, v)
            if emb is None:
                rec["status"] = "RUN_OK_NO_EMBEDDING"
                rec["note"] = (f"output kind={v.output.kind}; this method does not produce an "
                               "embedding, so embedding-based clustering metrics do not apply")
            else:
                rec["emb_shape"] = list(emb.shape)
                if not evaluate:
                    rec["status"] = "RUN_OK"
                else:
                    try:
                        _score_record(rec, emb, dataset, category, data_path, v,
                                      batch=batch_vec)
                    except TimeoutError:
                        raise
                    except Exception as e:      # scoring failed; the run itself succeeded
                        rec["status"] = "RUN_OK_EVAL_FAILED"
                        rec["error"] = f"{type(e).__name__}: {e}"
        except TimeoutError as e:
            rec["status"] = "TIMEOUT"
            rec["error"] = str(e)
            rec["run_sec"] = round(time.time() - t0, 1)
        except Exception as e:
            rec["status"] = "FAIL"
            em = f"{type(e).__name__}: {e}"
            # Truncate from the left: the text ends with the actual exception,
            # which a right-side cut would discard.
            rec["error"] = em if len(em) <= 600 else "... " + em[-596:]
            rec["traceback"] = traceback.format_exc()[-1200:]
            rec["run_sec"] = round(time.time() - t0, 1)
        finally:
            _disarm_deadline(_deadline_prev)
        if verbose:
            print(f"[run_all]   -> {rec['status']} ({rec.get('run_sec')}s) "
                  f"{(rec.get('metrics') or {}).get('ARI', '')}", flush=True)
        records.append(rec)

    result = BatchResult(records, dataset, category, out_dir)
    result.save()          # survive process exit; reload with load_batch()
    return result


def sweep(dataset: str, category: str, method: str, param: str, values, *,
          out_dir, modalities=None, data_path=None, timeout=None,
          verbose: bool = True) -> pd.DataFrame:
    """Run one method repeatedly over a range of one hyperparameter.

    Replaces a hand-written loop and its two usual mistakes: settings that
    share one ``out_dir`` and overwrite each other, and results that no
    longer say which value produced them.

    Parameters
    ----------
    dataset : str
        Dataset folder name under ``data_path``, as for ``mtb.run_all``.
    category : str
        Integration category of the variant to run.
    method : str
        Registry method id, e.g. ``"Matilda"``.
    param : str
        Hyperparameter to sweep; one of the variant's ``tunable`` keys
        (``mtb.params_for``).
    values : iterable
        Settings to try; each one is a separate ``mtb.run_all``.
    out_dir : path
        Root folder; each setting runs under ``<out_dir>/<param>_<value>/``.
    modalities : list[str] | None
        Modality tokens of the variant, when the method has several in
        ``category``; ``None`` = every variant.
    data_path : path | None
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.DEFAULT.data_path``.
    timeout : float | None
        Per-setting wall-clock cap in seconds, passed to ``run_all``;
        ``None`` = no cap.
    verbose : bool
        Print ``run_all``'s progress lines.

    Returns
    -------
    pandas.DataFrame
        The settings' ``BatchResult.summary`` rows stacked, the swept value
        first (column named ``param``). A tidy frame for plotting is in
        ``df.attrs["long"]``.

    Raises
    ------
    KeyError
        Unknown ``method``, or ``param`` not among the tunable keys of a single variant.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.params_for("Multigrate", "vertical", ["rna", "adt"])["tunable"]   # what can be swept
    >>> df = mtb.sweep("MYDATA", "vertical", "Multigrate", "lr",
    ...                [1e-4, 1e-3, 1e-2], out_dir="out/lr")
    >>> df[["lr", "status", "ARI", "NMI"]]
    >>> mtb.plot.bubble(df.attrs["long"])      # one series per setting

    Notes
    -----
    **Folder names.** Each setting's folder is ``<param>_<value>`` with
    ``.`` -> ``p`` and ``-`` -> ``m`` (``lr=0.001`` runs under
    ``<out_dir>/lr_0p001/``).

    **The tidy frame.** ``df.attrs["long"]`` makes each setting a separate
    series (``"Multigrate (lr=0.001)"``), so it can go straight into
    ``mtb.plot.bubble``; ``.long`` keys rows by method, so without it every
    setting would collapse onto one row. ``DataFrame.attrs`` does not
    survive ``to_csv``, so the frame is also written to
    ``<out_dir>/sweep_long.csv`` (path in ``df.attrs["long_path"]``) when any
    setting produced metrics.

    **Failed settings.** A setting that fails is not fatal: ``run_all``
    records it, so that value's row appears with ``status`` ``FAIL`` (or
    ``TIMEOUT``) and empty metrics rather than aborting the sweep. Check the
    ``status`` column before reading the curve - a failed setting and a poor
    one must not be confused.

    **Untunable methods.** Check ``mtb.params_for`` first: a method whose
    ``tunable`` is empty hardcodes its hyperparameters upstream and cannot be
    swept. ``sweep`` does not reject it up front; every setting is recorded
    as ``FAIL``.

    **Errors.** An unknown ``method`` raises ``KeyError`` with a did-you-mean
    hint; the ``KeyError`` for an unknown ``param`` lists the keys the
    variant accepts. The ``param`` check needs one variant: with
    ``modalities=None`` and several variants in ``category`` (Matilda under
    ``vertical``), an unknown ``param`` is recorded as ``FAIL`` for every
    setting instead - pass ``modalities`` to get the ``KeyError``. Errors of
    ``mtb.run_all`` (e.g. nothing is runnable) propagate.

    See Also
    --------
    mtb.params_for : the tunable hyperparameters of the variant.

    mtb.run_all : what each setting runs through.
    """
    registry.check_method(method)          # KeyError with a did-you-mean hint
    tune = _params_for_method(method, category, modalities)
    if tune is not None and tune != {} and param not in tune:
        raise KeyError(
            f"{method} does not expose {param!r}; it accepts {sorted(tune)}. "
            f"(An empty set means it hardcodes its hyperparameters upstream.)")
    frames, longs = [], []
    for v in values:
        tag = str(v).replace(".", "p").replace("-", "m")
        res = run_all(dataset, category, out_dir=Path(out_dir) / f"{param}_{tag}",
                      methods=[method], modalities=modalities, data_path=data_path,
                      params={method: {param: v}}, timeout=timeout, verbose=verbose)
        df = res.summary
        df.insert(0, param, v)
        frames.append(df)
        lg = res.long
        if not lg.empty:                       # tag the tidy frame too: each setting
            lg = lg.copy()                     # becomes its own series instead of
            lg[param] = v                      # collapsing onto one method row
            lg["method"] = lg["method"].astype(str) + f" ({param}={v})"
            longs.append(lg)
    out = pd.concat(frames, ignore_index=True)
    lng = pd.concat(longs, ignore_index=True) if longs else pd.DataFrame()
    out.attrs["long"] = lng
    # DataFrame.attrs does not survive to_csv, so also write the tidy frame beside
    # the run, where a later session can re-plot it.
    if not lng.empty:
        lp = Path(out_dir) / "sweep_long.csv"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lng.to_csv(lp, index=False)
        out.attrs["long_path"] = str(lp)
    return out


def _params_for_method(method, category, modalities):
    """The method's tunable set, or None if it cannot be determined."""
    try:
        from .discover import params_for
        return params_for(method, category,
                          list(modalities) if modalities else None)["tunable"]
    except Exception:
        return None
