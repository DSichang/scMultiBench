"""High-level workflow: point at a dataset, run everything that applies, get metrics and a figure.

The low-level API (``inputs_for`` -> ``run`` -> ``evaluate`` -> ``plot``) requires
you to know a method's name, its integration category AND its exact modality
combination. This module removes that burden:

    mtb.scan("D11")                     # what can I run on this data?
    res = mtb.run_all("D11", "vertical", out_dir="out/")  # run all of it, with metrics
    res.plot()                            # one figure

It also handles two traps that otherwise yield silently WRONG numbers:

* **output kind** - not every method returns an embedding. Methods emitting a
  graph or spatial coordinates are recorded as such instead of being scored with
  embedding metrics (scoring a KNN index matrix quietly gives ARI ~ 0).
* **label order** - ``evaluate`` needs labels in the embedding's cell order, and
  matching by LENGTH cannot distinguish orders because every permutation has the
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

#: internal alias of ``method_info(m)["runtime"]`` (the public ``mtb.runtime_hint``
#: is the deprecated, warning spelling)
runtime_hint = _runtime_hint


def load_batch(out_dir) -> "BatchResult":
    """Reload a :class:`BatchResult` that :meth:`BatchResult.save` wrote.

    ``run_all`` saves automatically, so after an overnight sweep you can come back
    and re-plot or re-inspect without re-running anything::

        res = mtb.load_batch("out/")
        res.summary
        res.plot().savefig("compare.png")
    """
    d = Path(out_dir)
    with open(d / "batch_result.json") as fh:
        blob = json.load(fh)
    recs = blob["records"]
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
    "vertical": "Several modalities measured in the SAME cells (e.g. CITE-seq "
                "RNA+ADT, or 10x multiome RNA+ATAC). Cells are already matched.",
    "diagonal": "Modalities measured in DIFFERENT cells, with no pairing "
                "(e.g. an RNA experiment and a separate ATAC experiment).",
    "mosaic":   "Several batches where only SOME share a modality; a paired batch "
                "bridges the others.",
    "cross":    "Several batches in which ALL modalities are present; the task is "
                "removing batch effects. (Spatial slice registration also lives here.)",
}

#: Modality role -> the file the loader looks for in <data_path>/<dataset>/.
ROLES = {
    "rna":       "rna.h5      - gene expression",
    "adt":       "adt.h5      - surface protein (CITE-seq antibody-derived tags)",
    "atac":      "atac.h5     - chromatin accessibility",
    "atac_gas":  "atac.h5     - ATAC as GENE-ACTIVITY scores  <-- note: plain atac.h5",
    "atac_peak": "peak.h5     - ATAC as PEAKS                 <-- note: peak.h5, NOT atac.h5",
    "rna1/rna2/...": "rna1.h5, rna2.h5, ... - one file per BATCH (mosaic/cross)",
    "adt1/adt2/...": "adt1.h5, adt2.h5, ... - one file per BATCH (mosaic/cross)",
    "cty":       "cty.csv     - cell-type labels, ONE label set (vertical)",
    "rna_cty / atac_cty":
                 "rna_cty.csv, atac_cty.csv - one label file PER MODALITY, used when "
                 "RNA and ATAC come from different cells (diagonal)",
    "cty1/cty2/...":
                 "cty1.csv, cty2.csv, ... - one label file per BATCH (mosaic/cross)",
}


def list_categories() -> dict:
    """The valid ``category`` values, with a plain-language description of each.

    ``category`` is a required argument of :func:`run_all`; this is the list.

        >>> mtb.list_categories()["vertical"]
        'Several modalities measured in the SAME cells ...'
    """
    return dict(CATEGORIES)


def describe_layout(category: str | None = None) -> str:
    """How to lay out your OWN dataset so the package can find it.

    Prints the directory layout and the role -> filename mapping. Start here when
    bringing your own data, then confirm with :func:`scan`.

    A "role" is just the name of one input a method takes. For CITE-seq the roles
    are ``rna`` (``rna.h5``) and ``adt`` (``adt.h5``, surface protein /
    antibody-derived tags), plus ``cty.csv`` for cell-type labels::

        <data_path>/MYCITE/
            rna.h5
            adt.h5
            cty.csv

    **Several batches** (mosaic / cross integration) use one NUMBERED file per
    batch, in the same flat directory - not sub-folders, and not one
    pre-concatenated matrix. Three batches of CITE-seq::

        <data_path>/COREBATCH/
            rna1.h5   adt1.h5   cty1.csv     # batch 1
            rna2.h5   adt2.h5   cty2.csv     # batch 2
            rna3.h5   adt3.h5   cty3.csv     # batch 3

    Batch membership is carried by the file numbering; there is no batch column.

    **Spatial registration** (PASTE/PASTE2/SPIRAL/GPSA, category ``cross``) takes
    a DIRECTORY of per-slice ``.h5ad`` files instead - see the SPATIAL
    REGISTRATION block of the output. ``category`` is validated: a typo raises
    ``ValueError`` listing the four categories, and ``'spatial'`` raises with a
    pointer to ``describe_layout('cross')`` (it is a task, not a category).

    The lists of methods that need gene-activity vs peak ATAC matrices are built
    from the method registry at call time (``find_methods(atac=...)``), so they
    cannot drift from what ``method_info`` / ``scan``'s ``atac`` column say.
    """
    if category == "spatial":
        raise ValueError(
            "'spatial' is not a category; registration methods live under "
            "category 'cross' - see describe_layout('cross')")
    registry.check_category(category)       # None passes; typo -> ValueError
    # ATAC representation lists come from the registry, not from prose that rots.
    from .discover import find_methods as _find_methods
    gas_methods = _find_methods(atac="gene_activity")
    peak_methods = _find_methods(atac="peak")
    lines = ["Put your files in  <data_path>/<DATASET_NAME>/ , e.g. ./data/MYDATA/",
             "  (dataset = the folder NAME; data_path = the folder that CONTAINS it)",
             ""]
    LAYOUTS = {
        "vertical": ["  rna.h5 + adt.h5 (CITE-seq)  or  rna.h5 + atac.h5 (multiome)",
                     "  cty.csv        <- ONE label file; the cells are already matched"],
        "diagonal": ["  rna.h5         <- the RNA cells",
                     "  atac.h5        <- the ATAC cells (gene activity); peak.h5 for peaks",
                     "  rna_cty.csv AND atac_cty.csv",
                     "                 <- ONE LABEL FILE PER MODALITY. The two cell sets are",
                     "                    disjoint, so they cannot share a single cty.csv."],
        "mosaic":   ["  rna1.h5 rna2.h5 atac2.h5 atac3.h5   <- numbered, one per batch",
                     "  cty1.csv cty2.csv cty3.csv          <- one per batch"],
        "cross":    ["  rna1.h5 rna2.h5 rna3.h5 + adt1.h5 adt2.h5 adt3.h5",
                     "  cty1.csv cty2.csv cty3.csv          <- one per batch",
                     "  (spatial registration instead takes a directory of .h5ad slices)"],
    }
    if category in LAYOUTS:
        lines += [f"LAYOUT FOR {category.upper()}:"] + LAYOUTS[category] + [""]
    else:
        for _c, _ls in LAYOUTS.items():
            lines += [f"{_c}:"] + _ls
        lines += [""]
    lines += ["  (numbered files live in the SAME flat dir; there is no batch column)",
             "", "Modality roles and the filenames they resolve to:"]
    lines += [f"    {k:16s} {v}" for k, v in ROLES.items()]
    lines += ["",
              "!! ATAC: the role name does NOT guarantee the representation.",
              "   atac_gas resolves to atac_gas.h5 if present, otherwise FALLS BACK",
              "   to atac.h5 - and in the shipped multiome datasets (D12-D17) atac.h5",
              "   contains PEAKS, not gene activity. Only D27/D28 ship a real",
              "   atac_gas.h5. Verified by feature names: 12 atac.h5 files are peaks",
              "   (chr1:3094772-3095489), 2 atac_gas.h5 files are gene activity.",
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
              "  If you do build it yourself, ALL THREE datasets are required:",
              "    matrix/data      the matrix, stored FEATURES x CELLS",
              "    matrix/features  one entry per feature (row of matrix/data)",
              "    matrix/barcodes  one entry per cell    (column of matrix/data)",
              "  e.g. 2,000 genes x 5,000 cells -> matrix/data has shape (2000, 5000),",
              "  matrix/features has 2000 entries and matrix/barcodes has 5000.",
              "  NOTE this is the TRANSPOSE of the scanpy/AnnData convention",
              "  (AnnData.X is cells x genes). scan() rejects a transposed file, and",
              "  a file with only matrix/data fails with a KeyError about 'features'.",
              "Labels are a single-column CSV: one header line (typically 'x'),",
              "then one cell-type label per cell; the evaluator reads the first",
              "column and skips the header.", ""]
    if category is None or category == "cross":
        reg = sorted(_find_methods(task="registration"))
        lines += ["SPATIAL REGISTRATION (task 'registration', category 'cross'):",
                  f"  Methods: {', '.join(reg)}. They take a DIRECTORY of slices, not",
                  "  modality files - the role is `data_dir` and scan() shows modalities",
                  "  as '(data_dir)'; pass NO modalities to run_all/inputs_for.",
                  "    <data_path>/MYVISIUM/            (or <data_path>/MYVISIUM/processed/)",
                  "        slice_0.h5ad  slice_1.h5ad  ...   one AnnData per slice, >= 2",
                  "  Each .h5ad needs .X (expression, spots x genes) and",
                  "  .obsm['spatial'] (spot coordinates, spots x 2). GPSA additionally",
                  "  needs obs['Ground_Truth'] (a region/layer label per spot) in",
                  "  EVERY slice - its driver reads that column at load; PASTE and",
                  "  PASTE2 read no obs column. The upstream scripts glob",
                  "  data_dir + '*.h5ad' WITHOUT sorting (directory order, which the",
                  "  filesystem decides), so run() stages the slices as zero-padded",
                  "  symlinks (00_<name>.h5ad ... in sorted order) under <out_dir>/inputs/",
                  "  and writes <out_dir>/slices_manifest.json: aligned_slice_<i>.h5ad ->",
                  "  the source file, in the order the script's glob returned - that",
                  "  manifest, not the prefix, is authoritative. Keep it - it is the",
                  "  ONLY link back:",
                  "  PASTE writes its slices WITHOUT any obs column (upstream",
                  "  main_PASTE_pairwise.py drops them all at load), PASTE2 rewrites .X",
                  "  (normalize_total + log1p + a 2,000-HVG subset) before writing, and",
                  "  GPSA keeps obs['Ground_Truth'] only. SPIRAL also wants a UNIQUE",
                  "  leading token per filename (the part before the first '_'); the",
                  "  staged 00_/01_ prefixes satisfy that.",
                  "  Output: aligned_slice_<i>.h5ad per slice (coordinates, not an",
                  "  embedding), so run_all records RUN_OK_NO_EMBEDDING - clustering",
                  "  metrics do not apply, and registration metrics are NOT wired into",
                  "  mtb.evaluate in this version.",
                  "  scan() checks the dir for >= 2 .h5ad slices with obsm['spatial'],",
                  "  and for GPSA that every slice carries obs['Ground_Truth'].",
                  ""]
    if category:
        lines += [f"{category}: {CATEGORIES.get(category, '(unknown category)')}", ""]
    lines += ["", "ENVIRONMENTS",
              "  Every method runs in its OWN conda env (they need mutually",
              "  incompatible framework versions). scan() checks TWO gates per row -",
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
    """Does this dataset really hold what a ``data_dir`` method needs?

    ``data_dir`` resolves to the dataset directory itself when there is no
    ``processed/`` subdir, so the path ALWAYS exists and existence proves nothing.
    Without a content check, spatial-registration methods look runnable on every
    dataset and a sweep would burn hours before failing. The check itself lives
    in :func:`multibench.engine.resolve._check_data_dir` (shared with
    ``inputs_for(check=True)``); this name is kept for callers of the old helper.
    """
    return _resolve._check_data_dir(variant, ds_dir)



@functools.lru_cache(maxsize=1)
def _installed_envs() -> frozenset:
    """Conda envs present on this machine (cached; see mtb.env.doctor())."""
    try:
        return frozenset(envs.installed_envs())
    except Exception:      # never let an env probe break discovery
        return frozenset()


#: Known method x dataset incompatibilities that file/env checks cannot see.
#: These are CONTENT problems - the files exist and the env is installed, but the
#: method still cannot finish. Surfaced by scan() so a sweep does not discover them
#: hours in.
_CAVEATS = {
    ("GLUE", "D28"): ("GLUE parses coordinates out of peak NAMES and needs them "
                      "colon-delimited (chr1:1-200); D28's are underscore-delimited "
                      "and it IndexErrors. Use D27, or rename the peaks."),
}


def _missing_script(variant, *, method: str | None = None) -> str:
    """Why this variant's script is unreachable here, or "" when it is fine.

    A method whose script is absent must not be reported runnable: the run
    would fail minutes later with a shell error instead of here, instantly.
    Three cases are checked and no more:

    * an ABSOLUTE entrypoint that does not exist - it names one machine's
      filesystem, so no download can supply it;
    * a repo-relative entrypoint missing from a reference checkout that IS
      present. When no checkout exists at all, nothing is reported: `run` and
      `run_all` fetch it on first use, and flagging every method as broken
      before that first fetch would be wrong;
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
    if ep.is_absolute():
        if ep.exists():
            return ""
        return _resolve.benchmark_host_only_reason(str(ep)) or (
                f"method script not found at {ep} - this entrypoint is an "
                f"absolute path on another machine; the script is not part of "
                f"the public scMultiBench repository, so it cannot be fetched")
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


#: scan() columns, in order. The first eleven are the historical set; the
#: two-gate columns, the registry-derived flags and (0.3.0) the ``command``
#: column are APPENDED so positional readers keep working. Add columns here,
#: never remove.
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
    # keep the TAIL: the filename is at the END of the message and is the whole
    # point of it. Right-truncation ate exactly what the user needs.
    return msg if len(msg) <= limit else msg[:100] + " ... " + msg[-(limit - 120):]


_ABS_PATH_RE = re.compile(r"(?<![\w./-])/(?:[^\s'\"\[\]{}(),:;]+/)+[^\s'\"\[\]{}(),:;]*")
_EXC_PREFIX_RE = re.compile(r"^[A-Z]\w*(?:Error|Exception|Warning): ")
_HOST_ONLY_SHORT = ("benchmark-host-only: script not published "
                    "(see method_info(m)['availability'])")
# the full host-only text (resolve.benchmark_host_only_reason) ends with this
# marker and contains '; ' itself, so it is collapsed BEFORE splitting on '; '
_HOST_ONLY_RE = re.compile(r"benchmark-host-only:.*?\(method_info\(m\)\['availability'\]\)",
                           re.S)


def _short_reason(text: str, method: str, dataset: str, category: str | None) -> str:
    """The ``reason`` column form of a ``files_reason``: what is missing, no noise.

    ``files_reason`` keeps the verbatim exception text (``FileNotFoundError:
    UnitedNet/D11/vertical: input files not found on disk: {'atac_gas':
    '/home/wen/data/D11/atac_gas.h5', ...}``) because the full path is what a
    user greps for. ``reason`` is what the 17-column frame, the CLI table
    and the "nothing is runnable" error show, so it drops what every row
    repeats: the exception class, the ``method/dataset/category:`` prefix and
    the absolute directory (each path becomes its basename). A
    benchmark-host-only row collapses to one sentence. The env half of
    ``reason`` is untouched - it carries the copy-pasteable install command.
    """
    if not text:
        return text
    text = _HOST_ONLY_RE.sub(_HOST_ONLY_SHORT, text)
    parts = []
    for part in text.split("; "):
        if part.startswith("benchmark-host-only:"):
            parts.append(_HOST_ONLY_SHORT)
            continue
        part = _EXC_PREFIX_RE.sub("", part)
        prefix = f"{method}/{dataset}/{category}: "
        if part.startswith(prefix):
            part = part[len(prefix):]
        part = _ABS_PATH_RE.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], part)
        parts.append(part)
    return "; ".join(parts)


def _list_of_ids(value, name: str):
    """Reject a bare string where a list of ids is expected.

    ``methods='StabMap'`` iterates the CHARACTERS and used to die with
    ``KeyError: unknown method 'S'``; say what was meant instead.
    """
    if isinstance(value, str):
        raise TypeError(
            f"{name} must be a list of ids, got the string {value!r} - did you "
            f"mean {name}=[{value!r}]?")
    return value


def _variant_consumes_atac(variant) -> bool:
    """Does THIS variant take an ATAC input (role or const filename)?"""
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
    """Report every method that can run on ``dataset``, why the rest cannot,
    and the exact command each one would run.

    Returns one row per (method, category, modalities) variant - runnable rows
    first - with TWO independent gates and their verdicts:

    ``files_ok`` / ``files_reason``
        the input files resolve on disk, are oriented features x cells, every
        label CSV has one row per cell of the modality it labels, and a
        ``data_dir`` method finds what it needs (>= 2 ``.h5ad`` slices with
        ``obsm['spatial']`` for registration; scBridge's bare filenames). This
        gate ALWAYS runs, whether or not any conda env is installed, so a
        laptop without envs still tells you whether your layout is right.
    ``env_ok`` / ``env_reason``
        the method's conda env exists on this machine. The reason names the
        env and the one-method install command
        (``multibench env install --methods X --packed --run``).

    ``runnable = files_ok & env_ok`` and ``reason`` joins the non-empty reasons
    with ``"; "`` (empty iff runnable) - the two columns every older caller
    reads. ``reason`` is the SHORT form: the file half drops the exception
    class, the ``method/dataset/category:`` prefix and the absolute directory
    (``input files not found on disk: {'atac': 'atac.h5'}. Available files
    in D11: [...]``), while ``files_reason`` / ``env_reason`` keep the
    verbatim text with full paths. Further columns: ``modalities`` is a
    ``+``-joined STRING here (e.g. ``"rna+adt"``); ``run_all``/``inputs_for``
    take it as a LIST (``["rna", "adt"]``), so split on ``"+"``. The sentinel
    ``"(data_dir)"`` marks a method that consumes a whole DIRECTORY rather
    than named modality files (the spatial-registration methods, and
    scBridge) - for those, pass no ``modalities`` at all (``modalities=[]``
    selects exactly them). ``needs_labels`` says whether THIS variant demands
    a label file (``cty.csv`` ...) as an input; ``atac`` is the ATAC
    representation the method expects (``'peak'`` / ``'gene_activity'`` /
    ``None`` when the variant takes no ATAC); ``runtime_tier`` /
    ``observed_worst_sec`` (see ``method_info(m)['runtime']``) let you size a
    sweep BEFORE launching it; ``caveat`` carries known content traps (e.g.
    an ``atac_gas`` role that fell back to a PEAK matrix). ``command`` is the
    shell line each variant would run (``run(..., dry_run=True)``,
    ``shlex``-joined), writing under ``<out_dir>/<method>_<dataset>/`` -
    the literal ``'<out_dir>'`` placeholder unless ``out_dir`` is given;
    ``""`` for a row whose inputs do not resolve (there is nothing to hand
    the script), while a row blocked only by ``env_ok`` still shows its
    command - the line to paste into a job script once the env is built.
    Nothing is executed. This is the first call to make when pointing the
    benchmark at a NEW dataset, and the frame ``run_all(dry_run=True)``
    returns (the 0.2 ``plan`` / ``plan_commands`` are deprecated aliases).

    The full frame is 18 columns wide (``SCAN_COLUMNS``). At the REPL select
    the four that answer "what can I run and why not the rest"::

        df[["method", "modalities", "runnable", "reason"]]

    and go to ``files_reason`` / ``env_reason`` only for a row you are
    debugging (``multibench scan`` prints that compact view by default;
    ``--columns all`` includes ``command``).

    Parameters
    ----------
    dataset : str
        Folder NAME under ``data_path`` (not a path). If the folder does
        not exist, ``FileNotFoundError`` lists the folders that do. A spelling
        that differs from the folder only in case (``'d52'`` on macOS) is
        replaced by the on-disk spelling with a ``UserWarning``
        (:func:`multibench.engine.resolve.canonical_dataset`), so the reasons,
        ``out_dir`` names and records never carry a name Linux would reject.
    category : str, optional
        Restrict to one integration category (``ValueError`` listing the
        valid ones on a typo); default: all four.
    methods : list of str, keyword-only, optional
        Restrict to these registry ids, as a LIST (``KeyError`` with a
        did-you-mean hint on a typo; a bare string such as
        ``methods="StabMap"`` raises ``TypeError`` saying to pass a list).
        Blocked rows are KEPT with their reason; ``ValueError`` when no
        variant of the requested methods exists under ``category`` (a known
        id with no diagonal variant, say) - never a silently empty frame.
    modalities : list of str, keyword-only, optional
        Restrict to ONE modality combination, given as a list of role names
        (``["rna", "adt"]``; ``protein`` is accepted for ``adt``; a bare
        string raises ``TypeError``). It is an exact selector, so
        directory-input variants (``"(data_dir)"``: scBridge, the
        registration methods) are excluded by any non-empty list; when that
        happens a ``UserWarning`` names them and says ``modalities=[]``
        selects them.
    data_path : path, keyword-only, optional
        The folder that CONTAINS ``dataset``; default the configured data
        root.
    out_dir : path or str, keyword-only
        Root the ``command`` lines write under
        (``<out_dir>/<method>_<dataset>/``, exactly like ``run_all``).
        Default: the literal placeholder ``'<out_dir>'``; pass the real one
        for paste-ready lines.
    params : dict, keyword-only, optional
        ``{method: {key: value}}`` hyperparameter overrides merged into each
        command the way ``run_all(params=)`` merges them; a key no variant of
        that method accepts raises ``KeyError`` naming the accepted keys, so
        a typo is caught here rather than hours into a sweep.
    verbose : bool, keyword-only
        Print one line ``[scan] files OK for k/n method rows; e/n envs
        installed`` (default True; ``run_all`` passes its own ``verbose``).

    Returns
    -------
    pandas.DataFrame
        One row per variant, columns ``SCAN_COLUMNS``, runnable rows first.

    ::

        mtb.scan("MYCITE", "vertical", data_path="/home/wen/data")
        #   method    category  modalities  env      output_kind  runnable  reason
        #   Matilda   vertical  rna+adt     matilda  embedding    True
        #   totalVI   vertical  rna+adt     scmb_scvi embedding   True

    A CITE-seq folder (``rna.h5`` + ``adt.h5`` + ``cty.csv``) is ``vertical`` with
    modalities ``["rna", "adt"]``; RNA and ATAC from different cells is
    ``diagonal``. See :func:`list_categories` and :func:`describe_layout`.

    Each method runs in its OWN conda environment (they need mutually
    incompatible framework versions). ``runnable=True`` verifies BOTH that the
    input files exist AND that that environment is installed, so a sweep never
    starts a method that cannot finish. List them with ``multibench env
    doctor``; build them with ``multibench env install --run``.
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
            f"{dirs}. dataset= is the folder NAME and data_path= the folder that "
            f"CONTAINS it (see mtb.describe_layout())")
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
        # --- gate 1: files. ALWAYS runs, env or no env. ---------------------
        # Both halves are checked (the method's script AND the dataset's files)
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
        # REQUEST problem (Matilda is not a cross method), reported as such
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
    """Read a label CSV - ONE reader for the package (``eval.io.read_labels``)."""
    return _eio.read_labels(path)


_NOT_ARMED = object()


def _arm_deadline(seconds):
    """Start a SIGALRM deadline covering an ENTIRE per-method step.

    The alarm used to wrap only the dispatch call and was cancelled immediately
    after it, so everything downstream - reading the output back, and above all
    computing the metrics - ran unbounded. A method finishing in 35s could then
    spend 105 minutes in the metric layer with timeout= set and never fire.

    Returns the previous handler, or ``_NOT_ARMED`` when no deadline was asked
    for. ``None`` is not usable as that sentinel: signal.signal legitimately
    returns None when the previous handler was not installed from Python.
    """
    if not seconds:
        return _NOT_ARMED
    import math
    import signal
    import threading
    import warnings

    if threading.current_thread() is not threading.main_thread():
        # signal.signal raises off the main thread; a deadline that cannot be
        # armed must be SAID, not silently dropped
        warnings.warn("timeout= is unavailable off the main thread; "
                      "running without a deadline")
        return _NOT_ARMED

    def _fire(signum, frame):
        raise TimeoutError(f"exceeded timeout of {seconds}s")

    prev = signal.signal(signal.SIGALRM, _fire)
    # int() truncation turned timeout=0.5 into alarm(0) = NO alarm at all
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
        # Only orderings whose TOTAL length equals n can survive the filter
        # below, and length is order-independent - so pick length-matching
        # SUBSETS first (combinations) and permute only those. The old code
        # materialized a concatenated array for every permutation of every
        # subset: factorial in files, gigabytes for nothing.
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
        # batch = which label FILE each cell came from. For multi-batch designs
        # (mosaic / cross) that IS the batch, so batch-correction metrics become
        # computable without asking the caller for anything extra.
        bat = np.concatenate([np.full(len(ctys[k]), i + 1) for i, k in enumerate(names)])
        out.append((list(names), lab, bat))
    return out


#: Below this ARI the winning ordering is itself at chance, so the confidence
#: ratio would compare two noise values and mean nothing.
_CHANCE_ARI = 0.05


def _order_confidence(cands) -> float | None:
    """How clearly the winning label order beat the alternatives, on a 0-1 scale.

    ``(best - runner_up) / best``. A plain DIFFERENCE would be useless here: the
    runner-up sits near chance (ARI ~ 0), so the difference is bounded above by the
    ARI itself, and any method scoring 0.3 could never look "clearly separated" no
    matter how unambiguous its ordering. Dividing by the winner removes that
    coupling, so a genuinely unambiguous order reads ~1.0 whether the method scored
    0.9 or 0.2.
    """
    if not cands or len(cands) < 2:
        return None
    best, second = cands[0]["ARI"], cands[1]["ARI"]
    # A ratio of two chance-level numbers is noise, not confidence: 0.0004 vs
    # 0.0002 would read 0.5 while BOTH orderings are garbage. Report None so the
    # column cannot be misread as evidence when no ordering worked at all.
    if best < _CHANCE_ARI:
        return None
    return round(max(0.0, (best - second) / best), 4)


def _accepts_metrics(fn) -> bool:
    """Does ``fn`` take the 0.3 ``metrics=`` knob (or ``**kwargs``)?"""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return "metrics" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _evaluate_compat(fn, emb, *, metrics=None, batch=None, **kw):
    """Call an ``evaluate`` with the ONE metric-selection knob ``metrics=``.

    Transitional: when ``fn`` predates the 0.3 signature (no ``metrics``
    parameter) the request is translated to its ``task=`` / ``only=`` pair -
    a family token becomes ``task``, a list of codes becomes ``only`` under
    ``task='all'`` (batch given) or ``'clustering'``. Drop this shim once
    ``eval/`` carries ``metrics=``.
    """
    if _accepts_metrics(fn):
        return fn(output=emb, metrics=metrics, batch=batch, **kw)
    if isinstance(metrics, str):
        return fn(output=emb, task=metrics, batch=batch, **kw)
    task = "all" if batch is not None else "clustering"
    only = None if metrics is None else set(metrics)
    return fn(output=emb, task=task, only=only, batch=batch, **kw)


def _evaluate_best_order(emb, category, cands, *, batch=None, metrics=None):
    """Score each candidate label order, keep the best, return the full spread.

    ``batch`` (optional, one entry per cell in embedding order) REPLACES the
    file-of-origin batch vector carried by each candidate - ``run_all(batch=)``
    / ``BatchResult.rescore(batch=)``. ``metrics`` (a family token or a list
    of metric codes, ``evaluate(metrics=)``) restricts the set the winner is
    scored on; ``None`` = the family the batch structure implies (screening
    still needs ARI only).
    """
    def _full(lab, bat, clustering=None):
        # several distinct source files (or a user batch with >1 level) => a
        # real batch structure, so ask for BOTH metric families; else clustering.
        if batch is not None:
            bat = np.asarray(batch)
        grp = "all" if len(set(np.asarray(bat).tolist())) > 1 else "clustering"
        return _evaluate_compat(_evaluate, emb, category=category, labels=lab,
                                verbose=False, batch=(bat if grp == "all" else None),
                                clustering=clustering,
                                metrics=(grp if metrics is None else metrics))

    if len(cands) == 1:
        # nothing to disambiguate - do not pay for a screening pass
        names, lab, bat = cands[0]
        try:
            val = _full(lab, bat)
        except Exception as e:  # noqa: BLE001 - surfaced on the record, not swallowed
            # the single-candidate path used to return nothing, so a bad
            # only=/batch= combination looked like 'no label file matched'
            return names, None, [{"order": names,
                                  "error": f"{type(e).__name__}: {str(e)[:300]}"}]
        return names, val, [{"order": names,
                             "ARI": round(float(val["Value"]["ARI"]), 4)}]

    # Ranking orderings needs ARI and nothing else, and the Leiden sweep ARI rests
    # on depends only on the embedding - not on which label vector it is scored
    # against. So sweep ONCE and reuse it for every candidate, then pay for the
    # full metric set exactly once, on the winner. Both of those were previously
    # per-candidate: on D52 cross that is 6 permutations each running their own
    # 10-resolution sweep over 23,478 cells (~250s apiece), which is what actually
    # timed the cross tutorial out.
    import scib.metrics as _me

    try:
        sweep_adata, sweep_keys = _escib.leiden_sweep(emb)
    except Exception:
        sweep_adata = None

    scored = []
    for names, lab, bat in cands:
        try:
            if sweep_adata is None:      # fall back to a self-contained screen
                val = _evaluate_compat(_evaluate, emb, category=category, verbose=False,
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
        # repeat the sweep a seventh time
        val = _full(lab, bat, clustering=clus)
    except Exception as e:
        raise RuntimeError(
            f"evaluation failed for the winning label order {names}: "
            f"{type(e).__name__}: {e}") from e
    spread = [{"order": n, "ARI": round(a, 4)} for a, n, _, _, _ in scored]
    return names, val, spread


# ------------------------------------------------------------------------ results
def _with_label_order_note(sm: "pd.DataFrame") -> "pd.DataFrame":
    """Attach ``label_order_note`` explaining a blank confidence (docstring
    promise of :attr:`BatchResult.summary`; used by the property AND save())."""
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
    """Outcome of :func:`run_all` - a summary table, a tidy frame and a figure."""

    def __init__(self, records, dataset, category, out_dir=None):
        self.records = records
        self.dataset = dataset
        self.category = category
        self.out_dir = out_dir

    @property
    def summary(self) -> pd.DataFrame:
        """One row per method: ``method, status, run_sec, output_kind, emb_shape,
        n_tunable`` plus one column per metric (``ARI``, ``NMI``, ``ASW``, ...).

        ``status`` is ``CHAIN_OK`` (ran and scored), ``CHAIN_OK_GRAPH_METHOD``
        (scored via a secondary embedding), ``RUN_OK_NO_EMBEDDING`` (ran, but the
        method emits a graph/coordinates so clustering metrics do not apply),
        ``RUN_OK_EVAL_FAILED`` (the method ran and produced an embedding, but every
        candidate label ordering failed to score - usually no label file matches the
        embedding's cell count), ``TIMEOUT`` (exceeded ``run_all(timeout=...)``) or
        ``FAIL`` (the method itself errored; see ``error``).
        ``FAIL``, ``TIMEOUT`` and ``RUN_OK_EVAL_FAILED`` appear in :attr:`failures`.

        Two methods can both be ``output_kind=graph`` and still end differently:
        scMoMaT also writes a UMAP embedding among its ``extra_outputs``, so it is
        scored through that (``CHAIN_OK_GRAPH_METHOD``); Seurat_WNN writes only a
        neighbour graph, so there is nothing to score (``RUN_OK_NO_EMBEDDING``) and
        its ``emb_shape`` is ``None`` - there is no embedding to describe.

        ``batch_source`` / ``n_batches`` say which batch vector the batch metrics
        (ASW_batch, GC, iLISI ...) were computed against: ``'file_of_origin'``
        - each cell's label FILE (``cty1.csv`` -> 1, ``cty2.csv`` -> 2 ...), the
        rule for multi-batch datasets; ``'user'`` - the vector passed as
        ``run_all(batch=)`` / :meth:`rescore` ``batch=``; ``None`` - a single
        label file, so no batch structure and clustering metrics only.
        ``n_batches`` is the number of distinct batch values used (1 = none).

        Two columns describe how the cells were matched to labels:

        ``label_order``
            WHICH label file(s), in which order, the metrics were computed against
            (e.g. ``rna_cty.csv+atac_cty.csv``). For unpaired/diagonal data the
            embedding holds two disjoint cell sets stacked in a method-specific
            order, so this is the difference between a meaningful ARI and a
            meaningless one.
        ``label_order_confidence``
            ``(best - runner_up) / best`` over the candidate orderings, on a 0-1
            scale, or ``None`` when only one ordering was possible (so there was
            nothing to choose).

            **Near 1.0** - every alternative ordering scored near chance, so the
            correspondence is unambiguous and the metrics can be read normally.
            **Below ~0.5** - two orderings explained the embedding comparably well,
            which should not happen for a correct one; treat that row with suspicion.
            The column stays NUMERIC so ``> 0.5`` and ``.isna()`` behave; when it is
            empty, the sibling column ``label_order_note`` says which case applies
            (``"single ordering"`` / ``"winner at chance"`` / ``"not scored"``).

            **``None``** - either only one ordering was possible (normal for a
            paired/vertical dataset with a single ``cty.csv``: there is nothing to
            choose between), or the WINNING
            ordering was itself at chance (ARI < 0.05), in which case the ratio would
            just compare two noise values. A ``None`` next to a near-zero ARI means
            no ordering explained the embedding; the method failed at the task, and
            the ordering machinery has nothing to say about it.

            It is deliberately a RATIO, not a difference. The runner-up sits near
            chance, so a difference is bounded above by the ARI itself and a method
            scoring 0.3 could never look well-separated however unambiguous its
            ordering. The ratio is scale-free.

            .. note::
               When more than one ordering is possible the reported metrics are the
               MAXIMUM over them, so they carry a small optimistic bias. That is the
               price of not making the caller guess the order; this column is how you
               see whether the choice was clear-cut.
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

        This is what :meth:`plot` and ``mtb.plot.bubble`` consume. Derived from
        each record: the unrounded frame ``run_all`` attached (or ``long.csv``
        via :func:`load_batch`) when present, otherwise the record's ``metrics``
        dict - so a result built or reloaded without ``long.csv`` still plots.
        Empty (with the seven columns) if no method produced metrics.
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
        """Raw per-method records: status, out_dir, metrics, and
        ``label_order_candidates`` - every label ordering that was tried, with the
        ARI each achieved (the evidence behind
        :attr:`summary`'s ``label_order_confidence``).

        Keeps a long sweep's outputs addressable so you can re-score or re-plot
        WITHOUT re-running the methods.
        """
        return self.records

    @property
    def failures(self) -> pd.DataFrame:
        """Methods that genuinely went wrong: ``method, status, error``.

        ``run_all`` records failures instead of raising, so ALWAYS check this - a
        sweep can finish "successfully" with several methods having failed.

        ``FAIL``, ``TIMEOUT``, ``RUN_OK_EVAL_FAILED`` and
        ``RUN_OK_NO_LABEL_MATCH`` (ran, but no label file matched the output's
        cell count, so nothing could be scored - usually a data-layout problem
        worth fixing) appear here.
        ``RUN_OK_NO_EMBEDDING`` does NOT: those methods ran correctly and merely
        emit a graph or spatial coordinates instead of an embedding, so there is
        nothing for clustering metrics to score. See :attr:`summary` for them.
        """
        # A method that RAN and simply has no embedding to score is NOT a failure -
        # listing it here sends people hunting for a bug that does not exist.
        bad = [r for r in self.records
               if str(r.get("status", "")).startswith(("FAIL", "TIMEOUT"))
               or r.get("status") in ("RUN_OK_EVAL_FAILED",
                                      "RUN_OK_NO_LABEL_MATCH")]
        if not bad:
            return pd.DataFrame(columns=["method", "status", "error"])
        return pd.DataFrame([{k: r.get(k) for k in ("method", "status", "error")} for r in bad])

    def plot(self, **kw):
        """Bubble figure of every method that produced metrics.

        Methods are rows (best first), metrics are columns; bubble SIZE encodes the
        method's rank - **rank 1 is the LARGEST bubble** - and bubble COLOUR the
        value, darker being higher. Both are relative to the methods in this figure. Read it next to :attr:`summary` - with few methods a small
        absolute gap still spans the whole colour scale.

        Returns a matplotlib ``Figure``; save it with ``fig.savefig("out.png")``.
        Keyword arguments are passed to ``mtb.plot.bubble``: ``metrics=`` (column
        order), ``methods=`` (subset), ``order=`` (row order; unlisted methods
        follow best-first; unknown names raise ``ValueError``), ``title=``,
        ``cmap=``, ``save=``.

        Raises ``ValueError`` if nothing scored - check :attr:`failures` then.
        """
        from . import plot as _plot
        lng = self.long
        if lng.empty:
            raise ValueError("no method produced metrics; see .failures / .summary")
        # No default title: dataset-name banners add nothing a caption
        # cannot say better, and they crowd the page layout. Pass
        # title= explicitly when one is wanted.
        return _plot.bubble(lng, **kw)

    def rescore(self, *, batch=None, labels=None, metrics=None,
                verbose: bool = False) -> "BatchResult":
        """Re-evaluate the STORED outputs with different labels / batch / metrics.

        Nothing is re-run: each record's embedding is read back from its
        ``out_dir`` (the file ``run_all`` loaded) and scored again, so an
        overnight sweep can be re-scored in minutes - e.g. with the batch
        vector the dataset really has instead of the file-of-origin rule,
        or with your own labels. Returns a NEW :class:`BatchResult` (this one is
        untouched; call ``.save(out_dir)`` on the result to persist it -
        ``load_batch`` keeps returning the original until you do).

        Parameters
        ----------
        batch : one batch id per cell in the embedding's row order (array-like,
            Series, or a CSV path read like a label file); ``None`` keeps the
            file-of-origin rule. Recorded as ``batch_source='user'``.
        labels : one cell-type label per cell in embedding order (same forms);
            ``None`` re-runs the label-order search over the dataset's label
            files (``label_order`` / ``label_order_confidence`` are refilled).
        metrics : the metric selection handed to ``evaluate(metrics=)``: a
            family token (``"clustering"`` / ``"batch"`` / ``"all"``) or a
            list of metric codes (``["ARI", "NMI"]``); ``None`` = every
            metric the batch structure allows. (The 0.2 ``only=`` is gone.)
        verbose : print one line per method.

        A record whose output cannot be read back (no embedding - registration
        methods - or a deleted ``out_dir``) keeps its status and gains an
        ``error`` note; ``RUN_OK_EVAL_FAILED`` when the new scoring fails
        (wrong ``batch`` length, say - the error says ``batch has N entries,
        embedding has M cells``).
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

        Produces ``summary.csv``, ``long.csv``, ``failures.csv`` and
        ``batch_result.json``. Reload with :func:`load_batch` to re-score or
        re-plot the next morning WITHOUT re-running any method.
        """
        d = Path(out_dir or self.out_dir or ".")
        d.mkdir(parents=True, exist_ok=True)
        sm = self.summary.copy()
        # "one ordering only" is a RESULT, "never ran" is an absence - a bare NaN
        # cannot tell them apart on disk. Put the explanation in its OWN column:
        # mixing a sentinel string into the numeric one breaks `> 0.5`, makes
        # .isna() miss the very rows it should catch, and trips a pandas
        # incompatible-dtype FutureWarning.
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

    Scoped to what the caller ASKED for: with ``methods=`` every requested
    variant is listed with its own reason (one per line); without it the
    first three blocked variants are shown and the message says how many
    there are in total. Reasons for methods the user never asked for used to
    be listed here (``methods=['Matilda']`` -> Concerto's missing env), which
    sent people fixing the wrong thing.
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
    simply not run); unknown METHOD names are caught earlier by
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
    """Run every method that applies to ``dataset`` under ``category``.

    Parameters
    ----------
    dataset : str
        The DIRECTORY NAME of your data, e.g. ``"MYCITE"`` - not a full path.
        A folder that does not exist raises ``FileNotFoundError`` (listing the
        folders that do) before anything else happens - on the dry run and
        the real run alike, so a typo never reaches the per-method loop. A
        spelling that differs from the folder only in case (``'d52'`` on a
        case-insensitive filesystem) is replaced by the on-disk spelling, with
        a ``UserWarning``, before anything is named after it.
    category : str
        Integration category (``ValueError`` listing the four on a typo).
    out_dir : path, optional
        Where each method's output goes (one sub-directory per method).
        Required for a real run (``TypeError`` otherwise); with
        ``dry_run=True`` it only names the directory the ``command`` column
        is rendered for (default: the literal ``'<out_dir>'`` placeholder).
    methods : list of str, keyword-only, optional
        Restrict to these method ids, as a LIST, e.g. ``["Matilda",
        "totalVI"]`` (default: everything runnable). An unknown id raises
        ``KeyError`` with a did-you-mean hint; a bare string
        (``methods="StabMap"``) raises ``TypeError`` saying to pass a list.
    modalities : list of str, keyword-only, optional
        Restrict to ONE modality combination, given as a list of role
        names, e.g. ``["rna", "adt"]`` for CITE-seq, ``["rna", "atac_gas"]`` for
        RNA + ATAC gene-activity, or ``["rna", "atac_peak"]`` for RNA + ATAC peaks.
        See :func:`describe_layout` for every role name. Default: all combinations.

        .. warning::
           The two ATAC representations do NOT map to the obvious filenames:
           gene-activity is ``atac.h5`` but peaks are ``peak.h5``. Putting a peak
           matrix in ``atac.h5`` runs every method on the wrong representation and
           raises NO error - you simply get confident, wrong numbers.
    params : dict, keyword-only, optional
        Per-method hyperparameters, ``{"Cobolt": {"lr": 1e-3}}``. Discover
        what a method accepts with :func:`multibench.params_for`.
    data_path : path, keyword-only, optional
        The folder that CONTAINS ``dataset``, e.g. ``"/home/wen/data"``
        (so the files live in ``/home/wen/data/MYCITE/``). Defaults to the
        package's configured data root.
    evaluate : bool, keyword-only
        Score every embedding with the benchmark metrics (default True);
        ``False`` runs only (status ``RUN_OK``).
    dry_run : bool, keyword-only
        Return :func:`scan` for the same selection - the identical frame:
        one row per (method, modalities) variant, runnable rows first,
        blocked rows KEPT with their ``reason`` (and the ``files_ok`` /
        ``env_ok`` gate columns) and the ``command`` column (the shell line
        each variant would run, rendered for ``out_dir`` or the literal
        ``'<out_dir>'`` placeholder; ``""`` for a row whose files do not
        resolve) - the same column ``multibench run-all --dry-run --format
        csv`` writes. Never empty: ``ValueError`` if nothing matches. Free;
        do it first. Filter ``plan[plan.runnable]`` for what will actually be
        attempted - ``len(plan)`` is NOT the sweep size; the readable view is
        ``plan[["method", "modalities", "runnable", "reason"]]``. A dry run
        also validates ``params``: a key no planned variant of that method
        accepts raises ``KeyError`` (naming the accepted keys) instead of
        being discovered hours in.
    verbose : bool, keyword-only
        Print progress (``[run_all] ...`` lines; the dry-run summary).
    timeout : float, keyword-only, optional
        Per-method wall-clock cap in SECONDS. Size it from the
        ``runtime_tier`` / ``observed_worst_sec`` columns of :func:`scan` (or
        ``method_info(m)['runtime']``); the slowest methods observed here need
        >4 h. A method exceeding it is recorded as ``TIMEOUT`` and the sweep
        moves on. Strongly recommended for unattended runs - without it a
        single hanging method blocks everything.
    skip_existing : bool, keyword-only
        If a method's output file is already present in ``out_dir``, reuse
        it instead of recomputing. Lets an interrupted overnight sweep
        resume without repeating the hours already done.

        .. warning::
           **``skip_existing=True`` together with ``params=...`` raises
           ``ValueError``.** Reuse is keyed on the output FILE, not on ``params``,
           so without that guard you would silently receive results computed with
           the OLD parameters. When tuning, give each setting a fresh ``out_dir``
           (or leave ``skip_existing`` False).

        .. warning::
           Reuse only checks that the output file EXISTS, not that it is complete.
           A method killed mid-write leaves a truncated file that would be reused
           as if it had succeeded. After a hard kill, delete that method's
           sub-directory before resuming.
    batch : array-like, keyword-only, optional
        One batch id per cell, in the embedding's row order (array-like /
        Series / a CSV path), used for the batch-correction metrics INSTEAD
        of the default rule (batch = the label FILE each cell came from,
        ``cty1.csv`` -> 1 ...). The summary records it as
        ``batch_source='user'``. A vector of the wrong length marks that method
        ``RUN_OK_EVAL_FAILED`` (``batch has N entries, embedding has M cells``).
        You can also re-score a finished sweep with :meth:`BatchResult.rescore`.

    Returns
    -------
    BatchResult or pandas.DataFrame
        The sweep's :class:`BatchResult` (saved under ``out_dir``); the
        :func:`scan` frame when ``dry_run=True``.

    Only methods that :func:`scan` marks runnable are attempted, which means their
    conda environment was found - a missing env is reported there rather than
    failing hours in (``multibench env doctor`` / ``env install --run``).

    Raises ``KeyError`` (did-you-mean) for an unknown id in ``methods`` or
    ``params`` BEFORE any file or env is looked at; ``ValueError`` when no
    variant of the requested methods exists under ``category`` ("no 'cross'
    variant matches ..."), and ``ValueError`` "nothing is runnable ..." when
    variants exist but not one passes both gates - that message lists the
    reason of EVERY requested variant (or the first 3 of N when ``methods`` was
    not given), never the reasons of methods you did not ask for.

    Methods can take minutes to hours; a failure is recorded, never raised, so one
    bad method cannot abort the sweep.
    """
    registry.check_category(category)      # raises with the valid list on a typo
    _list_of_ids(methods, "methods")       # 'StabMap' is not ['S','t',...]
    _list_of_ids(modalities, "modalities")
    # validate the ARGUMENT COMBINATION before anything is resolved or touched,
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
    # the on-disk spelling, decided ONCE here so out_dir names, records and
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
                   # run validates per method and RECORDS a bad override as FAIL
                   params=params if dry_run else None,
                   out_dir=OUT_DIR_PLACEHOLDER if out_dir is None else out_dir)
    if plan_df.empty:
        # only reachable through a modalities= selector that matches nothing:
        # a REQUEST problem, reported as such rather than as "nothing is
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
        # A per-method failure is recorded, never raised - but "not one method could
        # even start" is a different class: the REQUEST is wrong (bad dataset name,
        # wrong category, missing files, missing env). Returning an empty result
        # would report "0 failed", which reads as success and hides a typo. The
        # reasons listed are those of the REQUESTED variants only.
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
               # reproduction provenance: what actually ran, where, with what
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

            # Validate overrides BEFORE dispatching, on every path: an unknown key
            # must fail loudly instead of being dropped from the command line.
            mp = params.get(m)
            if mp:
                allowed = set(v0.tunable) | set(v0.params)
                # NB: an EMPTY allowed-set must reject everything - a method that
                # hardcodes its hyperparameters accepts no overrides at all. The
                # earlier guard skipped exactly that case, which is the one that
                # matters most.
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
                    except Exception as e:      # scoring failed; the RUN itself succeeded
                        rec["status"] = "RUN_OK_EVAL_FAILED"
                        rec["error"] = f"{type(e).__name__}: {e}"
        except TimeoutError as e:
            rec["status"] = "TIMEOUT"
            rec["error"] = str(e)
            rec["run_sec"] = round(time.time() - t0, 1)
        except Exception as e:
            rec["status"] = "FAIL"
            em = f"{type(e).__name__}: {e}"
            # Truncate from the LEFT only. A traceback ends with the actual
            # exception; any right-side cut discards precisely the payload.
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
    """Run ONE method repeatedly over a range of one hyperparameter.

    Answers "did you try changing the learning rate?" without hand-rolling the loop
    - and, importantly, without the two mistakes that loop invites: reusing an
    ``out_dir`` between settings (which silently returns the previous result) and
    losing track of which row came from which value.

    Returns the per-setting metrics with the swept value as a column. A tidy frame
    is written to ``<out_dir>/sweep_long.csv`` (and attached as ``df.attrs["long"]``,
    which does not survive ``to_csv``) in which each setting is a separate series
    (``"Multigrate (lr=0.001)"``), so it can go straight into ``mtb.plot.bubble`` -
    ``.long`` keys rows by method, so without this every setting would collapse onto
    one row::

        df = mtb.sweep("MYDATA", "vertical", "Multigrate", "lr",
                       [1e-4, 1e-3, 1e-2], out_dir="out/lr")
        df[["lr", "ARI", "NMI"]]

    A setting that fails is not fatal: ``run_all`` records it, so that value's row
    appears with ``status`` ``FAIL`` (or ``TIMEOUT``) and empty metrics rather than
    aborting the sweep. Check the ``status`` column before reading the curve - a
    failed setting and a genuinely poor one must not be confused.

    Check :func:`multibench.params_for` first - a method whose ``tunable`` is empty
    hardcodes its hyperparameters and cannot be swept at all.
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
        if not lg.empty:                       # tag the tidy frame too, so the sweep
            lg = lg.copy()                     # is plottable: each setting becomes its
            lg[param] = v                      # own series instead of collapsing onto
            lg["method"] = lg["method"].astype(str) + f" ({param}={v})"
            longs.append(lg)
    out = pd.concat(frames, ignore_index=True)
    lng = pd.concat(longs, ignore_index=True) if longs else pd.DataFrame()
    out.attrs["long"] = lng
    # DataFrame.attrs does NOT survive to_csv, so also write the tidy frame beside
    # the run. An overnight sweep must be re-plottable tomorrow, not only in-process.
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
