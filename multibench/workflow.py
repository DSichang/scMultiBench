"""High-level workflow: point at a dataset, run everything that applies, get metrics and a figure.

The low-level API (``inputs_for`` -> ``run`` -> ``evaluate`` -> ``plot``) requires
a method's name, its integration category and its exact modality combination.
This module works from the dataset instead:

    mtb.scan("D11")                     # what can I run on this data?
    res = mtb.run_all("D11", "vertical", out_dir="out/")  # run all of it, with metrics
    res.plot()                            # one figure

It also handles two common mistakes that give wrong numbers without an error:

* **output kind** - not every method returns an embedding. A method that
  writes a graph is recorded as such, not scored with embedding metrics
  (scoring a KNN index matrix gives ARI ~ 0).
* **label order** - ``evaluate`` needs labels in the embedding's cell order,
  and length alone cannot tell two orders apart. Candidate orders are scored,
  the best is kept, and the scores of all of them are recorded.
"""
from __future__ import annotations

import functools
import glob
import itertools
import json
import os
import re
import shlex
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
from .engine.schema import base_modality as _schema_base
from .engine.runner import run as _run
from .eval import io as _eio, scib as _escib
from .eval.pipeline import evaluate as _evaluate, to_long as _to_long

__all__ = ["scan", "run_all", "BatchResult", "list_categories", "describe_layout",
           "load_batch", "sweep"]



def load_batch(out_dir, *, methods=None, data_path=None) -> "BatchResult":
    """Reload a saved ``run_all`` result.

    Parameters
    ----------
    out_dir : path-like
        Folder holding ``batch_result.json``: a ``run_all`` ``out_dir`` or a
        ``mtb.data.fetch_outputs`` tree.
    methods : list[str] | None
        Methods whose records to keep; ``None`` = every record.
    data_path : path-like | None
        Folder that holds the dataset folder; ``None`` = the path each record saved.

    Returns
    -------
    BatchResult
        The reloaded sweep. It remembers ``out_dir``, so ``save()`` with no
        argument writes back to the same folder.

    Raises
    ------
    FileNotFoundError
        ``out_dir`` holds no ``batch_result.json``.
    ValueError
        ``data_path`` does not hold the dataset folder.
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
    method's unrounded long table; without it ``BatchResult.long`` is rebuilt
    from the records' rounded ``metrics``.

    **Record order.** ``methods=`` only filters: the kept records stay in the
    order the tree ran them, not the order of ``methods``.

    **Moved folders.** A record whose ``out_dir`` does not exist is pointed
    at the folder of the same name next to ``batch_result.json``. The
    dataset folder is looked up under ``data_path=``, the recorded
    ``data_path`` and then ``data_root``, and the one found is recorded as
    an absolute path. So a ``fetch_outputs`` tree or a copied ``run_all``
    folder can be re-scored.

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
    _check_data_path(data_path, blob["dataset"])
    for r in recs:
        # a tree from another host or another folder: the method's output
        # folder sits next to batch_result.json under the same name
        od = r.get("out_dir")
        if od and not Path(od).exists() and (d / Path(od).name).is_dir():
            r["out_dir"] = str((d / Path(od).name).resolve())
        # the first data root that holds the dataset, as an absolute path;
        # left as recorded when none does, and rescore then names the roots
        root, found = _dataset_root(r, blob["dataset"], data_path)
        if found and root is not None:
            r["data_path"] = r["data_root"] = root
    lp = d / "long.csv"
    if lp.exists():
        lng = pd.read_csv(lp)
        for r in recs:
            sub = lng[lng["method"] == r.get("method")]
            r["_long"] = sub if len(sub) else None
    else:
        for r in recs:
            r["_long"] = None
    res = BatchResult(recs, blob["dataset"], blob["category"], out_dir=d)
    for r in recs:
        res._batch_text(r.get("batch_file"))       # the saved batch, when present
    return res

#: The four integration scenarios, and what each one's data looks like.
CATEGORIES = {
    "vertical": "Several modalities measured in the same cells (e.g. CITE-seq "
                "RNA+ADT, or 10x multiome RNA+ATAC). Cells are already matched.",
    "diagonal": "Modalities measured in different cells, with no pairing "
                "(e.g. an RNA experiment and a separate ATAC experiment).",
    "mosaic":   "Several batches where only some share a modality; a paired batch "
                "bridges the others.",
    "cross":    "Several batches, each measured with RNA and ADT, for example one "
                "CITE-seq assay from several donors; the task is removing batch "
                "effects.",
}

#: Modality role -> the file the loader looks for in <data_path>/<dataset>/.
#: One canonical name per role: the file ``mtb.io.export_dataset`` writes.
ROLES = {
    "rna":       "rna.h5         - gene expression",
    "adt":       "adt.h5         - surface protein (CITE-seq antibody-derived tags)",
    "atac":      "atac.h5        - ATAC; method_info(m)['atac'] says peaks or gene activity",
    "atac_peak": "atac_peak.h5   - ATAC as peaks (diagonal)",
    "atac_gas":  "atac_gas.h5    - ATAC as gene-activity scores (diagonal)",
    "rna1/adt1/atac2 ...": "rna1.h5, adt1.h5, atac2.h5 ... - one file per batch (mosaic, cross)",
    "cty":       "cty.csv        - cell-type labels, one per cell (vertical)",
    "rna_cty / atac_cty":
                 "rna_cty.csv, atac_cty.csv - one label file per modality (diagonal)",
    "cty1/cty2 ...":
                 "cty1.csv, cty2.csv ... - one label file per batch (mosaic, cross)",
}

#: Older file names the loader still reads, in one line.
OLDER_NAMES = ("Older names still read: peak.h5 for peaks, and atac.h5 for gene "
               "activity.")


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

    Start here when bringing your own data, then check the folder with
    ``mtb.scan``.

    Parameters
    ----------
    category : str | None
        Integration category to describe; ``None`` = an overview of all four
        and the full file table.

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
    >>> print(mtb.describe_layout("vertical"))  # CITE-seq or multiome
    >>> print(mtb.describe_layout("mosaic"))    # batch patterns methods accept
    >>> print(mtb.describe_layout())            # every category

    Notes
    -----
    **What the text covers.** For one category: the files its methods read,
    one name per file, as ``mtb.io.export_dataset`` writes them. It also
    gives the ATAC representation each method needs, the ``.h5`` and label
    formats, and the install command. For mosaic and cross it lists each
    batch pattern the methods accept. ``multibench layout`` prints the same
    text, with ``multibench`` commands in place of the Python calls.

    **One rule for ATAC files.** Vertical reads ``atac.h5``;
    ``method_info(m)["atac"]`` says whether it must hold peaks or gene
    activity. Diagonal reads ``atac_peak.h5`` (peaks) and ``atac_gas.h5``
    (gene activity). Mosaic reads ``atac<i>.h5`` (peaks). ``peak.h5``, and
    ``atac.h5`` for gene activity, are accepted as older names.

    **Several batches.** Mosaic and cross use one numbered file per batch in
    the same folder, not sub-folders and not one concatenated matrix. The
    file number is the batch; there is no batch column:

    ```text
    <data_path>/COREBATCH/
        rna1.h5   adt1.h5   cty1.csv     # batch 1
        rna2.h5   adt2.h5   cty2.csv     # batch 2
        rna3.h5   adt3.h5   cty3.csv     # batch 3
    ```

    **Source of the lists.** The methods per ATAC representation and the
    batch patterns are read from the package's method list at call time, so they
    agree with ``method_info`` and ``mtb.find_methods``.

    See Also
    --------
    mtb.list_categories : the four categories with a description of each.

    mtb.scan : checks a laid-out folder (``files_ok`` / ``files_reason`` per method).

    mtb.io.export_dataset : writes a whole dataset in this layout from an AnnData.
    """
    registry.check_category(category)       # None passes; typo -> ValueError
    cats = [category] if category else list(CATEGORIES)
    lines = ["Put the files of one dataset in one folder: <data_path>/<DATASET>/, "
             "for example ./data/MYDATA/.",
             "The dataset name is the folder name; data_path is the folder that "
             "contains it.", ""]
    for cat in cats:
        lines += _layout_block(cat, full=category is not None) + [""]
    if category is None:
        lines += ["Every file name the loader reads:"]
        lines += [f"  {k:20s} {v.replace(_METHOD_INFO_ATAC, _check_atac_call())}"
                  for k, v in ROLES.items()]
        lines += [f"  {OLDER_NAMES}", ""]
    cat = category or "<category>"
    lines += ["Each .h5 file holds matrix/data (features x cells, the transpose of "
              "AnnData.X),"]
    lines += [config.hint("matrix/features and matrix/barcodes. mtb.io.export_dataset "
                          "writes the whole folder",
                          "matrix/features and matrix/barcodes. multibench convert "
                          "writes the whole folder"),
              config.hint("from an AnnData or MuData; mtb.io.to_canonical writes one "
                          "file.",
                          "from an .h5ad or .h5mu file, or one file; multibench convert "
                          "--help has examples."),
              "A label file is a single-column CSV: one header line, then one label "
              "per cell.",
              "",
              "Each method runs in its own environment, on Linux. Install one with:",
              "  multibench env install --methods X --packed --run",
              config.hint("mtb.scan", "multibench scan") + " checks the files and the "
              "environment for each method (columns files_ok, env_ok).",
              config.hint(
                  f"Next: mtb.scan('MYDATA', '{cat}'), then, on Linux, "
                  f"mtb.run_all('MYDATA', '{cat}', out_dir='out/')",
                  f"Next: multibench scan MYDATA --category {cat}; on Linux, "
                  f"multibench run-all MYDATA --category {cat} --out-dir out/")]
    return "\n".join(lines)


#: The modality files of the downloadable demo datasets (``mtb.data.fetch``).
#: ``describe_layout`` names the demo that has each batch pattern;
#: tests/test_workflow_layout.py checks this table against the data folders.
DEMO_FILES = {
    "D11": ("rna", "adt"),
    "D28": ("rna", "atac_peak", "atac_gas"),
    "D45": ("rna1", "rna2", "atac2", "atac3"),
    "D46": ("rna1", "rna2", "rna3", "adt1", "atac2"),
    "D52": ("rna1", "rna2", "rna3", "adt1", "adt2", "adt3"),
}


def _demo_for(roles) -> str:
    """The demo dataset whose files match ``roles`` (as a set), or ``""``."""
    from .engine.schema import modality_family
    want = {modality_family(r) for r in roles}
    for ds, files in DEMO_FILES.items():
        if {modality_family(r) for r in files} == want:
            return ds
    return ""


def batch_patterns(category: str) -> list[tuple[tuple, list[str]]]:
    """The batch patterns a category's variants accept, with their methods.

    Returns ``[(roles, methods), ...]``: ``roles`` sorted by batch then
    modality (``('rna1', 'rna2', 'atac2', 'atac3')``), ``methods`` in
    registry order. Variants that read the same set of files share one
    pattern whatever their argument order (Cobolt and MultiVI).
    """
    from .engine.schema import _batch_of, base_modality, modality_family
    order = {"rna": 0, "adt": 1, "atac": 2}
    pats: dict = {}
    for spec, v, _cat, mods in _variant_rows(category):
        if not mods:
            continue
        key = tuple(sorted((modality_family(m) for m in mods),
                           key=lambda r: (_batch_of(r) or 0,
                                          order.get(base_modality(r), 9), r)))
        ids = pats.setdefault(key, [])
        if spec.id not in ids:
            ids.append(spec.id)
    return list(pats.items())


def _atac_lines(category: str) -> list[str]:
    """The ATAC part of one category's layout: which representation each
    method needs, from the registry (``find_methods(category, atac=...)``)."""
    from .discover import find_methods as _find
    peak = _find(category, atac="peak")
    gas = _find(category, atac="gene_activity")
    if not peak and not gas:
        return []
    out = []
    if category == "vertical":
        out.append("atac.h5 holds peaks or gene activity; each method needs one of them:")
    elif category == "diagonal":
        out.append("Give atac_peak.h5, atac_gas.h5 or both. Each method needs one of "
                   "them, or both:")
        both = list(dict.fromkeys(s.id for s, v, _c, mods in _variant_rows("diagonal")
                                  if {"atac_peak", "atac_gas"} <= set(mods)))
        if both:
            out.append(f"  need both files:       {', '.join(both)}")
        # each method on one line: the ones that need both files are not
        # listed again under the representation they also read
        peak = [m for m in peak if m not in both]
        gas = [m for m in gas if m not in both]
    elif gas:
        out.append("atac<i>.h5 holds peaks or gene activity; each method needs one of them:")
    else:
        out.append(f"atac<i>.h5 holds peaks: every {category} method that reads ATAC "
                   f"needs peaks.")
    if category in ("vertical", "diagonal") or gas:
        if peak:
            out.append(f"  need peaks:            {', '.join(peak)}")
        if gas:
            out.append(f"  need gene activity:    {', '.join(gas)}")
    out += ["A method whose ATAC file holds the other representation gives a wrong "
            "embedding."]
    # method_info(m)['atac'] names one representation, so it cannot tell a
    # method that reads both files: point to the list printed above instead
    check = ("the list above" if category == "diagonal" else _check_atac_call())
    out += config.hint(
        ["mtb.scan and mtb.run_all skip such a method unless allow_atac_mismatch=True.",
         f"mtb.run only warns, so check {check} first."],
        ["multibench scan and run-all skip such a method unless --allow-atac-mismatch "
         "is given.",
         f"multibench run only warns, so check {check} first."])
    if category == "diagonal":
        out.append(OLDER_NAMES)
    return out


#: The call that says which ATAC representation a method reads, as the
#: layout text names it; the command line shows ``multibench info``.
_METHOD_INFO_ATAC = "method_info(m)['atac']"


def _check_atac_call() -> str:
    return config.hint(_METHOD_INFO_ATAC, "multibench info METHOD")


#: How ``describe_layout`` shows a batch's modalities as ``convert`` flags.
#: A ``.h5mu`` batch keeps its labels in the RNA modality's obs (muon's
#: usual layout), read with the documented ``<mod>:<col>`` selector.
_CONVERT_FLAGS = {
    ("rna",): ("h5ad", "--rna X"),
    ("adt", "rna"): ("h5ad", "--rna X --adt obsm:protein"),
    ("atac", "rna"): ("h5mu", "--rna mod:rna --atac mod:atac --atac-kind peak"),
}
_CONVERT_LABELS = {"h5ad": "obs:cell_type", "h5mu": "rna:cell_type"}


def _batch_recipe(category: str, patterns) -> list[str]:
    """One ``multibench convert`` line per batch of the most varied pattern,
    plus the ``export_dataset`` form of the first batch."""
    from .engine.schema import _batch_of, base_modality

    def per_batch(roles):
        per: dict = {}
        for r in roles:
            per.setdefault(_batch_of(r), set()).add(base_modality(r))
        return {b: tuple(sorted(ms)) for b, ms in per.items()}

    usable = [p for p, _ in patterns
              if all(ms in _CONVERT_FLAGS for ms in per_batch(p).values())]
    if not usable:
        return []
    batches = per_batch(max(usable, key=lambda p: len(set(per_batch(p).values()))))
    lines = ["Write one file per batch, numbered to match the pattern:"]
    same = len(set(batches.values())) == 1
    for b, ms in batches.items():
        ext, flags = _CONVERT_FLAGS[ms]
        lines.append(f"  multibench convert batch{b}.{ext} data/MYDATA {flags} "
                     f"--labels {_CONVERT_LABELS[ext]} --category {category} "
                     f"--batch-index {b}")
        if same:
            rest = [str(k) for k in batches if k != b]
            if rest:
                lines.append(f"  (the same for batch{'es' if len(rest) > 1 else ''} "
                             f"{' and '.join(rest)}, with --batch-index "
                             f"{' and '.join(rest)})")
            break
    if any(_CONVERT_FLAGS[ms][0] == "h5mu" for ms in batches.values()):
        lines.append("  In a .h5mu file, rna:cell_type reads the RNA modality's obs; "
                     "obs:<col> reads the global obs.")
    if not config._CLI:
        first = next(iter(batches.values()))
        kw = "rna='X', adt='obsm:protein'" if "adt" in first else "rna='X'"
        lines.append(f"  in Python: mtb.io.export_dataset(adata, 'data/MYDATA', {kw}, "
                     f"labels='obs:cell_type', category='{category}', batch_index=1)")
    return lines


def _layout_block(category: str, *, full: bool) -> list[str]:
    """The lines that describe one category's folder (``full`` adds the ATAC
    and batch-pattern details that ``describe_layout(category)`` prints)."""
    from .engine.schema import _batch_of, base_modality
    head = f"{category}: {CATEGORIES[category]}"
    if category == "vertical":
        demo = _demo_for(["rna", "adt"])
        files = ["  rna.h5 + adt.h5    CITE-seq" + (f" (demo {demo})" if demo else ""),
                 "  rna.h5 + atac.h5   multiome",
                 "  cty.csv            cell-type labels, one per cell"]
    elif category == "diagonal":
        demo = _demo_for(["rna", "atac_peak", "atac_gas"])
        files = ["  rna.h5             the RNA cells" + (f" (demo {demo})" if demo else ""),
                 "  atac_peak.h5       the ATAC cells, as peaks",
                 "  atac_gas.h5        the same ATAC cells, as gene-activity scores",
                 "  rna_cty.csv        labels of the RNA cells",
                 "  atac_cty.csv       labels of the ATAC cells"]
    else:
        seen = [r for roles, _ in batch_patterns(category) for r in roles]
        example = ", ".join(f"{r}.h5" for r in list(dict.fromkeys(seen))[:4])
        files = [f"  One numbered file per batch and modality ({example} ...)",
                 "  and one label file per batch (cty1.csv, cty2.csv ...), in the same "
                 "folder."]
    if not full:
        return [head] + files
    lines = [head] + files
    if category in ("mosaic", "cross"):
        patterns = batch_patterns(category)
        lines.append("Each method accepts one batch pattern. Number your batches to "
                     "match one of them:")
        for roles, ids in patterns:
            per: dict = {}
            for r in roles:
                per.setdefault(_batch_of(r), []).append(base_modality(r))
            desc = ", ".join(f"{b} = {'+'.join(ms)}" for b, ms in per.items())
            demo = _demo_for(roles)
            lines.append(f"  batch {desc}: {', '.join(ids)}"
                         + (f" (demo {demo})" if demo else ""))
        lines += _batch_recipe(category, patterns)
    lines += _atac_lines(category)
    return lines


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

    A checkout at another commit than ``$MULTIBENCH_SCRIPTS_REF`` is not a
    file problem: ``scan`` blocks every row with its own reason.

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
                        f"{root}: " + _runner.missing_script_fix(root))
            gone = [h for h in (getattr(variant, "helpers", None) or [])
                    if not (root / ep).parent.joinpath(h).exists()]
            if gone:
                who = config.hint(
                    (f"mtb.method_info({method!r})" if method else "method_info(m)")
                    + "['setup_hint']",
                    f"multibench info {method or 'METHOD'}")
                name = f"{method}'s script" if method else f"The script {ep.name}"
                files = " and ".join(gone)
                one = len(gone) == 1
                return (f"{name} imports {files}. The public scMultiBench repository "
                        f"does not include {'it' if one else 'them'}. Put "
                        f"{'a ' if one else ''}{files} next to {ep.name}. {who} shows "
                        f"how.")
            return ""
    return ""            # no checkout yet: run()/run_all() fetch one


def _join_clauses(parts) -> str:
    """Join the ``caveat`` parts as sentences (:func:`_join_sentences`)."""
    return _join_sentences(parts)


def _rows_word(df: "pd.DataFrame", k: int) -> str:
    """What ``k`` counts in a count line over the scan frame ``df``:
    ``methods`` when each method has one row, else ``rows`` (singular for 1)."""
    noun = "method" if len(df) and df["method"].is_unique else "row"
    return noun if k == 1 else noun + "s"


def _have_their(k: int, n: int) -> str:
    """The verb after ``k of n <rows>``: ``has its`` when ``k`` or ``n`` is 1
    (``0 of 1 method has its``), else ``have their`` (``3 of 14 rows have their``)."""
    return "has its" if 1 in (k, n) else "have their"


def _join_sentences(parts) -> str:
    """Join the ``reason`` parts as sentences, each ending with a period."""
    parts = [p.strip() for p in parts if p and p.strip()]
    return " ".join(p if p[-1] in ".!?" else p + "." for p in parts)


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
    and then the category-wide alternative. The command comes first, so a
    clipped reason still holds it.

    Off Linux the install command refuses, so the reason only names the
    environment and says it cannot be installed here; the scan summary line
    says what this computer can do instead.
    """
    if _runner.linux_only_sentence():
        return f"Environment {env} runs only on Linux, not on this computer."
    alt = (f" Use --category {category} to install the environments of every "
           f"{category} method." if category else "")
    return (f"Environment {env} is not installed. Run "
            f"multibench env install --methods {method} --packed --run.{alt}")


#: ``.format(method=)`` caveat of a GPU-only row that ``scan(assume_gpu=True)``
#: keeps runnable.
GPU_NODE_CAVEAT = ("{method} needs an NVIDIA GPU. This check assumes the job runs on a "
                   "GPU node.")

#: What the scan summary line adds off Linux, after the counts.
LINUX_ONLY_SUMMARY = ("Method environments run only on Linux. On this computer you can "
                      "check files, score embeddings and plot. The commands use this "
                      "computer's paths, so run scan again on the Linux machine.")


def _first_sentence(text: str) -> str:
    """The first sentence of ``text`` (up to a '.', '!' or '?' followed by a
    space or the end); a dotted file name does not end it."""
    m = re.match(r"(.+?[.!?])(\s|$)", text.strip(), re.S)
    return (m.group(1) if m else text).strip()


def _atac_kind_of(path: Path) -> str:
    """What an ATAC-family file holds, from its name, else from its features."""
    stem = path.stem.rstrip("0123456789")
    if stem in ("atac_peak", "peak"):
        return "peaks"
    if stem == "atac_gas":
        return "gene activity"
    frac = _resolve._peak_fraction_of(path)
    if frac is not None and frac >= 0.9:
        return "peaks"
    if frac is not None and frac <= 0.1:
        return "gene activity"
    return "ATAC"


def _missing_files_reason(spec, variant, category: str, mods: list, dataset: str,
                          data_path) -> str:
    """The ``reason`` text for input files that are not on disk, or ``""``.

    Built from the resolved paths, not from the exception text, so no path
    is ever cut. An ATAC file leads with what the method needs and what the
    folder holds instead: ``UnitedNet needs gene-activity ATAC (atac_gas.h5),
    and the folder has peaks (atac_peak.h5).`` Other files follow as
    ``adt.h5 is missing.``, then the per-batch hint of ``inputs_for`` when
    the folder holds ``rna1.h5, rna2.h5, ...`` for a vertical or diagonal
    method. Each part is a sentence with its own subject.
    """
    try:
        paths = _resolve.inputs_for(dataset, category, spec.id, modalities=mods or None,
                                    data_path=data_path, check=False)
    except Exception:           # noqa: BLE001 - fall back to the exception text
        return ""
    from .engine.schema import is_label_role
    missing = {r: Path(p) for r, p in paths.items() if not Path(p).exists()}
    if not missing:
        return ""
    atac_parts, other = [], []
    for role, p in missing.items():
        if is_label_role(role) or _schema_base(role) != "atac":
            other.append(p.name)
            continue
        digits = role[len(role.rstrip("0123456789")):]
        stem = role.rstrip("0123456789")
        want_file = f"atac{digits}.h5" if category == "vertical" else p.name
        if stem == "atac_peak":
            want = "peak"
        elif stem == "atac_gas" and category != "vertical":
            want = "gene-activity"
        else:
            want = {"peak": "peak", "gene_activity": "gene-activity"}.get(spec.atac or "", "")
        need = f"{spec.id} needs {want + ' ' if want else ''}ATAC ({want_file})"
        found = [p.parent / f"{b}{digits}.h5" for b in ("atac", "atac_peak", "atac_gas", "peak")]
        found = [f for f in found if f.is_file() and f.name != want_file]
        # the right representation under a diagonal name: only the name is wrong
        same = [f for f in found if category == "vertical" and want
                and _atac_kind_of(f) == {"peak": "peaks",
                                         "gene-activity": "gene activity"}[want]]
        if same:
            atac_parts.append(
                f"{spec.id} reads {want_file} for vertical. Rename {same[0].name} to "
                f"{want_file}, or write it with " + config.hint('category="vertical"',
                                                                "--category vertical"))
        elif found:
            has = ", ".join(f"{_atac_kind_of(f)} ({f.name})" for f in found)
            atac_parts.append(f"{need}, and the folder has {has}")
        else:
            atac_parts.append(f"{need}, which is not in the folder")
    if other:
        atac_parts.append(", ".join(other) + (" is" if len(other) == 1 else " are")
                          + " missing")
    # rna1.h5 + rna2.h5 where a vertical / diagonal method reads one rna.h5
    ds_dir = next(iter(missing.values())).parent
    batch_hint = _resolve._per_batch_hint(
        ds_dir, category, {st for r in missing for st in _resolve._role_stems(r)[0]})
    if batch_hint:
        atac_parts.append(batch_hint[:1].upper() + batch_hint[1:])
    return _join_sentences(atac_parts)


_ABS_PATH_RE = re.compile(r"(?<![\w./-])/(?:[^\s'\"\[\]{}(),:;]+/)+[^\s'\"\[\]{}(),:;]*")
_EXC_PREFIX_RE = re.compile(r"^[A-Z]\w*(?:Error|Exception|Warning): ")


def _dropped_dirs_message(modalities, dropped: list) -> str:
    """The warning of ``scan(modalities=...)`` for the methods that read a
    folder (scBridge): no modality token selects them."""
    one = len(dropped) == 1
    mods = _resolve._and_list([str(m) for m in modalities])
    head = (f"The modality {mods} leaves" if len(modalities) == 1
            else f"The modalities {mods} leave")
    fix = config.hint(
        f"Pass modalities=[] to select {'it' if one else 'them'}, or leave out "
        f"modalities= to see every variant.",
        f'Pass --modalities "" to select {"it" if one else "them"}, or leave out '
        f"--modalities to see every variant.")
    return (f"{head} out {_resolve._and_list(dropped)}, which "
            f"{'reads' if one else 'read'} a folder instead of modality files. {fix}")


def _short_reason(text: str, method: str, dataset: str, category: str | None) -> str:
    """The ``reason`` column form of a ``files_reason``: what is missing, no noise.

    ``text`` is one part of ``files_reason``: a missing-script sentence, or
    the verbatim exception text (``FileNotFoundError: SCALEX (diagonal)
    needs atac_gas.h5 in /path/to/data/LUNG. ...``), which keeps the full
    path because that is what a user greps for. ``reason`` is what the scan
    frame, the CLI table and the "No method can run" error show, so it drops
    what every row repeats: the exception class, the absolute directory
    (each path becomes its basename) and the ``<method> reads <file> of
    <dataset>, which`` opening of a cell check, which leaves the file as the
    subject (``atac_gas.h5 lists the ATAC cells in another order than
    atac_peak.h5. ...``). The 80-character column then keeps the fact that
    differs. The env half of ``reason`` is untouched - it carries the
    copy-pasteable install command.
    """
    if not text:
        return text
    part = _EXC_PREFIX_RE.sub("", text)
    part = re.sub(rf"^{re.escape(method)} reads (\S+) of {re.escape(str(dataset))}, which ",
                  r"\1 ", part)
    return _ABS_PATH_RE.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], part)


# Reject a bare string where a list of ids is expected (shared with mtb.env.*).
_list_of_ids = registry.check_id_list


#: The three representation-mismatch caveats of ``_resolve._preflight_caveats``
#: (PEAK_IN_GAS / PEAK_FED_TO_GAS / GAS_FED_TO_PEAK) after the method name:
#: wanted, file, held.
_WRONG_ATAC_BODY = (r"needs (gene-activity|peak) ATAC\. The features of (\S+) (?:do not )?"
                    r"look like chr:start-end, so it (?:seems to hold|holds) "
                    r"(peaks|gene activity)\.")
_WRONG_ATAC_RE = re.compile(r"^\S+ " + _WRONG_ATAC_BODY)
#: What the reason and the caveat of a peak-name row say after the method name
#: (``cli._strict_problem`` counts them apart): ``_resolve.PEAK_NAMES_CAVEAT``,
#: the peak file of a method whose peak names ``mtb.run`` rewrites holds names
#: the rewrite cannot turn into chr:start-end.
PEAK_NAMES_REASON = "reads peak names such as "
_PEAK_NAMES_RE = re.compile(r"^\S+ " + re.escape(PEAK_NAMES_REASON)
                            + r"(\S+)\. (\S+) holds other names")
#: The override every blocking-ATAC reason names, in its Python and CLI
#: spellings: the marker :func:`_is_wrong_atac` looks for, whatever the prose.
_ATAC_OVERRIDE = ("allow_atac_mismatch=True", "--allow-atac-mismatch")


def _blocking_caveats(caveats) -> list[str]:
    """The caveats that block a row: the ATAC file holds the other
    representation, or peak names the method's script cannot read."""
    return [c for c in caveats if _WRONG_ATAC_RE.match(c) or _PEAK_NAMES_RE.match(c)]


def _wrong_atac_reason(method: str, caveats) -> str:
    """The ``reason`` of a row whose ATAC file the method cannot read, or "".

    Two caveats block: the file holds the other representation, or the peak
    file of GLUE / Seurat_v3 holds names that are not chr:start-end (the
    script casts their parts to int). ``scan`` marks such a row not runnable
    unless ``allow_atac_mismatch=True``, so ``run_all`` skips it by default
    (the one rule of both). The reason gives the fix first, the override last.
    """
    anyway = (", or pass " + config.hint(_ATAC_OVERRIDE[0], _ATAC_OVERRIDE[1])
              + f" to run {method} anyway.")
    for text in _blocking_caveats(caveats):
        m = _WRONG_ATAC_RE.match(text)
        if m:
            as_ = "gene activity" if m.group(1) == "gene-activity" else "peaks"
            return (f"{method} needs {m.group(1)} ATAC, and {m.group(2)} holds "
                    f"{m.group(3)}. Export the ATAC as {as_}" + anyway)
        # the caveat starts with the method and ends with the fix (rename
        # them); the override follows it
        return text.rstrip(".") + anyway
    return ""


def _is_wrong_atac(reason) -> "pd.Series | bool":
    """Whether a scan ``reason`` (or a Series of them) carries :func:`_wrong_atac_reason`."""
    if isinstance(reason, pd.Series):
        text = reason.astype(str)
        return text.str.contains(_ATAC_OVERRIDE[0], regex=False) | text.str.contains(
            _ATAC_OVERRIDE[1], regex=False)
    return any(o in str(reason or "") for o in _ATAC_OVERRIDE)


def _atac_mismatch_caveats(method: str, category: str, inputs) -> list[str]:
    """The blocking caveats (:func:`_blocking_caveats`) of the files in a
    ``run`` call's ``inputs``; ``[]`` when there are none or they cannot be
    read. In-memory inputs are not checked."""
    try:
        spec = registry.get(method)
        paths = {r: str(p) for r, p in inputs.items()
                 if isinstance(p, (str, os.PathLike)) and Path(p).exists()}
        return _blocking_caveats(_resolve._preflight_caveats(
            paths, atac=spec.atac, category=category, method=method))
    except Exception:  # noqa: BLE001 - a check must never stop the run
        return []


#: How the note that the method scripts are not here begins
#: (``runner.SCRIPTS_NOT_HERE``); scan appends it, then ``runner._prepared_note``
#: (found by ``runner._prepared_at``), after every other sentence.
_START_NOTES = (_runner.SCRIPTS_NOT_HERE,)


def _run_caveat(caveat) -> str:
    """A scan ``caveat`` without the notes on starting the method, which
    ``run_all`` does itself: what its log and records keep of the caveat."""
    text = str(caveat or "")
    cuts = [i for i in (text.find(_START_NOTES[0]), _runner._prepared_at(text)) if i >= 0]
    return text[:min(cuts)].rstrip() if cuts else text


def _is_wrong_ref(reason) -> "pd.Series | bool":
    """Whether a scan ``reason`` (or a Series of them) says the method scripts
    are not at ``$MULTIBENCH_SCRIPTS_REF`` (``config.scripts_ref_problem``)."""
    tag = f"({config.SCRIPTS_REF_VAR})"
    if isinstance(reason, pd.Series):
        return reason.astype(str).str.contains(tag, regex=False)
    return tag in str(reason or "")


def _would_run(row) -> bool:
    """Whether the sweep would run this scan row: runnable, or blocked only by its env."""
    return bool(row["files_ok"]) and (not row["reason"]
                                      or row["reason"] == row.get("env_reason"))


def _scripts_ref_note() -> str | None:
    """The sentence a dry run prints under its count line when the method
    scripts are not at ``$MULTIBENCH_SCRIPTS_REF``, or their folder cannot be
    filled; ``None`` otherwise."""
    note = config.scripts_ref_problem() or config.scripts_folder_problem()
    return _join_sentences([note]) if note else None


def _dry_run_notes(plan: "pd.DataFrame") -> tuple:
    """What a dry run prints under its count line: ``(scripts_note, [(method, caveat)])``.

    ``scripts_note`` is the note that the method scripts are not on this
    machine yet, once, or ``None``. The list holds the caveat of each row
    the sweep would run - runnable, or blocked only by its env - without the
    notes on starting the method (:func:`_run_caveat`); rows with nothing
    left are not listed.
    """
    scripts = None
    lines = []
    for _, r in plan.iterrows():
        text = str(r.get("caveat") or "")
        i = text.find(_START_NOTES[0])
        if scripts is None and i >= 0:
            j = _runner._prepared_at(text[i:])
            scripts = text[i:][:j].rstrip() if j >= 0 else text[i:]
        cav = _run_caveat(text)
        if _would_run(r) and cav:
            lines.append((r["method"], cav))
    return scripts, lines


def _variant_consumes_atac(variant) -> bool:
    """Whether this variant takes an ATAC input.

    Judged on the declared modalities, not on every argument: Seurat_WNN's
    rna+adt variant passes its unused ATAC slot as a constant ``NULL``. A
    variant fed a folder (scBridge) declares none; its constant file names
    decide.
    """
    mods = variant.when.get("modalities") or []
    if mods:
        return any(_schema_base(m) == "atac" for m in mods)
    return any(a.const and "atac" in str(a.const) for a in variant.args)


def _modality_matcher(modalities, named=None):
    """A test ``(spec, variant_modalities) -> bool`` for ``scan(modalities=)``.

    A method in ``named`` (``scan(methods=)``) keeps a variant whose roles
    the list spells exactly as ``scan`` and ``method_info`` show them.
    Otherwise one representation token (``atac_peak`` / ``peak``,
    ``atac_gas`` / ``gas`` / ``gene_activity``) first drops every variant
    whose ``method_info(m)['atac']`` is the other representation, as
    ``find_methods(atac=...)`` does: moETM, scMM and iPOLNG read peaks
    through a role named ``atac_gas``. Then a list that names a variant's
    roles exactly (a scan row's ``modalities`` split on ``+``) keeps that
    variant. Otherwise the tokens name one
    combination: a base token (``rna``, ``adt`` / ``protein``, ``atac``)
    matches every role of that base (``atac`` matches ``atac``, ``atac_gas``,
    ``atac_peak`` and the numbered ``atac2``); a representation token
    (``atac_peak`` / ``peak``, ``atac_gas`` / ``gene_activity``) matches the
    ATAC roles of a method whose ``method_info(m)['atac']`` is that
    representation; a numbered token (``rna1``) matches that role. A variant
    matches when every token matches one of its roles and every role is
    matched. Both representation tokens together therefore keep only the
    variants with both roles (MultiMAP, Seurat_v3). ``[]`` matches only the
    variants fed a folder. Unknown tokens raise ``ValueError`` as in
    ``find_methods``.
    """
    from .discover import _modality_filter
    from .engine.schema import modality_family
    # find_methods' checks: ValueError on an unknown token, TypeError on a
    # bare string
    _modality_filter(modalities, None)
    exact = set(registry.normalize_modalities(modalities))
    spelled = {str(t).lower() for t in modalities}
    named = set(named or ())
    toks = []
    for tok in modalities:
        t = registry.MODALITY_ALIASES.get(str(tok).lower(), str(tok))
        stem = t.rstrip("0123456789")
        rep = {"atac_peak": "peak", "atac_gas": "gene_activity"}.get(stem)
        fam = modality_family(t)
        toks.append((fam, rep, fam in ("rna", "adt", "atac")))
    reps = _token_reps(modalities)

    def covers(tok, role):
        fam, _rep, is_base = tok
        return (_schema_base(role) == fam) if is_base else modality_family(role) == fam

    def match(spec, mods) -> bool:
        if not mods or not toks:
            return not mods and not toks
        # a named method keeps the row whose roles are spelled as scan shows them
        if spec.id in named and spelled == set(mods):
            return True
        # the representation before the spelling: moETM's atac_gas role reads peaks
        if len(reps) == 1 and spec.atac not in reps:
            return False
        if exact == set(mods):
            return True
        if any(rep and spec.atac != rep for _f, rep, _b in toks):
            return False
        return (all(any(covers(t, r) for r in mods) for t in toks)
                and all(any(covers(t, r) for t in toks) for r in mods))
    return match


#: representation token stem -> ``method_info(m)['atac']``, and back
_REP_OF_TOKEN = {"atac_peak": "peak", "atac_gas": "gene_activity"}
_TOKEN_OF_REP = {v: k for k, v in _REP_OF_TOKEN.items()}
_REP_WORDS = {"peak": "peaks", "gene_activity": "gene activity"}


def _token_reps(modalities) -> set:
    """The ATAC representations (``peak``, ``gene_activity``) the tokens name,
    as :func:`_modality_matcher` reads them; empty for ``atac`` or no ATAC token."""
    out = set()
    for tok in modalities or ():
        t = registry.MODALITY_ALIASES.get(str(tok).lower(), str(tok))
        rep = _REP_OF_TOKEN.get(t.rstrip("0123456789"))
        if rep:
            out.add(rep)
    return out


def _representation_note(category, methods, modalities) -> str:
    """Why a representation token dropped every row of the named ``methods``, or "".

    For each named method that the list would select with ``atac`` in place
    of the representation token: what it reads, the role named after the
    other representation (moETM's ``atac_gas``), and the list to pass.
    """
    if not methods or not modalities:
        return ""
    toks = [registry.MODALITY_ALIASES.get(str(t).lower(), str(t).lower())
            for t in modalities]

    def rep(tok):
        return _REP_OF_TOKEN.get(tok.rstrip("0123456789"))
    reps = {rep(t) for t in toks if rep(t)}
    if len(reps) != 1:
        return ""
    loose = _modality_matcher(["atac" if rep(t) else t for t in toks])
    notes, seen = [], set()
    for spec, _v, _cat, mods in _variant_rows(category):
        if (spec.id not in methods or spec.id in seen or not spec.atac
                or spec.atac in reps or not loose(spec, mods)):
            continue
        seen.add(spec.id)
        other = next((r for r in mods if rep(r) and rep(r) != spec.atac), None)
        via = f" through its {other} input" if other else ""
        fixed = [_TOKEN_OF_REP[spec.atac] if rep(t) else t for t in toks]
        base = ["atac" if rep(t) else t for t in toks]
        notes.append(f"{spec.id} reads {_REP_WORDS[spec.atac]}{via}. Pass " + config.hint(
            f"modalities={fixed!r} or {base!r}.",
            f"--modalities {','.join(fixed)} or {','.join(base)}."))
    return " ".join(notes)


def _no_variant_error(category, dataset, methods, modalities) -> ValueError:
    """The ``ValueError`` of a selection that matches no variant: what does
    not run, then why (a representation token) or where to look."""
    from .engine.schema import no_category_message
    from .plot.bubble import _and
    where = f"{category} data" if category else "any category"
    if modalities is None:
        # the named methods have no variant under this category
        head = " ".join(no_category_message(
            m, dict.fromkeys(c for _s, _v, c, _m in _variant_rows()
                             if _s.id == m and c), category) for m in methods or ()
        ) or f"No method runs on {where}."
    else:
        mods = "+".join(map(str, modalities or ())) or "a data folder"
        head = (f"{_and(methods)} {'does' if len(methods) == 1 else 'do'} not read "
                f"{mods} in {where}." if methods else f"No method reads {mods} in {where}.")
    why = _representation_note(category, methods, modalities)
    one = methods[0] if methods and len(methods) == 1 else None
    fix = "" if modalities is None else config.hint(
        f"mtb.method_info({one!r})['supports'] lists what {one} reads." if one else
        "mtb.method_info(m)['supports'] lists what each method reads.",
        f"multibench info {one} lists what {one} reads." if one else
        "multibench info METHOD lists what each method reads.")
    err = ValueError(" ".join(t for t in (head, why or fix) if t))
    err.representation = bool(why)      # the CLI keeps this message as it is
    return err


def _command_line(method: str, category: str, inputs: dict, *, out_dir, dataset: str,
                  params: dict | None, gpu: bool | None = None) -> tuple[str, list[str]]:
    """The shell line ``run`` would execute for one scan row (``shlex``-joined),
    and the note when that line reads files ``run`` writes first.

    ``(no preview: ...)`` when building it failed - a preview must never
    abort the scan.
    """
    import shlex
    try:
        # the runner's preview, not the module-level ``_run`` hook the
        # dispatch tests replace: a preview must never count as a dispatch.
        # Its setup notes are not used here; scan builds its own caveat.
        argv, notes = _runner.preview(method, category, inputs=inputs,
                                      out_dir=Path(out_dir) / f"{method}_{dataset}",
                                      params=params, gpu=gpu)
        prepared = [n for n in notes if _runner._prepared_at(n) == 0]
        return shlex.join(argv), prepared
    except Exception as e:  # noqa: BLE001 - a preview must never abort the scan
        return f"(no preview: {type(e).__name__}: {e})", []


def scan(dataset: str, category: str | None = None, *,
         methods: list[str] | None = None,
         modalities: list[str] | None = None,
         data_path: Path | str | None = None,
         out_dir="<out_dir>",
         params: dict | None = None,
         verbose: bool = True,
         assume_gpu: bool = False,
         allow_atac_mismatch: bool = False) -> pd.DataFrame:
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
        Print one line: how many rows have their input files and their
        environment.
    assume_gpu : bool
        ``True`` = skip this host's GPU test; for a login node without a GPU
        that checks a GPU-node job.
    allow_atac_mismatch : bool
        ``True`` = keep a row runnable when its ATAC file holds the other
        representation or peak names the method cannot read.

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
        Unknown ``category`` or modality token, or no variant of ``methods``
        exists under ``category``.
    KeyError
        Unknown id in ``methods`` or ``params``, or a ``params`` key no variant accepts.
    TypeError
        ``methods`` or ``modalities`` given as a bare string.

    Warns
    -----
    UserWarning
        ``dataset`` matches a folder only up to letter case.
    UserWarning
        ``modalities`` drops a folder-fed method whose ATAC representation it allows.

    Examples
    --------
    >>> import multibench as mtb
    >>> df = mtb.scan("D11", "vertical")
    >>> df[["method", "modalities", "runnable", "reason"]]
    >>> # what blocks the rest
    >>> df.loc[~df.runnable, ["method", "files_reason", "env_reason"]]
    >>> print(df.loc[df.files_ok, "command"].iloc[0])  # the line the run executes
    >>> mtb.scan("MYCITE", "vertical", modalities=["rna", "adt"],
    ...          data_path="/path/to/data")

    Notes
    -----
    **Column reference.** The full frame has 18 columns:

    ```text
    method              method id
    category            integration category of the variant
    modalities          '+'-joined string ("rna+adt"); "(data_dir)" for a
                        directory-fed variant
    env                 the conda env the method runs in
    output_kind         embedding / graph
    n_tunable           number of command-line hyperparameters
    runtime_tier        fast / medium / slow / very_slow / unknown
    observed_worst_sec  the slowest observed run, seconds (None = unmeasured)
    caveat              what the run needs besides the files, or ""
    runnable            both checks pass and nothing below blocks it
    reason              short form of the non-empty reasons, as sentences
    files_ok            the inputs resolve, are oriented and labelled
    files_reason        full file-check text, full paths
    env_ok              the env exists (and a GPU, when the script needs one)
    env_reason          full env-check text
    needs_labels        this variant demands a label file as an input
    atac                ATAC representation the method expects: 'peak' /
                        'gene_activity'; None when the variant takes no ATAC
    command             the shell line the variant would run; "" if the
                        inputs do not resolve
    ```

    **Two checks.** Every row carries two independent checks, each a flag
    plus a reason; ``runnable`` needs both:

    - ``files_ok`` / ``files_reason`` - the method's script is present, the
      input files resolve on disk and are oriented features x cells, every
      label CSV has one row per cell of the modality it labels, and a
      ``data_dir`` method (scBridge) finds the files it names. Seurat_v5's
      ``rna.h5`` and ``atac_peak.h5`` must hold the same cells. A diagonal
      ``atac_gas.h5`` must list the cells of ``atac_peak.h5`` in its order.
    - ``env_ok`` / ``env_reason`` - the method's conda env exists on this
      machine; the reason names the env and the one-method install command
      (``multibench env install --methods X --packed --run``). On macOS
      and Windows it says the environment runs only on Linux.
    - ``env_ok`` on a GPU-only method - when the upstream script calls CUDA
      unconditionally, ``env_ok`` also checks for an NVIDIA GPU
      (``mtb.env.host_has_gpu()``). Without one, ``env_reason`` gives the
      sentence ``run`` would raise: ``"<method> needs an NVIDIA GPU, and
      this computer has none. See ..."``. ``method_info(m)`` shows
      ``requires_gpu`` and the code line in ``gpu_evidence``.
    - ``assume_gpu=True`` (``multibench scan --assume-gpu``) skips that GPU
      test, for a check on a login node before a GPU-node job. The row's
      ``caveat`` then says ``This check assumes the job runs on a GPU node.``,
      and ``command`` leaves out the CPU flags.

    The file check runs whether or not any conda env is installed.

    **Reason columns.** ``reason`` joins the non-empty reasons as sentences
    and is empty only when the row is runnable. It is the short form. A
    missing ATAC file leads with what the method needs and what the folder
    holds (``UnitedNet needs gene-activity ATAC (atac_gas.h5), and the folder
    has peaks (atac_peak.h5).``). Other missing files read ``adt.h5 is
    missing.``

    File names are never cut. ``files_reason`` / ``env_reason`` keep the
    full text with full paths; read them for a row you are debugging.

    **The caveat column.** ``caveat`` lists what a row with ``files_ok``
    still needs, or what may go wrong without an error:

    - an ATAC file that holds the other representation (a peak matrix where
      the method needs gene activity), or peak names the method cannot read;
    - an RNA, ADT or peak file whose values are not whole numbers: the
      methods expect raw counts;
    - diagonal: a folder whose only label file is ``cty.csv``; diagonal
      needs ``rna_cty.csv`` and ``atac_cty.csv``;
    - a method that reads fewer numbered batches than the folder holds
      (``UINMF reads batches 1-2 of 3. Batch 3 is not used.``);
    - a step the user must do first, the first sentence of
      ``method_info(m)['setup_hint']`` (GLUE's GENCODE annotation file);
    - method scripts that are not on this machine yet: the first real run
      clones them with ``git``; on a host without network, fetch them first
      with ``multibench fetch --scripts``;
    - a command that reads a file ``mtb.run`` writes first (see the command
      column below).

    A row given the other ATAC representation is not runnable, also when
    ``methods=`` names the method, and ``run_all`` skips it. So is a GLUE or
    Seurat_v3 row whose peak names are not chr:start-end.
    ``allow_atac_mismatch=True`` keeps such a row runnable, with its caveat.
    With ``MULTIBENCH_SCRIPTS_REF`` set to another commit than the method
    scripts, or a ``repo_path`` that holds other files and no method
    scripts, no row is runnable.

    **The command column.**

    - ``command`` is ``run(..., dry_run=True)``, ``shlex``-joined, writing
      under ``<out_dir>/<method>_<dataset>/`` exactly like ``run_all`` - the
      literal ``'<out_dir>'`` placeholder unless ``out_dir`` is given.
    - Paths are absolute: a relative ``out_dir``, the placeholder included,
      is resolved against the working directory.
    - ``params`` are merged in the way ``run_all(params=)`` merges them.
    - On a GPU-less host the command already carries each method's
      ``cpu_params``, the flags that turn CUDA off where a switch exists.
    - A row blocked only by ``env_ok`` still shows its command. Put it in a
      job script once the environment is built.
    - Some commands read a file that ``mtb.run`` writes first under
      ``inputs/`` (Seurat_v3's renamed peak file, a converted input). The
      ``caveat`` names the file; start such a method with ``mtb.run`` or
      ``multibench run`` instead of the shell line.

    **The modalities column.** ``modalities`` is a ``+``-joined string here
    (``"rna+adt"``); ``run_all`` / ``inputs_for`` take a list
    (``["rna", "adt"]``), so split on ``"+"``. The sentinel ``"(data_dir)"``
    marks a method that consumes a whole directory rather than named modality
    files (scBridge); for it, pass no ``modalities`` at all.

    **Sizing a sweep.** ``runtime_tier`` / ``observed_worst_sec`` (see
    ``method_info(m)['runtime']``) let you size a sweep before launching it.

    **Selection and input checks.**

    - ``category`` - a typo raises ``ValueError`` listing the four valid
      values.
    - ``methods`` - an unknown id raises ``KeyError`` with a did-you-mean
      hint; blocked rows of the selected methods are kept, with their reason.
      A selection with no variant under ``category`` (a known id with no
      diagonal variant, say) raises ``ValueError`` instead of returning an
      empty frame.
    - ``params`` - a key no variant of that method accepts raises
      ``KeyError`` naming the accepted keys.
    - ``dataset`` - a spelling that differs from the folder only in case
      (``'d52'`` on macOS) is replaced by the on-disk spelling, with a
      ``UserWarning``.

    **Modality tokens.** ``modalities`` names one combination, in any
    order. Base tokens keep a row whose modalities are exactly that
    combination. Representation tokens select by what the method reads.
    ``mtb.find_methods`` keeps every method that reads at least the named
    modalities, so the two can list different methods.

    - a list that spells a row's modalities keeps that row, with one
      exception: moETM, scMM and iPOLNG read peaks through a role named
      ``atac_gas``, so ``atac_peak`` selects them and ``atac_gas`` does not,
      unless ``methods=`` names them;
    - a base token (``rna``, ``adt`` or its alias ``protein``, ``atac``)
      matches every role of that base: ``atac`` matches ``atac``,
      ``atac_gas``, ``atac_peak`` and numbered roles such as ``atac2``;
    - a representation token (``atac_peak`` / ``peak``, ``atac_gas`` /
      ``gas`` / ``gene_activity``) keeps the methods whose
      ``method_info(m)['atac']`` is that representation, like
      ``find_methods(atac=...)``. ``atac_peak`` alone also keeps MultiMAP
      and Seurat_v3, which read both files;
    - ``atac_peak`` together with ``atac_gas`` keeps only those two;
    - a numbered token (``rna1``) matches that role only;
    - an unknown token raises ``ValueError`` listing the vocabulary.

    A folder-fed variant (scBridge) has no modality roles. ``modalities=[]``
    selects exactly those. Other lists drop them with a ``UserWarning``,
    unless the tokens exclude their ATAC representation.

    **Choosing a category.** A CITE-seq folder (``rna.h5`` + ``adt.h5`` +
    ``cty.csv``) is ``vertical`` with modalities ``["rna", "adt"]``; RNA and
    ATAC from different cells is ``diagonal``. See ``mtb.list_categories``
    and ``mtb.describe_layout``.

    **Environments and CLI.** Each method runs in its own conda environment,
    on Linux. List them with ``multibench env doctor``; install one with
    ``multibench env install --methods X --packed --run``. Off Linux the
    summary line says what this computer can do. ``multibench scan`` prints
    a compact view by default; ``--columns all`` adds the rest, including
    ``command``.

    See Also
    --------
    mtb.run_all : run the runnable rows, with metrics; ``dry_run=True`` returns this frame.

    mtb.describe_layout : how to lay out a dataset folder so ``files_ok`` passes.

    mtb.env.doctor : the env check on its own, per env.

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
    wanted = _modality_matcher(modalities, methods) if modalities is not None else None
    base = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    dataset = _resolve.canonical_dataset(base, dataset)
    ds_dir = base / dataset
    if not ds_dir.is_dir():
        from .plot.bubble import _and
        dirs = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
        holds = (f"{base} holds {_and(dirs)}." if dirs else
                 f"{base} holds no folders." if base.is_dir() else
                 f"{base} does not exist either.")
        raise FileNotFoundError(
            f"The folder {ds_dir} does not exist. {holds} "
            + config.hint("dataset= is the folder name, and data_path= the folder that "
                          "holds it. mtb.describe_layout() shows the layout.",
                          "DATASET is the folder name, and --data-path the folder that "
                          "holds it. multibench layout shows the layout."))
    installed = _installed_envs()
    repo = _runner._repo_root_no_fetch()
    # scripts at another commit than $MULTIBENCH_SCRIPTS_REF, or a scripts
    # folder that holds other files: the real run refuses, so every row is
    # blocked; the files and the command stay checked
    wrong_ref = config.scripts_ref_problem(repo) or config.scripts_folder_problem(repo) or ""
    rows = []
    dropped_dirs: list[str] = []
    # one representation token already excludes a method that reads the other
    reps = _token_reps(modalities) if modalities is not None else set()
    for spec, v, cat, mods in _variant_rows(category):
        if methods is not None and spec.id not in methods:
            continue
        mod_str = "+".join(mods) or "(data_dir)"
        if wanted is not None and not wanted(spec, mods):
            other_rep = len(reps) == 1 and spec.atac and spec.atac not in reps
            if not mods and not other_rep and spec.id not in dropped_dirs:
                dropped_dirs.append(spec.id)
            continue
        rt = _runtimes().get(spec.id, {})
        rec = {"method": spec.id, "category": cat, "modalities": mod_str,
               "env": envs.group_for(spec.id), "output_kind": v.output.kind,
               "n_tunable": len(v.tunable),
               "runtime_tier": rt.get("tier", "unknown"),
               "observed_worst_sec": rt.get("worst_sec"),
               "caveat": "", "runnable": False, "reason": "",
               "files_ok": True, "files_reason": "", "env_ok": True, "env_reason": "",
               "needs_labels": bool(v.needs_labels),
               "atac": spec.atac if _variant_consumes_atac(v) else None,
               "command": ""}
        # --- check 1: files. Runs whether or not any env is installed. ------
        # Both halves are checked (the method's script and the dataset's files)
        # so a missing script does not hide a layout problem or vice versa.
        # files_reason keeps every text in full; reason gets the short form.
        file_problems, short_problems = [], []
        why_script = _missing_script(v, method=spec.id)
        if why_script:
            file_problems.append(why_script)
            short_problems.append(_short_reason(why_script, spec.id, dataset, cat))
        got, wrong_atac = None, ""
        try:
            got = _resolve.inputs_for(dataset, cat, spec.id, modalities=mods or None,
                                      data_path=data_path, check=True)
            extra = _resolve._preflight_caveats(got, atac=spec.atac, category=cat,
                                                method=spec.id)
            if extra:
                rec["caveat"] = _join_clauses([rec["caveat"], *extra])
            if not allow_atac_mismatch:
                wrong_atac = _wrong_atac_reason(spec.id, extra)
        except Exception as e:  # missing files / no variant / bad layout
            full = f"{type(e).__name__}: {e}"
            file_problems.append(full)
            short_problems.append(
                _missing_files_reason(spec, v, cat, mods, dataset, data_path)
                or _short_reason(full, spec.id, dataset, cat))
        if file_problems:
            rec["files_ok"], rec["files_reason"] = False, " ".join(file_problems)
        # --- check 2: env. ----------------------------------------------------
        if rec["env"] and rec["env"] not in installed:
            rec["env_ok"] = False
            rec["env_reason"] = _env_hint(rec["env"], spec.id, cat)
        # ... and the host: a script that calls CUDA unconditionally cannot
        # finish without an NVIDIA GPU, however complete the env - the same
        # sentence run() raises as OSError, so the sweep never starts it.
        # assume_gpu checks a job for a GPU node from a host without one.
        if spec.requires_gpu and not envs.host_has_gpu():
            if assume_gpu:
                rec["caveat"] = _join_clauses([rec["caveat"],
                                               GPU_NODE_CAVEAT.format(method=spec.id)])
            else:
                rec["env_ok"] = False
                rec["env_reason"] = " ".join(
                    r for r in (rec["env_reason"], spec.requires_gpu_reason) if r)
        rec["runnable"] = bool(rec["files_ok"] and rec["env_ok"] and not wrong_atac
                               and not wrong_ref)
        # the ATAC reason ends with its override, so it comes last
        rec["reason"] = _join_sentences([wrong_ref, *short_problems, rec["env_reason"],
                                         wrong_atac])
        # --- the command line: only when the files resolved (something to
        # hand the script); an env-blocked row still gets one. A setup step
        # the user must do first, and method scripts not yet on this machine,
        # go to caveat: the files are fine, but the run needs them.
        if rec["files_ok"] and got is not None:
            rec["command"], prepared = _command_line(spec.id, cat, got, out_dir=out_dir,
                                                     dataset=dataset,
                                                     params=params.get(spec.id),
                                                     gpu=True if assume_gpu else None)
            # the setup note in its first sentence, which starts with the
            # method name: the full hint is one method_info(m)['setup_hint']
            # away. A same-cells requirement (Seurat_v5) is checked on the
            # files above; its caveat appears only when the files fail it.
            notes = [n if n != spec.setup_hint else
                     "" if spec.id in _resolve._SAME_CELL_ROLES else _first_sentence(n)
                     for n in _runner.script_notes(spec, v, repo)]
            notes = [n for n in notes if n] + prepared
            if notes:
                rec["caveat"] = _join_clauses([rec["caveat"], *notes])
        rows.append(rec)
    df = pd.DataFrame(rows, columns=SCAN_COLUMNS)
    df = df.sort_values(["runnable", "category", "method"],
                        ascending=[False, True, True]).reset_index(drop=True)
    if df.empty and methods is not None:
        # no variant of the requested methods exists under this category: a
        # request problem (Matilda is not a cross method), reported as such
        # rather than as a silently empty frame
        raise _no_variant_error(category, dataset, methods, modalities)
    if params:
        _check_param_keys(df, params)       # a typo'd key must not start a sweep
    if dropped_dirs and modalities:
        # a variant fed a folder names no modality roles, so no token can
        # select it; say what was left out rather than dropping it unnoticed
        warnings.warn(_dropped_dirs_message(modalities, dropped_dirs),
                      UserWarning, stacklevel=2)
    if verbose:
        n, k_files, k_env = len(df), int(df["files_ok"].sum()), int(df["env_ok"].sum())
        rows = _rows_word(df, n)
        line = (f"[scan] {k_files} of {n} {rows} {_have_their(k_files, n)} input files. "
                f"{k_env} of {n} {_have_their(k_env, n)} environment installed.")
        if _runner.linux_only_sentence():
            line += f" {LINUX_ONLY_SUMMARY}"
        elif any(e and e not in installed for e in df["env"]):
            # the env reasons lead with the install command; doctor is named
            # once here (a row blocked only by the GPU test needs no doctor)
            doctor = config.hint("mtb.env.doctor()", "multibench env doctor")
            line += f" {doctor} checks the environments."
        print(line, flush=True)
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

    ``(best - max(runner_up, 0)) / best``. A ratio, not a difference: the
    runner-up sits near chance (ARI ~ 0), so a difference is bounded above by the
    ARI itself and a method scoring 0.3 could never look clearly separated.
    Dividing by the winner makes an unambiguous order read ~1.0 whether the
    method scored 0.9 or 0.2. ARI can fall slightly below 0, so the runner-up
    is clipped at 0 to keep the value within 0-1.
    """
    if not cands or len(cands) < 2:
        return None
    best, second = cands[0]["ARI"], cands[1]["ARI"]
    # A ratio of two chance-level values is noise (0.0004 vs 0.0002 would read
    # 0.5): report None so the column is not misread when no ordering worked.
    if best < _CHANCE_ARI:
        return None
    return round(max(0.0, (best - max(second, 0.0)) / best), 4)


def _metric_codes(metrics):
    """The metric codes ``metrics=`` selects; ``None`` = every metric, also
    for a value ``evaluate`` will reject (it raises there, with its message)."""
    from .data import catalog
    try:
        return catalog.metric_selection(metrics).codes
    except (TypeError, ValueError):
        return None


#: the metrics that need the Leiden clustering, and so a sweep
_SWEEP_METRICS = ("ARI", "NMI", "iF1")


def _needs_sweep(metrics) -> bool:
    """Whether ``metrics=`` names a metric computed on the Leiden clustering."""
    codes = _metric_codes(metrics)
    return codes is None or any(c in _SWEEP_METRICS for c in codes)


def _names_batch_metric(metrics) -> bool:
    """Whether ``metrics=`` is ``None`` or names a metric of the batch family."""
    from .plot.bar import BATCH_METRICS
    codes = _metric_codes(metrics)
    return codes is None or any(c in BATCH_METRICS for c in codes)


def _evaluate_best_order(emb, category, cands, *, batch=None, metrics=None,
                         user_batch=False, batch_given=True, on_rank=None):
    """Score each candidate label order, keep the best, return the full spread.

    ``batch`` (optional, one entry per cell in embedding order) replaces the
    file-of-origin batch vector carried by each candidate - ``run_all(batch=)``
    / ``BatchResult.rescore(batch=)``. ``user_batch`` says that the
    candidates already carry such a vector. ``metrics`` (a family token or a
    list of metric codes, ``evaluate(metrics=)``) restricts the set the
    winner is scored on; ``None`` = the family the batch structure implies
    (screening still needs ARI only). A file-of-origin batch, or a user batch
    with ``batch_given=False`` (the one ``run_all`` saved, reused by
    ``rescore``), goes to ``evaluate`` only when ``metrics`` can use it, so
    that ``evaluate`` does not warn about a batch the caller never gave.
    ``on_rank(orders, cells)`` is called before several orders are ranked.
    """
    user_batch = user_batch or batch is not None
    use_batch = (user_batch and batch_given) or _names_batch_metric(metrics)

    def _full(lab, bat, clustering=None):
        # several distinct source files (or a user batch with >1 level) => a
        # real batch structure, so ask for both metric families; else clustering.
        if batch is not None:
            bat = np.asarray(batch)
        grp = ("all" if use_batch and len(set(np.asarray(bat).tolist())) > 1
               else "clustering")
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
        # metrics= may leave ARI out: the one order needs no ranking
        # + 0.0: a tiny negative ARI rounds to 0.0, not -0.0
        ari = round(float(val["Value"]["ARI"]), 4) + 0.0 if "ARI" in val.index else None
        return names, val, [{"order": names, "ARI": ari}]

    # Ranking orderings needs only ARI, and the Leiden sweep behind ARI depends
    # on the embedding alone, not on the label vector. So sweep once, reuse it
    # for every candidate, and compute the full metric set once, on the winner;
    # doing either per candidate multiplies the cost by the number of orderings.
    import scib.metrics as _me

    if on_rank is not None:
        on_rank(len(cands), emb.shape[0])
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
    spread = [{"order": n, "ARI": round(a, 4) + 0.0} for a, n, _, _, _ in scored]
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


#: provenance of a reused output when ``out_dir`` holds no earlier record of it
_UNKNOWN_PROVENANCE = {"scripts_commit": None, "env_flavor": "unknown", "hostname": ""}


def _earlier_provenance(d: Path) -> dict:
    """``{method: {scripts_commit, env_flavor, hostname}}`` from the records
    saved in ``d/batch_result.json``; ``{}`` when there are none. A field an
    earlier record lacks gets its :data:`_UNKNOWN_PROVENANCE` value."""
    p = Path(d) / "batch_result.json"
    try:
        with open(p) as fh:
            recs = json.load(fh).get("records") or []
    except (OSError, ValueError, AttributeError):
        return {}
    return {r["method"]: {k: r.get(k, v) for k, v in _UNKNOWN_PROVENANCE.items()}
            for r in recs if isinstance(r, dict) and r.get("method")
            and r.get("status") != "SKIPPED"}


def _check_save_target(d: Path, dataset: str, category: str) -> None:
    """Raise ``ValueError`` when ``d`` holds a saved result of another
    dataset or category: merging those records would mix two sweeps."""
    p = Path(d) / "batch_result.json"
    if not p.exists():
        return
    try:
        with open(p) as fh:
            blob = json.load(fh)
    except (OSError, ValueError) as e:
        raise ValueError(f"{p} exists but cannot be read ({e}); save to another "
                         f"folder or remove it") from e
    have = (blob.get("dataset"), blob.get("category"))
    if have != (dataset, category):
        raise ValueError(
            f"{d} already holds a saved result for dataset={have[0]!r} "
            f"category={have[1]!r}; this one is dataset={dataset!r} "
            f"category={category!r}. Save it to another folder.")


def _scorable(rec: dict) -> bool:
    """Whether ``rescore`` scores ``rec``: not SKIPPED, FAIL or TIMEOUT."""
    status = str(rec.get("status", ""))
    return status != "SKIPPED" and not status.startswith(("FAIL", "TIMEOUT"))


def _rank_line(method):
    """``rescore(verbose=True)``'s line before the label orders are ranked."""
    def say(orders, cells):
        print(f"[rescore] {method}: ranking {orders} label orders on {cells:,} cells "
              f"with one Leiden sweep ...", flush=True)
    return say


def _warn_unsaved_batch(records, result) -> None:
    """Warn once when a record was scored with a user batch that is not saved."""
    old = [r for r in records if r.get("batch_source") == "user" and "batch_file" not in r]
    lost = [r for r in records if "batch_file" in r
            and result._batch_text(r.get("batch_file")) is None
            and (r.get("batch_file") or r.get("batch_source") == "user")]
    if old:
        warnings.warn("This result was scored with your batch vector, which older "
                      "versions did not save. Pass batch= again to keep the batch "
                      "metrics.", UserWarning, stacklevel=3)
    elif lost:
        warnings.warn("This result was scored with your batch vector, which is not "
                      "saved with it. Pass batch= again to keep the batch metrics.",
                      UserWarning, stacklevel=3)


class BatchResult:
    """The result of ``mtb.run_all``: its summary table, long table and figure.

    ``mtb.run_all`` and ``mtb.load_batch`` build it. It keeps one record per
    method, which ``rescore`` and ``plot`` read.

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
        Long table ``metric, value, method, dataset, category, ...`` for
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
    scored, the skipped ones and the failures.

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
        # the batch files the records name ({file name: CSV text}), until saved
        self._batches: dict = {}

    def _batch_text(self, name) -> str | None:
        """The text of the batch file ``name``: in memory, else in ``out_dir``."""
        if name and name not in self._batches and self.out_dir is not None:
            p = Path(self.out_dir) / name
            if p.is_file():
                self._batches[name] = p.read_text()
        return self._batches.get(name)

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
        >>> ok = res.summary.query("status == 'CHAIN_OK'")
        >>> ok.sort_values("ARI", ascending=False)

        Notes
        -----
        **Column reference.** When nothing ran the frame is empty, with the
        columns below except the metrics:

        ```text
        method                  method id
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
        caveat                  scan's caveat for the method, or ""
        reason                  why a SKIPPED method did not run, or ""
        ```

        A ``caveat`` of NaN: the record was saved before this column existed;
        ``n_batches`` and ``label_order`` still show which batches were read.

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
        - ``SKIPPED`` - blocked before the run, with the reason in the
          ``reason`` column; in ``failures`` only when ``methods=`` named the
          method.

        ``FAIL``, ``TIMEOUT``, ``RUN_OK_EVAL_FAILED`` and
        ``RUN_OK_NO_LABEL_MATCH`` also appear in ``failures``.

        **Graph methods.** scMoMaT also writes a UMAP, which is scored:
        ``CHAIN_OK_GRAPH_METHOD``. Seurat_WNN writes only a neighbour graph:
        ``RUN_OK_NO_EMBEDDING``, and its ``emb_shape`` is ``None``.

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
        ``rna_cty.csv+atac_cty.csv``). For diagonal data the embedding stacks
        two disjoint cell sets in a method-specific order.

        **Label-order confidence.** ``label_order_confidence`` is
        ``(best - max(runner_up, 0)) / best`` over the ARI of the candidate
        label orders, from 0 to 1. Near 1, one order clearly fits. Below about 0.5,
        two orders scored alike; check that row's label order.

        **Optimistic bias.** When more than one ordering is possible, the
        reported metrics are those of the ordering with the highest ARI. So
        they are slightly optimistic. ``label_order_confidence`` shows how far
        ahead the chosen order was.

        **Blank confidence.** The column is numeric, and a blank is ``NaN``.
        So ``> 0.5`` is ``False`` for it and ``.isna()`` finds it. It is blank
        in three cases, named by ``label_order_note``:

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
                        | {m: v for m, v in (r.get("metrics") or {}).items()}
                        # NaN, not None, for a record saved before the field
                        | {"caveat": np.nan if r.get("caveat") is None
                           else r.get("caveat"),
                           # a SKIPPED record keeps its reason in error
                           "reason": (r.get("error") or "")
                           if r.get("status") == "SKIPPED" else ""})
        if not rows:      # nothing ran (e.g. no method was runnable on this dataset)
            return pd.DataFrame(columns=["method", "status", "run_sec", "output_kind",
                                         "emb_shape", "n_tunable", "label_order",
                                         "label_order_confidence", "batch_source",
                                         "n_batches", "label_order_note", "caveat",
                                         "reason"]).astype({"label_order_confidence":
                                                            "float64"})
        sm = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
        # float64 with NaN also when every row is blank (else object, None)
        sm["label_order_confidence"] = pd.to_numeric(
            sm["label_order_confidence"], errors="coerce").astype("float64")
        sm = _with_label_order_note(sm)
        for col in ("caveat", "reason"):         # new columns go last
            sm[col] = sm.pop(col)
        # whole numbers stay whole next to the blanks of SKIPPED and FAIL rows
        for col in ("n_batches", "n_tunable"):
            try:
                sm[col] = pd.to_numeric(sm[col]).astype("Int64")
            except (TypeError, ValueError):     # not whole numbers: left as they are
                pass
        return sm

    @property
    def long(self) -> pd.DataFrame:
        """The scores as a long table, for plotting.

        This is what ``plot`` and ``mtb.plot.bubble`` consume.

        Returns
        -------
        pandas.DataFrame
            Columns ``metric``, ``value``, ``method``, ``dataset``,
            ``category``, ``clustering`` and ``source``. When the scores carry
            ``scored_with``, so does the table. Empty, with the first seven
            columns, when no method produced metrics.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> mtb.plot.bubble(res.long, metrics=["ARI", "NMI"])
        >>> res.long.pivot_table(index="method", columns="metric", values="value")

        Notes
        -----
        **Source of the rows.** Each record contributes the unrounded frame
        ``run_all`` attached (or ``long.csv`` via ``mtb.load_batch``) when
        present. Otherwise the record contributes its ``metrics`` dict.
        Neither that dict nor the folders that ``mtb.data.fetch_outputs``
        downloads carry ``scored_with``.

        See Also
        --------
        BatchResult.plot : draws the bubble figure from this frame.

        mtb.to_long : the wide -> long conversion used for the metrics dict.
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
                with warnings.catch_warnings():
                    # a metrics dict has no scoring record: keep the seven
                    # columns without to_long's wide-CSV warning
                    warnings.simplefilter("ignore", UserWarning)
                    frames.append(_to_long(
                        wide, method=r.get("method"),
                        dataset=r.get("dataset", self.dataset),
                        category=r.get("category", self.category)
                    ).drop(columns="scored_with"))
        if not frames:
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True)

    @property
    def results(self) -> list:
        """The raw per-method records: status, out_dir, metrics and the orderings tried.

        Each record keeps the method's ``out_dir``, which ``rescore`` reads.

        Returns
        -------
        list[dict]
            One dict per method. Read ``method``, ``status``, ``out_dir`` and
            ``metrics`` first; the other keys are listed in Notes.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> [r["out_dir"] for r in res.results]
        >>> res.results[0].get("label_order_candidates")  # None with a single ordering

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
        batch_file                  the file in out_dir that holds a user batch
        error, traceback, note      why a method failed, was skipped or was not scored
        requested                   SKIPPED records: True when methods= named it
        reused                      True when skip_existing reused the output
        env, output_kind, n_tunable the scan row the method ran from
        caveat                      that row's caveat, or ""
        data_path, multibench_version, started_at   provenance of the run
        data_root                   data_path as an absolute path
        scripts_commit, env_flavor, hostname        the scripts, env build and computer
        _long                       internal; read BatchResult.long instead
        ```

        **Reused outputs.** A record with ``reused`` True copies
        ``scripts_commit``, ``env_flavor`` and ``hostname`` from the method's
        earlier record in ``out_dir``: the values of the run that made the
        output. Without an earlier record they are ``None``, ``'unknown'`` and
        ``''``.

        **Label-order evidence.** ``label_order_candidates`` holds every
        ordering tried and its ARI. ``summary``'s ``label_order_confidence``
        is computed from them. It is present only when more than one ordering
        was possible.

        See Also
        --------
        BatchResult.summary : the same records as a table.
        """
        return self.records

    @property
    def failures(self) -> pd.DataFrame:
        """Methods that failed, timed out or could not be scored.

        ``run_all`` records failures instead of raising. Check this frame.

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
        - ``SKIPPED``, for a method that ``methods=`` named and that did not run.
        - ``RUN_OK_EVAL_FAILED`` - the embedding exists, scoring it failed.
        - ``RUN_OK_NO_LABEL_MATCH`` - ran, but no label file has as many
          cells as the output, so nothing was scored. Check the folder with
          ``mtb.inputs_for(check=True)`` and ``mtb.labels_for``.

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
                                      "RUN_OK_NO_LABEL_MATCH")
               or (r.get("status") == "SKIPPED" and r.get("requested"))]
        if not bad:
            return pd.DataFrame(columns=["method", "status", "error"])
        return pd.DataFrame([{k: r.get(k) for k in ("method", "status", "error")} for r in bad])

    def plot(self, **kw):
        """Bubble figure of every method that produced metrics.

        Rows are methods, best first. Circle size shows the rank within a
        column (bigger is better). Fill compares the values in a column: the
        lightest is the lowest in this figure, not zero.

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

        **Reading the figure.** Size and colour are relative to the methods
        in this figure, so with few methods a small gap fills the whole colour
        scale. Check the values in ``summary``.

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
        """Score the saved outputs again with new labels, batch or metrics.

        No method is re-run. Each record's embedding is read back from its
        ``out_dir``.

        Parameters
        ----------
        batch : array-like | Series | path | None
            Batch ids in the order of ``mtb.labels_for(dataset)``, or a
            barcode-indexed Series or CSV. ``None`` = the batch ``run_all`` was
            given, else each cell's label file.
        labels : array-like | Series | path | None
            One cell-type label per cell; a Series or a barcode-indexed CSV is
            aligned by barcode. ``None`` = search the dataset's label files
            again.
        metrics : str | list[str] | None
            Metric family (``"clustering"``, ``"batch"``, ``"all"``) or metric
            codes; ``None`` = every metric the batch structure allows.
        verbose : bool
            Print one line per method, and one before each label-order ranking.

        Returns
        -------
        BatchResult
            A new result; this one is untouched. Call ``.save(out_dir)`` on it
            to persist it.

        Raises
        ------
        ValueError
            ``labels`` or ``batch`` holds ids that are not cells of the dataset.

        Warns
        -----
        UserWarning
            A Series or barcode-indexed CSV cannot be aligned and is matched by position.

        Examples
        --------
        >>> res = mtb.load_batch("out/")
        >>> res.rescore(metrics=["ARI", "NMI"]).summary
        >>> new = res.rescore(batch="data/D11/donor.csv")
        >>> new.summary[["method", "batch_source", "iLISI"]]
        >>> res.rescore(labels=my_labels).save("out/rescored")

        Notes
        -----
        **Arguments.** A ``labels`` array, list or plain CSV follows the
        embedding rows. Without ``labels=``, a ``batch`` array follows the
        order of ``mtb.labels_for(dataset)``. With a ``labels`` array in
        embedding row order, the ``batch`` array follows the embedding rows
        too. A CSV path is read like a label file. The batch is recorded as
        ``batch_source='user'``.

        **Aligned by barcode.** A Series or one-column DataFrame with a
        non-default index, or a CSV whose first column holds barcodes, is
        aligned to the dataset's cells as in ``run_all(batch=)``, with the
        same errors and warnings. Aligned labels go into each record's
        label-order search, so ``label_order`` names the file order chosen.
        Labels matched by position read ``(user labels)``.

        **Label order.** With ``labels=None`` and several label files,
        ``rescore`` ranks the file orders by ARI, which needs one Leiden sweep.
        It keeps the stored order and skips the sweep when ``metrics=`` has no
        ARI, NMI or iF1.

        **Labels without batch.** Given ``labels`` and no ``batch``, the batch
        that ``run_all(batch=)`` saved is reused. Without a saved batch, every
        cell is in one batch (``batch_source`` ``None``, ``n_batches`` 1), so
        only clustering metrics are computed. Pass ``batch`` as well to get the
        batch metrics.

        **Record status.** A method that emits no embedding (graph-only) is
        marked ``RUN_OK_NO_EMBEDDING`` with a ``note``. ``SKIPPED``, ``FAIL``
        and ``TIMEOUT`` records are kept as they are. A record whose output
        file is gone becomes ``RUN_OK_EVAL_FAILED``, with the reason in
        ``error``. So does a record whose new scoring fails, for example on a
        ``batch`` of the wrong length (``batch has N entries, embedding has M
        cells``).

        **Other hosts.** When the dataset folder has moved, pass the folder
        that now holds it as ``mtb.load_batch(data_path=)``. When the dataset
        folder is not found, ``labels=None`` gives ``RUN_OK_NO_LABEL_MATCH``
        with a ``note``. A Series or CSV is then matched by position, with a warning,
        and the saved batch is not used.

        **Persisting.** ``mtb.load_batch`` keeps returning the original result
        until the new one is saved.

        See Also
        --------
        mtb.evaluate : the scoring function applied per record.

        BatchResult.save : persist the re-scored result.
        """
        import copy
        new = BatchResult([], self.dataset, self.category, out_dir=self.out_dir)
        # the data folder of each record: the recorded data_path, else data_root
        found = [_dataset_root(r, self.dataset) for r in self.records]
        roots = [dp for dp, _ in found]
        # the roots searched for each data folder, named when it is not found
        tried: dict = {}
        for dp, r in zip(roots, self.records):
            tried.setdefault(dp, _root_tries(r))
        # labels and batch per data folder, before any record is scored: ids
        # that are not cells of the dataset raise here
        lab_vec: dict = {}
        bat_vec: dict = {}
        for dp in dict.fromkeys(dp for dp, r in zip(roots, self.records)
                                if r.get("status") != "SKIPPED"):
            if labels is not None:
                lab_vec[dp] = _cell_vector(labels, self.dataset, dp, what="labels",
                                           order=_ROWS, stacklevel=4, moved=True,
                                           tried=tried[dp])
            # a batch vector follows labels given in embedding row order
            rows = labels is not None and not lab_vec[dp][1]
            if batch is not None:
                vec, by_cell = _cell_vector(batch, self.dataset, dp, what="batch",
                                            order=_ROWS if rows else None,
                                            stacklevel=4, moved=True, tried=tried[dp])
                # saved with the result, unless it follows the embedding rows
                file = None if rows and not by_cell else _batch_csv(vec, self.dataset, dp)
                bat_vec[dp] = (vec, by_cell, file)
        if batch is None and _names_batch_metric(metrics):
            _warn_unsaved_batch([r for r in self.records if _scorable(r)], self)
        for r, (dp, ok) in zip(self.records, found):
            rec = copy.deepcopy({k: v for k, v in r.items() if k != "_long"})
            rec["_long"] = None
            m = rec.get("method")
            # nothing was run, or the run failed: there is no output to score
            if not _scorable(rec):
                new.records.append(rec)
                continue
            rec["data_path"] = dp
            if ok and dp is not None:
                rec["data_root"] = dp           # the folder found, absolute
            try:
                mods = rec.get("modalities") or []
                v = registry.get(m).select(self.category, set(mods))
                emb = _load_embedding(Path(rec["out_dir"]), v)
                if emb is None:
                    rec["status"] = "RUN_OK_NO_EMBEDDING"
                    rec["note"] = (f"output kind={v.output.kind}; this method does not "
                                   "produce an embedding, so embedding-based metrics do not apply")
                else:
                    lab, lab_by_cell = lab_vec.get(dp, (None, False))
                    if batch is not None:
                        bat, bat_by_cell, file = bat_vec[dp]
                        rec["batch_file"] = file and file[0]
                    else:
                        # the batch run_all was given, when it is saved
                        text = self._batch_text(rec.get("batch_file"))
                        file = text and (rec["batch_file"], text)
                        bat, bat_by_cell = ((_read_batch_text(text), True) if text
                                            else (None, False))
                    if file:
                        new._batches[file[0]] = file[1]
                    # metrics without clustering: the stored label order is kept
                    keep = (rec.get("labels_used")
                            if labels is None and not _needs_sweep(metrics) else None)
                    _score_record(rec, emb, self.dataset, self.category, dp, v,
                                  batch=bat, labels=lab, metrics=metrics,
                                  labels_by_cell=lab_by_cell, batch_by_cell=bat_by_cell,
                                  keep_order=keep, batch_given=batch is not None,
                                  on_rank=_rank_line(m) if verbose else None,
                                  tried=tried[dp])
            except Exception as e:  # noqa: BLE001 - one bad record must not abort the rest
                _drop_scores(rec)
                rec["status"] = "RUN_OK_EVAL_FAILED"
                em = f"{type(e).__name__}: {e}"
                rec["error"] = em if len(em) <= 600 else "... " + em[-596:]
            if verbose:
                print(f"[rescore] {m} -> {rec['status']} {_ari_tail(rec)}".rstrip(),
                      flush=True)
            new.records.append(rec)
        # a saved batch that fits no label order: the batch metrics are gone
        unused = next((r["note"] for r in new.records
                       if str(r.get("note") or "").startswith("The saved batch is not used")),
                      None)
        if unused:
            warnings.warn(unused, UserWarning, stacklevel=2)
        return new

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

        Raises
        ------
        ValueError
            The folder holds a saved result of another dataset or category.

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
        - ``batch_<hash>.csv`` - a batch vector the records were scored with.

        Reload it with ``mtb.load_batch``.

        **Saving into a folder that has a result.** When the folder already
        holds ``batch_result.json`` for the same dataset and category, the
        records are merged. A method in this result replaces its earlier
        record, unless it is ``SKIPPED`` and the earlier one is not; the other
        earlier records are kept, and every file in the list above is
        rewritten from the merged set. This result object is not changed.

        A line ``# Merged with 1 earlier record in <folder> (StabMap).``
        names the kept methods.

        **Jobs in parallel.** Jobs running at the same time should use one
        ``out_dir`` each; combine them with
        ``multibench plot --input dir1 --input dir2``.

        **Blank confidence on disk.** In ``summary.csv`` the
        ``label_order_note`` column says why ``label_order_confidence`` is
        empty on a row: "single ordering", "winner at chance" or "not scored".

        See Also
        --------
        mtb.load_batch : reads the folder back.
        """
        d = Path(out_dir or self.out_dir or ".")
        _check_save_target(d, self.dataset, self.category)
        before = load_batch(d).records if (d / "batch_result.json").exists() else []
        # a SKIPPED record never replaces an earlier record of a run
        ran = {r.get("method") for r in before if r.get("status") != "SKIPPED"}
        new = [r for r in self.records
               if not (r.get("status") == "SKIPPED" and r.get("method") in ran)]
        mine = {r.get("method") for r in new}
        kept = [r for r in before if r.get("method") not in mine]
        merged = BatchResult(kept + new, self.dataset, self.category,
                             out_dir=d) if kept else self
        d.mkdir(parents=True, exist_ok=True)
        sm = merged.summary.copy()
        # "single ordering" is a result, "never ran" an absence, and a bare NaN
        # cannot tell them apart on disk. The note has its own column: a sentinel
        # string in the numeric one breaks `> 0.5` and `.isna()` and trips a
        # pandas incompatible-dtype FutureWarning.
        sm = _with_label_order_note(sm)
        sm.to_csv(d / "summary.csv", index=False)
        lng = merged.long
        if not lng.empty:
            lng.to_csv(d / "long.csv", index=False)
        elif (d / "long.csv").exists():
            # an earlier long.csv would attach stale metrics on load_batch
            (d / "long.csv").unlink()
        merged.failures.to_csv(d / "failures.csv", index=False)
        # the batch vectors the records were scored with, for a later rescore
        for name in dict.fromkeys(r["batch_file"] for r in merged.records
                                  if r.get("batch_file")):
            text = None if (d / name).exists() else self._batch_text(name)
            if text is not None:
                (d / name).write_text(text)
        slim = [{k: v for k, v in r.items() if k != "_long"} for r in merged.records]
        with open(d / "batch_result.json", "w") as fh:
            json.dump({"dataset": self.dataset, "category": self.category,
                       "records": slim}, fh, indent=1, default=str)
        if kept:
            names = ", ".join(dict.fromkeys(str(r.get("method")) for r in kept))
            records = "record" if len(kept) == 1 else "records"
            print(f"# Merged with {len(kept)} earlier {records} in {d} ({names}).",
                  flush=True)
        return d

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        ok = sum(1 for r in self.records if str(r.get("status", "")).startswith("CHAIN_OK"))
        noemb = sum(1 for r in self.records if r.get("status") == "RUN_OK_NO_EMBEDDING")
        nolab = sum(1 for r in self.records if r.get("status") == "RUN_OK_NO_LABEL_MATCH")
        skipped = sum(1 for r in self.records if r.get("status") == "SKIPPED")
        fails = self.failures
        # a named SKIPPED method is in failures too; count it once, as skipped
        named = int((fails["status"] == "SKIPPED").sum()) if len(fails) else 0
        bad = len(fails) - named
        extra = (f", {noemb} ran but not scorable" if noemb else "") + (
            f", {nolab} ran but no labels matched" if nolab else "") + (
            f", {skipped} skipped" if skipped else "") + (
            f" ({named} named)" if named else "")
        return (f"<BatchResult {self.category}/{self.dataset}: "
                f"{ok}/{len(self.records)} with metrics{extra}, {bad} failed>")


# ---------------------------------------------------------------------- run_all
#: closing lines that name no cause (R's after any error); the line before them does
_ERROR_TRAILERS = ("Execution halted",)
#: R prints the call trace and then any warnings after the error message
_R_CALLS = re.compile(r"^Calls: ")
_R_WARNINGS = re.compile(r"^In addition: Warning messages?:")


def _ari_tail(rec) -> str:
    """The end of a ``run_all`` / ``rescore`` result line: ``ARI 0.629``,
    rounded to 3 decimals and never ``-0.000``; ``""`` when the record has
    no ARI (missing, ``None`` or NaN)."""
    ari = (rec.get("metrics") or {}).get("ARI")
    try:
        ari = float(ari)
    except (TypeError, ValueError):
        return ""
    if ari != ari:          # NaN
        return ""
    return f"ARI {round(ari, 3) + 0.0:.3f}"


def _error_tail(error, width: int = 200) -> str:
    """The last line of ``error`` that names a cause, clipped from the left to ``width``.

    ``run_all``'s ``-> FAIL`` / ``-> TIMEOUT`` progress line ends with it, so
    a job log shows why without opening ``failures.csv``. R's call trace and
    trailing warning block are skipped, and an R message that continues on
    the line after ``Error in f() :`` is joined to it.
    """
    lines = [l.strip() for l in str(error or "").splitlines()]
    lines = [l for l in lines if l and l not in _ERROR_TRAILERS]
    cut = next((i for i in range(len(lines) - 1, 0, -1) if _R_WARNINGS.match(lines[i])), None)
    if cut is not None:
        lines = lines[:cut]
    lines = [l for l in lines if not _R_CALLS.match(l)]
    if not lines:
        return ""
    last = lines[-1]
    if len(lines) > 1 and lines[-2].startswith("Error") and lines[-2].endswith(":"):
        last = f"{lines[-2]} {last}"
    if len(last) <= width:
        return last
    # clipped at a word boundary: the tail starts with a whole word
    tail = last[-(width - 3):]
    space = tail.find(" ")
    if last[-(width - 3) - 1] != " " and 0 <= space < len(tail) - 1:
        tail = tail[space + 1:]
    return "..." + tail


def _scan_hint(dataset: str, category: str, *, data_path=None, methods=None,
               modalities=None, allow_atac_mismatch: bool = False,
               assume_gpu: bool = False) -> str:
    """The ``mtb.scan`` call (``multibench scan`` command under the CLI) that
    shows the rows a ``run_all`` call with these arguments selected.

    Every argument not at its default is spelled out as the caller gave it
    (a relative ``data_path`` stays relative), so the hint, run from the same
    directory, finds the same dataset folder and the same rows. A
    ``data_path`` that is the default data root is left out, as ``None``.
    """
    py, cli = [repr(dataset), repr(category)], [dataset, "--category", category]
    if data_path is not None and (os.path.realpath(data_path)
                                  != os.path.realpath(config.DEFAULT.data_path)):
        py.append(f"data_path={os.fspath(data_path)!r}")
        cli += ["--data-path", os.fspath(data_path)]
    if methods:
        py.append(f"methods={list(methods)}")
        cli += ["--methods", ",".join(methods)]
    if modalities:
        py.append(f"modalities={list(modalities)}")
        cli += ["--modalities", ",".join(modalities)]
    if allow_atac_mismatch:
        py.append("allow_atac_mismatch=True")
        cli.append("--allow-atac-mismatch")
    if assume_gpu:
        py.append("assume_gpu=True")
        cli.append("--assume-gpu")
    return config.hint(f"mtb.scan({', '.join(py)})",
                       "multibench scan " + " ".join(shlex.quote(str(a)) for a in cli))


def _nothing_runnable_message(dataset: str, category: str, blocked: pd.DataFrame,
                              methods, *, data_path=None, modalities=None,
                              allow_atac_mismatch: bool = False,
                              assume_gpu: bool = False) -> str:
    """The ``ValueError`` text for "not one requested variant can start".

    Scoped to what the caller asked for: with ``methods=`` every requested
    variant is listed with its own reason (one per line); without it the
    first three blocked variants are shown and the message says how many
    there are in total. Rows whose input files are in place come first
    (then by method): an env install unblocks those. Reasons of methods the
    caller did not request are never listed: they would point at the wrong
    fix. Off Linux, when an environment blocks a row, the line after the
    head says where methods run. The last line names the ``scan`` call with
    the caller's selection (:func:`_scan_hint`); the list counts methods when
    each method has one row (:func:`_rows_word`).
    """
    def _line(r):
        return f"  {r['method']} ({r['modalities']}): {r['reason']}"
    # rows whose files are in place first: they are the ones an env install fixes
    if "files_ok" in blocked.columns:
        blocked = blocked.assign(_files=~blocked["files_ok"].astype(bool)).sort_values(
            ["_files", "method"], kind="stable").drop(columns="_files")
    platform = _platform_line(blocked)
    doctor = config.hint("mtb.env.doctor()", "multibench env doctor")
    where = _scan_hint(dataset, category, data_path=data_path, methods=methods,
                       modalities=modalities, allow_atac_mismatch=allow_atac_mismatch,
                       assume_gpu=assume_gpu)
    if methods:
        lines = [_line(r) for _, r in blocked.iterrows()]
        head = (f"None of the requested methods ({', '.join(methods)}) can run on "
                f"{dataset} ({category})")
        return (f"{head}.\n{platform}Blocked, one line per "
                f"requested {_rows_word(blocked, 1)}:\n" + "\n".join(lines) +
                f"\n{where} shows these rows. Its files_ok and env_ok columns say "
                f"which check failed. {doctor} checks the environments.")
    head = f"No method can run on {dataset} ({category})"
    n, k = len(blocked), min(3, len(blocked))
    lines = [_line(r) for _, r in blocked.head(k).iterrows()]
    rows = _rows_word(blocked, n)
    shown = (f"The first {k} of {n} blocked {rows}" if n > k
             else f"The blocked {rows}" if n == 1 else f"The {n} blocked {rows}")
    return (f"{head}.\n{platform}{shown}:\n" + "\n".join(lines) +
            f"\n{where} shows every row. Its files_ok and env_ok columns say "
            f"which check failed. {doctor} checks the environments.")


def _platform_line(blocked: pd.DataFrame) -> str:
    """The platform sentence (with its newline) for the "nothing is runnable"
    error on a non-Linux host whose rows are blocked by the env check, else
    ``""``. It comes right after the head: the environments cannot be
    installed here, so the per-row reasons are not the fix."""
    linux_only = _runner.linux_only_sentence()
    if not linux_only or "env_ok" not in blocked or blocked["env_ok"].all():
        return ""
    return (f"{linux_only} On this computer you can check files, score embeddings and "
            f"plot. Run the methods on a Linux machine.\n")


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


def _label_file_sizes(dataset, category=None, method=None, *, modalities=None,
                      data_path=None) -> list:
    """``[(file name, cells)]`` of ``labels_for(...)``, in its order; ``[]`` when
    the files cannot be read."""
    try:
        files = _resolve.labels_for(dataset, category, method, modalities=modalities,
                                    data_path=data_path, check=False)
        return [(Path(p).name, len(_read_cty(p))) for p in files.values()]
    except Exception:  # noqa: BLE001 - no label files: nothing to compare against
        return []


def _batch_segments(dataset, data_path, batch) -> dict | None:
    """``{label file name: its cells' batch ids}`` when ``batch`` has one id per
    cell in the order of ``labels_for(dataset)``; else ``None``."""
    sizes = _label_file_sizes(dataset, data_path=data_path)
    if not sizes or sum(n for _, n in sizes) != len(batch):
        return None
    out, i = {}, 0
    for name, n in sizes:
        out[name] = batch[i:i + n]
        i += n
    return out


def _label_data_files(label: Path) -> list[Path]:
    """The data files whose barcodes are the cells of one label file:
    ``cty2.csv`` -> ``rna2.h5, adt2.h5, ...``; ``atac_cty.csv`` -> the ATAC files."""
    stem = label.stem
    digits = stem[len(stem.rstrip("0123456789")):]
    base = stem[:len(stem) - len(digits)]
    if base == "cty":
        stems = [f"{b}{digits}" for b in ("rna", "adt", "atac", "atac_peak", "atac_gas")]
    elif base in ("rna_cty", "adt_cty"):
        stems = [base[:-4]]
    elif base in ("atac_cty", "peak_cty"):
        stems = ["atac_peak", "atac_gas", "atac", "peak"]
    else:
        stems = []
    return [label.parent / f"{s}.h5" for s in stems]


def _dataset_folder(dataset, data_path) -> Path:
    """``<data_path>/<dataset>``, with ``None`` = config's data root."""
    base = config.DEFAULT.data_path if data_path is None else data_path
    return Path(base) / dataset


def _root_tries(rec: dict, override=None) -> list:
    """The data roots a saved record's dataset folder is looked up under, in
    order: ``override``, the recorded ``data_path`` (read from the current
    directory; ``None`` = config's) and ``data_root``."""
    tries = [override] if override is not None else []
    tries.append(rec.get("data_path"))
    if rec.get("data_root"):
        tries.append(rec["data_root"])
    return tries


def _dataset_root(rec: dict, dataset: str, override=None) -> tuple:
    """``(root, found)``: the data root a saved record is scored from.

    The first root of :func:`_root_tries` whose ``<root>/<dataset>`` is a
    folder, as an absolute path (``None`` stays ``None``: config's). When
    none is, the recorded ``data_path`` and ``False``.
    """
    for root in _root_tries(rec, override):
        if _dataset_folder(dataset, root).is_dir():
            return (None if root is None else str(Path(root).resolve())), True
    return rec.get("data_path"), False


def _check_data_path(data_path, dataset: str) -> None:
    """``ValueError`` when ``load_batch(data_path=)`` does not hold ``dataset``."""
    if data_path is None or _dataset_folder(dataset, data_path).is_dir():
        return
    root = Path(data_path)
    here = f", here {root.parent}" if root.name == dataset else ""
    raise ValueError(f"{root / dataset} is not a folder. data_path= is the folder "
                     f"that holds {dataset}{here}.")


#: what to do when a saved result's dataset folder is not found
_MOVED_FIX = ("Pass data_path= to mtb.load_batch, or run rescore from the folder "
              "where run_all ran.")


def _or(values) -> str:
    """``'A'``, ``'A or B'``, ``'A, B or C'``."""
    v = list(values)
    return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " or " + v[-1]


def _folder_missing(dataset: str, roots) -> str:
    """Why a saved result's cell ids cannot be read, as a clause: the dataset
    folder is under none of the data ``roots`` (``None`` = config's), each
    named as an absolute path."""
    where = dict.fromkeys(os.path.abspath(config.DEFAULT.data_path if r is None else r)
                          for r in roots)
    return f"the dataset folder {dataset} is not found in {_or(where)}"


def _unplaced_batch_note(dataset: str, data_path, tried, n_ids: int) -> str:
    """The ``note`` of a record whose saved batch fits no label order."""
    if not _dataset_folder(dataset, data_path).is_dir():
        why = _folder_missing(dataset, tried or [data_path])
        return f"The saved batch is not used, because {why}. {_MOVED_FIX}"
    total = sum(n for _, n in _label_file_sizes(dataset, data_path=data_path))
    return (f"The saved batch is not used, because it has {n_ids:,} ids and the label "
            f"files of {dataset} have {total:,} cells.")


def _batch_csv(vec, dataset, data_path) -> tuple[str, str]:
    """``(file name, CSV text)`` of a batch vector that ``run_all`` or
    ``rescore`` scored with: one row per cell, the dataset's cell ids first
    when it has as many. The name holds the text's sha1."""
    import hashlib
    frame = pd.DataFrame({"batch": np.asarray(vec)})
    if _dataset_folder(dataset, data_path).is_dir():
        ids = _dataset_cell_ids(dataset, data_path)
        if ids is not None and len(ids) == len(frame):
            frame.insert(0, "cell", ids)
    text = frame.to_csv(index=False)
    return f"batch_{hashlib.sha1(text.encode()).hexdigest()[:8]}.csv", text


def _read_batch_text(text: str) -> np.ndarray:
    """The batch ids of a file :func:`_batch_csv` wrote, as text."""
    import io
    return pd.read_csv(io.StringIO(text), dtype=str,
                       keep_default_na=False)["batch"].to_numpy()


def _dataset_cell_ids(dataset, data_path) -> list | None:
    """The dataset's cell ids in the order of ``labels_for(dataset)``: for
    each label file, the barcodes of a data file with as many cells. ``None``
    when a label file has no such data file, or ids repeat."""
    ids: list = []
    try:
        files = _resolve.labels_for(dataset, data_path=data_path, check=False)
        for p in map(Path, files.values()):
            n = len(_read_cty(p))
            bars = next((b for b in map(_resolve._barcodes_of, _label_data_files(p))
                         if b is not None and len(b) == n), None)
            if bars is None:
                return None
            ids.extend(bars)
    except Exception:  # noqa: BLE001 - no readable files: no ids
        return None
    return ids if ids and len(set(ids)) == len(ids) else None


_n_ids = _eio._n_ids


#: the order an argument of run_all / rescore follows when it is matched by position
_ROWS = "the embedding rows"


def _cell_vector(x, dataset, data_path, *, what: str = "batch", order: str | None = None,
                 stacklevel: int = 5, moved: bool = False,
                 tried=None) -> tuple[np.ndarray, bool]:
    """``x`` as one value per cell, and whether it was aligned by cell id.

    Aligned (``True``): a Series or one-column DataFrame whose index is not
    a ``RangeIndex``, or a CSV whose first column holds cell ids, is put in
    the order of ``labels_for(dataset)`` by the barcodes of the dataset's
    files, as ``evaluate`` aligns to an AnnData. Ids that are not cells of
    the dataset, repeated ids and cells without a value raise
    ``ValueError``. Without usable barcodes (missing or repeated) the match
    is by position, with a ``UserWarning``. Arrays, lists and other CSVs are
    returned as given (``False``): positional, in ``order`` (default: the
    order of ``labels_for(dataset)``). ``stacklevel`` is that of the CSV
    warning, which is raised one call deeper than the Series warning.
    A dataset folder that is not found is named with the data roots
    ``tried`` (default: ``data_path``); ``moved`` (rescore) adds the fix for
    a saved result read from another directory.
    """
    from .eval.pipeline import _carries_ids, _pick
    order = order or config.hint(f"the order of mtb.labels_for({dataset!r})",
                                 f"the order of the label files of {dataset}")
    folder = _dataset_folder(dataset, data_path)
    if not folder.is_dir():
        why = _folder_missing(dataset, tried or [data_path])
        fix = _MOVED_FIX if moved else f"Check that it follows {order}."
        no_ids = f"{why}. {fix[:-1]}" if moved else why
    else:
        why = (f"the files of {dataset} have no usable cell ids (missing or "
               f"repeated barcodes)")
        fix, no_ids = f"Check that it follows {order}.", why
    if isinstance(x, (str, Path)):
        vals, first = _eio.read_labels_ids(x, what=what, pick=_pick(what))
        if first is None:
            return np.asarray(vals), False
        return _eio.by_id_column(
            vals, first, _dataset_cell_ids(dataset, data_path), what=what,
            name=Path(x).name, target=dataset, order=order, no_ids=no_ids,
            stacklevel=stacklevel)
    vals = _eio.as_vector(x, what=what)
    if not _carries_ids(x):
        return vals, False
    ids = _dataset_cell_ids(dataset, data_path)
    if ids is None:
        warnings.warn(f"The {what} {type(x).__name__} is matched by position, because "
                      f"{why}. {fix}", UserWarning, stacklevel=stacklevel - 1)
        return vals, False
    index = pd.Index([str(i) for i in x.index])
    if list(index) == ids:
        return vals, True
    # to_numpy() is right only for a vector already in that order
    by_position = (f"pass {what}.to_numpy() only when {what} "
                   + (f"already follows {order}" if order == _ROWS
                      else f"is already in {order}"))
    if not index.is_unique:
        dup = index[index.duplicated()]
        raise ValueError(f"{what}: the index repeats {_n_ids(len(dup))} (first: "
                         f"{list(dup[:3])}), so it cannot be aligned to the cells of "
                         f"{dataset}. Give each cell one id, or {by_position}.")
    foreign = index.difference(pd.Index(ids), sort=False)
    if len(foreign):
        if pd.api.types.is_integer_dtype(x.index):
            raise ValueError(f"{what}: the index holds row numbers (first: "
                             f"{list(x.index[:3])}), not cell barcodes. Set the "
                             f"index to the dataset's barcodes, or {by_position}.")
        raise ValueError(f"{what}: {_n_ids(len(foreign))} "
                         f"{'is not a cell' if len(foreign) == 1 else 'are not cells'} "
                         f"of {dataset} (first: {list(foreign[:3])}). Rename the index "
                         f"to the dataset's barcodes, or {by_position}.")
    missing = pd.Index(ids).difference(index, sort=False)
    if len(missing):
        unit = "a batch id" if what == "batch" else "a label"
        raise ValueError(f"{what}: {len(missing):,} of the {len(ids):,} cells of {dataset} "
                         f"have no id in {what} (first: {list(missing[:3])}). Give "
                         f"{unit} for every cell.")
    return np.asarray(pd.Series(vals, index=index).reindex(ids).to_numpy()), True


def _batch_vector(batch, dataset, data_path) -> np.ndarray:
    """``batch`` as one id per cell, in the order of ``labels_for(dataset)``
    when it carries cell ids (:func:`_cell_vector`); arrays as given."""
    return _cell_vector(batch, dataset, data_path, what="batch", stacklevel=5)[0]


def _inputs_on_disk(row, dataset, data_path) -> bool:
    """Whether every input file of a scan row is in the folder; its file
    checks may still fail (a missing method script, a label count)."""
    if row["files_ok"]:
        return True
    mods = [] if row["modalities"] == "(data_dir)" else row["modalities"].split("+")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            paths = _resolve.inputs_for(dataset, row["category"], row["method"],
                                        modalities=mods or None, data_path=data_path,
                                        check=False)
        if "data_dir" in paths:
            v = registry.get(row["method"]).select(row["category"], set(mods))
            if not _resolve._check_data_dir(v, paths["data_dir"])[0]:
                return False
    except Exception:  # noqa: BLE001 - unresolvable: not on disk
        return False
    return all(Path(p).exists() for p in paths.values())


def _skipped_rows(blocked, attempted, methods, dataset, data_path) -> tuple[list, int]:
    """The blocked rows a real run reports, and how many others it only counts.

    Reported: a row whose input files are in the folder (blocked by its env,
    the GPU, the method script or an ATAC check), and every row of a method
    ``methods=`` names that has no runnable row. The rest need files the
    folder does not have.
    """
    named = set(methods or ())
    shown, others = [], 0
    for _, r in blocked.iterrows():
        on_disk = _inputs_on_disk(r, dataset, data_path)
        if on_disk or (r["method"] in named and r["method"] not in attempted):
            shown.append((r, on_disk))
        else:
            others += 1
    return shown, others


def _skipped_records(shown, attempted, methods, *, category, dataset, data_path,
                     version, data_root=None) -> list[dict]:
    """One ``SKIPPED`` record per method with no attempted row: its first
    reported row, a row with its files on disk first."""
    named = set(methods or ())
    first: dict = {}
    for r, on_disk in sorted(shown, key=lambda t: not t[1]):
        if r["method"] not in attempted:
            first.setdefault(r["method"], r)
    return [{"method": m, "category": category, "dataset": dataset,
             "modalities": [] if r["modalities"] == "(data_dir)"
             else r["modalities"].split("+"),
             "output_kind": r["output_kind"], "env": r["env"],
             "n_tunable": int(r["n_tunable"]), "status": "SKIPPED",
             "error": r["reason"], "requested": m in named, "_long": None,
             "caveat": _run_caveat(r["caveat"]),
             "data_path": str(data_path) if data_path else None,
             "data_root": data_root,
             "multibench_version": version,
             "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            for m, r in first.items()]


def _batch_length_problem(plan, batch, dataset, category, data_path) -> str:
    """A dry run's note when ``batch`` fits neither the dataset's label files nor
    the cells of some row the sweep would run; "" when it fits."""
    total = sum(n for _, n in _label_file_sizes(dataset, data_path=data_path))
    if not total or len(batch) == total:
        return ""
    bad = []
    for _, r in plan.iterrows():
        if not _would_run(r) or r["method"] in bad:
            continue
        mods = None if r["modalities"] == "(data_dir)" else r["modalities"].split("+")
        own = sum(n for _, n in _label_file_sizes(dataset, category, r["method"],
                                                  modalities=mods, data_path=data_path))
        if own and len(batch) != own:
            bad.append(r["method"])
    if not bad:
        return ""
    who = ", ".join(bad[:3]) + (f" and {len(bad) - 3} more" if len(bad) > 3 else "")
    return (config.hint(f"batch has {len(batch):,} entries",
                        f"--batch gave {len(batch):,} batch ids")
            + f" for the {total:,} cells of the label files of {dataset}. "
            f"The batch metrics of {who} would fail.")


#: what scoring writes into a record; a record that is not scored keeps none of it
_SCORE_KEYS = ("metrics", "labels_used", "label_order_candidates", "batch_source",
               "n_batches", "_long", "note", "error")


def _drop_scores(rec: dict) -> dict:
    """Remove the fields of an earlier scoring from ``rec``."""
    for k in _SCORE_KEYS:
        rec.pop(k, None)
    return rec


def _score_record(rec, emb, dataset, category, data_path, variant, *,
                  batch=None, labels=None, metrics=None, labels_by_cell=False,
                  batch_by_cell=False, keep_order=None, on_rank=None,
                  batch_given=True, tried=None):
    """Fill ``rec`` with metrics for ``emb`` (shared by run_all and rescore).

    Sets ``status`` (``CHAIN_OK`` / ``CHAIN_OK_GRAPH_METHOD`` /
    ``RUN_OK_NO_LABEL_MATCH`` / ``RUN_OK_EVAL_FAILED``), ``metrics``,
    ``labels_used``, ``label_order_candidates``, ``batch_source``,
    ``n_batches``, ``emb_shape`` and the tidy ``_long`` frame. ``metrics``
    restricts the metric set (``evaluate(metrics=)``). The fields of an
    earlier scoring are removed first, so a record that is not scored has
    none of them.

    ``keep_order`` (a stored ``labels_used``) is scored directly, with no
    ranking, when it is one of the candidate orders; the stored
    ``label_order_candidates`` are then kept. ``on_rank`` is handed to
    :func:`_evaluate_best_order`.

    ``labels`` (one per cell) replaces the label files. In embedding row
    order it bypasses the label-order search; with ``labels_by_cell`` it is
    in the order of ``labels_for(dataset)`` and is put into each candidate
    order of the search, as the file labels are. Either way the cells form
    one batch unless ``batch`` is given.

    ``batch`` (one per cell) replaces the file-of-origin batch. It is in the
    order of ``labels_for(dataset)`` and goes into each candidate order,
    except that a vector without ``batch_by_cell`` follows labels given in
    embedding row order. A batch in the dataset's order that meets such
    labels is put in the stored ``labels_used`` order when its files hold
    the embedding's cells, else in the variant's own label-file order. A
    positional vector as long as the embedding is used as given when no
    order fits. A ``batch_by_cell`` batch that fits no order (the dataset
    folder is not found, or its label files changed) is not used: the record
    gets a ``note`` and no batch metric. ``batch_given=False`` marks a batch
    the caller did not pass (the saved one ``rescore`` reuses): it is
    recorded as ``'user'`` but reaches ``evaluate`` only for a batch metric.
    ``tried`` are the data roots searched for the dataset folder, named in
    the note when it is not found (default: ``data_path``).
    """
    stored = rec.get("label_order_candidates")
    stored_used = rec.get("labels_used")
    _drop_scores(rec)
    rec["emb_shape"] = list(emb.shape)
    n = emb.shape[0]
    rows = labels is not None and not labels_by_cell      # labels in embedding row order
    segments = None
    if batch is not None:
        batch = np.asarray(batch)
        # in the dataset's cell order: each label order gets the ids of its files
        if batch_by_cell or not rows:
            segments = _batch_segments(dataset, data_path, batch)
        if batch_by_cell and segments is None:
            # cells in the dataset's order, but no label files to place them
            # by: the batch metrics would see the ids in the wrong rows
            rec["note"] = _unplaced_batch_note(dataset, data_path, tried, len(batch))
            batch = None
        elif len(batch) != n and segments is None:
            raise ValueError(f"batch has {len(batch)} entries, embedding has {n} cells")
    if rows:
        labels = np.asarray(labels)
        if len(labels) != n:
            raise ValueError(f"labels has {len(labels)} entries, embedding has {n} cells")
        cands = [(["(user labels)"], labels, np.ones(n, dtype=int))]
    else:
        cands = _label_candidates(dataset, n, data_path)
    if labels is not None and not rows:
        # the user's labels, in the dataset's cell order, replace each
        # candidate's file labels; the cells form one batch
        labels = np.asarray(labels)
        segs = _batch_segments(dataset, data_path, labels)
        cands = [(names, np.concatenate([segs[k] for k in names]), np.ones(n, dtype=int))
                 for names, _, _ in cands
                 if segs is not None and all(k in segs for k in names)]
        if not cands:
            if len(labels) != n:
                raise ValueError(f"labels has {len(labels)} entries, embedding has "
                                 f"{n} cells")
            cands = [(["(user labels)"], labels, np.ones(n, dtype=int))]
    kept = [c for c in cands if keep_order and c[0] == list(keep_order)]
    if kept:
        # the stored order is still a candidate: no ranking, no sweep
        cands = kept
    if batch is not None:
        # the user's ids replace each candidate's file-of-origin batch, put in
        # that candidate's order. Labels given in embedding rows follow the
        # stored label order when its files hold the embedding's cells, else
        # the variant's own order; a vector as long as the embedding is kept
        own = None
        if rows and segments is not None:
            used = list(stored_used or ())
            own = (used if used and all(k in segments for k in used)
                   and sum(len(segments[k]) for k in used) == n else
                   [k for k, _ in _label_file_sizes(dataset, category, rec.get("method"),
                                                    modalities=rec.get("modalities") or None,
                                                    data_path=data_path)])
        placed = []
        for names, lab, bat in cands:
            keys = own if rows else names
            if segments is not None and keys and all(k in segments for k in keys) \
                    and sum(len(segments[k]) for k in keys) == n:
                placed.append((names, lab, np.concatenate([segments[k] for k in keys])))
            elif len(batch) == n:
                placed.append((names, lab, batch))
        if cands and not placed:
            raise ValueError(f"batch has {len(batch)} entries, embedding has {n} cells")
        cands = placed
    if not cands:
        rec["status"] = "RUN_OK_NO_LABEL_MATCH"
        if not _dataset_folder(dataset, data_path).is_dir():
            why = _folder_missing(dataset, tried or [data_path])
            rec["note"] = f"{why[:1].upper()}{why[1:]}. {_MOVED_FIX}"
        return rec
    names, val, spread = _evaluate_best_order(emb, category, cands, metrics=metrics,
                                              user_batch=batch is not None,
                                              batch_given=batch_given,
                                              on_rank=on_rank)
    if val is None:
        rec["status"] = "RUN_OK_EVAL_FAILED"
        errs = [s["error"] for s in spread if isinstance(s, dict) and s.get("error")]
        if errs:
            rec["error"] = errs[0]
        return rec
    # + 0.0: a tiny negative score rounds to 0.0, not -0.0
    rec["metrics"] = {k: (None if pd.isna(x) else round(float(x), 4) + 0.0)
                      for k, x in val["Value"].items()}
    rec["labels_used"] = names
    if kept and stored:
        rec["label_order_candidates"] = stored
    elif len(spread) > 1:
        rec["label_order_candidates"] = spread
    # which batch vector the batch metrics saw (summary columns batch_source/n_batches)
    bat = next(b for nm, _, b in cands if nm == names)
    nb = int(len(set(np.asarray(bat).tolist())))
    if batch is not None:
        rec["batch_source"], rec["n_batches"] = "user", nb
    else:
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
            batch=None,
            assume_gpu: bool = False,
            allow_atac_mismatch: bool = False) -> "BatchResult | pd.DataFrame":
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
    batch : array-like | Series | path | None
        Batch ids, cells in the order of ``mtb.labels_for(dataset)``; a Series
        or a barcode-indexed CSV is aligned by barcode. ``None`` = each cell's
        label file.
    assume_gpu : bool
        Dry run only: skip this host's GPU test, as ``mtb.scan(assume_gpu=True)``
        does.
    allow_atac_mismatch : bool
        ``True`` = also run a method whose ATAC file holds the other
        representation or unreadable peak names.

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
        Unknown ``category``, no matching variant, nothing runnable, a
        mismatched ``out_dir``, or conflicting arguments (Notes).
    ValueError
        A ``batch`` Series or CSV holds ids that are not cells of the dataset.
    KeyError
        Unknown id in ``methods`` or ``params``; on a dry run, a rejected ``params`` key.
    TypeError
        A real run without ``out_dir``; ``methods`` or ``modalities`` given as
        a bare string.

    Warns
    -----
    UserWarning
        ``dataset`` matches a folder only up to letter case.
    UserWarning
        ``modalities`` drops a folder-fed method whose ATAC representation it allows.
    UserWarning
        A ``batch`` Series or barcode-indexed CSV cannot be aligned and is matched by position.

    Examples
    --------
    >>> import multibench as mtb
    >>> plan = mtb.run_all("D11", "vertical", dry_run=True)  # what would run?
    >>> plan[["method", "modalities", "runnable", "reason"]]
    >>> res = mtb.run_all("D11", "vertical", out_dir="out/", timeout=3600)
    >>> res.summary        # one row per method, metrics as columns
    >>> res.failures       # failures are recorded, not raised

    Notes
    -----
    **Dry run.** ``dry_run=True`` runs nothing and returns the
    ``mtb.scan`` frame for the same selection: blocked rows are kept with
    their ``reason``, and ``command`` is rendered for ``out_dir`` (or the
    literal ``'<out_dir>'`` placeholder). ``plan[plan.runnable]`` lists what
    will run. ``len(plan)`` also counts blocked rows.
    ``multibench run-all --dry-run --format csv`` writes the same frame.

    **Before the sweep.** Every attempted row passed both ``mtb.scan`` checks
    (input files and conda env, plus a GPU where the script needs one), so a
    missing env is reported before any method starts (``multibench env
    doctor``). Methods take minutes to hours each.

    **Skipped rows.** A blocked row whose input files are in the folder,
    or a named method with no runnable row, is logged and recorded as
    ``SKIPPED`` with its reason. Other blocked rows are only counted.

    **Failures are recorded.** In a real run a method that raises is
    recorded as ``FAIL`` (with its ``error``), one that exceeds ``timeout``
    as ``TIMEOUT``, and the sweep moves on; a ``params`` key the variant does
    not accept is a ``FAIL`` too. Check ``res.failures``.

    **Timeout.** Without a cap, one method that hangs stops the whole
    sweep. Size it from the
    ``runtime_tier`` / ``observed_worst_sec`` columns of ``mtb.scan`` (or
    ``method_info(m)['runtime']``); the slowest methods take more than 4 h.
    The cap covers the run and its scoring; off the main thread it is
    unavailable, with a warning.

    **Saved files.** The result is saved automatically under ``out_dir``
    (``summary.csv``, ``failures.csv``, ``batch_result.json``, ``long.csv``
    when some method produced metrics); reload it with ``mtb.load_batch``.
    With ``batch=``, the vector is saved as ``batch_<hash>.csv``.

    **Several jobs, one folder.** A later run into the same ``out_dir`` is
    merged with the records already there: methods it re-ran are replaced,
    the others kept. An ``out_dir`` that holds another dataset or category
    raises ``ValueError`` before any method runs. Jobs running at the same
    time should use one ``out_dir`` each; combine them with
    ``multibench plot --input dir1 --input dir2``.

    **Resuming.** ``skip_existing=True`` reuses each method's existing
    output. Reuse only checks that the output file exists, not
    that it is complete: a method killed mid-write leaves a truncated file
    that would be reused as if it had succeeded. After a hard kill, delete
    that method's sub-directory before resuming.

    A reused output's record copies ``scripts_commit``, ``env_flavor`` and
    ``hostname`` from the method's earlier record in ``out_dir``. Without an
    earlier record they are ``None``, ``'unknown'`` and ``''``.

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
    is recorded as ``batch_source='user'``.

    Give ``batch`` in the cell order of ``mtb.labels_for(dataset)``.
    ``run_all`` puts it in each method's cell order, as it does the labels,
    and keeps only the cells of the batches a method reads. A vector as long
    as one method's output is used as given.

    A Series or one-column DataFrame indexed by cell id is aligned to the
    barcodes of the dataset's files. So is a CSV whose first column holds
    them, as ``obs[["sample"]].to_csv(path)`` writes it. Ids that are not
    cells of the dataset raise ``ValueError`` before any method runs.

    Files without usable barcodes give a match by position, with a
    ``UserWarning``. So does a CSV whose first column holds text but none of
    the barcodes. R's row numbers in that column give no warning.

    Any other length marks that method ``RUN_OK_EVAL_FAILED`` (``batch has N
    entries, embedding has M cells``); the dry run says so first. Re-score a
    finished sweep with ``BatchResult.rescore``.

    **ATAC files.** Vertical reads ``atac.h5``; ``method_info(m)["atac"]``
    says whether it must hold peaks or gene activity. Diagonal reads
    ``atac_peak.h5`` (peaks) and ``atac_gas.h5`` (gene activity). Mosaic
    reads ``atac<i>.h5`` (peaks). ``peak.h5``, and ``atac.h5`` for gene
    activity, are accepted as older names.

    ``run_all`` skips a method given the other representation, also when
    ``methods=`` names it. With ``allow_atac_mismatch=True`` the method runs
    without an error and gives a wrong embedding.

    **Modality tokens.** ``modalities`` follows the rule of ``mtb.scan``.
    Base tokens keep a row whose modalities are exactly that combination;
    ``atac`` matches every ATAC role. Representation tokens select by what
    the method reads: ``atac_peak`` / ``atac_gas`` keep the methods that
    need that representation. moETM, scMM and iPOLNG read
    peaks through a role named ``atac_gas``, so ``atac_peak`` selects them
    and ``atac_gas`` does not, unless ``methods=`` names them.

    **Errors raised.**

    - An unknown ``category``: ``ValueError`` listing the four.
    - An unknown id in ``methods`` or ``params``: ``KeyError`` with a
      did-you-mean hint, before anything runs.
    - A selection that matches no variant: ``ValueError``, such as
      "Matilda does not run on cross data."; a dry run is never empty.
    - A dry run with a ``params`` key no planned variant of that method
      accepts: ``KeyError`` naming the accepted keys.
    - Nothing runnable: ``ValueError``. Its first line is
      ``No method can run on D11 (vertical).`` With ``methods=``, it starts
      ``None of the requested methods (Matilda, totalVI) can run on D11 (vertical).``
      The message lists the reason of every requested variant. Without
      ``methods``, it gives the first 3 of N. It never lists the reasons of
      methods you did not ask for. On macOS or Windows, when an environment
      blocks a row, its second line says that methods run only on Linux.
    - An ``out_dir`` that holds a saved result of another dataset or
      category: ``ValueError``, before any method runs.
    - ``skip_existing=True`` with ``params``, or ``assume_gpu=True`` in a
      real run: ``ValueError``; a real run checks this host's GPU.

    **Dataset spelling.** A ``dataset`` that differs from the folder only in
    case (``'d52'``) is replaced by the on-disk spelling, with a
    ``UserWarning``, before anything is named after it.

    See Also
    --------
    mtb.scan : the preflight frame this function runs from.

    mtb.BatchResult : what is returned - ``summary``, ``long``, ``failures``, ``plot``, ``rescore``.

    mtb.sweep : one method over a range of one hyperparameter.

    mtb.load_batch : reload a saved sweep.

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
    if assume_gpu and not dry_run:
        raise ValueError(
            "assume_gpu=True applies to a dry run only; a real run checks this "
            "host's GPU. Pass dry_run=True, or run the sweep on the GPU node "
            "without assume_gpu.")
    if not dry_run and skip_existing and params:
        raise ValueError(
            "skip_existing=True with params=... would return results computed with the "
            "earlier parameters (reuse is keyed on the output file, not on params). "
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
                   modalities=modalities, verbose=False, assume_gpu=assume_gpu,
                   allow_atac_mismatch=allow_atac_mismatch,
                   # the dry run renders (and validates) params in the frame; a real
                   # run validates per method and records a bad override as FAIL
                   params=params if dry_run else None,
                   out_dir=OUT_DIR_PLACEHOLDER if out_dir is None else out_dir)
    if plan_df.empty:
        # only reachable through a modalities= selector that matches nothing:
        # a request problem, reported as such rather than as "nothing is
        # runnable" with other methods' reasons attached
        raise _no_variant_error(category, dataset, methods, modalities)
    if dry_run:
        # a Series with ids that are not cells of the dataset raises here too
        batch_vec = None if batch is None else _batch_vector(batch, dataset, data_path)
        if verbose:
            k, n = int(plan_df["runnable"].sum()), len(plan_df)
            msg = (f"[run_all] Dry run: {k} of {n} requested {_rows_word(plan_df, n)} "
                   f"can run on {dataset} ({category}).")
            if n > k:
                doctor = config.hint("mtb.env.doctor()", "multibench env doctor")
                msg += (f" {n - k} {'is' if n - k == 1 else 'are'} blocked. The table's "
                        f"reason column says why, "
                        f"and its files_ok and env_ok columns say which check "
                        f"failed. {doctor} checks the environments.")
            print(msg, flush=True)
            # every row also carries it as its reason
            wrong_ref = _scripts_ref_note()
            if wrong_ref:
                print(f"[run_all] {wrong_ref}", flush=True)
            # the caveats of the rows the sweep would run: the compact views clip them
            scripts, lines = _dry_run_notes(plan_df)
            if scripts:
                print(f"[run_all] {scripts}", flush=True)
            for _m, cav in lines:          # each caveat starts with its method
                print(f"[run_all] {cav}", flush=True)
            if batch_vec is not None:
                bad = _batch_length_problem(plan_df, batch_vec, dataset, category,
                                            data_path)
                if bad:
                    print(f"[run_all] {bad}", flush=True)
        return plan_df                     # = scan(): runnable rows first, blocked rows keep `reason`
    blocked = plan_df[~plan_df["runnable"]]
    plan_df = plan_df[plan_df["runnable"]]
    if plan_df.empty:
        # A per-method failure is recorded, never raised - but "not one method
        # could start" means the request is wrong (bad dataset name, wrong
        # category, missing files, missing env). An empty result would report
        # "0 failed", which reads as success and hides a typo.
        raise ValueError(_nothing_runnable_message(
            dataset, category, blocked, methods, data_path=data_path,
            modalities=modalities, allow_atac_mismatch=allow_atac_mismatch,
            assume_gpu=assume_gpu))

    batch_vec = None if batch is None else _batch_vector(batch, dataset, data_path)
    # saved in out_dir so that rescore can reuse it
    batch_file = (_batch_csv(batch_vec, dataset, data_path)
                  if batch_vec is not None and evaluate else None)
    # the data root as an absolute path: a saved result read from another
    # directory still finds the dataset folder
    data_root = os.path.abspath(config.DEFAULT.data_path if data_path is None
                                else data_path)
    out_dir = Path(out_dir)
    # a folder holding another dataset's saved result would refuse the save
    # after the sweep; refuse now, before any method runs
    _check_save_target(out_dir, dataset, category)
    # every blocked row a reader of the log or the summary would look for is
    # named with its reason; the rest (files this folder lacks) are counted
    attempted = set(plan_df["method"])
    shown, others = _skipped_rows(blocked, attempted, methods, dataset, data_path)
    if verbose:
        several = pd.Series([r["method"] for r, _ in shown]).value_counts()
        for r, _ in shown:
            mods = f" ({r['modalities']})" if several.get(r["method"], 0) > 1 else ""
            print(f"[run_all] skipping {r['method']}{mods}: {r['reason']}", flush=True)
        if others:
            rows = "row needs" if others == 1 else "rows need"
            print(f"[run_all] {others} more {rows} files this folder does not have. "
                  + config.hint("mtb.scan", "multibench scan")
                  + f" shows {'it' if others == 1 else 'them'}.", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    # a reused output keeps the provenance of the run that made it
    earlier = _earlier_provenance(out_dir) if skip_existing else {}
    from . import __version__ as _pkg_version

    for _, row in plan_df.iterrows():
        m, mods = row["method"], row["modalities"]
        mod_list = [] if mods == "(data_dir)" else mods.split("+")
        rec = {"method": m, "category": category, "dataset": dataset,
               "modalities": mod_list, "output_kind": row["output_kind"],
               "env": row["env"], "n_tunable": row["n_tunable"], "status": "?", "_long": None,
               "caveat": _run_caveat(row["caveat"]),
               # provenance: what ran, where, with what
               "params_used": dict(params.get(m) or {}),
               "out_dir": str(out_dir / f"{m}_{dataset}"),
               "data_path": str(data_path) if data_path else None,
               "data_root": data_root,
               "multibench_version": _pkg_version,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        t0 = time.time()
        if verbose:
            print(f"[run_all] {m} ({category}/{dataset}) ...", flush=True)
            if rec["caveat"]:              # it starts with the method name
                print(f"[run_all]   {rec['caveat']}", flush=True)
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
                with warnings.catch_warnings():
                    # an allowed ATAC caveat is already in the log and the record
                    warnings.filterwarnings(
                        "ignore", category=UserWarning,
                        message=re.escape(f"{m} ") + rf"({_WRONG_ATAC_BODY}"
                                rf"|{re.escape(PEAK_NAMES_REASON)})")
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
                    if batch_file:
                        rec["batch_file"] = batch_file[0]
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
            tail = _error_tail(rec["error"]) if rec.get("error") else _ari_tail(rec)
            print(f"[run_all]   -> {rec['status']} ({rec.get('run_sec')}s) {tail}".rstrip(),
                  flush=True)
        if rec.get("reused"):
            rec.update(earlier.get(m, _UNKNOWN_PROVENANCE))
        else:
            rec.update(config.run_provenance(row["env"]))   # scripts_commit, env_flavor, hostname
        records.append(rec)
    records += _skipped_records(shown, attempted, methods, category=category,
                                dataset=dataset, data_path=data_path,
                                version=_pkg_version, data_root=data_root)

    result = BatchResult(records, dataset, category, out_dir)
    if batch_file and any(r.get("batch_file") for r in records):
        result._batches[batch_file[0]] = batch_file[1]
    result.save()          # survive process exit; reload with load_batch()
    return result


def sweep(dataset: str, category: str, method: str, param: str, values, *,
          out_dir, modalities=None, data_path=None, timeout=None,
          verbose: bool = True) -> pd.DataFrame:
    """Run one method once per value of one hyperparameter, each in its own out_dir.

    Parameters
    ----------
    dataset : str
        Dataset folder name under ``data_path``, as for ``mtb.run_all``.
    category : str
        Integration category of the variant to run.
    method : str
        Method id, e.g. ``"Matilda"``; see ``mtb.list_methods()``.
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
        first (column named ``param``). A long table for plotting is in
        ``df.attrs["long"]``.

    Raises
    ------
    KeyError
        Unknown ``method``, or ``param`` not among the tunable keys of a single variant.

    Examples
    --------
    >>> import multibench as mtb
    >>> # what can be swept
    >>> mtb.params_for("Multigrate", "vertical", ["rna", "adt"])["tunable"]
    >>> df = mtb.sweep("MYDATA", "vertical", "Multigrate", "lr",
    ...                [1e-4, 1e-3, 1e-2], out_dir="out/lr")
    >>> df[["lr", "status", "ARI", "NMI"]]
    >>> mtb.plot.bubble(df.attrs["long"])      # one series per setting

    Notes
    -----
    **Folder names.** Each setting's folder is ``<param>_<value>`` with
    ``.`` -> ``p`` and ``-`` -> ``m`` (``lr=0.001`` runs under
    ``<out_dir>/lr_0p001/``).

    **The long table.** ``df.attrs["long"]`` makes each setting a separate
    series (``"Multigrate (lr=0.001)"``), so it can go straight into
    ``mtb.plot.bubble``; ``.long`` keys rows by method, so without it every
    setting would collapse onto one row. ``DataFrame.attrs`` does not
    survive ``to_csv``, so the frame is also written to
    ``<out_dir>/sweep_long.csv`` (path in ``df.attrs["long_path"]``) when any
    setting produced metrics.

    **Failed settings.** A setting that fails is not fatal: ``run_all``
    records it, so that value's row appears with ``status`` ``FAIL`` (or
    ``TIMEOUT``) and empty metrics rather than aborting the sweep. Check the
    ``status`` column before reading the curve.

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
    ``mtb.run_all`` (e.g. no method can run) propagate.

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
