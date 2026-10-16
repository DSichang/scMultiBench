"""High-level workflow: point at a dataset, run everything that applies, get metrics and a figure.

The low-level API (``inputs_for`` -> ``run`` -> ``evaluate`` -> ``plot``) needs a
method, its category and its modality combination; this module starts from the dataset:

    mtb.scan("D11")                                       # what can run on this data?
    res = mtb.run_all("D11", "vertical", out_dir="out/")  # run it all, with metrics
    res.plot()                                            # one figure

Two mistakes that give wrong numbers without an error are handled here. A method
that writes a graph is recorded as such, not scored with embedding metrics (a kNN
index matrix scores ARI ~ 0). Labels must follow the embedding's cell order, which
length alone cannot tell: every candidate order is scored and the best is kept.
"""
from __future__ import annotations

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
from .config import _and_list
from .discover import _REPRESENTATION_TOKENS, _runtimes
from .engine import envs, registry, resolve as _resolve
from .engine import runner as _runner
from .engine.schema import base_modality as _schema_base
from .engine.runner import run as _run
from .eval import io as _eio, scib as _escib
from .eval.pipeline import LONG_COLUMNS as _LONG_COLUMNS, evaluate as _evaluate, to_long as _to_long

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
    ``long.csv`` restores each method's unrounded long table; without it
    ``BatchResult.long`` is rebuilt from the records' rounded ``metrics``.
    ``methods=`` only filters: records keep the order the tree ran them.

    A record whose ``out_dir`` does not exist is pointed at the folder of the
    same name next to ``batch_result.json``. The dataset folder is looked up
    under ``data_path=``, the recorded ``data_path`` and then ``data_root``,
    and the one found is recorded as an absolute path. So a ``fetch_outputs``
    tree or a copied ``run_all`` folder can be re-scored.

    See Also
    --------
    mtb.BatchResult : the object returned.

    mtb.run_all : writes the folder this function reads.

    mtb.data.fetch_outputs : downloads recorded run outputs in the same layout.
    """
    folder = Path(out_dir)
    with open(folder / "batch_result.json") as fh:
        blob = json.load(fh)
    records = blob["records"]
    if methods is not None:
        have = [r.get("method") for r in records]
        unknown = [m for m in methods if m not in have]
        if unknown:
            raise KeyError(f"no record for {unknown} in {folder}; methods in the "
                           f"tree: {have}")
        records = [r for r in records if r.get("method") in set(methods)]
    _check_data_path(data_path, blob["dataset"])
    for r in records:
        # a tree copied from another host: the output folder sits next to the JSON
        saved = r.get("out_dir")
        if saved and not Path(saved).exists() and (folder / Path(saved).name).is_dir():
            r["out_dir"] = str((folder / Path(saved).name).resolve())
        # left as recorded when no root holds the dataset; rescore then names the roots
        root, found = _dataset_root(r, blob["dataset"], data_path)
        if found and root is not None:
            r["data_path"] = r["data_root"] = root
    long_path = folder / "long.csv"
    long_table = pd.read_csv(long_path) if long_path.exists() else None
    for r in records:
        rows = [] if long_table is None else long_table[long_table["method"] == r.get("method")]
        r["_long"] = rows if len(rows) else None
    res = BatchResult(records, blob["dataset"], blob["category"], out_dir=folder)
    for r in records:
        res._batch_text(r.get("batch_file"))       # loads the saved batch, when present
    return res


#: The four integration scenarios, and what each one's data looks like.
CATEGORIES = {
    "vertical": "Several modalities measured in the same cells (e.g. CITE-seq "
                "RNA+ADT, or 10x multiome RNA+ATAC). Cells are already matched.",
    "diagonal": "Modalities measured in different cells, with no pairing "
                "(e.g. an RNA experiment and a separate ATAC experiment).",
    "mosaic":   "Several batches where only some share a modality. A paired batch "
                "bridges the others.",
    "cross":    "Several batches that all measure the same modalities, for example "
                "one CITE-seq assay (RNA+ADT) from several donors. The task is "
                "removing batch effects.",
}

#: Modality role -> the file the loader looks for in <data_path>/<dataset>/.
#: One canonical name per role: the file ``mtb.io.export_dataset`` writes.
ROLES = {
    "rna":       "rna.h5         - gene expression",
    "adt":       "adt.h5         - surface protein (CITE-seq antibody-derived tags)",
    "atac":      "atac.h5        - ATAC. method_info(m)['atac'] says peaks or gene activity",
    "atac_peak": "atac_peak.h5   - ATAC as peaks (diagonal)",
    "atac_gas":  "atac_gas.h5    - ATAC as gene-activity scores (diagonal)",
    "rna1/adt1/atac2 ...": "rna1.h5, adt1.h5, atac2.h5 ... - one file per batch (mosaic, cross)",
    "cty":       "cty.csv        - cell-type labels, one per cell (vertical)",
    "rna_cty / atac_cty":
                 "rna_cty.csv, atac_cty.csv - one label file per modality (diagonal)",
    "rna_cty1/atac_cty1":
                 "rna_cty1.csv, atac_cty1.csv ... - the same per batch (diagonal, several "
                 "batches)",
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
    For one category the text gives the files its methods read, as
    ``mtb.io.export_dataset`` writes them, the ATAC representation each
    method needs, the ``.h5`` and label formats, the install command and, for
    mosaic and cross, each batch pattern the methods accept. The lists are
    read from the method registry at call time. ``multibench layout`` prints
    the same text with ``multibench`` commands.

    Vertical reads ``atac.h5``; ``method_info(m)["atac"]`` says whether it
    must hold peaks or gene activity. Diagonal reads ``atac_peak.h5`` (peaks)
    and ``atac_gas.h5`` (gene activity). Mosaic reads ``atac<i>.h5`` (peaks).
    ``peak.h5``, and ``atac.h5`` for gene activity, are accepted as older names.

    Mosaic and cross use one numbered file per batch in one folder, not
    sub-folders or one concatenated matrix. The file number is the batch:

    ```text
    <data_path>/COREBATCH/
        rna1.h5   adt1.h5   cty1.csv     # batch 1
        rna2.h5   adt2.h5   cty2.csv     # batch 2
        rna3.h5   adt3.h5   cty3.csv     # batch 3
    ```

    See Also
    --------
    mtb.list_categories : the four categories with a description of each.

    mtb.scan : checks a laid-out folder (``files_ok`` / ``files_reason`` per method).

    mtb.io.export_dataset : writes a whole dataset in this layout from an AnnData.
    """
    registry.check_category(category)
    cats = [category] if category else list(CATEGORIES)
    lines = ["Put the files of one dataset in one folder: <data_path>/<DATASET>/, "
             "for example ./data/MYDATA/.",
             "The dataset name is the folder name. data_path is the folder that "
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
              config.hint("from an AnnData or MuData. mtb.io.to_canonical writes one "
                          "file.",
                          "from an .h5ad or .h5mu file, or one file. multibench convert "
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
                  f"Next: multibench scan MYDATA --category {cat}, then, on Linux, "
                  f"multibench run-all MYDATA --category {cat} --out-dir out/")]
    return "\n".join(lines)


#: The modality files of the demo datasets (``mtb.data.fetch``); ``describe_layout``
#: names the demo that has each batch pattern.
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
    patterns: dict = {}
    for spec, v, _cat, mods in _variant_rows(category):
        if not mods:
            continue
        key = tuple(sorted((modality_family(m) for m in mods),
                           key=lambda r: (_batch_of(r) or 0,
                                          order.get(base_modality(r), 9), r)))
        ids = patterns.setdefault(key, [])
        if spec.id not in ids:
            ids.append(spec.id)
    return list(patterns.items())


def _atac_lines(category: str) -> list[str]:
    """The ATAC part of one category's layout: which representation each
    method needs, from the registry (``find_methods(category, atac=...)``)."""
    from .discover import find_methods
    peak = find_methods(category, atac="peak")
    gas = find_methods(category, atac="gene_activity")
    if not peak and not gas:
        return []
    out = []
    if category == "vertical" and gas:
        out.append("atac.h5 holds peaks or gene activity. Each method needs one of them:")
    elif category == "vertical":
        out.append("atac.h5 holds peaks, not gene activity: every vertical method that "
                   "reads ATAC needs peaks.")
    elif category == "diagonal":
        out.append("Give atac_peak.h5, atac_gas.h5 or both. Each method needs one of "
                   "them, or both:")
        both = list(dict.fromkeys(s.id for s, v, _c, mods in _variant_rows("diagonal")
                                  if {"atac_peak", "atac_gas"} <= set(mods)))
        if both:
            out.append(f"  need both files:       {', '.join(both)}")
        peak = [m for m in peak if m not in both]
        gas = [m for m in gas if m not in both]
    elif gas:
        out.append("atac<i>.h5 holds peaks or gene activity. Each method needs one of them:")
    else:
        out.append(f"atac<i>.h5 holds peaks: every {category} method that reads ATAC "
                   f"needs peaks.")
    if category == "diagonal" or gas:
        if peak:
            out.append(f"  need peaks:            {', '.join(peak)}")
        if gas:
            out.append(f"  need gene activity:    {', '.join(gas)}")
    out += ["A method whose ATAC file holds the other representation gives a wrong "
            "embedding."]
    # method_info(m)['atac'] names one representation, so it cannot tell a method
    # that reads both files
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


#: The call that says which ATAC representation a method reads.
_METHOD_INFO_ATAC = "method_info(m)['atac']"


def _check_atac_call() -> str:
    return config.hint(_METHOD_INFO_ATAC, "multibench info METHOD")


#: How ``describe_layout`` shows a batch's modalities as ``convert`` flags. A
#: ``.h5mu`` batch keeps its labels in the RNA modality's obs (muon's layout).
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
        lines.append("  In a .h5mu file, rna:cell_type reads the RNA modality's obs. "
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
                 "  rna.h5 + adt.h5 + atac.h5   all three from the same cells",
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
    lines = [head] + files
    if not full:
        return lines
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
    if category == "diagonal":
        several = _resolve._several_batch_diagonal_methods()
        if several:
            lines += ["For several RNA and several ATAC batches, number the files: "
                      "rna1.h5, atac_gas1.h5 or",
                      "atac_peak1.h5, rna_cty1.csv, atac_cty1.csv, then 2, 3 ... These "
                      "methods read them:",
                      f"  {', '.join(several)}"]
    lines += _atac_lines(category)
    return lines


# --------------------------------------------------------------------------- scan
def _installed_envs() -> frozenset:
    """The envs on this machine, read on every call so that an env installed
    mid-session counts at once; empty when the probe fails."""
    try:
        return frozenset(envs.installed_envs())
    except Exception:      # an env probe must not break discovery
        return frozenset()


def _other_env(env: str):
    """:func:`envs.other_env_prefix`, ``None`` when the probe fails."""
    try:
        return envs.other_env_prefix(env)
    except Exception:      # an env probe must not break discovery
        return None


def _missing_script(variant, *, method: str | None = None) -> str:
    """Why this variant's script or a helper module it imports is missing from
    the scripts checkout, or "" (also when there is no checkout yet: run fetches it)."""
    entry = Path(variant.entrypoint)
    for root in (Path(config.DEFAULT.repo_path), Path(config.__file__).resolve().parent.parent):
        if (root / "tools_scripts").is_dir():
            # a script the package ships is copied in before the run
            shipped = (method and entry.name in variant.helpers
                       and _runner.shipped_helper(method, entry.name))
            if not (root / entry).exists() and not shipped:
                return (f"method script {entry} is missing from the reference checkout at "
                        f"{root}: " + _runner.missing_script_fix(root))
            gone = [h for h in variant.helpers
                    if not (root / entry).parent.joinpath(h).exists()
                    and not (method and _runner.shipped_helper(method, h))]
            if gone:
                who = config.hint(
                    (f"mtb.method_info({method!r})" if method else "method_info(m)")
                    + "['setup_hint']",
                    f"multibench info {method or 'METHOD'}")
                name = f"{method}'s script" if method else f"The script {entry.name}"
                files = " and ".join(gone)
                one = len(gone) == 1
                return (f"{name} imports {files}. The public scMultiBench repository "
                        f"does not include {'it' if one else 'them'}. Put "
                        f"{'a ' if one else ''}{files} next to {entry.name}. {who} shows "
                        f"how.")
            return ""
    return ""


def _rows_word(df: "pd.DataFrame", k: int) -> str:
    """``method(s)`` when each method of the scan frame has one row, else ``row(s)``."""
    noun = "method" if len(df) and df["method"].is_unique else "row"
    return noun if k == 1 else noun + "s"


def _have_their(k: int, n: int) -> str:
    """The verb after ``k of n <rows>``: ``has its`` when ``k`` or ``n`` is 1."""
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


#: scan() columns. New columns go last, so positional readers keep working.
SCAN_COLUMNS = ["method", "category", "modalities", "env", "output_kind", "n_tunable",
                "runtime_tier", "observed_worst_sec", "caveat", "runnable", "reason",
                "files_ok", "files_reason", "env_ok", "env_reason", "needs_labels", "atac",
                "command"]

#: Default ``out_dir`` of :func:`scan`'s ``command`` column: a preview needs no
#: real directory.
OUT_DIR_PLACEHOLDER = "<out_dir>"


def _env_hint(env: str, method: str, category: str | None, other=None) -> str:
    """The ``env_reason`` text: the install command first, so a clipped reason
    still holds it; off Linux only the platform sentence; ``other`` = the
    prefix of a same-named env multibench did not install."""
    if _runner.linux_only_sentence():
        return f"Environment {env} runs only on Linux, not on this computer."
    if other is not None:
        return (f"{envs.other_env_text(env, other)} Rename or remove it, then run "
                f"multibench env install --methods {method} --packed --run.")
    alt = (f" Use --category {category} to install the environments of every "
           f"{category} method." if category else "")
    return (f"Environment {env} is not installed. Run "
            f"multibench env install --methods {method} --packed --run.{alt}")


#: ``.format(method=)`` caveat of a GPU-only row that ``scan(assume_gpu=True)``
#: keeps runnable.
GPU_NODE_CAVEAT = ("{method} needs an NVIDIA GPU. This check assumes the job runs on a "
                   "GPU node.")

#: What the first line of ``scan(per_method=True)`` adds off Linux.
LINUX_ONLY_CHECK = ("Method environments run only on Linux. On this computer you can "
                    "check files, score embeddings and plot.")
#: What the scan summary line adds off Linux, after the counts.
LINUX_ONLY_SUMMARY = (f"{LINUX_ONLY_CHECK} The commands use this computer's paths, so "
                      "run scan again on the Linux machine.")


def _first_sentence(text: str) -> str:
    """The first sentence of ``text``; a dotted file name does not end it."""
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


def _missing_files_reason(spec, category: str, mods: list, dataset: str,
                          data_path) -> str:
    """The ``reason`` for input files that are not on disk, or ``""``: ``SCALEX needs
    gene-activity ATAC (atac_gas.h5), and the folder has peaks (atac_peak.h5).``"""
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
    has_other = {}         # (need, kinds found) -> its part: numbered files share one
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
        need = f"{spec.id} needs {want + ' ' if want else ''}ATAC"
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
            kinds = tuple(_atac_kind_of(f) for f in found)
            part = has_other.get((need, kinds))
            if part is None:
                part = has_other[need, kinds] = ([], [[] for _ in found])
                atac_parts.append(part)
            part[0].append(want_file)
            for names, f in zip(part[1], found):
                names.append(f.name)
        else:
            atac_parts.append(f"{need} ({want_file}), which is not in the folder")
    for (need, kinds), part in has_other.items():
        has = ", ".join(f"{k} ({', '.join(names)})" for k, names in zip(kinds, part[1]))
        atac_parts[atac_parts.index(part)] = (
            f"{need} ({', '.join(part[0])}), and the folder has {has}")
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
    """The warning of ``scan(modalities=...)`` for the methods that read a folder."""
    one = len(dropped) == 1
    mods = _and_list([str(m) for m in modalities])
    head = (f"The modality {mods} leaves" if len(modalities) == 1
            else f"The modalities {mods} leave")
    fix = config.hint(
        f"Pass modalities=[] to select {'it' if one else 'them'}, or leave out "
        f"modalities= to see every variant.",
        f'Pass --modalities "" to select {"it" if one else "them"}, or leave out '
        f"--modalities to see every variant.")
    return (f"{head} out {_and_list(dropped)}, which "
            f"{'reads' if one else 'read'} a folder instead of modality files. {fix}")


def _short_reason(text: str, method: str, dataset: str) -> str:
    """A ``files_reason`` part as the ``reason`` column shows it: without the
    exception class, directories, or the ``<method> reads <file> of <dataset>`` opening."""
    if not text:
        return text
    part = _EXC_PREFIX_RE.sub("", text)
    part = re.sub(rf"^{re.escape(method)} reads (\S+) of {re.escape(str(dataset))}, which ",
                  r"\1 ", part)
    return _ABS_PATH_RE.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], part)


#: The representation-mismatch caveats of ``_resolve._preflight_caveats`` after the
#: method name; groups: wanted, file, held.
_WRONG_ATAC_BODY = (r"needs (gene-activity|peak) ATAC\. The features of (\S+) (?:do not )?"
                    r"look like chr:start-end, so it (?:seems to hold|holds) "
                    r"(peaks|gene activity)\.")
_WRONG_ATAC_RE = re.compile(r"^\S+ " + _WRONG_ATAC_BODY)
#: What a peak-name caveat (``_resolve.PEAK_NAMES_CAVEAT``) says after the method
#: name; ``cli._strict_problem`` counts these rows apart.
PEAK_NAMES_REASON = "reads peak names such as "
_PEAK_NAMES_RE = re.compile(r"^\S+ " + re.escape(PEAK_NAMES_REASON)
                            + r"(\S+)\. (\S+) holds other names")
#: The override every blocking-ATAC reason names (Python, CLI); what
#: :func:`_is_wrong_atac` looks for.
_ATAC_OVERRIDE = ("allow_atac_mismatch=True", "--allow-atac-mismatch")


def _blocking_caveats(caveats) -> list[str]:
    """The caveats that block a row: the ATAC file holds the other
    representation, or peak names the method's script cannot read."""
    return [c for c in caveats if _WRONG_ATAC_RE.match(c) or _PEAK_NAMES_RE.match(c)]


def _wrong_atac_reason(method: str, caveats) -> str:
    """The ``reason`` of a row whose ATAC file the method cannot read (the other
    representation, or peak names it cannot parse), or ""; the override comes last."""
    anyway = (", or pass " + config.hint(_ATAC_OVERRIDE[0], _ATAC_OVERRIDE[1])
              + f" to run {method} anyway.")
    for text in _blocking_caveats(caveats):
        m = _WRONG_ATAC_RE.match(text)
        if m:
            as_ = "gene activity" if m.group(1) == "gene-activity" else "peaks"
            return (f"{method} needs {m.group(1)} ATAC, and {m.group(2)} holds "
                    f"{m.group(3)}. Export the ATAC as {as_}" + anyway)
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
    """The blocking caveats of the files in a ``run`` call's ``inputs``; ``[]`` when
    none or unreadable. In-memory inputs are not checked."""
    try:
        spec = registry.get(method)
        paths = {r: str(p) for r, p in inputs.items()
                 if isinstance(p, (str, os.PathLike)) and Path(p).exists()}
        return _blocking_caveats(_resolve._preflight_caveats(
            paths, atac=spec.atac, category=category, method=method))
    except Exception:  # noqa: BLE001 - a check must never stop the run
        return []


def _run_caveat(caveat) -> str:
    """A scan ``caveat`` without the trailing notes on starting the method
    (scripts not here, a prepared input), which ``run_all`` handles itself."""
    text = str(caveat or "")
    cuts = [i for i in (text.find(_runner.SCRIPTS_NOT_HERE), _runner._prepared_at(text))
            if i >= 0]
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
    """The dry-run note that the method scripts are not at
    ``$MULTIBENCH_SCRIPTS_REF`` or their folder cannot be filled, else ``None``."""
    note = config.scripts_ref_problem() or config.scripts_folder_problem()
    return _join_sentences([note]) if note else None


def _dry_run_notes(plan: "pd.DataFrame", modalities=None) -> tuple:
    """``(scripts_note, [(method, caveat)])``: the note that the method scripts are
    not here yet, and the caveat of each row the sweep would run (nested rows out)."""
    scripts = None
    lines = []
    left_out = set()
    if modalities is None and len(plan):
        left_out = set(_nested_rows(plan[[_would_run(r) for _, r in plan.iterrows()]])[0])
    for index, r in plan.iterrows():
        text = str(r.get("caveat") or "")
        start = text.find(_runner.SCRIPTS_NOT_HERE)
        if scripts is None and start >= 0:
            end = _runner._prepared_at(text[start:])
            scripts = text[start:][:end].rstrip() if end >= 0 else text[start:]
        caveat = _run_caveat(text)
        if _would_run(r) and caveat and index not in left_out:
            lines.append((r["method"], caveat))
    return scripts, lines


def _variant_consumes_atac(variant) -> bool:
    """Whether this variant takes an ATAC input, judged on its declared modalities
    (Seurat_WNN's rna+adt variant passes ``NULL`` to its ATAC slot)."""
    mods = variant.when.get("modalities") or []
    if mods:
        return any(_schema_base(m) == "atac" for m in mods)
    return any(a.const and "atac" in str(a.const) for a in variant.args)


def _modality_matcher(modalities, named=None):
    """A test ``(spec, variant_modalities) -> bool`` for ``scan(modalities=)``; the
    rules are those of the scan docstring. ``named`` are the ``methods=`` ids."""
    from .discover import _modality_filter
    from .engine.schema import modality_family
    _modality_filter(modalities, None)      # ValueError / TypeError as in find_methods
    exact = set(registry.normalize_modalities(modalities))
    spelled = {str(t).lower() for t in modalities}
    named = set(named or ())
    tokens = []                             # (family, representation, is a base token)
    for given in modalities:
        token = registry.MODALITY_ALIASES.get(str(given).lower(), str(given))
        family = modality_family(token)
        tokens.append((family, _REPRESENTATION_TOKENS.get(token.rstrip("0123456789")),
                       family in ("rna", "adt", "atac")))
    reps = _token_reps(modalities)

    def covers(token, role):
        family, _rep, is_base = token
        return (_schema_base(role) == family) if is_base else modality_family(role) == family

    def match(spec, mods) -> bool:
        if not mods or not tokens:
            return not mods and not tokens
        if spec.id in named and spelled == set(mods):
            return True
        # the representation before the spelling: moETM's atac_gas role reads peaks
        if len(reps) == 1 and spec.atac not in reps:
            return False
        if exact == set(mods):
            return True
        if any(rep and spec.atac != rep for _f, rep, _b in tokens):
            return False
        return (all(any(covers(t, r) for r in mods) for t in tokens)
                and all(any(covers(t, r) for t in tokens) for r in mods))
    return match


#: ``method_info(m)['atac']`` -> representation token stem
_TOKEN_OF_REP = {v: k for k, v in _REPRESENTATION_TOKENS.items()}


def _token_reps(modalities) -> set:
    """The ATAC representations (``peak``, ``gene_activity``) the tokens name."""
    out = set()
    for given in modalities or ():
        token = registry.MODALITY_ALIASES.get(str(given).lower(), str(given))
        rep = _REPRESENTATION_TOKENS.get(token.rstrip("0123456789"))
        if rep:
            out.add(rep)
    return out


def _representation_note(category, methods, modalities) -> str:
    """Why a representation token dropped the named ``methods`` (what each reads,
    and the list to pass), or ""."""
    if not methods or not modalities:
        return ""
    tokens = [registry.MODALITY_ALIASES.get(str(t).lower(), str(t).lower())
              for t in modalities]

    def rep(token):
        return _REPRESENTATION_TOKENS.get(token.rstrip("0123456789"))
    reps = {rep(t) for t in tokens if rep(t)}
    if len(reps) != 1:
        return ""
    loose = _modality_matcher(["atac" if rep(t) else t for t in tokens])
    notes, seen = [], set()
    for spec, _v, _cat, mods in _variant_rows(category):
        if (spec.id not in methods or spec.id in seen or not spec.atac
                or spec.atac in reps or not loose(spec, mods)):
            continue
        seen.add(spec.id)
        other = next((r for r in mods if rep(r) and rep(r) != spec.atac), None)
        via = f" through its {other} input" if other else ""
        fixed = [_TOKEN_OF_REP[spec.atac] if rep(t) else t for t in tokens]
        base = ["atac" if rep(t) else t for t in tokens]
        notes.append(f"{spec.id} reads {_resolve._READS[spec.atac]}{via}. Pass " + config.hint(
            f"modalities={fixed!r} or {base!r}.",
            f"--modalities {','.join(fixed)} or {','.join(base)}."))
    return " ".join(notes)


def _no_variant_error(category, methods, modalities) -> ValueError:
    """The ``ValueError`` for named methods without a row, or a selection that
    matches no variant: what does not run, then why."""
    from .engine.schema import no_category_message
    where = f"{category} data" if category else "any category"
    methods = list(methods or ())
    cats = {m: dict.fromkeys(c for s, _v, c, _m in _variant_rows() if s.id == m and c)
            for m in methods}
    # no variant under this category, whatever the modalities
    off = [m for m in methods
           if modalities is None or (category and category not in cats[m])]
    dropped = [m for m in methods if m not in off]
    parts = [no_category_message(m, cats[m], category) for m in off]
    if modalities is not None and (dropped or not methods):
        mods = "+".join(map(str, modalities or ())) or "a data folder"
        parts.append(f"{_and_list(dropped)} {'does' if len(dropped) == 1 else 'do'} not read "
                     f"{mods} in {where}." if dropped else
                     f"No method reads {mods} in {where}.")
        one = dropped[0] if len(dropped) == 1 else None
        parts.append(_representation_note(category, dropped, modalities) or config.hint(
            f"mtb.method_info({one!r})['supports'] lists what {one} reads." if one else
            "mtb.method_info(m)['supports'] lists what each method reads.",
            f"multibench info {one} lists what {one} reads." if one else
            "multibench info METHOD lists what each method reads."))
    return ValueError(" ".join(parts) or f"No method runs on {where}.")


def _command_line(method: str, category: str, inputs: dict, *, out_dir, dataset: str,
                  params: dict | None, gpu: bool | None = None) -> tuple[str, list[str]]:
    """The shell line ``run`` would execute for one scan row, and the notes on
    files that ``run`` writes first; ``(no preview: ...)`` when it cannot be built."""
    try:
        # runner.preview, not _run: a preview is not a dispatch
        argv, notes = _runner.preview(method, category, inputs=inputs,
                                      out_dir=Path(out_dir) / f"{method}_{dataset}",
                                      params=params, gpu=gpu)
        prepared = [n for n in notes if _runner._prepared_at(n) == 0]
        return shlex.join(argv), prepared
    except Exception as e:  # noqa: BLE001 - a preview must never abort the scan
        return f"(no preview: {type(e).__name__}: {e})", []


#: what :func:`_scan` keeps of each row besides the shown columns (its second frame)
_PRIVATE_COLUMNS = ["_other", "_label_inputs", "_folder_caveat"]

_FAMILY_WORDS = {"rna": "RNA", "adt": "ADT", "atac": "ATAC"}


def _every_combination(modalities) -> tuple:
    """``(every, modalities)``: ``"all"`` becomes ``(True, None)``; another bare
    string is left for the id-list check to reject."""
    if isinstance(modalities, str) and modalities == "all":
        return True, None
    return False, modalities


def _row_modalities(text) -> list:
    """A scan row's ``modalities`` as a list; ``[]`` for a row that reads a folder."""
    return [] if text == "(data_dir)" else str(text).split("+")


def _family(role) -> str:
    """The modality family of a role: ``rna2`` -> ``rna``, ``atac_peak`` -> ``atac``."""
    return _schema_base(str(role)).rstrip("0123456789")


def _row_families(modalities) -> set:
    """The families a scan row reads; none for a row that reads a folder."""
    roles = _row_modalities(modalities)
    return {_family(r) for r in roles} & set(registry._BASE_MODALITIES)


def _role_files(ds_dir: Path, category: str | None) -> dict:
    """``{file name: family}`` of the matrix files a role of some variant of
    ``category`` resolves to; an absent ATAC role counts its batch's other ATAC name."""
    from .engine.schema import is_label_role
    files = {}
    for spec, v, *_ in _variant_rows(category):
        for role, path in _resolve._resolve_variant_inputs(v, ds_dir, spec.id).items():
            fam, p = _family(role), Path(path)
            if fam not in registry._BASE_MODALITIES or is_label_role(role):
                continue
            if p.name not in files and p.exists():
                files[p.name] = fam
            elif fam == "atac" and not p.exists():
                digits = role[len(role.rstrip("0123456789")):]
                for base in _resolve._ATAC_FILE_BASES:
                    q = ds_dir / f"{base}{digits}.h5"
                    if q.name not in files and q.is_file():
                        files[q.name] = fam
    return files


def _held_families(ds_dir: Path, category: str | None) -> list:
    """The families the folder holds a file of (:func:`_role_files`)."""
    held = set(_role_files(ds_dir, category).values())
    return [f for f in registry._BASE_MODALITIES if f in held]


def _listed(df: "pd.DataFrame", need, held, methods) -> list:
    """One flag per row: whether a scan without ``modalities`` lists it (the rule
    of the scan docstring, Listed rows). ``need`` is each row's families."""
    keep = [n <= set(held) or bool(ok) for n, ok in zip(need, df["files_ok"])]
    if methods:
        left = set(df["method"][keep])
        keep = [k or m not in left for k, m in zip(keep, df["method"])]
    if not held or not any(keep):
        return [True] * len(df)
    return keep


def _tenx_files(ds_dir: Path, files: dict) -> list:
    """The names in ``files`` that are 10x Genomics files, by batch, then RNA, ADT, ATAC."""
    from .engine.ingest import _is_canonical_h5, _is_tenx_h5

    def order(name):
        stem = name.rsplit(".", 1)[0]
        return int(stem[len(stem.rstrip("0123456789")):] or 0), registry._BASE_MODALITIES.index(files[name])
    return sorted((n for n in files if not _is_canonical_h5(ds_dir / n)
                   and _is_tenx_h5(ds_dir / n)), key=order)


def _holds_sentence(dataset: str, ds_dir: Path, held, category: str | None = None) -> str:
    """``D52 holds RNA and ADT, in batches 1-3.``; a 10x Genomics file is named as
    such and does not count for its family."""
    if not held:
        if not _cells_per_file(ds_dir):
            return f"{dataset} holds no RNA, ADT or ATAC file."
        return (f"{dataset} holds no RNA, ADT or ATAC file under a name that a "
                f"{category + ' ' if category else ''}method reads.")
    files = _role_files(ds_dir, category)
    tenx = _tenx_files(ds_dir, files)
    is_tenx = ""
    if tenx:
        is_tenx = f" {_and_list(tenx)} " + (
            "is a 10x Genomics file under the name of a dataset file." if len(tenx) == 1
            else "are 10x Genomics files under the names of dataset files.")
        held = [f for f in held
                if any(fam == f and n not in tenx for n, fam in files.items())]
        if not held:
            return f"{dataset} holds no dataset file:{is_tenx}"
    nums = sorted(_resolve._folder_batches(ds_dir))
    where = (f", in {'batch' if len(nums) == 1 else 'batches'} {_resolve._span(nums)}"
             if nums else "")
    return (f"{dataset} holds {_and_list(_FAMILY_WORDS[f] for f in held)}{where}."
            + is_tenx)


def _all_spelling() -> str:
    """How a caller asks for every combination."""
    return config.hint('modalities="all"', "--modalities all")


def _lacking(families) -> str:
    """``ATAC``, ``ADT or ATAC``: the families as words."""
    return _and_list([_FAMILY_WORDS[f] for f in families], "or")


def _not_listed_note(df: "pd.DataFrame", dataset: str, ds_dir: Path,
                     category: str | None = None) -> str:
    """What the folder holds, how many rows are not listed, and how to list them."""
    d, one = df.attrs["not_listed"], df.attrs["not_listed"] == 1
    return (f"{_holds_sentence(dataset, ds_dir, df.attrs['held'], category)} {d} "
            f"{'row that reads' if one else 'rows that read'} "
            f"{_lacking(df.attrs['not_held'])} {'is' if one else 'are'} not listed. "
            f"{_all_spelling()} lists {'it' if one else 'them'}.")


def _scan_note(df: "pd.DataFrame", dataset: str, data_path=None,
               category: str | None = None) -> str:
    """:func:`_not_listed_note` from a scan frame and the dataset as typed (the
    command line); ``""`` when every row is listed."""
    if not df.attrs.get("not_listed"):
        return ""
    base = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # scan warned about the spelling
        dataset = _resolve.canonical_dataset(base, dataset)
    return _not_listed_note(df, dataset, base / dataset, category)


def scan(dataset: str, category: str | None = None, *,
         methods: list[str] | None = None,
         modalities: list[str] | str | None = None,
         data_path: Path | str | None = None,
         out_dir="<out_dir>",
         params: dict | None = None,
         verbose: bool = True,
         assume_gpu: bool = False,
         allow_atac_mismatch: bool = False,
         per_method: bool = False) -> pd.DataFrame:
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
    modalities : list[str] | str | None
        Modality tokens of one combination, e.g. ``["rna", "adt"]``; ``None`` =
        the combinations whose modalities the folder holds; ``"all"`` = every
        combination.
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
    per_method : bool
        ``True`` = one row per method with the columns ``method``,
        ``runnable`` and ``reason``; needs ``category``.

    Returns
    -------
    pandas.DataFrame
        One row per (method, category, modalities) variant, runnable rows
        first. Read ``df[["method", "modalities", "runnable", "reason"]]``;
        all 18 columns are listed in Notes. ``df.attrs`` holds ``held``,
        ``not_listed`` and ``not_held``, and with ``per_method=True`` also
        ``cells`` (Notes).

    Raises
    ------
    FileNotFoundError
        ``<data_path>/<dataset>`` does not exist; the message lists the folders present.
    ValueError
        Unknown ``category`` or modality token.
    ValueError
        A method in ``methods`` has no variant in ``category`` or ``modalities``,
        even when other named methods have one.
    ValueError
        ``per_method=True`` without ``category``.
    KeyError
        Unknown id in ``methods`` or ``params``, or a ``params`` key no variant accepts.
    TypeError
        ``methods`` given as a bare string, or ``modalities`` as a string
        other than ``"all"``.

    Warns
    -----
    UserWarning
        ``dataset`` matches a folder only up to letter case.
    UserWarning
        ``modalities`` leaves out scBridge, which reads a folder, without excluding its ATAC form.

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
    >>> mtb.scan("D11", "vertical", modalities="all")   # every combination
    >>> # one row per method: can it run on this folder?
    >>> mtb.scan("MYCITE", "vertical", methods=["Matilda", "totalVI"],
    ...          data_path="/path/to/data", per_method=True)

    Notes
    -----
    **Columns.**

    ```text
    method              method id
    category            integration category of the variant
    modalities          '+'-joined roles ("rna+adt"); run_all and inputs_for take
                        a list, so split on "+". "(data_dir)" marks a variant
                        that reads a whole folder (scBridge): pass no modalities
    env                 the conda env the method runs in
    output_kind         embedding / graph
    n_tunable           number of command-line hyperparameters
    runtime_tier        fast / medium / slow / very_slow / unknown
    observed_worst_sec  the slowest observed run, seconds (None = unmeasured)
    caveat              what the run needs besides the files, or ""
    runnable            both checks pass and nothing below blocks it
    reason              short form of the non-empty reasons; "" when runnable
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

    **Listed rows.** Without ``modalities`` the frame lists the variants
    whose modalities the folder holds, and every variant whose input files
    are all there, so the ``files_ok`` rows are those of
    ``modalities="all"``. A method named in ``methods`` keeps its rows when
    none would be listed. When the folder holds no RNA, ADT or ATAC file, or
    no row would be left, every variant is listed. ``df.attrs`` says what was
    left out:

    ```text
    held        the modalities the folder holds, e.g. ['rna', 'adt']
    not_listed  how many variants are not listed
    not_held    the modalities those variants read and the folder does not
                hold, e.g. ['atac']
    cells       per_method=True only: {file name: cells} of the matrix
                files in the folder; None for a file that cannot be read
    ```

    **One row per method.** With ``per_method=True``, ``runnable`` is
    ``True`` when one of the method's listed variants is runnable, which is
    when ``run_all`` tries the method. Otherwise ``reason`` is that of one
    variant: one whose input files are all in the folder, else one that
    reads the modalities the folder holds, else the one that lacks the
    fewest files. Runnable methods come first, then by name. The printout
    adds the cells of each matrix file and the full reason of each method
    that cannot run.

    **The two checks.** ``files_ok``: the method's script is present, the
    input files resolve on disk and are oriented features x cells, no
    matrix file is a 10x Genomics ``.h5`` under the name of a dataset file
    or holds 0 features or 0 cells, every label CSV has one row per cell of
    the modality it labels, and a ``data_dir`` method (scBridge) finds the
    files it names. Seurat_v5's ``rna.h5`` and ``atac_peak.h5`` must hold
    the same cells; a diagonal ``atac_gas.h5`` must list the cells of
    ``atac_peak.h5`` in its order. ``env_ok``: the method's conda env exists
    and multibench installed it, on Linux; ``env_reason`` names the env and
    the one-method install command (``multibench env install --methods X
    --packed --run``), names an env of the same name that multibench did not
    install (it is not used), or on macOS and Windows says it runs only on
    Linux (``multibench env doctor`` checks every env). For
    a method with ``requires_gpu``, the host also needs an NVIDIA GPU. Without
    one, ``env_reason`` is the sentence ``run`` would raise: ``"<method>
    needs an NVIDIA GPU, and this computer has none. See ..."``. With
    ``assume_gpu=True`` the row's ``caveat`` says ``This check assumes the
    job runs on a GPU node.`` and ``command`` leaves out the CPU flags.

    **Caveats.** ``caveat`` lists what a row with ``files_ok`` still needs,
    or what may go wrong without an error: an ATAC file that holds the other
    representation, or peak names the method cannot read; values that are
    not whole numbers (the methods expect raw counts); a diagonal folder
    whose only label file is ``cty.csv``; a method that reads fewer
    numbered batches than the folder holds; a step to do first (the first
    sentence of ``method_info(m)['setup_hint']``); method scripts not on
    this machine yet (the first real run clones them with ``git``; on a host
    without network, fetch them first with ``multibench fetch --scripts``); a
    command that reads a file ``mtb.run`` writes first, so start that
    method with ``mtb.run``.

    A row given the other ATAC representation is not runnable, also when
    ``methods=`` names the method, and ``run_all`` skips it. So is a GLUE or
    Seurat_v3 row whose peak names are not chr:start-end.
    ``allow_atac_mismatch=True`` keeps such a row runnable, with its caveat.
    With ``MULTIBENCH_SCRIPTS_REF`` set to another commit than the method
    scripts, or a ``repo_path`` that holds other files and no method
    scripts, no row is runnable.

    **Commands.** ``command`` is ``run(..., dry_run=True)``, ``shlex``-joined,
    writing under ``<out_dir>/<method>_<dataset>/`` as ``run_all`` does, with
    absolute paths, ``params`` merged and, on a host without a GPU, each
    method's ``cpu_params``. A row blocked only by ``env_ok`` still shows its
    command. ``multibench scan`` prints compact columns; ``--columns all``
    adds ``command`` and the rest.

    **Modality tokens.** A list of ``modalities`` names one combination, in
    any order. Base tokens keep a row whose modalities are exactly that
    combination. They are ``rna``, ``adt`` (or ``protein``) and ``atac``,
    which matches every ATAC role (``atac_gas``, ``atac_peak``, ``atac2`` ...).
    Representation tokens select by what the method reads: ``atac_peak`` /
    ``peak`` and ``atac_gas`` / ``gas`` / ``gene_activity`` keep the methods
    whose ``method_info(m)['atac']`` is that representation, like
    ``find_methods(atac=...)``. ``atac_peak`` alone also keeps MultiMAP and
    Seurat_v3, which read both files; ``atac_peak`` together with
    ``atac_gas`` keeps only those two. moETM, scMM and iPOLNG read peaks
    through a role named ``atac_gas``, so ``atac_peak`` selects them and
    ``atac_gas`` does not, unless ``methods=`` names them. A numbered token
    (``rna1``) matches that role only. ``modalities=[]`` selects the
    variants that read a folder. Other lists drop them, with a
    ``UserWarning`` unless the tokens already exclude their ATAC
    representation. ``mtb.find_methods`` keeps every method that reads at
    least the named modalities, so the two can list different methods.

    See Also
    --------
    mtb.run_all : run the runnable rows, with metrics; ``dry_run=True`` returns this frame.

    mtb.describe_layout : how to lay out a dataset folder so ``files_ok`` passes.

    mtb.env.doctor : the env check on its own, per env.

    mtb.inputs_for : the ``{role: path}`` resolution behind ``files_ok``.
    """
    if per_method and category is None:
        raise ValueError(f"per_method=True needs a category, as "
                         f"mtb.scan({dataset!r}, 'vertical', per_method=True).")
    df, private = _scan(dataset, category, methods=methods, modalities=modalities,
                        data_path=data_path, out_dir=out_dir, params=params,
                        verbose=verbose and not per_method, assume_gpu=assume_gpu,
                        allow_atac_mismatch=allow_atac_mismatch, stacklevel=3)
    if not per_method:
        return df
    return _per_method(df, private, data_path=data_path, verbose=verbose,
                       nested=not isinstance(modalities, (list, tuple)))


def _scan(dataset: str, category: str | None = None, *, methods=None, modalities=None,
          data_path=None, out_dir="<out_dir>", params: dict | None = None,
          verbose: bool = True, assume_gpu: bool = False,
          allow_atac_mismatch: bool = False,
          stacklevel: int = 2) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """:func:`scan`'s frame, and a second frame aligned with it: ``_other`` (the
    reason without the off-Linux sentence), ``_label_inputs`` and ``_folder_caveat``."""
    registry.check_category(category)
    registry.check_id_list(methods, "methods")
    every, modalities = _every_combination(modalities)
    registry.check_id_list(modalities, "modalities")
    if methods is not None:
        methods = [registry.check_method(m) for m in methods]
    params = params or {}
    for method in params:                   # did-you-mean KeyError before any I/O
        registry.check_method(method)
    wanted = _modality_matcher(modalities, methods) if modalities is not None else None
    base = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    dataset = _resolve.canonical_dataset(base, dataset, stacklevel=stacklevel + 1)
    ds_dir = base / dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(_resolve.missing_dataset_message(base, dataset))
    installed = _installed_envs()
    repo = _runner._repo_root_no_fetch()
    label_files = {q.resolve(): q.name for q in _label_files(dataset, data_path)}
    # the real run refuses these scripts, so every row is blocked
    wrong_ref = config.scripts_ref_problem(repo) or config.scripts_folder_problem(repo) or ""
    rows = []
    dropped_dirs: list[str] = []
    off_linux = bool(_runner.linux_only_sentence())
    others: dict = {}          # env -> a same-named env multibench did not install
    reps = _token_reps(modalities) if modalities is not None else set()
    for spec, v, cat, mods in _variant_rows(category):
        if methods is not None and spec.id not in methods:
            continue
        mod_str = "+".join(mods) or "(data_dir)"
        if wanted is not None and not wanted(spec, mods):
            # one representation token already excludes a method that reads the other
            other_rep = len(reps) == 1 and spec.atac and spec.atac not in reps
            if not mods and not other_rep and spec.id not in dropped_dirs:
                dropped_dirs.append(spec.id)
            continue
        runtime = _runtimes().get(spec.id, {})
        rec = {"method": spec.id, "category": cat, "modalities": mod_str,
               "env": envs.group_for(spec.id), "output_kind": v.output.kind,
               "n_tunable": len(v.tunable),
               "runtime_tier": runtime.get("tier", "unknown"),
               "observed_worst_sec": runtime.get("worst_sec"),
               "caveat": "", "runnable": False, "reason": "",
               "files_ok": True, "files_reason": "", "env_ok": True, "env_reason": "",
               "needs_labels": bool(v.needs_labels),
               "atac": spec.atac if _variant_consumes_atac(v) else None,
               "command": ""}
        # check 1, files: both the script and the dataset files, so neither hides
        # the other. files_reason keeps the full text; reason the short form.
        file_problems, short_problems = [], []
        why_script = _missing_script(v, method=spec.id)
        if why_script:
            file_problems.append(why_script)
            short_problems.append(_short_reason(why_script, spec.id, dataset))
        got, wrong_atac = None, ""
        rec["_label_inputs"], rec["_folder_caveat"] = (), ""
        try:
            got = _resolve.inputs_for(dataset, cat, spec.id, modalities=mods or None,
                                      data_path=data_path, check=True)
            rec["_label_inputs"] = _label_inputs(got, v, label_files)
            extra = _resolve._preflight_caveats(got, atac=spec.atac, category=cat,
                                                method=spec.id)
            if extra:
                rec["caveat"] = _join_sentences([rec["caveat"], *extra])
                rec["_folder_caveat"] = _join_sentences(extra)
            if not allow_atac_mismatch:
                wrong_atac = _wrong_atac_reason(spec.id, extra)
        except Exception as e:  # missing files, no variant, bad layout
            full = f"{type(e).__name__}: {e}"
            file_problems.append(full)
            short_problems.append(
                _missing_files_reason(spec, cat, mods, dataset, data_path)
                or _short_reason(full, spec.id, dataset))
        if file_problems:
            rec["files_ok"], rec["files_reason"] = False, " ".join(file_problems)
        env_hint = gpu_hint = ""
        if rec["env"] and (off_linux or rec["env"] not in installed):
            rec["env_ok"] = False
            if not off_linux and rec["env"] not in others:
                others[rec["env"]] = _other_env(rec["env"])
            env_hint = _env_hint(rec["env"], spec.id, cat, others.get(rec["env"]))
        # a script that calls CUDA unconditionally: run() raises this sentence
        if spec.requires_gpu and not envs.host_has_gpu():
            if assume_gpu:
                rec["caveat"] = _join_sentences([rec["caveat"],
                                               GPU_NODE_CAVEAT.format(method=spec.id)])
            else:
                rec["env_ok"] = False
                gpu_hint = spec.requires_gpu_reason or ""
        rec["env_reason"] = " ".join(r for r in (env_hint, gpu_hint) if r)
        rec["runnable"] = bool(rec["files_ok"] and rec["env_ok"] and not wrong_atac
                               and not wrong_ref)
        # the ATAC reason ends with its override, so it comes last
        rec["reason"] = _join_sentences([wrong_ref, *short_problems, rec["env_reason"],
                                         wrong_atac])
        rec["_other"] = _join_sentences([wrong_ref, *short_problems,
                                         gpu_hint if off_linux else rec["env_reason"],
                                         wrong_atac])
        # the command line, also for an env-blocked row; setup steps go to caveat
        if rec["files_ok"] and got is not None:
            rec["command"], prepared = _command_line(spec.id, cat, got, out_dir=out_dir,
                                                     dataset=dataset,
                                                     params=params.get(spec.id),
                                                     gpu=True if assume_gpu else None)
            # the setup hint's first sentence; a same-cells hint (Seurat_v5) is
            # checked on the files above instead
            notes = [n if n != spec.setup_hint else
                     "" if spec.id in _resolve._SAME_CELL_ROLES else _first_sentence(n)
                     for n in _runner.script_notes(spec, v, repo)]
            notes = [n for n in notes if n] + prepared
            if notes:
                rec["caveat"] = _join_sentences([rec["caveat"], *notes])
        rows.append(rec)
    df = pd.DataFrame(rows, columns=[*SCAN_COLUMNS, *_PRIVATE_COLUMNS])
    df = df.sort_values(["runnable", "category", "method"],
                        ascending=[False, True, True]).reset_index(drop=True)
    private = pd.concat([df.pop(c) for c in _PRIVATE_COLUMNS], axis=1)
    if methods is not None:
        # a named method with no row is a request problem, also next to others
        missing = [m for m in dict.fromkeys(methods) if m not in set(df["method"])]
        if missing or df.empty:
            raise _no_variant_error(category, missing or methods, modalities)
    if params:
        _check_param_keys(df, params)
    if dropped_dirs and modalities:
        warnings.warn(_dropped_dirs_message(modalities, dropped_dirs),
                      UserWarning, stacklevel=stacklevel)
    # last, so every check above saw every variant
    held = _held_families(ds_dir, category)
    need = [_row_families(m) for m in df["modalities"]]
    keep = ([True] * len(df) if every or modalities is not None
            else _listed(df, need, held, methods))
    left_out = [(m, [f for f in registry._BASE_MODALITIES if f in n and f not in held])
                for m, n, k in zip(df["method"], need, keep) if not k]
    if left_out:
        df = df[keep].reset_index(drop=True)
        private = private[keep].reset_index(drop=True)
    df.attrs.update(held=held, not_listed=len(left_out),
                    not_held=[f for f in registry._BASE_MODALITIES
                              if any(f in lack for _, lack in left_out)])
    private.attrs.update(dataset=dataset, folder=ds_dir, unlisted=left_out,
                         category=category)
    if verbose:
        n, k_files, k_env = len(df), int(df["files_ok"].sum()), int(df["env_ok"].sum())
        rows = _rows_word(df, n)
        line = (f"[scan] {k_files} of {n} {rows} {_have_their(k_files, n)} input files. "
                f"{k_env} of {n} {_have_their(k_env, n)} environment installed.")
        if left_out:
            line += " " + _not_listed_note(df, dataset, ds_dir, category)
        if _runner.linux_only_sentence():
            line += f" {LINUX_ONLY_SUMMARY}"
        elif any(e and e not in installed for e in df["env"]):
            doctor = config.hint("mtb.env.doctor()", "multibench env doctor")
            line += f" {doctor} checks the environments."
        print(line, flush=True)
    return df, private


def _row_paths(row, dataset, data_path) -> dict:
    """The unchecked ``inputs_for`` paths of a scan row, without its warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _resolve.inputs_for(dataset, row["category"], row["method"],
                                   modalities=_row_modalities(row["modalities"]) or None,
                                   data_path=data_path, check=False)


def _missing_inputs(row, dataset, data_path) -> int:
    """How many input files of a scan row are not in the folder."""
    if row["files_ok"]:
        return 0
    try:
        paths = _row_paths(row, dataset, data_path)
    except Exception:  # noqa: BLE001 - unresolvable: the last row to speak for its method
        return 10 ** 6
    return sum(not Path(p).exists() for p in paths.values())


def _cells_per_file(ds_dir: Path) -> dict:
    """``{file name: cells}`` of the layout's matrix files in ``ds_dir``, in layout
    order; ``None`` for a file that cannot be read."""
    from .engine.ingest import _layout_order
    names = sorted((q.name for q in ds_dir.iterdir() if q.suffix == ".h5" and q.is_file()
                    and _resolve._LAYOUT_FILE_RE.match(q.name)), key=_layout_order)
    cells = {}
    for name in names:
        sniff = _resolve._sniff_h5(str(ds_dir / name), (ds_dir / name).stat().st_mtime_ns)
        cells[name] = sniff[2] if sniff else None
    return cells


#: how many files the ``Cells per file`` line of ``scan(per_method=True)`` names
_CELLS_SHOWN = 8


def _cells_line(cells: dict) -> str:
    """``Cells per file: rna.h5 1,718, adt.h5 1,718.``; files after the first 8
    are only counted."""
    shown = [f"{name} {'?' if n is None else format(n, ',')}"
             for name, n in list(cells.items())[:_CELLS_SHOWN]]
    more = len(cells) - len(shown)
    return (f"Cells per file: {', '.join(shown)}"
            + (f" and {more} more" if more else "") + ".")


def _per_method(df: "pd.DataFrame", private: "pd.DataFrame", *, data_path=None,
                verbose: bool = True, nested: bool = True) -> "pd.DataFrame":
    """The frame of ``scan(per_method=True)`` from :func:`_scan`'s two frames; with
    ``verbose`` it prints the per-method lines (``nested``: leave out nested rows)."""
    dataset, ds_dir = private.attrs["dataset"], private.attrs["folder"]
    rows, reasons, n_files = [], [], 0
    held = set(df.attrs["held"])
    for method, group in df.groupby("method", sort=False):
        n_files += bool(group["files_ok"].any())
        if group["runnable"].any():
            rows.append((method, True, ""))
            continue
        # the row that speaks for the method: files on disk, families held, fewest missing
        i = min(group.index, key=lambda i: (
            not _inputs_on_disk(group.loc[i], dataset, data_path),
            len(_row_families(group.at[i, "modalities"]) - held),
            _missing_inputs(group.loc[i], dataset, data_path), i))
        rows.append((method, False, group.at[i, "reason"]))
        reasons.append((method, private.at[i, "_other"]))
    out = pd.DataFrame(rows, columns=["method", "runnable", "reason"])
    order = out.sort_values(["runnable", "method"], ascending=[False, True]).index
    out = out.loc[order].reset_index(drop=True)
    cells = _cells_per_file(ds_dir)
    out.attrs = {**df.attrs, "cells": cells}
    if not verbose:
        return out
    n, k = len(out), int(out["runnable"].sum())
    holds = _holds_sentence(dataset, ds_dir, df.attrs["held"], private.attrs["category"])
    line = (f"[scan] {n_files} of {n} {'method' if n == 1 else 'methods'} "
            f"{_have_their(n_files, n)} input files in {dataset}. {k} of {n} can run on "
            f"this computer. {holds}")
    # the methods with no listed row, and the families they lack
    listed, gone = set(df["method"]), {}
    for m, lack in private.attrs["unlisted"]:
        if m not in listed:
            gone.setdefault(m, set()).update(lack)
    if gone:
        one = len(gone) == 1
        lack = _lacking([f for f in registry._BASE_MODALITIES if any(f in v for v in gone.values())])
        line += (f" {len(gone)} {'method that needs' if one else 'methods that need'} "
                 f"{lack} {'is' if one else 'are'} not listed. {_all_spelling()} lists "
                 f"{'it' if one else 'them'}.")
    installed = _installed_envs()
    if _runner.linux_only_sentence():
        line += f" {LINUX_ONLY_CHECK}"
    elif any(e and e not in installed for e in df["env"]):
        doctor = config.hint("mtb.env.doctor()", "multibench env doctor")
        line += f" {doctor} checks the environments."
    print(line, flush=True)
    if cells:
        print(f"[scan] {_cells_line(cells)}", flush=True)
    by_name = dict(reasons)
    for method in out["method"]:
        if by_name.get(method):
            print(f"[scan] {method}: {by_name[method]}", flush=True)
    would = [i for i, r in df.iterrows() if _would_run(r)]
    if nested:
        would = [i for i in would if i not in set(_nested_rows(df.loc[would])[0])]
    for note in dict.fromkeys(private.loc[would, "_folder_caveat"]):
        if note:
            print(f"[scan] {note}", flush=True)
    return out


# ------------------------------------------------------------------- label order
#: "no deadline armed"; not None, which signal.signal returns for a handler
#: that was not installed from Python
_NOT_ARMED = object()


def _arm_deadline(seconds):
    """Start a SIGALRM deadline over a whole per-method step (run and scoring);
    return the previous handler, or ``_NOT_ARMED``."""
    if not seconds:
        return _NOT_ARMED
    import math
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        # signal.signal raises off the main thread
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


def _label_files(dataset, data_path=None) -> list:
    """The label files of a dataset folder, sorted by name: every
    ``*cty*.csv`` except the tool-specific ``*scjoint*`` reformats."""
    base = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    return [p for p in sorted((base / dataset).glob("*cty*.csv"))
            if "scjoint" not in p.name.lower()]


def _label_inputs(inputs: dict, variant, label_files: dict) -> tuple:
    """The names of the dataset's label files among a scan row's inputs (also the
    constant file names of a folder-fed variant), whatever ``needs_labels`` says."""
    paths = [Path(p) for role, p in inputs.items() if role != "data_dir"]
    if "data_dir" in inputs:
        paths += [Path(inputs["data_dir"]) / str(a.const) for a in variant.args
                  if a.const and str(a.const).endswith(".csv")]
    names = (label_files.get(p.resolve()) for p in paths)
    return tuple(dict.fromkeys(n for n in names if n))


def _label_candidates(dataset, n, data_path=None):
    """``[(file names, labels, batch per cell)]``: every label ordering of ``n``
    cells; length alone cannot pick one. The batch is each cell's label file."""
    by_file = {}
    for p in _label_files(dataset, data_path):
        try:
            by_file[p.name] = _eio.read_labels(p)
        except Exception:
            pass
    cands = [([k], v) for k, v in by_file.items()]
    rna = [k for k in by_file if k.startswith("rna_cty")]
    atac = [k for k in by_file if k.startswith(("atac_cty", "peak_cty"))]
    for a, b in itertools.product(rna, atac):
        cands += [([a, b], np.concatenate([by_file[a], by_file[b]])),
                  ([b, a], np.concatenate([by_file[b], by_file[a]]))]
    # diagonal, several batches: the RNA files in number order then the ATAC
    # files, or the reverse; each file counts as one batch
    per_batch = [sorted((k for k in by_file if re.fullmatch(rf"{side}_cty\d+\.csv", k)),
                        key=lambda k: _resolve._label_sort_key(Path(k).stem))
                 for side in ("rna", "atac")]
    if all(per_batch) and sum(map(len, per_batch)) > 2:
        for order in (per_batch[0] + per_batch[1], per_batch[1] + per_batch[0]):
            cands.append((order, np.concatenate([by_file[k] for k in order])))
    nums = sorted(k for k in by_file if k.startswith("cty") and any(c.isdigit() for c in k))
    if len(nums) > 1:
        # permute only the subsets of n cells, not every subset (factorial in files)
        sizes = {k: len(by_file[k]) for k in nums}
        for r in range(len(nums), 1, -1):
            for combo in itertools.combinations(nums, r):
                if sum(sizes[k] for k in combo) != n:
                    continue
                for perm in itertools.permutations(combo):
                    cands.append((list(perm),
                                  np.concatenate([by_file[k] for k in perm])))
    out, seen = [], set()
    for names, lab in cands:
        if len(lab) != n or tuple(names) in seen:
            continue
        seen.add(tuple(names))
        batch = np.concatenate([np.full(len(by_file[k]), i + 1) for i, k in enumerate(names)])
        out.append((list(names), lab, batch))
    return out


#: Below this ARI the winning ordering is at chance: no confidence is reported.
_CHANCE_ARI = 0.05


def _order_confidence(cands) -> float | None:
    """``(best - max(runner_up, 0)) / best`` over the ARI of the ranked label orders;
    a ratio, as the runner-up sits near ARI 0 and a difference is capped by the winner."""
    if len(cands) < 2:
        return None
    best, second = cands[0]["ARI"], cands[1]["ARI"]
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
    """Whether ``metrics=`` is ``None`` or names a metric that needs the
    batches: one of the batch or the classification family."""
    from .plot.bar import BATCH_METRICS, CLASSIFICATION_METRICS
    codes = _metric_codes(metrics)
    return codes is None or any(c in BATCH_METRICS + CLASSIFICATION_METRICS
                                for c in codes)


def _without_classification(metrics, multi_batch: bool):
    """``metrics=`` without the classification family, for the UMAP a graph method
    writes; ``None`` when nothing else is left."""
    from .plot.bar import BATCH_METRICS, CLASSIFICATION_METRICS, CLUSTERING_METRICS
    if metrics is None or metrics == "all":
        if multi_batch:
            return CLUSTERING_METRICS + [c for c in BATCH_METRICS if c != "kBET"]
        return "clustering" if metrics is None else metrics
    codes = _metric_codes(metrics)
    if codes is None:
        return metrics                     # evaluate rejects it with its message
    left = [c for c in codes if c not in CLASSIFICATION_METRICS]
    if len(left) == len(codes):
        return metrics
    return left or None


def _evaluate_best_order(emb, category, cands, *, metrics=None, user_batch=False,
                         batch_given=True, on_rank=None, embedding=True):
    """Score each candidate label order and keep the best: ``(names, scores, spread)``.
    ``user_batch``: the candidates carry the caller's batch; ``batch_given``: passed now."""
    # a file-of-origin batch, or the saved one rescore reuses, goes to evaluate only
    # for a metric that uses it: evaluate would warn about a batch never given
    use_batch = (user_batch and batch_given) or _names_batch_metric(metrics)

    def _full(lab, batch, clustering=None):
        several = len(set(np.asarray(batch).tolist())) > 1
        multi = use_batch and several
        chosen = metrics if embedding else _without_classification(metrics, multi)
        if not embedding:
            # the batches go only with a selection that still uses them
            multi = several and ((user_batch and batch_given)
                                 or _names_batch_metric(chosen))
        return _evaluate(emb, category=category, labels=lab,
                         verbose=False, batch=(batch if multi else None),
                         clustering=clustering, metrics=chosen)

    if len(cands) == 1:
        names, lab, batch = cands[0]
        try:
            val = _full(lab, batch)
        except Exception as e:  # noqa: BLE001 - surfaced on the record
            # an empty result would make a bad metrics=/batch= look like 'no label file matched'
            return names, None, [{"order": names,
                                  "error": f"{type(e).__name__}: {str(e)[:300]}"}]
        # metrics= may leave ARI out; + 0.0: a tiny negative ARI rounds to 0.0, not -0.0
        ari = round(float(val["Value"]["ARI"]), 4) + 0.0 if "ARI" in val.index else None
        return names, val, [{"order": names, "ARI": ari}]

    # The Leiden sweep behind ARI depends on the embedding alone: sweep once, rank
    # every order on it, and compute the full metric set only for the winner.
    import scib.metrics

    if on_rank is not None:
        on_rank(len(cands), emb.shape[0])
    try:
        sweep_adata, sweep_keys = _escib.leiden_sweep(emb)
    except Exception:
        sweep_adata = None

    scored, last_error = [], ""
    for names, lab, batch in cands:
        try:
            if sweep_adata is None:      # a self-contained screen per order
                val = _evaluate(emb, category=category, verbose=False,
                                labels=lab, metrics=["ARI"])
                scored.append((float(val["Value"]["ARI"]), names, lab, batch, None))
                continue
            sweep_adata.obs["celltype"] = pd.Categorical(
                np.asarray(lab).astype(str))
            best_key, best_nmi = None, -1.0
            for k in sweep_keys:
                nmi = float(scib.metrics.nmi(sweep_adata, cluster_key=k,
                                             label_key="celltype"))
                if nmi > best_nmi:
                    best_nmi, best_key = nmi, k
            ari = float(scib.metrics.ari(sweep_adata, cluster_key=best_key,
                                         label_key="celltype"))
            scored.append((ari, names, lab, batch,
                           np.asarray(sweep_adata.obs[best_key].values)))
        except Exception as e:  # noqa: BLE001 - keep screening other orders
            last_error = f"{type(e).__name__}: {e}"
    if not scored:
        raise RuntimeError("no label ordering could be screened"
                           + (f"; last error: {last_error}" if last_error else ""))
    scored.sort(key=lambda r: -r[0])
    ari, names, lab, batch, clusters = scored[0]
    try:
        val = _full(lab, batch, clustering=clusters)    # reuses the winning clustering
        if clusters is not None and val.attrs.get("clustering") == "user":
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
def _with_label_order_note(summary: "pd.DataFrame") -> "pd.DataFrame":
    """Add ``label_order_note``: why ``label_order_confidence`` is blank. A column of
    its own, since a sentinel string in the numeric one would break ``> 0.5``."""
    scored = summary["status"].astype(str).str.startswith("CHAIN_OK")
    blank = scored & summary["label_order_confidence"].isna()
    ari = pd.to_numeric(summary.get("ARI"), errors="coerce")
    why = pd.Series([None] * len(summary), index=summary.index, dtype=object)
    why[blank & (ari < _CHANCE_ARI)] = "winner at chance"
    why[blank & ~(ari < _CHANCE_ARI)] = "single ordering"
    why[~scored] = "not scored"
    summary["label_order_note"] = why
    return summary


#: provenance of a reused output when ``out_dir`` holds no earlier record of it
_UNKNOWN_PROVENANCE = {"scripts_commit": None, "env_flavor": "unknown", "hostname": ""}


def _earlier_provenance(folder: Path) -> dict:
    """``{method: {scripts_commit, env_flavor, hostname}}`` of the runs saved in
    ``folder``; records without a method are skipped, an unreadable file gives ``{}``."""
    path = Path(folder) / "batch_result.json"
    try:
        with open(path) as fh:
            records = json.load(fh).get("records") or []
    except (OSError, ValueError, AttributeError):
        return {}
    # a saved file may hold any records BatchResult was given
    return {r["method"]: {k: r.get(k, v) for k, v in _UNKNOWN_PROVENANCE.items()}
            for r in records if isinstance(r, dict) and r.get("method")
            and r.get("status") != "SKIPPED"}


def _check_save_target(folder: Path, dataset: str, category: str) -> None:
    """Raise ``ValueError`` when ``folder`` holds a saved result of another dataset
    or category: merging the records would mix two sweeps."""
    path = Path(folder) / "batch_result.json"
    if not path.exists():
        return
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except (OSError, ValueError) as e:
        raise ValueError(f"{path} exists but cannot be read ({e}); save to another "
                         f"folder or remove it") from e
    have = (blob.get("dataset"), blob.get("category"))
    if have != (dataset, category):
        raise ValueError(
            f"{folder} already holds a saved result for dataset={have[0]!r} "
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
        The per-method records (the same list ``results`` returns).
    dataset : str
        Dataset folder name.
    category : str
        Integration category.
    out_dir : path | None
        The sweep's output root, or ``None`` for an in-memory result.

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
    ``len(res)`` is the number of method records.

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
        self._batches: dict = {}       # {batch file name: CSV text}, until saved

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
        - ``RUN_OK_NO_EMBEDDING`` - ran; the method emits only a graph.
        - ``RUN_OK_EVAL_FAILED`` - ran, but scoring the embedding failed.
        - ``RUN_OK_NO_LABEL_MATCH`` - ran, but no label file matches the
          embedding's cell count.
        - ``RUN_OK`` - ran with ``evaluate=False``.
        - ``TIMEOUT`` - exceeded ``run_all(timeout=...)``.
        - ``FAIL`` - the method itself errored.
        - ``SKIPPED`` - blocked before the run, with the reason in the
          ``reason`` column.

        **Graph methods.** scMoMaT also writes a UMAP, which is scored:
        ``CHAIN_OK_GRAPH_METHOD``. Seurat_WNN writes only a neighbour graph:
        ``RUN_OK_NO_EMBEDDING``, and its ``emb_shape`` is ``None``. Neither
        has classification metrics, which need the method's own embedding.

        **Batch columns.** The batch vector the batch metrics used, and its
        number of distinct values (1 = none). ``batch_source`` is
        ``'file_of_origin'`` (each cell's label file: ``cty1.csv`` -> 1 ...),
        ``'user'`` (``run_all(batch=)`` / ``rescore(batch=)``) or ``None`` (a
        single label file: clustering metrics only).

        **Label-order confidence.** ``label_order`` names the label file(s),
        in order, the metrics used (``rna_cty.csv+atac_cty.csv``).
        ``label_order_confidence`` is ``(best - max(runner_up, 0)) / best``
        over the ARI of the candidate label orders, from 0 to 1. Near 1, one
        order clearly fits. Below about 0.5, two orders scored alike; check
        that row's label order.

        **Optimistic bias.** When more than one ordering is possible, the
        reported metrics are those of the ordering with the highest ARI. So
        they are slightly optimistic. ``label_order_confidence`` shows how far
        ahead the chosen order was.

        **Blank confidence.** The column is numeric, and a blank is ``NaN``.
        So ``> 0.5`` is ``False`` for it and ``.isna()`` finds it. It is blank
        in three cases, named by ``label_order_note``:

        - ``"single ordering"`` - only one ordering was possible (a vertical
          dataset with one ``cty.csv``).
        - ``"winner at chance"`` - the winning ordering scored ARI < 0.05.
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
                        # + 0.0: an older saved record may hold -0.0
                        | {m: (v + 0.0 if isinstance(v, float) else v)
                           for m, v in (r.get("metrics") or {}).items()}
                        # NaN, not None, for a record saved before the field
                        | {"caveat": np.nan if r.get("caveat") is None
                           else r.get("caveat"),
                           "reason": (r.get("error") or "")
                           if r.get("status") == "SKIPPED" else ""})
        if not rows:
            return pd.DataFrame(columns=["method", "status", "run_sec", "output_kind",
                                         "emb_shape", "n_tunable", "label_order",
                                         "label_order_confidence", "batch_source",
                                         "n_batches", "label_order_note", "caveat",
                                         "reason"]).astype({"label_order_confidence":
                                                            "float64"})
        table = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
        # float64 with NaN also when every row is blank (else object, None)
        table["label_order_confidence"] = pd.to_numeric(
            table["label_order_confidence"], errors="coerce").astype("float64")
        table = _with_label_order_note(table)
        for col in ("caveat", "reason"):
            table[col] = table.pop(col)
        # whole numbers stay whole next to the blanks of SKIPPED and FAIL rows
        for col in ("n_batches", "n_tunable"):
            try:
                table[col] = pd.to_numeric(table[col]).astype("Int64")
            except (TypeError, ValueError):     # not whole numbers: left as they are
                pass
        return table

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
        frames = []
        for r in self.records:
            if r.get("_long") is not None:
                frames.append(r["_long"])
            elif r.get("metrics"):
                # through to_long, for the same seven columns as an attached frame
                wide = pd.DataFrame({"Value": list(r["metrics"].values())},
                                    index=list(r["metrics"]))
                with warnings.catch_warnings():
                    # a metrics dict has no scoring record: no wide-CSV warning
                    warnings.simplefilter("ignore", UserWarning)
                    frames.append(_to_long(
                        wide, method=r.get("method"),
                        dataset=r.get("dataset", self.dataset),
                        category=r.get("category", self.category)
                    ).drop(columns="scored_with"))
        if not frames:
            return pd.DataFrame(columns=_LONG_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    @property
    def results(self) -> list:
        """The raw per-method records: status, out_dir, metrics and the orderings tried.

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
        scripts_commit, env_flavor, hostname        the scripts, env build and
                                    computer of the run that made the output
        _long                       internal; read BatchResult.long instead
        ```

        ``label_order_candidates`` holds every ordering tried and its ARI.
        ``summary``'s ``label_order_confidence`` is computed from them.

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
            barcode-indexed Series or CSV. Every cell needs a batch. ``None`` =
            the batch ``run_all`` was given, else each cell's label file.
        labels : array-like | Series | path | None
            One cell-type label per cell; a Series or a barcode-indexed CSV is
            aligned by barcode. ``None`` = search the dataset's label files
            again.
        metrics : str | list[str] | None
            Metric family (``"clustering"``, ``"batch"``, ``"classification"``,
            ``"all"``) or metric codes; ``None`` = every metric the batch
            structure allows, as in ``mtb.evaluate``.
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
            ``labels`` or ``batch`` holds ids that are not cells of the dataset,
            or a cell has no batch.
        FileNotFoundError
            ``labels`` or ``batch`` is the name of an obs column, such as
            ``"donor"``. Pass the column itself: ``batch=adata.obs["donor"]``.

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
        **Order.** A ``labels`` array, list or plain CSV follows the embedding
        rows. Without ``labels=``, a ``batch`` array follows the order of
        ``mtb.labels_for(dataset)``. With a ``labels`` array, the ``batch``
        array follows the embedding rows too. A CSV path is read like a label
        file. A Series or one-column DataFrame with a non-default index, or a
        CSV whose first column holds barcodes, is aligned to the dataset's
        cells as in ``run_all(batch=)``, with the same errors and warnings.
        Aligned labels go into the label-order search; labels matched by
        position read ``(user labels)``. The batch is recorded as
        ``batch_source='user'``.

        **Label order.** With ``labels=None`` and several label files,
        ``rescore`` ranks the file orders by ARI, which needs one Leiden sweep.
        It keeps the stored order and skips the sweep when ``metrics=`` has no
        ARI, NMI or iF1.

        **Labels without batch.** Given ``labels`` and no ``batch``, the batch
        that ``run_all(batch=)`` saved is reused; without one, every cell is
        in one batch, so only clustering metrics are computed. Pass ``batch``
        as well for the batch metrics.

        **Record status.** A graph-only method is marked
        ``RUN_OK_NO_EMBEDDING``. ``SKIPPED``, ``FAIL`` and ``TIMEOUT`` records
        are kept as they are. A record whose output file is gone, or whose new
        scoring fails (a ``batch`` of the wrong length), becomes
        ``RUN_OK_EVAL_FAILED`` with the reason in ``error``. So does a record
        whose labels, from the folder or ``labels=``, have a cell without one
        (NaN, blank text, or text such as ``NA``): a missing label is not a
        cell type.

        **Other hosts.** When the dataset folder has moved, pass the folder
        that now holds it as ``mtb.load_batch(data_path=)``. When the dataset
        folder is not found, ``labels=None`` gives ``RUN_OK_NO_LABEL_MATCH``
        with a ``note``; a Series or CSV is matched by position, with a
        warning, and the saved batch is not used.

        See Also
        --------
        mtb.evaluate : the scoring function applied per record.

        BatchResult.save : persist the re-scored result.
        """
        import copy
        new = BatchResult([], self.dataset, self.category, out_dir=self.out_dir)
        found = [_dataset_root(r, self.dataset) for r in self.records]
        roots = [root for root, _ in found]
        tried: dict = {}           # the roots searched, named when the folder is not found
        for root, r in zip(roots, self.records):
            tried.setdefault(root, _root_tries(r))
        # labels and batch per data folder, before any record is scored: ids
        # that are not cells of the dataset raise here
        labels_at: dict = {}
        batch_at: dict = {}
        used = dict.fromkeys(root for root, r in zip(roots, self.records)
                             if r.get("status") != "SKIPPED")
        for root in used:
            if labels is not None:
                # as written: a record names the word for a missing label
                labels_at[root] = _cell_vector(labels, self.dataset, root, what="labels",
                                               order=_ROWS, stacklevel=4, moved=True,
                                               tried=tried[root], as_written=True)
            # a batch vector follows labels given in embedding row order
            rows = labels is not None and not labels_at[root][1]
            if batch is not None:
                vec, by_cell = _cell_vector(batch, self.dataset, root, what="batch",
                                            order=_ROWS if rows else None,
                                            stacklevel=4, moved=True, tried=tried[root],
                                            as_written=True)
                _check_batch(vec, self.dataset)
                # saved with the result, unless it follows the embedding rows
                file = None if rows and not by_cell else _batch_csv(vec, self.dataset, root)
                batch_at[root] = (vec, by_cell, file)
        if batch is None and _names_batch_metric(metrics):
            _warn_unsaved_batch([r for r in self.records if _scorable(r)], self)
        for r, (root, ok) in zip(self.records, found):
            rec = copy.deepcopy({k: v for k, v in r.items() if k != "_long"})
            rec["_long"] = None
            method = rec.get("method")
            if not _scorable(rec):
                new.records.append(rec)
                continue
            rec["data_path"] = root
            if ok and root is not None:
                rec["data_root"] = root
            try:
                mods = rec.get("modalities") or []
                variant = registry.get(method).select(self.category, set(mods))
                emb = _load_embedding(Path(rec["out_dir"]), variant)
                if emb is None:
                    rec["status"] = "RUN_OK_NO_EMBEDDING"
                    rec["note"] = (f"output kind={variant.output.kind}; this method does not "
                                   "produce an embedding, so embedding-based metrics do not apply")
                else:
                    label_ids, labels_by_cell = labels_at.get(root, (None, False))
                    if batch is not None:
                        batch_ids, batch_by_cell, file = batch_at[root]
                        rec["batch_file"] = file and file[0]
                    else:
                        # the batch run_all was given, when it is saved
                        text = self._batch_text(rec.get("batch_file"))
                        file = text and (rec["batch_file"], text)
                        batch_ids, batch_by_cell = ((_read_batch_text(text), True) if text
                                                    else (None, False))
                    if file:
                        new._batches[file[0]] = file[1]
                    # metrics without clustering: the stored label order is kept
                    keep = (rec.get("labels_used")
                            if labels is None and not _needs_sweep(metrics) else None)
                    _score_record(rec, emb, self.dataset, self.category, root, variant,
                                  batch=batch_ids, labels=label_ids, metrics=metrics,
                                  labels_by_cell=labels_by_cell, batch_by_cell=batch_by_cell,
                                  keep_order=keep, batch_given=batch is not None,
                                  on_rank=_rank_line(method) if verbose else None,
                                  tried=tried[root])
            except Exception as e:  # noqa: BLE001 - one bad record must not abort the rest
                _drop_scores(rec)
                rec["status"] = "RUN_OK_EVAL_FAILED"
                rec["error"] = _clip_error(f"{type(e).__name__}: {e}")
            if verbose:
                print(f"[rescore] {method} -> {rec['status']} {_ari_tail(rec)}".rstrip(),
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

        **Saving into a folder that has a result.** When the folder already
        holds ``batch_result.json`` for the same dataset and category, the
        records are merged. A method in this result replaces its earlier
        record, unless it is ``SKIPPED`` and the earlier one is not; the other
        earlier records are kept, every file in the list above is rewritten
        from the merged set, and a line ``# Merged with 1 earlier record in <folder>
        (StabMap).`` names them. This result object is not changed. Jobs
        running at the same time should use one ``out_dir`` each; combine
        them with ``multibench plot --input dir1 --input dir2``.

        See Also
        --------
        mtb.load_batch : reads the folder back.
        """
        folder = Path(out_dir or self.out_dir or ".")
        _check_save_target(folder, self.dataset, self.category)
        before = (load_batch(folder).records if (folder / "batch_result.json").exists()
                  else [])
        # a SKIPPED record never replaces an earlier record of a run
        ran = {r.get("method") for r in before if r.get("status") != "SKIPPED"}
        new = [r for r in self.records
               if not (r.get("status") == "SKIPPED" and r.get("method") in ran)]
        mine = {r.get("method") for r in new}
        kept = [r for r in before if r.get("method") not in mine]
        merged = BatchResult(kept + new, self.dataset, self.category,
                             out_dir=folder) if kept else self
        folder.mkdir(parents=True, exist_ok=True)
        merged.summary.to_csv(folder / "summary.csv", index=False)
        long_table = merged.long
        if not long_table.empty:
            long_table.to_csv(folder / "long.csv", index=False)
        elif (folder / "long.csv").exists():
            # an earlier long.csv would attach stale metrics on load_batch
            (folder / "long.csv").unlink()
        merged.failures.to_csv(folder / "failures.csv", index=False)
        for name in dict.fromkeys(r["batch_file"] for r in merged.records
                                  if r.get("batch_file")):
            text = None if (folder / name).exists() else self._batch_text(name)
            if text is not None:
                (folder / name).write_text(text)
        slim = [{k: v for k, v in r.items() if k != "_long"} for r in merged.records]
        with open(folder / "batch_result.json", "w") as fh:
            json.dump({"dataset": self.dataset, "category": self.category,
                       "records": slim}, fh, indent=1, default=str)
        if kept:
            names = ", ".join(dict.fromkeys(str(r.get("method")) for r in kept))
            records = "record" if len(kept) == 1 else "records"
            print(f"# Merged with {len(kept)} earlier {records} in {folder} ({names}).",
                  flush=True)
        return folder

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
#: closing lines that name no cause (R's after any error)
_ERROR_TRAILERS = ("Execution halted",)
#: R prints the call trace and then any warnings after the error message
_R_CALLS = re.compile(r"^Calls: ")
_R_WARNINGS = re.compile(r"^In addition: Warning messages?:")


def _clip_error(text: str) -> str:
    """``text`` cut to 600 characters from the left: it ends with the exception."""
    return text if len(text) <= 600 else "... " + text[-596:]


def _ari_tail(rec) -> str:
    """The end of a ``run_all`` / ``rescore`` result line: ``ARI 0.629`` (never
    ``-0.000``); ``""`` without an ARI."""
    ari = (rec.get("metrics") or {}).get("ARI")
    try:
        ari = float(ari)
    except (TypeError, ValueError):
        return ""
    if np.isnan(ari):
        return ""
    return f"ARI {round(ari, 3) + 0.0:.3f}"


def _error_tail(error, width: int = 200) -> str:
    """The last line of ``error`` that names a cause, clipped from the left to
    ``width``; R's call trace and warnings are skipped, ``Error in f() :`` joined."""
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
               assume_gpu: bool = False, every: bool = False) -> str:
    """The ``mtb.scan`` call (or ``multibench scan`` command) that shows the rows of
    a ``run_all`` call, with every non-default argument as the caller gave it."""
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
    elif every:
        py.append('modalities="all"')
        cli += ["--modalities", "all"]
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
                              assume_gpu: bool = False, others=None,
                              every: bool = False, note: str = "") -> str:
    """The ``ValueError`` text when no requested row can start (run_all's Notes);
    ``others``: each row's blocks besides the platform."""
    def _line(r):
        return f"  {_row_label(r)}: {r['reason']}"
    # rows whose files are in place first: an env install fixes those
    blocked = blocked.assign(_files=~blocked["files_ok"].astype(bool)).sort_values(
        ["_files", "method"], kind="stable").drop(columns="_files")
    platform = _platform_line(blocked)
    doctor = config.hint("mtb.env.doctor()", "multibench env doctor")
    where = _scan_hint(dataset, category, data_path=data_path, methods=methods,
                       modalities=modalities, allow_atac_mismatch=allow_atac_mismatch,
                       assume_gpu=assume_gpu, every=every)
    lacking = ""
    if platform and others is not None:
        # as for SKIPPED records: the rest are only counted
        shown, n_lacking = _skipped_rows(blocked, set(), methods, dataset, data_path)
        if shown and n_lacking:
            blocked = blocked.loc[[r.name for r, _ in shown]]
            lacking = (f"\n{n_lacking} {'row needs' if n_lacking == 1 else 'rows need'} "
                       f"files that {dataset} does not have.")
    if methods:
        head = (f"None of the requested methods ({', '.join(methods)}) can run on "
                f"{dataset} ({category})")
        listing = (_other_blocks(blocked, others, requested=True) + lacking
                   if platform and others is not None else
                   f"Blocked, one line per requested {_rows_word(blocked, 1)}:\n"
                   + "\n".join(_line(r) for _, r in blocked.iterrows()))
        return (f"{head}.\n{platform}{listing}"
                f"\n{where} shows these rows. Its files_ok and env_ok columns say "
                f"which check failed. {doctor} checks the environments.")
    head = f"No method can run on {dataset} ({category})"
    if platform and others is not None:
        listing = _other_blocks(blocked, others, requested=False) + lacking
    else:
        n, k = len(blocked), min(3, len(blocked))
        rows = _rows_word(blocked, n)
        shown = (f"The first {k} of {n} blocked {rows}" if n > k
                 else f"The blocked {rows}" if n == 1 else f"The {n} blocked {rows}")
        listing = f"{shown}:\n" + "\n".join(_line(r) for _, r in blocked.head(k).iterrows())
    return (f"{head}.\n{platform}{listing}" + (f"\n{note}" if note else "") +
            f"\n{where} shows {'the listed rows' if note else 'every row'}. Its "
            f"files_ok and env_ok columns say "
            f"which check failed. {doctor} checks the environments.")


def _other_blocks(blocked: pd.DataFrame, others: pd.Series, *, requested: bool) -> str:
    """The list under the off-Linux line of the "No method can run" error: the rows
    something else also blocks, with that reason (all when ``requested``, else 3)."""
    rest = others.reindex(blocked.index).fillna("")
    listed = [(r, rest[i]) for i, r in blocked.iterrows() if rest[i]]
    n, k = len(blocked), len(listed)
    noun = ("requested " if requested else "") + _rows_word(blocked, n)
    if not listed:
        return (f"Nothing else blocks this {noun}." if n == 1
                else f"Nothing else blocks these {n} {noun}.")
    count = f"{k} of {n} {noun} {'is' if k == 1 else 'are'} also blocked by something else"
    shown = listed if requested else listed[:3]
    lines = "\n".join(f"  {_row_label(r)}: {text}" for r, text in shown)
    return (f"{count}:" if len(shown) == k
            else f"{count}. The first {len(shown)}:") + "\n" + lines


def _row_label(r) -> str:
    """``Matilda (rna+adt)``, or the method alone for a variant that reads a folder."""
    mods = r["modalities"]
    return r["method"] if mods == "(data_dir)" else f"{r['method']} ({mods})"


def _platform_line(blocked: pd.DataFrame) -> str:
    """The off-Linux line (with its newline) of the "No method can run" error when
    an env check blocks a row, else ``""``."""
    linux_only = _runner.linux_only_sentence()
    if not linux_only or blocked["env_ok"].all():
        return ""
    return (f"{linux_only} On this computer you can check files, score embeddings and "
            f"plot. Run the methods on a Linux machine.\n")


def _unknown_params_error(method: str, unknown: list, allowed: set) -> KeyError:
    return KeyError(f"{method} does not accept {unknown}; it accepts {sorted(allowed)}. "
                    "An empty set means it hardcodes its hyperparameters upstream.")


def _check_param_keys(plan_df: pd.DataFrame, params: dict) -> None:
    """Raise ``KeyError`` when ``params`` names a key no scanned variant of that
    method accepts, so a dry run catches the typo; a method without rows is skipped."""
    for method, overrides in params.items():
        rows = plan_df[plan_df["method"] == method]
        if rows.empty or not overrides:
            continue
        allowed: set = set()
        for _, r in rows.iterrows():
            variant = registry.get(method).select(r["category"],
                                                  set(_row_modalities(r["modalities"])))
            allowed |= set(variant.tunable) | set(variant.params)
        unknown = [k for k in overrides if k not in allowed]
        if unknown:
            raise _unknown_params_error(method, unknown, allowed)


def _load_embedding(mdir: Path, variant):
    """The embedding a finished run left in ``mdir`` as cells x dims, or ``None``: the
    output, or a graph method's extra embedding output (scMoMaT's UMAP)."""
    import h5py

    def read(path, key=None):
        with h5py.File(path) as h:
            return np.array(h[key or ("data" if "data" in h else list(h.keys())[0])])

    emb = None
    if variant.output.kind == "embedding":
        p = mdir / variant.output.file
        if not p.exists():
            raise FileNotFoundError(f"no output {p} to score - was the run deleted?")
        emb = read(p, variant.output.dataset)
    else:
        extra = [mdir / o.file for o in variant.extra_outputs if o.kind == "embedding"]
        found = next((p for p in extra if p.exists()), None)
        if found is not None:
            emb = read(found)
    if emb is not None and emb.ndim == 2 and emb.shape[0] < emb.shape[1]:
        emb = emb.T
    return emb


def _label_file_sizes(dataset, category=None, method=None, *, modalities=None,
                      data_path=None) -> list:
    """``[(file name, cells)]`` of ``labels_for(...)``; ``[]`` when unreadable."""
    try:
        files = _resolve.labels_for(dataset, category, method, modalities=modalities,
                                    data_path=data_path, check=False)
        return [(Path(p).name, len(_eio.read_labels(p))) for p in files.values()]
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
        stems = ["rna", "adt", "atac", "atac_peak", "atac_gas"]
    elif base in ("rna_cty", "adt_cty"):
        stems = [base[:-4]]
    elif base in ("atac_cty", "peak_cty"):
        stems = ["atac_peak", "atac_gas", "atac", "peak"]
    else:
        stems = []
    return [label.parent / f"{s}{digits}.h5" for s in stems]


def _dataset_folder(dataset, data_path) -> Path:
    """``<data_path>/<dataset>``, with ``None`` = config's data root."""
    base = config.DEFAULT.data_path if data_path is None else data_path
    return Path(base) / dataset


def _root_tries(rec: dict, override=None) -> list:
    """The data roots a saved record's dataset is looked up under: ``override``,
    the recorded ``data_path`` (``None`` = config's), then ``data_root``."""
    tries = [override] if override is not None else []
    tries.append(rec.get("data_path"))
    if rec.get("data_root"):
        tries.append(rec["data_root"])
    return tries


def _dataset_root(rec: dict, dataset: str, override=None) -> tuple:
    """``(root, found)``: the first root of :func:`_root_tries` that holds the
    dataset, absolute (``None`` = config's); else the recorded ``data_path``, ``False``."""
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


def _folder_missing(dataset: str, roots) -> str:
    """The clause that the dataset folder is under none of ``roots``, each absolute."""
    where = dict.fromkeys(os.path.abspath(config.DEFAULT.data_path if r is None else r)
                          for r in roots)
    return f"the dataset folder {dataset} is not found in {_and_list(where, 'or')}"


def _unplaced_batch_note(dataset: str, data_path, tried, n_ids: int) -> str:
    """The ``note`` of a record whose saved batch fits no label order."""
    if not _dataset_folder(dataset, data_path).is_dir():
        why = _folder_missing(dataset, tried or [data_path])
        return f"The saved batch is not used, because {why}. {_MOVED_FIX}"
    total = sum(n for _, n in _label_file_sizes(dataset, data_path=data_path))
    return (f"The saved batch is not used, because it has {n_ids:,} ids and the label "
            f"files of {dataset} have {total:,} cells.")


def _batch_csv(vec, dataset, data_path) -> tuple[str, str]:
    """``(batch_<sha1>.csv, CSV text)`` of a batch vector, with the dataset's cell
    ids when it has as many."""
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
    """The dataset's cell ids in ``labels_for(dataset)`` order (each label file's
    data-file barcodes); ``None`` when some are missing or ids repeat."""
    ids: list = []
    try:
        with warnings.catch_warnings():       # the default order is meant here
            warnings.simplefilter("ignore", UserWarning)
            files = _resolve.labels_for(dataset, data_path=data_path, check=False)
        for p in map(Path, files.values()):
            n = len(_eio.read_labels(p))
            bars = next((b for b in map(_resolve._barcodes_of, _label_data_files(p))
                         if b is not None and len(b) == n), None)
            if bars is None:
                return None
            ids.extend(bars)
    except Exception:  # noqa: BLE001 - no readable files: no ids
        return None
    return ids if ids and len(set(ids)) == len(ids) else None


#: the order an argument of run_all / rescore follows when it is matched by position
_ROWS = "the embedding rows"


def _cell_vector(x, dataset, data_path, *, what: str = "batch", order: str | None = None,
                 stacklevel: int = 5, moved: bool = False,
                 tried=None, as_written: bool = False) -> tuple[np.ndarray, bool]:
    """``(x per cell, aligned by id)``: an id-carrying Series, DataFrame or CSV in
    ``labels_for`` order. ``stacklevel`` is the CSV warning's; the Series one is 1 less."""
    from .eval.pipeline import _carries_ids, _pick
    if not config._CLI and _eio._names_no_file(x):
        # the name of a column: the command line says so before it reads --batch
        raise FileNotFoundError(_column_name_error(what, x))
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
        vals, first = _eio.read_labels_ids(x, what=what, pick=_pick(what),
                                           as_written=as_written)
        if first is None:
            return np.asarray(vals), False
        return _eio.by_id_column(
            vals, first, _dataset_cell_ids(dataset, data_path), what=what,
            name=Path(x).name, target=dataset, order=order, no_ids=no_ids,
            stacklevel=stacklevel)
    vals = _eio.as_vector(x, what=what)
    if not _carries_ids(x):
        if what == "batch" and isinstance(x, pd.Series) \
                and _eio._numbers_rows(x, _dataset_cell_ids(dataset, data_path)):
            warnings.warn(_eio._by_position(what), UserWarning,
                          stacklevel=stacklevel - 1)
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
        raise ValueError(f"{what}: the index repeats {_eio._n_ids(len(dup))} (first: "
                         f"{list(dup[:3])}), so it cannot be aligned to the cells of "
                         f"{dataset}. Give each cell one id, or {by_position}.")
    foreign = index.difference(pd.Index(ids), sort=False)
    if len(foreign):
        if pd.api.types.is_integer_dtype(x.index):
            raise ValueError(f"{what}: the index holds row numbers (first: "
                             f"{list(x.index[:3])}), not cell barcodes. Set the "
                             f"index to the dataset's barcodes, or {by_position}.")
        raise ValueError(f"{what}: {_eio._n_ids(len(foreign))} "
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


def _column_name_error(what: str, x: str) -> str:
    """The error for ``batch=`` / ``labels=`` given the name of an obs column."""
    col = x[4:] if x.startswith("obs:") else x
    return config.hint(
        f"{what}={x!r} is not a file. Pass the column itself, as "
        f"{what}=adata.obs[{col!r}].",
        f"--{what} {x} is not a file. Pass a CSV file, as "
        f'adata.obs[[{col!r}]].to_csv("{what}.csv") writes in Python.')


def _has_gap(values) -> bool:
    """Whether some per-cell value is missing (checked on the distinct values)."""
    return bool(_eio.missing_mask(pd.unique(np.asarray(values, dtype=object))).any())


def _check_batch(vec, dataset: str) -> None:
    """``ValueError`` when the batch vector has a cell without a batch."""
    if not _has_gap(vec):
        return
    miss = _eio.missing_mask(vec)
    raise ValueError(
        config.hint("batch", "--batch") + f": {int(miss.sum()):,} of the "
        f"{len(miss):,} cells of {dataset} have no batch{_eio._missing_example(vec)}. "
        f"Give every cell a batch.")


def _batch_vector(batch, dataset, data_path) -> np.ndarray:
    """``batch`` as one id per cell (:func:`_cell_vector`), checked for gaps."""
    vec = _cell_vector(batch, dataset, data_path, what="batch", stacklevel=5,
                       as_written=True)[0]
    _check_batch(vec, dataset)
    return vec


def _label_gaps(dataset, data_path) -> dict:
    """``{file name: (missing, total, labels as written)}`` of the label files
    with a cell without a label; an unreadable file is left to the scoring."""
    gaps = {}
    for p in _label_files(dataset, data_path):
        try:
            labels = _eio.read_labels(p)
        except Exception:  # noqa: BLE001 - not a readable label file
            continue
        if not _has_gap(labels):
            continue
        miss = _eio.missing_mask(labels)
        gaps[p.name] = (int(miss.sum()), len(miss),
                        np.asarray(_eio.read_labels(p, as_written=True), dtype=object))
    return gaps


def _label_readers(rows: "pd.DataFrame", private: "pd.DataFrame", gaps: dict) -> dict:
    """``{method: the label files with a gap it reads as inputs}`` for ``rows``."""
    readers: dict = {}
    for i, m in rows["method"].items():
        read = [n for n in private.at[i, "_label_inputs"] if n in gaps]
        if read:
            readers.setdefault(m, set()).update(read)
    return readers


def _missing_labels_problem(dataset: str, gaps: dict, readers: dict, *,
                            evaluate: bool) -> str:
    """Why ``run_all`` does not start on label files with a missing label, or "".
    Without ``evaluate`` only the methods that read such a file as an input stop it."""
    if not gaps or not (evaluate or readers):
        return ""
    if not evaluate:
        read = set().union(*readers.values())
        gaps = {name: g for name, g in gaps.items() if name in read}
    (name, (n, total, shown)), more = next(iter(gaps.items())), len(gaps) - 1
    head = (f"{name} of {dataset} has {n:,} of {total:,} cells without a label"
            f"{_eio._missing_example(shown)}")
    if more:
        head += f", and {more} more label {'file has' if more == 1 else 'files have'} some"
    if not evaluate:
        who, others = next(iter(readers)), len(readers) - 1
        head += (f", and {who} reads the label files" if not others else
                 f", and {who} and {others} more {'method' if others == 1 else 'methods'} "
                 f"read the label files")
    return f"{head}. Write the folder again with a label for every cell."


def _unlabelled_error(cands, dataset, data_path, *, user: bool) -> str | None:
    """The ``error`` of a record whose candidate labels have a gap, or ``None``;
    ``user``: the labels came from the caller."""
    for names, lab, _ in cands:
        if not _has_gap(lab):
            continue
        miss, shown, files = _eio.missing_mask(lab), lab, "the labels you passed"
        if not user:
            folder = _dataset_folder(dataset, data_path)
            files = _and_list(names)
            shown = np.concatenate([np.asarray(_eio.read_labels(folder / k, as_written=True),
                                               dtype=object) for k in names])
        return (f"ValueError: {int(miss.sum()):,} of the {len(miss):,} labels in {files} "
                f"are missing{_eio._missing_example(shown)}. Give every cell a label, then "
                f"score again.")
    return None


def _inputs_on_disk(row, dataset, data_path) -> bool:
    """Whether every input file of a scan row is in the folder (its other file
    checks may still fail)."""
    if row["files_ok"]:
        return True
    try:
        paths = _row_paths(row, dataset, data_path)
        if "data_dir" in paths:
            variant = registry.get(row["method"]).select(
                row["category"], set(_row_modalities(row["modalities"])))
            if not _resolve._check_data_dir(variant, paths["data_dir"])[0]:
                return False
    except Exception:  # noqa: BLE001 - unresolvable: not on disk
        return False
    return all(Path(p).exists() for p in paths.values())


def _skipped_rows(blocked, attempted, methods, dataset, data_path) -> tuple[list, int]:
    """``([(row, files on disk)], others)``: the blocked rows a real run reports
    (files on disk, or a named method not attempted) and how many it only counts."""
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
    """One ``SKIPPED`` record per method not attempted, from its first reported row
    (one with its files on disk first)."""
    named = set(methods or ())
    first: dict = {}
    for r, on_disk in sorted(shown, key=lambda t: not t[1]):
        if r["method"] not in attempted:
            first.setdefault(r["method"], r)
    return [{"method": m, "category": category, "dataset": dataset,
             "modalities": _row_modalities(r["modalities"]),
             "output_kind": r["output_kind"], "env": r["env"],
             "n_tunable": int(r["n_tunable"]), "status": "SKIPPED",
             "error": r["reason"], "requested": m in named, "_long": None,
             "caveat": _run_caveat(r["caveat"]),
             "data_path": str(data_path) if data_path else None,
             "data_root": data_root,
             "multibench_version": version,
             "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            for m, r in first.items()]


def _nested_rows(plan: "pd.DataFrame") -> tuple[list, list[str]]:
    """``(indices, notes)``: the rows a sweep without ``modalities=`` leaves out
    because another row of the method holds their modalities, and why."""
    drop, notes = [], []
    for method, rows in plan.groupby("method", sort=False):
        sets = {i: set(str(mods).split("+")) for i, mods in rows["modalities"].items()}
        inside = [i for i, a in sets.items() if any(a < b for b in sets.values())]
        if not inside:
            continue
        drop += inside
        kept = [rows.at[i, "modalities"] for i in sets if i not in inside]
        left = [rows.at[i, "modalities"] for i in inside]
        notes.append(f"{method} runs on {_and_list(kept)}, not on "
                     f"{_and_list(left)}, whose files are part of it. Pass "
                     + config.hint("modalities=", "--modalities")
                     + " to run a smaller combination.")
    return drop, notes


def _batch_length_problem(plan, batch, dataset, category, data_path) -> str:
    """A dry run's note when ``batch`` fits neither the dataset's label files nor
    some row the sweep would run; "" when it fits."""
    total = sum(n for _, n in _label_file_sizes(dataset, data_path=data_path))
    if not total or len(batch) == total:
        return ""
    bad = []
    for _, r in plan.iterrows():
        if not _would_run(r) or r["method"] in bad:
            continue
        mods = _row_modalities(r["modalities"]) or None
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


#: what scoring writes into a record
_SCORE_KEYS = ("metrics", "labels_used", "label_order_candidates", "batch_source",
               "n_batches", "_long", "note", "error")


def _drop_scores(rec: dict) -> dict:
    """Remove the fields of an earlier scoring."""
    for k in _SCORE_KEYS:
        rec.pop(k, None)
    return rec


def _score_record(rec, emb, dataset, category, data_path, variant, *,
                  batch=None, labels=None, metrics=None, labels_by_cell=False,
                  batch_by_cell=False, keep_order=None, on_rank=None,
                  batch_given=True, tried=None):
    """Fill ``rec`` with the status and scores of ``emb``. ``labels`` / ``batch`` in
    ``labels_for`` order (``*_by_cell``) go into each label order, else follow the rows."""
    stored = rec.get("label_order_candidates")
    stored_used = rec.get("labels_used")
    _drop_scores(rec)
    rec["emb_shape"] = list(emb.shape)
    embedding = variant.output.kind == "embedding"
    if not embedding and _without_classification(metrics, True) is None:
        from .eval.classify import _graph_text
        raise ValueError(_graph_text(rec.get("method")) + " "
                         + config.hint("Pass metrics='clustering'.",
                                       "Pass --metrics clustering."))
    n = emb.shape[0]
    rows = labels is not None and not labels_by_cell      # labels in embedding row order
    segments = None
    if batch is not None:
        batch = np.asarray(batch)
        # in the dataset's cell order: each label order gets the ids of its files
        if batch_by_cell or not rows:
            segments = _batch_segments(dataset, data_path, batch)
        if batch_by_cell and segments is None:
            # no label files to place the ids by: they would land in the wrong rows
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
        by_file = _batch_segments(dataset, data_path, labels)
        cands = [(names, np.concatenate([by_file[k] for k in names]), np.ones(n, dtype=int))
                 for names, _, _ in cands
                 if by_file is not None and all(k in by_file for k in names)]
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
        # the user's ids replace each candidate's file-of-origin batch, in that
        # candidate's order. With labels in embedding rows: the stored label order
        # when its files hold the embedding's cells, else the variant's own order;
        # a vector as long as the embedding is kept as given
        own = None
        if rows and segments is not None:
            stored_order = list(stored_used or ())
            fits = (stored_order and all(k in segments for k in stored_order)
                    and sum(len(segments[k]) for k in stored_order) == n)
            own = (stored_order if fits else
                   [k for k, _ in _label_file_sizes(dataset, category, rec.get("method"),
                                                    modalities=rec.get("modalities") or None,
                                                    data_path=data_path)])
        placed = []
        for names, lab, _ in cands:
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
    # not inside the ranking: it would take the error of each order and rank on
    unlabelled = _unlabelled_error(cands, dataset, data_path, user=labels is not None)
    if unlabelled:
        rec["status"], rec["error"] = "RUN_OK_EVAL_FAILED", unlabelled
        return rec
    names, val, spread = _evaluate_best_order(emb, category, cands, metrics=metrics,
                                              user_batch=batch is not None,
                                              batch_given=batch_given,
                                              on_rank=on_rank, embedding=embedding)
    if val is None:
        rec["status"] = "RUN_OK_EVAL_FAILED"
        errs = [s["error"] for s in spread if s.get("error")]
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
    # the batch vector the batch metrics saw
    winning_batch = next(b for order, _, b in cands if order == names)
    n_batches = int(len(set(np.asarray(winning_batch).tolist())))
    if batch is not None:
        rec["batch_source"] = "user"
    else:
        rec["batch_source"] = "file_of_origin" if n_batches > 1 else None
    rec["n_batches"] = n_batches
    rec["_long"] = _to_long(val, method=rec["method"], dataset=dataset, category=category)
    rec["status"] = "CHAIN_OK" if embedding else "CHAIN_OK_GRAPH_METHOD"
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
    modalities : list[str] | str | None
        Modality tokens of one combination, e.g. ``["rna", "adt"]``; ``None`` =
        every combination. ``"all"`` runs the same methods; its dry run and
        messages also list the combinations whose modalities the folder
        does not hold, as ``mtb.scan(modalities="all")`` does.
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
        or a barcode-indexed CSV is aligned by barcode. Every cell needs a
        batch. ``None`` = each cell's label file.
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
    FileNotFoundError
        ``batch`` is the name of an obs column, such as ``"donor"``.
    ValueError
        Unknown ``category``; a named method or a selection with no variant,
        before anything runs, dry run included; nothing runnable (Notes); an
        ``out_dir`` that holds a saved result of another dataset or category;
        ``skip_existing=True`` with ``params``, or ``assume_gpu=True``, in a
        real run.
    ValueError
        A ``batch`` Series or CSV holds ids that are not cells of the dataset.
    ValueError
        A cell has no batch in ``batch``.
    ValueError
        A real run on a folder whose label files have a cell without a label
        (Notes).
    KeyError
        Unknown id in ``methods`` or ``params``; on a dry run, a rejected ``params`` key.
    TypeError
        A real run without ``out_dir``; ``methods`` given as a bare string, or
        ``modalities`` as a string other than ``"all"``.

    Warns
    -----
    UserWarning
        ``dataset`` matches a folder only up to letter case.
    UserWarning
        ``modalities`` leaves out scBridge, which reads a folder, without excluding its ATAC form.
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
    **Dry run.** ``dry_run=True`` runs nothing and returns the ``mtb.scan``
    frame for the same selection: blocked rows are kept with their
    ``reason``, and ``command`` is rendered for ``out_dir`` (or the literal
    ``'<out_dir>'`` placeholder). ``plan[plan.runnable]`` lists what will
    run. Like ``mtb.scan``, it lists the combinations whose modalities the
    folder holds, and every combination with ``modalities="all"``; the
    methods a real run tries are the same either way. ``multibench run-all
    --dry-run --format csv`` writes the same frame.

    **What runs.** Only the rows ``mtb.scan`` marks runnable: input files,
    conda env, and a GPU where the script needs one. Without
    ``modalities``, a method whose runnable rows are nested runs once, on
    the row with the most modalities: on a folder with ``rna.h5``,
    ``adt.h5`` and ``atac.h5``, scMoMaT runs ``rna+adt+atac``. Pass
    ``modalities=`` to run a smaller combination; it selects as in
    ``mtb.scan``. A blocked row whose input files are in the folder, or a
    named method with no runnable row, is logged and recorded as
    ``SKIPPED`` with its reason; other blocked rows are only counted. Each
    method reads one ATAC representation (``mtb.describe_layout``), and
    ``run_all`` skips a method given the other representation, also when
    ``methods=`` names it. With ``allow_atac_mismatch=True`` the method runs
    without an error and gives a wrong embedding.

    **Failures.** A method that raises is recorded as ``FAIL`` (with its
    ``error``), one that exceeds ``timeout`` as ``TIMEOUT``, and the sweep
    moves on; a ``params`` key the variant does not accept is a ``FAIL``
    too. Check ``res.failures``. Without a timeout, one method that hangs
    stops the sweep: size it from the ``runtime_tier`` /
    ``observed_worst_sec`` columns of ``mtb.scan``; the slowest methods take
    more than 4 h. The timeout covers the run and its scoring; off the main
    thread it is unavailable, with a warning.

    Nothing runnable: ``ValueError``. Its first line is ``No method can run
    on D11 (vertical).`` With ``methods=``, it starts ``None of the
    requested methods (Matilda, totalVI) can run on D11 (vertical).`` and
    lists every requested variant. Without ``methods``, it gives the first 3
    of N. It never lists the reasons of methods you did not ask for. On
    macOS or Windows, its second line says that methods run only on Linux,
    and it lists only the rows something else also blocks. Its last line
    names the ``mtb.scan`` call that shows the rows. When combinations are
    not listed because the folder does not hold their modalities, the line
    before says so.

    **Saved files.** The result is saved under ``out_dir`` as
    ``BatchResult.save`` writes it; reload it with ``mtb.load_batch``. With
    ``batch=``, the vector is saved as ``batch_<hash>.csv``. A later run
    into the same ``out_dir`` is merged with the records already there:
    methods it re-ran are replaced, the others kept. Jobs running at the
    same time should use one ``out_dir`` each.

    **Resuming.** ``skip_existing=True`` reuses each method's existing
    output. Reuse only checks that the output file exists, not that it is
    complete: after a hard kill, delete that method's sub-directory before
    resuming. A reused output's record copies ``scripts_commit``,
    ``env_flavor`` and ``hostname`` from the method's earlier record in
    ``out_dir``; without one they are ``None``, ``'unknown'`` and ``''``.
    Reuse is keyed on the output file, not on ``params``, so the two
    together raise ``ValueError``: give each setting a fresh ``out_dir``, as
    ``mtb.sweep`` does.

    **Missing labels.** A missing label is not a cell type. A real run
    stops with ``ValueError`` before its first method when a label file of
    the folder has a cell without a label: NaN, blank text, or text that a
    label file reads as missing, such as ``NA`` or ``None``. Write the
    folder again with a label for every cell. ``evaluate=False`` runs the
    methods without scores, unless one of them reads the label file as an
    input (Matilda does). A dry run prints the same fact and raises nothing.
    ``mtb.evaluate(missing_labels="drop")`` scores one embedding on its
    labelled cells only.

    **Batch vector.** By default the batch metrics use the label file each
    cell came from (``cty1.csv`` -> 1 ...); ``batch=`` replaces that rule and
    is recorded as ``batch_source='user'``. With two or more batches outside
    ``vertical``, the classification metrics are scored too. ``run_all``
    puts the vector in each method's cell order, as it does the labels, and
    keeps only the cells of the batches a method reads. A vector as long as
    one method's output is used as given; any other length marks that
    method ``RUN_OK_EVAL_FAILED``, which the dry run says first.

    A Series or one-column DataFrame indexed by cell id is aligned to the
    barcodes of the dataset's files. So is a CSV whose first column holds
    them, as ``obs[["sample"]].to_csv(path)`` writes it. Without usable
    barcodes, or with a first column of text that holds none of them, the
    match is by position, with a ``UserWarning``; R's row numbers there give
    no warning.

    ``"obs:<column>"`` and ``"<column>"`` name a column of the object you
    pass. ``export_dataset`` and ``evaluate`` take both. ``run_all`` and
    ``rescore`` are given a folder, not the object, so they take the column
    itself, for example ``batch=adata.obs["donor"]``. A cell without a batch
    (NaN, blank text, or text such as ``NA``) raises ``ValueError``, in a dry
    run too.

    See Also
    --------
    mtb.scan : the check table this function runs from.

    mtb.BatchResult : what is returned - ``summary``, ``long``, ``failures``, ``plot``, ``rescore``.

    mtb.sweep : one method over a range of one hyperparameter.

    mtb.load_batch : reload a saved sweep.

    mtb.run : one method, one variant, with explicit inputs.
    """
    # the arguments first, so a bad combination is not reported as an I/O error
    registry.check_category(category)
    registry.check_id_list(methods, "methods")
    every, modalities = _every_combination(modalities)
    registry.check_id_list(modalities, "modalities")
    params = params or {}
    for method in params:
        registry.check_method(method)
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
    # the on-disk spelling, once, so out_dir names and records agree (one warning)
    dataset = _resolve.canonical_dataset(
        Path(data_path) if data_path is not None else config.DEFAULT.data_path, dataset)
    if not dry_run and out_dir is None:
        raise TypeError("run_all() needs out_dir= for a real run (dry_run=True "
                        "returns the scan frame without one)")
    # a real run checks params per method and records a bad key as FAIL
    plan_df, private = _scan(dataset, category, data_path=data_path, methods=methods,
                             modalities="all" if every else modalities, verbose=False,
                             assume_gpu=assume_gpu,
                             allow_atac_mismatch=allow_atac_mismatch,
                             params=params if dry_run else None,
                             out_dir=OUT_DIR_PLACEHOLDER if out_dir is None else out_dir)
    not_listed = (_not_listed_note(plan_df, dataset, private.attrs["folder"], category)
                  if plan_df.attrs["not_listed"] else "")
    if plan_df.empty:
        # a modalities= selection that matches nothing, without methods=
        raise _no_variant_error(category, methods, modalities)
    if dry_run:
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
            if not_listed:
                print(f"[run_all] {not_listed}", flush=True)
            wrong_ref = _scripts_ref_note()
            if wrong_ref:
                print(f"[run_all] {wrong_ref}", flush=True)
            # in full: the compact table views clip them
            scripts, lines = _dry_run_notes(plan_df, modalities)
            if scripts:
                print(f"[run_all] {scripts}", flush=True)
            for _method, caveat in lines:
                print(f"[run_all] {caveat}", flush=True)
            if modalities is None:
                for note in _nested_rows(plan_df[plan_df["runnable"]])[1]:
                    print(f"[run_all] {note}", flush=True)
            if batch_vec is not None:
                bad = _batch_length_problem(plan_df, batch_vec, dataset, category,
                                            data_path)
                if bad:
                    print(f"[run_all] {bad}", flush=True)
            gaps = _label_gaps(dataset, data_path)
            if gaps:
                would = plan_df[[_would_run(r) for _, r in plan_df.iterrows()]]
                if modalities is None:
                    would = would.drop(index=_nested_rows(would)[0])
                unlabelled = _missing_labels_problem(
                    dataset, gaps, _label_readers(would, private, gaps),
                    evaluate=evaluate)
                if unlabelled:
                    print(f"[run_all] {unlabelled}", flush=True)
        return plan_df
    blocked = plan_df[~plan_df["runnable"]]
    plan_df = plan_df[plan_df["runnable"]]
    if plan_df.empty:
        # not one method could start: the request is wrong, and an empty result
        # would read "0 failed", as success
        raise ValueError(_nothing_runnable_message(
            dataset, category, blocked, methods, data_path=data_path,
            modalities=modalities, allow_atac_mismatch=allow_atac_mismatch,
            assume_gpu=assume_gpu, others=private["_other"], every=every,
            note="" if methods else not_listed))
    if modalities is None:
        nested, nested_notes = _nested_rows(plan_df)
        plan_df = plan_df.drop(index=nested)
        if verbose:
            for note in nested_notes:
                print(f"[run_all] {note}", flush=True)
    # the scores would count a missing label as a cell type, and a method that
    # reads the file would train on it: stop before out_dir is made
    gaps = _label_gaps(dataset, data_path)
    unlabelled = _missing_labels_problem(
        dataset, gaps, _label_readers(plan_df, private, gaps), evaluate=evaluate)
    if unlabelled:
        raise ValueError(unlabelled)

    batch_vec = None if batch is None else _batch_vector(batch, dataset, data_path)
    # saved in out_dir so that rescore can reuse it
    batch_file = (_batch_csv(batch_vec, dataset, data_path)
                  if batch_vec is not None and evaluate else None)
    # absolute: a saved result read from another directory still finds the data
    data_root = os.path.abspath(config.DEFAULT.data_path if data_path is None
                                else data_path)
    out_dir = Path(out_dir)
    # the save after the sweep would refuse this folder: refuse it now
    _check_save_target(out_dir, dataset, category)
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
    earlier = _earlier_provenance(out_dir) if skip_existing else {}
    from . import __version__ as _pkg_version

    for _, row in plan_df.iterrows():
        method = row["method"]
        mods = _row_modalities(row["modalities"])
        method_dir = out_dir / f"{method}_{dataset}"
        rec = {"method": method, "category": category, "dataset": dataset,
               "modalities": mods, "output_kind": row["output_kind"],
               "env": row["env"], "n_tunable": row["n_tunable"], "status": "?", "_long": None,
               "caveat": _run_caveat(row["caveat"]),
               "params_used": dict(params.get(method) or {}),
               "out_dir": str(method_dir),
               "data_path": str(data_path) if data_path else None,
               "data_root": data_root,
               "multibench_version": _pkg_version,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        start = time.time()
        if verbose:
            print(f"[run_all] {method} ({category}/{dataset}) ...", flush=True)
            if rec["caveat"]:              # it starts with the method name
                print(f"[run_all]   {rec['caveat']}", flush=True)
        deadline = _NOT_ARMED
        try:
            deadline = _arm_deadline(timeout)
            inputs = _resolve.inputs_for(dataset, category, method, modalities=mods or None,
                                         data_path=data_path, check=True)
            variant = registry.get(method).select(category, set(mods))
            reused = skip_existing and (method_dir / variant.output.file).exists()
            # on every path, also a reuse: an unknown key must fail, not be dropped
            overrides = params.get(method)
            if overrides:
                allowed = set(variant.tunable) | set(variant.params)
                unknown = [k for k in overrides if k not in allowed]
                if unknown:
                    raise _unknown_params_error(method, unknown, allowed)
            if reused:
                if verbose:
                    print(f"[run_all]   reusing existing output in {method_dir}", flush=True)
                res = None
            else:
                with warnings.catch_warnings():
                    # an allowed ATAC caveat is already in the log and the record
                    warnings.filterwarnings(
                        "ignore", category=UserWarning,
                        message=re.escape(f"{method} ") + rf"({_WRONG_ATAC_BODY}"
                                rf"|{re.escape(PEAK_NAMES_REASON)})")
                    res = _run(method=method, category=category, inputs=inputs,
                               out_dir=str(method_dir), params=overrides)
            rec["reused"] = bool(reused)
            rec["run_sec"] = round(time.time() - start, 1)
            if variant.output.kind == "embedding" and res is not None:
                emb = np.asarray(res.output)
                if emb.ndim == 2 and emb.shape[0] < emb.shape[1]:
                    emb = emb.T
            else:
                emb = _load_embedding(method_dir, variant)
            if emb is None:
                rec["status"] = "RUN_OK_NO_EMBEDDING"
                rec["note"] = (f"output kind={variant.output.kind}; this method does not "
                               "produce an embedding, so embedding-based clustering metrics "
                               "do not apply")
            else:
                rec["emb_shape"] = list(emb.shape)
                if not evaluate:
                    rec["status"] = "RUN_OK"
                else:
                    if batch_file:
                        rec["batch_file"] = batch_file[0]
                    try:
                        _score_record(rec, emb, dataset, category, data_path, variant,
                                      batch=batch_vec)
                    except TimeoutError:
                        raise
                    except Exception as e:      # the run itself succeeded
                        rec["status"] = "RUN_OK_EVAL_FAILED"
                        rec["error"] = f"{type(e).__name__}: {e}"
        except TimeoutError as e:
            rec["status"] = "TIMEOUT"
            rec["error"] = str(e)
            rec["run_sec"] = round(time.time() - start, 1)
        except Exception as e:
            rec["status"] = "FAIL"
            rec["error"] = _clip_error(f"{type(e).__name__}: {e}")
            rec["traceback"] = traceback.format_exc()[-1200:]
            rec["run_sec"] = round(time.time() - start, 1)
        finally:
            _disarm_deadline(deadline)
        if verbose:
            tail = _error_tail(rec["error"]) if rec.get("error") else _ari_tail(rec)
            print(f"[run_all]   -> {rec['status']} ({rec.get('run_sec')}s) {tail}".rstrip(),
                  flush=True)
        if rec.get("reused"):
            rec.update(earlier.get(method, _UNKNOWN_PROVENANCE))
        else:
            rec.update(config.run_provenance(row["env"]))   # scripts_commit, env_flavor, hostname
        records.append(rec)
    records += _skipped_records(shown, attempted, methods, category=category,
                                dataset=dataset, data_path=data_path,
                                version=_pkg_version, data_root=data_root)

    result = BatchResult(records, dataset, category, out_dir)
    if batch_file and any(r.get("batch_file") for r in records):
        result._batches[batch_file[0]] = batch_file[1]
    result.save()
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
    Each setting's folder is ``<param>_<value>`` with ``.`` -> ``p`` and
    ``-`` -> ``m`` (``lr=0.001`` runs under ``<out_dir>/lr_0p001/``).

    ``df.attrs["long"]`` names each setting as its own series
    (``"Multigrate (lr=0.001)"``), so ``mtb.plot.bubble`` draws one row per
    setting. It is also written to ``<out_dir>/sweep_long.csv`` (path in
    ``df.attrs["long_path"]``) when any setting produced metrics, since
    ``DataFrame.attrs`` does not survive ``to_csv``.

    A setting that fails is recorded by ``run_all``: its row has ``status``
    ``FAIL`` or ``TIMEOUT`` and empty metrics. Check the ``status`` column
    before reading the curve. A method whose ``tunable`` is empty cannot be
    swept; every setting is then a ``FAIL``. The ``param`` check needs one
    variant: with ``modalities=None`` and several variants in ``category``
    (Matilda under ``vertical``), an unknown ``param`` is a ``FAIL`` for
    every setting; pass ``modalities`` to get the ``KeyError``. Errors of
    ``mtb.run_all`` (no method can run) propagate.

    See Also
    --------
    mtb.params_for : the tunable hyperparameters of the variant.

    mtb.run_all : what each setting runs through.
    """
    registry.check_method(method)
    tune = _params_for_method(method, category, modalities)
    if tune and param not in tune:
        raise KeyError(
            f"{method} does not expose {param!r}; it accepts {sorted(tune)}.")
    frames, longs = [], []
    for v in values:
        tag = str(v).replace(".", "p").replace("-", "m")
        res = run_all(dataset, category, out_dir=Path(out_dir) / f"{param}_{tag}",
                      methods=[method], modalities=modalities, data_path=data_path,
                      params={method: {param: v}}, timeout=timeout, verbose=verbose)
        df = res.summary
        df.insert(0, param, v)
        frames.append(df)
        setting = res.long
        if not setting.empty:                  # each setting its own series
            setting = setting.copy()
            setting[param] = v
            setting["method"] = setting["method"].astype(str) + f" ({param}={v})"
            longs.append(setting)
    out = pd.concat(frames, ignore_index=True)
    long_table = pd.concat(longs, ignore_index=True) if longs else pd.DataFrame()
    out.attrs["long"] = long_table
    if not long_table.empty:
        long_path = Path(out_dir) / "sweep_long.csv"
        long_path.parent.mkdir(parents=True, exist_ok=True)
        long_table.to_csv(long_path, index=False)
        out.attrs["long_path"] = str(long_path)
    return out


def _params_for_method(method, category, modalities):
    """The variant's tunable keys, or ``None`` when ``params_for`` raises."""
    try:
        from .discover import params_for
        return params_for(method, category,
                          list(modalities) if modalities else None)["tunable"]
    except Exception:
        return None
