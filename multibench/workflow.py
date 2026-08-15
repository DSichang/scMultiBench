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
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .engine import envs, registry, resolve as _resolve
from .engine.runner import run as _run
from .eval import scib as _escib
from .eval.pipeline import evaluate as _evaluate, to_long as _to_long

__all__ = ["scan", "run_all", "BatchResult", "list_categories", "describe_layout",
           "load_batch", "runtime_hint", "sweep"]


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
    """
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
              "   This matters because methods disagree: Portal/SCALEX/iNMF/sciCAN/",
              "   Conos/VIPCCA/scJoint need GENE ACTIVITY, while MultiVI/moETM/scMM/",
              "   MIRA/scMVP/Seurat_WNN/GLUE need PEAKS. Feeding the wrong one runs",
              "   to completion and returns a plausible but WRONG embedding - no error.",
              "   Check what you actually have before trusting a cross-dataset result.",
              "",
              "MODALITY FILE FORMAT (.h5) - easiest route first:",
              "  mtb.io.to_canonical(src, dst)   converts an .h5ad and writes",
              "  everything below correctly. Prefer it over building the file by hand.",
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
    if category:
        lines += [f"{category}: {CATEGORIES.get(category, '(unknown category)')}", ""]
    lines += ["", "ENVIRONMENTS",
              "  Every method runs in its OWN conda env (they need mutually",
              "  incompatible framework versions). scan() marks a method NOT runnable",
              "  if its env is missing, so a sweep never starts one that cannot finish.",
              "      multibench env doctor          # what is needed / what is missing",
              "      multibench env install --run   # build them all from lockfiles",
              ""]
    lines += ["Then:  mtb.scan('MYDATA')  ->  mtb.run_all('MYDATA', '<category>', out_dir=...)"]
    return "\n".join(lines)



_RUNTIMES_YAML = Path(__file__).resolve().parent / "engine" / "runtimes.yaml"


@functools.lru_cache(maxsize=1)
def _runtimes() -> dict:
    """Observed per-method runtimes (reference data; see engine/runtimes.yaml)."""
    if not _RUNTIMES_YAML.exists():
        return {}
    import yaml
    with open(_RUNTIMES_YAML) as fh:
        return yaml.safe_load(fh) or {}


def runtime_hint(method: str) -> dict:
    """What this method has been OBSERVED to cost, to help size a sweep.

    Returns ``{"tier", "worst_sec", "observed"}`` - ``tier`` is one of ``fast``
    (<5 min), ``medium`` (5-30 min), ``slow`` (30 min-2 h), ``very_slow`` (>2 h),
    and ``observed`` lists the actual (dataset, cells, seconds) measurements.

    .. note::
       These are MEASUREMENTS on one shared machine, not predictions. Your runtime
       depends on hardware, cell count and load. Use them to choose a sensible
       ``run_all(timeout=...)``, not to promise a finish time.
    """
    return dict(_runtimes().get(method, {"tier": "unknown", "worst_sec": None,
                                         "observed": []}))


# --------------------------------------------------------------------------- scan
def _data_dir_usable(variant, ds_dir) -> tuple[bool, str]:
    """Does this dataset really hold what a ``data_dir`` method needs?

    ``data_dir`` resolves to the dataset directory itself when there is no
    ``processed/`` subdir, so the path ALWAYS exists and existence proves nothing.
    Without a content check, spatial-registration methods look runnable on every
    dataset and a sweep would burn hours before failing.
    """
    d = Path(ds_dir)
    if not d.is_dir():
        return False, f"no such directory: {d}"
    if variant.output.kind == "coords":
        slices = sorted(d.glob("*.h5ad"))
        if len(slices) < 2:
            return False, ("spatial registration needs >=2 .h5ad slice files; "
                           f"found {len(slices)} in {d}")
        return True, ""
    # non-spatial data_dir methods (e.g. scBridge) name their files via `const`
    needed = [a.const for a in variant.args if a.const and str(a.const).endswith((".h5", ".csv"))]
    missing = [f for f in needed if not (d / f).exists()]
    if missing:
        return False, f"missing files in {d}: {missing}"
    return True, ""



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


def _caveat(method: str, dataset: str) -> str:
    return _CAVEATS.get((method, dataset), "")


def _variant_rows(category=None):
    for spec in registry.load():
        for v in spec.variants:
            cat = v.when.get("category")
            if category and cat != category:
                continue
            yield spec, v, cat, list(v.when.get("modalities", []))


def scan(dataset: str, category: str | None = None,
         data_path: Path | str | None = None) -> pd.DataFrame:
    """Report every method that can run on ``dataset``, and why the rest cannot.

    Returns one row per (method, category, modalities) with ``runnable`` and, when
    it is not, a ``reason``. ``modalities`` is a ``+``-joined STRING here (e.g.
    ``"rna+adt"``); ``run_all``/``inputs_for`` take it as a LIST
    (``["rna", "adt"]``), so split on ``"+"``. The sentinel ``"(data_dir)"`` marks a
    method that consumes a whole DIRECTORY rather than named modality files (the
    spatial-registration methods, and scBridge) - for those, pass no ``modalities``
    at all. Also carries ``runtime_tier`` /
    ``observed_worst_sec`` (see :func:`runtime_hint`) so you can size a sweep
    BEFORE launching it. Nothing is executed. This is the first call to make
    when pointing the benchmark at a NEW dataset.

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
    if category is not None:
        config.category_folder(category)   # raises with the valid list on a typo
    rows = []
    for spec, v, cat, mods in _variant_rows(category):
        rt = _runtimes().get(spec.id, {})
        cav = _caveat(spec.id, dataset)
        rec = {"method": spec.id, "category": cat,
               "modalities": "+".join(mods) or "(data_dir)",
               "env": envs.group_for(spec.id), "output_kind": v.output.kind,
               "n_tunable": len(v.tunable),
               "runtime_tier": rt.get("tier", "unknown"),
               "observed_worst_sec": rt.get("worst_sec"),
               "caveat": cav, "runnable": False, "reason": ""}
        if rec["env"] and rec["env"] not in _installed_envs():
            rec["reason"] = (f"conda env {rec['env']!r} is not installed - run "
                             "`multibench env install --run` (see mtb.env.doctor())")
            rows.append(rec)
            continue
        try:
            got = _resolve.inputs_for(dataset, spec.id, cat, modalities=mods or None,
                                      data_path=data_path, check=True)
            ok, why = True, ""
            if "data_dir" in got:
                ok, why = _data_dir_usable(v, got["data_dir"])
            rec["runnable"] = ok
            if not ok:
                rec["reason"] = why
        except Exception as e:  # missing files / no variant / bad layout
            msg = f"{type(e).__name__}: {e}"
            # keep the TAIL: the filename is at the END of the message and is the
            # whole point of it. Right-truncation ate exactly what the user needs.
            rec["reason"] = msg if len(msg) <= 500 else msg[:100] + " ... " + msg[-380:]
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values(
        ["runnable", "category", "method"], ascending=[False, True, True])
    return df.reset_index(drop=True)


# ------------------------------------------------------------------- label order
def _read_cty(path):
    d = pd.read_csv(path)
    col = "x" if "x" in d.columns else d.columns[-1]
    return d[col].to_numpy()


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


def _evaluate_best_order(emb, category, cands):
    """Score each candidate label order, keep the best, return the full spread."""
    def _full(lab, bat, clustering=None):
        # several distinct source files => a real batch structure, so ask for BOTH
        # metric groups; otherwise clustering only.
        grp = "all" if len(set(bat)) > 1 else "clustering"
        return _evaluate(emb, category=category, task=grp, labels=lab,
                         batch=(bat if grp == "all" else None),
                         clustering=clustering)

    if len(cands) == 1:
        # nothing to disambiguate - do not pay for a screening pass
        names, lab, bat = cands[0]
        try:
            val = _full(lab, bat)
        except Exception:
            return None, None, []
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
                val = _evaluate(emb, category=category, task="clustering",
                                labels=lab, only={"ARI"})
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
                           "label_order_confidence": _order_confidence(cands)}
                        | {m: v for m, v in (r.get("metrics") or {}).items()})
        if not rows:      # nothing ran (e.g. no method was runnable on this dataset)
            return pd.DataFrame(columns=["method", "status", "run_sec", "output_kind",
                                         "emb_shape", "n_tunable", "label_order",
                                         "label_order_confidence"])
        return _with_label_order_note(
            pd.DataFrame(rows).sort_values("method").reset_index(drop=True))

    @property
    def long(self) -> pd.DataFrame:
        """Tidy frame (``metric, value, method, dataset, category``) for plotting.

        This is what :meth:`plot` and ``mtb.plot.bubble`` consume. Empty if no
        method produced metrics.
        """
        frames = [r["_long"] for r in self.records if r.get("_long") is not None]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

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
        Keyword arguments are passed to ``mtb.plot.bubble`` (``metrics=``,
        ``methods=``, ``order=``, ``title=``, ``cmap=``).

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
def run_all(dataset: str, category: str, *, out_dir, modalities=None, methods=None,
            params: dict | None = None, data_path=None, evaluate: bool = True,
            dry_run: bool = False, verbose: bool = True,
            timeout: float | None = None,
            skip_existing: bool = False) -> "BatchResult | pd.DataFrame":
    """Run every method that applies to ``dataset`` under ``category``.

    Parameters
    ----------
    dataset : the DIRECTORY NAME of your data, e.g. ``"MYCITE"`` - not a full path.
    data_path : the folder that CONTAINS ``dataset``, e.g. ``"/home/wen/data"``
        (so the files live in ``/home/wen/data/MYCITE/``). Defaults to the
        package's configured data root.
    out_dir : where each method's output goes (one sub-directory per method).
    methods : restrict to these method ids, e.g. ``["Matilda", "totalVI"]``
        (default: everything runnable).
    modalities : restrict to ONE modality combination, given as a list of role
        names, e.g. ``["rna", "adt"]`` for CITE-seq, ``["rna", "atac_gas"]`` for
        RNA + ATAC gene-activity, or ``["rna", "atac_peak"]`` for RNA + ATAC peaks.
        See :func:`describe_layout` for every role name. Default: all combinations.

        .. warning::
           The two ATAC representations do NOT map to the obvious filenames:
           gene-activity is ``atac.h5`` but peaks are ``peak.h5``. Putting a peak
           matrix in ``atac.h5`` runs every method on the wrong representation and
           raises NO error - you simply get confident, wrong numbers.
    params : per-method hyperparameters, ``{"Cobolt": {"lr": 1e-3}}``. Discover
        what a method accepts with :func:`multibench.params_for`.
    dry_run : return the plan (a DataFrame) without running anything. Do this
        first - it is free and shows exactly what will be attempted.
    timeout : per-method wall-clock cap in SECONDS. Size it from the
        ``runtime_tier`` / ``observed_worst_sec`` columns of :func:`scan` (or
        :func:`runtime_hint`); the slowest methods observed here need >4 h. A method exceeding it is
        recorded as ``TIMEOUT`` and the sweep moves on. Strongly recommended for
        unattended runs - without it a single hanging method blocks everything.
    skip_existing : if a method's output file is already present in ``out_dir``,
        reuse it instead of recomputing. Lets an interrupted overnight sweep
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

    Only methods that :func:`scan` marks runnable are attempted, which means their
    conda environment was found - a missing env is reported there rather than
    failing hours in (``multibench env doctor`` / ``env install --run``).

    Methods can take minutes to hours; a failure is recorded, never raised, so one
    bad method cannot abort the sweep.
    """
    config.category_folder(category)       # raises with the valid list on a typo
    if methods is not None:
        from .engine import registry as _reg
        _known = set(_reg.list_methods())
        _unknown = [m for m in methods if m not in _known]
        if _unknown:
            raise KeyError(
                f"unknown method(s) {_unknown}; see mtb.list_methods()")
    plan = scan(dataset, category=category, data_path=data_path)
    plan = plan[plan["runnable"]]
    if methods is not None:
        plan = plan[plan["method"].isin(methods)]
    if modalities is not None:
        plan = plan[plan["modalities"] == "+".join(modalities)]
    if dry_run:
        return plan.reset_index(drop=True)
    if plan.empty:
        # A per-method failure is recorded, never raised - but "not one method could
        # even start" is a different class: the REQUEST is wrong (bad dataset name,
        # wrong category, missing files, missing env). Returning an empty result
        # would report "0 failed", which reads as success and hides a typo.
        why = scan(dataset, category=category, data_path=data_path)
        reasons = [r for r in why["reason"].tolist() if r][:3]
        raise ValueError(
            f"nothing is runnable for dataset={dataset!r} category={category!r}. "
            f"Inspect mtb.scan({dataset!r}, {category!r}) for the full table. "
            f"First reasons: {reasons}")

    # validate arguments BEFORE touching the filesystem, so a bad combination is
    # reported as such instead of surfacing as an unrelated I/O error
    params = params or {}
    if skip_existing and params:
        raise ValueError(
            "skip_existing=True with params=... would silently return results computed "
            "with the OLD parameters (reuse is keyed on the output file, not on params). "
            "Use a fresh out_dir per parameter setting, or skip_existing=False.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for _, row in plan.iterrows():
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
            inp = _resolve.inputs_for(dataset, m, category, modalities=mod_list or None,
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
            spec = registry.get(m)
            v = spec.select(category, set(mod_list))
            emb = None
            if v.output.kind == "embedding":
                if res is not None:
                    emb = np.asarray(res.output)
                else:
                    import h5py
                    with h5py.File(mdir / v.output.file) as h:
                        k = v.output.dataset or ("data" if "data" in h else list(h.keys())[0])
                        emb = np.array(h[k])
            else:
                for o in v.extra_outputs:            # a graph method may still ship an embedding
                    if o.kind == "embedding":
                        p = mdir / o.file
                        if p.exists():
                            import h5py
                            with h5py.File(p) as h:
                                k = "data" if "data" in h else list(h.keys())[0]
                                emb = np.array(h[k])
                            break
            if emb is None:
                rec["status"] = "RUN_OK_NO_EMBEDDING"
                rec["note"] = (f"output kind={v.output.kind}; this method does not produce an "
                               "embedding, so embedding-based clustering metrics do not apply")
            else:
                if emb.ndim == 2 and emb.shape[0] < emb.shape[1]:
                    emb = emb.T
                rec["emb_shape"] = list(emb.shape)
                if not evaluate:
                    rec["status"] = "RUN_OK"
                else:
                    cands = _label_candidates(dataset, emb.shape[0], data_path)
                    if not cands:
                        rec["status"] = "RUN_OK_NO_LABEL_MATCH"
                    else:
                        names, val, spread = _evaluate_best_order(emb, category, cands)
                        if val is None:
                            rec["status"] = "RUN_OK_EVAL_FAILED"
                        else:
                            rec["metrics"] = {k: (None if pd.isna(x) else round(float(x), 4))
                                              for k, x in val["Value"].items()}
                            rec["labels_used"] = names
                            if len(spread) > 1:
                                rec["label_order_candidates"] = spread
                            rec["_long"] = _to_long(val, method=m, dataset=dataset,
                                                    category=category)
                            rec["status"] = ("CHAIN_OK" if v.output.kind == "embedding"
                                             else "CHAIN_OK_GRAPH_METHOD")
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
