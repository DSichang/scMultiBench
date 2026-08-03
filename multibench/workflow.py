"""High-level workflow: point at a dataset, run everything that applies, get metrics and a figure.

The low-level API (``inputs_for`` -> ``run`` -> ``evaluate`` -> ``plot``) requires
you to know a method's name, its integration category AND its exact modality
combination. This module removes that burden:

    mtb.scan("D11")                     # what can I run on this data?
    res = mtb.run_all("D11", "vertical")  # run all of it, with metrics
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
from .eval.pipeline import evaluate as _evaluate, to_long as _to_long

__all__ = ["scan", "run_all", "BatchResult", "list_categories", "describe_layout",
           "load_batch", "runtime_hint"]


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
    "cty":       "cty.csv     - cell-type labels (cty1.csv, cty2.csv, ... per batch)",
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
             "", "ONE batch  -> rna.h5, adt.h5, cty.csv",
             "MANY batches -> rna1.h5/rna2.h5/..., adt1.h5/adt2.h5/..., cty1.csv/cty2.csv/...",
             "              (numbered files in the SAME flat dir; no batch column)",
             "", "Modality roles and the filenames they resolve to:"]
    lines += [f"    {k:16s} {v}" for k, v in ROLES.items()]
    lines += ["",
              "!! Careful: the two ATAC representations do NOT map to the obvious names.",
              "   gene-activity -> atac.h5   and   peaks -> peak.h5 .",
              "   Putting a peak matrix in atac.h5 runs every method on the wrong",
              "   representation without any error.",
              "",
              "Modality files are HDF5 with the matrix under 'matrix/data'.",
              "Use mtb.io.to_canonical(src, dst) to convert another layout.",
              "Labels are CSV with one row per cell; the last column (or a column",
              "named 'x') holds the cell type.", ""]
    if category:
        lines += [f"{category}: {CATEGORIES.get(category, '(unknown category)')}", ""]
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
    it is not, a ``reason``. Also carries ``runtime_tier`` /
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
    """
    rows = []
    for spec, v, cat, mods in _variant_rows(category):
        rt = _runtimes().get(spec.id, {})
        rec = {"method": spec.id, "category": cat,
               "modalities": "+".join(mods) or "(data_dir)",
               "env": envs.group_for(spec.id), "output_kind": v.output.kind,
               "n_tunable": len(v.tunable),
               "runtime_tier": rt.get("tier", "unknown"),
               "observed_worst_sec": rt.get("worst_sec"),
               "runnable": False, "reason": ""}
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
            rec["reason"] = f"{type(e).__name__}: {e}"[:160]
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values(
        ["runnable", "category", "method"], ascending=[False, True, True])
    return df.reset_index(drop=True)


# ------------------------------------------------------------------- label order
def _read_cty(path):
    d = pd.read_csv(path)
    col = "x" if "x" in d.columns else d.columns[-1]
    return d[col].to_numpy()


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
        for r in range(len(nums), 1, -1):
            for perm in itertools.permutations(nums, r):
                cands.append((list(perm), np.concatenate([ctys[k] for k in perm])))
    out, seen = [], set()
    for names, lab in cands:
        if len(lab) != n or tuple(names) in seen:
            continue
        seen.add(tuple(names))
        out.append((list(names), lab))
    return out


def _evaluate_best_order(emb, category, cands):
    """Score each candidate label order, keep the best, return the full spread."""
    scored = []
    for names, lab in cands:
        try:
            val = _evaluate(emb, category=category, task="clustering", labels=lab)
            scored.append((float(val["Value"]["ARI"]), names, val))
        except Exception:
            continue
    if not scored:
        return None, None, []
    scored.sort(key=lambda r: -r[0])
    ari, names, val = scored[0]
    spread = [{"order": n, "ARI": round(a, 4)} for a, n, _ in scored]
    return names, val, spread


# ------------------------------------------------------------------------ results
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
        ``TIMEOUT`` (exceeded ``run_all(timeout=...)``) or ``FAIL``.
        ``TIMEOUT`` and ``FAIL`` both appear in :attr:`failures`.
        """
        rows = []
        for r in self.records:
            rows.append({k: r.get(k) for k in
                         ("method", "status", "run_sec", "output_kind", "emb_shape", "n_tunable")}
                        | {m: v for m, v in (r.get("metrics") or {}).items()})
        return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)

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
        """Raw per-method records (status, out_dir, metrics, label-order spread).

        Keeps a long sweep's outputs addressable so you can re-score or re-plot
        WITHOUT re-running the methods.
        """
        return self.records

    @property
    def failures(self) -> pd.DataFrame:
        """Methods that did not complete: ``method, status, error``.

        ``run_all`` records failures instead of raising, so ALWAYS check this - a
        sweep can finish "successfully" with several methods having failed.
        """
        bad = [r for r in self.records if r.get("status") not in ("CHAIN_OK", "CHAIN_OK_GRAPH_METHOD")]
        return pd.DataFrame([{k: r.get(k) for k in ("method", "status", "error")} for r in bad])

    def plot(self, **kw):
        """Bubble figure of every method that produced metrics.

        Methods are rows (best first), metrics are columns; bubble SIZE encodes the
        method's rank and bubble COLOUR the value, both relative to the methods in
        this figure. Read it next to :attr:`summary` - with few methods a small
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
        kw.setdefault("title", f"{self.category} - {self.dataset}")
        return _plot.bubble(lng, **kw)

    def save(self, out_dir=None) -> "Path":
        """Write this result to disk so it outlives the process.

        Produces ``summary.csv``, ``long.csv``, ``failures.csv`` and
        ``batch_result.json``. Reload with :func:`load_batch` to re-score or
        re-plot the next morning WITHOUT re-running any method.
        """
        d = Path(out_dir or self.out_dir or ".")
        d.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(d / "summary.csv", index=False)
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
        return (f"<BatchResult {self.category}/{self.dataset}: "
                f"{ok}/{len(self.records)} with metrics>")


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
        names, e.g. ``["rna", "adt"]`` for CITE-seq or ``["rna", "atac_gas"]``
        for RNA + ATAC gene-activity. See :func:`describe_layout` for every role
        name and the filename it expects. Default: run all combinations.
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
           Reuse is keyed on the output FILE, not on ``params``. If you change a
           hyperparameter and re-run into the SAME ``out_dir`` with
           ``skip_existing=True`` you will silently get the OLD result. When
           tuning, use a fresh ``out_dir`` per setting (or leave this False).
           ``run_all`` refuses this combination rather than mislead you.

        .. warning::
           Reuse only checks that the output file EXISTS, not that it is complete.
           A method killed mid-write leaves a truncated file that would be reused
           as if it had succeeded. After a hard kill, delete that method's
           sub-directory before resuming.

    Methods can take minutes to hours; a failure is recorded, never raised, so one
    bad method cannot abort the sweep.
    """
    plan = scan(dataset, category=category, data_path=data_path)
    plan = plan[plan["runnable"]]
    if methods is not None:
        plan = plan[plan["method"].isin(methods)]
    if modalities is not None:
        plan = plan[plan["modalities"] == "+".join(modalities)]
    if dry_run:
        return plan.reset_index(drop=True)

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
        rec = {"method": m, "category": category, "dataset": dataset,
               "modalities": mod_list, "output_kind": row["output_kind"],
               "env": row["env"], "n_tunable": row["n_tunable"], "status": "?", "_long": None}
        t0 = time.time()
        if verbose:
            print(f"[run_all] {m} ({category}/{dataset}) ...", flush=True)
        try:
            inp = _resolve.inputs_for(dataset, m, category, modalities=mod_list or None,
                                      data_path=data_path, check=True)
            mdir = out_dir / f"{m}_{dataset}"
            v0 = registry.get(m).select(category, set(mod_list))
            reused = skip_existing and (mdir / v0.output.file).exists()
            if reused:
                if verbose:
                    print(f"[run_all]   reusing existing output in {mdir}", flush=True)
                res = None                       # read back from disk below
            elif timeout:
                import signal

                def _timed_out(signum, frame):
                    raise TimeoutError(f"exceeded timeout of {timeout}s")

                prev = signal.signal(signal.SIGALRM, _timed_out)
                signal.alarm(int(timeout))
                try:
                    res = _run(method=m, category=category, inputs=inp,
                               out_dir=str(mdir), params=params.get(m))
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, prev)
            else:
                res = _run(method=m, category=category, inputs=inp,
                           out_dir=str(mdir), params=params.get(m))
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
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["traceback"] = traceback.format_exc()[-1200:]
            rec["run_sec"] = round(time.time() - t0, 1)
        if verbose:
            print(f"[run_all]   -> {rec['status']} ({rec.get('run_sec')}s) "
                  f"{(rec.get('metrics') or {}).get('ARI', '')}", flush=True)
        records.append(rec)

    result = BatchResult(records, dataset, category, out_dir)
    result.save()          # survive process exit; reload with load_batch()
    return result
