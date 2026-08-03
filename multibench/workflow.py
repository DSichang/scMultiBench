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

import glob
import itertools
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

__all__ = ["scan", "run_all", "BatchResult"]


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
    it is not, a ``reason``. Nothing is executed. This is the first call to make
    when pointing the benchmark at a NEW dataset.
    """
    rows = []
    for spec, v, cat, mods in _variant_rows(category):
        rec = {"method": spec.id, "category": cat,
               "modalities": "+".join(mods) or "(data_dir)",
               "env": envs.group_for(spec.id), "output_kind": v.output.kind,
               "n_tunable": len(v.tunable), "runnable": False, "reason": ""}
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

    def __init__(self, records, dataset, category):
        self.records = records
        self.dataset = dataset
        self.category = category

    @property
    def summary(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            rows.append({k: r.get(k) for k in
                         ("method", "status", "run_sec", "output_kind", "emb_shape", "n_tunable")}
                        | {m: v for m, v in (r.get("metrics") or {}).items()})
        return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)

    @property
    def long(self) -> pd.DataFrame:
        frames = [r["_long"] for r in self.records if r.get("_long") is not None]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @property
    def failures(self) -> pd.DataFrame:
        bad = [r for r in self.records if r.get("status") not in ("CHAIN_OK", "CHAIN_OK_GRAPH_METHOD")]
        return pd.DataFrame([{k: r.get(k) for k in ("method", "status", "error")} for r in bad])

    def plot(self, **kw):
        """Bubble figure of everything that produced metrics."""
        from . import plot as _plot
        lng = self.long
        if lng.empty:
            raise ValueError("no method produced metrics; see .failures / .summary")
        kw.setdefault("title", f"{self.category} - {self.dataset}")
        return _plot.bubble(lng, **kw)

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        ok = sum(1 for r in self.records if str(r.get("status", "")).startswith("CHAIN_OK"))
        return (f"<BatchResult {self.category}/{self.dataset}: "
                f"{ok}/{len(self.records)} with metrics>")


# ---------------------------------------------------------------------- run_all
def run_all(dataset: str, category: str, *, out_dir, modalities=None, methods=None,
            params: dict | None = None, data_path=None, evaluate: bool = True,
            dry_run: bool = False, verbose: bool = True) -> BatchResult | pd.DataFrame:
    """Run every method that applies to ``dataset`` under ``category``.

    Parameters
    ----------
    methods : restrict to these method ids (default: everything runnable).
    modalities : restrict to one modality combination (default: all of them).
    params : per-method hyperparameters, ``{"Cobolt": {"lr": 1e-3}}``. Discover
        what a method accepts with :func:`multibench.params_for`.
    dry_run : return the plan (a DataFrame) without running anything.

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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params or {}
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
            res = _run(method=m, category=category, inputs=inp,
                       out_dir=str(out_dir / f"{m}_{dataset}"), params=params.get(m))
            rec["run_sec"] = round(time.time() - t0, 1)
            spec = registry.get(m)
            v = spec.select(category, set(mod_list))
            emb = None
            if v.output.kind == "embedding":
                emb = np.asarray(res.output)
            else:
                for o in v.extra_outputs:            # a graph method may still ship an embedding
                    if o.kind == "embedding":
                        p = out_dir / f"{m}_{dataset}" / o.file
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
        except Exception as e:
            rec["status"] = "FAIL"
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            rec["traceback"] = traceback.format_exc()[-1200:]
            rec["run_sec"] = round(time.time() - t0, 1)
        if verbose:
            print(f"[run_all]   -> {rec['status']} ({rec.get('run_sec')}s) "
                  f"{(rec.get('metrics') or {}).get('ARI', '')}", flush=True)
        records.append(rec)

    return BatchResult(records, dataset, category)
