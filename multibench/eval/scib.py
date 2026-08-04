"""scib metric computation (clustering + batch), ported from qc/scib_metrics."""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd


def _build_adata(emb, celltype, cluster, batch):
    adata = ad.AnnData(np.asarray(emb, dtype=float))
    adata.obsm["X_emb"] = adata.X
    adata.obs["celltype"] = pd.Categorical(np.asarray(celltype).astype(str))
    if cluster is not None:
        adata.obs["cluster"] = pd.Categorical(np.asarray(cluster))
    # kBET converts this to an R factor via rpy2, which refuses non-string
    # categories ("Converting pandas Category series to R factor is only
    # possible when categories are strings"). Integer batch ids are the
    # natural thing for a caller to pass, so coerce here rather than making
    # every caller remember.
    adata.obs["batch"] = pd.Categorical(np.asarray(batch).astype(str))
    return adata


def compute(emb, celltype, cluster, batch, group: str = "clustering",
            slow_metrics: bool = False) -> pd.DataFrame:
    """Compute scib metrics. group in {'clustering','batch','all'}.

    Returns a metric.csv-shaped DataFrame (index = metric, column 'Value').
    """
    if group not in {"clustering", "batch", "all"}:
        raise ValueError(f"unknown group {group!r}; valid: clustering|batch|all")

    import scanpy as sc
    import scib.metrics as me

    n = np.asarray(emb).shape[0]
    n_ct = len(np.asarray(celltype))
    if n_ct != n:
        raise ValueError(
            f"input length mismatch: emb has {n} cells, celltype has {n_ct}"
        )
    if cluster is not None:
        n_cl = len(np.asarray(cluster))
        if n_cl != n:
            raise ValueError(
                f"input length mismatch: emb has {n} cells, cluster has {n_cl}"
            )
    if batch is not None:
        n_ba = len(np.asarray(batch))
        if n_ba != n:
            raise ValueError(
                f"input length mismatch: emb has {n} cells, batch has {n_ba}"
            )

    adata = _build_adata(emb, celltype, cluster, batch)
    sc.pp.neighbors(adata, use_rep="X_emb")

    # When clustering metrics are requested but no precomputed clustering was
    # supplied, derive one from the embedding with scIB optimal-resolution
    # Leiden: sweep resolutions and keep the assignment that maximises NMI
    # vs. the cell-type labels. This is the standard scib clustering protocol
    # and is what lets evaluate() run directly on a method's embedding output.
    if group in ("clustering", "all") and cluster is None:
        me.cluster_optimal_resolution(
            adata, label_key="celltype", cluster_key="cluster", verbose=False
        )
    out: dict[str, float] = {}

    def _safe(name, fn):
        """Compute one metric defensively: record NaN (with a warning) if it fails.

        Some scib metrics (notably the LISI graph metrics) rely on a prebuilt
        binary that may not load on every platform (macOS arm64, older glibc).
        Degrading gracefully lets evaluate() still return every metric that does
        compute, instead of failing the whole evaluation on one optional metric.
        """
        try:
            out[name] = float(fn())
        except Exception as exc:  # noqa: BLE001 - report and continue
            warnings.warn(
                f"scib metric {name!r} could not be computed "
                f"({type(exc).__name__}: {str(exc)[:160]}); recording NaN."
            )
            out[name] = float("nan")

    want_clu = group in ("clustering", "all")
    want_bat = group in ("batch", "all")

    if want_clu:
        _safe("ARI", lambda: me.ari(adata, cluster_key="cluster", label_key="celltype"))
        _safe("NMI", lambda: me.nmi(adata, cluster_key="cluster", label_key="celltype"))
        _safe("ASW", lambda: me.silhouette(adata, label_key="celltype", embed="X_emb"))
        # Isolated-label convention: treat EVERY cell type as isolated and score
        # them all. scib's default picks only types confined to few batches, and
        # returns NOTHING when every type appears in every batch (it short-circuits
        # on iso_threshold == n_batches) - so a well-balanced dataset silently got
        # no iASW/iF1 at all. n_batches + 1 clears that check and admits every label.
        _iso = int(adata.obs["batch"].nunique()) + 1
        _safe("iASW", lambda: me.isolated_labels_asw(adata, batch_key="batch", label_key="celltype",
                                                     embed="X_emb", iso_threshold=_iso))
        _safe("iF1", lambda: me.isolated_labels_f1(adata, batch_key="batch", label_key="celltype",
                                                   embed="X_emb", iso_threshold=_iso))
        _safe("cLISI", lambda: me.clisi_graph(adata, label_key="celltype", type_="embed", use_rep="X_emb"))
    if want_bat:
        _safe("ASW_batch", lambda: me.silhouette_batch(adata, batch_key="batch", label_key="celltype", embed="X_emb"))
        _safe("GC", lambda: me.graph_connectivity(adata, label_key="celltype"))
        _safe("iLISI", lambda: me.ilisi_graph(adata, batch_key="batch", type_="embed", use_rep="X_emb"))
        # kBET shells out to R once per method and dominates the runtime of a
        # sweep (hours per dataset at 10-30k cells), so it is opt-in. Everything
        # it needs IS installed - pass slow_metrics=True to compute it.
        if slow_metrics:
            _safe("kBET", lambda: me.kBET(adata, batch_key="batch", label_key="celltype", type_="embed", embed="X_emb"))

    return pd.DataFrame.from_dict(out, orient="index", columns=["Value"])
