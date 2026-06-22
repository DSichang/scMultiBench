"""scib metric computation (clustering + batch), ported from qc/scib_metrics."""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd


def _build_adata(emb, celltype, cluster, batch):
    adata = ad.AnnData(np.asarray(emb, dtype=float))
    adata.obsm["X_emb"] = adata.X
    adata.obs["celltype"] = pd.Categorical(np.asarray(celltype))
    adata.obs["cluster"] = pd.Categorical(np.asarray(cluster))
    adata.obs["batch"] = pd.Categorical(np.asarray(batch))
    return adata


def compute(emb, celltype, cluster, batch, group: str = "clustering") -> pd.DataFrame:
    """Compute scib metrics. group in {'clustering','batch','all'}.

    Returns a metric.csv-shaped DataFrame (index = metric, column 'Value').
    """
    if group not in {"clustering", "batch", "all"}:
        raise ValueError(f"unknown group {group!r}; valid: clustering|batch|all")

    import scanpy as sc
    import scib.metrics as me

    n = np.asarray(emb).shape[0]
    n_ct = len(np.asarray(celltype))
    n_cl = len(np.asarray(cluster))
    if not (n_ct == n_cl == n):
        raise ValueError(
            f"input length mismatch: emb has {n} cells, celltype has {n_ct}, "
            f"cluster has {n_cl}"
        )
    if batch is not None:
        n_ba = len(np.asarray(batch))
        if n_ba != n:
            raise ValueError(
                f"input length mismatch: emb has {n} cells, batch has {n_ba}"
            )

    adata = _build_adata(emb, celltype, cluster, batch)
    sc.pp.neighbors(adata, use_rep="X_emb")
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
        _safe("iASW", lambda: me.isolated_labels_asw(adata, batch_key="batch", label_key="celltype", embed="X_emb"))
        _safe("iF1", lambda: me.isolated_labels_f1(adata, batch_key="batch", label_key="celltype", embed="X_emb"))
        _safe("cLISI", lambda: me.clisi_graph(adata, label_key="celltype", type_="embed", use_rep="X_emb"))
    if want_bat:
        _safe("ASW_batch", lambda: me.silhouette_batch(adata, batch_key="batch", label_key="celltype", embed="X_emb"))
        _safe("GC", lambda: me.graph_connectivity(adata, label_key="celltype"))
        _safe("iLISI", lambda: me.ilisi_graph(adata, batch_key="batch", type_="embed", use_rep="X_emb"))
        _safe("kBET", lambda: me.kBET(adata, batch_key="batch", label_key="celltype", type_="embed", embed="X_emb"))

    return pd.DataFrame.from_dict(out, orient="index", columns=["Value"])
