"""scib metric computation (clustering + batch), ported from qc/scib_metrics."""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd


def _build_adata(emb, celltype, cluster, batch):
    # anndata is an evaluation-only dependency: importing it lazily keeps
    # `import multibench` working on environments (e.g. Colab) that have
    # no anndata installed and only use discovery + plotting.
    import anndata as ad
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



def leiden_sweep(emb):
    """Run the scIB optimal-resolution Leiden sweep ONCE, reusably.

    ``cluster_optimal_resolution`` clusters the embedding at 10 resolutions and
    keeps whichever maximises NMI against a label vector. The clustering at a
    given resolution depends only on the embedding - the label vector enters
    solely through the argmax - so ranking N candidate label orderings needs one
    sweep, not N. On D52 cross (6 candidate orderings, 23,478 cells) the per-
    candidate sweeps cost ~250s each and dominated the whole evaluation.

    Returns ``(adata, keys)``. The caller assigns ``adata.obs["celltype"]`` and
    scores with scib's OWN ``nmi``/``ari`` against each key, so the selection
    protocol stays identical to ``cluster_optimal_resolution``'s rather than
    being reimplemented.
    """
    import scanpy as sc
    from scib.metrics.clustering import get_resolutions

    # anndata is an evaluation-only dependency: importing it lazily keeps
    # `import multibench` working on environments (e.g. Colab) that have
    # no anndata installed and only use discovery + plotting.
    import anndata as ad
    adata = ad.AnnData(np.asarray(emb, dtype=float))
    adata.obsm["X_emb"] = adata.X
    sc.pp.neighbors(adata, use_rep="X_emb")
    keys = []
    for res in get_resolutions(n=10, max=2):
        key = f"_mb_res_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key)
        keys.append(key)
    return adata, keys


def _isolated_labels_f1(adata, label_key, batch_key, embed, iso_threshold):
    """Isolated-label F1, identical to scib's but without the per-label re-clustering.

    ``scib.metrics.isolated_labels_f1`` calls ``cluster_optimal_resolution`` once
    per isolated label, and every call recomputes the kNN graph and a full Leiden
    resolution sweep. The clustering at a given resolution does not depend on
    which label is being scored - only the F1 target does. Under our convention
    that EVERY label is isolated, scib therefore repeats the same 10 clusterings
    once per label: on a 28-label dataset that is 280 Leiden runs where 10 suffice,
    which is ~90 min on 23k cells.

    Each resolution is clustered once here, then each label takes its max F1 over
    all resolutions - the same quantity scib's per-label optimisation returns.
    ``tests/test_eval_isolated_f1.py`` pins this to scib's own result.
    """
    import scanpy as sc
    from sklearn.metrics import f1_score
    from scib.metrics.clustering import get_resolutions
    from scib.metrics.isolated_labels import get_isolated_labels

    labels = get_isolated_labels(adata, label_key, batch_key, iso_threshold,
                                 verbose=False)
    if len(labels) == 0:
        return float("nan")

    sc.pp.neighbors(adata, use_rep=embed)
    resolutions = get_resolutions(n=10, max=2)

    keys = []
    for res in resolutions:
        key = f"_mb_isof1_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key)
        keys.append(key)

    try:
        scores = []
        for label in labels:
            y_true = (adata.obs[label_key] == label).values
            best = 0.0
            for key in keys:
                col = adata.obs[key]
                for cluster in col.unique():
                    # argument order mirrors scib's max_f1 exactly; F1 is
                    # symmetric under swapping y_true/y_pred, but match it anyway
                    f1 = f1_score((col == cluster).values, y_true)
                    if f1 > best:
                        best = f1
            scores.append(best)
    finally:
        for key in keys:
            if key in adata.obs:
                del adata.obs[key]

    return float(np.mean(scores))

def compute(emb, celltype, cluster, batch, group: str = "clustering",
            slow_metrics: bool = False, only=None) -> pd.DataFrame:
    """Compute scib metrics. group in {'clustering','batch','all'}.

    Returns a metric.csv-shaped DataFrame (index = metric, column 'Value').

    ``only`` restricts the computation to the named metrics, e.g.
    ``only={"ARI"}``. Everything not named is skipped rather than computed and
    discarded, and the optimal-resolution Leiden sweep is skipped too when no
    requested metric needs it. This exists because ranking candidate label
    orderings needs ARI alone, and paying for iF1/cLISI/iLISI once per candidate
    made that ranking cost more than the entire rest of the evaluation.
    """
    if only is not None:
        only = set(only)
    if group not in {"clustering", "batch", "all"}:
        raise ValueError(f"unknown group {group!r}; valid: clustering|batch|all")

    try:
        import scanpy as sc
        import scib.metrics as me
    except ImportError as e:
        raise RuntimeError(
            "metrics need scib and scanpy, which are not installed here - "
            "run: pip install scib scanpy"
        ) from e

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
    _needs_clustering = only is None or bool(only & {"ARI", "NMI"})
    if group in ("clustering", "all") and cluster is None and _needs_clustering:
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
        if only is not None and name not in only:
            return
        try:
            # scib prints per-chunk progress from inside some metrics (the LISI
            # family especially) and emits third-party deprecation warnings;
            # neither carries information for the caller, so both are swallowed
            # here. Our own could-not-compute warning below stays visible.
            with contextlib.redirect_stdout(io.StringIO()), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
        _safe("iF1", lambda: _isolated_labels_f1(adata, label_key="celltype",
                                                 batch_key="batch", embed="X_emb",
                                                 iso_threshold=_iso))
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
