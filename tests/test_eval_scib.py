import numpy as np
import pytest

anndata = pytest.importorskip("anndata")
scib = pytest.importorskip("scib")  # heavy; skip if absent

from multibench.eval import scib as escib

# Note: scib's LISI graph metrics (cLISI/iLISI) rely on a prebuilt binary that
# may not load on every platform (macOS arm64, older glibc). `compute()` degrades
# gracefully — those metrics come back NaN with a warning — so these tests run
# wherever scib is installed and assert on the always-computable metrics + that
# every metric key is present.


def _toy():
    rng = np.random.default_rng(0)
    # two well-separated cell types in 5-D embedding, two batches
    a = rng.normal(0, 0.1, size=(40, 5))
    b = rng.normal(5, 0.1, size=(40, 5))
    emb = np.vstack([a, b])
    celltype = np.array([0] * 40 + [1] * 40)
    cluster = celltype.copy()
    batch = np.array(([0, 1] * 40))
    return emb, celltype, cluster, batch


def test_clustering_metrics_present_and_bounded():
    emb, ct, cl, ba = _toy()
    res = escib.compute(emb, ct, cl, ba, group="clustering")
    for k in ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI"]:
        assert k in res.index
    assert 0.0 <= float(res.loc["ARI", "Value"]) <= 1.0
    # identical cluster==celltype -> high ARI
    assert float(res.loc["ARI", "Value"]) > 0.9


def test_batch_metrics_present():
    emb, ct, cl, ba = _toy()
    res = escib.compute(emb, ct, cl, ba, group="batch")
    for k in ["ASW_batch", "GC", "iLISI", "kBET"]:
        assert k in res.index


def test_clustering_derived_when_no_cluster_supplied():
    """evaluate()/compute() must run directly on an embedding: when no
    precomputed clustering is passed, one is derived via optimal-resolution
    Leiden. Two well-separated cell types should be recovered (high ARI)."""
    emb, ct, _cl, ba = _toy()
    res = escib.compute(emb, ct, None, ba, group="clustering")
    for k in ["ARI", "NMI", "ASW"]:
        assert k in res.index
    assert float(res.loc["ARI", "Value"]) > 0.9


def test_pipeline_evaluate_without_clustering():
    """Public evaluate() closes the chain with labels only (clustering=None)."""
    from multibench.eval.pipeline import evaluate
    emb, ct, _cl, _ba = _toy()
    df = evaluate(emb, category="vertical", task="clustering", labels=ct)
    assert "ARI" in df.index
    assert float(df.loc["ARI", "Value"]) > 0.9
