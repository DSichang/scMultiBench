"""The article's clustering protocol (SINFONIA's ``get_N_clusters`` + scib_metrics.py).

ARI, NMI, ASW, iASW and iF1 score Leiden clusters searched to the number of
cell types; ARI_batch and NMI_batch = 1 - |ARI| / 1 - |NMI| of clusters searched to
the number of batches, only with two or more batches.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scanpy")
pytest.importorskip("scib")

import multibench as mtb
from multibench.eval import scib as S


def _blobs(k=4, n_per=50, dims=5, seed=0):
    rng = np.random.default_rng(seed)
    emb = np.vstack([rng.normal(8 * i, 0.5, size=(n_per, dims)) for i in range(k)])
    return emb, np.repeat([f"t{i}" for i in range(k)], n_per)


def test_the_search_finds_one_cluster_per_cell_type():
    emb, ct = _blobs(k=4)
    cl = S.leiden_n_clusters(emb, 4)
    assert len(cl) == len(ct) and len(set(cl)) == 4
    assert pd.crosstab(cl, ct).gt(0).sum(axis=1).eq(1).all()   # each cluster one blob


def test_the_search_keeps_the_last_clustering_when_it_cannot_reach_n(monkeypatch):
    emb, _ = _blobs(k=2)
    seen = []

    def two_clusters(adata, resolution, key_added, flavor):
        seen.append(resolution)
        # two clusters whatever the resolution, labelled by the step
        adata.obs[key_added] = pd.Categorical(
            np.where(np.arange(adata.n_obs) % 2, f"a{len(seen)}", f"b{len(seen)}"))

    monkeypatch.setattr(S, "_leiden", two_clusters)
    with pytest.warns(UserWarning, match=r"^No Leiden resolution gave 5 clusters; using "
                                         r"the 2 clusters of the last try\.$"):
        cl = S.leiden_n_clusters(emb, 5, flavor="leidenalg")
    assert len(seen) == S._SEARCH_STEPS == 30
    assert seen[:3] == [1.5, 2.25, 2.625]       # bisected upwards: too few clusters
    assert set(cl) == {f"a{len(seen)}", f"b{len(seen)}"}


def test_the_graph_is_scanpys_default_one_on_50_pcs_with_nan_as_zero():
    import anndata as ad
    import scanpy as sc
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(120, 60))
    emb[3, 7] = np.nan
    ref = ad.AnnData(np.nan_to_num(emb))
    with pytest.warns(UserWarning):           # scanpy: falling back to PCA
        sc.pp.neighbors(ref)                  # the call of SINFONIA.py
    got = S._clustering_graph(emb)
    assert got.obsm["X_pca"].shape[1] == 50
    assert (got.obsp["connectivities"] != ref.obsp["connectivities"]).nnz == 0
    narrow = S._clustering_graph(emb[:, :10])
    assert "X_pca" not in narrow.obsm


def test_asw_and_iasw_score_the_clusters_not_the_cell_types():
    from sklearn.metrics import silhouette_score
    emb, ct = _blobs(k=3)
    clusters = np.where(emb[:, 0] < 4, "low", "high")       # two clusters, three types
    batch = np.tile([0, 1], len(ct) // 2)
    a = S.compute(emb, ct, clusters, batch, only={"ASW", "iASW"}, verbose=False)
    b = S.compute(emb, ct[::-1], clusters, batch, only={"ASW", "iASW"}, verbose=False)
    assert a["Value"].tolist() == b["Value"].tolist()        # labels play no part
    assert a.loc["ASW", "Value"] == pytest.approx((silhouette_score(emb, clusters) + 1) / 2)
    assert a.attrs["clustering"] == "user"


def test_if1_scores_the_given_clusters():
    emb, ct = _blobs(k=3)
    batch = np.tile([0, 1], len(ct) // 2)
    same = S.compute(emb, ct, ct, batch, only={"iF1"}, verbose=False)
    merged = S.compute(emb, ct, np.where(ct == "t2", "t1", ct), batch, only={"iF1"},
                       verbose=False)
    assert same.loc["iF1", "Value"] == pytest.approx(1.0)
    # t1 and t2 share one cluster of 100 cells: F1 = 2 * 0.5 / 1.5 for each
    assert merged.loc["iF1", "Value"] == pytest.approx((1 + 2 * (2 / 3)) / 3)
    assert same.attrs["leiden_flavor"] is None             # no Leiden ran


def test_without_a_clustering_the_searched_clusters_are_scored():
    emb, ct = _blobs(k=3)
    out = S.compute(emb, ct, None, np.zeros(len(ct)), only={"ARI", "NMI"}, verbose=False)
    assert out.loc["ARI", "Value"] == pytest.approx(1.0)
    assert out.attrs["clustering"] == "search"


def test_a_failed_cell_type_search_says_so(monkeypatch):
    emb, ct = _blobs(k=2)
    monkeypatch.setattr(S, "_SEARCH_STEPS", 1)               # 1.5 gives > 2 clusters here
    monkeypatch.setattr(S, "_SEARCH_RANGE", (2.0, 4.0))
    with pytest.warns(UserWarning, match=r"gave 2 clusters, one per cell type; using the "
                                         r"\d+ clusters of the last try\.$"):
        S.compute(emb, ct, None, np.zeros(len(ct)), only={"ARI"}, verbose=False)


@pytest.mark.filterwarnings("ignore:No Leiden resolution")
def test_ari_batch_and_nmi_batch_are_one_minus_the_batch_scores():
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    rng = np.random.default_rng(2)
    emb, ct = _blobs(k=3)
    batch = rng.integers(0, 3, len(ct)).astype(str)
    emb = emb + 4 * (batch == "1")[:, None]                  # a visible batch effect
    out = S.compute(emb, ct, None, batch, group="batch",
                    only={"ARI_batch", "NMI_batch"}, verbose=False)
    cl = S.leiden_n_clusters(emb, 3)
    assert out.loc["ARI_batch", "Value"] == pytest.approx(
        1 - abs(adjusted_rand_score(batch, cl)))
    assert out.loc["NMI_batch", "Value"] == pytest.approx(
        1 - normalized_mutual_info_score(batch, cl, average_method="arithmetic"))


@pytest.mark.filterwarnings("ignore:No Leiden resolution")
def test_ari_batch_and_nmi_batch_are_absent_with_one_batch():
    emb, ct = _blobs(k=3)
    out = S.compute(emb, ct, None, np.zeros(len(ct)), group="all", verbose=False)
    assert "ASW_batch" in out.index
    assert not {"ARI_batch", "NMI_batch"} & set(out.index)
    two = S.compute(emb, ct, None, np.tile([0, 1], len(ct) // 2), group="batch",
                    verbose=False)
    assert {"ARI_batch", "NMI_batch"} <= set(two.index)


def test_the_batch_family_lists_them():
    assert mtb.plot.BATCH_METRICS[:3] == ["ASW_batch", "ARI_batch", "NMI_batch"]
    assert {"ARI_batch", "NMI_batch"} <= set(mtb.catalog.metrics()["metric"])
    assert mtb.catalog.canonical_metric("ari_batch") == "ARI_batch"


def test_published_tables_read_ari_batch_and_nmi_batch_from_their_file():
    df = mtb.load_results("diagonal", dataset="D28", source="published")
    v = df[df.method == "iNMF"].set_index("metric")["value"]
    assert v["ARI_batch"] == pytest.approx(0.944977134818114)
    assert v["NMI_batch"] == pytest.approx(0.9562164841965417)
    assert v["ARI"] == pytest.approx(0.1986774572091863)    # metric.csv's own ARI


def test_leiden_sweep_is_deprecated():
    emb, _ = _blobs(k=2, n_per=20)
    with pytest.warns(FutureWarning, match="leiden_sweep is deprecated and will be "
                                           "removed in 0.5. Use leiden_n_clusters."):
        S.leiden_sweep(emb)
