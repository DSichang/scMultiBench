from multibench.data import results


def test_load_results_returns_tidy_long_frame(result_dir):
    df = results.load_results(category="diagonal", dataset="D27", result_path=result_dir)
    assert set(["metric", "value", "method", "dataset", "category"]).issubset(df.columns)
    assert (df["category"] == "diagonal").all()
    assert (df["dataset"] == "D27").all()


def test_known_value_matches_disk(result_dir):
    df = results.load_results(category="diagonal", dataset="D27", result_path=result_dir)
    row = df[(df["method"] == "scBridge") & (df["metric"] == "ARI")]
    assert len(row) == 1
    assert abs(float(row["value"].iloc[0]) - 0.84374879) < 1e-6


def test_clustering_variant_changes_values(result_dir):
    default = results.load_results(category="diagonal", dataset="D27",
                                   clustering="default", result_path=result_dir)
    louvain = results.load_results(category="diagonal", dataset="D27",
                                   clustering="louvain", result_path=result_dir)
    # ASW genuinely differs between default (corrected ~0.8478) and louvain
    # (~0.8481): louvain must reflect its own file, not the default correction.
    d = float(default[(default.method == "scBridge") & (default.metric == "ASW")]["value"].iloc[0])
    lo = float(louvain[(louvain.method == "scBridge") & (louvain.metric == "ASW")]["value"].iloc[0])
    assert abs(lo - d) > 1e-6
    # ARI also differs between the two clustering variants.
    d_ari = float(default[(default.method == "scBridge") & (default.metric == "ARI")]["value"].iloc[0])
    lo_ari = float(louvain[(louvain.method == "scBridge") & (louvain.metric == "ARI")]["value"].iloc[0])
    assert abs(lo_ari - d_ari) > 1e-6


def test_metric_filter(result_dir):
    df = results.load_results(category="diagonal", dataset="D27",
                              metric=["ARI", "NMI"], result_path=result_dir)
    assert set(df["metric"].unique()) <= {"ARI", "NMI"}


def test_methods_are_canonicalized(result_dir):
    df = results.load_results(category="diagonal", result_path=result_dir)
    # no result-dir-style dotted names leak through
    assert not any("." in m for m in df["method"].unique())


# ---------------------------------------------------------------------------
# provenance (`source`), clustering-variant dirs, filters, error texts
# ---------------------------------------------------------------------------
import warnings

import pandas as pd
import pytest

import multibench as mtb

COLS = ["metric", "value", "method", "dataset", "category", "clustering", "source"]


def _val(df, method, metric):
    row = df[(df.method == method) & (df.metric == metric)]
    assert len(row) == 1, row
    return float(row.value.iloc[0])


def test_source_column_and_values(result_dir):
    pub = results.load_results("diagonal", dataset="D28", source="published",
                               result_path=result_dir)
    assert list(pub.columns) == COLS
    assert (pub.source == "published").all() and (pub.clustering == "default").all()
    assert abs(_val(pub, "iNMF", "ARI") - 0.198677) < 1e-6
    rr = results.load_results("diagonal", dataset="D28", source="rerun",
                              result_path=result_dir)
    assert (rr.source == "rerun-0.2.1").all()
    assert rr.method.nunique() == 12
    assert abs(_val(rr, "iNMF", "ARI") - 0.472126) < 1e-6


def test_source_both_concat(result_dir):
    import matplotlib; matplotlib.use("Agg")
    pub = results.load_results("diagonal", dataset="D28", result_path=result_dir)
    rr = results.load_results("diagonal", dataset="D28", source="rerun", result_path=result_dir)
    both = results.load_results("diagonal", dataset="D28", source="both", result_path=result_dir)
    assert len(both) == len(pub) + len(rr)
    assert list(both.columns) == COLS
    assert set(both.source.unique()) == {"published", "rerun-0.2.1"}
    # a 7-column frame (one source at a time) still plots
    from matplotlib.figure import Figure
    assert isinstance(mtb.plot.bubble(rr), Figure)


def test_suffix_dir_splits_method_and_clustering(result_dir):
    lo = results.load_results("vertical", dataset="D3", clustering="louvain",
                              result_path=result_dir)
    assert "Concerto" in set(lo.method)
    assert (lo[lo.method == "Concerto"].clustering == "louvain").all()
    default = results.load_results("vertical", dataset="D3", result_path=result_dir)
    assert not any("_louvain" in m for m in default.method.unique())
    assert "Concerto" not in set(default.method)   # the dir IS the louvain variant
    for cat in ("vertical", "diagonal", "cross"):
        df = results.load_results(cat, result_path=result_dir)
        assert not any(m.endswith(("_louvain", "_kmeans")) for m in df.method.unique()), cat


def test_zero_row_filter_warns(result_dir):
    with pytest.warns(UserWarning, match="Concerto"):
        df = results.load_results("vertical", dataset="D11", method="Concerto",
                                  result_path=result_dir)
    assert len(df) == 0 and list(df.columns) == COLS
    with pytest.warns(UserWarning, match="ZZZ"):
        df = results.load_results("vertical", dataset="D11", metric="ZZZ",
                                  result_path=result_dir)
    assert len(df) == 0


def test_method_filter_accepts_list_and_aliases(result_dir):
    df = results.load_results("diagonal", dataset="D27", method=["scbridge", "Seurat v3"],
                              result_path=result_dir)
    assert set(df.method) == {"scBridge", "Seurat_v3"}
    df2 = results.load_results("diagonal", dataset="D27", methods=["scBridge"],
                               result_path=result_dir)
    assert set(df2.method) == {"scBridge"}
    with pytest.raises(ValueError, match="either method= or methods="):
        results.load_results("diagonal", dataset="D27", method="scBridge",
                             methods=["scBridge"], result_path=result_dir)


def test_task_filters_by_family(result_dir):
    full = results.load_results("diagonal", dataset="D28", result_path=result_dir)
    b = results.load_results("diagonal", dataset="D28", task="batch", result_path=result_dir)
    assert set(b.metric) <= set(mtb.plot.BATCH_METRICS) and len(b)
    c = results.load_results("diagonal", dataset="D28", task="clustering", result_path=result_dir)
    assert set(c.metric) <= set(mtb.plot.CLUSTERING_METRICS) and len(c)
    assert len(b) + len(c) == len(full)      # the two families partition the frame
    with pytest.raises(ValueError, match="unknown task"):
        results.load_results("diagonal", dataset="D28", task="bogus", result_path=result_dir)
    # default task=None == task='all' == today's frame (positional call, as api_verify does)
    pos = results.load_results("diagonal", None, "scib", "D28", result_path=result_dir)
    pd.testing.assert_frame_equal(pos, full)
    pd.testing.assert_frame_equal(
        results.load_results("diagonal", dataset="D28", task="all", result_path=result_dir), full)


def test_result_path_single_csv(tmp_path, result_dir):
    rr = results.load_results("cross", dataset="D52", source="rerun", result_path=result_dir)
    five = rr[["metric", "value", "method", "dataset", "category"]]
    f = tmp_path / "mine.csv"
    five.to_csv(f, index=False)
    back = results.load_results(result_path=f)
    assert list(back.columns) == COLS
    assert (back.source == "user").all() and len(back) == len(five)
    assert set(back.method) == set(five.method)
    # a round-trip of a 7-col frame keeps its own provenance
    rr.to_csv(f, index=False)
    assert (results.load_results(result_path=f).source == "rerun-0.2.1").all()
    # not a long CSV -> ValueError naming what is missing
    pd.DataFrame({"Value": [1.0]}, index=["ARI"]).to_csv(tmp_path / "wide.csv")
    with pytest.raises(ValueError, match=r"missing column\(s\) \['metric', 'value', 'method'\]"):
        results.load_results(result_path=tmp_path / "wide.csv")


def test_missing_published_error_text(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        results.load_results("vertical", result_path=tmp_path)
    msg = str(e.value)
    assert "result_path" in msg and "multibench/result/scib_metric" in msg
    assert "source='rerun'" in msg


def test_mosaic_rerun_loads(result_dir):
    m = results.load_results("mosaic", source="rerun", result_path=result_dir)
    assert set(m.dataset) == {"D45", "D45s"}
    with pytest.raises(FileNotFoundError):
        results.load_results("mosaic", source="published", result_path=result_dir)
    # 'both' on mosaic = rerun only, no error
    b = results.load_results("mosaic", source="both", result_path=result_dir)
    assert set(b.source) == {"rerun-0.2.1"}


def test_category_none_unions_everything(result_dir):
    df = results.load_results(result_path=result_dir)
    assert set(df.category) == {"cross", "diagonal", "vertical"}    # mosaic has no published
    both = results.load_results(source="both", result_path=result_dir)
    assert "mosaic" in set(both.category)
    with pytest.raises(ValueError, match="unknown category"):
        results.load_results("sideways", result_path=result_dir)
    with pytest.raises(ValueError, match="unknown source"):
        results.load_results("vertical", source="paper", result_path=result_dir)


def test_dataset_list_and_unknown_dataset(result_dir):
    df = results.load_results("diagonal", dataset=["D27", "D28"], result_path=result_dir)
    assert set(df.dataset) == {"D27", "D28"}
    with pytest.raises(FileNotFoundError, match="D999"):
        results.load_results("diagonal", dataset="D999", result_path=result_dir)
    with pytest.raises(FileNotFoundError, match="no re-run sweep"):
        results.load_results("diagonal", dataset="D999", source="rerun", result_path=result_dir)
