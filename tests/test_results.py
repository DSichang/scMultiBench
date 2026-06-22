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
