import pytest

import multibench as mtb
from multibench.engine import resolve


def test_available_datasets_lists_diagonal(result_dir):
    ds = mtb.available_datasets("diagonal", result_path=result_dir)
    assert isinstance(ds, list) and "D27" in ds
    # mosaic has no published results -> empty list (not an error)
    assert mtb.available_datasets("mosaic", result_path=result_dir) == []


def test_available_datasets_rejects_unwired_metric_set(result_dir):
    with pytest.raises(NotImplementedError):
        mtb.available_datasets("vertical", metric_set="classification", result_path=result_dir)


def test_inputs_for_check_raises_on_missing(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")  # peak.h5 deliberately missing
    with pytest.raises(FileNotFoundError):
        resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path, check=True)


def test_inputs_for_check_passes_when_present(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    for n in ["rna.h5", "peak.h5"]:
        (d / n).write_text("")
    got = resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path, check=True)
    assert "atac_peak" in got


def test_to_long_exposed_top_level():
    import pandas as pd
    wide = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    long = mtb.to_long(wide, method="M", dataset="D", category="vertical")
    assert list(long.columns) == ["metric", "value", "method", "dataset", "category"]


def test_namespace_all_hygiene():
    # env/config/io expose a curated __all__ (no leaked stdlib imports)
    assert "recipe" in mtb.env.__all__ and "subprocess" not in mtb.env.__all__
    assert "category_folder" in mtb.config.__all__ and "Path" not in mtb.config.__all__
    assert "to_canonical" in mtb.io.__all__


# --- P12: inputs_for's default warns about phantom paths ---------------------

def test_inputs_for_default_warns_on_missing(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")   # peak.h5 deliberately missing
    with pytest.warns(UserWarning, match="atac_peak") as rec:
        got = resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path)
    assert got["atac_peak"].endswith("/D27/atac_peak.h5")     # fallback path still returned
    msg = str(rec[0].message)
    assert "1 resolved input path(s) do not exist" in msg and "check=True to raise" in msg


def test_inputs_for_check_false_is_silent(tmp_path):
    import warnings
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path,
                                 check=False)
    assert "atac_peak" in got


def test_inputs_for_default_no_warning_when_present(tmp_path):
    import warnings
    d = tmp_path / "D27"; d.mkdir()
    for n in ["rna.h5", "peak.h5"]:
        (d / n).write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = resolve.inputs_for("D27", "Seurat_v5", "diagonal", data_path=tmp_path)
    assert got["atac_peak"].endswith("/D27/peak.h5")
def test_available_datasets_warns_on_missing_root(tmp_path, result_dir):
    with pytest.warns(UserWarning, match="does not exist"):
        assert mtb.available_datasets("vertical", result_path=tmp_path / "nope") == []
    assert mtb.available_datasets("mosaic", source="rerun", result_path=result_dir) == ["D45", "D45s"]
    assert mtb.available_datasets("mosaic", result_path=result_dir) == []   # published: still none
    assert "D27" in mtb.available_datasets()
    # only datasets holding the requested clustering file are listed
    assert mtb.available_datasets("vertical", clustering="louvain", result_path=result_dir) == ["D3"]
    assert "D27" in mtb.available_datasets("diagonal", clustering="kmeans", result_path=result_dir)


def test_results_coverage(result_dir):
    from multibench.data.results import results_coverage
    cov = results_coverage("cross", result_path=result_dir)
    assert list(cov.columns) == ["category", "dataset", "method", "clustering", "source"]
    d52 = cov[cov.dataset == "D52"]
    assert set(d52[d52.source == "published"].method) == {"scMoMaT"}
    assert d52[d52.source == "rerun-0.2.1"].method.nunique() == 8
    # clustering variants surface: Concerto's louvain-only D3 directory
    allc = results_coverage(result_path=result_dir)
    row = allc[(allc.dataset == "D3") & (allc.method == "Concerto")]
    assert set(row.clustering) == {"louvain"}
    assert results_coverage("mosaic", source="published", result_path=result_dir).empty


def test_recommend_ranks_with_coverage(result_dir):
    from multibench.data.results import recommend
    with pytest.warns(UserWarning, match="incomplete"):
        r = recommend("diagonal", result_path=result_dir)
    assert list(r.columns) == ["method", "grand_score", "n_datasets", "n_datasets_total",
                               "coverage", "needs_labels", "runtime_tier", "worst_sec",
                               "env", "output_kind"]
    assert (r.n_datasets <= r.n_datasets_total).all()
    assert r.grand_score.is_monotonic_decreasing
    sb = r[r.method == "scBridge"].iloc[0]
    assert sb.needs_labels is True or sb.needs_labels == True   # noqa: E712
    assert sb.runtime_tier in {"fast", "medium", "slow", "very_slow", "unknown"}


def test_recommend_drops_singleton_datasets_and_warns(result_dir):
    from multibench.data.results import recommend
    with pytest.warns(UserWarning, match="fewer than 2 methods") as rec:
        r = recommend("cross", result_path=result_dir)
    # the shipped cross tree has ONE method with a metric.csv in every dataset
    # but D53: those datasets must be dropped, so scMoMaT (D52/D58/D59 alone)
    # cannot score 1.0 on the strength of singleton min-max
    assert r.n_datasets_total.iloc[0] == 1       # only D53 holds >= 2 methods
    assert "scMoMaT" not in set(r.method)
    assert not ((r.n_datasets == 1) & (r.grand_score == 1.0)).sum() > 1
    msg = str(rec[0].message)
    assert "D52" in msg and "D57" in msg and "1.0 by construction" in msg
    with pytest.raises(ValueError, match="single-method"):
        recommend("cross", min_methods=50, result_path=result_dir)


def test_recommend_unknown_result_id_and_modalities(result_dir):
    import pandas as pd
    from multibench.data.results import recommend
    # a non-registry id (e.g. a result-dir token or the user's own method) gets
    # None metadata, no KeyError
    long = mtb.load_results("vertical", dataset="D11", result_path=result_dir)
    mine = long[long.method == "scMM"].assign(method="Concerto_louvain")
    mine = mine.assign(value=mine.value * 0.5)
    r = recommend("vertical", long_df=pd.concat([long, mine]))
    row = r[r.method == "Concerto_louvain"].iloc[0]
    assert row.needs_labels is None and row.env is None and row.runtime_tier is None
    # modalities filter goes through find_methods
    r2 = recommend("vertical", long_df=long, modalities=["rna", "adt"])
    assert set(r2.method) <= set(mtb.find_methods(category="vertical", modalities=["rna", "adt"]))
    # sciPENN / scMSI are RNA+ADT methods: asking for RNA+ATAC leaves nothing
    sub = long[long.method.isin(["sciPENN", "scMSI"])]
    with pytest.raises(ValueError, match="consumes modalities"):
        recommend("vertical", long_df=sub, modalities=["rna", "atac"])
    with pytest.raises(ValueError, match="unknown task"):
        recommend("vertical", long_df=long, task="bogus")
