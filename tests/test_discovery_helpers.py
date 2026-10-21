import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import resolve

# source= / result_path= and available_datasets are deprecated in 0.4 (one
# FutureWarning each); the published-table tests below pass them on purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.* is deprecated and will be removed in 0.5:FutureWarning")


def test_inputs_for_check_raises_on_missing(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")  # peak.h5 deliberately missing
    with pytest.raises(FileNotFoundError):
        resolve.inputs_for("D27", "diagonal", "Seurat_v5", data_path=tmp_path, check=True)


def test_inputs_for_check_passes_when_present(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    for n in ["rna.h5", "peak.h5"]:
        (d / n).write_text("")
    got = resolve.inputs_for("D27", "diagonal", "Seurat_v5", data_path=tmp_path, check=True)
    assert "atac_peak" in got


# --- P12: inputs_for(check=None) warns about phantom paths; the default is silent

def test_inputs_for_check_none_warns_on_missing(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")   # peak.h5 deliberately missing
    with pytest.warns(UserWarning, match="atac_peak") as rec:
        got = resolve.inputs_for("D27", "diagonal", "Seurat_v5", data_path=tmp_path,
                                 check=None)
    assert got["atac_peak"].endswith("/D27/atac_peak.h5")     # fallback path still returned
    msg = str(rec[0].message)
    assert msg.startswith(f"Seurat_v5 (diagonal) needs atac_peak.h5 in {d}. The folder "
                          f"holds rna.h5.")
    assert "Pass check=True to raise this as an error" in msg


def test_inputs_for_check_false_is_silent_and_check_is_the_default(tmp_path):
    import warnings
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = resolve.inputs_for("D27", "diagonal", "Seurat_v5", data_path=tmp_path,
                                 check=False)
    assert "atac_peak" in got
    # since 0.4 the default checks the files (it was check=False)
    with pytest.raises(FileNotFoundError, match="needs atac_peak.h5"):
        resolve.inputs_for("D27", "diagonal", "Seurat_v5", data_path=tmp_path)


def test_inputs_for_default_no_warning_when_present(tmp_path):
    import warnings
    d = tmp_path / "D27"; d.mkdir()
    for n in ["rna.h5", "peak.h5"]:
        (d / n).write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = resolve.inputs_for("D27", "diagonal", "Seurat_v5", data_path=tmp_path)
    assert got["atac_peak"].endswith("/D27/peak.h5")
def test_recommend_unknown_result_id_and_modalities(result_dir):
    import pandas as pd
    from multibench.data.results import recommend
    # a non-registry id (e.g. a result-dir token or the user's own method) gets
    # None metadata, no KeyError
    long = mtb.load_results("vertical", dataset="D11")
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
    with pytest.raises(ValueError, match=r"Unknown metrics= token bogus\."):
        recommend("vertical", long_df=long, metrics="bogus")


# --- P04: unscored methods are named, source/family recorded on the frame ---

def test_recommend_task_error_names_real_metrics(result_dir):
    from multibench.data.results import recommend
    # vertical ships no batch metrics; the error must list what IS there,
    # not "metrics present: []" (the loader used to pre-filter by task)
    # the family on a single-batch category says so (X89)
    with pytest.raises(ValueError, match=r'^The vertical tables have no batch scores '
                                         r'\(D11 has one batch\)\. Use metrics="clustering"\.$'):
        recommend("vertical", metrics="batch")
    with pytest.raises(ValueError, match=r"metrics present: \['ARI'"):
        recommend("vertical", metrics=["GC", "iLISI"])


def test_namespace_all_hygiene():
    # env/config/io expose a curated __all__ (no leaked stdlib imports)
    # recipe left __all__ in 0.4.0 (X34); it stays importable
    assert "recipe" not in mtb.env.__all__ and "subprocess" not in mtb.env.__all__
    assert callable(mtb.env.recipe)
    # mtb.config is the settings object: its names are the settings
    assert mtb.config.__all__ == ["data_path", "repo_path", "envs_dir", "leiden_flavor"]
    assert "category_folder" not in mtb.config.__all__      # internal token map
    assert sorted(mtb.io.__all__) == ["export_dataset", "read_canonical"]


def _rec(*a, **k):
    import warnings
    from multibench.data.results import recommend
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = recommend(*a, **k)
    msgs = [str(w.message) for w in rec if "recommend(" in str(w.message)
            and not issubclass(w.category, FutureWarning)]
    return r, (msgs[0] if msgs else "")


def test_recommend_rerun_missing_only_seurat_wnn(result_dir):
    r, msg = _rec("vertical", modalities=["rna", "adt"])
    assert set(r[r.grand_score.isna()].method) == {"Seurat_WNN"}
    assert r.attrs["source"] == "rerun" and r.attrs["missing"] == ["Seurat_WNN"]
    assert "Seurat_WNN has no stored results" not in msg            # c1 (C17)
    assert 'Try source="rerun"' not in msg


def test_recommend_long_df_records_source_and_family(result_dir):
    long = mtb.load_results("vertical", dataset="D11")
    r, _ = _rec("vertical", long_df=long, metrics="clustering")
    assert r.attrs["source"] == "long_df" and r.attrs["metrics"] == "clustering"
    r2, _ = _rec("vertical", long_df=long, metrics=["ARI", "NMI"])
    assert r2.attrs["metrics"] == ["ARI", "NMI"]


def test_recommend_metrics_batch_on_diagonal_rerun(result_dir):
    r, _ = _rec("diagonal", metrics="batch")
    assert r.attrs["metrics"] == "batch" and r.grand_score.notna().any()
    # the explicit list of the same family scores the same
    m = mtb.catalog.metrics()
    batch = m.loc[m.family == "batch", "metric"].tolist()
    r2, _ = _rec("diagonal", metrics=batch)
    assert r2.attrs["metrics"] == batch
    pd.testing.assert_frame_equal(r, r2)

