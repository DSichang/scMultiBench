import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import resolve

# source= / result_path= and available_datasets are deprecated in 0.4 (one
# FutureWarning each); the published-table tests below pass them on purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.* is deprecated and will be removed in 0.5:FutureWarning")


def test_available_datasets_lists_diagonal(result_dir):
    with pytest.warns(FutureWarning, match=r"available_datasets is deprecated .* "
                                           r"Use mtb\.catalog\.datasets\(\)\."):
        ds = mtb.available_datasets("diagonal")
    assert isinstance(ds, list) and "D28" in ds
    # mosaic has no published results -> empty list (not an error)
    assert mtb.available_datasets("mosaic", source="published") == []


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


def test_to_long_exposed_top_level():
    import pandas as pd
    wide = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    long = mtb.to_long(wide, method="M", dataset="D", category="vertical")
    assert list(long.columns) == ["metric", "value", "method", "dataset", "category",
                                  "source", "scored_with"]


def test_namespace_all_hygiene():
    # env/config/io expose a curated __all__ (no leaked stdlib imports)
    # recipe left __all__ in 0.4.0 (X34); it stays importable
    assert "recipe" not in mtb.env.__all__ and "subprocess" not in mtb.env.__all__
    assert callable(mtb.env.recipe)
    # config.Config and io.to_canonical are deprecated in 0.4.0: out of __all__
    assert mtb.config.__all__ == ["DEFAULT"]
    assert "category_folder" not in mtb.config.__all__      # internal token map
    assert sorted(mtb.io.__all__) == ["export_dataset", "read_canonical"]


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
def test_available_datasets_warns_on_missing_root(tmp_path, result_dir):
    with pytest.warns(UserWarning, match="does not exist"):
        assert mtb.available_datasets("vertical", result_path=tmp_path / "nope") == []
    # the deprecated function lists the stored (re-run) datasets in 0.4.x (X59)
    assert mtb.available_datasets("mosaic") == ["D45"]
    assert mtb.available_datasets("mosaic", source="published") == []
    assert "D28" in mtb.available_datasets()
    assert "D28" in mtb.available_datasets("diagonal")


def test_results_coverage(result_dir, layout_tree):
    from multibench.data.results import results_coverage
    cov = results_coverage("cross", source="both")
    assert list(cov.columns) == ["category", "dataset", "method", "clustering", "source"]
    d52 = cov[cov.dataset == "D52"]
    assert set(d52[d52.source == "published"].method) == {"scMoMaT"}
    assert d52[d52.source == "stored"].method.nunique() == 7      # UINMF's rows left (r3)
    # clustering variants surface: a louvain-only Concerto_louvain directory
    # (vertical D3 of the layout tree) shows up under clustering='louvain'
    allc = results_coverage(source="published", result_path=layout_tree)
    row = allc[(allc.dataset == "D3") & (allc.method == "Concerto")]
    assert set(row.clustering) == {"louvain"}
    assert results_coverage("mosaic", source="published").empty


def test_recommend_ranks_with_coverage(result_dir):
    import warnings
    from multibench.data.results import recommend
    # the shipped diagonal tables score the same 14 methods on each of D24,
    # D25 and D28: a complete matrix, so no coverage note at all
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        r = recommend("diagonal", source="published")
    # 0.4.0: no n_datasets, n_datasets_total, coverage or datasets columns
    assert list(r.columns) == ["method", "grand_score", "needs_labels", "runtime_tier",
                               "worst_sec", "env"]
    assert r.grand_score.notna().all()
    scored = r.grand_score.dropna()
    assert scored.is_monotonic_decreasing
    sb = r[r.method == "scBridge"].iloc[0]
    assert sb.needs_labels is True or sb.needs_labels == True   # noqa: E712
    assert sb.runtime_tier in {"fast", "medium", "slow", "very_slow", "unknown"}
    # source='both': the stored D28 rows join the published D28 ones (one
    # table per category since 0.4.0, X04)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        rb = recommend("diagonal", source="both")
    assert rb.grand_score.dropna().is_monotonic_decreasing
    # scored rows precede the NaN (unscored) tail
    assert rb.grand_score.notna().tolist() == sorted(rb.grand_score.notna().tolist(), reverse=True)


def test_recommend_drops_singleton_datasets_and_warns(layout_tree):
    from multibench.data.results import recommend
    with pytest.warns(UserWarning, match="fewer than 2 methods") as rec:
        r = recommend("cross", source="published", result_path=layout_tree)
    rec = [w for w in rec if not issubclass(w.category, FutureWarning)]
    # the layout tree's cross part has ONE method with a metric table in every
    # dataset but D53 (six methods) and D57 (UINMF and MOFA2's nested
    # filtered5/metric.csv). The singleton datasets must be dropped, so
    # scMoMaT (D52/D58/D59 alone), Concerto (D54) and MOFA2 (D56) cannot score
    # 1.0 on the strength of singleton min-max
    # scMoMaT has rows only in the dropped singleton datasets: it is LISTED
    # (wired for cross) but unscored, and the warning says why
    smt = r[r.method == "scMoMaT"].iloc[0]
    assert pd.isna(smt.grand_score)
    assert "scMoMaT" in r.attrs["not_scored"]
    assert ("scMoMaT has scores only on datasets left out of the "
            "ranking, so it is listed last.") in str(rec[0].message)
    # a 1.0 is now only ever the best of >= 2 methods on a kept dataset
    # (sciPENN on D53, UINMF over MOFA2 on D57), never a singleton artefact:
    # every scored method sits in a dataset that holds another method
    assert {"UINMF", "MOFA2"} <= set(r.method)
    msg = str(rec[0].message)
    # D56's only published table is MOFA2's: a singleton, left out and named
    assert "Datasets D52, D54, D56, D58 and D59 have fewer than 2 methods" in msg
    assert "D57" not in msg.splitlines()[1] and "is always 1.0" in msg
    # ONE warning, one line per finding, the dropped-datasets line first
    assert len([w for w in rec if issubclass(w.category, UserWarning)
                and "recommend(" in str(w.message)]) == 1
    assert msg.splitlines()[1].strip().startswith(
        "- Datasets D52, D54, D56, D58 and D59 have fewer than 2 methods and are left "
        "out of the ranking.")


def test_recommend_single_dataset_category_names_the_other_source(result_dir):
    """The shipped published cross table holds one method (scMoMaT on D52):
    nothing can be ranked, and the error must point at the stored results,
    which hold seven methods for it."""
    import warnings
    from multibench.data.results import recommend
    with pytest.raises(ValueError, match=r"^No cross dataset has 2 or more methods\. D52 has "
                                         r"1\. A min-max score over one method is always "
                                         r"1\.0\.") as e:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            recommend("cross", source="published")
    msg = str(e.value)
    assert ("The stored results have 7 methods for cross: Concerto, sciPENN, scMDC, scMM, "
            "scMoMaT, StabMap, totalVI. Leave out source= to rank them.") in msg
    assert "meaningless" not in msg and ">=" not in msg and "(s)" not in msg
    assert msg.endswith("Otherwise pass long_df= with more methods.")   # no min_methods
    # that source ranks them on D52, nothing dropped
    r, m = _rec("cross")
    assert r.grand_score.notna().sum() == 7
    assert "dropped" not in m
    # a single-dataset category with several methods ranks (vertical: D11)
    r2, _ = _rec("vertical", source="published")
    assert r2.grand_score.notna().sum() == 6


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

def _rec(*a, **k):
    import warnings
    from multibench.data.results import recommend
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = recommend(*a, **k)
    msgs = [str(w.message) for w in rec if "recommend(" in str(w.message)
            and not issubclass(w.category, FutureWarning)]
    return r, (msgs[0] if msgs else "")


def test_recommend_lists_unscored_methods(result_dir):
    r, msg = _rec("vertical", modalities=["rna", "adt"], source="published")
    nan_rows = r[r.grand_score.isna()]
    assert set(nan_rows.method) == {"Concerto", "Matilda", "MOFA2", "Seurat_WNN", "UINMF",
                                    "VIMCCA", "scMDC", "totalVI"}
    assert nan_rows.env.notna().all() and nan_rows.runtime_tier.notna().all()
    # column list (0.4.0: no coverage columns); scored rows first
    assert list(r.columns) == ["method", "grand_score", "needs_labels", "runtime_tier",
                               "worst_sec", "env"]
    assert r.grand_score.iloc[: len(r) - len(nan_rows)].notna().all()
    # c1 (C04): every metric present by default; (C17) no line for the unscored rows
    assert r.attrs["family"] == "all" and r.attrs["metrics"] == "all"
    assert r.attrs["source"] == "published"
    assert r.attrs["not_scored"] == r.attrs["missing"] == sorted(nan_rows.method, key=str.lower)
    assert "listed last: " not in msg
    # the coverage line appears only where the matrix IS incomplete
    # (the single published vertical dataset scores its 6 methods completely;
    # diagonal source='both' lacks GLUE / Seurat_v5 in the re-run rows)
    assert "only part of the datasets" not in msg
    d28 = mtb.load_results("diagonal")
    part = pd.concat([d28, d28[d28.method != "iNMF"].assign(dataset="D28b")])
    _, msg_b = _rec("diagonal", long_df=part)
    assert "Some methods have scores on only part of the datasets: iNMF 1 of 2" in msg_b
    assert "coverage column" not in msg_b        # that column left in 0.4.0


def test_recommend_rerun_missing_only_seurat_wnn(result_dir):
    r, msg = _rec("vertical", modalities=["rna", "adt"])
    assert set(r[r.grand_score.isna()].method) == {"Seurat_WNN"}
    assert r.attrs["source"] == "rerun" and r.attrs["not_scored"] == ["Seurat_WNN"]
    assert "Seurat_WNN has no stored results" not in msg            # c1 (C17)
    assert 'Try source="rerun"' not in msg


def test_recommend_long_df_records_source_and_family(result_dir):
    long = mtb.load_results("vertical", dataset="D11")
    r, _ = _rec("vertical", long_df=long, metrics="clustering")
    assert r.attrs["source"] == "long_df" and r.attrs["family"] == "clustering"
    r2, _ = _rec("vertical", long_df=long, metrics=["ARI", "NMI"])
    assert r2.attrs["family"] is None and r2.attrs["metrics"] == ["ARI", "NMI"]


def test_recommend_attrs_are_the_documented_keys(result_dir, layout_tree):
    """``frame.attrs`` carries exactly these keys."""
    keys = {"metrics", "family", "source", "not_scored", "missing", "dropped_methods"}
    for r, _ in (_rec("cross", source="published", result_path=layout_tree), _rec("vertical")):
        assert set(r.attrs) == keys


def test_recommend_scores_only_methods_the_registry_lists_for_the_category(
        result_dir, layout_tree_foreign):
    """recommend('cross') ranks only the methods list_methods('cross') lists.
    Rows of another registry method in the cross table (here scMSI and
    Seurat_WNN, vertical methods) used to be ranked, and shaped every other
    method's within-dataset rank."""
    layout_tree = layout_tree_foreign
    r, msg = _rec("cross", source="published", result_path=layout_tree)
    listed = set(mtb.find_methods(category="cross"))
    assert set(r.method) <= listed
    assert not ({"scMSI", "Seurat_WNN"} & set(r.method))
    assert r.attrs["dropped_methods"] == ["scMSI", "Seurat_WNN"]
    assert ("scMSI and Seurat_WNN have scores in the published table, but this "
            "package does not run them for cross, so they are left out of the "
            "ranking.") in msg
    assert "mtb.find_methods(category='cross') does not list them" in msg
    # MOFA2 and Multigrate are cross methods of the package and are ranked
    assert {"MOFA2", "Multigrate"} <= set(r.method[r.grand_score.notna()])
    # the dropped-datasets line stays first; the drop line follows it
    lines = [ln.strip() for ln in msg.splitlines()[1:]]
    assert lines[0].startswith("- Datasets ") and "left out of the ranking" in lines[0]
    assert lines[1].startswith("- scMSI and Seurat_WNN have scores in")
    # the same rule on a user frame: registry methods foreign to the category
    # are dropped and named ("long_df frame"), an unknown name (yours) is kept
    long = mtb.load_results("cross", dataset="D53", source="published",
                            result_path=layout_tree)
    mine = mtb.to_long(pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"]),
                       method="MyMethod", dataset="D53", category="cross")
    r2, msg2 = _rec("cross", long_df=pd.concat([long, mine]))
    assert "MyMethod" in set(r2.method) and "scMSI" not in set(r2.method)
    assert ("scMSI and Seurat_WNN have scores in the long_df frame, but this package "
            "does not run them for cross") in msg2
    # a category where every stored method is listed: nothing dropped, no line
    r3, msg3 = _rec("vertical")
    assert r3.attrs["dropped_methods"] == [] and "does not run" not in msg3
    # nothing rankable left -> ValueError naming the culprits
    with pytest.raises(ValueError, match=r"every row in long_df belongs to a method this package does not run for cross \(scMSI"):
        mtb.recommend("cross", long_df=long[long.method == "scMSI"])


def test_recommend_methods_keyword(layout_tree):
    """methods= for parity with load_results / scan / run_all (the instructor
    reached for it and got a TypeError)."""
    from multibench.data.results import recommend
    result_dir = layout_tree        # cross: D53 holds several methods, D52 one
    r, msg = _rec("cross", methods=["scmdc", "sciPENN", "scMoMaT"], source="published",
                  result_path=result_dir)
    assert r.method.tolist() == ["sciPENN", "scMDC", "scMoMaT"]        # alias/case tolerant
    assert r.attrs["not_scored"] == ["scMoMaT"]                        # restricted to the request
    assert "StabMap" not in msg and "totalVI" not in msg
    # a requested method without rows is still listed as unscored
    r2, msg2 = _rec("cross", methods=["sciPENN", "scMDC", "totalVI"], source="published",
                    result_path=result_dir)
    assert r2.method.tolist() == ["sciPENN", "scMDC", "totalVI"]
    assert "totalVI has no published scores" not in msg2            # c1 (C17)
    with pytest.raises(KeyError, match=r"Unknown method Matlida. Did you mean Matilda\?"):
        recommend("cross", methods=["Matlida"], source="published", result_path=result_dir)
    with pytest.raises(ValueError, match=r"none of methods=\['totalVI'\] has rows in the published table for cross"):
        recommend("cross", methods=["totalVI"], source="published", result_path=result_dir)
    # category is the only positional; every selector is keyword-only
    import inspect
    params = list(inspect.signature(recommend).parameters)
    assert params[0] == "category" and "methods" in params and "metrics" in params
    assert inspect.signature(recommend).parameters["methods"].kind is inspect.Parameter.KEYWORD_ONLY


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


def test_recommend_metrics_batch_on_diagonal_rerun(result_dir):
    r, _ = _rec("diagonal", metrics="batch")
    assert r.attrs["family"] == "batch" and r.grand_score.notna().any()
    # the explicit list of the same family scores the same
    m = mtb.catalog.metrics()
    batch = m.loc[m.family == "batch", "metric"].tolist()
    r2, _ = _rec("diagonal", metrics=batch)
    assert r2.attrs["family"] is None and r2.attrs["metrics"] == batch
    pd.testing.assert_frame_equal(r, r2)
