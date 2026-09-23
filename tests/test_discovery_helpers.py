import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import resolve


def test_available_datasets_lists_diagonal(result_dir):
    ds = mtb.available_datasets("diagonal", result_path=result_dir)
    assert isinstance(ds, list) and "D28" in ds
    # mosaic has no published results -> empty list (not an error)
    assert mtb.available_datasets("mosaic", result_path=result_dir) == []


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
                                  "clustering", "source"]


def test_namespace_all_hygiene():
    # env/config/io expose a curated __all__ (no leaked stdlib imports)
    assert "recipe" in mtb.env.__all__ and "subprocess" not in mtb.env.__all__
    assert "Config" in mtb.config.__all__ and "Path" not in mtb.config.__all__
    assert "category_folder" not in mtb.config.__all__      # internal token map
    assert "to_canonical" in mtb.io.__all__


# --- P12: inputs_for(check=None) warns about phantom paths; the default is silent

def test_inputs_for_check_none_warns_on_missing(tmp_path):
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")   # peak.h5 deliberately missing
    with pytest.warns(UserWarning, match="atac_peak") as rec:
        got = resolve.inputs_for("D27", "diagonal", "Seurat_v5", data_path=tmp_path,
                                 check=None)
    assert got["atac_peak"].endswith("/D27/atac_peak.h5")     # fallback path still returned
    msg = str(rec[0].message)
    assert "1 resolved input path(s) do not exist" in msg and "check=True to raise" in msg


def test_inputs_for_check_false_is_silent_and_is_the_default(tmp_path):
    import warnings
    d = tmp_path / "D27"; d.mkdir()
    (d / "rna.h5").write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = resolve.inputs_for("D27", "diagonal", "Seurat_v5", data_path=tmp_path,
                                 check=False)
        assert resolve.inputs_for("D27", "diagonal", "Seurat_v5", data_path=tmp_path) == got
    assert "atac_peak" in got


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
    assert mtb.available_datasets("mosaic", source="rerun", result_path=result_dir) == ["D45", "D45s"]
    assert mtb.available_datasets("mosaic", result_path=result_dir) == []   # published: still none
    assert "D28" in mtb.available_datasets()
    assert "D28" in mtb.available_datasets("diagonal", result_path=result_dir)


def test_results_coverage(result_dir, layout_tree):
    from multibench.data.results import results_coverage
    cov = results_coverage("cross", result_path=result_dir)
    assert list(cov.columns) == ["category", "dataset", "method", "clustering", "source"]
    d52 = cov[cov.dataset == "D52"]
    assert set(d52[d52.source == "published"].method) == {"scMoMaT"}
    assert d52[d52.source == "rerun"].method.nunique() == 8
    # clustering variants surface: a louvain-only Concerto_louvain directory
    # (vertical D3 of the layout tree) shows up under clustering='louvain'
    allc = results_coverage(result_path=layout_tree)
    row = allc[(allc.dataset == "D3") & (allc.method == "Concerto")]
    assert set(row.clustering) == {"louvain"}
    assert results_coverage("mosaic", source="published", result_path=result_dir).empty


def test_recommend_ranks_with_coverage(result_dir):
    import warnings
    from multibench.data.results import recommend
    # the shipped diagonal tables score the same 14 methods on each of D24,
    # D25 and D28: a complete matrix, so no coverage note at all
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        r = recommend("diagonal", result_path=result_dir)
    assert list(r.columns) == ["method", "grand_score", "n_datasets", "n_datasets_total",
                               "coverage", "needs_labels", "runtime_tier", "worst_sec",
                               "env", "output_kind", "datasets"]
    assert (r.n_datasets == r.n_datasets_total).all() and (r.coverage == 1.0).all()
    assert (r.n_datasets_total == 3).all() and r.grand_score.notna().all()
    scored = r.grand_score.dropna()
    assert scored.is_monotonic_decreasing
    sb = r[r.method == "scBridge"].iloc[0]
    assert sb.needs_labels is True or sb.needs_labels == True   # noqa: E712
    assert sb.runtime_tier in {"fast", "medium", "slow", "very_slow", "unknown"}
    # the re-run sweep scores 12 of the 14 on D28/D28s: source='both' makes
    # the matrix incomplete and the warning says so
    with pytest.warns(UserWarning, match="incomplete"):
        rb = recommend("diagonal", source="both", result_path=result_dir)
    assert (rb.n_datasets <= rb.n_datasets_total).all() and (rb.coverage < 1.0).any()
    assert (rb.n_datasets_total == 4).all()
    assert rb.grand_score.dropna().is_monotonic_decreasing
    # scored rows precede the NaN (unscored) tail
    assert rb.grand_score.notna().tolist() == sorted(rb.grand_score.notna().tolist(), reverse=True)


def test_recommend_drops_singleton_datasets_and_warns(layout_tree):
    from multibench.data.results import recommend
    with pytest.warns(UserWarning, match="fewer than 2 methods") as rec:
        r = recommend("cross", result_path=layout_tree)
    # the layout tree's cross part has ONE rankable method with a metric table
    # in every dataset but D53 (six methods, four of them wired for cross). D57
    # holds UINMF + MOFA2's nested filtered5/metric.csv, but MOFA2 is not a
    # cross method of this package (list_methods('cross') does not list it),
    # so once its rows are dropped D57 is a singleton too. The singleton
    # datasets must be dropped, so scMoMaT (D52/D58/D59 alone) and UINMF (D57
    # alone) cannot score 1.0 on the strength of singleton min-max
    assert (r.n_datasets_total == 1).all()       # only D53 holds >= 2 rankable methods
    # scMoMaT / UINMF have rows only in the dropped singleton datasets: they
    # are LISTED (wired for cross) but unscored, and the warning says why
    smt = r[r.method == "scMoMaT"].iloc[0]
    assert pd.isna(smt.grand_score) and smt.n_datasets == 0 and smt.coverage == 0.0
    assert {"scMoMaT", "UINMF"} <= set(r.attrs["not_scored"])
    assert "rows only in dropped dataset(s) for: scMoMaT, UINMF" in str(rec[0].message)
    # a 1.0 is now only ever the best of >= 2 methods on a kept dataset
    # (sciPENN on D53), never a singleton artefact: every scored method sits
    # in a dataset that holds another method
    assert "UINMF" in set(r.method) and "MOFA2" not in set(r.method)
    assert (r[r.grand_score == 1.0].n_datasets == 1).all()
    msg = str(rec[0].message)
    # D56's only published table is MOFA2's (dropped): it has no rankable rows
    # at all, so it is neither ranked nor listed as a singleton
    assert "D52" in msg and "D57" in msg and "D56" not in msg and "is always 1.0" in msg
    # ONE warning, one line per finding, the dropped-datasets line first
    assert len([w for w in rec if issubclass(w.category, UserWarning)
                and "recommend(" in str(w.message)]) == 1
    assert msg.splitlines()[1].strip().startswith("- dropped")
    with pytest.raises(ValueError, match="single-method") as e:
        recommend("cross", min_methods=50, result_path=layout_tree)
    assert "pass source=" not in str(e.value)       # that tree has no other source


def test_recommend_single_dataset_category_names_the_other_source(result_dir):
    """The shipped published cross table holds one method (scMoMaT on D52):
    nothing can be ranked, and the error must point at the re-run sweeps,
    which hold eight methods for it."""
    import warnings
    from multibench.data.results import recommend
    with pytest.raises(ValueError, match=r"no dataset in cross holds >= 2 methods \(1 dataset\(s\): \['D52'\]\)") as e:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            recommend("cross", result_path=result_dir)
    msg = str(e.value)
    assert ("The rerun tables hold 8 methods for cross (Concerto, sciPENN, scMDC, scMM, "
            "scMoMaT, StabMap, totalVI, UINMF): pass source='rerun' (or 'both')") in msg
    assert msg.endswith("Otherwise pass long_df= with more methods, or lower min_methods.")
    # that source ranks them on D52 + D52s, nothing dropped
    r, m = _rec("cross", source="rerun", result_path=result_dir)
    assert r.grand_score.notna().sum() == 8 and (r.n_datasets_total == 2).all()
    assert "dropped" not in m
    # a single-dataset category with several methods ranks (vertical: D11)
    r2, _ = _rec("vertical", result_path=result_dir)
    assert (r2.n_datasets_total == 1).all() and r2.grand_score.notna().sum() == 6


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
    with pytest.raises(ValueError, match=r"unknown metrics= token 'bogus'"):
        recommend("vertical", long_df=long, metrics="bogus")


# --- P04: unscored methods are named, source/family recorded on the frame ---

def _rec(*a, **k):
    import warnings
    from multibench.data.results import recommend
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = recommend(*a, **k)
    msgs = [str(w.message) for w in rec if "recommend(" in str(w.message)]
    return r, (msgs[0] if msgs else "")


def test_recommend_lists_unscored_methods(result_dir):
    r, msg = _rec("vertical", modalities=["rna", "adt"], result_path=result_dir)
    nan_rows = r[r.grand_score.isna()]
    assert set(nan_rows.method) == {"Concerto", "Matilda", "MOFA2", "Seurat_WNN", "UINMF",
                                    "VIMCCA", "scMDC", "totalVI"}
    assert (nan_rows.n_datasets == 0).all() and (nan_rows.coverage == 0.0).all()
    assert (nan_rows.n_datasets_total == r.n_datasets_total.iloc[0]).all()
    assert nan_rows.env.notna().all() and nan_rows.runtime_tier.notna().all()
    # column list (datasets appended in study round 1); scored rows first
    assert list(r.columns) == ["method", "grand_score", "n_datasets", "n_datasets_total",
                               "coverage", "needs_labels", "runtime_tier", "worst_sec",
                               "env", "output_kind", "datasets"]
    assert r.grand_score.iloc[: len(r) - len(nan_rows)].notna().all()
    assert r.attrs["family"] == "clustering" and r.attrs["metrics"] == "clustering"
    assert r.attrs["source"] == "published"
    assert r.attrs["not_scored"] == r.attrs["missing"] == sorted(nan_rows.method, key=str.lower)
    import re
    assert re.search(r"no rows in source='published' for: .*totalVI", msg)
    assert 'try source="rerun"' in msg and "listed with grand_score NaN" in msg
    # the old load-bearing phrases survive where the matrix IS incomplete
    # (the single published vertical dataset scores its 6 methods completely;
    # diagonal source='both' lacks GLUE / Seurat_v5 in the re-run rows)
    assert "incomplete" not in msg
    _, msg_b = _rec("diagonal", source="both", result_path=result_dir)
    assert "incomplete" in msg_b and "partial coverage" in msg_b


def test_recommend_rerun_missing_only_seurat_wnn(result_dir):
    r, msg = _rec("vertical", modalities=["rna", "adt"], source="rerun",
                  result_path=result_dir)
    assert set(r[r.grand_score.isna()].method) == {"Seurat_WNN"}
    assert r.attrs["source"] == "rerun" and r.attrs["not_scored"] == ["Seurat_WNN"]
    assert "no rows in source='rerun' for: Seurat_WNN" in msg
    assert 'try source="rerun"' not in msg


def test_recommend_long_df_records_source_and_family(result_dir):
    long = mtb.load_results("vertical", dataset="D11", source="rerun", result_path=result_dir)
    r, _ = _rec("vertical", long_df=long, metrics="clustering")
    assert r.attrs["source"] == "long_df" and r.attrs["family"] == "clustering"
    r2, _ = _rec("vertical", long_df=long, metrics=["ARI", "NMI"])
    assert r2.attrs["family"] is None and r2.attrs["metrics"] == ["ARI", "NMI"]


def test_recommend_attrs_are_the_documented_keys(result_dir, layout_tree):
    """``frame.attrs`` carries exactly the keys recommend's Notes list."""
    keys = {"metrics", "family", "source", "not_scored", "missing", "dropped_methods"}
    doc = mtb.recommend.__doc__
    for r, _ in (_rec("cross", result_path=layout_tree), _rec("vertical", result_path=result_dir)):
        assert set(r.attrs) == keys
    assert all(f'``"{k}"``' in doc for k in keys)


def test_recommend_scores_only_methods_the_registry_lists_for_the_category(result_dir, layout_tree):
    """recommend('cross') used to rank MOFA2 and Multigrate (rows in the
    published cross table) although list_methods('cross') does not list them
    - and their rows shaped every other method's within-dataset rank."""
    r, msg = _rec("cross", result_path=layout_tree)
    listed = set(mtb.list_methods(category="cross"))
    assert set(r.method) <= listed
    assert not ({"MOFA2", "Multigrate"} & set(r.method))
    assert r.attrs["dropped_methods"] == ["MOFA2", "Multigrate"]
    assert ("also scored in the published table but not run by this package for "
            "cross: MOFA2, Multigrate") in msg
    assert "mtb.list_methods(category='cross') does not list them" in msg
    # the dropped-datasets line stays first; the drop line follows it
    lines = [ln.strip() for ln in msg.splitlines()[1:]]
    assert lines[0].startswith("- dropped") and lines[1].startswith("- also scored")
    # the same rule on a user frame: registry methods foreign to the category
    # are dropped and named ("long_df frame"), an unknown name (yours) is kept
    long = mtb.load_results("cross", dataset="D53", result_path=layout_tree)
    mine = mtb.to_long(pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"]),
                       method="MyMethod", dataset="D53", category="cross")
    r2, msg2 = _rec("cross", long_df=pd.concat([long, mine]))
    assert "MyMethod" in set(r2.method) and "MOFA2" not in set(r2.method)
    assert "also scored in the long_df frame but not run by this package for cross: MOFA2, Multigrate" in msg2
    # a category where every stored method is listed: nothing dropped, no line
    r3, msg3 = _rec("vertical", result_path=result_dir)
    assert r3.attrs["dropped_methods"] == [] and "also scored" not in msg3
    # nothing rankable left -> ValueError naming the culprits
    with pytest.raises(ValueError, match=r"every row in long_df belongs to a method this package does not run for cross \(MOFA2"):
        mtb.recommend("cross", long_df=long[long.method == "MOFA2"])


def test_recommend_methods_keyword(layout_tree):
    """methods= for parity with load_results / scan / run_all (the instructor
    reached for it and got a TypeError)."""
    from multibench.data.results import recommend
    result_dir = layout_tree        # cross: D53 holds several methods, D52 one
    r, msg = _rec("cross", methods=["scmdc", "sciPENN", "scMoMaT"], result_path=result_dir)
    assert r.method.tolist() == ["sciPENN", "scMDC", "scMoMaT"]        # alias/case tolerant
    assert r.attrs["not_scored"] == ["scMoMaT"]                        # restricted to the request
    assert "StabMap" not in msg and "totalVI" not in msg
    # a requested method without rows is still listed as unscored
    r2, msg2 = _rec("cross", methods=["sciPENN", "scMDC", "totalVI"], result_path=result_dir)
    assert r2.method.tolist() == ["sciPENN", "scMDC", "totalVI"]
    assert "no rows in source='published' for: totalVI" in msg2
    with pytest.raises(KeyError, match=r"unknown method 'Matlida'; did you mean 'Matilda'\?"):
        recommend("cross", methods=["Matlida"], result_path=result_dir)
    with pytest.raises(ValueError, match=r"none of methods=\['totalVI'\] has rows in source='published' for cross"):
        recommend("cross", methods=["totalVI"], result_path=result_dir)
    # category is the only positional; every selector is keyword-only
    import inspect
    params = list(inspect.signature(recommend).parameters)
    assert params[0] == "category" and "methods" in params and "metrics" in params
    assert inspect.signature(recommend).parameters["methods"].kind is inspect.Parameter.KEYWORD_ONLY


def test_recommend_task_error_names_real_metrics(result_dir):
    from multibench.data.results import recommend
    # vertical ships no batch metrics; the error must list what IS there,
    # not "metrics present: []" (the loader used to pre-filter by task)
    with pytest.raises(ValueError, match=r"metrics present: \['ARI'"):
        recommend("vertical", metrics="batch", result_path=result_dir)
    with pytest.raises(ValueError, match=r"metrics present: \['ARI'"):
        recommend("vertical", metrics=["GC", "iLISI"], result_path=result_dir)


def test_recommend_metrics_batch_on_diagonal_rerun(result_dir):
    r, _ = _rec("diagonal", metrics="batch", source="rerun", result_path=result_dir)
    assert r.attrs["family"] == "batch" and r.grand_score.notna().any()
    # the explicit list of the same family scores the same
    r2, _ = _rec("diagonal", metrics=list(mtb.plot.BATCH_METRICS), source="rerun",
                 result_path=result_dir)
    assert r2.attrs["family"] is None and r2.attrs["metrics"] == list(mtb.plot.BATCH_METRICS)
    pd.testing.assert_frame_equal(r, r2)
