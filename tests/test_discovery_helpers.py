import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import resolve


def test_available_datasets_lists_diagonal(result_dir):
    ds = mtb.available_datasets("diagonal", result_path=result_dir)
    assert isinstance(ds, list) and "D27" in ds
    # mosaic has no published results -> empty list (not an error)
    assert mtb.available_datasets("mosaic", result_path=result_dir) == []


def test_available_datasets_rejects_unknown_metric_set(result_dir):
    # a typo is a ValueError listing the valid tokens (the same message
    # config.metric_set_dir gives), not a "declared but not wired" claim
    with pytest.raises(ValueError, match=r"unknown metric_set 'classification'; valid: \['scib'\]"):
        mtb.available_datasets("vertical", metric_set="classification", result_path=result_dir)
    with pytest.raises(ValueError, match="unknown metric_set 'scibb'"):
        mtb.load_results("vertical", metric_set="scibb", result_path=result_dir)


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
    assert list(long.columns) == ["metric", "value", "method", "dataset", "category",
                                  "clustering", "source"]


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
    assert d52[d52.source == "rerun"].method.nunique() == 8
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
    scored = r.grand_score.dropna()
    assert scored.is_monotonic_decreasing
    # scored rows precede the NaN (unscored) tail
    assert r.grand_score.notna().tolist() == sorted(r.grand_score.notna().tolist(), reverse=True)
    sb = r[r.method == "scBridge"].iloc[0]
    assert sb.needs_labels is True or sb.needs_labels == True   # noqa: E712
    assert sb.runtime_tier in {"fast", "medium", "slow", "very_slow", "unknown"}


def test_recommend_drops_singleton_datasets_and_warns(result_dir):
    from multibench.data.results import recommend
    with pytest.warns(UserWarning, match="fewer than 2 methods") as rec:
        r = recommend("cross", result_path=result_dir)
    # the shipped cross tree has ONE rankable method with a metric table in
    # every dataset but D53 (six methods, four of them wired for cross). D57
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
    assert "D52" in msg and "D57" in msg and "D56" not in msg and "1.0 by construction" in msg
    # ONE warning, one line per finding, the dropped-datasets line first
    assert len([w for w in rec if issubclass(w.category, UserWarning)
                and "recommend(" in str(w.message)]) == 1
    assert msg.splitlines()[1].strip().startswith("- dropped")
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
    with pytest.raises(ValueError, match=r"unknown family 'bogus' \(given as task=\)"):
        recommend("vertical", long_df=long, task="bogus")
    with pytest.raises(ValueError, match=r"unknown family 'bogus' \(given as family=\)"):
        recommend("vertical", long_df=long, family="bogus")


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
    assert set(nan_rows.method) == {"Concerto", "Matilda", "Seurat_WNN", "UINMF",
                                    "VIMCCA", "scMDC", "totalVI"}
    assert (nan_rows.n_datasets == 0).all() and (nan_rows.coverage == 0.0).all()
    assert (nan_rows.n_datasets_total == r.n_datasets_total.iloc[0]).all()
    assert nan_rows.env.notna().all() and nan_rows.runtime_tier.notna().all()
    # column list unchanged; scored rows first
    assert list(r.columns) == ["method", "grand_score", "n_datasets", "n_datasets_total",
                               "coverage", "needs_labels", "runtime_tier", "worst_sec",
                               "env", "output_kind"]
    assert r.grand_score.iloc[: len(r) - len(nan_rows)].notna().all()
    assert r.attrs["family"] == "clustering" and r.attrs["task"] == "clustering"
    assert r.attrs["source"] == "published"
    assert r.attrs["not_scored"] == r.attrs["missing"] == sorted(nan_rows.method, key=str.lower)
    import re
    assert re.search(r"no rows in source='published' for: .*totalVI", msg)
    assert 'try source="rerun"' in msg and "listed with grand_score NaN" in msg
    # the old load-bearing phrases survive
    assert "incomplete" in msg and "partial coverage" in msg


def test_recommend_rerun_missing_only_seurat_wnn(result_dir):
    r, msg = _rec("vertical", modalities=["rna", "adt"], source="rerun",
                  result_path=result_dir)
    assert set(r[r.grand_score.isna()].method) == {"Seurat_WNN"}
    assert r.attrs["source"] == "rerun" and r.attrs["not_scored"] == ["Seurat_WNN"]
    assert "no rows in source='rerun' for: Seurat_WNN" in msg
    assert 'try source="rerun"' not in msg


def test_recommend_long_df_records_source_and_family(result_dir):
    long = mtb.load_results("vertical", dataset="D11", source="rerun", result_path=result_dir)
    r, _ = _rec("vertical", long_df=long, family="clustering")
    assert r.attrs["source"] == "long_df" and r.attrs["family"] == "clustering"
    r2, _ = _rec("vertical", long_df=long, metrics=["ARI", "NMI"])
    assert r2.attrs["family"] is None and r2.attrs["task"] is None


def test_recommend_cross_skips_registration_methods(result_dir):
    """The registration (coords-output) methods are never rows of the table,
    but they are no longer silently absent: the warning names them with the
    reason and attrs lists them (re-test round 3, spatial user)."""
    r, msg = _rec("cross", result_path=result_dir)
    assert not ({"PASTE", "PASTE2", "SPIRAL", "GPSA"} & set(r.method))
    assert r.attrs["unranked_registration"] == ["GPSA", "PASTE", "PASTE2", "SPIRAL"]
    assert ("registration methods (coords output: GPSA, PASTE, PASTE2, SPIRAL) produce "
            "aligned coordinates, not an embedding - no scIB metric applies") in msg
    assert "are not ranked" in msg
    # a category without registration methods has neither the line nor the ids
    r2, msg2 = _rec("vertical", result_path=result_dir)
    assert r2.attrs["unranked_registration"] == [] and "registration methods" not in msg2


def test_recommend_scores_only_methods_the_registry_lists_for_the_category(result_dir):
    """recommend('cross') used to rank MOFA2 and Multigrate (rows in the
    published cross table) although list_methods('cross') does not list them
    - and their rows shaped every other method's within-dataset rank."""
    r, msg = _rec("cross", result_path=result_dir)
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
    long = mtb.load_results("cross", dataset="D53", result_path=result_dir)
    mine = mtb.to_long(pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"]),
                       "MyMethod", "D53", "cross")
    r2, msg2 = _rec("cross", long_df=pd.concat([long, mine]))
    assert "MyMethod" in set(r2.method) and "MOFA2" not in set(r2.method)
    assert "also scored in the long_df frame but not run by this package for cross: MOFA2, Multigrate" in msg2
    # a category where every stored method is listed: nothing dropped, no line
    r3, msg3 = _rec("vertical", result_path=result_dir)
    assert r3.attrs["dropped_methods"] == [] and "also scored" not in msg3
    # nothing rankable left -> ValueError naming the culprits
    with pytest.raises(ValueError, match=r"every row in long_df belongs to a method this package does not run for cross \(MOFA2"):
        mtb.recommend("cross", long_df=long[long.method == "MOFA2"])


def test_recommend_methods_keyword(result_dir):
    """methods= for parity with load_results / scan / run_all (the instructor
    reached for it and got a TypeError)."""
    from multibench.data.results import recommend
    r, msg = _rec("cross", methods=["scmdc", "sciPENN", "scMoMaT", "paste"], result_path=result_dir)
    assert r.method.tolist() == ["sciPENN", "scMDC", "scMoMaT"]        # alias/case tolerant
    assert r.attrs["not_scored"] == ["scMoMaT"]                        # restricted to the request
    assert r.attrs["unranked_registration"] == ["PASTE"]
    assert "registration methods (coords output: PASTE)" in msg
    assert "StabMap" not in msg and "totalVI" not in msg
    # a requested method without rows is still listed as unscored
    r2, msg2 = _rec("cross", methods=["sciPENN", "scMDC", "totalVI"], result_path=result_dir)
    assert r2.method.tolist() == ["sciPENN", "scMDC", "totalVI"]
    assert "no rows in source='published' for: totalVI" in msg2
    with pytest.raises(KeyError, match=r"unknown method 'Matlida'; did you mean 'Matilda'\?"):
        recommend("cross", methods=["Matlida"], result_path=result_dir)
    with pytest.raises(ValueError, match=r"none of methods=\['totalVI'\] has rows in source='published' for cross"):
        recommend("cross", methods=["totalVI"], result_path=result_dir)
    # positional order and the old keywords are untouched
    import inspect
    params = list(inspect.signature(recommend).parameters)
    assert params[0] == "category" and params[-1] == "methods"
    assert inspect.signature(recommend).parameters["methods"].kind is inspect.Parameter.KEYWORD_ONLY


def test_recommend_task_error_names_real_metrics(result_dir):
    from multibench.data.results import recommend
    # vertical ships no batch metrics; the error must list what IS there,
    # not "metrics present: []" (the loader used to pre-filter by task)
    with pytest.raises(ValueError, match=r"metrics present: \['ARI'"):
        recommend("vertical", family="batch", result_path=result_dir)
    with pytest.raises(ValueError, match=r"metrics present: \['ARI'"):
        recommend("vertical", task="batch", result_path=result_dir)


def test_recommend_family_batch_on_diagonal_rerun(result_dir):
    r, _ = _rec("diagonal", family="batch", source="rerun", result_path=result_dir)
    assert r.attrs["family"] == "batch" and r.grand_score.notna().any()
    # family wins over the task alias when both are given
    r2, _ = _rec("diagonal", task="clustering", family="batch", source="rerun",
                 result_path=result_dir)
    assert r2.attrs["family"] == "batch"
    pd.testing.assert_frame_equal(r, r2)
