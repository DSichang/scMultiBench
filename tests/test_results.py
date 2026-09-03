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
    assert (rr.source == "rerun").all()
    assert rr.method.nunique() == 12
    assert abs(_val(rr, "iNMF", "ARI") - 0.472126) < 1e-6


def test_source_both_concat(result_dir):
    import matplotlib; matplotlib.use("Agg")
    pub = results.load_results("diagonal", dataset="D28", result_path=result_dir)
    rr = results.load_results("diagonal", dataset="D28", source="rerun", result_path=result_dir)
    both = results.load_results("diagonal", dataset="D28", source="both", result_path=result_dir)
    assert len(both) == len(pub) + len(rr)
    assert list(both.columns) == COLS
    assert set(both.source.unique()) == {"published", "rerun"}
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
    # a KNOWN method with no rows: empty frame + warning that answers the
    # "where are its rows then?" question (results_coverage says rerun has D11)
    with pytest.warns(UserWarning, match="no published rows for method 'Concerto'") as rec:
        df = results.load_results("vertical", dataset="D11", method="Concerto",
                                  result_path=result_dir)
    assert len(df) == 0 and list(df.columns) == COLS
    msg = str(rec[0].message)
    assert "rerun has 1 dataset(s) (D11)" in msg and "pass source='rerun'" in msg
    # a KNOWN metric absent from the tables: still the empty-frame-plus-warning
    with pytest.warns(UserWarning, match="kBET"):
        df = results.load_results("vertical", dataset="D11", metric="kBET",
                                  result_path=result_dir)
    assert len(df) == 0
    # an UNKNOWN metric code is a typo: ValueError listing both vocabularies
    with pytest.raises(ValueError, match=r"unknown metric\(s\) \['ZZZ'\]; valid codes: \['ARI'") as e:
        results.load_results("vertical", dataset="D11", metric="ZZZ", result_path=result_dir)
    assert "present in this frame: ['ARI'" in str(e.value)
    with pytest.raises(ValueError, match=r"unknown metric\(s\) \['NOPE'\]"):
        results.load_results("vertical", dataset="D11", metric=["ARI", "NOPE"],
                             result_path=result_dir)


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
    with pytest.raises(ValueError, match=r"unknown family 'bogus' \(given as task=\); valid: None"):
        results.load_results("diagonal", dataset="D28", task="bogus", result_path=result_dir)
    # family= is the documented name and means the same thing
    pd.testing.assert_frame_equal(
        results.load_results("diagonal", dataset="D28", family="batch", result_path=result_dir), b)
    with pytest.raises(ValueError, match=r"unknown family 'bogus' \(given as family=\)"):
        results.load_results("diagonal", dataset="D28", family="bogus", result_path=result_dir)
    with pytest.raises(ValueError, match="pass either family= or task="):
        results.load_results("diagonal", dataset="D28", task="batch", family="clustering",
                             result_path=result_dir)
    # a list_tasks() token in the family slot is explained, not just rejected
    with pytest.raises(ValueError, match="selects a METRIC FAMILY, not a mtb.list_tasks"):
        results.load_results("diagonal", task="dimension_reduction", result_path=result_dir)
    # the positional slip load_results('vertical', 'D11') names the slot
    with pytest.raises(ValueError, match=r"did you mean dataset='D11'"):
        results.load_results("vertical", "D11", result_path=result_dir)
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
    assert (results.load_results(result_path=f).source == "rerun").all()
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
    assert set(b.source) == {"rerun"}


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


# ---------------------------------------------------------------------------
# P08 / Rin: provenance through CSV, list-filter validation, degenerate rows
# ---------------------------------------------------------------------------
from multibench.data.results import DegenerateRerunWarning


def _quiet_rerun(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegenerateRerunWarning)
        return results.load_results(*a, **k)


def test_concat_to_csv_reload_keeps_provenance(tmp_path, result_dir):
    rr = results.load_results("vertical", dataset="D11", source="rerun", result_path=result_dir)
    mine = mtb.to_long(pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"]),
                       "MyMethod", "D11", "vertical")
    f = tmp_path / "all.csv"
    pd.concat([rr, mine]).to_csv(f, index=False)
    back = results.load_results(result_path=f)            # default keeps every row
    assert len(back) == len(rr) + 2
    assert back.clustering.notna().all() and back.source.notna().all()
    assert set(back.source) == {"rerun", "user"}
    user = results.load_results(result_path=f, source="user")
    assert set(user.method) == {"MyMethod"} and len(user) == 2
    rerun = results.load_results(result_path=f, source="rerun")   # prefix match
    assert set(rerun.source) == {"rerun"} and "MyMethod" not in set(rerun.method)
    assert len(results.load_results(result_path=f, source="both")) == len(back)
    with pytest.raises(ValueError, match=r"source 'bogus' not in .*present: \['rerun', 'user'\]"):
        results.load_results(result_path=f, source="bogus")
    # the directory branch still validates against the fixed vocabulary
    with pytest.raises(ValueError, match="unknown source 'user'"):
        results.load_results("vertical", source="user", result_path=result_dir)


def test_load_long_csv_fills_nan_provenance(tmp_path):
    df = pd.DataFrame({"metric": ["ARI", "NMI", "ARI"], "value": [0.1, 0.2, 0.3],
                       "method": ["A", "A", "B"], "dataset": "D1", "category": "vertical",
                       "clustering": ["default", None, "louvain"],
                       "source": [None, "rerun-0.2.1", None]})
    f = tmp_path / "gap.csv"
    df.to_csv(f, index=False)
    back = results.load_results(result_path=f)
    assert back.clustering.tolist() == ["default", "default", "louvain"]
    assert back.source.tolist() == ["user", "rerun-0.2.1", "user"]
    # dataset filter on a file validates every element
    with pytest.raises(FileNotFoundError, match=r"no rows for dataset 'D9'.*datasets in the file: \['D1'\]"):
        results.load_results(result_path=f, dataset="D9")


def test_dataset_list_validates_every_element(result_dir):
    with pytest.raises(FileNotFoundError, match=r"no published results for diagonal/D99;") as e:
        results.load_results("diagonal", dataset=["D27", "D99"], result_path=result_dir)
    assert "datasets with published tables: ['D24'" in str(e.value)
    with pytest.raises(FileNotFoundError, match=r"no rerun results for diagonal/\['D98', 'D99'\]"):
        _quiet_rerun("diagonal", dataset=["D28", "D98", "D99"], source="rerun",
                     result_path=result_dir)
    # source='both': an id present in ONE source is fine (D27 is published-only)
    both = _quiet_rerun("diagonal", dataset=["D27", "D28"], source="both", result_path=result_dir)
    assert set(both.dataset) == {"D27", "D28"}
    # category=None: the union across categories must cover every id
    df = results.load_results(dataset=["D11", "D28"], result_path=result_dir)
    assert set(df.dataset) == {"D11", "D28"}


def test_method_list_validates_every_element(result_dir):
    with pytest.raises(KeyError, match=r"unknown method 'Matlida'; did you mean 'Matilda'\?"):
        results.load_results("vertical", method="Matlida", result_path=result_dir)
    with pytest.raises(KeyError, match="unknown method 'Nope'"):
        results.load_results("vertical", method=["Matilda", "Nope"], result_path=result_dir)
    # case-folded names resolve (registry id 'totalVI')
    df = _quiet_rerun("vertical", method="totalvi", source="rerun", result_path=result_dir)
    assert set(df.method) == {"totalVI"}
    # a name that exists only in a user file is accepted for that file
    long = mtb.to_long(pd.DataFrame({"Value": [0.5]}, index=["ARI"]), "MyMethod", "D1", "vertical")


def test_user_method_name_accepted_in_file(tmp_path):
    long = mtb.to_long(pd.DataFrame({"Value": [0.5]}, index=["ARI"]), "MyMethod", "D1", "vertical")
    f = tmp_path / "mine.csv"
    long.to_csv(f, index=False)
    assert set(results.load_results(result_path=f, method="mymethod").method) == {"MyMethod"}
    with pytest.raises(KeyError, match="unknown method 'MyMethd'; did you mean 'MyMethod'"):
        results.load_results(result_path=f, method="MyMethd")


def test_degenerate_rerun_rows_are_flagged(result_dir):
    with pytest.warns(DegenerateRerunWarning, match=r"Conos/D28 \(rerun-0.2.1 ARI 0.0004 vs published 0.27\)") as rec:
        rr = results.load_results("diagonal", dataset="D28", source="rerun", result_path=result_dir)
    assert "Conos" in set(rr.method)          # flagged, never dropped silently
    msg = str(rec[0].message)
    assert "df[df.method != 'Conos']" in msg and "ARI < 0.01" in msg
    # published-only loads and a filter that removes the row are silent
    with warnings.catch_warnings():
        warnings.simplefilter("error", DegenerateRerunWarning)
        results.load_results("diagonal", dataset="D28", result_path=result_dir)
        results.load_results("diagonal", dataset="D28", source="rerun", method="iNMF",
                             result_path=result_dir)
        results.load_results("diagonal", dataset="D28s", source="rerun", result_path=result_dir)
    # results_coverage is a WHERE scan and stays quiet
    with warnings.catch_warnings():
        warnings.simplefilter("error", DegenerateRerunWarning)
        mtb.results_coverage("diagonal", result_path=result_dir)
    # recommend inherits the flag (Conos sits in its ranking)
    with pytest.warns(DegenerateRerunWarning):
        mtb.recommend("diagonal", source="rerun", result_path=result_dir)


def test_catalog_datasets_covers_every_dataset_with_results(result_dir):
    cat = mtb.catalog.datasets()
    have = set(mtb.available_datasets(source="both", result_path=result_dir))
    assert have <= set(cat.dataset)
    extra = cat[cat.dataset.isin({"D11s", "D24", "D28s", "D45s", "D52s", "SD7", "SD8", "SD9", "SD10"})]
    assert len(extra) == 9 and extra.has_results.all()
    assert extra.set_index("dataset").loc["D11s", "category"] == "vertical"
    assert extra.set_index("dataset").loc["SD7", "simulated"] == True   # noqa: E712
    # the CSV rows come first, the appended ids after them in natural order
    tail = cat.dataset.tolist()[-9:]
    assert tail == ["D11s", "D24", "D28s", "D45s", "D52s", "SD7", "SD8", "SD9", "SD10"]


# ---------------------------------------------------------------------------
# re-test round 3: plain 'rerun' provenance, one-method tables, fetchable()
# ---------------------------------------------------------------------------
def test_source_column_is_plain_rerun_and_version_in_attrs(result_dir, tmp_path):
    """Two of three re-testers filtered df[df.source == 'rerun'] and got an
    empty frame: the column carried the file stamp 'rerun-0.2.1'. The column
    now holds exactly 'published' | 'rerun' | 'user'; the stamp lives in
    frame.attrs['rerun_version']."""
    rr = _quiet_rerun("vertical", dataset="D11", source="rerun", result_path=result_dir)
    assert set(rr.source) == {"rerun"}
    assert rr.attrs["rerun_version"] == "0.2.1"
    assert len(rr[rr.source == "rerun"]) == len(rr)
    both = _quiet_rerun("cross", dataset="D52", source="both", result_path=result_dir)
    assert set(both.source) == {"published", "rerun"}
    assert both.attrs["rerun_version"] == "0.2.1"
    # published-only frames carry the key too, as None
    pub = results.load_results("diagonal", dataset="D28", result_path=result_dir)
    assert set(pub.source) == {"published"} and pub.attrs["rerun_version"] is None
    # results_coverage reports the same stamp and plain 'rerun' rows
    cov = mtb.results_coverage("cross", result_path=result_dir)
    assert set(cov.source) == {"published", "rerun"} and cov.attrs["rerun_version"] == "0.2.1"
    # a user file: the column keeps whatever it carries, an old stamp is
    # still matched by source='rerun' (prefix) and parsed into attrs
    old = rr.assign(source="rerun-0.1.0")
    f = tmp_path / "old.csv"
    pd.concat([old, mtb.to_long(pd.DataFrame({"Value": [0.5]}, index=["ARI"]),
                                "MyMethod", "D11", "vertical")]).to_csv(f, index=False)
    back = results.load_results(result_path=f, source="rerun")
    assert set(back.source) == {"rerun-0.1.0"} and back.attrs["rerun_version"] == "0.1.0"
    assert set(results.load_results(result_path=f, source="user").method) == {"MyMethod"}
    # a rerun frame without a stamp (plain 'rerun' written by this version)
    rr.to_csv(f, index=False)
    back = results.load_results(result_path=f, source="rerun")
    assert set(back.source) == {"rerun"} and back.attrs["rerun_version"] is None
    # the DegenerateRerunWarning keeps naming the full stamp
    with pytest.warns(DegenerateRerunWarning, match=r"Conos/D28 \(rerun-0.2.1 ARI"):
        results.load_results("diagonal", dataset="D28", source="rerun", result_path=result_dir)


def test_single_method_table_warns_when_other_source_has_more(result_dir):
    """The instructor got a silent 1-method cross/D52 table under the default
    source='published' (the CLI warns, the API did not)."""
    with pytest.warns(UserWarning, match=r"only one method \(scMoMaT\) in the published table "
                                         r"for cross/D52; ranks and Overall bars are not "
                                         r"meaningful with a single method") as rec:
        df = results.load_results("cross", dataset="D52", result_path=result_dir)
    assert set(df.method) == {"scMoMaT"}          # a warning, never an error
    msgs = [str(w.message) for w in rec if "only one method" in str(w.message)]
    assert len(msgs) == 1
    assert "the rerun tables hold 8 methods for it" in msgs[0]
    assert "pass source='rerun' (or 'both')" in msgs[0]
    # source='rerun' / 'both' for the same selection are quiet; so is a
    # published selection with >= 2 methods and a coverage scan
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("ignore", DegenerateRerunWarning)
        results.load_results("cross", dataset="D52", source="rerun", result_path=result_dir)
        results.load_results("cross", dataset="D52", source="both", result_path=result_dir)
        results.load_results("cross", dataset="D53", result_path=result_dir)
        results.load_results("cross", result_path=result_dir)
        mtb.results_coverage("cross", result_path=result_dir)
        # the source the user asked for is the ONLY one with rows: no warning
        # (D54 has a single published method and no re-run sweep)
        results.load_results("cross", dataset="D54", result_path=result_dir)


def test_fetchable_lists_the_release_assets(result_dir):
    from multibench.data.fetch import AVAILABLE      # mtb.data.fetch is the function
    ids = results.fetchable()
    assert isinstance(ids, list) and ids == sorted(AVAILABLE, key=mtb.catalog._dataset_sort_key)
    assert "D11" in ids and "D12" not in ids
    # the two vocabularies overlap but are DIFFERENT (D46 downloads and has no
    # stored results; D12 has published tables and no download) - which is
    # exactly why they are two functions; the docstring says which is which
    have = set(mtb.available_datasets(source="both", result_path=result_dir))
    assert set(ids) & have and set(ids) != have
    assert "D46" in ids and "D46" not in have and "D12" in have and "D12" not in ids
    doc = mtb.available_datasets.__doc__
    assert "STORED RESULTS" in doc and "fetch" in doc and "NOT the datasets that can be downloaded" in doc
