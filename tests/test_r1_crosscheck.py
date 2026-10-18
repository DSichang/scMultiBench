"""Round 1 of the 0.4.0 cross-validation: one test per fixed defect and per
deprecation (ids X.. of rounds040/r1/crosscheck/items.json)."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

import multibench as mtb


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec]


def _d11():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results("vertical", dataset="D11")


def _mine(base, name="Mine", scored_with="leidenalg/search/0.4.0"):
    rows = base[base.method == "Matilda"].assign(method=name, source="user")
    if scored_with is not None:
        rows = rows.assign(scored_with=scored_with)
    return rows


# --- stored tables: X03, X04, X42, X59, X69 ----------------------------------

def test_x03_conos_rows_left_the_d28_table():
    d28 = mtb.load_results("diagonal", dataset="D28")
    assert "Conos" not in set(d28.method)
    r, msgs = _messages(mtb.recommend, "diagonal")
    assert pd.isna(r.set_index("method").loc["Conos", "grand_score"])
    assert not any("ARI below" in m for m in msgs)


def test_x04_one_stored_table_per_category():
    cov = mtb.results_coverage()
    assert cov.groupby("category").dataset.nunique().to_dict() == {
        "cross": 1, "diagonal": 1, "mosaic": 1, "vertical": 1}
    assert not cov.dataset.str.endswith("s").any()
    r = mtb.recommend("diagonal")
    assert set(r.loc[r.grand_score.notna(), "n_datasets"]) == {1}


def test_x04_bar_of_one_dataset_has_no_whiskers():
    fig = mtb.plot.bar(mtb.load_results("diagonal"), overall="mean_overall")
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "overall — D28"
    assert ax.get_xlabel() == "overall score (D28)"
    from matplotlib.collections import LineCollection
    assert not [c for c in ax.collections if isinstance(c, LineCollection)
                and len(c.get_segments())]


def test_x42_results_coverage_has_three_columns():
    cov = mtb.results_coverage("cross")
    assert list(cov.columns) == ["category", "dataset", "method"]
    assert cov.attrs["rerun_version"] == "0.4.0"


def test_x59_available_datasets_lists_the_stored_datasets():
    with pytest.warns(FutureWarning, match="available_datasets is deprecated"):
        got = mtb.available_datasets()
    assert got == sorted(mtb.results_coverage().dataset.unique())


def test_x69_recommend_cross_names_uinmf_two_batches():
    _, msgs = _messages(mtb.recommend, "cross")
    assert any("UINMF's stored D52 scores cover batches 1-2 of 3" in m for m in msgs)
    _, msgs = _messages(mtb.recommend, "vertical")
    assert not any("batches 1-2" in m for m in msgs)


# --- recommend: X05, X14, X89 ------------------------------------------------

def test_x05_recommend_docstring_names_the_min_max_scaling():
    doc = " ".join(mtb.recommend.__doc__.split())
    assert ("each per-dataset score is the mean rank over the metrics, min-max scaled "
            "within the dataset (1 = best, 0 = worst there)") in doc


@pytest.mark.parametrize("fn", ["recommend", "bar"])
def test_x14_rows_repeating_stored_rows_are_refused(fn):
    d28 = mtb.load_results("diagonal", dataset="D28")
    both = pd.concat([d28, d28[d28.method == "iNMF"].assign(source="user")])
    call = (lambda: mtb.recommend("diagonal", long_df=both)) if fn == "recommend" \
        else (lambda: mtb.plot.bar(both))
    with pytest.raises(ValueError, match=r"Your rows and the stored table both have iNMF "
                                         r"on D28\. Give your rows another name, e\.g\. "
                                         r"rows\.assign\(method=\"iNMF_rerun\"\)\."):
        call()


def test_x16_the_rename_advice_works_on_a_long_frame():
    d11 = _d11()
    rows = d11[d11.method == "scMM"].assign(source="user")
    with pytest.raises(ValueError, match=r'rows\.assign\(method="scMM_rerun"\)') as e:
        mtb.plot.build_table(pd.concat([d11, rows]))
    assert "to_long(method" not in str(e.value)
    # the advice runs as written
    fixed = rows.assign(method="scMM_rerun")
    mtb.plot.build_table(pd.concat([d11, fixed]))


def test_x89_batch_family_on_vertical_says_why():
    with pytest.raises(ValueError, match=r'^The vertical tables have no batch scores '
                                         r'\(D11 has one batch\)\. Use metrics="clustering"\.$'):
        mtb.recommend("vertical", metrics="batch")


# --- figures: X07, X86, X87, X88, X94 ----------------------------------------

def test_x07_a_family_only_your_row_has_does_not_lift_it():
    d11 = _d11()
    mine = _mine(d11)
    cls = pd.DataFrame({"metric": ["OCA", "F1"], "value": [0.9, 0.8], "method": "Mine",
                        "dataset": "D11", "category": "vertical", "clustering": "default",
                        "source": "user", "scored_with": "leidenalg/search/0.4.0"})
    plain, _ = _messages(mtb.plot.build_table, pd.concat([d11, mine]))
    tbl, msgs = _messages(mtb.plot.build_table, pd.concat([d11, mine, cls]))
    assert tbl.overall["Mine"] == pytest.approx(plain.overall["Mine"])
    assert "Classification" in [b.label for b in tbl.blocks]          # still drawn
    assert any("Classification has a value for one method only, so it does not count "
               "in the row Overall." in m for m in msgs), msgs
    # bar: the lone metrics are left out of the ranks, with one line
    _, msgs = _messages(mtb.plot.bar, pd.concat([d11, mine, cls]))
    assert "F1 and OCA have a value for one method only, so they do not count in " \
           "the Overall." in msgs


def test_x86_language_chips_and_summary_scale_have_a_legend():
    d11 = _d11()
    fig, _ = _messages(mtb.plot.bubble, pd.concat([d11, _mine(d11)]))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "Py / R: the method's language; ?: not a package method" in texts
    fig, _ = _messages(mtb.plot.bubble, d11)
    assert "Py / R: the method's language" in [t.get_text() for t in fig.axes[0].texts]
    two = pd.concat([d11, d11.assign(dataset="D11b")])
    fig, _ = _messages(mtb.plot.bubble, two, aggregate="summary")
    assert "(mean rank over datasets, scaled per column)" in \
        [t.get_text() for t in fig.axes[0].texts]


def test_x87_clustering_columns_follow_catalog_metrics():
    tbl = mtb.plot.build_table(_d11())
    order = mtb.catalog.metrics().query("family == 'clustering'").metric.tolist()
    assert tbl.blocks[0].raw.columns.tolist() == order


def test_x88_bubble_table_repr_is_short():
    text = repr(mtb.plot.build_table(_d11()))
    assert len(text.splitlines()) == 1 and text.startswith("BubbleTable: 13 methods (")
    assert "families DR and clustering (6)" in text and "datasets D11" in text


@pytest.mark.parametrize("scored_with,expect", [
    ("leidenalg/sweep/0.3.9", "were scored before multibench 0.4.0"),
    (None, "do not record how they were scored"),
    ("leidenalg/search/0.4.0", None),
    ("leidenalg/user/0.4.0", None),
    ("unknown", None)])
@pytest.mark.parametrize("fn", [mtb.plot.build_table, mtb.plot.bar])
def test_x94_rows_scored_before_040_warn_next_to_stored_rows(fn, scored_with, expect):
    d11 = _d11()
    _, msgs = _messages(fn, pd.concat([d11, _mine(d11, scored_with=scored_with)]))
    hits = [m for m in msgs if "multibench 0.4.0" in m]
    if expect is None:
        assert not hits, msgs
    else:
        assert len(hits) == 1 and expect in hits[0] and hits[0].startswith(
            "The rows for Mine "), msgs


def test_x94_recommend_warns_for_old_rows_too():
    d11 = _d11()
    _, msgs = _messages(mtb.recommend, "vertical",
                        long_df=pd.concat([d11, _mine(d11, scored_with="igraph/sweep/0.3.1")]))
    assert any("The rows for Mine were scored before multibench 0.4.0." in m for m in msgs)


# --- deprecations and catalog: X39, X40, X41, X43 ----------------------------

def test_x39_no_tasks_column_or_key():
    assert "tasks" not in mtb.catalog.methods().columns
    assert "tasks" not in mtb.method_info("Matilda", verbose=True)


def test_x40_method_info_default_keys_and_the_verbose_hint():
    info = mtb.method_info("Matilda")
    assert set(info) == {"id", "language", "notes", "categories", "env", "atac",
                         "needs_labels", "supports", "runtime", "gpu", "requires_gpu",
                         "cpu_params", "setup_hint", "repo_url", "reference"}
    with pytest.raises(KeyError, match=r"'params' is in method_info\('Matilda', "
                                       r"verbose=True\)"):
        info["params"]
    full = mtb.method_info("Matilda", verbose=True)
    assert {"params", "scripts_url", "notes_long", "upstream_knobs"} <= set(full)


def test_x41_build_table_na_is_deprecated():
    with pytest.warns(FutureWarning, match=r"^plot\.build_table\(na=\) is deprecated and "
                                           r"will be removed in 0\.5; leave it out"):
        mtb.plot.build_table(_d11(), na="skip")
    import inspect
    assert "na" not in inspect.signature(mtb.plot.build_table).parameters


@pytest.mark.parametrize("fn", ["methods", "datasets", "metrics"])
def test_x43_catalog_files_dir_is_deprecated(fn):
    from multibench import config
    f = getattr(mtb.catalog, fn)
    import inspect
    assert "files_dir" not in inspect.signature(f).parameters
    want = f()
    with pytest.warns(FutureWarning, match=rf"^catalog\.{fn}\(files_dir=\) is deprecated "
                                           r"and will be removed in 0\.5\. Use "
                                           r"config\.DEFAULT\.files_path\.$"):
        by_kw = f(files_dir=config.DEFAULT.files_path)
    with pytest.warns(FutureWarning):
        by_pos = f(config.DEFAULT.files_path)               # 0.3 took it by position
    pd.testing.assert_frame_equal(by_kw, want)
    pd.testing.assert_frame_equal(by_pos, want)
