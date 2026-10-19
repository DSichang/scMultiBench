"""The stored results and recommend: what they hold and how they rank."""
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
    cat = mtb.catalog.datasets()
    stored = cat[cat.has_results]
    assert stored.groupby("category").dataset.nunique().to_dict() == {
        "cross": 1, "diagonal": 1, "mosaic": 1, "vertical": 1}
    assert not stored.dataset.str.endswith("s").any()
    # recommend's n_datasets column left in 0.4.0: the stored results hold one dataset
    assert set(mtb.load_results("diagonal").dataset) == {"D28"}


def test_x04_bar_of_one_dataset_has_no_whiskers():
    fig = mtb.plot.bar(mtb.load_results("diagonal"), overall="mean_overall")
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "overall — D28"
    assert ax.get_xlabel() == "overall score, scaled 0-1 (D28)"
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


def test_x69_uinmf_two_batch_rows_left_the_d52_table():
    # owner, round 3: the rows of a run the package no longer makes are removed
    assert "UINMF" not in set(mtb.load_results("cross", dataset="D52").method)
    r, msgs = _messages(mtb.recommend, "cross")
    assert pd.isna(r.set_index("method").loc["UINMF", "grand_score"])
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
    with pytest.raises(ValueError, match=r"Your rows and the stored results both have iNMF "
                                         r"on D28\. Give your rows another name, e\.g\. "
                                         r"rows\.assign\(method=rows\.method \+ "
                                         r"' \(this run\)'\)\."):
        call()


def test_x16_the_rename_advice_works_on_a_long_frame():
    d11 = _d11()
    rows = d11[d11.method == "scMM"].assign(source="user")
    # Y23 (r3): the advice keeps each method its own name
    # 0.4.0: the suffix is " (this run)"; recommend(long_df=) replaces build_table
    with pytest.raises(ValueError, match=r"rows\.assign\(method=rows\.method \+ "
                                         r"' \(this run\)'\)") as e:
        mtb.recommend("vertical", long_df=pd.concat([d11, rows]))
    assert "to_long(method" not in str(e.value)
    # the advice runs as written
    fixed = rows.assign(method=rows.method + " (this run)")
    r, _ = _messages(mtb.recommend, "vertical", long_df=pd.concat([d11, fixed]))
    assert "scMM (this run)" in set(r.method)


def test_x89_batch_family_on_vertical_says_why():
    with pytest.raises(ValueError, match=r'^The vertical tables have no batch scores '
                                         r'\(D11 has one batch\)\. Use metrics="clustering"\.$'):
        mtb.recommend("vertical", metrics="batch")
