"""Rows from a new dataset next to the stored tables must not plot silently.

Five students added their own dataset's rows to a stored table (S1-02,
S2-07, S3-06, S4-07, S5-09). bubble(aggregate="summary") ranked the new rows
last by construction and plot.bar ranked a lone baseline first with Overall
1.0, without a warning. Both now warn about an incomplete method x dataset
matrix, a dataset that holds one method, and datasets that share no method.
"""
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb

matplotlib.use("Agg")


def _rows(method, dataset, base=0.5):
    return [{"metric": m, "value": base + 0.01 * k, "method": method,
             "dataset": dataset, "category": "mosaic"}
            for k, m in enumerate(("ARI", "NMI", "ASW", "cLISI"))]


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


def _stored_plus_lone_baseline():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stored = mtb.load_results("mosaic", source="rerun")
    return pd.concat([stored, pd.DataFrame(_rows("PCA_standin", "LABMOS"))],
                     ignore_index=True)


def test_bar_warns_about_a_lone_method_and_the_incomplete_matrix():
    fig, msgs = _messages(mtb.plot.bar, _stored_plus_lone_baseline())
    lone = [m for m in msgs if m.startswith("dataset LABMOS has only one method")]
    assert lone == ["dataset LABMOS has only one method (PCA_standin), so its Overall "
                    "there is always 1.0. Plot it with methods scored on the same "
                    "dataset."]
    inc = [m for m in msgs if m.startswith("The summary ranks ")]
    assert len(inc) == 1 and "PCA_standin has scores on 1 of them." in inc[0]
    assert "Filter long_df to the methods scored on every dataset" in inc[0]


def test_build_table_summary_warns_about_the_lone_method_too():
    tbl, msgs = _messages(mtb.plot.build_table, _stored_plus_lone_baseline(),
                          aggregate="summary", overall="mean_overall")
    assert tbl.methods[0] == "PCA_standin"            # the reason for the warning
    assert any(m.startswith("dataset LABMOS has only one method (PCA_standin), so "
                            "its Overall there is always 1.0") for m in msgs)
    _, msgs = _messages(mtb.plot.build_table, _stored_plus_lone_baseline(),
                        aggregate="summary")          # overall="rank"
    assert any(m.startswith("dataset LABMOS has only one method (PCA_standin), so "
                            "its rank there is always the lowest")
               for m in msgs)


def test_datasets_sharing_no_method_get_the_stronger_warning():
    df = pd.DataFrame(_rows("A", "D1") + _rows("B", "D1", 0.6)
                      + _rows("X", "MINE") + _rows("Y", "MINE", 0.4))
    for fn, kw in ((mtb.plot.bar, {}),
                   (mtb.plot.build_table, {"aggregate": "summary"}),
                   (mtb.plot.bubble, {"aggregate": "summary"})):
        _, msgs = _messages(fn, df, **kw)
        strong = [m for m in msgs if m.startswith("rows come from")]
        assert strong == ["rows come from 2 datasets (D1, MINE) that share no method, "
                          "so the figure ranks unrelated rows against each other. Plot "
                          "each dataset on its own, or score the same methods on every "
                          "dataset. If these datasets hold the same cells, give their "
                          "rows one dataset name first."], fn
        # it replaces the incomplete-matrix message, which says less
        assert not any(m.startswith("The summary ranks ") for m in msgs), fn


def test_a_complete_frame_plots_without_these_warnings():
    # B leads on both datasets: with A ahead on D2, every mean rank would tie
    # and the summary would warn that the columns compare nothing
    df = pd.DataFrame(_rows("A", "D1") + _rows("B", "D1", 0.6)
                      + _rows("A", "D2", 0.3) + _rows("B", "D2", 0.4))
    for fn, kw in ((mtb.plot.bar, {}), (mtb.plot.build_table, {"aggregate": "summary"})):
        _, msgs = _messages(fn, df, **kw)
        assert msgs == [], (fn, msgs)


def test_one_dataset_frame_needs_no_cross_dataset_warning():
    # a single method on a single dataset: nothing is ranked against anything
    _, msgs = _messages(mtb.plot.bar, pd.DataFrame(_rows("A", "D1")))
    assert msgs == []
    _, msgs = _messages(mtb.plot.bar, pd.DataFrame(_rows("A", "D1") + _rows("B", "D1", 0.6)))
    assert msgs == []


def test_require_complete_drops_the_incomplete_method_without_a_coverage_warning():
    df = pd.DataFrame(_rows("A", "D1") + _rows("B", "D1", 0.6) + _rows("C", "D1", 0.7)
                      + _rows("A", "D2", 0.3) + _rows("B", "D2", 0.4))
    tbl, msgs = _messages(mtb.plot.build_table, df, aggregate="summary",
                          require_complete=True)
    assert set(tbl.methods) == {"A", "B"}
    assert [m for m in msgs if not m.startswith("require_complete=True dropped")] == []


@pytest.mark.parametrize("fn", [mtb.plot.bar, mtb.plot.build_table])
def test_docstrings_list_the_warnings(fn):
    doc = " ".join(fn.__doc__.split())
    assert "Warns" in fn.__doc__
    assert "a dataset holds one method, or no method spans two datasets" in doc.lower()
