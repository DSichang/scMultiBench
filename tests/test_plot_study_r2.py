"""Plot warnings and drawing after the second student study (fix round 2).

M01: the default aggregate='dataset' figure of a stored table plus rows from
the user's own dataset advised aggregate='summary', which then warned the
opposite. The no-overlap and one-method checks now run in both modes, and the
one-method message has no idiom ('by construction').
M02: a constant column (or a one-method figure) was drawn at the 'High' end
of the colour ramp without a warning. It is now grey, named in the footnote,
and warned about.
M03: igraph-scored rows next to the leidenalg-scored stored tables now warn.
M26: the footnote and the chip key read as plain sentences.
"""
import importlib
import warnings

import matplotlib
import pandas as pd
import pytest
from matplotlib import colors
from matplotlib.patches import Circle, Rectangle

import multibench as mtb
from multibench import config

B = importlib.import_module("multibench.plot.bubble")

matplotlib.use("Agg")



def _grey():
    return colors.to_rgba(B.CONSTANT_FILL)


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


def _stored(dataset, source="rerun", category="vertical"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results(category, dataset=dataset, source=source)


def _mine(method="PCA", dataset="MYCITE", scored_with="igraph/sweep/0.3.1",
          values=(0.351, 0.5, 0.55, 0.9)):
    return pd.DataFrame([
        {"metric": m, "value": v, "method": method, "dataset": dataset,
         "category": "vertical", "clustering": "default", "source": "user",
         "scored_with": scored_with}
        for m, v in zip(("ARI", "NMI", "ASW", "cLISI"), values)])


# --- M01: no summary advice when the datasets share no method ----------------

def test_dataset_mode_own_dataset_gets_the_no_overlap_warning():
    df = pd.concat([_stored("D11"), _mine()], ignore_index=True)
    _, msgs = _messages(mtb.plot.build_table, df)
    assert ("The rows come from 2 datasets, D11 and MYCITE, that share no method. The "
            "figure ranks unrelated rows against each other. Plot each dataset on "
            "its own, or score your method on D11 and add it to that table.") in msgs
    assert ("Dataset MYCITE has only one method, PCA. Its row is ranked against "
            "rows from other datasets. Plot it with methods scored on the same "
            "dataset.") in msgs
    assert not any("aggregate='summary'" in m for m in msgs), msgs


def test_dataset_mode_lone_dataset_next_to_shared_ones_gets_no_summary_advice():
    # S3: a stand-in on LUNG added to the three diagonal tables
    diag = _stored(["D24", "D25", "D28"], source="published", category="diagonal")
    own = _mine(method="PCA_standin", dataset="LUNG", scored_with="leidenalg/sweep/0.3.1")
    _, msgs = _messages(mtb.plot.build_table, pd.concat([diag, own], ignore_index=True))
    assert not any("aggregate='summary'" in m for m in msgs), msgs
    assert any(m.startswith("This figure averages each method over 4 datasets")
               and m.endswith("Plot each dataset on its own.") for m in msgs), msgs
    assert any(m.startswith("Dataset LUNG has only one method, PCA_standin.")
               for m in msgs), msgs


def test_dataset_mode_rerun_pair_still_suggests_the_summary():
    df = pd.concat([_stored("D11"), _stored("D11s")], ignore_index=True)
    _, msgs = _messages(mtb.plot.build_table, df)
    assert msgs == [
        "This figure averages each method over 2 datasets (D11, D11s), so its rows "
        "mix datasets. Pass aggregate='summary' for the rank-averaged summary panel, "
        "or filter to one dataset."]


def test_dataset_mode_advice_uses_the_cli_spelling(monkeypatch):
    monkeypatch.setattr(config, "_CLI", True)
    _, msgs = _messages(mtb.plot.build_table,
                        pd.concat([_stored("D11"), _stored("D11s")], ignore_index=True))
    assert msgs[0].startswith("This figure averages each method over 2 datasets")
    assert "Pass --aggregate summary" in msgs[0] and "aggregate='summary'" not in msgs[0]


def test_no_plot_warning_says_by_construction():
    lone = pd.concat([_stored("D11"), _mine(scored_with="leidenalg/sweep/0.3.1")],
                     ignore_index=True)
    calls = [(mtb.plot.build_table, {}),
             (mtb.plot.build_table, {"aggregate": "summary"}),
             (mtb.plot.build_table, {"aggregate": "summary", "overall": "mean_overall"}),
             (mtb.plot.bar, {}), (mtb.plot.bar, {"overall": "rank"})]
    seen = []
    for fn, kw in calls:
        _, msgs = _messages(fn, lone, **kw)
        seen += msgs
        assert not any("by construction" in m for m in msgs), (fn, kw, msgs)
    assert "Dataset MYCITE has only one method, PCA. Its rank there is always " \
           "the lowest. Plot it with methods scored on the same dataset." in seen
    assert "Dataset MYCITE has only one method, PCA. Its Overall there is " \
           "always 1.0. Plot it with methods scored on the same dataset." in seen


# --- M02: constant columns and one-method figures ---------------------------

def _circles(fig, facecolor):
    return [p for p in fig.axes[0].patches if isinstance(p, Circle)
            and p.get_zorder() == 3 and p.get_facecolor() == facecolor]


def _ilisi_tie():
    rows = []
    for m, asw in (("A", 0.9), ("B", 0.5), ("C", 0.2)):
        rows += [{"method": m, "metric": "iLISI", "value": 0.0},
                 {"method": m, "metric": "ASW_batch", "value": asw},
                 {"method": m, "metric": "GC", "value": asw / 2}]
    return pd.DataFrame(rows)


def test_one_method_figure_warns_and_is_grey():
    _, msgs = _messages(mtb.plot.build_table, _mine())
    assert msgs == ["Only one method, PCA, is in this figure. With nothing to rank "
                    "it against, the fills are grey. Plot it with methods scored on "
                    "the same dataset."]
    fig, msgs = _messages(mtb.plot.bubble, _mine())
    assert len(msgs) == 1
    assert len(_circles(fig, _grey())) == 4                  # every metric marker
    bars = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)
            and p.get_zorder() == 3 and p.get_facecolor() == _grey()]
    assert len(bars) == 1                                 # the Overall bar too
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "Grey fill: one method, nothing to compare." in texts


def test_constant_column_warns_and_uses_the_neutral_fill():
    tbl, msgs = _messages(mtb.plot.build_table, _ilisi_tie())
    assert msgs == ["All methods have the same iLISI (0.000), so that column is grey."]
    assert tbl.norm["iLISI"].tolist() == [1.0, 1.0, 1.0]    # R parity kept
    assert tbl.ranks["iLISI"].tolist() == [3.0, 3.0, 3.0]
    fig = B.render(tbl)
    grey = _circles(fig, _grey())
    assert len(grey) == 3
    ilisi_x = {round(c.center[0], 3) for c in grey}
    assert len(ilisi_x) == 1                              # one column, three rows
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "Grey fill: all rows equal in iLISI (0.000)." in texts


def test_stored_demo_figures_are_unchanged():
    for source, category, datasets in (
            ("rerun", "vertical", ["D11", "D11s"]), ("rerun", "diagonal", ["D28", "D28s"]),
            ("rerun", "mosaic", ["D45", "D45s"]), ("rerun", "cross", ["D52", "D52s"]),
            ("published", "vertical", ["D11"]),
            ("published", "diagonal", ["D24", "D25", "D28"])):
        for ds in datasets:
            df = _stored(ds, source=source, category=category)
            tbl, msgs = _messages(B.build_table, df, na="skip")
            assert not any("is grey" in m or "are grey" in m for m in msgs), (ds, msgs)
            fig = B.render(tbl)
            assert not [p for p in fig.axes[0].patches if p.get_facecolor() == _grey()], ds
            assert not any(t.get_text().startswith("Grey fill")
                           for t in fig.axes[0].texts), ds


def test_bubble_table_notes_document_the_grey_rule():
    doc = " ".join(B.BubbleTable.__doc__.split())
    assert "a constant (or all-NaN) column is all ones, as in the R code" in doc
    assert "The figure draws a column whose rows all hold one value in grey" in doc


# --- M03: igraph rows next to the stored tables -----------------------------

BACKEND = ("The rows for {m} were clustered with the igraph Leiden backend. The stored "
           "tables used leidenalg, which can move ARI by up to about 0.1. Set "
           "mtb.config.DEFAULT.leiden_flavor = 'leidenalg' before evaluate to compare "
           "them.")


@pytest.mark.parametrize("fn", [mtb.plot.build_table, mtb.plot.bubble, mtb.plot.bar])
def test_igraph_rows_next_to_stored_rows_warn(fn):
    df = pd.concat([_stored("D11"), _mine(method="MyMethod", dataset="D11")],
                   ignore_index=True)
    _, msgs = _messages(fn, df)
    assert msgs.count(BACKEND.format(m="MyMethod")) == 1, msgs


@pytest.mark.parametrize("fn", [mtb.plot.build_table, mtb.plot.bar])
def test_no_backend_warning_for_leidenalg_or_own_rows(fn):
    lalg = _mine(method="MyMethod", dataset="D11", scored_with="leidenalg/sweep/0.3.1")
    _, msgs = _messages(fn, pd.concat([_stored("D11"), lalg], ignore_index=True))
    assert not any("igraph Leiden backend" in m for m in msgs), msgs
    own = pd.concat([_mine(method="M1"), _mine(method="M2", values=(0.4, 0.6, 0.5, 0.8))],
                    ignore_index=True)
    _, msgs = _messages(fn, own)
    assert not any("igraph Leiden backend" in m for m in msgs), msgs
    # rows without scored_with (stored rows only) never trigger it
    _, msgs = _messages(fn, _stored("D11"))
    assert not any("igraph Leiden backend" in m for m in msgs), msgs


def test_backend_warning_needs_a_sweep_metric_on_show():
    df = pd.concat([_stored("D11"), _mine(method="MyMethod", dataset="D11")],
                   ignore_index=True)
    _, msgs = _messages(mtb.plot.build_table, df, metrics=["ASW", "cLISI"])
    assert not any("igraph Leiden backend" in m for m in msgs), msgs
    # clusters from the user: only iF1 depends on the backend
    user = _mine(method="MyMethod", dataset="D11", scored_with="igraph/user/0.3.1")
    _, msgs = _messages(mtb.plot.build_table,
                        pd.concat([_stored("D11"), user], ignore_index=True))
    assert not any("igraph Leiden backend" in m for m in msgs), msgs


def test_backend_warning_cli_spelling(monkeypatch):
    monkeypatch.setattr(config, "_CLI", True)
    df = pd.concat([_stored("D11"), _mine(method="MyMethod", dataset="D11")],
                   ignore_index=True)
    _, msgs = _messages(mtb.plot.build_table, df)
    hit = [m for m in msgs if "igraph Leiden backend" in m]
    assert hit and hit[0].endswith("Score them with multibench evaluate "
                                   "--leiden-flavor leidenalg to compare them.")


# --- M26: plain footnote and chip key ---------------------------------------

def test_footnote_and_chip_key_are_plain_sentences():
    assert B.CHIP_KEY == "Py / R = language · L = uses cell-type labels · " \
                         "? = a name the package does not know · " \
                         "DR = dimension reduction"
    fig = B.render(B.build_table(_ilisi_tie()))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert ("Overall: each method's mean rank over the metrics, scaled 0-1. "
            "Rows: best mean Overall first.") in texts
    assert not any("minmax(" in t or "rows ordered by" in t for t in texts)
    two = pd.concat([_ilisi_tie().assign(dataset="D1"),
                     _ilisi_tie().assign(dataset="D2")], ignore_index=True)
    for basis in ("rank", "mean_overall"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = B.render(B.build_table(two, aggregate="summary", overall=basis))
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any(t.startswith(f"Overall (overall='{basis}'): ")
                   and t.endswith("Rows: best mean Overall first.") for t in texts), texts
        assert not any("minmax" in t for t in texts)
