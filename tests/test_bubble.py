import importlib

import pandas as pd

# the module: mtb.plot.bubble is the function, without the 0.2 attribute paths.
# The table tests call its internal _build_table: the public build_table is
# deprecated in 0.4.0 (mtb.rank(category, long_df=...)) and forwards to it.
bubble = importlib.import_module("multibench.plot.bubble")


def _matrix(tbl, what):
    """The ``what`` frames (``raw`` or ``norm``) of every block, in figure order."""
    return pd.concat([getattr(b, what) for b in tbl.blocks], axis=1)


def _ranks(raw):
    """The per-column max-ranks of a method x metric frame (``n`` = best)."""
    return raw.apply(lambda col: bubble.style.rank_max(col.to_numpy()), axis=0)


def _toy_long():
    return pd.DataFrame(
        {
            "method": ["A", "A", "B", "B"],
            "metric": ["ARI", "NMI", "ARI", "NMI"],
            "value": [0.2, 0.4, 0.8, 0.6],
            "dataset": ["D1"] * 4,
            "category": ["vertical"] * 4,
        }
    )


def test_build_table_pivots_and_adds_overall():
    tbl = bubble._build_table(_toy_long(), metrics=["ARI", "NMI"])
    norm = _matrix(tbl, "norm")
    assert set(norm.columns) == {"ARI", "NMI"}
    assert set(norm.index) == {"A", "B"}
    # B dominates both metrics -> higher overall
    assert tbl.blocks[0].overall["B"] > tbl.blocks[0].overall["A"]
    # rows sorted by overall descending by default
    assert list(norm.index) == tbl.methods == ["B", "A"]


def test_build_table_respects_method_filter():
    tbl = bubble._build_table(_toy_long(), metrics=["ARI"], methods=["A"])
    assert list(_matrix(tbl, "norm").index) == ["A"]


def test_render_returns_figure_with_circles(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.patches import Circle

    fig = bubble.render(bubble._build_table(_toy_long(), metrics=["ARI", "NMI"]))
    ax = fig.axes[0]
    circles = [p for p in ax.patches if isinstance(p, Circle)]
    # Shiny knit-table contract: metric markers are circles whose RADIUS varies
    # with the within-column rank (2 methods x 2 metrics, plus the fixed-size
    # legend circles and any language chips).
    from matplotlib.patches import Rectangle
    metric_circles = [c for c in circles if c.get_zorder() == 3
                      and abs(c.center[1] % 1 - 0.5) < 1e-6 and c.center[0] > 0]
    assert len(metric_circles) == 4
    radii = sorted({round(c.radius, 4) for c in metric_circles})
    assert len(radii) > 1, "circle radius must encode rank, not be constant"
    # per family Overall: a horizontal bar per method whose WIDTH varies
    bars = [p for p in ax.patches if isinstance(p, Rectangle)
            and p.get_zorder() == 3 and abs(p.get_height() - 0.76) < 1e-6]
    assert len(bars) == 2, "one horizontal Overall bar per method"
    widths = sorted({round(b.get_width(), 4) for b in bars})
    assert len(widths) == 2, "Overall bar length must encode the ranked score"
    labels = [t.get_text() for t in ax.texts]
    assert "Score" in labels and "Rank" in labels, "scIB legends present"
    out = tmp_path / "fig.pdf"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_bubble_save(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    out = tmp_path / "b.png"
    fig = bubble.bubble(_toy_long(), metrics=["ARI", "NMI"], save=out)
    assert out.exists()


def test_top_level_plot_namespace(tmp_path):
    import matplotlib; matplotlib.use("Agg")
    import multibench as mtb
    assert hasattr(mtb.plot, "bubble")
    out = tmp_path / "n.png"
    mtb.plot.bubble(_toy_long(), metrics=["ARI", "NMI"], save=out)
    assert out.exists()


# ---------------------------------------------------------------------------
# plain-function API, strict selectors, metrics order, honesty warnings
# ---------------------------------------------------------------------------
import inspect
import pydoc
import warnings

import pytest


def _three(datasets=("D1",)):
    rows = []
    for ds in datasets:
        for i, m in enumerate(["A", "B", "C"]):
            for k, met in enumerate(["ARI", "NMI", "iF1"]):
                rows.append({"method": m, "metric": met, "value": 0.1 * (i + 1) + 0.05 * k,
                             "dataset": ds, "category": "vertical"})
    return pd.DataFrame(rows)


def test_bubble_is_plain_function_with_signature_and_docstring():
    import multibench as mtb
    assert inspect.isfunction(mtb.plot.bubble)
    params = list(inspect.signature(mtb.plot.bubble).parameters)
    # 0.4.0: order=, aggregate=, require_complete= and overall= are deprecated
    assert params == ["long_df", "metrics", "methods", "title", "save"]
    assert "Parameters" in mtb.plot.bubble.__doc__
    txt = pydoc.render_doc(mtb.plot.bubble)
    assert "bubble(" in txt.replace("\b", "") and "long_df" in txt


def test_metrics_order_is_honoured():
    import matplotlib; matplotlib.use("Agg")
    tbl = bubble._build_table(_three(), metrics=["NMI", "ARI", "iF1"])
    assert list(_matrix(tbl, "raw").columns) == ["NMI", "ARI", "iF1"]
    fig = bubble.render(tbl)
    labels = [t.get_text() for t in fig.axes[0].texts if t.get_text() in {"NMI", "ARI", "iF1"}]
    assert labels == ["NMI", "ARI", "iF1"]
    # family block order still paper order: clustering block before batch block
    long = _three()
    extra = long[long.metric == "ARI"].assign(metric="GC")
    tbl2 = bubble._build_table(pd.concat([long, extra]), metrics=["GC", "ARI"])
    assert list(_matrix(tbl2, "raw").columns) == ["ARI", "GC"]


def test_unknown_metric_raises_with_available():
    with pytest.raises(ValueError, match=r"^F1 is not in the table\. ") as e:
        bubble._build_table(_three(), metrics=["ARI", "NMI", "F1"])
    assert "The table has ARI, NMI and iF1." in str(e.value)


def test_unknown_method_lists_the_table():
    long = _three().replace({"method": {"A": "scJoint"}})
    tbl = bubble._build_table(long, methods=["scjoint"])      # case-insensitive: ok
    assert tbl.methods == ["scJoint"]
    with pytest.raises(ValueError) as e:
        bubble._build_table(long, methods=["scjiont"])
    assert str(e.value) == "scjiont is not in the table. The table has B, C and scJoint."


def test_metric_alias_resolves():
    tbl = bubble._build_table(_three(), metrics=["ari", "nmi"])
    assert list(_matrix(tbl, "raw").columns) == ["ARI", "NMI"]
    # a name given twice counts once
    tbl = bubble._build_table(_three(), metrics=["ARI", "ari"])
    assert list(_matrix(tbl, "raw").columns) == ["ARI"]


def test_duplicate_rows_raise():
    long = _three()
    with pytest.raises(ValueError, match="rows repeat the same method, metric and dataset"):
        bubble._build_table(pd.concat([long, long]))
    nods = long.drop(columns=["dataset"])
    with pytest.raises(ValueError, match="rows repeat the same method and metric"):
        bubble._build_table(pd.concat([nods, nods]))
    # a plain frame without dataset still works
    assert bubble._build_table(nods).methods == ["C", "B", "A"]


def test_nan_cell_drawn_as_dash():
    import matplotlib; matplotlib.use("Agg")
    long = _three()
    long = long[~((long.method == "B") & (long.metric == "NMI"))]
    fig = bubble.render(bubble._build_table(long))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts.count(bubble.NA_MARK) == 1


def test_chips_unknown_and_known(monkeypatch):
    import matplotlib; matplotlib.use("Agg")
    from matplotlib.patches import Circle
    # unknown (non-registry) methods get a '?' chip
    fig = bubble.render(bubble._build_table(_three()))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts.count("?") == 3 and "L" not in texts
    # a registry method gets its language chip, and no badge for using labels
    from multibench.engine import registry

    class _Spec:
        language = "python"; needs_labels = True; variants = []
    monkeypatch.setattr(registry, "get", lambda name: _Spec())
    monkeypatch.setattr(registry, "method_of", lambda name: name)   # c4: the chip's lookup
    fig = bubble.render(bubble._build_table(_three()))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "L" not in texts and texts.count("Py") == 3
    badges = [p for p in fig.axes[0].patches if isinstance(p, Circle) and p.center[0] < 0]
    assert len(badges) == 3


# --- P10: the per-column ranks and min-max values ---------------------------
def test_table_ranks_are_max_ranks():
    tbl = bubble._build_table(_three())
    n = len(tbl.methods)
    ranks = _ranks(_matrix(tbl, "raw"))
    for c in ranks.columns:
        assert ranks[c].max() == n and ranks[c].min() == 1   # n = best
    # the legend's 1 = best is the row position
    assert tbl.methods.index(ranks["ARI"].idxmax()) + 1 == 1


def test_table_norm_is_matrix():
    tbl = bubble._build_table(_three())
    assert _matrix(tbl, "norm").index.tolist() == tbl.methods


# --- the figure carries its legends and nothing under them -------------------

# --- P19: the missing-metric Overall rule is said out loud ------------------

def _with_gap():
    long = _three()
    return long[~((long.method == "B") & (long.metric == "NMI"))]


def test_na_warn_default_names_method_and_rule():
    with pytest.warns(UserWarning) as rec:
        tbl = bubble._build_table(_with_gap())
    assert [str(w.message) for w in rec if "has no NMI" in str(w.message)] == [
        "B has no NMI. Pass metrics= without NMI to compare on the same metrics."]
    # the arithmetic the message describes: B's Overall is the mean over ARI, cLISI ranks
    b = tbl.blocks[0]
    assert b.overall["B"] == pytest.approx(
        bubble.style.minmax(_ranks(b.raw).mean(axis=1).to_numpy())[tbl.methods.index("B")])


def test_circles_of_values_tied_by_rounding_have_one_size():
    # the ranks count 0.0 and 2.2e-16 as a tie; the circle sizes must agree
    import importlib
    from matplotlib.patches import Circle
    B = importlib.import_module("multibench.plot.bubble")
    rows = [dict(metric=k, value=v, method=m, dataset="D1")
            for m, ari, il in (("A", .5, 0.0), ("B", .6, 2.2e-16), ("C", .7, 4.4e-16))
            for k, v in (("ARI", ari), ("iLISI", il))]
    tbl = B._build_table(pd.DataFrame(rows))
    x_ilisi = max(c.center[0] for c in B.render(tbl).axes[0].patches if isinstance(c, Circle))
    radii = {round(c.radius, 6) for c in B.render(tbl).axes[0].patches
             if isinstance(c, Circle) and c.center[0] == x_ilisi}
    assert len(radii) == 1


def test_python_bubble_titles_its_one_category_and_dataset():
    # X86 (r3): the command line's default title
    import warnings
    import multibench as mtb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        long = mtb.load_results("vertical", dataset="D11")
        assert mtb.plot.bubble(long).axes[0].get_title() == "vertical D11"
        assert mtb.plot.bubble(long, title="").axes[0].get_title() == ""


def test_a_table_without_values_says_so():
    with pytest.raises(ValueError, match=r"^The table has no values\.$"):
        bubble._build_table(_three().assign(value=float("nan")))
