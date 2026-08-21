import pandas as pd
from multibench.plot import bubble


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
    tbl = bubble.build_table(_toy_long(), metrics=["ARI", "NMI"])
    assert set(tbl.matrix.columns) == {"ARI", "NMI"}
    assert set(tbl.matrix.index) == {"A", "B"}
    # B dominates both metrics -> higher overall
    assert tbl.overall["B"] > tbl.overall["A"]
    # rows sorted by overall descending by default
    assert list(tbl.matrix.index) == ["B", "A"]


def test_build_table_respects_method_filter():
    tbl = bubble.build_table(_toy_long(), metrics=["ARI"], methods=["A"])
    assert list(tbl.matrix.index) == ["A"]


def test_render_returns_figure_with_circles(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.patches import Circle

    fig = bubble.render(bubble.build_table(_toy_long(), metrics=["ARI", "NMI"]))
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
    fig = bubble.plot_bubble(_toy_long(), metrics=["ARI", "NMI"], save=out)
    assert out.exists()


def test_summary_aggregate_ranks_within_dataset_then_averages():
    long = pd.DataFrame({
        "method": ["A", "B", "A", "B"],
        "metric": ["ARI", "ARI", "ARI", "ARI"],
        "value":  [0.1, 0.9, 0.9, 0.1],   # A wins D2, B wins D1 -> tie on average
        "dataset": ["D1", "D1", "D2", "D2"],
        "category": ["vertical"] * 4,
    })
    tbl = bubble.build_table(long, metrics=["ARI"], aggregate="summary")
    assert abs(tbl.overall["A"] - tbl.overall["B"]) < 1e-9

    # asymmetric case: B wins BOTH datasets -> B must outrank A overall
    long2 = pd.DataFrame({
        "method": ["A", "B", "A", "B"],
        "metric": ["ARI", "ARI", "ARI", "ARI"],
        "value":  [0.1, 0.9, 0.2, 0.8],   # B wins D1 and D2
        "dataset": ["D1", "D1", "D2", "D2"],
        "category": ["vertical"] * 4,
    })
    tbl2 = bubble.build_table(long2, metrics=["ARI"], aggregate="summary")
    assert tbl2.overall["B"] > tbl2.overall["A"]


def test_top_level_plot_namespace(tmp_path):
    import matplotlib; matplotlib.use("Agg")
    import multibench as mtb
    assert hasattr(mtb.plot, "bubble")
    out = tmp_path / "n.png"
    mtb.plot.bubble(_toy_long(), metrics=["ARI", "NMI"], save=out)
    assert out.exists()


def test_summary_bars_carry_no_error_bars():
    """The paper's summary panels carry no error bars - neither do ours."""
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    rows = []
    for ds, bump in (("DS1", 0.0), ("DS2", 0.2)):
        for m, v in (("A", 0.5), ("B", 0.8)):
            for metric in ("ARI", "NMI"):
                rows.append({"method": m, "metric": metric,
                             "value": v + bump, "dataset": ds})
    # method C exists in ONE dataset only -> must get no whisker
    rows += [{"method": "C", "metric": "ARI", "value": 0.3, "dataset": "DS1"},
             {"method": "C", "metric": "NMI", "value": 0.3, "dataset": "DS1"}]
    fig = bubble.plot_bubble(pd.DataFrame(rows), aggregate="summary")
    ax = fig.axes[0]
    whiskers = [l for l in ax.lines if l.get_gid() == "whisker"]
    assert whiskers == [], f"expected no whisker artists, got {len(whiskers)}"


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
    assert params[0] == "long_df"
    for p in ("metrics", "methods", "order", "aggregate", "cmap", "title", "save",
              "show_language", "require_complete", "overall"):
        assert p in params
    assert "Parameters" in mtb.plot.bubble.__doc__
    assert "mean_overall" in mtb.plot.bubble.__doc__      # OVERALL_DOC spliced in
    txt = pydoc.render_doc(mtb.plot.bubble)
    assert "bubble(" in txt.replace("\b", "") and "long_df" in txt


def test_bubble_attribute_aliases(tmp_path):
    import matplotlib; matplotlib.use("Agg")
    import multibench as mtb
    assert mtb.plot.bubble.build_table is mtb.plot.build_table
    assert mtb.plot.bubble.render is mtb.plot.render
    assert mtb.plot.bubble.plot_bubble is mtb.plot.bubble
    assert mtb.plot.plot_bubble is mtb.plot.bubble
    from multibench.plot import bubble as b
    tbl = b.build_table(_toy_long(), metrics=["ARI", "NMI"])
    assert isinstance(tbl, b.BubbleTable)
    b.render(tbl).savefig(tmp_path / "x.png")


def test_metrics_order_is_honoured():
    import matplotlib; matplotlib.use("Agg")
    tbl = bubble.build_table(_three(), metrics=["NMI", "ARI", "iF1"])
    assert list(tbl.raw.columns) == ["NMI", "ARI", "iF1"]
    fig = bubble.render(tbl)
    labels = [t.get_text() for t in fig.axes[0].texts if t.get_text() in {"NMI", "ARI", "iF1"}]
    assert labels == ["NMI", "ARI", "iF1"]
    # family block order still paper order: clustering block before batch block
    long = _three()
    extra = long[long.metric == "ARI"].assign(metric="GC")
    tbl2 = bubble.build_table(pd.concat([long, extra]), metrics=["GC", "ARI"])
    assert list(tbl2.raw.columns) == ["ARI", "GC"]


def test_unknown_metric_raises_with_available():
    with pytest.raises(ValueError, match=r"unknown metric\(s\) \['F1'\]") as e:
        bubble.build_table(_three(), metrics=["ARI", "NMI", "F1"])
    assert "available in this frame" in str(e.value) and "ARI" in str(e.value)


def test_unknown_method_raises_with_suggestion():
    long = _three().replace({"method": {"A": "scJoint"}})
    tbl = bubble.build_table(long, methods=["scjoint"])      # case-insensitive: ok
    assert tbl.methods == ["scJoint"]
    with pytest.raises(ValueError) as e:
        bubble.build_table(long, methods=["scjiont"])
    assert "did you mean" in str(e.value) and "scJoint" in str(e.value)


def test_metric_alias_resolves():
    tbl = bubble.build_table(_three(), metrics=["ari", "nmi"])
    assert list(tbl.raw.columns) == ["ARI", "NMI"]
    with pytest.raises(ValueError, match="more than once"):
        bubble.build_table(_three(), metrics=["ARI", "ari"])


def test_bad_aggregate_raises():
    with pytest.raises(ValueError, match="aggregate must be 'dataset' or 'summary'"):
        bubble.build_table(_three(), aggregate="mean")
    with pytest.raises(ValueError, match="overall must be one of"):
        bubble.build_table(_three(), overall="median")


def test_order_reorders_without_filtering():
    tbl = bubble.build_table(_three(), order=["A"])
    assert tbl.methods == ["A", "C", "B"]          # A pinned first, rest best-first
    assert tbl.overall["C"] > tbl.overall["B"]
    with pytest.raises(ValueError, match=r"unknown method\(s\) \['nope'\]"):
        bubble.build_table(_three(), order=["nope"])


def test_duplicate_rows_raise():
    long = _three()
    with pytest.raises(ValueError, match="duplicate"):
        bubble.build_table(pd.concat([long, long]))
    nods = long.drop(columns=["dataset"])
    with pytest.raises(ValueError, match="duplicate"):
        bubble.build_table(pd.concat([nods, nods]))
    # a plain frame without dataset still works
    assert bubble.build_table(nods).methods == ["C", "B", "A"]


def test_multi_dataset_under_aggregate_dataset_warns():
    import matplotlib; matplotlib.use("Agg")
    two = _three(datasets=("D1", "D2"))
    with pytest.warns(UserWarning, match="summary"):
        tbl = bubble.build_table(two)
    assert tbl.methods == ["C", "B", "A"]             # bare names, unchanged
    assert tbl.datasets == ("D1", "D2")
    fig = bubble.render(tbl)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert all("2 ds" in lab for lab in labels)
    # a method present in ONE of the two datasets is tagged with that dataset
    mixed = pd.concat([two, two[(two.method == "A") & (two.dataset == "D1")].assign(method="X")])
    with pytest.warns(UserWarning):
        fig = bubble.render(bubble.build_table(mixed))
    assert any(lab == "X · D1" for lab in [t.get_text() for t in fig.axes[0].get_yticklabels()])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bubble.build_table(two, aggregate="summary")   # complete matrix: silent


def test_summary_warns_on_incomplete_matrix():
    long = _three(datasets=("DS1", "DS2"))
    long = long[~((long.method == "C") & (long.dataset == "DS2"))]
    with pytest.warns(UserWarning, match="C seen in 1/2"):
        tbl = bubble.build_table(long, aggregate="summary")
    assert set(tbl.methods) == {"A", "B", "C"}
    assert tbl.coverage.to_dict() == {"A": 2, "B": 2, "C": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bubble.build_table(_three(datasets=("DS1", "DS2")), aggregate="summary")


def test_summary_require_complete_restricts_and_raises():
    long = _three(datasets=("DS1", "DS2"))
    long = long[~((long.method == "C") & (long.dataset == "DS2"))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tbl = bubble.build_table(long, aggregate="summary", require_complete=True)
    assert set(tbl.methods) == {"A", "B"}
    only_partial = long[((long.method == "A") & (long.dataset == "DS1"))
                        | ((long.method == "B") & (long.dataset == "DS2"))]
    with pytest.raises(ValueError, match="no method has results on all 2 datasets"):
        bubble.build_table(only_partial, aggregate="summary", require_complete=True)


def _disagreeing():
    # the D11/D11s pattern: the two Overall formulas order B and C differently
    rows = []
    vals = {"D1": {"A": [0.9, 0.9], "B": [0.5, 0.2], "C": [0.4, 0.3]},
            "D2": {"A": [0.9, 0.9], "B": [0.1, 0.8], "C": [0.2, 0.7]},
            "D3": {"A": [0.9, 0.9], "B": [0.3, 0.3], "C": [0.31, 0.29]}}
    for ds, d in vals.items():
        for m, (a, n) in d.items():
            rows += [{"method": m, "metric": "ARI", "value": a, "dataset": ds},
                     {"method": m, "metric": "NMI", "value": n, "dataset": ds}]
    return pd.DataFrame(rows)


def test_overall_basis_shared_with_bar():
    import matplotlib; matplotlib.use("Agg")
    import multibench as mtb
    long = _disagreeing()

    def bar_order(**kw):
        fig = mtb.plot.bar(long, **kw)
        return [t.get_text() for t in fig.axes[0].get_yticklabels()][::-1]
    for basis in ("rank", "mean_overall"):
        bb = bubble.build_table(long, aggregate="summary", overall=basis).methods
        assert bb == bar_order(overall=basis), basis
    # defaults pin today's behaviour: bubble == 'rank', bar == 'mean_overall'
    assert bubble.build_table(long, aggregate="summary").methods == \
        bubble.build_table(long, aggregate="summary", overall="rank").methods
    assert bar_order() == bar_order(overall="mean_overall")
    assert bubble.build_table(long, aggregate="summary").overall_basis == "rank"
    tbl = bubble.build_table(long, aggregate="summary", overall="mean_overall")
    assert tbl.overall_basis == "mean_overall"
    fig = bubble.render(tbl)
    assert any("mean_overall" in t.get_text() for t in fig.axes[0].texts)


def test_nan_cell_drawn_as_dash_with_legend():
    import matplotlib; matplotlib.use("Agg")
    long = _three()
    long = long[~((long.method == "B") & (long.metric == "NMI"))]
    fig = bubble.render(bubble.build_table(long))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts.count(bubble.NA_MARK) == 1
    assert any("n/a" in t for t in texts)
    fig0 = bubble.render(bubble.build_table(_three()))
    assert not any("n/a" in t.get_text() for t in fig0.axes[0].texts)


def test_badges_unknown_and_supervised(monkeypatch):
    import matplotlib; matplotlib.use("Agg")
    from matplotlib.patches import Circle
    # unknown (non-registry) methods get a '?' chip
    fig = bubble.render(bubble.build_table(_three()))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts.count("?") == 3 and "L" not in texts
    # a registry method that needs labels gets an 'L' badge next to the chip
    from multibench.engine import registry

    class _Spec:
        language = "python"; needs_labels = True
    monkeypatch.setattr(registry, "get", lambda name: _Spec())
    fig = bubble.render(bubble.build_table(_three()))
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts.count("L") == 3 and texts.count("Py") == 3
    badges = [p for p in fig.axes[0].patches if isinstance(p, Circle) and p.center[0] < 0]
    assert all(p.center[0] < 0 for p in badges) and len(badges) == 6
