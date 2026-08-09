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


def test_summary_bars_carry_sd_whiskers():
    """The across-datasets summary draws its error bars ON the bars, not as a
    separate chart: SD whiskers appear at bar tips when a method+metric exists
    in >=2 datasets, and are absent for single-dataset methods."""
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
    assert whiskers, "summary mode must draw SD whiskers on the bars"
    # Expected whiskers (3 segments each: body + 2 caps):
    #   A, B on ARI and NMI (rank spread across the two datasets)  -> 4
    #   A on Overall                                               -> 1
    #   B on Overall has ZERO spread (top-ranked in both datasets,
    #   minmax overall = 1 in each) -> correctly NO whisker; a zero
    #   spread must not draw a fake one.
    #   C is single-dataset -> none anywhere.
    assert len(whiskers) == 5 * 3, (
        f"expected 15 whisker segments, got {len(whiskers)}")
    # and none of them may sit on C's row
    tbl = bubble.build_table(pd.DataFrame(rows), aggregate="summary")
    c_y = len(tbl.methods) - tbl.methods.index("C") - 0.5
    ys = {round(float(l.get_ydata()[0]), 2) for l in whiskers
          if l.get_ydata()[0] == l.get_ydata()[1]}
    assert round(c_y, 2) not in ys, (
        "single-dataset method C grew a whisker from structural zeros")
