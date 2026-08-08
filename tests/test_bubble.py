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
    # paper layout: circles are METRIC markers only - 2 methods x 2 metrics.
    # the per-family Overall is a SQUARE, and the top-3 markers carry rank labels.
    assert len(circles) == 4
    from matplotlib.patches import Rectangle
    squares = [p for p in ax.patches
               if isinstance(p, Rectangle) and abs(p.get_width() - 0.64) < 1e-6]
    assert len(squares) == 2, "one Overall square per method per family"
    labels = [t.get_text() for t in ax.texts]
    assert "1" in labels and "2" in labels, "top ranks must be annotated"
    assert any(t == "Overall" for t in labels), "family Overall chip present"
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
