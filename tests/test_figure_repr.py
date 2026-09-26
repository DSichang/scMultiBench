"""A bubble figure shows itself in a notebook without %matplotlib inline.

It is built without pyplot (so it is never shown twice); before 0.3.3 a
notebook then printed only '<Figure size ...>'.
"""
import multibench as mtb


def test_bubble_figure_has_a_png_repr():
    long = mtb.load_results("vertical", dataset="D11", source="rerun")
    fig = mtb.plot.bubble(long)
    png = fig._repr_png_()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
