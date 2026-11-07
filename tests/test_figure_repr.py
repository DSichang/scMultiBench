"""A bubble figure shows itself in a notebook without %matplotlib inline.

It is built without pyplot (so it is never shown twice); before 0.3.3 a
notebook then printed only '<Figure size ...>'.
"""
import multibench as mtb


def test_bubble_figure_has_a_png_repr():
    long = mtb.load_results("vertical", dataset="D11")
    fig = mtb.plot.bubble(long)
    png = fig._repr_png_()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)


def test_bubble_figure_pickles():
    import pickle
    fig = mtb.plot.bubble(mtb.load_results("vertical", dataset="D11"))
    back = pickle.loads(pickle.dumps(fig))
    assert type(back) is type(fig) and back._repr_png_()[:8] == b"\x89PNG\r\n\x1a\n"


def test_importing_the_package_does_not_import_matplotlib():
    import subprocess
    import sys
    code = "import sys, multibench; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "False"
