"""Plot fixes after the seventh student study (fix round 7).

R7-04: a renamed re-run such as ``SCALEX_rerun`` keeps its ``?`` chip (0.4.0:
the key says "?: not a package method", and ``SCALEX (this run)`` gets SCALEX's
chip); the legends and row labels of a narrow figure fit inside the figure at its
own size.
"""
import importlib
import inspect
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb

matplotlib.use("Agg")

B = importlib.import_module("multibench.plot.bubble")


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def _d28():
    return _quiet(mtb.load_results, "diagonal", dataset="D28")


# --- R7-04: the chip of a renamed re-run --------------------------------------

KEY = "Py / R: the method's language; ?: not a package method"


# --- R7-04: the legends and row labels fit the narrow figures -----------------

def _outside(fig, pad=0.5):
    """Texts of the axes (legends, row labels, title) that leave the figure."""
    fig.draw_without_rendering()
    ax, box = fig.axes[0], fig.bbox
    arts = [t for t in ax.texts if t.get_text()]
    arts += [t for t in ax.get_yticklabels() if t.get_text()]
    if ax.title.get_text():
        arts.append(ax.title)
    bad = []
    for a in arts:
        bb = a.get_window_extent()
        if bb.x0 < box.x0 - pad or bb.x1 > box.x1 + pad:
            bad.append((a.get_text()[:40], round(bb.x0 - box.x0, 1),
                        round(bb.x1 - box.x1, 1)))
    return bad


@pytest.mark.parametrize("kw", [
    {"metrics": ["ARI"], "methods": ["SCALEX"]},                    # one method, one metric
    {"metrics": ["ARI"], "methods": ["SCALEX"], "title": "D28"},
    {"metrics": ["ARI", "NMI"]},                                    # two metrics
    {"metrics": ["ARI", "NMI"], "methods": ["SCALEX"]},
    {"metrics": ["ARI", "NMI", "cLISI"]},                           # the Quickstart figure
])
def test_narrow_figures_keep_legends_and_row_labels_inside(kw):
    fig = _quiet(mtb.plot.bubble, _d28(), **kw)
    assert _outside(fig) == [], kw


def test_wide_figure_is_not_widened():
    tbl = _quiet(B._build_table, _d28())
    fig = _quiet(B.render, tbl)
    total_w = sum(1.5 + 1.1 * b.raw.shape[1] + 0.5 for b in tbl.blocks) - 0.5
    assert fig.get_size_inches()[0] == pytest.approx(0.6 * total_w + 2.8)
