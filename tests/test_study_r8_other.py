"""Fixes after the eighth student study (fix round 8, package 'other').

R8-03: a family of one metric in a bubble figure had a header wider than its
coloured pill. The pill now holds its header, and two widened pills keep a
gap; figures whose headers already fit are drawn as before.
R8-07: the last messages in the old style (grey columns, the one-method
table, the label-order errors, a failed scripts fetch) are short capitalised
sentences, and the fetch error fits run's FAIL line whole.
R8-08: the help of ``multibench scan`` and ``multibench convert`` is short
sentences without ';', nested parentheses or slash lists.
"""
import importlib
import re
import subprocess
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench.eval import pipeline as P

matplotlib.use("Agg")

B = importlib.import_module("multibench.plot.bubble")
R = importlib.import_module("multibench._data.results")
W = importlib.import_module("multibench.workflow")

HEADERS = {label for label, _, _ in B.FAMILIES} | {"Other"}


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def _d52():
    return _quiet(mtb.load_results, "cross", dataset="D52")


# ============================================================ R8-03 pills
def _pills_and_headers(fig):
    from matplotlib.patches import FancyBboxPatch
    fig.draw_without_rendering()
    ax = fig.axes[0]
    pills = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    heads = [t for t in ax.texts if t.get_text() in HEADERS]
    assert len(pills) == len(heads) >= 1
    return ax, pills, heads


@pytest.mark.parametrize("metrics", [["ARI", "iLISI"], ["cLISI"],
                                     ["ARI", "NMI", "ASW_batch"]])
def test_each_family_header_lies_inside_its_pill(metrics):
    fig = _quiet(mtb.plot.bubble, _d52(), metrics=metrics)
    _, pills, heads = _pills_and_headers(fig)
    pt = fig.dpi / 72.0
    for pill, head in zip(pills, heads):
        p, t = pill.get_window_extent(), head.get_window_extent()
        assert p.x0 <= t.x0 - 2.5 * pt and t.x1 + 2.5 * pt <= p.x1, (
            metrics, head.get_text(), (t.x0, t.x1), (p.x0, p.x1))
        assert p.y0 <= t.y0 and t.y1 <= p.y1, (metrics, head.get_text())
    boxes = sorted((p.get_window_extent() for p in pills), key=lambda b: b.x0)
    for a, b in zip(boxes, boxes[1:]):
        assert a.x1 < b.x0, (metrics, a, b)


@pytest.mark.parametrize("n_rows,title", [(1, None), (1, "D52"), (2, "D52")])
def test_widened_pills_of_a_small_figure_keep_a_gap(n_rows, title):
    df = _d52()
    methods = sorted(df["method"].unique())[:n_rows]
    fig = _quiet(mtb.plot.bubble, df, metrics=["ARI", "iLISI"], methods=methods,
                 title=title)
    _, pills, heads = _pills_and_headers(fig)
    pt = fig.dpi / 72.0
    for pill, head in zip(pills, heads):
        p, t = pill.get_window_extent(), head.get_window_extent()
        assert p.x0 <= t.x0 and t.x1 <= p.x1, (n_rows, head.get_text())
    a, b = (p.get_window_extent() for p in pills)
    assert b.x0 - a.x1 >= 1.9 * pt, (n_rows, title, a.x1, b.x0)
    # the widened pills stay inside the figure
    assert a.x0 >= fig.bbox.x0 and b.x1 <= fig.bbox.x1


def test_pills_that_hold_their_header_keep_their_width():
    fig = _quiet(mtb.plot.bubble, _d52(), metrics=["ARI", "NMI", "iLISI", "GC"])
    _, pills, _ = _pills_and_headers(fig)
    # [Overall 1.5][2 metrics x 1.1] minus 0.35: the width before round 8
    assert [round(p.get_width(), 6) for p in pills] == [3.35, 3.35]
    xs = [round(p.get_x(), 6) for p in pills]
    assert xs == [0.05, 3.7 + 0.5 + 0.05]


# ============================================================ R8-07 text
def _plain(msg, *code, lead=None):
    """A capitalised sentence text: no ', so', no ';' and no '[' outside code.

    ``lead`` is a code name the text may start with instead of a capital.
    """
    body = msg
    for c in code:
        body = body.replace(c, "")
    if lead:
        assert body.startswith(lead), msg
    else:
        assert body[:1].isupper(), msg
    assert ", so" not in body and ";" not in body and "[" not in body, msg
    return msg


# ---- restored from before the 0.4 cut: kept behaviour
def _warned(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec]


def _two(values):
    """Two methods on one dataset with the given (cLISI, iLISI) per method."""
    rows = []
    for m, (cl, il, ari) in zip(("A", "B"), values):
        rows += [{"method": m, "dataset": "D1", "metric": k, "value": v}
                 for k, v in (("cLISI", cl), ("iLISI", il), ("ARI", ari))]
    return pd.DataFrame(rows)


