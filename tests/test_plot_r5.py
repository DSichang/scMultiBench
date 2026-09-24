"""Plot and evaluate fixes after the fifth student study (fix round 5).

R5-08: a bubble figure of few metrics clipped its Rank legend (and, with one
metric, its Score ramp) at the right edge of the axes.
R5-14: the evaluate Notes say how LISI turns per-cell scores into one number.
"""
import importlib
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config

matplotlib.use("Agg")

B = importlib.import_module("multibench.plot.bubble")
P = importlib.import_module("multibench.eval.pipeline")

QUICKSTART = ["ARI", "NMI", "cLISI"]


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


def _stored(dataset, source="published", category="diagonal"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results(category, dataset=dataset, source=source)


def _bubble(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.plot.bubble(df, **kw)


# --- R5-08: the legends stay inside the axes and the figure ------------------

def _legend_outside(fig):
    """Legend artists whose drawn extent leaves the axes or the figure."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    ax_box, fig_box = ax.get_window_extent(renderer), fig.bbox
    legend = [a for a in list(ax.patches) + list(ax.texts) if a.get_gid() == "legend"]
    assert legend, "no artist carries gid='legend'"
    bad = []
    for a in legend:
        bb = a.get_window_extent(renderer)
        for box, where in ((ax_box, "axes"), (fig_box, "figure")):
            if (bb.x0 < box.x0 - 0.5 or bb.x1 > box.x1 + 0.5
                    or bb.y0 < box.y0 - 0.5 or bb.y1 > box.y1 + 0.5):
                bad.append((type(a).__name__, getattr(a, "get_text", lambda: "")(), where))
    return bad


def test_quickstart_figure_keeps_its_rank_legend_whole():
    df = _stored("D28")
    fig = _bubble(df, metrics=QUICKSTART)
    assert _legend_outside(fig) == []
    # all five Rank circles are drawn, the largest (rank 1) last
    circles = [p for p in fig.axes[0].patches if p.get_gid() == "legend"
               and type(p).__name__ == "Circle"]
    assert len(circles) == 5


@pytest.mark.parametrize("kw", [
    {"metrics": ["ARI"]},                                   # one metric
    {"metrics": QUICKSTART, "methods": ["Conos", "GLUE"]},  # two rows
    {"metrics": ["ARI"], "methods": ["Conos"]},               # one row, one metric
    {"metrics": ["ARI", "iLISI"]},                           # two families, one metric each
    {"metrics": QUICKSTART, "show_language": False},
])
def test_narrow_figures_keep_their_legends_inside(kw):
    fig = _bubble(_stored("D28"), **kw)
    assert _legend_outside(fig) == [], kw


def test_narrow_summary_figure_keeps_its_score_legend_inside():
    df = _stored(["D24", "D25", "D28"])
    fig = _bubble(df, metrics=["ARI"], aggregate="summary")
    assert _legend_outside(fig) == []


def _old_width(tbl):
    """The figure width before R5-08: 0.6 * table width + 2.8 inches."""
    total_w = sum(1.5 + 1.1 * b.raw.shape[1] + 0.5 for b in tbl.blocks) - 0.5
    return 0.6 * total_w + 2.8


@pytest.mark.parametrize("metrics", [
    ["cLISI", "ARI", "ASW", "iASW", "NMI"],            # five, one family
    ["ARI", "NMI", "cLISI", "ASW", "iLISI"],           # five, two families
    None,                                              # every metric
])
def test_figures_of_five_or_more_metrics_keep_their_size(metrics):
    df = _stored("D28")
    tbl = B.build_table(df, metrics=metrics, na="skip")
    fig = B.render(tbl)
    w, h = fig.get_size_inches()
    assert w == pytest.approx(_old_width(tbl))
    assert h == pytest.approx(0.42 * len(tbl.methods) + 2.9)
    assert _legend_outside(fig) == []


# --- R5-14: the LISI median and scaling are written down ----------------------

def test_evaluate_notes_state_the_lisi_median_and_scaling():
    doc = mtb.evaluate.__doc__
    sentence = ("cLISI and iLISI take the median m of the per-cell scores and scale "
                "it: cLISI = (L - m)/(L - 1), iLISI = (m - 1)/(B - 1), with L cell "
                "types and B batches.")
    assert sentence in " ".join(doc.split())
    # after the scib-call table, before the re-run paragraph
    flat = " ".join(doc.split())
    assert flat.index("computed only when named in metrics=[...]") < flat.index(sentence)
    assert flat.index(sentence) < flat.index("The re-run tables were scored")


def test_lisi_formula_matches_scib():
    """The documented formula is what scib computes from the per-cell scores."""
    scib_lisi = pytest.importorskip("scib.metrics.lisi")
    import inspect
    src = inspect.getsource(scib_lisi)
    assert "np.nanmedian" in src
    assert "(ilisi - 1) / (nbatches - 1)" in src
    assert "(nlabs - clisi) / (nlabs - 1)" in src
