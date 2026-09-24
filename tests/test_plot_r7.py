"""Plot fixes after the seventh student study (fix round 7).

R7-04: the chip key under a bubble figure called a renamed re-run such as
``SCALEX_rerun`` "not a package method". The key now says what is true for
every ``?`` row, and the footnote lines and row labels of a narrow figure fit
inside the figure at its own size.
R7-12: plain wording for ``order`` (build_table, bubble) and for a one-entry
label dict (evaluate Notes).
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

NEW_KEY = ("Py / R = language · L = uses cell-type labels · "
           "? = a name the package does not know · DR = dimension reduction")
UNKNOWN = "a name the package does not know, such as your own or a renamed re-run"


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def _d28():
    return _quiet(mtb.load_results, "diagonal", dataset="D28", source="published")


def _rerun(name, **extra):
    """A re-run scored in Python and renamed, as in Quickstart Step 5."""
    wide = pd.DataFrame({"Value": [0.31, 0.42, 0.88]}, index=["ARI", "NMI", "cLISI"])
    long = _quiet(mtb.to_long, wide, method=name, dataset="D28", category="diagonal")
    for k, v in extra.items():
        long[k] = v
    return long


def _texts(fig):
    return [t.get_text() for t in fig.axes[0].texts]


def _chips(fig):
    """Row label -> the chip / badge letters drawn on that row."""
    ax = fig.axes[0]
    rows = {round(float(y), 3): lab.get_text()
            for y, lab in zip(ax.get_yticks(), ax.get_yticklabels())}
    out = {}
    for t in ax.texts:
        x, y = t.get_position()
        if t.get_text() in ("Py", "R", "?", "L") and round(y, 3) in rows:
            out.setdefault(rows[round(y, 3)], []).append(t.get_text())
    return out


# --- R7-04: the chip key is true for a renamed re-run -------------------------

def test_chip_key_wording():
    assert B.CHIP_KEY == NEW_KEY


def test_renamed_rerun_footer_does_not_say_not_a_package_method():
    df = pd.concat([_d28(), _rerun("SCALEX_rerun")], ignore_index=True)
    fig = _quiet(mtb.plot.bubble, df, metrics=["ARI", "NMI", "cLISI"])
    texts = _texts(fig)
    assert not any("not a package method" in t for t in texts), texts
    assert NEW_KEY in texts
    # the chip still follows the row name: the renamed row keeps its '?'
    assert _chips(fig)["SCALEX_rerun"] == ["?"]
    assert _chips(fig)["SCALEX"] == ["Py"]


def test_renamed_supervised_rerun_shows_question_mark_and_badge_under_a_true_key():
    df = pd.concat([_d28(), _rerun("Seurat_v3_rerun", needs_labels=True)],
                   ignore_index=True)
    fig = _quiet(mtb.plot.bubble, df, metrics=["ARI", "NMI", "cLISI"])
    assert _chips(fig)["Seurat_v3_rerun"] == ["?", "L"]
    assert not any("not a package method" in t for t in _texts(fig))


def test_docstrings_use_the_key_wording():
    for fn in (mtb.plot.bubble, mtb.to_long):
        doc = " ".join(inspect.getdoc(fn).split())
        assert UNKNOWN in doc, fn.__name__
        assert "not a package method" not in doc, fn.__name__
    assert "sweep variant" not in inspect.getdoc(mtb.plot.bubble)
    assert "add a boolean ``needs_labels`` column" in inspect.getdoc(mtb.to_long)


# --- R7-04: the footnote lines and row labels fit the narrow figures ----------

def _outside(fig, pad=0.5):
    """Texts of the axes (footnotes, row labels, title) that leave the figure."""
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
    {"metrics": ["ARI"], "methods": ["Conos"]},                     # one method, one metric
    {"metrics": ["ARI"], "methods": ["Conos"], "title": "D28"},
    {"metrics": ["ARI", "NMI"]},                                    # two metrics
    {"metrics": ["ARI", "NMI"], "methods": ["Conos"]},
    {"metrics": ["ARI", "NMI", "cLISI"]},                           # the Quickstart figure
    {"metrics": ["ARI"], "show_language": False},
])
def test_narrow_figures_keep_footnotes_and_row_labels_inside(kw):
    fig = _quiet(mtb.plot.bubble, _d28(), **kw)
    assert _outside(fig) == [], kw


def test_renamed_rerun_alone_keeps_its_footnotes_inside():
    fig = _quiet(mtb.plot.bubble, _rerun("SCALEX_rerun"))
    assert _outside(fig) == []


def test_narrow_summary_figure_keeps_its_footnotes_inside():
    df = pd.concat([_quiet(mtb.load_results, "diagonal", dataset=d, source="published")
                    for d in ("D24", "D25", "D28")], ignore_index=True)
    fig = _quiet(mtb.plot.bubble, df, metrics=["ARI"], aggregate="summary")
    assert _outside(fig) == []


def test_fit_widens_only_and_keeps_the_height():
    tbl = _quiet(B.build_table, _d28(), metrics=["ARI"], methods=["Conos"])
    fig = _quiet(B.render, tbl)
    w, h = fig.get_size_inches()
    assert h == pytest.approx(0.42 * 1 + 2.9)
    # 0.6 inch per x-unit past the table, plus 2.8: the width before the fit
    assert w > 0.6 * (4.82 - 0.45) + 2.8


def test_wide_figure_is_not_widened():
    tbl = _quiet(B.build_table, _d28(), na="skip")
    fig = _quiet(B.render, tbl)
    total_w = sum(1.5 + 1.1 * b.raw.shape[1] + 0.5 for b in tbl.blocks) - 0.5
    assert fig.get_size_inches()[0] == pytest.approx(0.6 * total_w + 2.8)


# --- R7-12: plain wording -------------------------------------------------------

def test_order_parameter_reads_as_sentences():
    for fn in (mtb.plot.bubble, mtb.plot.build_table):
        doc = " ".join(inspect.getdoc(fn).split())
        assert ("Methods to put first, in this order. The rest follow, best first. "
                "To drop methods, use ``methods``.") in doc, fn.__name__
        assert "Reorders only" not in doc


def test_one_entry_label_dict_note():
    doc = " ".join(inspect.getdoc(mtb.evaluate).split())
    assert ("``label_order=list(d)`` trusts the dict's own order. A one-entry dict "
            "needs no ``label_order``.") in doc
    assert "order to get wrong" not in doc
