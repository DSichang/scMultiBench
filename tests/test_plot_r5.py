"""Plot and evaluate fixes after the fifth student study (fix round 5).

R5-08: a bubble figure of few metrics clipped its Rank legend (and, with one
metric, its Score ramp) at the right edge of the axes.
R5-11: the plot warnings, the CLI overlay line and the label-order error read
as plain sentences; the na='raise' error no longer describes a figure it does
not draw.
R5-14: the evaluate Notes say how LISI turns per-cell scores into one number.
"""
import importlib
import warnings

import matplotlib
import pandas as pd
import pytest

import multibench as mtb
# the table behind a bubble figure: mtb.plot.build_table is deprecated in 0.4.0
from multibench.plot.bubble import _build_table  # noqa: E402
from multibench import config

matplotlib.use("Agg")

B = importlib.import_module("multibench.plot.bubble")
P = importlib.import_module("multibench.eval.pipeline")

QUICKSTART = ["ARI", "NMI", "cLISI"]


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


# --- R5-08: the legends stay inside the axes and the figure ------------------

# --- R5-11 (a): the several-datasets warning -----------------------------------

# --- R5-11 (b): one method, equal columns ------------------------------------

def _rows(method, values):
    return [{"method": method, "metric": m, "value": v}
            for m, v in zip(("ARI", "NMI", "cLISI"), values)]


def test_one_method_warning_is_plain():
    _, msgs = _messages(_build_table,
                        pd.DataFrame(_rows("PCA_LSI_stack", (0.3, 0.4, 0.9))))
    assert msgs == [
        "Only one method, PCA_LSI_stack, is in this figure. With nothing to rank it "
        "against, the fills are grey. Plot it with methods scored on the same dataset."]


def test_equal_column_warning_is_plain():
    df = pd.DataFrame(_rows("A", (0.3, 0.4, 1.0)) + _rows("B", (0.5, 0.2, 1.0)))
    _, msgs = _messages(_build_table, df)
    # 0.4.0: the warning says how to leave the column out
    assert msgs == ["cLISI is 1.000 for every method. That column is grey. Leave it out "
                    "with metrics=."]
    df = pd.DataFrame(_rows("A", (0.3, 0.4, 1.0)) + _rows("B", (0.3, 0.2, 1.0)))
    _, msgs = _messages(_build_table, df)
    # columns in figure order, catalog.metrics()'s since 0.4.0 (X87): ARI first
    assert msgs == ["ARI is 0.300 and cLISI is 1.000 for every method. Those "
                    "columns are grey. Leave them out with metrics=."]


# --- R5-11 (c): the CLI overlay line -----------------------------------------

# --- R5-11 (d): na='raise' does not describe the figure it does not draw -------

# --- R5-11 (e): the label-order error names a method only when one was used ---

def _cty(tmp_path):
    p1, p2 = tmp_path / "cty1.csv", tmp_path / "cty2.csv"
    pd.DataFrame({"x": ["a", "b"] * 5}).to_csv(p1, index=False)
    pd.DataFrame({"x": ["a", "b"] * 5}).to_csv(p2, index=False)
    return str(p1), str(p2)


def test_reordered_plain_dict_error_names_only_the_default_order(tmp_path):
    p1, p2 = _cty(tmp_path)
    with pytest.raises(ValueError) as e:
        P._labels_from_dict({"cty2": p2, "cty1": p1})
    assert str(e.value) == (
        "The label keys cty2 and cty1 are not in the default order. Pass a list of "
        "paths in cell order, or dataset=, category= and method=. The default order is cty1, "
        "cty2, ... by number, with rna before adt before atac. It is not "
        "alphabetical.")


def test_reordered_labels_for_dict_names_the_method_order(tmp_path):
    from multibench.engine.resolve import LabelFiles
    p1, p2 = _cty(tmp_path)
    # labels_for with a method whose cell order puts cty2 first
    d = LabelFiles({"cty2": p2, "cty1": p1})
    assert P._labels_from_dict(d) == [p2, p1]
    d["cty2"] = d.pop("cty2")                     # now cty1, cty2: the default order
    assert P._labels_from_dict(d) == [p1, p2]
    three = LabelFiles({"cty2": p2, "cty1": p1, "cty3": p1})
    three["cty2"] = three.pop("cty2")             # cty1, cty3, cty2: neither order
    with pytest.raises(ValueError) as e:
        P._labels_from_dict(three)
    assert str(e.value).startswith(
        "The label keys cty1, cty3 and cty2 are in neither the method's cell "
        "order nor the default order. Pass a list of paths in cell order, or dataset=, "
        "category= and method=.")
    # labels_for without a method returns the default order: no method to name
    plain = LabelFiles({"cty1": p1, "cty2": p2})
    plain["cty1"] = plain.pop("cty1")
    with pytest.raises(ValueError) as e:
        P._labels_from_dict(plain)
    assert "method's cell order" not in str(e.value)
    assert "are not in the default order" in str(e.value)


# --- R5-14: the LISI median and scaling ---------------------------------------

def test_lisi_formula_matches_scib():
    """The documented formula is what scib computes from the per-cell scores."""
    scib_lisi = pytest.importorskip("scib.metrics.lisi")
    import inspect
    src = inspect.getsource(scib_lisi)
    assert "np.nanmedian" in src
    assert "(ilisi - 1) / (nbatches - 1)" in src
    assert "(nlabs - clisi) / (nlabs - 1)" in src
