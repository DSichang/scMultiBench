"""Plot and evaluate fixes after the fifth student study (fix round 5).

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
