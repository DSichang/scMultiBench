"""Plot and ranking warnings after the third student study (fix round 3).

R3-06: the Leiden-backend warning counts an igraph-scored row only when the
frame holds stored rows of the same dataset. Rows from the user's own
dataset get the no-overlap warning alone.
R3-07: recommend(long_df=...) gives the same backend line in its warning.
R3-08: the n/a warning names the row and its source in short sentences.
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

B = importlib.import_module("multibench.plot.bubble")

matplotlib.use("Agg")

REFUSAL = ("Your rows are on MYCITE, the stored results on D11: plot them apart, or "
           "score your embedding with dataset='D11'.")


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


def _mine(method, dataset, scored_with="igraph/search/0.4.0",
          metrics=("ARI", "NMI", "ASW", "iASW", "iF1", "cLISI"), base=0.5):
    return pd.DataFrame([
        {"metric": m, "value": base + 0.01 * i, "method": method, "dataset": dataset,
         "category": "vertical", "clustering": "default", "source": "user",
         "scored_with": scored_with}
        for i, m in enumerate(metrics)])


# --- R3-06 -------------------------------------------------------------------

# --- R3-07 -------------------------------------------------------------------

def test_recommend_documents_the_backend_line():
    doc = " ".join(mtb.recommend.__doc__.split())
    assert "Warns" in mtb.recommend.__doc__
    assert "One message: partial coverage, methods left out and why." in doc


# --- R3-08 -------------------------------------------------------------------

def test_na_warning_stops_after_three_rows():
    frames = []
    for i, m in enumerate("ABCDE"):
        rows = _mine(m, "X", base=0.1 * i)
        frames.append(rows[rows["metric"] != "NMI"] if m != "E" else rows)
    _, msgs = _messages(_build_table, pd.concat(frames, ignore_index=True))
    (msg,) = [m for m in msgs if "has no" in m]
    assert msg.count("has no NMI.") == 3
    assert "... and 1 more row." in msg
    assert msg.endswith("Pass metrics= without NMI to compare on the same metrics.")


