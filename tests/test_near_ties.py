"""Floating-point noise must not decide a rank or a bubble's fill (S4-05).

Three methods with iLISI 2.2e-16, 0.0 and 0.0 used to rank 3, 2, 2: one
method got the full-colour bubble and extra Batch Overall for a difference of
1e-16. Ranks and min-max values now treat values that agree to 9 decimals as
ties, and evaluate() records values within 1e-12 of a bound as the bound.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.plot import style


def test_rank_and_minmax_treat_noise_as_a_tie():
    assert style.rank_max(np.array([2.2e-16, 0.0, 0.0])).tolist() == [3.0, 3.0, 3.0]
    assert style.minmax(np.array([2.2e-16, 0.0, 0.0])).tolist() == [1.0, 1.0, 1.0]
    # a real difference still ranks
    assert style.rank_max(np.array([1e-6, 0.0, 0.0])).tolist() == [3.0, 2.0, 2.0]


def test_tidy_value_snaps_bounds():
    from multibench.eval.scib import _tidy_value
    assert _tidy_value("iLISI", 2.2e-16) == 0.0
    assert _tidy_value("ARI", -3e-13) == 0.0
    assert _tidy_value("cLISI", 1.0 + 4e-16) == 1.0
    assert _tidy_value("GC", 1 - 1e-13) == 1.0
    assert _tidy_value("ASW", -0.02) == 0.0              # bounded: clipped
    assert _tidy_value("ASW_batch", 1.2) == 1.0
    assert _tidy_value("ARI", -0.05) == -0.05            # ARI can be negative
    assert _tidy_value("NMI", 0.4321) == 0.4321
    assert np.isnan(_tidy_value("iLISI", float("nan")))


def test_compute_records_lisi_residue_as_zero(monkeypatch):
    pytest.importorskip("scib")
    import scib.metrics as me
    from multibench.eval import scib as escib
    monkeypatch.setattr(escib, "_lisi_helper_problem", lambda: None)
    monkeypatch.setattr(me, "ilisi_graph", lambda *a, **k: 2.2e-16)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 4))
    lab = np.repeat(["a", "b", "c"], 20)
    bat = np.tile(["x", "y"], 30)
    out = escib.compute(emb, lab, None, bat, group="batch", only={"iLISI"})
    assert out.loc["iLISI", "Value"] == 0.0
