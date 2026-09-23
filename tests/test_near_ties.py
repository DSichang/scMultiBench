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


def _noise_frame():
    rows = []
    for m, il in (("A", 2.2e-16), ("B", 0.0), ("C", 0.0)):
        rows += [{"method": m, "metric": "iLISI", "value": il},
                 {"method": m, "metric": "GC", "value": 0.5},
                 {"method": m, "metric": "ASW_batch", "value": 0.5}]
    return pd.DataFrame(rows)


def test_rank_and_minmax_treat_noise_as_a_tie():
    assert style.rank_max(np.array([2.2e-16, 0.0, 0.0])).tolist() == [3.0, 3.0, 3.0]
    assert style.minmax(np.array([2.2e-16, 0.0, 0.0])).tolist() == [1.0, 1.0, 1.0]
    # a real difference still ranks
    assert style.rank_max(np.array([1e-6, 0.0, 0.0])).tolist() == [3.0, 2.0, 2.0]


def test_build_table_near_tie_ranks_are_equal():
    tbl = mtb.plot.build_table(_noise_frame())
    assert tbl.ranks["iLISI"].nunique() == 1
    assert tbl.norm["iLISI"].tolist() == [1.0, 1.0, 1.0]
    assert tbl.overall.nunique() == 1


def _old_rank_max(x):
    return pd.Series(np.asarray(x, dtype=float)).rank(method="max").to_numpy()


def _old_minmax(x):
    x = np.asarray(x, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or lo == hi:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def _stored_frames():
    """Every stored table: each (source, dataset) alone, and each category's
    datasets together."""
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cat in mtb.list_categories():
            for source in ("published", "rerun"):
                try:
                    df = mtb.load_results(cat, source=source)
                except (FileNotFoundError, ValueError):
                    continue
                if df.empty:
                    continue
                for ds in sorted(df["dataset"].unique()):
                    out.append((f"{source}/{ds}", "dataset", df[df["dataset"] == ds]))
                if df["dataset"].nunique() > 1:
                    out.append((f"{source}/{cat}", "summary", df))
    return out


def _tables(df, aggregate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.plot.build_table(df, aggregate=aggregate, na="skip")


def test_stored_tables_rank_exactly_as_without_the_tolerance(monkeypatch):
    frames = _stored_frames()
    names = {n for n, _, _ in frames}
    # the demo tables the package ships must all be covered
    for want in ("rerun/D11", "rerun/D11s", "rerun/D28", "rerun/D28s", "rerun/D45",
                 "rerun/D45s", "rerun/D52", "rerun/D52s", "published/D11",
                 "published/D24", "published/D25", "published/D28", "published/D52"):
        assert want in names, sorted(names)
    new = {n: _tables(df, agg) for n, agg, df in frames}
    new_bar = {n: mtb.plot.bar(df).axes[0].get_yticklabels() for n, agg, df in frames
               if agg == "summary"}
    new_bar = {n: [t.get_text() for t in v] for n, v in new_bar.items()}
    monkeypatch.setattr(style, "rank_max", _old_rank_max)
    monkeypatch.setattr(style, "minmax", _old_minmax)
    for n, agg, df in frames:
        old = _tables(df, agg)
        assert new[n].methods == old.methods, n
        pd.testing.assert_frame_equal(new[n].ranks, old.ranks, obj=n)
        pd.testing.assert_frame_equal(new[n].norm, old.norm, obj=n, atol=1e-8)
        pd.testing.assert_series_equal(new[n].overall, old.overall, obj=n, atol=1e-8)
    import matplotlib.pyplot as plt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n, labels in new_bar.items():
            df = next(d for m, a, d in frames if m == n)
            old = [t.get_text() for t in mtb.plot.bar(df).axes[0].get_yticklabels()]
            assert labels == old, n
    plt.close("all")


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
