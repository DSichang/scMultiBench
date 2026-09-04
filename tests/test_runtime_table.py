"""engine/runtimes.yaml must never understate a run the package itself recorded.

Five 1-second placeholders (scBridge and the four registration methods) once
reported tier 'fast' while files/final_verification.tsv held 170-3729 s for
the same runs; scan/recommend/run_all(timeout=) all surfaced them. The table
is now the max over its three sources and these invariants pin it.
"""
import glob
import math
from pathlib import Path

import pandas as pd
import pytest

import multibench as mtb
from multibench import discover
from multibench.discover import _runtimes

ROOT = Path(__file__).resolve().parents[1]
TIERS = (("fast", 300), ("medium", 1800), ("slow", 7200))


def _tier(sec):
    for name, bound in TIERS:
        if sec < bound:
            return name
    return "very_slow"


def test_worst_sec_covers_verification_wall_s():
    for m in mtb.list_methods():
        rec = discover.verification_for(m)
        if not rec:
            continue
        wall = max(r["wall_s"] for r in rec if r["wall_s"] is not None)
        hint = mtb.method_info(m)["runtime"]
        assert hint["worst_sec"] is not None, f"{m} is verified but reports tier 'unknown'"
        assert hint["worst_sec"] >= wall, (m, hint["worst_sec"], wall)
        # the exact placeholder failure: a sub-5 s claim against a longer record
        assert not (hint["worst_sec"] < 5 and wall > 5), m


def test_no_placeholder_observations_and_worst_is_max():
    for m, rec in _runtimes().items():
        secs = [o["sec"] for o in rec["observed"]]
        assert secs, m
        assert min(secs) >= 5, (m, secs)
        assert rec["worst_sec"] == max(secs), m
        for o in rec["observed"]:
            assert o["source"] in {"manual", "summary_csv", "verification"}, (m, o)


def test_tier_matches_worst_sec_and_docstring_thresholds():
    import inspect
    for m, rec in _runtimes().items():
        assert rec["tier"] == _tier(rec["worst_sec"]), (m, rec["tier"], rec["worst_sec"])
    doc = inspect.getdoc(mtb.method_info)
    assert "<5 min" in doc and "5-30 min" in doc and "30 min-2 h" in doc


def test_summary_csv_run_sec_is_not_understated():
    files = sorted(glob.glob(str(ROOT / "notebooks" / "results" / "summary_D*.csv")))
    if not files:
        pytest.skip("notebook result tables not in this checkout")
    table = _runtimes()
    checked = 0
    for f in files:
        ds = Path(f).stem.split("_", 1)[1]
        if ds.endswith("s"):            # *s.csv = subsampled runs
            continue
        for _, r in pd.read_csv(f).iterrows():
            if pd.isna(r["run_sec"]) or not str(r["status"]).startswith(("CHAIN_OK", "RUN_OK")):
                continue
            obs = {o["dataset"]: o["sec"] for o in table[r["method"]]["observed"]}
            assert obs[ds] >= math.ceil(float(r["run_sec"])), (r["method"], ds)
            checked += 1
    assert checked > 30


def test_expected_tier_moves_after_regeneration():
    def hint(m):
        return mtb.method_info(m)["runtime"]
    for m in ("scBridge", "PASTE"):
        assert hint(m)["tier"] == "medium", m
    for m in ("PASTE2", "GPSA", "SPIRAL", "iPOLNG", "scMVP"):
        assert hint(m)["tier"] == "slow", m
    assert hint("scBridge")["worst_sec"] == 964     # summary_D28.csv, not the 170 s verification
    # keys of the public answer are unchanged; unmeasured methods stay 'unknown'
    assert set(hint("StabMap")) == {"tier", "worst_sec", "observed"}
    unmeasured = [m for m in mtb.list_methods() if m not in _runtimes()]
    for m in unmeasured:
        assert hint(m) == {"tier": "unknown", "worst_sec": None, "observed": []}
