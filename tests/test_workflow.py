"""Tests for the high-level workflow API (scan / run_all / BatchResult).

The runnable COUNTS below are reference-host values (every method env
installed). They are skipped, not failed, on a host without the envs; the
host-independent contracts live in tests/test_scan_gates.py and
tests/test_workflow_study2.py.
"""
import pandas as pd
import pytest

import multibench as mtb


def _envs_installed(category=None) -> bool:
    """Every env the category's methods need exists on this host."""
    try:
        rows = mtb.env.doctor(category=category)
    except Exception:  # noqa: BLE001 - no conda at all
        return False
    return bool(rows) and all(r["exists"] for r in rows)


_NO_ENVS = "reference-host runnable counts need the method conda envs (none here)"


@pytest.mark.skipif(not _envs_installed(), reason=_NO_ENVS)
def test_scan_returns_runnable_and_reasons():
    df = mtb.scan("D11")
    assert {"method", "category", "modalities", "runnable", "reason"} <= set(df.columns)
    assert df["runnable"].any()
    # anything not runnable must say why
    bad = df[~df["runnable"]]
    assert (bad["reason"].str.len() > 0).all()


def test_scan_filters_by_category():
    df = mtb.scan("D11", category="vertical")
    assert set(df["category"]) == {"vertical"}


def test_scan_rejects_spatial_methods_on_non_spatial_data():
    """A data_dir path always exists, so existence alone must not imply runnable."""
    df = mtb.scan("D11", category="cross")
    spatial = df[df["method"].isin(["PASTE", "PASTE2", "SPIRAL", "GPSA"])]
    assert len(spatial) == 4
    assert not spatial["runnable"].any()
    assert spatial["reason"].str.contains("h5ad").all()


def _has_dataset(name):
    from multibench import config
    return (config.DEFAULT.data_path / name).is_dir()


@pytest.mark.skipif(not _has_dataset("D63"), reason="D63 slices not on this host")
def test_scan_accepts_spatial_methods_on_spatial_data():
    df = mtb.scan("D63", category="cross")
    spatial = df[df["method"].isin(["PASTE", "PASTE2", "SPIRAL", "GPSA"])]
    assert spatial["runnable"].all()


@pytest.mark.skipif(not _envs_installed("vertical"), reason=_NO_ENVS)
def test_run_all_dry_run_lists_the_plan():
    """dry_run returns the WHOLE plan (blocked rows kept with a reason); the
    runnable subset is what will be attempted - 14 rows on the reference host."""
    plan = mtb.run_all("D11", "vertical", out_dir="/tmp/unused", dry_run=True,
                       verbose=False)
    assert isinstance(plan, pd.DataFrame)
    ok = plan[plan["runnable"]]
    assert (ok["reason"] == "").all()
    assert len(ok) == 14


def test_run_all_dry_run_respects_method_filter():
    plan = mtb.run_all("D11", "vertical", out_dir="/tmp/unused",
                       methods=["Matilda", "totalVI"], dry_run=True)
    assert set(plan["method"]) == {"Matilda", "totalVI"}


def test_run_all_dry_run_is_the_scan_frame():
    """Holds on every host: dry_run == scan(dataset, category), row for row."""
    for ds, cat in [("D11", "vertical"), ("D52", "cross"), ("D63", "cross"),
                    ("D45", "mosaic")]:
        if not _has_dataset(ds):
            continue
        plan = mtb.run_all(ds, cat, out_dir="/tmp/unused", dry_run=True, verbose=False)
        pd.testing.assert_frame_equal(plan, mtb.scan(ds, cat))


@pytest.mark.skipif(not (_envs_installed("vertical") and _envs_installed("cross")
                         and _envs_installed("mosaic")), reason=_NO_ENVS)
def test_run_all_plans_match_scan():
    """Runnable counts on the reference host (needs the method envs)."""
    for ds, cat, n in [("D11", "vertical", 14), ("D52", "cross", 8),
                       ("D63", "cross", 4), ("D45", "mosaic", 4)]:
        if not _has_dataset(ds):
            continue
        plan = mtb.run_all(ds, cat, out_dir="/tmp/unused", dry_run=True, verbose=False)
        assert int(plan["runnable"].sum()) == n, f"{ds}/{cat}: {int(plan['runnable'].sum())} != {n}"


def test_method_info_supports_multi_category_methods():
    """Requirement: a user must see what a multi-category method can be run as."""
    sup = mtb.method_info("Multigrate")["supports"]
    cats = {s["category"] for s in sup}
    assert {"vertical", "mosaic"} <= cats
    for s in sup:
        assert s["modalities"] and s["output_kind"]


def test_batch_result_shape():
    from multibench.workflow import BatchResult
    r = BatchResult([{"method": "X", "status": "CHAIN_OK", "run_sec": 1.0,
                      "output_kind": "embedding", "emb_shape": [10, 2],
                      "n_tunable": 0, "metrics": {"ARI": 0.5}, "_long": None}],
                    "D11", "vertical")
    assert len(r) == 1
    assert r.summary.loc[0, "ARI"] == 0.5
    assert r.failures.empty
    # .long is DERIVED from the record's metrics when no tidy frame was attached
    lng = r.long
    assert list(lng.columns) == ["metric", "value", "method", "dataset", "category"]
    assert lng.loc[0, "metric"] == "ARI" and lng.loc[0, "value"] == 0.5
    assert lng.loc[0, "dataset"] == "D11" and lng.loc[0, "category"] == "vertical"
    empty = BatchResult([{"method": "X", "status": "FAIL", "error": "boom", "_long": None}],
                        "D11", "vertical")
    assert empty.long.empty and list(empty.long.columns) == list(lng.columns)
    with pytest.raises(ValueError):
        empty.plot()      # nothing scored -> must refuse rather than draw nothing
