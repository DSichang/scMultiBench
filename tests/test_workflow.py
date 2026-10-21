"""Tests for the high-level workflow API (scan / run_all / BatchResult).

The runnable COUNTS below are reference-host values (every method env
installed). They are skipped, not failed, on a host without the envs; the
host-independent contracts live in tests/test_scan_gates.py and
tests/test_workflow_study2.py.
"""
import pandas as pd
import pytest

import multibench as mtb
from tests.conftest import full_scan


_NO_ENVS = "reference-host runnable counts need the method conda envs (none here)"


def test_scan_filters_by_category():
    df = full_scan("D11", category="vertical")
    assert set(df["category"]) == {"vertical"}


def _has_dataset(name):
    from multibench import config
    return (config.DEFAULT.data_path / name).is_dir()


def test_run_all_dry_run_respects_method_filter():
    plan = mtb.run_all("D11", "vertical", out_dir="/tmp/unused",
                       methods=["Matilda", "totalVI"], dry_run=True)
    assert set(plan["method"]) == {"Matilda", "totalVI"}


def test_run_all_dry_run_is_the_scan_frame():
    """Holds on every host: dry_run == scan(dataset, category), row for row, less
    the rows a run leaves out (0.4.0): a row whose files are part of another row
    of its method."""
    from multibench import workflow as W
    for ds, cat in [("D11", "vertical"), ("D52", "cross"), ("D45", "mosaic")]:
        if not _has_dataset(ds):
            continue
        plan = mtb.run_all(ds, cat, out_dir="/tmp/unused", dry_run=True, verbose=False)
        assert list(plan.columns)[-1] == "command"      # the CLI csv's column, in scan too
        scan = full_scan(ds, cat, out_dir="/tmp/unused", verbose=False)
        nested = W._nested_rows(scan[[W._would_run(r) for _, r in scan.iterrows()]])[0]
        pd.testing.assert_frame_equal(
            plan, W._shown_modalities(scan.drop(index=nested).reset_index(drop=True))[
                W.PLAN_COLUMNS])                                          # c1 (C32)


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
    assert list(lng.columns) == ["metric", "value", "method", "dataset", "category",
                                  "source"]
    assert lng.loc[0, "metric"] == "ARI" and lng.loc[0, "value"] == 0.5
    assert lng.loc[0, "dataset"] == "D11" and lng.loc[0, "category"] == "vertical"
    empty = BatchResult([{"method": "X", "status": "FAIL", "error": "boom", "_long": None}],
                        "D11", "vertical")
    assert empty.long.empty and list(empty.long.columns) == list(lng.columns)
    with pytest.raises(ValueError):
        empty.plot()      # nothing scored -> must refuse rather than draw nothing
