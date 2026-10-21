import pytest

from multibench._data import results

# source= / result_path= / clustering= and available_datasets are deprecated in
# 0.4 (one FutureWarning each); the published-table tests pass them on purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.* is deprecated and will be removed in 0.5:FutureWarning")


def test_load_results_returns_tidy_long_frame(result_dir):
    df = results.load_results(category="diagonal", dataset="D28")
    assert set(["metric", "value", "method", "dataset", "category"]).issubset(df.columns)
    assert (df["category"] == "diagonal").all()
    assert (df["dataset"] == "D28").all()


def test_metric_filter(result_dir):
    df = results.load_results(category="diagonal", dataset="D28",
                              metrics=["ARI", "NMI"])
    assert set(df["metric"].unique()) <= {"ARI", "NMI"}


def test_methods_are_canonicalized(result_dir):
    df = results.load_results(category="diagonal")
    # no result-dir-style dotted names leak through
    assert not any("." in m for m in df["method"].unique())


# ---------------------------------------------------------------------------
# provenance (`source`), clustering-variant dirs, filters, error texts
# ---------------------------------------------------------------------------
import warnings

import pandas as pd
import pytest

import multibench as mtb

# Y39 (r3): clustering only for a deprecated published variant other than the default
COLS = ["metric", "value", "method", "dataset", "category", "source"]


def test_methods_filter_accepts_list_and_aliases(result_dir):
    df = results.load_results("diagonal", dataset="D28", methods=["scbridge", "Seurat v3"])
    assert set(df.method) == {"scBridge", "Seurat_v3"}
    df2 = results.load_results("diagonal", dataset="D28", methods="scBridge")
    assert set(df2.method) == {"scBridge"}


def test_metrics_filters_by_family(result_dir):
    full = results.load_results("diagonal", dataset="D28")
    fam = mtb.catalog.metrics().groupby("family")["metric"].apply(set)
    b = results.load_results("diagonal", dataset="D28", metrics="batch")
    assert set(b.metric) <= fam["batch"] and len(b)
    c = results.load_results("diagonal", dataset="D28", metrics="clustering")
    assert set(c.metric) <= fam["clustering"] and len(c)
    assert len(b) + len(c) == len(full)      # the two families partition the frame
    with pytest.raises(ValueError, match=r"Unknown metrics= token bogus\. Pass 'all', "
                                         r"'clustering', 'batch' or 'classification'"):
        results.load_results("diagonal", dataset="D28", metrics="bogus")
    with pytest.raises(ValueError, match=r"Unknown metrics= token dimension_reduction\."):
        results.load_results("diagonal", metrics="dimension_reduction")
    # everything after category is keyword-only: the positional slip is a TypeError
    with pytest.raises(TypeError):
        results.load_results("vertical", "D11")
    # None == 'all' == today's frame
    pd.testing.assert_frame_equal(
        results.load_results("diagonal", dataset="D28", metrics="all"), full)
    pd.testing.assert_frame_equal(
        results.load_results("diagonal", dataset="D28", metrics=None), full)


# ---------------------------------------------------------------------------
# P08 / Rin: provenance through CSV, list-filter validation, degenerate rows
# ---------------------------------------------------------------------------


