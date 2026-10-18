"""recommend(long_df=) gives the numbers behind a figure, in its row order (core040 M7)."""
import warnings

import pandas as pd
import pytest

import multibench as mtb
from multibench.plot.bubble import _build_table

COLUMNS = ["method", "grand_score", "needs_labels", "runtime_tier", "worst_sec", "env"]


def test_the_six_columns():
    assert list(mtb.recommend("vertical").columns) == COLUMNS


@pytest.mark.parametrize("category", ["vertical", "diagonal", "mosaic", "cross"])
@pytest.mark.parametrize("metrics", ["clustering", "batch", "all"])
def test_the_order_is_the_bubble_order_for_the_same_metrics(category, metrics):
    df = mtb.load_results(category)
    if metrics == "batch" and not df["metric"].isin(["iLISI", "GC"]).any():
        pytest.skip("one batch: no batch scores")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ranked = mtb.recommend(category, long_df=df, metrics=metrics)
        drawn = _build_table(df, metrics=metrics).methods
    assert list(ranked.dropna(subset=["grand_score"])["method"]) == drawn


def test_min_methods_is_deprecated_and_changes_nothing():
    with pytest.warns(FutureWarning, match=r"recommend\(min_methods=\) is deprecated"):
        old = mtb.recommend("vertical", min_methods=5)
    pd.testing.assert_frame_equal(old, mtb.recommend("vertical"))
