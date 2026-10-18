"""catalog.datasets is the one dataset list: the benchmark's and the demo subsets, with
fetchable and has_results (core040 O8)."""
import multibench as mtb
from multibench.data.fetch import AVAILABLE, DEMO_CATEGORIES


def test_columns_and_rows():
    ds = mtb.catalog.datasets()
    assert list(ds.columns) == ["dataset", "category", "simulated", "fetchable", "has_results"]
    assert set(DEMO_CATEGORIES) <= set(ds["dataset"]) and len(DEMO_CATEGORIES) == 12
    assert ds["dataset"].is_unique


def test_fetchable_is_what_fetch_downloads():
    ds = mtb.catalog.datasets()
    assert set(ds.loc[ds.fetchable, "dataset"]) == set(AVAILABLE)


def test_has_results_marks_the_stored_datasets():
    ds = mtb.catalog.datasets()
    assert set(ds.loc[ds.has_results, "dataset"]) == {"D11", "D28", "D45", "D52"}


def test_a_demo_subset_has_its_tutorial_category():
    ds = mtb.catalog.datasets(category="cross")
    assert {"D52mini", "D56mini", "D58mini", "D59mini"} <= set(ds["dataset"])
