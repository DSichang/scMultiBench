"""catalog.datasets is the one dataset list: the benchmark's and the demo subsets, with
has_results (core040 O8); every row is one mtb.data.fetch downloads, so no fetchable
column (v1 u5#5)."""
import multibench as mtb
from multibench._data.fetch import AVAILABLE, DEMO_CATEGORIES


def test_columns_and_rows():
    ds = mtb.catalog.datasets()
    # c1 (C34): the rows a user can act on, each with its category
    assert list(ds.columns) == ["dataset", "category", "has_results"]
    assert set(DEMO_CATEGORIES) <= set(ds["dataset"]) and len(DEMO_CATEGORIES) == 13
    assert ds["category"].notna().all()
    assert ds["dataset"].is_unique


def test_the_rows_are_what_fetch_downloads():
    ds = mtb.catalog.datasets()
    assert set(ds["dataset"]) == set(AVAILABLE)


def test_has_results_marks_the_stored_datasets():
    ds = mtb.catalog.datasets()
    assert set(ds.loc[ds.has_results, "dataset"]) == {"D11", "D28", "D45", "D52"}


def test_a_demo_subset_has_its_tutorial_category():
    ds = mtb.catalog.datasets(category="cross")
    assert {"D52mini", "D56mini", "D58mini", "D59mini"} <= set(ds["dataset"])
