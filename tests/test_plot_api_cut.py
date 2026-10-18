"""The plot side's public surface: ``mtb.plot.__all__`` is ``bubble`` and
``bar`` (0.4.0; ``build_table`` / ``BubbleTable`` are deprecated for
``mtb.recommend(category, long_df=...)``), ``plot_bubble`` is removed (0.4) with
an error naming ``bubble``, ``render`` / ``FamilyBlock`` stay importable but
leave the listing, and ``bubble`` / ``bar`` take the same keyword-only
selectors after ``long_df``.
"""
import inspect
import warnings

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

import multibench as mtb


def _toy_long():
    return pd.DataFrame({
        "method": ["A", "A", "B", "B"],
        "metric": ["ARI", "NMI", "ARI", "NMI"],
        "value": [0.2, 0.4, 0.8, 0.6],
        "dataset": ["D1"] * 4,
        "category": ["vertical"] * 4,
    })


def test_plot_all_and_dir_are_the_contract():
    assert mtb.plot.__all__ == ["bubble", "bar"]
    listed = [n for n in dir(mtb.plot) if not n.startswith("__")]
    assert sorted(listed) == sorted(mtb.plot.__all__)
    # deprecated in 0.4 (catalog.metrics() has the family column): importable, warn
    for name in ("FAMILIES", "CLUSTERING_METRICS", "BATCH_METRICS",
                 "CLASSIFICATION_METRICS"):
        with pytest.warns(FutureWarning, match=f"plot.{name} is deprecated"):
            getattr(mtb.plot, name)
    # importable, just not listed
    from multibench.plot import render, FamilyBlock   # noqa: F401
    assert callable(mtb.plot.render) and mtb.plot.FamilyBlock is FamilyBlock


def test_plot_bubble_is_removed_naming_bubble():
    with pytest.raises(AttributeError) as e:
        mtb.plot.plot_bubble
    assert str(e.value) == "mtb.plot.plot_bubble was removed in 0.4. Use mtb.plot.bubble."
    with pytest.raises(ImportError):
        from multibench.plot import plot_bubble   # noqa: F401


def test_bubble_and_bar_are_keyword_only_after_long_df():
    for fn in (mtb.plot.bubble, mtb.plot.bar):
        params = list(inspect.signature(fn).parameters.values())
        assert [p.name for p in params] == ["long_df", "metrics", "methods", "title", "save"]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[1:]), fn.__name__


def test_build_table_warns_once_and_still_builds_the_table():
    with pytest.raises(TypeError):
        mtb.plot.build_table(_toy_long(), ["ARI"])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tbl = mtb.plot.build_table(_toy_long(), metrics=["ARI"])
        is_table = isinstance(tbl, mtb.plot.BubbleTable)
    assert [str(w.message) for w in rec if issubclass(w.category, FutureWarning)] == [
        "plot.build_table is deprecated and will be removed in 0.5. Use "
        "mtb.recommend(category, long_df=...).",
        "plot.BubbleTable is deprecated and will be removed in 0.5. Use "
        "mtb.recommend(category, long_df=...)."]
    assert is_table and tbl.methods == ["B", "A"]
