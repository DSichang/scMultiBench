"""0.3.0 public-surface cut, plot side: ``mtb.plot.__all__`` is exactly seven
names, ``plot_bubble`` is a deprecated alias of ``bubble``, ``render`` /
``FamilyBlock`` stay importable but leave the listing, and every selector of
``bubble`` / ``bar`` / ``build_table`` is keyword-only after ``long_df``.
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
    assert mtb.plot.__all__ == ["bubble", "bar", "build_table", "BubbleTable", "FAMILIES",
                                "CLUSTERING_METRICS", "BATCH_METRICS"]
    listed = [n for n in dir(mtb.plot) if not n.startswith("__")]
    assert sorted(listed) == sorted(mtb.plot.__all__)
    # importable, just not listed
    from multibench.plot import render, FamilyBlock, plot_bubble   # noqa: F401
    assert callable(mtb.plot.render) and mtb.plot.FamilyBlock is FamilyBlock


def test_plot_bubble_is_a_deprecated_alias_of_bubble(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = mtb.plot.bubble(_toy_long(), metrics=["ARI", "NMI"])
    with pytest.warns(DeprecationWarning, match=r"mtb.plot.plot_bubble is deprecated since 0.3.0.*use mtb.plot.bubble"):
        old = mtb.plot.plot_bubble(_toy_long(), metrics=["ARI", "NMI"])
    assert mtb.plot.plot_bubble.__wrapped__ is mtb.plot.bubble
    assert len(old.axes) == len(new.axes)
    assert [t.get_text() for t in old.axes[0].texts] == [t.get_text() for t in new.axes[0].texts]
    assert inspect.signature(mtb.plot.plot_bubble) == inspect.signature(mtb.plot.bubble)


def test_bubble_bar_build_table_are_keyword_only_after_long_df():
    for fn in (mtb.plot.bubble, mtb.plot.bar, mtb.plot.build_table):
        params = list(inspect.signature(fn).parameters.values())
        assert params[0].name == "long_df"
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[1:]), fn.__name__
    # today's defaults are kept
    sig = inspect.signature(mtb.plot.bubble).parameters
    assert sig["aggregate"].default == "dataset" and sig["overall"].default == "rank"
    assert sig["na"].default == "warn" and sig["show_language"].default is True
    with pytest.raises(TypeError):
        mtb.plot.build_table(_toy_long(), ["ARI"])
    tbl = mtb.plot.build_table(_toy_long(), metrics=["ARI"])
    assert isinstance(tbl, mtb.plot.BubbleTable)
