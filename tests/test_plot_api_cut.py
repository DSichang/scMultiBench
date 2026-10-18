"""The plot side's public surface: ``mtb.plot.__all__`` is four names,
``plot_bubble`` is removed (0.4) with an error naming ``bubble``, ``render`` /
``FamilyBlock`` stay importable but leave the listing, and every selector of
``bubble`` / ``bar`` / ``build_table`` is keyword-only after ``long_df``.
"""
import inspect

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
    assert mtb.plot.__all__ == ["bubble", "bar", "build_table", "BubbleTable"]
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


def test_bubble_bar_build_table_are_keyword_only_after_long_df():
    for fn in (mtb.plot.bubble, mtb.plot.bar, mtb.plot.build_table):
        params = list(inspect.signature(fn).parameters.values())
        assert params[0].name == "long_df"
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[1:]), fn.__name__
    # today's defaults are kept
    sig = inspect.signature(mtb.plot.bubble).parameters
    assert sig["aggregate"].default == "dataset" and sig["overall"].default == "rank"
    assert "na" not in sig and "show_language" not in sig          # deprecated in 0.4
    # build_table(na=) is deprecated since 0.4.0 (X41), as bubble(na=)
    assert "na" not in inspect.signature(mtb.plot.build_table).parameters
    with pytest.raises(TypeError):
        mtb.plot.build_table(_toy_long(), ["ARI"])
    tbl = mtb.plot.build_table(_toy_long(), metrics=["ARI"])
    assert isinstance(tbl, mtb.plot.BubbleTable)
