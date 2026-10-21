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


def test_bubble_and_bar_are_keyword_only_after_long_df():
    for fn in (mtb.plot.bubble, mtb.plot.bar):
        params = list(inspect.signature(fn).parameters.values())
        assert [p.name for p in params] == ["long_df", "metrics", "methods", "title", "save"]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[1:]), fn.__name__


