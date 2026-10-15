"""Plotting API.

``mtb.plot.bubble(long_df, ...)`` draws the paper-style bubble table and
``mtb.plot.bar(long_df, ...)`` the across-dataset summary bars;
``mtb.plot.build_table`` returns the numbers behind a bubble figure (a
:class:`BubbleTable`) so the ranks can be audited before drawing.
``help(mtb.plot.bubble)`` documents every parameter and the visual encoding.
The figure half, ``render``, and the per-family block ``FamilyBlock`` stay
importable but are not part of the public listing; ``plot_bubble`` is the
deprecated 0.2.x name of ``bubble``.
"""
from . import bubble as _bubble_module
from . import style
from .bubble import (bubble, plot_bubble, build_table, render, BubbleTable,
                     FamilyBlock, FAMILIES)
from .bar import bar, CLUSTERING_METRICS, BATCH_METRICS, CLASSIFICATION_METRICS

# the 0.2.x attribute paths (``mtb.plot.bubble.build_table`` ...) until 0.4
for _n in ("build_table", "render", "plot_bubble", "BubbleTable", "FamilyBlock",
           "FAMILIES", "NA_MARK", "_resolve"):
    setattr(bubble, _n, getattr(_bubble_module, _n))
bubble.style = style
bubble.__module__ = "multibench.plot"
del _n

__all__ = ["bubble", "bar", "build_table", "BubbleTable", "FAMILIES",
           "CLUSTERING_METRICS", "BATCH_METRICS", "CLASSIFICATION_METRICS"]


def __dir__() -> list[str]:
    """Only the public names and dunders (PEP 562); the other imports stay reachable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
