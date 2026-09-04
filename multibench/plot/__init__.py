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
from .bar import bar, CLUSTERING_METRICS, BATCH_METRICS

# back-compat: ``bubble`` used to be a callable namespace object, so
# ``from multibench.plot import bubble; bubble.build_table(...)`` and
# ``mtb.plot.bubble.render(...)`` worked. ``bubble`` is now a plain function
# (so help()/inspect.signature show its real parameters); the old attribute
# paths are kept as attributes on the function for one release (drop in 0.4).
for _n in ("build_table", "render", "plot_bubble", "BubbleTable", "FamilyBlock",
           "FAMILIES", "NA_MARK", "_resolve", "_pivot"):
    setattr(bubble, _n, getattr(_bubble_module, _n))
bubble.style = style
bubble.__module__ = "multibench.plot"
del _n

__all__ = ["bubble", "bar", "build_table", "BubbleTable", "FAMILIES",
           "CLUSTERING_METRICS", "BATCH_METRICS"]


def __dir__() -> list[str]:
    """``dir(mtb.plot)`` lists the public names and the dunders only (PEP
    562): ``render``, ``FamilyBlock``, ``style`` and the deprecated
    ``plot_bubble`` stay accessible but out of the listing."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
