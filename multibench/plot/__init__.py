"""Plotting API.

``mtb.plot.bubble(long_df, ...)`` draws the paper-style bubble table and
``mtb.plot.bar(long_df, ...)`` the across-dataset summary bars;
``mtb.plot.build_table`` returns the numbers behind a bubble figure (a
:class:`BubbleTable`) so the ranks can be audited before drawing.
``help(mtb.plot.bubble)`` documents every parameter and the visual encoding.
The figure half, ``render``, and the per-family block ``FamilyBlock`` stay
importable but are not part of the public listing.
"""
from . import style
from .. import _compat
from .bubble import (bubble, build_table, render, BubbleTable, FamilyBlock,
                     FAMILIES as _FAMILIES)
from .bar import bar, CLUSTERING_METRICS, BATCH_METRICS, CLASSIFICATION_METRICS

__all__ = ["bubble", "bar", "build_table", "BubbleTable"]
bubble.__module__ = "multibench.plot"       # the name it is reached by

_METRICS = _compat.use("catalog.metrics()")
__getattr__ = _compat.module_getattr(__name__, "plot", old={
    "FAMILIES": (_FAMILIES, _METRICS),
    "CLUSTERING_METRICS": (CLUSTERING_METRICS, _METRICS),
    "BATCH_METRICS": (BATCH_METRICS, _METRICS),
    "CLASSIFICATION_METRICS": (CLASSIFICATION_METRICS, _METRICS),
}, removed={
    "plot_bubble": ("0.4", _compat.use("mtb.plot.bubble")),
})
del CLUSTERING_METRICS, BATCH_METRICS, CLASSIFICATION_METRICS


def __dir__() -> list[str]:
    """Only the public names and dunders (PEP 562); the other imports stay reachable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
