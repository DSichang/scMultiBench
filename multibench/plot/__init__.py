"""Figures of a long table: ``mtb.plot.bubble`` draws the article's bubble table
and ``mtb.plot.bar`` one bar per method. ``mtb.recommend(category, long_df=...)``
gives the numbers behind both, in their row order."""
from . import style
from .. import _compat
from .bubble import (bubble, build_table, render, BubbleTable as _BubbleTable,  # noqa: F401
                     FamilyBlock, FAMILIES as _FAMILIES)
from .bar import bar, CLUSTERING_METRICS, BATCH_METRICS, CLASSIFICATION_METRICS

__all__ = ["bubble", "bar"]
bubble.__module__ = "multibench.plot"       # the name it is reached by

_METRICS = _compat.use("catalog.metrics()")
__getattr__ = _compat.module_getattr(__name__, "plot", old={
    "FAMILIES": (_FAMILIES, _METRICS),
    "CLUSTERING_METRICS": (CLUSTERING_METRICS, _METRICS),
    "BATCH_METRICS": (BATCH_METRICS, _METRICS),
    "CLASSIFICATION_METRICS": (CLASSIFICATION_METRICS, _METRICS),
    "BubbleTable": (_BubbleTable, _compat.use("mtb.recommend(category, long_df=...)")),
}, removed={
    "plot_bubble": ("0.4", _compat.use("mtb.plot.bubble")),
})
del CLUSTERING_METRICS, BATCH_METRICS, CLASSIFICATION_METRICS


def __dir__() -> list[str]:
    """Only the public names and dunders (PEP 562); the other imports stay reachable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
