"""Plotting API: ``plot.bubble(...)`` renders; ``plot.bubble.build_table`` etc."""
from . import bubble as _bubble_module
from .bubble import plot_bubble, build_table, render, BubbleTable


class _BubbleNamespace:
    """Callable shim so ``plot.bubble(df)`` renders while ``plot.bubble.foo``
    still reaches the underlying module's functions (build_table, render, ...)."""

    def __call__(self, *args, **kwargs):
        return _bubble_module.plot_bubble(*args, **kwargs)

    __call__.__doc__ = None  # replaced right below

    def __getattr__(self, name):
        return getattr(_bubble_module, name)


bubble = _BubbleNamespace()
# interactive help(mtb.plot.bubble) used to show an opaque shim; borrow the
# real function's signature and docstring
try:
    import functools as _ft
    _BubbleNamespace.__call__ = _ft.wraps(plot_bubble)(_BubbleNamespace.__call__.__wrapped__ if hasattr(_BubbleNamespace.__call__, "__wrapped__") else _BubbleNamespace.__call__)
    bubble.__doc__ = plot_bubble.__doc__
except Exception:
    pass
# Also expose the underlying callables directly for discoverability
# (``mtb.plot.plot_bubble`` / ``mtb.plot.build_table`` / ``mtb.plot.render``).
__all__ = ["bar", "bubble", "plot_bubble", "build_table", "render", "BubbleTable"]

from .bar import bar, CLUSTERING_METRICS, BATCH_METRICS
