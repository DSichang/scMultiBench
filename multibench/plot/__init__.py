"""Plotting API: ``plot.bubble(...)`` renders; ``plot.bubble.build_table`` etc."""
from . import bubble as _bubble_module


class _BubbleNamespace:
    """Callable shim so ``plot.bubble(df)`` renders while ``plot.bubble.foo``
    still reaches the underlying module's functions (build_table, render, ...)."""

    def __call__(self, *args, **kwargs):
        return _bubble_module.plot_bubble(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_bubble_module, name)


bubble = _BubbleNamespace()
__all__ = ["bubble"]
