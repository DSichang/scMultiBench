"""Plotting API: ``plot.bubble(...)`` renders; ``plot.bubble.build_table`` etc."""
from . import bubble as _bubble_module
from .bubble import plot_bubble, build_table, render, BubbleTable


class _BubbleNamespace:
    """Callable shim so ``plot.bubble(df)`` renders while ``plot.bubble.foo``
    still reaches the underlying module's functions (build_table, render, ...)."""

    def __call__(self, *args, **kwargs):
        return _bubble_module.plot_bubble(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_bubble_module, name)


bubble = _BubbleNamespace()
# Also expose the underlying callables directly for discoverability
# (``mtb.plot.plot_bubble`` / ``mtb.plot.build_table`` / ``mtb.plot.render``).
__all__ = ["bubble", "plot_bubble", "build_table", "render", "BubbleTable"]
