"""Figures of a long table: ``mtb.plot.bubble`` draws the article's bubble table
and ``mtb.plot.bar`` one bar per method. ``mtb.recommend(category, long_df=...)``
gives the numbers behind both, in their row order."""
from . import style  # noqa: F401
from .bubble import bubble
from .bar import bar

__all__ = ["bubble", "bar"]
bubble.__module__ = "multibench.plot"       # the name it is reached by


def __dir__() -> list[str]:
    """Only the public names and dunders (PEP 562); the submodules stay importable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
