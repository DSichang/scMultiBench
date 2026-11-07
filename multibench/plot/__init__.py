"""Figures of a long table: ``mtb.plot.bubble`` draws the paper's bubble table
and ``mtb.plot.bar`` one bar per method. ``mtb.rank(category, long_df=...)``
gives the numbers behind both, in their row order."""
from .._text import public_dir
from .bubble import bubble
from .bar import bar

__all__ = ["bubble", "bar"]
bubble.__module__ = "multibench.plot"       # the name it is reached by
__dir__ = public_dir(globals())
