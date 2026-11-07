"""The figure class of ``mtb.plot.bubble``, in its own module so that importing
the package does not import matplotlib."""
from matplotlib.figure import Figure


class _BubbleFigure(Figure):
    """A ``Figure`` with ``_repr_png_``. Built without pyplot, the figure is
    never shown twice, and a notebook without ``%matplotlib inline`` still
    shows the image rather than ``<Figure size ...>``."""

    def _repr_png_(self):
        import io
        buf = io.BytesIO()
        self.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        return buf.getvalue()
