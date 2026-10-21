"""Demo data: ``fetch`` downloads demo datasets and ``write_example`` writes a
tutorial's example ``.h5ad`` files; ``mtb.catalog.datasets()`` lists them."""
from ._data.examples import write_example
from ._data.fetch import fetch

__all__ = ["fetch", "write_example"]
