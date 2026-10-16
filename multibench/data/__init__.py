"""The benchmark's data: ``fetch`` downloads a dataset, ``fetch_outputs`` stored
method outputs, ``fetchable`` lists the ids ``fetch`` takes, and ``write_example``
writes a tutorial dataset as ``.h5ad`` files."""

from .examples import write_example  # noqa: F401
from .fetch import fetch, fetch_outputs  # noqa: F401
from .results import fetchable  # noqa: F401  (the downloadable reference datasets)

__all__ = ["fetch", "fetch_outputs", "fetchable", "write_example"]
