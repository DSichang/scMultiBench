"""Demo data: ``fetch`` downloads demo datasets and ``write_example`` writes a
tutorial's example ``.h5ad`` files; ``mtb.catalog.datasets()`` lists them."""

from .. import _compat
from .examples import write_example  # noqa: F401
from .fetch import fetch, fetch_outputs as _fetch_outputs  # noqa: F401
from .results import fetchable as _fetchable

__all__ = ["fetch", "write_example"]

__getattr__ = _compat.module_getattr(__name__, "data", old={
    "fetch_outputs": (_fetch_outputs, _compat.note(
        "use mtb.load_results for the stored results")),
    "fetchable": (_fetchable, _compat.use("mtb.catalog.datasets()")),
})


def __dir__() -> list[str]:
    """Only the public names and dunders (PEP 562); the submodules stay importable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
