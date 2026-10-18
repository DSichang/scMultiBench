"""The benchmark's data: ``fetch`` downloads a dataset, ``fetchable`` lists the ids
``fetch`` takes, and ``write_example`` writes a tutorial dataset as ``.h5ad`` files."""

from .. import _compat
from .examples import write_example  # noqa: F401
from .fetch import fetch, fetch_outputs as _fetch_outputs  # noqa: F401
from .results import fetchable  # noqa: F401  (the downloadable reference datasets)

__all__ = ["fetch", "fetchable", "write_example"]

__getattr__ = _compat.module_getattr(__name__, "data", old={
    "fetch_outputs": (_fetch_outputs, _compat.note(
        "use mtb.load_results for the stored scores")),
})


def __dir__() -> list[str]:
    """Only the public names and dunders (PEP 562); the submodules stay importable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
