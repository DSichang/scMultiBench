"""Demo data: ``fetch`` downloads demo datasets and ``write_example`` writes a
tutorial's example ``.h5ad`` files; ``mtb.catalog.datasets()`` lists them."""

from .examples import write_example  # noqa: F401
from .fetch import fetch  # noqa: F401

__all__ = ["fetch", "write_example"]


def __dir__() -> list[str]:
    """Only the public names and dunders (PEP 562); the submodules stay importable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
