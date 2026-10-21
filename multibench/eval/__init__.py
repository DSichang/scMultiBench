"""evaluate(): the benchmark's metrics of an embedding, as the long results table."""
from .pipeline import evaluate

__all__ = ["evaluate"]


def __dir__() -> list[str]:
    """Only the public names and dunders (PEP 562); the submodules stay importable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
