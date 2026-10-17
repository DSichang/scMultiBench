"""evaluate(): compute metrics; to_long(): reshape to the long results frame."""
from .pipeline import evaluate, to_long

__all__ = ["evaluate", "to_long"]


def __dir__() -> list[str]:
    """Only the public names and dunders (PEP 562); the submodules stay importable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
