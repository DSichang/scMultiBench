"""evaluate(): the benchmark's metrics of an embedding, as the long results table."""
from .._text import public_dir
from .pipeline import evaluate

__all__ = ["evaluate"]
__dir__ = public_dir(globals())
