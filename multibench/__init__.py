"""scmbench — Python API for the scMultiBench benchmark."""

from . import config
from . import plot
from . import eval
from .data import catalog
from .data.results import load_results
from .eval.pipeline import evaluate
from .engine.runner import run
from .engine.registry import list_methods, list_tasks
from .engine.resolve import inputs_for
from .engine import ingest as io
from .engine import envs as env
from .discover import find_methods, method_info

__version__ = "0.1.0"

__all__ = ["config", "plot", "eval", "catalog", "load_results", "evaluate",
           "run", "list_methods", "list_tasks", "inputs_for", "io", "env",
           "method_info", "find_methods", "__version__"]
