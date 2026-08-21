"""multibench - run and compare single-cell multimodal integration methods.

Quickstart
----------
::

    import multibench as mtb

    mtb.list_categories()            # the four integration scenarios
    print(mtb.describe_layout())     # how to lay out YOUR dataset

    mtb.scan("MYDATA")               # which methods can run on it (and why not the rest)

    res = mtb.run_all("MYDATA", "vertical", out_dir="out/")
    print(res.summary)               # per-method status + metrics
    print(res.failures)              # what did not finish, and why
    res.plot().savefig("compare.png")

The four ``category`` values are ``vertical``, ``diagonal``, ``mosaic`` and
``cross`` - see :func:`list_categories` for what each one means, and
:func:`describe_layout` for the filenames each expects.
"""

from . import config
from . import plot
from . import eval
from .data import catalog
from .data.results import load_results, available_datasets, results_coverage, recommend
from .eval.pipeline import evaluate, to_long
from .engine.runner import run
from .engine.registry import list_methods, list_tasks
from .engine.resolve import inputs_for, labels_for
from .engine import ingest as io
from .engine import envs as env
from .discover import find_methods, method_info, params_for, cite
from .workflow import (scan, run_all, BatchResult, list_categories,
                       describe_layout, load_batch, runtime_hint, sweep, plan)

__version__ = "0.3.0"

__all__ = ["config", "plot", "eval", "catalog", "load_results", "available_datasets",
           "evaluate", "to_long", "run", "list_methods", "list_tasks", "inputs_for",
           "labels_for", "io", "env", "method_info", "find_methods", "params_for",
           "scan", "run_all", "BatchResult",
           "list_categories", "describe_layout", "load_batch",
           "runtime_hint", "sweep", "__version__", "results_coverage", "recommend", "cite", "plan"]
