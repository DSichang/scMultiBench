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
from . import data
from .data import catalog
from .data.results import (load_results, available_datasets, results_coverage, recommend,
                           DegenerateRerunWarning)
from .eval.pipeline import evaluate, to_long
from .engine.runner import run
from .engine.registry import list_tasks
from .engine.schema import AmbiguousVariantError
from .engine.resolve import inputs_for, labels_for
from .engine import ingest as io
from .engine import envs as env
from .discover import list_methods, find_methods, method_info, params_for, cite
from .workflow import (scan, run_all, BatchResult, list_categories,
                       describe_layout, load_batch, sweep)
from ._compat import deprecated_alias as _deprecated_alias

__version__ = "0.3.0"

__all__ = [
    # sub-namespaces
    "config", "plot", "eval", "catalog", "io", "env", "data",
    # discover
    "list_methods", "find_methods", "method_info", "params_for", "describe_layout", "cite",
    "list_tasks", "list_categories",
    # inputs
    "inputs_for", "labels_for",
    # run
    "scan", "run", "run_all", "sweep", "load_batch", "BatchResult",
    # score
    "evaluate", "to_long",
    # compare
    "load_results", "available_datasets", "results_coverage", "recommend",
    "AmbiguousVariantError", "DegenerateRerunWarning", "__version__",
]


def _runtime_of(method: str) -> dict:
    """``method_info(method)["runtime"]`` (target of the deprecated ``runtime_hint``)."""
    return method_info(method)["runtime"]


# Deprecated 0.2 spellings, kept for ONE release: each warns (DeprecationWarning)
# and forwards. They stay reachable as attributes but are not in __all__ and
# do not show in dir(mtb).
plan = _deprecated_alias("plan", "scan", scan)
plan_commands = _deprecated_alias("plan_commands", "scan", scan)
runtime_hint = _deprecated_alias("runtime_hint", "method_info(m)['runtime']", _runtime_of)


def __dir__() -> list[str]:
    """``dir(mtb)`` lists the public surface (``__all__``) and the dunders only
    (PEP 562): imported submodules and the deprecated aliases are reachable
    but not advertised."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
