"""multibench - run and compare single-cell multimodal integration methods.

Quickstart
----------
::

    import multibench as mtb

    print(mtb.describe_layout())     # the four categories and how to lay out your dataset

    mtb.scan("MYDATA")               # which methods can run on it (and why not the rest)

    res = mtb.run_all("MYDATA", "vertical", out_dir="out/")
    print(res.summary)               # per-method status + metrics
    print(res.failures)              # what did not finish, and why
    res.plot().savefig("compare.png")

The four ``category`` values are ``vertical``, ``diagonal``, ``mosaic`` and
``cross`` - see :func:`describe_layout` for what each one means and the
filenames each expects.
"""

from . import config
from . import plot
from . import eval
from . import data
from .data import catalog
from .data.results import (load_results, available_datasets, results_coverage, recommend,
                           DegenerateRerunWarning)
from .eval.pipeline import evaluate, to_long
from .eval.classify import classify, classification_metrics
from .engine.runner import run
from .engine import registry as _registry
from .engine.schema import AmbiguousVariantError
from .engine.resolve import inputs_for, labels_for
from .engine import ingest as io
from .engine import envs as env
from .discover import list_methods, find_methods, method_info, params_for, cite
from .workflow import (scan, run_all, BatchResult, list_categories,
                       describe_layout, load_batch, sweep)
from . import _compat
from ._compat import deprecated_alias as _deprecated_alias

# `import multibench.io` / `from multibench.env import plan` work as for a real
# submodule: the three namespaces live elsewhere in the package
import sys as _sys
for _name, _module in (("io", io), ("catalog", catalog), ("env", env)):
    _sys.modules.setdefault(f"{__name__}.{_name}", _module)
del _sys, _name, _module

__version__ = "0.3.9"

__all__ = [
    # sub-namespaces
    "config", "plot", "catalog", "io", "env", "data",
    # discover
    "find_methods", "method_info", "params_for", "describe_layout", "cite",
    # inputs
    "inputs_for", "labels_for",
    # run
    "scan", "run", "run_all", "sweep", "load_batch", "BatchResult",
    # score
    "evaluate", "to_long", "classify", "classification_metrics",
    # compare
    "load_results", "results_coverage", "recommend",
    "AmbiguousVariantError", "DegenerateRerunWarning", "__version__",
]


def _runtime_of(method: str) -> dict:
    """``method_info(method)["runtime"]`` (target of the deprecated ``runtime_hint``)."""
    return method_info(method)["runtime"]


# deprecated 0.2 spellings: reachable, not in __all__ or dir(mtb)
plan = _deprecated_alias("plan", "scan", scan)
plan_commands = _deprecated_alias("plan_commands", "scan", scan)
runtime_hint = _deprecated_alias("runtime_hint", "method_info(m)['runtime']", _runtime_of)

# deprecated 0.4 names: reachable until 0.5, not in __all__ or dir(mtb)
list_tasks = _compat.renamed("list_tasks", "category= and metrics=", _registry.list_tasks)


def __dir__() -> list[str]:
    """``dir(mtb)`` lists ``__all__`` and the dunders only (PEP 562)."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
