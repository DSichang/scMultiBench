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
from .data.results import (load_results, available_datasets, recommend,
                           results_coverage as _results_coverage)
from .data import results as _results
from .eval.pipeline import evaluate, to_long
from .eval.classify import (classify as _classify,
                            classification_metrics as _classification_metrics)
from .engine.runner import run as _run
from .engine import registry as _registry
from .engine.schema import AmbiguousVariantError as _AmbiguousVariantError
from .engine.resolve import inputs_for as _inputs_for, labels_for as _labels_for
from .engine import ingest as io
from .engine import envs as env
from .discover import list_methods, find_methods, method_info, params_for, cite
from .workflow import (scan, run_all, BatchResult, list_categories,
                       describe_layout, load_batch, sweep as _sweep)
from . import _compat

# `import multibench.io` / `from multibench.env import install` work as for a
# real submodule: the three namespaces live elsewhere in the package
import sys as _sys
for _name, _module in (("io", io), ("catalog", catalog), ("env", env)):
    _sys.modules.setdefault(f"{__name__}.{_name}", _module)
del _sys, _name, _module

__version__ = "0.4.0"

__all__ = [
    # namespaces
    "config", "plot", "catalog", "io", "env", "data",
    # methods
    "find_methods", "method_info", "params_for", "cite",
    # run
    "scan", "run_all", "load_batch", "BatchResult",
    # metrics
    "evaluate", "load_results", "recommend",
    # setup
    "describe_layout", "__version__",
]


# deprecated: reachable until 0.5, not in __all__ or dir(mtb)
list_tasks = _compat.renamed("list_tasks", "category= and metrics=", _registry.list_tasks)
_RUN_ONE = "mtb.run_all(dataset, category, methods=[method])"
run = _compat.renamed("run", _RUN_ONE, _run)
inputs_for = _compat.renamed("inputs_for", _RUN_ONE, _inputs_for)
labels_for = _compat.renamed("labels_for", "mtb.evaluate(..., dataset=, method=)", _labels_for)
sweep = _compat.renamed("sweep", "a loop over mtb.run_all(params=...)", _sweep)
_CLASSIFICATION = "mtb.evaluate(..., metrics='classification')"
classify = _compat.renamed("classify", _CLASSIFICATION, _classify)
classification_metrics = _compat.renamed("classification_metrics", _CLASSIFICATION,
                                         _classification_metrics)
results_coverage = _compat.renamed("results_coverage", "mtb.load_results(category)",
                                   _results_coverage)

# the 0.2 names, removed in 0.4 as their warnings said
__getattr__ = _compat.module_getattr(__name__, "", old={
    # nothing raises it since the Conos rows left the stored tables
    "DegenerateRerunWarning": (_results.DegenerateRerunWarning,
                               _compat.note("nothing raises it")),
    "AmbiguousVariantError": (_AmbiguousVariantError, _compat.use("except ValueError")),
}, removed={
    "plan": ("0.4", _compat.use("mtb.run_all(..., dry_run=True)")),
    "plan_commands": ("0.4", _compat.use("mtb.run_all(..., dry_run=True)")),
    "runtime_hint": ("0.4", _compat.use("mtb.method_info(method)['runtime']")),
})


def __dir__() -> list[str]:
    """``dir(mtb)`` lists ``__all__`` and the dunders only (PEP 562)."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
