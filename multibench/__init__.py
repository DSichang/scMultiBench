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
from .data.results import load_results, recommend
from .eval.pipeline import evaluate
from .engine import ingest as io
from .engine import envs as env
from .discover import find_methods, method_info, params_for, cite
from .workflow import scan, run_all, BatchResult, describe_layout, load_batch

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


def __dir__() -> list[str]:
    """``dir(mtb)`` lists ``__all__`` and the dunders only (PEP 562)."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
