"""multibench - run and compare single-cell multimodal integration methods.

Quickstart
----------
::

    import multibench as mtb

    mtb.scan("MYDATA")               # which methods can run on it (and why not the rest)

    res = mtb.run("MYDATA", "vertical", out_dir="out/")
    print(res.summary)               # per method: status, metrics, and why one failed
    res.plot().savefig("compare.png")

The four ``category`` values are ``vertical``, ``diagonal``, ``mosaic`` and
``cross``. The files each one expects in a dataset folder are listed at
https://dsichang.github.io/scMultiBench/tutorials/run/#folder-layout.
"""

from ._settings import config
from . import plot, catalog, io, env, data
from ._data.results import load_results, recommend
from .eval.pipeline import evaluate
from .discover import find_methods, method_info, params_for, cite
from .workflow import scan, run, RunResult, load_run

__version__ = "0.4.0"

__all__ = [
    # namespaces
    "config", "plot", "catalog", "io", "env", "data",
    # methods
    "find_methods", "method_info", "params_for", "cite",
    # run
    "scan", "run", "load_run", "RunResult",
    # metrics
    "evaluate", "load_results", "recommend",
    # version
    "__version__",
]


def __dir__() -> list[str]:
    """``dir(mtb)`` lists ``__all__`` and the dunders only (PEP 562)."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})
