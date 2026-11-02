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

from . import _text
from ._settings import config
from . import plot, catalog, io, env, data
from ._data.results import load_results, rank
from .eval.pipeline import evaluate
from .discover import find_methods, method_info, params_for
from .workflow import scan, run, RunResult, load_run

__version__ = "0.5.2"

__all__ = [
    # namespaces
    "config", "plot", "catalog", "io", "env", "data",
    # methods
    "find_methods", "method_info", "params_for",
    # run
    "scan", "run", "load_run", "RunResult",
    # metrics
    "evaluate", "load_results", "rank",
    # version
    "__version__",
]

__dir__ = _text.public_dir(globals())
