"""Round-5 student study, package work 'cfg': R5-04 and R5-05.

- R5-04: while the method scripts are not fetched, ``mtb.scan`` keeps the
  rows the scripts block runnable.
- R5-05: a failed download in ``mtb.data.fetch`` / ``fetch_outputs`` raises
  ``OSError`` naming the dataset, the URL and the manual route, and leaves
  nothing under ``data_path``. The progress line names the URL.
"""
import importlib

import multibench as mtb
from multibench.engine import envs, registry

F = importlib.import_module("multibench._data.fetch")
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


def test_raises_tables_list_the_download_error():
    import inspect
    for fn in (mtb.data.fetch,):
        raises = inspect.getdoc(fn).split("Raises\n------\n")[1].split("\n\n")[0]
        assert "OSError\n    The download failed; the message names the URL." in raises, \
            fn.__name__


