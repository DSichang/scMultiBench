"""Every line of a public docstring's Examples fits in 80 characters.

The API reference cuts code lines off at the right edge of the code box at
1280 px; trailing comments such as ``# no Leiden sweep`` were lost (S4-13).
A long comment goes on its own ``>>> # ...`` line above the code, and a long
call continues on a ``...`` line.
"""
import importlib
import inspect
import pkgutil
import re

import multibench

WIDTH = 80


def _public_objects():
    seen = set()
    for info in pkgutil.walk_packages(multibench.__path__, "multibench."):
        if info.name.endswith("__main__") or any(
                p.startswith("_") for p in info.name.split(".")[1:]):
            continue
        mod = importlib.import_module(info.name)
        for name, obj in vars(mod).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
                continue
            items = [(f"{mod.__name__}.{name}", obj)]
            if inspect.isclass(obj):
                for an, av in vars(obj).items():
                    if an.startswith("_"):
                        continue
                    if isinstance(av, property):
                        av = av.fget
                    elif isinstance(av, (staticmethod, classmethod)):
                        av = av.__func__
                    if inspect.isfunction(av):
                        items.append((f"{mod.__name__}.{name}.{an}", av))
            for path, o in items:
                if (inspect.isfunction(o) or inspect.isclass(o)) and id(o) not in seen:
                    seen.add(id(o))
                    yield path, o


def _examples(doc: str) -> list:
    m = re.search(r"^Examples\n-+\n(.*?)(?=^\S[^\n]*\n-{3,}\n|\Z)", doc, re.S | re.M)
    return m.group(1).splitlines() if m else []


def test_examples_lines_fit_the_code_box():
    long = []
    for path, obj in _public_objects():
        for line in _examples(inspect.getdoc(obj) or ""):
            if len(line) > WIDTH:
                long.append(f"{len(line)} {path}: {line}")
    assert long == []


def test_the_scan_reads_examples():
    paths = {p for p, o in _public_objects() if _examples(inspect.getdoc(o) or "")}
    for want in ("multibench.eval.pipeline.evaluate", "multibench.plot.bar.bar",
                 "multibench.workflow.run_all", "multibench.config.Config",
                 "multibench.workflow.BatchResult.summary"):
        assert want in paths, want
