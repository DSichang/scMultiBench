"""A numpydoc choice set must not show a default the signature does not have.

The API reference is rendered by mkdocstrings (griffe's numpy parser). For a
parameter typed as a choice set, ``name : {"a", "b"}``, griffe takes the first
choice as the default and prints it in the parameter table, whatever the
signature says. ``plot.bar`` showed ``group`` defaulting to "clustering" and
``overall`` to "rank" while the code defaults are ``None`` and
"mean_overall". So the first choice of every set must be the signature
default, and a parameter without a default must not be typed as a choice set.
"""
import ast
import importlib
import inspect
import pkgutil
import re

import pytest

import multibench

# griffe's own pattern (griffe/_internal/docstrings/numpy.py, _RE_PARAMETER):
# the choices are everything between the first "{" and the last "}" of the
# type, and the default is the text before the first ", " inside them
_CHOICE_LINE = re.compile(r"^(?P<names>\w+(?:, \w+)*) : \{(?P<choices>.+)\}")


def _modules():
    for info in pkgutil.walk_packages(multibench.__path__, "multibench."):
        if info.name.endswith("__main__"):
            continue                      # running it would start the CLI
        yield importlib.import_module(info.name)


def _callables():
    """Every function, class and class method defined in the package."""
    seen = set()
    for mod in _modules():
        for name, obj in vars(mod).items():
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            targets = [(f"{mod.__name__}.{name}", obj)]
            if inspect.isclass(obj):
                for an, av in vars(obj).items():
                    if isinstance(av, (staticmethod, classmethod)):
                        av = av.__func__
                    if inspect.isfunction(av):
                        targets.append((f"{mod.__name__}.{name}.{an}", av))
            for path, target in targets:
                if id(target) in seen or not (inspect.isfunction(target)
                                              or inspect.isclass(target)):
                    continue
                seen.add(id(target))
                yield path, target


def _parameter_lines(doc: str):
    """The item lines (``name : type``) of the docstring's Parameters section."""
    lines = inspect.cleandoc(doc).splitlines()
    is_rule = lambda k: k < len(lines) and re.fullmatch(r"-{3,}", lines[k].strip())
    start = next((k + 2 for k, line in enumerate(lines)
                  if line.strip() == "Parameters" and is_rule(k + 1)), None)
    if start is None:
        return
    for k in range(start, len(lines)):
        if is_rule(k + 1):
            return                        # the next section's header
        if lines[k] and not lines[k][0].isspace():
            yield lines[k]


def _literal(text: str):
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return text.strip()


def _mismatches():
    out = []
    for path, obj in _callables():
        doc = obj.__doc__
        if not doc or "Parameters" not in doc:
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        for line in _parameter_lines(doc):
            m = _CHOICE_LINE.match(line)
            if not m:
                continue
            shown = _literal(m.group("choices").split(", ", 1)[0])
            for name in m.group("names").split(", "):
                p = sig.parameters.get(name)
                if p is None:
                    continue
                if p.default is inspect.Parameter.empty:
                    out.append(f"{path}({name}): required, but the choice set "
                               f"renders the default {shown!r}")
                elif shown != p.default:
                    out.append(f"{path}({name}): the reference shows default "
                               f"{shown!r}, the signature has {p.default!r}")
    return out


def test_first_choice_of_every_choice_set_is_the_signature_default():
    assert _mismatches() == []


def test_the_scan_sees_the_known_choice_sets():
    # guard against a scan that silently checks nothing
    seen = {}
    for path, obj in _callables():
        doc = obj.__doc__ or ""
        for line in _parameter_lines(doc) if "Parameters" in doc else ():
            m = _CHOICE_LINE.match(line)
            if m:
                seen.setdefault(path, set()).update(m.group("names").split(", "))
    assert {"group", "overall"} <= seen["multibench.plot.bar.bar"]
    # bubble's __module__ is set to "multibench.plot" (plot/__init__.py)
    assert {"aggregate", "overall", "na"} <= seen["multibench.plot.bubble"]
    assert {"aggregate", "overall", "na"} <= seen["multibench.plot.bubble.build_table"]
    assert "flavor" in seen["multibench.eval.scib.compute"]
    assert "flavor" in seen["multibench.eval.scib.leiden_sweep"]


@pytest.mark.parametrize("name, expected", [("group", None), ("overall", "mean_overall")])
def test_bar_reference_defaults_match_the_code(name, expected):
    griffe = pytest.importorskip("griffe")
    # the package attribute multibench.plot.bar is the function; reach the module
    bar_mod = importlib.import_module("multibench.plot.bar")
    # a docstring without a parent: griffe renders the choice-set default
    ds = griffe.Docstring(inspect.getdoc(bar_mod.bar))
    params = {p.name: p for s in ds.parse("numpy") if s.kind.value == "parameters"
              for p in s.value}
    assert _literal(str(params[name].default)) == expected
    assert inspect.signature(bar_mod.bar).parameters[name].default == expected
