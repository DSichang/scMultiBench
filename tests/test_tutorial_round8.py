"""The tutorials after fix round 8 of the student study.

Round 8 split the ';' joins in the prose of the docs tutorial pages (R8-09)
and took ';' out of the package messages (R8-07). The notebook tutorials
render on the same site, so their prose and the lines their cells print
follow the same rule. Code and code comments are left as they are.

R8-04 makes ``scan`` and ``run_all`` raise ``ValueError`` when ``methods=``
names a method with no variant in the category, where they used to drop it.
The run cells of the task tutorials pass a fixed method list to
``run_all``. The guard below checks that list against the live package, so a
list that would now raise fails here instead of in a Run all.
"""
import ast
import importlib.util
import json
import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
KEYS = list(GEN.TASKS)
GENERATED = [f"tutorial_{k}{suffix}" for k in KEYS for suffix in ("", "_all")
             if not suffix or GEN.has_all(k)]
NOTEBOOKS = GENERATED + ["colab_quickstart", "tutorial_end_to_end"]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _prose(md):
    """Markdown without fenced code blocks and with each code span blanked."""
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    return re.sub(r"`[^`\n]*`", "`x`", md)


def _printed_strings(src):
    """The string parts of every ``print(...)`` argument in a code cell."""
    ipy = pytest.importorskip("IPython.core.inputtransformer2")
    tree = ast.parse(ipy.TransformerManager().transform_cell(src))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        out.append(sub.value)
    return out


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ------------------------------------------------ R8-09 style: no ';' joins
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_prose_joins_no_two_facts_with_a_semicolon(name):
    hits = [ln for kind, src in _cells(name) if kind == "markdown"
            for ln in _prose(src).splitlines() if ";" in ln]
    assert not hits, f"{name}: split these sentences at the ';': {hits}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_printed_lines_hold_no_semicolon(name):
    hits = [s for kind, src in _cells(name) if kind == "code"
            for s in _printed_strings(src) if ";" in s]
    assert not hits, f"{name}: a printed line joins two facts with ';': {hits}"


def test_split_sentences_keep_their_facts():
    """The sentences split at the ';' keep both halves."""
    for name in GENERATED:
        md = "\n".join(src for kind, src in _cells(name) if kind == "markdown")
        assert "Circle size shows the rank within a column, and bigger is better." in md, name
    for key in KEYS:
        if GEN.TASKS[key]["cat"] != "mosaic":
            continue
        md = "\n".join(src for kind, src in _cells(f"tutorial_{key}") if kind == "markdown")
        assert "writes one batch per call. Number the batches as below: " in md, key
        assert "`mtb.describe_layout(\"mosaic\")` lists the batch patterns" in md, key


# ------------------------------------------ R8-04 guards: the named methods
def _method_list(name):
    """The list a notebook assigns to ``METHODS``."""
    found = []
    for kind, src in _cells(name):
        if kind != "code" or "METHODS = " not in src:
            continue
        ipy = pytest.importorskip("IPython.core.inputtransformer2")
        for node in ast.walk(ast.parse(ipy.TransformerManager().transform_cell(src))):
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "METHODS":
                found.append(ast.literal_eval(node.value))
    assert len(found) == 1, name
    return found[0]


@pytest.mark.parametrize("key", KEYS)
def test_run_cells_name_only_methods_with_a_variant_on_their_dataset(key):
    """The run cell passes the notebook's method list to run_all on its
    dataset. Since R8-04 a listed method with no variant there raises
    ValueError, so every listed method must get a row. The tutorial lists its
    default methods, and the ``_all`` notebook every method of the task."""
    import multibench as mtb
    t = GEN.TASKS[key]
    cat, ds = t["cat"], t["ds"]
    lists = {f"tutorial_{key}": t["methods"]}
    if GEN.has_all(key):
        lists[f"tutorial_{key}_all"] = GEN.task_methods(key)
    for name, methods in lists.items():
        assert _method_list(name) == methods, name
    assert set(t["methods"]) <= set(GEN.task_methods(key))
    if not (mtb.config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")
    for name, methods in lists.items():
        sc = _quiet(mtb.scan, ds, cat, methods=methods, verbose=False)
        assert set(sc.method) == set(methods), (name, ds)
    # the check the guard protects against: an off-category name raises
    methods = t["methods"]
    other = next(m for m in mtb.list_methods() if cat not in mtb.method_info(m)["categories"])
    with pytest.raises(ValueError, match=rf"{re.escape(other)} does not run on {cat} data"):
        _quiet(mtb.scan, ds, cat, methods=methods + [other], verbose=False)
