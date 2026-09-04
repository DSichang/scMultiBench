"""The user-facing text matches the 0.3.0 public surface.

Three mechanical checks, so the docs cannot drift back to the 0.2.1 spellings
the surface cut retired: (a) no retired spelling survives in the README, the
notebooks or their generator; (b) every ``mtb.<name>`` the README and the
docs quickstart mention exists in the live package and sits in the relevant
``__all__``; (c) every ``mtb.<fn>(...)`` call in a python fence of the README
and the docs pages binds to the live signature (a keyword that no longer
exists fails here before a reader hits the TypeError).
"""
import ast
import inspect
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))

# The 0.2.1 spellings the 0.3.0 cut retired (deprecated aliases included:
# the docs teach the new name, the alias only keeps old scripts alive).
STALE = {
    r"\bplan_commands\b": "scan()",
    r"\bcommand_preview\b": "run(..., dry_run=True)",
    r"\bruntime_hint\b": "method_info(m)['runtime']",
    r"\bplot_bubble\b": "plot.bubble",
    r"\bfrom_mudata\b": "io.export_dataset(mdata, ...)",
    r"\bwrite_labels\b": "export_dataset writes the label files",
    r"\bslow_metrics\b": "metrics=[..., 'kBET']",
    r"\bmetric_set\b": "removed - the scIB set is the only one",
    r"\bfiles_dir=": "method_info() lost files_dir; catalog.methods() has the columns",
    r"--only\b": "multibench evaluate --metrics",
    r"\bmtb\.plan\(": "mtb.scan(",
    r"\bplan_df\b|\bplan\[plan\.": "scan frame",
    r"env\.(create|create_all|create_env|create_group|freeze|lockfile|group_for|groups|"
    r"default_env_name|own_env_name|installed_envs|required_envs|environment_yml|"
    r"create_commands|group_create_commands|install_packed|packed_sizes|"
    r"host_platform_problem|DIFFICULTY|VERIFIED_STAR)\b": "left env.__all__ in 0.3.0 (status/plan/install/doctor/recipe)",
    r"list_methods\([^)]*\b(task|runnable)=": "find_methods(category, task=/runnable=)",
    r"cite\([^)]*\bmethods=": "cite(*ids) or cite([ids])",
    r"needs_labels=True": "to_long() lost needs_labels; add the column to the frame",
    r"\b(evaluate|load_results|recommend|rescore)\([^)]*\b(task|family|only|method|metric)=": "metrics= / methods=",
    r"\b(inputs_for|labels_for)\(\s*(\"D\d+s?\"|DATASET)\s*,\s*\"(?!vertical|diagonal|mosaic|cross)\w+\"":
        "(dataset, category, method) order",
    r"17 columns|18 columns\b(?! - )|plan_commands adds": "scan's frame is SCAN_COLUMNS + command",
}


def _text(path):
    text = path.read_text()
    if path.suffix == ".ipynb":
        text = "\n".join("".join(c["source"]) for c in json.loads(text)["cells"])
    return text


@pytest.mark.parametrize(
    "path", [ROOT / "README.md", ROOT / "tools" / "gen_tut.py"] + NOTEBOOKS,
    ids=lambda p: p.name)
def test_no_stale_spelling_survives(path):
    text = _text(path)
    for pattern, fix in STALE.items():
        hit = re.search(pattern, text)
        assert hit is None, f"{path.name} still says {hit.group(0)!r}: use {fix}"


# ---- (b) every mtb.<name> mentioned in the README / quickstart exists -------
def _docs_pages(*names):
    root = os.environ.get("SCMULTIBENCH_DOCS")
    if not root or not Path(root).is_dir():
        return []
    return [Path(root) / n for n in names if (Path(root) / n).is_file()]


def _mtb_tokens(text):
    """Dotted names after ``mtb.`` (``mtb.plot.bubble``, ``mtb.config.DEFAULT.data_path``)."""
    return sorted(set(re.findall(r"\bmtb\.([A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*)", text)))


@pytest.mark.parametrize(
    "path", [ROOT / "README.md"] + _docs_pages("quickstart.md"), ids=lambda p: p.name)
def test_every_mtb_name_mentioned_is_public(path):
    import multibench as mtb
    text = _text(path)
    tokens = _mtb_tokens(text)
    assert tokens, f"{path.name} mentions no mtb.<name>?"
    for tok in tokens:
        parts = tok.split(".")
        obj = mtb
        assert parts[0] in mtb.__all__, f"{path.name}: mtb.{parts[0]} is not in mtb.__all__"
        obj = getattr(mtb, parts[0])
        for depth, part in enumerate(parts[1:], start=1):
            assert hasattr(obj, part), f"{path.name}: mtb.{tok} does not exist ({part!r})"
            if depth == 1 and inspect.ismodule(obj) and getattr(obj, "__all__", None) is not None:
                assert part in obj.__all__, \
                    f"{path.name}: mtb.{'.'.join(parts[:2])} is not in {obj.__name__}.__all__"
            obj = getattr(obj, part)


# ---- (c) every mtb.<fn>(...) call in a python fence binds to the live signature
def _python_fences(text):
    """Each python fence as source; a REPL-style fence (``>>>`` prompts) yields
    one statement per prompt, continuation lines joined, output lines dropped."""
    for block in re.findall(r"```python[^\n]*\n(.*?)```", text, flags=re.S):
        if ">>> " not in block:
            yield block
            continue
        stmt = []
        for line in block.splitlines() + [">>> "]:
            if line.startswith(">>> "):
                if stmt:
                    yield "\n".join(stmt)
                stmt = [line[4:]]
            elif line.startswith("... ") and stmt:
                stmt.append(line[4:])
            else:
                if stmt:
                    yield "\n".join(stmt)
                stmt = []


def _dotted(func):
    """``mtb.a.b`` for a plain attribute chain rooted at ``mtb``; None for
    anything else (a call on a call's result, a subscript, ...)."""
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name) and func.id == "mtb":
        return ".".join(["mtb", *reversed(parts)])
    return None


def _calls(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:          # signature blocks (``x: str = ...`` inside a call) and prompts
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name:
                yield name, node


def _bind(fn, node):
    sig = inspect.signature(fn)
    kwargs = {k.arg: object() for k in node.keywords if k.arg}
    if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
        return
    if any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in node.args):
        # ``mtb.run(..., convert=False)``: the docs idiom for "as above"; only
        # the spelled-out keywords can be checked.
        if not any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            unknown = set(kwargs) - set(sig.parameters)
            if unknown:
                raise TypeError(f"unexpected keyword(s) {sorted(unknown)}")
        return
    sig.bind(*([object()] * len(node.args)), **kwargs)


@pytest.mark.parametrize(
    "path",
    [ROOT / "README.md"] + _docs_pages("quickstart.md", "installation.md", "api.md",
                                        "tutorials/discover.md", "tutorials/run.md",
                                        "tutorials/evaluate.md", "tutorials/plot.md"),
    ids=lambda p: p.name)
def test_every_documented_call_binds_to_the_live_signature(path):
    import multibench as mtb
    seen = 0
    for src in _python_fences(_text(path)):
        for name, node in _calls(src):
            obj = mtb
            for part in name.split(".")[1:]:
                assert hasattr(obj, part), f"{path.name}: {name} does not exist"
                obj = getattr(obj, part)
            if not callable(obj) or inspect.isclass(obj):
                continue
            try:
                _bind(obj, node)
            except TypeError as e:
                raise AssertionError(f"{path.name}: {ast.unparse(node)[:100]} -> {e}") from None
            seen += 1
    assert seen, f"{path.name}: no mtb.<fn>(...) call parsed"
