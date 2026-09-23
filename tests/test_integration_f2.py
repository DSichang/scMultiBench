"""Checks that guard the merge of the round-2 work packages."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "multibench"


def _module_imports(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def _own_nodes(fn):
    """The nodes of ``fn``'s body, without nested functions and classes."""
    for child in ast.iter_child_nodes(fn):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                              ast.ClassDef)):
            continue
        yield child
        yield from _own_nodes(child)


def test_no_module_level_name_is_used_before_a_local_reimport():
    """A function-local ``from .. import config`` next to a module-level one makes
    ``config`` local to the whole function, so a use above that import raises
    UnboundLocalError. Two branches merged that way broke export_dataset."""
    found = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text())
        top = _module_imports(tree)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nodes = list(_own_nodes(fn))
            first = {}
            for node in nodes:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        name = (a.asname or a.name).split(".")[0]
                        if name in top:
                            first[name] = min(first.get(name, node.lineno), node.lineno)
            for node in nodes:
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and node.id in first and node.lineno < first[node.id]):
                    found.append(f"{path.relative_to(PKG.parent)}:{node.lineno} "
                                 f"{fn.name} uses {node.id} above its local import "
                                 f"(line {first[node.id]})")
    assert not found, "\n".join(found)
