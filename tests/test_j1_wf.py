"""Journal review, round 1: run, scan and RunResult."""
import ast
from pathlib import Path

import multibench as mtb


def test_engine_and_eval_never_import_workflow():
    """workflow is built on engine and eval; neither imports it back."""
    pkg = Path(mtb.__file__).parent
    for path in sorted([*pkg.glob("engine/*.py"), *pkg.glob("eval/*.py")]):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] if node.module else [a.name for a in node.names]
                top = [n.split(".")[0] for n in names]
                assert not (node.level == 2 and "workflow" in top), (path.name, node.lineno)
                assert not (node.level == 0 and str(node.module).startswith(
                    "multibench.workflow")), (path.name, node.lineno)
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith("multibench.workflow") for a in node.names), \
                    (path.name, node.lineno)
