"""scib 1.1.7 calls ``warnings.simplefilter("default")`` when it is imported,
which shows every warning of the session from then on: on Colab (Python 3.13)
the output of ``run`` filled with jupyter_client's DeprecationWarning after the
first method was scored. The package imports scib through ``import_scib``,
which keeps the caller's filters."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scib")

PROBE = """
import warnings
before = list(warnings.filters)
from multibench.eval import pipeline, scib as s
s.import_scib()
s._lisi_helper_problem()
pipeline._scib_version()
# a new rule that shows every DeprecationWarning, from any module
print([f[0] for f in warnings.filters if f not in before and f[0] != "ignore"
       and issubclass(DeprecationWarning, f[2]) and f[1] is None and f[3] is None])
"""


def test_scoring_imports_leave_the_warning_filters_alone():
    # a fresh interpreter: an earlier test may have imported scib already
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                         check=True, cwd=Path(__file__).resolve().parents[1])
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_only_import_scib_imports_scib():
    root = Path(__file__).resolve().parents[1] / "multibench"
    direct = [f"{p.relative_to(root)}:{i}"
              for p in root.rglob("*.py") if p != root / "eval" / "scib.py"
              for i, line in enumerate(p.read_text().splitlines(), 1)
              if re.match(r"\s*(import|from) scib\b", line)]
    assert direct == []
