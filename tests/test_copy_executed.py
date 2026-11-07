"""tools/copy_executed.py: the docs copies show what a reader saw in Jupyter."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import copy_executed as tool  # noqa: E402


def test_a_progress_line_shows_its_last_state():
    assert tool.apply_carriage_returns("a\n\r[env] 0.2 GB\r[env] 2.1 GB\ndone\n") == \
        "a\n[env] 2.1 GB\ndone\n"
    # a shorter rewrite leaves the end of the longer text, as a terminal does
    assert tool.apply_carriage_returns("12345\r89") == "89345"


def test_outputs_of_one_stream_are_joined_and_nothing_else_changes(tmp_path):
    cell = {"cell_type": "code", "source": ["x"], "metadata": {}, "execution_count": 1,
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["a\n", "\rb"]},
                        {"output_type": "stream", "name": "stdout", "text": ["\rc\n"]},
                        {"output_type": "stream", "name": "stderr", "text": ["w\n"]},
                        {"output_type": "execute_result", "data": {"text/plain": ["1"]},
                         "metadata": {}, "execution_count": 1}]}
    nb = {"cells": [cell, {"cell_type": "markdown", "source": ["# T"], "metadata": {}}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    src, dest = tmp_path / "run", tmp_path / "docs"
    src.mkdir(), dest.mkdir()
    (src / "tutorial_demo.ipynb").write_text(json.dumps(nb))
    tool.main([str(src), str(dest)])
    out = json.loads((dest / "demo.ipynb").read_text())
    got = out["cells"][0]["outputs"]
    assert [o.get("name") for o in got] == ["stdout", "stderr", None]
    assert got[0]["text"] == ["a\n", "c\n"]
    assert got[2] == cell["outputs"][3] and out["cells"][1] == nb["cells"][1]
