"""Maintainer tool: copy executed tutorials into the docs, as a reader saw them.

The 13 tutorials are executed on the benchmark host (``jupyter nbconvert
--execute`` of ``notebooks/tutorial_<name>.ipynb``); the docs publish those
copies as ``docs/tutorials/<name>.ipynb``. A saved notebook keeps every
fragment of a progress line that rewrites itself with ``\\r`` and splits one
cell's printing into several outputs; Jupyter shows one line and one block.
This tool applies the carriage returns and joins consecutive outputs of one
stream, so the docs show what the reader saw. Nothing else changes::

    python tools/copy_executed.py <folder of executed tutorial_*.ipynb> <docs>/docs/tutorials
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def apply_carriage_returns(text: str) -> str:
    """``text`` as a terminal shows it: after a ``\\r`` the line is written
    again from its start, over what was there."""
    out = []
    for line in text.split("\n"):
        shown = ""
        for part in line.split("\r"):
            shown = part + shown[len(part):]
        out.append(shown)
    return "\n".join(out)


def clean_outputs(outputs: list) -> list:
    """Consecutive stream outputs of one name joined, carriage returns applied."""
    merged = []
    for o in outputs:
        if (o.get("output_type") == "stream" and merged
                and merged[-1].get("output_type") == "stream"
                and merged[-1].get("name") == o.get("name")):
            merged[-1]["text"] += "".join(o["text"])
        else:
            o = dict(o)
            if o.get("output_type") == "stream":
                o["text"] = "".join(o["text"])
            merged.append(o)
    for o in merged:
        if o.get("output_type") == "stream":
            o["text"] = apply_carriage_returns(o["text"]).splitlines(keepends=True)
    return merged


def main(argv=None) -> int:
    src, dest = (Path(a) for a in (argv or sys.argv[1:]))
    for nb_path in sorted(src.glob("tutorial_*.ipynb")):
        nb = json.loads(nb_path.read_text())
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                cell["outputs"] = clean_outputs(cell.get("outputs", []))
        target = dest / f"{nb_path.stem.removeprefix('tutorial_')}.ipynb"
        target.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
