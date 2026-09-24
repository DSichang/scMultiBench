"""The tutorials after fix round 6 of the student study.

R6-12 (h): the count line that ``scan`` prints says ``methods`` when each
method has one row ("[scan] 13 of 14 methods have their input files.") and
``rows`` otherwise. The two scan cells of each category tutorial then printed
a count line of their own, "files_ok 13, env_ok 0, runnable 0 of 14 method
variants": the file and environment counts a second time, under a third
noun. They now print only the count that scan's line lacks,
"runnable: 0 of 14".

The other round-6 changes leave the notebooks' calls and prose true; the
guards below check the facts the prose states on the live package.
"""
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
CATS = list(GEN.SCEN)


def _code(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _print_lines(src):
    """The lines of a cell that print from the scan frame ``sc`` or ``avail``."""
    return [ln for ln in src.splitlines()
            if ln.startswith("print(") and re.search(r"\b(sc|avail)\.", ln)]


def _scan_cells(cat):
    """The code cells of the tutorial that bind a scan frame and print a count
    from it: section 3 (``sc``, the user's folder) and section 5 (``avail``)."""
    return [src for src in _code(f"tutorial_{cat}")
            if re.search(r"^(sc|avail) = mtb\.scan\(", src, re.M) and _print_lines(src)]


@pytest.mark.parametrize("cat", CATS)
def test_each_scan_cell_prints_one_count_line_of_its_own(cat):
    cells = _scan_cells(cat)
    assert len(cells) == 2, f"tutorial_{cat}: expected the section-3 and section-5 scan cells"
    for src in cells:
        assert len(_print_lines(src)) == 1, src


@pytest.mark.parametrize("cat", CATS)
def test_scan_cells_add_the_runnable_count_without_repeating_scan_line(cat, capsys):
    """Run each scan cell's print line on the scan of the tutorial's dataset:
    scan's own line carries the file and environment counts in its noun, and
    the cell adds only ``runnable: k of n``."""
    import multibench as mtb
    ds = GEN.SCEN[cat]["ds"]
    if not (mtb.config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = mtb.scan(ds, cat)
    scan_out = capsys.readouterr().out.strip().splitlines()
    assert len(scan_out) == 1 and scan_out[0].startswith("[scan] "), scan_out
    for src in _scan_cells(cat):
        exec(_print_lines(src)[0], {"sc": frame, "avail": frame})
        line = capsys.readouterr().out.strip()
        assert line == f"runnable: {int(frame.runnable.sum())} of {len(frame)}", line
        # nothing that scan's line already says, and no second noun for its rows
        assert "files_ok" not in line and "env_ok" not in line, line
        assert "variant" not in line and "method" not in line, line


@pytest.mark.parametrize("cat,noun", [("vertical", "rows"), ("diagonal", "methods"),
                                      ("mosaic", "rows"), ("cross", "methods")])
def test_scan_count_line_noun_on_the_tutorial_datasets(cat, noun, capsys):
    """The package fact the change follows (R6-12 h): D28 and D52 have one row
    per method, D11 and D45 have methods with several rows."""
    import multibench as mtb
    ds = GEN.SCEN[cat]["ds"]
    if not (mtb.config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = mtb.scan(ds, cat)
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"[scan] {int(frame.files_ok.sum())} of {len(frame)} {noun} "), line
    assert frame.method.is_unique == (noun == "methods")
