"""The tutorials after fix round 6 of the student study.

R6-12 (h): the count line that ``scan`` prints says ``methods`` when each
method has one row ("[scan] 13 of 14 methods have their input files.") and
``rows`` otherwise. The tutorials no longer print a scan table; the guards
below check that line on the tutorials' datasets.
"""
import importlib.util
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


@pytest.mark.parametrize("cat", CATS)
def test_scan_prints_one_count_line_on_the_tutorial_dataset(cat, capsys):
    import multibench as mtb
    ds = GEN.SCEN[cat]["ds"]
    if not (mtb.config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mtb.scan(ds, cat)
    scan_out = capsys.readouterr().out.strip().splitlines()
    assert len(scan_out) == 1 and scan_out[0].startswith("[scan] "), scan_out


@pytest.mark.parametrize("cat,noun", [("vertical", "rows"), ("diagonal", "methods"),
                                      ("mosaic", "rows"), ("cross", "methods")])
def test_scan_count_line_noun_on_the_tutorial_datasets(cat, noun, capsys):
    """The package fact the change follows (R6-12 h): D28 and D52 have one row
    per method, D11 and D46 have methods with several rows."""
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
