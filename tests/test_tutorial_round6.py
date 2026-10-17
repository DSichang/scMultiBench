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
from tests.conftest import full_scan

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
KEYS = list(GEN.TASKS)


@pytest.mark.parametrize("key", KEYS)
def test_scan_prints_one_count_line_on_the_tutorial_dataset(key, capsys):
    import multibench as mtb
    t = GEN.TASKS[key]
    if not (mtb.config.DEFAULT.data_path / t["ds"]).is_dir():
        pytest.skip(f"{t['ds']} is not on disk")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_scan(t["ds"], t["cat"])
    scan_out = capsys.readouterr().out.strip().splitlines()
    assert len(scan_out) == 1 and scan_out[0].startswith("[scan] "), scan_out


#: the tasks whose folder holds two modalities, and for them one listed
#: variant per method: the rows scan lists (0.3.5: the variants whose
#: modalities the folder holds)
ONE_ROW_PER_METHOD = {"vertical_rna_adt": 14, "vertical_rna_atac": 14, "mosaic_rna_adt": 3,
                      "mosaic_rna_atac": 6, "cross_adt_atac": 5}


@pytest.mark.parametrize("key", KEYS)
def test_scan_count_line_noun_on_the_tutorial_datasets(key, capsys):
    """The package fact the change follows (R6-12 h): the line says
    ``methods`` when each method has one row, and ``rows`` otherwise. Since
    the article's tasks, each of the four categories has methods with several
    variants (before: all but diagonal and cross), so with every variant
    listed (``modalities="all"``) every line says ``rows``. By default scan
    lists the variants whose modalities the folder holds, and on the five
    tasks of two modalities that is one row per method."""
    import multibench as mtb
    t = GEN.TASKS[key]
    if not (mtb.config.DEFAULT.data_path / t["ds"]).is_dir():
        pytest.skip(f"{t['ds']} is not on disk")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = full_scan(t["ds"], t["cat"], modalities="all")
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"[scan] {int(frame.files_ok.sum())} of {len(frame)} rows "), line
    assert not frame.method.is_unique
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        listed = full_scan(t["ds"], t["cat"])
    line = capsys.readouterr().out.strip()
    noun = "methods" if key in ONE_ROW_PER_METHOD else "rows"
    assert line.startswith(f"[scan] {int(listed.files_ok.sum())} of {len(listed)} {noun} "), line
    assert listed.method.is_unique == (key in ONE_ROW_PER_METHOD)
    if key in ONE_ROW_PER_METHOD:
        assert len(listed) == ONE_ROW_PER_METHOD[key]
