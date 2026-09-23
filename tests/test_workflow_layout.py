"""``describe_layout`` per category (study round 1, L33 / L34).

Each category prints only its own files, one name per file, the ATAC rule,
and for mosaic / cross the batch patterns the registry declares.
"""
import inspect
import re
from pathlib import Path

import pytest

import multibench as mtb
from multibench import config, workflow as W

CATS = ("vertical", "diagonal", "mosaic", "cross")


@pytest.mark.parametrize("cat", CATS)
def test_one_category_prints_only_that_category(cat):
    txt = mtb.describe_layout(cat)
    n = len(txt.splitlines())
    assert 15 <= n <= 30, n                                # was about 65 lines
    others = [c for c in CATS if c != cat]
    assert f"\n{cat}: " in txt
    assert not any(f"\n{c}: " in txt for c in others)
    assert "Every file name the loader reads" not in txt    # the full table: overview only
    # the last line runs scan on this category, and the install line is per method
    assert txt.splitlines()[-1].startswith(f"Next: mtb.scan('MYDATA', '{cat}')")
    assert "multibench env install --methods X --packed --run" in txt
    assert "lockfiles" not in txt and "two gates" not in txt
    assert "mtb.scan checks the files and the environment for each method" in txt


@pytest.mark.parametrize("cat", ["vertical", "diagonal"])
def test_numbered_files_only_for_batch_categories(cat):
    assert "batch column" not in mtb.describe_layout(cat)
    assert "rna1" not in mtb.describe_layout(cat)


@pytest.mark.parametrize("cat", ["mosaic", "cross"])
def test_batch_categories_explain_numbered_files(cat):
    txt = mtb.describe_layout(cat)
    assert "cty1.csv" in txt and "one label file per batch" in txt


def test_one_peak_file_name_per_category():
    """describe_layout('diagonal') used to give three peak file names."""
    diag = mtb.describe_layout("diagonal")
    bare_peak = re.findall(r"(?<![\w_])peak\.h5", diag)
    assert len(bare_peak) == 1                          # only in the older-names line
    assert "Older names still read: peak.h5 for peaks, and atac.h5 for gene activity." in diag
    vert = mtb.describe_layout("vertical")
    assert not re.search(r"(?<![\w_])peak\.h5", vert) and "atac_peak.h5" not in vert
    assert "atac.h5 holds peaks or gene activity" in vert


def test_mosaic_lists_every_registry_pattern_with_methods_and_demo():
    txt = mtb.describe_layout("mosaic")
    assert "Number your batches to match one of them" in txt
    lines = txt.splitlines()
    pats = W.batch_patterns("mosaic")
    assert len(pats) >= 2
    for roles, ids in pats:
        line = next(l for l in lines if l.strip().startswith("batch 1 =")
                    and l.rstrip().split(": ", 1)[-1].split(" (demo")[0] == ", ".join(ids))
        assert line
    d45 = next(l for l in lines if "(demo D45)" in l)
    d46 = next(l for l in lines if "(demo D46)" in l)
    assert "Cobolt" in d45 and "MultiVI" in d45 and "atac" in d45
    assert "StabMap" in d46 and "scMoMaT" in d46 and "2 = rna+atac, 3 = rna" in d46
    # the per-batch recipe (one file per batch) and its Python form
    assert txt.count("--batch-index") == 3 and "batch_index=1" in txt
    assert "atac<i>.h5 holds peaks" in txt


def test_cross_patterns_come_from_the_registry():
    txt = mtb.describe_layout("cross")
    for roles, ids in W.batch_patterns("cross"):
        assert ", ".join(ids) in txt
    assert "(demo D52)" in txt and "--atac" not in txt       # cross is RNA + ADT only


def test_demo_table_matches_the_demo_folders():
    base = Path(config.DEFAULT.data_path)
    checked = 0
    for ds, roles in W.DEMO_FILES.items():
        d = base / ds
        if not d.is_dir():
            continue
        on_disk = {p.stem for p in d.glob("*.h5")}
        assert on_disk == set(roles), ds
        checked += 1
    assert checked or not base.is_dir(), "no demo folder to check the table against"


def test_overview_keeps_every_category_and_the_full_table():
    txt = mtb.describe_layout()
    for c in CATS:
        assert f"\n{c}: " in txt
    assert "Every file name the loader reads:" in txt
    for k in W.ROLES:
        assert k in txt
    assert "mtb.scan('MYDATA', '<category>')" in txt


def test_one_atac_rule_in_run_all_and_describe_layout():
    rule = ("Vertical reads ``atac.h5``; ``method_info(m)[\"atac\"]`` says whether it "
            "must hold peaks or gene activity. Diagonal reads ``atac_peak.h5`` (peaks) "
            "and ``atac_gas.h5`` (gene activity). Mosaic reads ``atac<i>.h5`` (peaks). "
            "``peak.h5``, and ``atac.h5`` for gene activity, are accepted as older names.")
    for fn in (mtb.run_all, mtb.describe_layout):
        flat = " ".join(inspect.getdoc(fn).split())
        assert rule in flat, fn.__name__
    flat = " ".join(inspect.getdoc(mtb.run_all).split())
    assert "plausible" not in flat and "peaks are ``peak.h5``" not in flat
