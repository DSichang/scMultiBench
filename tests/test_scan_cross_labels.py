"""scan's file gate catches a truncated ``cty<i>.csv`` in the numbered (cross /
mosaic) layout.

``_check_label_lengths`` paired label ROLES with modality roles, but no cross
method takes ``cty<i>`` as an input role - only the evaluator reads it - so a
D52 copy with five rows cut from ``cty1.csv`` scanned as ``files_ok=True`` on
every RNA+ADT method and failed hours later, after the run, inside evaluate.
"""
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import resolve


@pytest.fixture
def no_envs(monkeypatch):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


def _h5(path, n_feat, n_cells):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(n_feat, n_cells)).astype(float))
        g.create_dataset("features", data=np.array([f"g{i}" for i in range(n_feat)], dtype="S12"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(n_cells)], dtype="S12"))


def _labels(path, n):
    pd.DataFrame({"x": (["A", "B"] * n)[:n]}).to_csv(path, index=False)


def _numbered(root, name, cells=(40, 50, 60), short=None):
    """Three CITE-seq batches, rna<i>.h5 + adt<i>.h5 + cty<i>.csv; ``short``
    = (batch, rows to drop from that batch's label file)."""
    d = root / name
    d.mkdir(parents=True)
    for i, n in enumerate(cells, start=1):
        _h5(d / f"rna{i}.h5", 30, n)
        _h5(d / f"adt{i}.h5", 10, n)
        _labels(d / f"cty{i}.csv", n - (short[1] if short and short[0] == i else 0))
    return d


def test_intact_numbered_layout_passes(tmp_path, no_envs):
    _numbered(tmp_path, "GOOD")
    df = mtb.scan("GOOD", "cross", data_path=tmp_path)
    rows = df[df["modalities"] != "(data_dir)"]
    assert len(rows) > 0 and rows["files_ok"].all(), rows[["method", "files_reason"]]


def test_truncated_cty_of_one_batch_blocks_every_cross_method(tmp_path, no_envs):
    _numbered(tmp_path, "BADLAB", short=(2, 5))
    df = mtb.scan("BADLAB", "cross", data_path=tmp_path)
    rows = df[df["modalities"] != "(data_dir)"]
    assert len(rows) > 0 and not rows["files_ok"].any()
    for _, r in rows.iterrows():
        why = r["files_reason"]
        # the file, the two counts and the batch index
        assert "cty2.csv has 45 labels" in why and "has 50 cells" in why
        assert "batch 2" in why and "ValueError" in why
        assert "cty2.csv" in r["reason"]           # visible in the short form too


def test_only_the_broken_batch_is_named(tmp_path, no_envs):
    _numbered(tmp_path, "BAD3", short=(3, 1))
    df = mtb.scan("BAD3", "cross", data_path=tmp_path)
    why = df[df["modalities"] != "(data_dir)"].iloc[0]["files_reason"]
    assert "cty3.csv has 59 labels" in why and "batch 3" in why
    assert "cty1.csv" not in why and "cty2.csv" not in why


def test_extra_label_rows_are_caught_too(tmp_path, no_envs):
    _numbered(tmp_path, "LONG", short=(1, -4))          # 4 rows too MANY
    df = mtb.scan("LONG", "cross", data_path=tmp_path)
    why = df[df["modalities"] != "(data_dir)"].iloc[0]["files_reason"]
    assert "cty1.csv has 44 labels" in why and "rna1.h5 has 40 cells" in why


def test_missing_cty_file_is_not_a_length_problem(tmp_path, no_envs):
    """No cty<i>.csv at all: nothing to compare, the gate stays on the files."""
    d = _numbered(tmp_path, "NOCTY")
    for i in (1, 2, 3):
        (d / f"cty{i}.csv").unlink()
    df = mtb.scan("NOCTY", "cross", data_path=tmp_path)
    rows = df[df["modalities"] != "(data_dir)"]
    assert rows["files_ok"].all()


def test_inputs_for_check_true_raises_with_the_same_message(tmp_path):
    _numbered(tmp_path, "BADLAB", short=(1, 5))
    with pytest.raises(ValueError, match=r"cty1\.csv has 35 labels but rna1\.h5 has 40 cells .* batch 1"):
        mtb.inputs_for("BADLAB", "cross", "Concerto", data_path=tmp_path, check=True)
    # default (check=None) and check=False stay silent about content
    mtb.inputs_for("BADLAB", "cross", "Concerto", data_path=tmp_path)


def test_batch_label_file_pairing_rule():
    assert resolve._batch_label_file("rna1", "/d/rna1.h5") == ("1", Path("/d/cty1.csv"))
    assert resolve._batch_label_file("adt12", "/d/adt12.h5") == ("12", Path("/d/cty12.csv"))
    assert resolve._batch_label_file("rna", "/d/rna.h5") is None       # unnumbered
    assert resolve._batch_label_file("cty1", "/d/cty1.csv") is None    # a label role
    assert resolve._batch_label_file("data_dir", "/d/") is None


@pytest.mark.skipif(not (Path(config.DEFAULT.data_path) / "D52" / "cty1.csv").is_file(),
                    reason="reference dataset D52 not fetched")
def test_reference_d52_copy_with_five_rows_cut(tmp_path, no_envs):
    """The exact repro from the re-test: copy D52, cut 5 rows from cty1.csv."""
    src = Path(config.DEFAULT.data_path) / "D52"
    bad = tmp_path / "BADLAB"
    bad.mkdir()
    for f in src.iterdir():
        if f.is_file() and f.name != "cty1.csv":
            os.symlink(f.resolve(), bad / f.name)
    lines = (src / "cty1.csv").read_text().splitlines()
    (bad / "cty1.csv").write_text("\n".join(lines[:-5]) + "\n")
    df = mtb.scan("BADLAB", "cross", data_path=tmp_path)
    rows = df[df["modalities"] != "(data_dir)"]
    assert len(rows) >= 8 and not rows["files_ok"].any()
    assert rows["files_reason"].str.contains(r"cty1\.csv has \d+ labels but rna1\.h5 has \d+ cells").all()
    assert rows["files_reason"].str.contains("batch 1").all()
    # and the untouched reference dataset still passes the label check
    ok = mtb.scan("D52", "cross")
    assert not ok["files_reason"].str.contains("labels but").any()
