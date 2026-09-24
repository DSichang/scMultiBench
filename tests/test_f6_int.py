"""Fix round 6, integration: behaviour that needs two work packages at once.

- eval + wf: ``run-all --batch SHEET --batch-column COL`` saves the column it
  scored with (``batch_<hash>.csv``, the same file a one-column barcode CSV
  gives), and ``rescore()`` reuses it.
- eval + cli: ``evaluate --name`` without ``--method`` takes the batch from a
  sample sheet with ``--batch-column``.
- cli + msg: the dry run of ``multibench run`` prints its header, then the
  missing-input note, then the setup note as a sentence, then the command.
"""
import re
import warnings

import pandas as pd
import pytest

import multibench as mtb
from multibench import cli
from tests.test_f4_cli import _run_all
from tests.test_study_r4_wf import _quiet
from tests.test_study_r5_wf import (BARS, CELLTYPE, SAMPLES, _adata,  # noqa: F401
                                    _barcode_csv, batchy)

pytest.importorskip("scib")


def _sheet(path, index):
    """``mdata.obs.to_csv(path)``: barcodes, then celltype and sample; rows shuffled."""
    pd.DataFrame({"celltype": CELLTYPE, "sample": SAMPLES}, index=index).sample(
        frac=1.0, random_state=11).to_csv(path)
    return path


def test_batch_column_run_saves_the_column_and_rescore_reuses_it(batchy, tmp_path, capsys):
    sheet = _sheet(tmp_path / "obs_sheet.csv", BARS)
    one = _barcode_csv(tmp_path / "one.csv", SAMPLES)
    assert _run_all(batchy, tmp_path / "o1", "--batch", str(sheet),
                    "--batch-column", "sample") == 0
    assert _run_all(batchy, tmp_path / "o2", "--batch", str(one)) == 0
    capsys.readouterr()
    a, b = (mtb.load_batch(tmp_path / o).results[0] for o in ("o1", "o2"))
    # the same cells and batch ids: the same saved file
    assert re.fullmatch(r"batch_[0-9a-f]{8}\.csv", a["batch_file"] or "")
    assert a["batch_file"] == b["batch_file"]
    saved = pd.read_csv(tmp_path / "o1" / a["batch_file"])
    assert list(saved.columns) == ["cell", "batch"] and list(saved["cell"]) == BARS
    assert list(saved["batch"]) == list(SAMPLES)
    # the copy of the sheet that --batch-column reads is gone
    assert sorted(p.name for p in (tmp_path / "o1").glob("*.csv")) == sorted(
        [a["batch_file"], "summary.csv", "long.csv"]
        + [p.name for p in (tmp_path / "o1").glob("failures.csv")])
    before = mtb.load_batch(tmp_path / "o1").summary.iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        again = mtb.load_batch(tmp_path / "o1").rescore().summary.iloc[0]
    assert (again["batch_source"], again["n_batches"]) == ("user", 2)
    for k in ("ASW_batch", "GC", "iLISI"):
        assert (pd.isna(before[k]) and pd.isna(again[k])) or before[k] == again[k], k


def test_name_alone_with_a_batch_column(tmp_path, capsys):
    a = _adata(tmp_path)
    a.write_h5ad(tmp_path / "mine.h5ad")
    sheet = _sheet(tmp_path / "obs_sheet.csv", list(a.obs_names))
    one = _barcode_csv(tmp_path / "one.csv", SAMPLES, index=list(a.obs_names))
    base = ["evaluate", "--output", str(tmp_path / "mine.h5ad"), "--labels",
            str(sheet), "--column", "celltype", "--name", "RNA+ADT PCA", "--dataset",
            "MYCITE", "--category", "vertical", "--metrics", "ASW_batch,iLISI"]
    assert _quiet(cli.main, base + ["--batch", str(sheet), "--batch-column", "sample",
                                    "--out", str(tmp_path / "sheet.csv")]) == 0
    assert _quiet(cli.main, base + ["--batch", str(one),
                                    "--out", str(tmp_path / "one_out.csv")]) == 0
    capsys.readouterr()
    got = pd.read_csv(tmp_path / "sheet.csv")
    want = pd.read_csv(tmp_path / "one_out.csv")
    assert set(got["method"]) == {"RNA+ADT PCA"}
    assert "needs_labels" not in got.columns          # no package method: no L badge
    pd.testing.assert_frame_equal(got, want)


def test_run_dry_run_header_notes_then_command(tmp_path, capsys):
    rc = _quiet(cli.main, ["run", "--method", "GLUE", "--category", "diagonal",
                           "--input", "rna=nope/rna.h5", "--input",
                           "atac_peak=nope/atac_peak.h5", "--out", str(tmp_path / "o"),
                           "--dry-run"])
    assert rc == 0
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "# Dry run. Nothing was executed."
    assert lines[1] == ("# GLUE reads nope/rna.h5, which does not exist. multibench scan "
                        "shows what the folder holds.")
    assert lines[-1] == "# multibench run would execute:"
    setup = [ln for ln in lines if "GENCODE" in ln]
    assert setup and setup[0].startswith("# GLUE needs the GENCODE v43 human annotation")
    assert not [ln for ln in lines if "setup:" in ln]
