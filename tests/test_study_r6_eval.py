"""Fix round 6, work package 'eval'.

R6-03: a barcode-keyed sample sheet with several columns (``mdata.obs.to_csv()``)
works as a batch file. ``--batch-column NAME`` on ``evaluate`` and ``run-all``
picks its column, as ``--column`` does for ``--labels``. Without a chosen
column, the error lists the columns after the cell ids and says how to choose
one (the flag on the command line, a Series in Python), without pandas'
'Unnamed: 0'. A two-column file whose first column has no header is read as
an index and its other column.
R6-15: the catalog's cLISI and iLISI rows name the median over cells.
"""
import json

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli
from multibench.eval import io as eio
from tests.test_f4_cli import _run_all
from tests.test_study_r4_wf import _quiet, cite  # noqa: F401
from tests.test_study_r5_wf import (BARS, CELLTYPE, KW, SAMPLES, _adata,  # noqa: F401
                                    _barcode_csv, _one_column, batchy)

pytest.importorskip("scib")

METRICS = ["ASW_batch", "iLISI"]


def _sheet(path, index, batch=SAMPLES, **kw):
    """``mdata.obs.to_csv(path)``: barcodes without a header, then celltype
    and sample; rows shuffled."""
    pd.DataFrame({"celltype": CELLTYPE, "sample": batch}, index=index).sample(
        frac=1.0, random_state=11).to_csv(path, **kw)
    return path


def _h5ad(tmp_path):
    a = _adata(tmp_path)
    a.write_h5ad(tmp_path / "mine_emb.h5ad")
    return a, tmp_path / "mine_emb.h5ad"


def _evaluate(h5ad, *flags):
    return cli.main(["evaluate", "--output", str(h5ad), *flags,
                     "--metrics", ",".join(METRICS)])


def _values(path):
    # iLISI is NaN where the LISI backend is unavailable
    return pd.read_csv(path, index_col=0)["Value"].round(6).fillna(-1).to_dict()


def _dry_run(data, *flags):
    return cli.main(["run-all", "MYCITE", "--category", "vertical", "--methods", "Matilda",
                     "--modalities", "rna,adt", "--data-path", str(data), "--dry-run",
                     *flags])


# ============================================================ --batch-column
def test_evaluate_batch_column_scores_as_a_one_column_barcode_file(tmp_path):
    a, h5ad = _h5ad(tmp_path)
    ids = list(a.obs_names)
    sheet = _sheet(tmp_path / "obs_sheet.csv", ids)
    one = _barcode_csv(tmp_path / "b1.csv", SAMPLES, index=ids)
    lab = _one_column(tmp_path / "cty.csv", CELLTYPE)
    runs = {"one": ["--labels", str(lab), "--batch", str(one)],
            "sheet": ["--labels", str(lab), "--batch", str(sheet),
                      "--batch-column", "sample"],
            # S2's whole call: one sheet for both, each flag with its column
            "both": ["--labels", str(sheet), "--column", "celltype",
                     "--batch", str(sheet), "--batch-column", "sample"]}
    for tag, flags in runs.items():
        assert _quiet(_evaluate, h5ad, *flags, "--out", str(tmp_path / tag)) == 0, tag
    want = _values(tmp_path / "one")
    assert _values(tmp_path / "sheet") == want
    assert _values(tmp_path / "both") == want
    assert want["ASW_batch"] < 0.5             # the file sees the sample shift
    py = mtb.evaluate(a, labels="celltype", batch="sample", metrics=METRICS)
    assert want == py["Value"].round(6).fillna(-1).to_dict()


def test_run_all_batch_column_dry_run_and_run(batchy, tmp_path, capsys):
    sheet = _sheet(tmp_path / "obs_sheet.csv", BARS)
    one = _barcode_csv(tmp_path / "b1.csv", SAMPLES)
    assert _dry_run(batchy, "--batch", str(sheet), "--batch-column", "sample") == 0
    assert _run_all(batchy, tmp_path / "o1", "--batch", str(sheet),
                    "--batch-column", "sample") == 0
    assert _run_all(batchy, tmp_path / "o2", "--batch", str(one)) == 0
    capsys.readouterr()
    a = mtb.load_batch(tmp_path / "o1").summary.iloc[0]
    b = mtb.load_batch(tmp_path / "o2").summary.iloc[0]
    assert a["status"] == b["status"] == "CHAIN_OK" and a["n_batches"] == 2
    for k in ("ASW_batch", "GC", "iLISI"):
        assert (pd.isna(a[k]) and pd.isna(b[k])) or a[k] == b[k], k


@pytest.mark.parametrize("command", ["evaluate", "run-all"])
def test_batch_column_without_batch_is_a_usage_error(command, tmp_path, capsys):
    argv = (["evaluate", "--output", str(tmp_path / "x.h5"), "--labels",
             str(tmp_path / "cty.csv")] if command == "evaluate"
            else ["run-all", "MYCITE", "--category", "vertical", "--dry-run"])
    with pytest.raises(SystemExit) as e:
        cli.main(argv + ["--batch-column", "sample"])
    assert e.value.code == 2
    assert "--batch-column needs --batch" in capsys.readouterr().err


def test_batch_column_help_is_one_line(capsys):
    for command in ("evaluate", "run-all"):
        with pytest.raises(SystemExit):
            cli.main([command, "--help"])
        text = " ".join(capsys.readouterr().out.split())
        assert ("--batch-column NAME the column to read in the --batch CSV when it has "
                "several columns") in text, command


# ============================================================ the error text
def test_a_batch_sheet_without_a_column_names_the_flag(cite, tmp_path, capsys):
    a, h5ad = _h5ad(tmp_path)
    sheet = _sheet(tmp_path / "obs_sheet.csv", list(a.obs_names))
    lab = _one_column(tmp_path / "cty.csv", CELLTYPE)
    assert _evaluate(h5ad, "--labels", str(lab), "--batch", str(sheet)) == 1
    err = capsys.readouterr().err
    assert ("error: The --batch file obs_sheet.csv has several columns after the cell "
            "ids: celltype, sample. Choose one with --batch-column.") in err
    assert "Unnamed" not in err and "named x" not in err
    # run-all reads the file in its dry run and says the same
    data, batch, _ = cite
    (tmp_path / "d").mkdir()
    dsheet = _sheet(tmp_path / "d" / "obs_sheet.csv", BARS, batch)
    assert _dry_run(data, "--batch", str(dsheet)) == 1
    err = capsys.readouterr().err
    assert ("error: The --batch file obs_sheet.csv has several columns after the cell "
            "ids: celltype, sample. Choose one with --batch-column.") in err
    assert "Unnamed" not in err
    # a label file keeps pointing to --column and to the label column x
    assert _evaluate(h5ad, "--labels", str(sheet), "--batch", str(sheet),
                     "--batch-column", "sample") == 1
    err = capsys.readouterr().err
    assert ("error: The --labels file obs_sheet.csv has several columns after the cell "
            "ids: celltype, sample. Choose one with --column, or name the label column "
            "x.") in err
    assert "Unnamed" not in err


def test_an_unknown_batch_column_lists_the_columns(tmp_path, capsys):
    a, h5ad = _h5ad(tmp_path)
    sheet = _sheet(tmp_path / "obs_sheet.csv", list(a.obs_names))
    lab = _one_column(tmp_path / "cty.csv", CELLTYPE)
    assert _evaluate(h5ad, "--labels", str(lab), "--batch", str(sheet),
                     "--batch-column", "donor") == 1
    err = capsys.readouterr().err
    assert ("error: The --batch file obs_sheet.csv has no column named 'donor'. Its "
            "columns after the cell ids are celltype, sample.") in err
    assert "error: batch:" not in err
    assert "Unnamed" not in err


def _example(msg: str) -> str:
    """The ``pd.read_csv(...)[...]`` expression a Python error suggests."""
    start = msg.index("pd.read_csv(")
    end = msg.index("]", start) + 1
    return msg[start:end]


def test_python_errors_suggest_a_series_that_works(cite, tmp_path):
    a, _ = _h5ad(tmp_path)
    sheet = _sheet(tmp_path / "obs_sheet.csv", list(a.obs_names))
    quoted = json.dumps(str(sheet))
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels="celltype", batch=str(sheet), metrics=METRICS)
    msg = str(e.value)
    assert msg == ("The batch file obs_sheet.csv has several columns after the cell "
                   "ids: celltype, sample. Pass one column as a Series, for example "
                   f'pd.read_csv({quoted}, index_col=0)["sample"].')
    # the suggested call, pasted, gives the scores of the obs column
    series = eval(_example(msg), {"pd": pd})
    pd.testing.assert_frame_equal(
        mtb.evaluate(a, labels="celltype", batch=series, metrics=METRICS),
        mtb.evaluate(a, labels="celltype", batch="sample", metrics=METRICS))
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels=str(sheet), metrics=["ASW"])
    assert str(e.value) == ("The labels file obs_sheet.csv has several columns after "
                            "the cell ids: celltype, sample. Pass one column as a "
                            "Series, for "
                            f'example pd.read_csv({quoted}, index_col=0)["celltype"], '
                            "or name the label column x.")
    # run_all reads the file before any method runs, with the same advice
    data, batch, _ = cite
    (tmp_path / "d").mkdir()
    dsheet = _sheet(tmp_path / "d" / "obs_sheet.csv", BARS, batch)
    with pytest.raises(ValueError, match=r"^The batch file obs_sheet\.csv has several "
                                         r"columns after the cell ids: celltype, sample\. "
                                         r"Pass "
                                         r"one column as a Series, for example "
                                         r"pd\.read_csv\("):
        mtb.run_all("MYCITE", "vertical", dry_run=True, batch=dsheet, data_path=data,
                    **KW)


def test_files_without_an_id_column_and_tsv_files(tmp_path):
    a, _ = _h5ad(tmp_path)
    plain = tmp_path / "meta.csv"
    pd.DataFrame({"celltype": CELLTYPE, "sample": SAMPLES}).to_csv(plain, index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels="celltype", batch=str(plain), metrics=METRICS)
    assert str(e.value) == ("The batch file meta.csv has several columns: celltype, "
                            "sample. "
                            "Pass one column as a Series, for example "
                            f'pd.read_csv({json.dumps(str(plain))})["sample"].')
    tsv = _sheet(tmp_path / "obs.tsv", list(a.obs_names), sep="\t")
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels="celltype", batch=str(tsv), metrics=METRICS)
    msg = str(e.value)
    assert f'pd.read_csv({json.dumps(str(tsv))}, sep="\\t", index_col=0)["sample"]' in msg
    assert list(eval(_example(msg), {"pd": pd}).index[:2]) != [0, 1]
    # R's row numbers before the columns: the example keeps the rows by position
    rn = tmp_path / "rn.csv"
    pd.DataFrame({"celltype": CELLTYPE, "sample": SAMPLES},
                 index=range(1, len(SAMPLES) + 1)).to_csv(rn)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels="celltype", batch=str(rn), metrics=METRICS)
    msg = str(e.value)
    assert msg.startswith("The batch file rn.csv has several columns after the row "
                          "numbers: celltype, sample. ")
    assert f'pd.read_csv({json.dumps(str(rn))})["sample"]' in msg
    pd.testing.assert_frame_equal(
        mtb.evaluate(a, labels="celltype", batch=eval(_example(msg), {"pd": pd}),
                     metrics=METRICS),
        mtb.evaluate(a, labels="celltype", batch=SAMPLES, metrics=METRICS))


def test_read_labels_error_names_column_and_no_pandas_header(tmp_path):
    p = tmp_path / "meta.csv"
    pd.DataFrame({"celltype": CELLTYPE, "sample": SAMPLES},
                 index=[f"bc{i}" for i in range(len(SAMPLES))]).to_csv(p)
    with pytest.raises(ValueError) as e:
        eio.read_labels(p)
    assert str(e.value) == (f"{p} has several columns after the cell ids: celltype, "
                            f"sample. Pass column=<name>.")
    with pytest.raises(ValueError) as e:
        eio.read_labels(p, column="donor")
    assert str(e.value) == (f"{p} has no column named 'donor'. Its columns after the "
                            f"cell ids are celltype, sample.")


def test_a_headerless_first_column_with_repeats_is_an_index(tmp_path):
    """pd.concat of two per-sample tables, written with their RangeIndex."""
    a, _ = _h5ad(tmp_path)
    half = len(SAMPLES) // 2
    p = tmp_path / "concat.csv"
    pd.concat([pd.DataFrame({"sample": SAMPLES[:half]}),
               pd.DataFrame({"sample": SAMPLES[half:]})]).to_csv(p)
    assert pd.read_csv(p).columns[0] == "Unnamed: 0"
    assert list(eio.read_labels(p)) == list(SAMPLES)
    pd.testing.assert_frame_equal(
        mtb.evaluate(a, labels="celltype", batch=str(p), metrics=METRICS),
        mtb.evaluate(a, labels="celltype", batch=SAMPLES, metrics=METRICS))


# ==================================================================== R6-15
def test_catalog_lisi_rows_name_the_median():
    d = mtb.catalog.metrics().set_index("metric")["description"]
    for code in ("cLISI", "iLISI"):
        assert "the median over cells, scaled to 0-1" in d[code], code
        assert d[code].endswith("higher = better."), code
        assert "by scib" not in d[code], code
    assert "how much each cell's neighbours share its cell type" in d["cLISI"]
    assert "how many batches appear among each cell's neighbours" in d["iLISI"]
