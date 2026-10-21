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
import os
import json

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.eval import io as eio
from tests.test_study_r4_wf import _quiet, cite  # noqa: F401
from tests.test_study_r5_wf import (BARS, CELLTYPE, KW, SAMPLES, _adata,  # noqa: F401
                                    _barcode_csv, _one_column)

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


def _example(msg: str) -> str:
    """The ``pd.read_csv(...)[...]`` expression a Python error suggests."""
    start = msg.index("pd.read_csv(")
    end = msg.index("]", start) + 1
    return msg[start:end] + (".to_numpy()" if msg[end:].startswith(".to_numpy()") else "")


def test_python_errors_suggest_a_series_that_works(cite, tmp_path):
    a, _ = _h5ad(tmp_path)
    sheet = _sheet(tmp_path / "obs_sheet.csv", list(a.obs_names))
    quoted = json.dumps(str(sheet))
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels="celltype", batch=str(sheet), metrics=METRICS)
    msg = str(e.value)
    assert msg == ("The batch file obs_sheet.csv has several columns after the cell "
                   "ids: celltype, sample. Pass one column, for example "
                   f'pd.read_csv({quoted}, index_col=0)["sample"].')
    # the suggested call, pasted, gives the scores of the obs column
    series = eval(_example(msg), {"pd": pd})
    pd.testing.assert_frame_equal(
        mtb.evaluate(a, labels="celltype", batch=series, metrics=METRICS),
        mtb.evaluate(a, labels="celltype", batch="sample", metrics=METRICS))
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels=str(sheet), metrics=["ASW"])
    assert str(e.value) == ("The labels file obs_sheet.csv has several columns after "
                            "the cell ids: celltype, sample. Pass one column, for "
                            f'example pd.read_csv({quoted}, index_col=0)["celltype"], '
                            "or name the label column x.")
    # run reads the file before any method runs, with the same advice
    data, batch, _ = cite
    (tmp_path / "d").mkdir()
    dsheet = _sheet(tmp_path / "d" / "obs_sheet.csv", BARS, batch)
    with pytest.raises(ValueError, match=r"^The batch file obs_sheet\.csv has several "
                                         r"columns after the cell ids: celltype, sample\. "
                                         r"Pass "
                                         r"one column, for example "
                                         r"pd\.read_csv\("):
        mtb.run(os.path.join(data, 'MYCITE'), 'vertical', dry_run=True, batch=dsheet, **KW)


def test_files_without_an_id_column_and_tsv_files(tmp_path):
    a, _ = _h5ad(tmp_path)
    plain = tmp_path / "meta.csv"
    pd.DataFrame({"celltype": CELLTYPE, "sample": SAMPLES}).to_csv(plain, index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels="celltype", batch=str(plain), metrics=METRICS)
    assert str(e.value) == ("The batch file meta.csv has several columns: celltype, "
                            "sample. "
                            "Pass one column, for example "
                            f'pd.read_csv({json.dumps(str(plain))})["sample"].to_numpy().')
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
    assert f'pd.read_csv({json.dumps(str(rn))})["sample"].to_numpy()' in msg
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
    # c3 (E05): row numbers are no cell ids, and the AnnData has them
    with pytest.raises(ValueError, match=r"^concat\.csv has no cell ids"):
        mtb.evaluate(a, labels="celltype", batch=str(p), metrics=METRICS)


# ==================================================================== R6-15
def test_catalog_lisi_rows_name_the_median():
    d = mtb.catalog.metrics().set_index("metric")["description"]
    for code in ("cLISI", "iLISI"):
        assert "the median over cells, scaled to 0-1" in d[code], code
        assert d[code].endswith("higher = better."), code
        assert "by scib" not in d[code], code
    assert "how much each cell's neighbours share its cell type" in d["cLISI"]
    assert "how many batches appear among each cell's neighbours" in d["iLISI"]
