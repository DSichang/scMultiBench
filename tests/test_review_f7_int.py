"""Review of round 7 (integration): the findings applied in the package.

- R7-10: on Linux, scan's summary line names the doctor check when an
  environment is missing; the category sentence of the env reason starts
  with 'Use'.
- R7-11: the reason column drops '<method> reads <file> of <dataset>, which'.
- A missing batch file is called a batch file, in sentences.
- The several-datasets warning of the plots lists its datasets in words.
- The no-conda install error, the modalities warning of scan, the missing
  feature-names warning and the several-variants errors are sentences.
- Docstrings: RunResult.long, method_info, bubble's L badge,
  normalize_peak_names' example.
"""
import inspect
import re
import shutil
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry
from multibench.engine import runner as R
from tests.test_docs_r4 import _flat, needs_docs
from tests.test_study_r7_msg import ALL_ENVS, GENES, _h5, _lung, _quiet
from tests.conftest import full_scan

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def test_env_reason_names_the_category_in_a_sentence(monkeypatch):
    monkeypatch.setattr(R, "linux_only_sentence", lambda: None)
    text = W._env_hint("glue", "GLUE", "diagonal")
    assert text == ("Environment glue is not installed. Run mtb.env.install(['GLUE']). "
                    "mtb.env.install(category='diagonal') installs the environments of "
                    "every diagonal method.")


# ======================================================================= R7-11
def test_reason_column_starts_with_the_file_of_a_cell_check(tmp_path):
    d = _lung(tmp_path)
    _h5(d / "atac_gas.h5", GENES, [f"a{i}" for i in reversed(range(20))])
    row = _quiet(full_scan, "LUNG", "diagonal", methods=["SCALEX"], data_path=tmp_path,
                 verbose=False).iloc[0]
    assert row["reason"].startswith("atac_gas.h5 lists the ATAC cells in another order "
                                    "than atac_peak.h5."), row["reason"]
    assert "than atac_peak.h5" in row["reason"][:80]
    assert "SCALEX reads" not in row["reason"]
    # files_reason keeps the exception verbatim
    assert row["files_reason"].startswith("ValueError: SCALEX reads atac_gas.h5 of LUNG, "
                                          "which lists")


def test_reason_column_of_the_other_cells_and_orientation_checks():
    other = ("ValueError: SCALEX reads atac_gas.h5 of LUNG, which holds other cells than "
             "atac_peak.h5. The files hold 18 and 20 cells and share 0.")
    assert W._short_reason(other, "SCALEX", "LUNG").startswith(
        "atac_gas.h5 holds other cells than atac_peak.h5.")
    turned = ("ValueError: totalVI reads rna.h5 of TRANS, which stores matrix/data as "
              "cells x features, shape (20, 60).")
    assert W._short_reason(turned, "totalVI", "TRANS") == (
        "rna.h5 stores matrix/data as cells x features, shape (20, 60).")
    # another method or dataset is left alone
    assert W._short_reason(other, "GLUE", "LUNG").startswith("SCALEX reads")


# ================================================================ batch file
def test_missing_batch_file_is_named_as_a_batch_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    emb = np.random.default_rng(0).normal(size=(20, 3))
    labels = ["A", "B"] * 10
    with pytest.raises(FileNotFoundError) as e:
        mtb.evaluate(emb, labels=labels, batch="missing.csv", metrics=["iLISI"])
    msg = str(e.value)
    assert msg == (f"The batch file missing.csv does not exist. From the working directory "
                   f"{tmp_path.resolve()}, the path resolves to "
                   f"{(tmp_path / 'missing.csv').resolve()}.")
    assert "labels file" not in msg and ";" not in msg and "(cwd" not in msg


# ==================================================== scan's modalities warning
def _lung_data(root):
    d = _lung(root / "data")
    return d


def test_modalities_warning_in_python_is_sentences(tmp_path):
    _lung_data(tmp_path)
    with pytest.warns(UserWarning) as rec:
        full_scan("LUNG", "diagonal", data_path=tmp_path / "data",
                 modalities=["rna", "atac_gas"], verbose=False)
    msg = next(str(w.message) for w in rec if "scBridge" in str(w.message))
    assert msg == ("The modalities rna and atac_gas leave out scBridge, which reads a "
                   "folder instead of modality files. Pass modalities=[] to select it, or "
                   "leave out modalities= to see every variant.")


# ==================================================== missing feature names
def _citeseq():
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, size=(30, 40)).astype(float))
    a.obs_names = [f"c{i}" for i in range(30)]
    a.var_names = [f"G{i}" for i in range(40)]
    a.obs["cell_type"] = ["T", "B", "NK"] * 10
    a.obsm["protein"] = rng.poisson(3.0, size=(30, 20)).astype(float)
    return a


def _names_warning(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn(*args, **kw)
    return [str(w.message) for w in rec if "feature names" in str(w.message)]


def test_missing_feature_names_of_to_canonical_name_its_argument(tmp_path):
    msgs = _names_warning(ingest._to_canonical, _citeseq(), tmp_path / "adt.h5",
                          modality="adt", obsm="protein")
    assert len(msgs) == 1 and "Pass feature_names=, store obsm['protein'] as a " \
        "DataFrame" in msgs[0], msgs


def test_missing_feature_names_of_a_bare_array(tmp_path):
    a = _citeseq()
    msgs = _names_warning(ingest.export_dataset, a, tmp_path / "ARR", rna="X",
                          adt=np.ones((30, 4)), labels="obs:cell_type")
    assert msgs == ["The adt matrix array has no feature names. The file names them "
                    "feature_0 to feature_3. Pass adt_names=, or give adt= as a DataFrame "
                    "or AnnData with named features."]


def test_method_info_returns_and_raises_read_as_a_card():
    doc = inspect.getdoc(mtb.method_info)
    returns = doc.split("Returns\n-------\n", 1)[1].split("\n\n", 1)[0]
    raises = doc.split("Raises\n------\n", 1)[1].split("\n\n", 1)[0]
    assert "(" not in returns and ";" not in returns
    assert "declared stub" not in raises
    flat = " ".join(doc.split())
    # 0.4: method_info(verbose=) is deprecated; the card points at params_for
    assert "``params`` (per variant" not in flat
    assert "mtb.params_for : the hyperparameters of one variant" in flat


# ==================================================== docs pages
@needs_docs
def test_run_page_names_the_methods_that_read_both_atac_files():
    text = _flat("tutorials/run.md")
    para = text.split("set `atac_kind=`", 1)[1].split("=== ", 1)[0]
    assert "Each method reads one of the two" not in para
    assert ("Most methods read one of the two, and `method_info(m)[\"atac\"]` names it. "
            "MultiMAP and Seurat_v3 read both.") in para
    assert all(m in para for m in ("MultiMAP", "Seurat_v3"))


@needs_docs
def test_installation_heading_quotes_the_no_conda_error():
    text = _flat("installation.md")
    assert (f"`RuntimeError: {envs.NO_CONDA} Environment <env> has a packed archive.`"
            in text)
    assert "no conda/mamba" not in text


@needs_docs
def test_plot_page_says_how_to_keep_the_methods_scored_everywhere():
    text = _flat("tutorials/plot.md")
    # 0.4: require_complete= is deprecated; methods= keeps the methods scored on
    # every dataset
    assert "require_complete" not in text
    # c4 (u5): the code's rule, a missing dataset is left out
    assert ("A method missing from a dataset is left out there: its score is the mean "
            "over the datasets it was scored on. A warning lists these methods, and "
            "`methods=` keeps the ones scored on every dataset.") in text
    # the subsample tables left in 0.4.0 (X04): the page says a summary needs datasets
    # of your own
    assert "The stored results hold one dataset per category" in text
