"""Review of round 8 (integration): the findings applied in the package.

- R8-01: a sweep saved before the fix keeps ARI -0.0 in batch_result.json;
  ``summary`` shows 0.0 for it.
- R8-07: the grey-column warning under ``aggregate="summary"`` gives each
  mean rank its own sentence, with no serial comma.
- R8-05: off Linux, the "No method can run" list counts only the rows whose
  files are in the folder, as for SKIPPED records; one line counts the rest.
  A variant that reads a folder is listed as ``scBridge``, not
  ``scBridge ((data_dir))``.
- labels_for names a missing dataset folder as scan does; load_results
  names missing tables in plain sentences without the package path.
- Old-style messages: an unknown modality, an unknown metrics= token, no
  conda, an empty bar table, the empty CLI table and the deprecated
  evaluate flags.
- The fetch error with a ref keeps its subject in run's FAIL line.
- The ``env plan`` and ``env install`` descriptions are short sentences
  (the option help of ``scan`` and ``convert``: tests/test_study_r8_other.py);
  'preflight' is gone.
- scan names a missing scripts checkout in the ``caveat`` column.
"""
import importlib.util
import inspect
import json
import math
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench._data import results as RS
from multibench.engine import envs
from multibench.engine import runner as R
from tests.test_docs_r4 import _flat, needs_docs
from tests.test_study_r8_other import _plain
from tests.conftest import full_scan

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
MACOS = "Methods run only on Linux, and this computer runs macOS."


# ======================================================================= R8-01
def test_a_stored_minus_zero_ari_reads_zero_in_summary(tmp_path):
    rec = {"method": "VIMCCA", "status": "CHAIN_OK", "run_sec": 1.0,
           "metrics": {"ARI": -0.0, "NMI": 0.0101}, "out_dir": str(tmp_path / "VIMCCA_X")}
    (tmp_path / "batch_result.json").write_text(json.dumps(
        {"dataset": "X", "category": "vertical", "records": [rec]}))
    assert '"ARI": -0.0' in (tmp_path / "batch_result.json").read_text()
    sm = mtb.load_run(tmp_path).summary
    ari = sm.loc[0, "ARI"]
    assert ari == 0.0 and math.copysign(1, ari) == 1
    assert "-0.0" not in sm.to_csv(index=False)
    assert sm.loc[0, "NMI"] == 0.0101


# ======================================================================= R8-05
@pytest.fixture
def macos_no_gpu(monkeypatch):
    monkeypatch.setattr("multibench.engine.envs.host_system", lambda: "macOS")
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)


def _nothing_runnable(dataset, category, **kw):
    with pytest.raises(ValueError) as e:
        mtb.run(dataset, category, out_dir="out", verbose=False, **kw)
    return str(e.value)


def test_the_macos_list_leaves_out_methods_whose_files_are_missing(macos_no_gpu):
    msg = _nothing_runnable("D11", "vertical", modalities="all")
    lines = msg.splitlines()
    # 0.4: off Linux the GPU sentence is left out (the OS already blocks), and the
    # ATAC methods, whose files D11 lacks, are left to scan
    assert len(lines) == 3, msg
    assert lines[2] == "mtb.scan('D11', 'vertical', modalities=\"all\") says why."
    assert "NVIDIA GPU" not in msg and "atac" not in msg


def test_a_mosaic_folder_lists_no_file_it_cannot_have(macos_no_gpu):
    msg = _nothing_runnable("D46", "mosaic")
    listed = [l for l in msg.splitlines() if l.startswith("  ")]
    assert not [l for l in listed if "atac3.h5" in l or "adt2.h5" in l], listed


def test_a_folder_variant_is_listed_by_its_method_name(macos_no_gpu, monkeypatch):
    for kw in ({}, {"methods": ["scBridge", "SCALEX"]}):
        msg = _nothing_runnable("D28", "diagonal", **kw)
        assert "((" not in msg
        assert "NVIDIA GPU" not in msg, msg          # 0.4: left out off Linux
    # on Linux the GPU sentence is said, on the line of the method's name
    monkeypatch.setattr("multibench.engine.envs.host_system", lambda: None)
    msg = _nothing_runnable("D28", "diagonal", methods=["scBridge", "SCALEX"])
    assert "((" not in msg
    line = next(l for l in msg.splitlines() if l.startswith("  scBridge"))
    assert line.startswith("  scBridge: ") and "scBridge needs an NVIDIA GPU" in line, msg


# ================================================= missing folders and tables
@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "D11").symlink_to(ROOT / "data" / "D11")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_labels_for_names_a_missing_folder_as_scan_does(in_tmp):
    with pytest.raises(FileNotFoundError) as e:
        mtb.engine.resolve.labels_for("MYCITE", "vertical", "Matilda", data_path="data")
    with pytest.raises(FileNotFoundError) as s:
        full_scan("MYCITE", "vertical", data_path="data", verbose=False)
    assert str(e.value) == str(s.value)
    assert str(e.value).startswith("The folder data/MYCITE does not exist. data holds D11.")


# ====================================================== old-style messages
def _raised(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fn(*args, **kw)
        except Exception as e:  # noqa: BLE001 - the text is checked
            return e
    raise AssertionError("no error")


def test_a_metrics_token_error_is_plain_outside_its_examples():
    emb, lab = np.random.default_rng(0).normal(size=(6, 2)), list("aabbcc")
    # 0.4: one code is a valid metrics= (metrics="ARI"); an unknown token is refused
    e = _raised(mtb.evaluate, emb, labels=lab, metrics="ARII")
    assert isinstance(e, ValueError)
    assert str(e) == ("Unknown metrics= token ARII. Pass 'all', 'clustering', 'batch' or "
                      "'classification', or metric codes such as metrics=['ARI', 'NMI'].")
    _plain(str(e), "metrics=['ARI', 'NMI']")
    e = _raised(mtb.evaluate, emb, labels=lab, metrics="dimension_reduction")
    _plain(str(e), "metrics=['ARI', 'NMI']")
    e = _raised(mtb.evaluate, emb, labels=lab, metrics=[])
    _plain(str(e), "metrics=[]", lead=" selects nothing")


def test_the_no_conda_error_is_plain(monkeypatch):
    monkeypatch.setattr(envs.shutil, "which", lambda _: None)
    e = _raised(envs._run_all, [["mamba", "create"]])
    assert isinstance(e, RuntimeError)
    assert str(e) == "Conda is not installed on this computer. Install conda first."
    _plain(str(e))


def test_an_empty_bar_table_is_plain():
    e = _raised(mtb.plot.bar, pd.DataFrame(columns=["method", "metric", "value"]))
    assert isinstance(e, ValueError)
    assert str(e).startswith("long_df has no rows to plot. ")
    _plain(str(e), lead="long_df")


# ================================================================ fetch error
def test_scan_says_nothing_of_a_missing_scripts_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_repo_root_no_fetch", lambda: tmp_path / "no_scripts")
    plan = full_scan("D28", "diagonal", methods=["uniPort"], verbose=False)
    assert "scripts are not in" not in plan.loc[0, "caveat"]


# ============================================================ Changes page

    # the 0.3 --only deprecation line the page quotes is history: 0.4 removed --only
