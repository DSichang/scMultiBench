"""Review of round 8 (integration): the findings applied in the package.

- R8-01: a sweep saved before the fix keeps ARI -0.0 in batch_result.json;
  ``summary`` shows 0.0 for it.
- R8-07: the grey-column warning under ``aggregate="summary"`` gives each
  mean rank its own sentence, with no serial comma.
- R8-05: off Linux, the "No method can run" list counts only the rows whose
  files are in the folder, as for SKIPPED records; one line counts the rest.
  A variant that reads a folder is listed as ``scBridge``, not
  ``scBridge ((data_dir))``.
- describe_layout and list_categories join no two facts with ';'.
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
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.data import results as RS
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
    sm = mtb.load_batch(tmp_path).summary
    ari = sm.loc[0, "ARI"]
    assert ari == 0.0 and math.copysign(1, ari) == 1
    assert "-0.0" not in sm.to_csv(index=False)
    assert sm.loc[0, "NMI"] == 0.0101


# ======================================================================= R8-05
@pytest.fixture
def macos_no_gpu(monkeypatch):
    monkeypatch.setattr(R, "linux_only_sentence", lambda: MACOS)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)


def _nothing_runnable(dataset, category, **kw):
    with pytest.raises(ValueError) as e:
        mtb.run(dataset, category, out_dir="out", verbose=False, **kw)
    return str(e.value)


def test_the_macos_list_counts_only_rows_whose_files_are_present(macos_no_gpu):
    msg = _nothing_runnable("D11", "vertical", modalities="all")
    plan = full_scan("D11", "vertical", verbose=False, modalities="all")
    lacking = int((~plan["files_ok"]).sum())
    lines = msg.splitlines()
    # 0.4: off Linux the GPU sentence is left out (the OS already blocks), so no
    # method is listed as also blocked by something else
    assert lines[2] == f"{lacking} rows need files that D11 does not have."
    assert lines[3].startswith("mtb.run('D11', 'vertical', modalities=\"all\", dry_run=True) shows "
                               "every row.")
    assert "NVIDIA GPU" not in msg and "also blocked" not in msg
    assert "atac" not in msg
    # with modalities= the rows the folder cannot have are not in the scan
    msg = _nothing_runnable("D11", "vertical", modalities=["rna", "adt"])
    assert "need files" not in msg and "also blocked" not in msg


def test_a_mosaic_folder_lists_no_file_it_cannot_have(macos_no_gpu):
    msg = _nothing_runnable("D46", "mosaic")
    listed = [l for l in msg.splitlines() if l.startswith("  ")]
    assert not [l for l in listed if "atac3.h5" in l or "adt2.h5" in l], listed
    assert re.search(r"^\d+ rows? needs? files that D46 does not have\.$", msg, re.M), msg


def test_a_folder_variant_is_listed_by_its_method_name(macos_no_gpu, monkeypatch):
    for kw in ({}, {"methods": ["scBridge", "SCALEX"]}):
        msg = _nothing_runnable("D28", "diagonal", **kw)
        assert "((" not in msg
        assert "NVIDIA GPU" not in msg, msg          # 0.4: left out off Linux
    # on Linux the GPU sentence is said, on the line of the method's name
    monkeypatch.setattr(R, "linux_only_sentence", lambda: None)
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
    e = _raised(mtb.evaluate, emb, labels=lab, metrics=3)
    assert isinstance(e, TypeError)
    assert str(e) == ("metrics= takes None, a family ('all', 'clustering', 'batch', "
                      "'classification'), a metric code, or a list of them. It got an int.")
    _plain(str(e), lead="metrics=")
    e = _raised(mtb.evaluate, emb, labels=lab, metrics=[])
    _plain(str(e), "metrics=[]", lead=" selects nothing")


def test_the_no_conda_error_is_plain(monkeypatch):
    monkeypatch.setattr(envs.shutil, "which", lambda _: None)
    e = _raised(envs._run_all, [["mamba", "create"]])
    assert isinstance(e, RuntimeError)
    assert str(e).startswith("Neither conda nor mamba is on this machine. Method "
                             "environments cannot be built here.")
    _plain(str(e))


def test_an_empty_bar_table_is_plain():
    e = _raised(mtb.plot.bar, pd.DataFrame(columns=["method", "metric", "value"]))
    assert isinstance(e, ValueError)
    assert str(e).startswith("long_df has no rows to plot. ")
    _plain(str(e), lead="long_df")


# ================================================================ fetch error
def test_scan_names_a_missing_scripts_checkout_in_the_caveat(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_repo_root_no_fetch", lambda: tmp_path / "no_scripts")
    plan = full_scan("D28", "diagonal", methods=["uniPort"], verbose=False)
    assert plan.loc[0, "caveat"].startswith(R.SCRIPTS_NOT_HERE)


# ============================================================ Changes page

@needs_docs
def test_changes_page_follows_the_review_of_round_8(capsys, monkeypatch):
    text = _flat("changes.md")
    assert text.count("- Reworded:") == 1
    for phrase in (
            "- `bubble` widens a family pill to fit its header, and a figure of one to "
            "three metrics to fit its row labels and key. The height is unchanged.",
            "0.3.1 dropped that method. `multibench run-all` exits with `1`.",
            "- For `batch=`, `labels=` and these CSVs, ids that are missing, repeated or "
            "not cells of the dataset raise `ValueError`.",
            "copy a fetched scripts folder and set `MULTIBENCH_REPO_PATH`.",
            "- Missing batch labels get `metrics='all' needs batch labels"):
        assert phrase in text, phrase
    for gone in ("The commands exit with `1`.", "In these three cases",
                 "family headers and key fit", "a scripts checkout"):
        assert gone not in text, gone
    # the 0.3 --only deprecation line the page quotes is history: 0.4 removed --only
