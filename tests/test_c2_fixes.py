"""Core round c2 (0.4.0): the fixes, owner decisions and simplifications after the
overall score (tests/test_c2_overall.py)."""
import importlib
import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, workflow
from tests.test_c1_decisions import _tree


def _old_tree(tmp_path):
    """A run folder whose record a 0.3.9 run wrote."""
    out = _tree(tmp_path)
    blob = json.loads((out / "batch_result.json").read_text())
    blob["records"][0].update(run_sec=17.0, multibench_version="0.3.9",
                              started_at="2026-10-17T16:00:33", labels_used=["cty.csv"])
    (out / "batch_result.json").write_text(json.dumps(blob))
    return out


# --------------------------------------------------------------- D03
@pytest.mark.parametrize("how", ["python", "cli"])
def test_skip_existing_rescores_the_saved_record(tmp_path, monkeypatch, how):
    from multibench.engine import envs
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "Methods run only on Linux.")
    out = _old_tree(tmp_path)
    ref = tmp_path / "ref"
    shutil.copytree(out, ref)
    want = mtb.load_batch(ref).rescore(verbose=False).records[0]
    if how == "python":
        mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir=out,
                    skip_existing=True, verbose=False)
    else:
        assert cli.main(["run-all", "D11", "--category", "vertical", "--methods", "Matilda",
                         "--out-dir", str(out), "--skip-existing"]) == 0
    got = mtb.load_batch(out).records[0]
    for key in ("run_sec", "multibench_version", "started_at", "status", "metrics",
                "labels_used", "scored_by_version"):
        assert got.get(key) == want.get(key), key
    assert (got["run_sec"], got["multibench_version"]) == (17.0, "0.3.9")


# --------------------------------------------------------------- D28
def test_rescore_does_not_rank_a_recorded_label_order(tmp_path, monkeypatch):
    seen = []
    real = workflow._evaluate_best_order

    def spy(emb, category, cands, **kw):
        seen.append(len(cands))
        return real(emb, category, cands, **kw)
    monkeypatch.setattr(workflow, "_evaluate_best_order", spy)
    mtb.load_batch(_old_tree(tmp_path)).rescore(metrics=["cLISI"], verbose=False)
    assert seen == [1]


# --------------------------------------------------------------- D04, D14, D15
def test_the_summary_has_no_label_order_confidence_and_the_records_keep_the_orders():
    cands = [{"order": ["a.csv", "b.csv"], "ARI": 0.6}, {"order": ["b.csv", "a.csv"], "ARI": 0.1}]
    rec = {"method": "M", "status": "CHAIN_OK", "metrics": {"ARI": 0.6},
           "labels_used": ["a.csv", "b.csv"], "label_order_candidates": cands}
    res = mtb.BatchResult([rec], "D28", "diagonal")
    assert "label_order_confidence" not in res.summary.columns
    assert res.records[0]["label_order_candidates"] == cands
    assert not hasattr(workflow, "_order_confidence")


def test_failures_name_their_text_reason_as_the_summary_does():
    recs = [{"method": "A", "status": "FAIL", "error": "boom"},
            {"method": "B", "status": "RUN_OK_EVAL_FAILED", "error": "no labels"},
            {"method": "C", "status": "RUN_OK_NO_EMBEDDING"},
            {"method": "D", "status": "SKIPPED", "error": "env", "requested": True},
            {"method": "E", "status": "SKIPPED", "error": "env"},
            {"method": "F", "status": "TIMEOUT_600", "error": "slow"}]
    res = mtb.BatchResult(recs, "D11", "vertical")
    f = res.failures
    assert list(f.columns) == ["method", "status", "reason"]
    assert f["method"].tolist() == ["A", "B", "D", "F"]
    s = res.summary.set_index("method")
    assert (f.set_index("method")["reason"] == s.loc[f["method"], "reason"]).all()


def test_every_record_code_the_package_writes_has_a_status():
    src = Path(workflow.__file__).read_text()
    written = set(re.findall(r'rec\["status"\](?:, rec\["error"\])? = "([A-Z_]+)"', src))
    written |= set(re.findall(r'"status": "([A-Z_]+)"', src))
    assert written and written <= set(workflow._STATUS)
    # outside _STATUS, the code reads a record's status through _status only
    body = src[src.index("_RAN_REASON = "):]
    assert not re.search(r'get\("status"\) *(==|!=) *"|get\("status"\) +in +\(', body)
    assert not re.search(r'status.*startswith\(\("FAIL"', body)


# --------------------------------------------------------------- D16
def test_one_plan_column_list():
    assert not hasattr(workflow, "COMPACT_COLUMNS")
    assert not hasattr(workflow, "compact_columns")
    assert workflow.PLAN_COLUMNS[:4] == ["method", "modalities", "runnable", "reason"]


# --------------------------------------------------------------- C43, C57
def test_fetch_returns_the_dataset_folder(monkeypatch):
    fetch = importlib.import_module("multibench.data.fetch")
    root = Path(mtb.config.DEFAULT.data_path)
    assert mtb.data.fetch("D11") == root / "D11"           # present: nothing downloads
    monkeypatch.setattr(fetch, "_download", lambda url: pytest.fail("downloaded"))
    assert mtb.data.fetch("D11", "D28") == [root / "D11", root / "D28"]


def test_the_labels_recipe_reads_a_folder_in_its_order():
    folder = mtb.data.fetch("D11")
    labels = pd.read_csv(folder / "cty.csv").iloc[:, 0]
    assert len(labels) == 2864


# --------------------------------------------------------------- D09, C41
def test_recommend_modalities_keeps_your_own_methods():
    rows = pd.DataFrame([{"method": m, "metric": c, "value": v + k / 10, "dataset": "MULTI",
                          "source": "user"}
                         for k, m in enumerate(["RNA PCA", "joint PCA", "random"])
                         for c, v in (("ARI", 0.3), ("NMI", 0.4))])
    out = mtb.recommend("vertical", long_df=rows, modalities=["rna", "atac"])
    assert out["method"].tolist() == ["random", "joint PCA", "RNA PCA"]


def test_recommend_leads_with_the_overall_score():
    assert mtb.recommend.__doc__.splitlines()[0] == (
        "The overall score and row order of ``mtb.plot.bubble``, best first.")
