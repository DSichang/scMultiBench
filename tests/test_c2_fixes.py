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


# --------------------------------------------------------------- D05, D19, D22, C12
def _mosaic(root):
    """A small mosaic folder (rna1, rna2+atac2, atac3) of 250 cells per batch."""
    from tests.test_f3_int import _h5
    d = root / "MOS"
    d.mkdir(parents=True)
    bars = {b: [f"b{b}c{i}" for i in range(250)] for b in (1, 2, 3)}
    genes, peaks = [f"g{i}" for i in range(30)], [f"chr1:{i}-{i + 9}" for i in range(0, 300, 10)]
    _h5(d / "rna1.h5", genes, bars[1])
    _h5(d / "rna2.h5", genes, bars[2])
    _h5(d / "atac2.h5", peaks, bars[2])
    _h5(d / "atac3.h5", peaks, bars[3])
    for b in (1, 2, 3):
        pd.DataFrame({"x": ["T", "B"] * 125}).to_csv(d / f"cty{b}.csv", index=False)
    return d


def test_a_size_caveat_states_the_folders_number(tmp_path):
    plan = mtb.run_all(_mosaic(tmp_path), "mosaic", dry_run=True, verbose=False)
    cav = plan.set_index("method").loc["Multigrate", "caveat"]
    assert cav == ("Multigrate needs about 1,300 cells in every batch; batch 1 of this "
                   "folder has 250: leave Multigrate out.")


def test_the_benchmarks_own_mosaic_batches_hold_the_same_cells():
    # c2 (C12): why nothing refuses batches of the same cells: D45's are, by design
    from multibench.engine import resolve
    d = Path(mtb.config.DEFAULT.data_path) / "D45"
    assert resolve._barcodes_of(d / "rna1.h5") == resolve._barcodes_of(d / "rna2.h5")
    assert mtb.scan("D45", "mosaic", verbose=False)["reason"].str.contains(
        "same").sum() == 0


def test_the_dry_run_command_keeps_the_out_dir_placeholder():
    plan = mtb.run_all("D28", "diagonal", methods=["iNMF"], dry_run=True, verbose=False)
    assert plan["command"].iloc[0].endswith("'<out_dir>/iNMF_D28/'")


def test_a_batch_of_the_wrong_length_is_refused_before_anything_runs(tmp_path):
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", methods=["Matilda"], dry_run=True, verbose=False,
                    batch=["a"] * 100)
    assert str(e.value) == ("batch has 100 entries for the 2,864 cells of D11: give one "
                            "batch per cell.")


# --------------------------------------------------------------- D07, D08, D10-D13
def _emb(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 4))


def test_the_length_messages_fit_the_call():
    with pytest.raises(ValueError) as e:
        mtb.evaluate(_emb(500), labels=["a", "b"] * 249, metrics=["ARI"], verbose=False)
    assert str(e.value) == "labels has 498 entries for 500 embedding rows: give one label per row."
    with pytest.raises(ValueError) as e:
        mtb.evaluate(_emb(2000), dataset="D11", category="vertical", method="sciPENN",
                     metrics=["ARI"], verbose=False)
    assert str(e.value) == ("The label files of D11 (cty.csv) hold 2,864 cells; the "
                            "embedding has 2,000 rows. Check that the embedding is the one "
                            "of method= on D11.")


def test_obsm_x_on_an_anndata_without_x():
    import anndata as ad
    a = ad.AnnData(obs=pd.DataFrame({"celltype": ["a", "b"] * 10},
                                    index=[f"c{i}" for i in range(20)]))
    a.obsm["X_joint"] = _emb(20)
    for kw, what in (({"obsm": "X"}, "no X"), ({}, "no obsm['X_emb']")):
        with pytest.raises(ValueError) as e:
            mtb.evaluate(a, labels="celltype", metrics=["ASW"], verbose=False, **kw)
        assert str(e.value) == (f"The AnnData has {what}; its obsm keys are X_joint: pass "
                                f"obsm='X_joint'.")


def test_label_files_without_ids_are_matched_by_position_without_a_warning():
    res_obs = pd.Index([f"cty:{i}" for i in range(2864)])
    import anndata as ad
    a = ad.AnnData(obs=pd.DataFrame(index=res_obs))
    a.obsm["X_emb"] = _emb(2864)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mtb.evaluate(a, dataset="D11", category="vertical", method="Matilda (this run)",
                     metrics=["ASW"], verbose=False)


def test_diagonal_classification_follows_the_folders_sides_whatever_batch_is():
    from multibench.eval import pipeline
    n = sum(len(pd.read_csv(Path(mtb.config.DEFAULT.data_path) / "D28" / f))
            for f in ("rna_cty.csv", "atac_cty.csv"))
    e, donor = _emb(n), np.random.default_rng(1).choice(["d1", "d2"], n)
    kw = dict(dataset="D28", category="diagonal", method="PCA", metrics="classification",
              verbose=False)
    a = mtb.evaluate(e, **kw)
    b = mtb.evaluate(e, batch=donor, **kw)
    assert a["value"].tolist() == b["value"].tolist()


def test_a_package_method_on_the_wrong_category():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as e:
            mtb.evaluate(_emb(10), dataset="D52", category="vertical", method="StabMap")
    assert str(e.value) == ("StabMap does not run on vertical data; D52 is cross: pass "
                            "category='cross'.")


def test_a_benchmark_dataset_gives_the_category(tmp_path, capsys):
    out = mtb.evaluate(_emb(2864), dataset="D11", method="sciPENN", metrics=["ARI"],
                       verbose=False)
    assert out["category"].tolist() == ["vertical"]
    np.save(tmp_path / "e.npy", _emb(2864))
    assert cli.main(["evaluate", str(tmp_path / "e.npy"), "--dataset", "D11",
                     "--method", "sciPENN", "--metrics", "ARI"]) == 0


# --------------------------------------------------------------- C15, D29
def test_every_long_table_has_scored_with():
    st = mtb.load_results("vertical")
    assert list(st.columns) == ["metric", "value", "method", "dataset", "category",
                                "source", "scored_with"]
    assert st["scored_with"].unique().tolist() == ["leidenalg/search/0.4.0"]


@pytest.mark.parametrize("metrics", ["kBET", ["clustering", "kBET"]])
def test_an_absent_metric_has_one_text(metrics):
    with pytest.warns(UserWarning) as rec:
        mtb.load_results("cross", metrics=metrics)
    assert [str(w.message) for w in rec] == ["The stored results have no kBET."]
