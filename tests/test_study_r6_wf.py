"""Study round 6, work package wf: R6-01, R6-02, R6-04, R6-05 and R6-06.

R6-01: a saved result read from another directory lost its dataset folder
(``data_path`` recorded as given, relative). rescore then matched a barcode
Series by position and blamed the files. run_all records ``data_root``;
load_batch takes ``data_path=``; a folder that is not found is named.

R6-02: run_all(batch=) kept only batch_source and n_batches, not the vector,
so every rescore without batch= lost the batch metrics.

R6-04: rescore(metrics=) without a clustering metric still ranked the label
orders with a Leiden sweep.

R6-05: rescore handed its own file-of-origin batch to evaluate, which warned
about a batch= the caller never gave.

R6-06: label_order_confidence went above 1 when the runner-up ARI was below 0.

The runs are faked (``_run`` writes an embedding), as in the R4-02 tests.
"""
import inspect
import json
import re
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs
from multibench.eval import pipeline
from tests.test_study_r4_wf import (ALL_ENVS, N, _batch_metrics, _h5, _quiet,  # noqa: F401
                                    _Res, cite)

pytest.importorskip("scib")

KW = dict(methods=["Matilda"], modalities=["rna", "adt"], verbose=False)
BARS = [f"c{i}" for i in range(N)]
CELLTYPE = np.array(["A"] * (N // 2) + ["B"] * (N // 2))
MOVED = ("Pass data_path= to mtb.load_batch, or run rescore from the folder where "
         "run_all ran.")


def _ari(res):
    return round(float(res.summary["ARI"].iloc[0]), 4)


def _shuffled_labels(seed=5):
    return pd.Series(CELLTYPE, index=BARS).sample(frac=1.0, random_state=seed)


def _by_position(messages) -> list:
    return [m for m in messages if "by position" in m or "looks like cell ids" in m]


def _rescore(res, **kw):
    """rescore, with the warnings it raised."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = res.rescore(**kw)
    return out, [str(w.message) for w in rec]


# ======================================================================= R6-01
@pytest.fixture
def rel(cite, tmp_path, monkeypatch):
    """A sweep run from ``tmp_path`` with the relative ``data_path='data'``."""
    monkeypatch.chdir(tmp_path)
    res = _quiet(mtb.run_all, "MYCITE", "vertical", "out", data_path="data", **KW)
    (tmp_path / "nb").mkdir()
    return res, tmp_path


def test_run_all_records_the_absolute_data_root(rel):
    res, root = rel
    rec = res.results[0]
    assert rec["data_path"] == "data"
    assert Path(rec["data_root"]).is_absolute()
    assert Path(rec["data_root"]).samefile(root / "data")
    saved = json.loads((root / "out" / "batch_result.json").read_text())["records"][0]
    assert saved["data_root"] == rec["data_root"]


def test_rescore_from_another_directory_aligns_the_labels_by_barcode(rel, monkeypatch):
    res, root = rel
    at_root = _quiet(mtb.load_batch("out").rescore, labels=_shuffled_labels())
    assert at_root.summary.iloc[0]["label_order"] == "cty.csv"
    assert _ari(at_root) == _ari(res) > 0.3
    monkeypatch.chdir(root / "nb")
    back = mtb.load_batch("../out")
    assert Path(back.results[0]["data_path"]).samefile(root / "data")
    new, caught = _rescore(back, labels=_shuffled_labels())
    assert not _by_position(caught), caught
    assert new.summary.iloc[0]["label_order"] == "cty.csv"
    assert _ari(new) == _ari(at_root)


def test_rescore_of_an_in_memory_result_after_a_chdir(rel, monkeypatch):
    _, root = rel
    res = _quiet(mtb.run_all, "MYCITE", "vertical", root / "abs", data_path="data", **KW)
    monkeypatch.chdir(root / "nb")
    new, caught = _rescore(res, labels=_shuffled_labels())
    assert not _by_position(caught), caught
    assert new.summary.iloc[0]["label_order"] == "cty.csv" and _ari(new) == _ari(res)


def test_a_copied_result_is_pointed_at_its_data_with_data_path(rel, monkeypatch):
    res, root = rel
    copy = root / "elsewhere" / "copy"
    shutil.copytree(root / "out", copy)
    monkeypatch.chdir(root / "nb")
    new = _quiet(mtb.load_batch(copy, data_path=root / "data").rescore,
                 labels=_shuffled_labels())
    assert new.summary.iloc[0]["label_order"] == "cty.csv" and _ari(new) == _ari(res)
    # the data moved too: only the argument finds it
    shutil.move(str(root / "data"), str(root / "moved"))
    back = mtb.load_batch(copy, data_path=root / "moved")
    assert back.results[0]["data_path"] == str(root / "moved")
    assert back.results[0]["out_dir"] == str(copy / "Matilda_MYCITE")
    new, caught = _rescore(back, labels=_shuffled_labels())
    assert not _by_position(caught), caught
    assert new.summary.iloc[0]["label_order"] == "cty.csv" and _ari(new) == _ari(res)
    assert _ari(_quiet(back.rescore)) == _ari(res)


def test_a_folder_that_is_not_found_is_named(rel, monkeypatch):
    res, root = rel
    shutil.move(str(root / "data"), str(root / "moved"))
    monkeypatch.chdir(root / "nb")
    back = mtb.load_batch("../out")
    assert back.results[0]["data_path"] == "data"         # left as recorded
    # the roots tried: the recorded data_path from here, then data_root
    where = f"{Path.cwd() / 'data'} or {back.results[0]['data_root']}"
    with pytest.warns(UserWarning, match=re.escape(
            f"The labels Series is matched by position, because the dataset folder "
            f"MYCITE is not found in {where}. {MOVED}")):
        back.rescore(labels=_shuffled_labels())
    csv = root / "labels.csv"
    _shuffled_labels().to_frame("celltype").to_csv(csv)
    with pytest.warns(UserWarning, match=re.escape(
            f"The first column of labels.csv looks like cell ids, but the dataset "
            f"folder MYCITE is not found in {where}. {MOVED[:-1]}. The file "
            f"is matched by position.")):
        back.rescore(labels=csv)
    new = _quiet(back.rescore, metrics=["ARI"])
    rec = new.results[0]
    assert rec["status"] == "RUN_OK_NO_LABEL_MATCH"
    assert rec["note"] == f"The dataset folder MYCITE is not found in {where}. {MOVED}"
    for k in ("metrics", "batch_source", "n_batches", "labels_used"):
        assert k not in rec, k
    row = new.summary.iloc[0]
    assert pd.isna(row["batch_source"]) and pd.isna(row["n_batches"])


def test_unusable_barcodes_keep_their_own_message(rel):
    res, root = rel
    for f in ("rna.h5", "adt.h5"):
        with h5py.File(root / "data" / "MYCITE" / f, "a") as h:
            del h["matrix/barcodes"]
            h["matrix/barcodes"] = np.array(["same"] * N, dtype="S8")
    with pytest.warns(UserWarning, match=r"because the files of MYCITE have no usable "
                                         r"cell ids \(missing or repeated barcodes\)"):
        res.rescore(labels=_shuffled_labels(), metrics=["ARI"])


def test_rescore_from_the_run_folder_and_of_a_fetched_tree_is_unchanged(rel, monkeypatch):
    res, root = rel
    again = _quiet(mtb.load_batch("out").rescore)
    # the folder found, as an absolute path (review of round 6)
    assert again.results[0]["data_path"] == again.results[0]["data_root"] == \
        str((root / "data").resolve())
    assert again.summary.iloc[0]["label_order"] == "cty.csv" and _ari(again) == _ari(res)
    # a fetch_outputs tree: data_path None (config's), out_dir from another host
    tree = root / "tree"
    shutil.copytree(root / "out", tree)
    blob = json.loads((tree / "batch_result.json").read_text())
    for r in blob["records"]:
        r["data_path"] = None
        r.pop("data_root", None)
        r["out_dir"] = "/benchmark/host/out/Matilda_MYCITE"
    (tree / "batch_result.json").write_text(json.dumps(blob))
    monkeypatch.setattr(config.DEFAULT, "data_path", root / "data")
    monkeypatch.chdir(root / "nb")
    back = mtb.load_batch(tree)
    assert back.results[0]["data_path"] is None
    new = _quiet(back.rescore)
    assert new.results[0]["data_path"] is None
    assert new.summary.iloc[0]["label_order"] == "cty.csv" and _ari(new) == _ari(res)


# ======================================================================= R6-02
@pytest.fixture
def batched(cite, tmp_path):
    """A sweep scored with a shuffled barcode-indexed batch Series."""
    data, batch, shuffled = cite
    res = _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "out", data_path=data,
                 batch=shuffled, **KW)
    return res, batch, tmp_path


def test_run_all_saves_the_batch_it_scored_with(batched):
    res, batch, tmp = batched
    rec = res.results[0]
    assert rec["batch_source"] == "user" and rec["n_batches"] == 2
    name = rec["batch_file"]
    assert re.fullmatch(r"batch_[0-9a-f]{8}\.csv", name)
    saved = pd.read_csv(tmp / "out" / name)
    assert list(saved.columns) == ["cell", "batch"]
    assert list(saved["cell"]) == BARS and list(saved["batch"]) == list(batch)
    blob = json.loads((tmp / "out" / "batch_result.json").read_text())
    assert blob["records"][0]["batch_file"] == name


def test_rescore_reuses_the_saved_batch(batched):
    res, batch, tmp = batched
    before = _batch_metrics(res)
    assert before[1] is not None
    back = mtb.load_batch(tmp / "out")
    for kw in ({}, {"labels": _shuffled_labels(2)}):
        new, caught = _rescore(back, **kw)
        rec = new.results[0]
        assert (rec["batch_source"], rec["n_batches"]) == ("user", 2), kw
        assert _batch_metrics(new) == before, kw
        assert not [m for m in caught if "batch vector" in m], caught
    # the reference example: the re-scored result keeps the batch on disk
    _quiet(back.rescore, labels=_shuffled_labels(2)).save(tmp / "rescored")
    assert (tmp / "rescored" / res.results[0]["batch_file"]).is_file()
    again = _quiet(mtb.load_batch(tmp / "rescored").rescore)
    assert _batch_metrics(again) == before
    # a batch that is given wins, and is saved in its turn
    ones = _quiet(back.rescore, batch=np.ones(N, dtype=int))
    assert ones.results[0]["n_batches"] == 1
    assert ones.results[0]["batch_file"] != res.results[0]["batch_file"]
    ones.save(tmp / "ones")
    assert _quiet(mtb.load_batch(tmp / "ones").rescore).results[0]["n_batches"] == 1


def test_a_folder_without_the_saved_batch_warns_once(batched):
    res, batch, tmp = batched
    p = tmp / "out" / "batch_result.json"
    blob = json.loads(p.read_text())
    for r in blob["records"]:
        (tmp / "out" / r.pop("batch_file")).unlink()
    p.write_text(json.dumps(blob))
    old = mtb.load_batch(tmp / "out")
    text = ("This result was scored with your batch vector, which older versions did "
            "not save. Pass batch= again to keep the batch metrics.")
    new, caught = _rescore(old)
    assert caught.count(text) == 1, caught
    assert new.results[0]["n_batches"] == 1
    failed, caught = _rescore(old, metrics="all")
    assert caught.count(text) == 1, caught
    rec = failed.results[0]
    assert rec["status"] == "RUN_OK_EVAL_FAILED"
    assert ("metrics='all' needs batch labels for ASW_batch, GC and iLISI. "
            "Pass batch=<vector>, or metrics='clustering'.") in rec["error"]
    for k in ("batch_source", "n_batches", "metrics"):
        assert k not in rec, k
    assert pd.isna(failed.summary.iloc[0]["n_batches"])


def test_the_batch_label_errors_are_short_sentences(monkeypatch):
    rng = np.random.default_rng(0)
    emb, ct = rng.normal(size=(60, 4)), np.array(["A", "B"] * 30)
    cases = {
        "all": ("metrics='all' needs batch labels for ASW_batch, GC and iLISI. "
                "Pass batch=<vector>, or metrics='clustering'.",
                "--metrics all needs batch labels for ASW_batch, GC and iLISI. "
                "Pass --batch CSV, or --metrics clustering."),
        "batch": ("metrics='batch' needs batch labels for ASW_batch, GC and iLISI. "
                  "Pass batch=<vector>, or labels as a list of two or more files.",
                  "--metrics batch needs batch labels for ASW_batch, GC and iLISI. "
                  "Pass --batch CSV, or two or more --labels files."),
    }
    for metrics, (py, cli) in cases.items():
        with pytest.raises(ValueError) as exc:
            pipeline.evaluate(emb, labels=ct, metrics=metrics)
        assert str(exc.value) == py
    with pytest.raises(ValueError) as exc:
        pipeline.evaluate(emb, labels=ct, metrics=["GC", "iLISI"])
    assert str(exc.value) == ("metrics=['GC', 'iLISI'] needs batch labels for GC and "
                              "iLISI. Pass batch=<vector>, or labels as a list of two "
                              "or more files. Each file then counts as one batch.")
    monkeypatch.setattr(config, "_CLI", True)
    for metrics, (py, cli) in cases.items():
        with pytest.raises(ValueError) as exc:
            pipeline.evaluate(emb, labels=ct, metrics=metrics)
        assert str(exc.value) == cli
    with pytest.raises(ValueError) as exc:
        pipeline.evaluate(emb, labels=ct, metrics=["GC", "iLISI"])
    assert str(exc.value).startswith("--metrics GC,iLISI needs batch labels for GC and "
                                     "iLISI. Pass --batch CSV, or two or more --labels")


# ============================================================ R6-04 and R6-05
SIZES = {1: 40, 2: 50, 3: 60}


@pytest.fixture
def cross(tmp_path, monkeypatch):
    """Three RNA+ADT batches whose label files differ, and fake runs whose
    embedding stacks the cells in file order (cty1, cty2, cty3)."""
    d = tmp_path / "data" / "MYCROSS"
    d.mkdir(parents=True)
    labs = {}
    rng = np.random.default_rng(3)
    for b, n in SIZES.items():
        bars = [f"b{b}c{i}" for i in range(n)]
        _h5(d / f"rna{b}.h5", [f"g{i}" for i in range(30)], bars)
        _h5(d / f"adt{b}.h5", [f"p{i}" for i in range(6)], bars)
        labs[b] = rng.choice(["A", "B", "C"], n)
        pd.DataFrame({"x": labs[b]}).to_csv(d / f"cty{b}.csv", index=False)
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)

    def fake_run(method, category, inputs, out_dir, params=None):
        batches = sorted({int(k[-1]) for k in inputs if k[-1].isdigit()})
        lab = np.concatenate([labs[b] for b in batches])
        bat = np.concatenate([np.full(SIZES[b], b) for b in batches])
        emb = np.random.default_rng(7).normal(size=(len(lab), 4))
        emb[:, 0] += 8.0 * (lab == "B")
        emb[:, 1] += 8.0 * (lab == "C")
        emb[:, 2] += 3.0 * bat
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        return _Res(emb)
    monkeypatch.setattr(W, "_run", fake_run)
    return tmp_path / "data"


def _cross_run(data, tmp, method):
    return _quiet(mtb.run_all, "MYCROSS", "cross", tmp / method, data_path=data,
                  methods=[method], verbose=False)


def _count_sweeps(monkeypatch, fail=False):
    calls = []
    real = W._escib.leiden_sweep

    def counted(*a, **k):
        calls.append(1)
        if fail:
            raise AssertionError("leiden_sweep was called")
        return real(*a, **k)
    monkeypatch.setattr(W._escib, "leiden_sweep", counted)
    return calls


def test_rescore_without_clustering_metrics_keeps_the_stored_order(cross, tmp_path,
                                                                    monkeypatch):
    res = _cross_run(cross, tmp_path, "totalVI")
    rec = res.results[0]
    assert rec["labels_used"] == ["cty1.csv", "cty2.csv", "cty3.csv"]
    assert len(rec["label_order_candidates"]) == 6
    back = mtb.load_batch(tmp_path / "totalVI")
    calls = _count_sweeps(monkeypatch, fail=True)
    new = back.rescore(metrics=["ASW"])
    assert calls == []
    out = new.results[0]
    assert out["status"] == "CHAIN_OK"
    assert out["metrics"]["ASW"] == rec["metrics"]["ASW"]
    assert out["labels_used"] == rec["labels_used"]
    assert out["label_order_candidates"] == rec["label_order_candidates"]
    cols = ["label_order", "label_order_confidence", "batch_source", "n_batches"]
    assert new.summary[cols].equals(res.summary[cols])


def test_rescore_with_a_clustering_metric_ranks_with_one_sweep(cross, tmp_path,
                                                              monkeypatch, capsys):
    _cross_run(cross, tmp_path, "totalVI")
    back = mtb.load_batch(tmp_path / "totalVI")
    calls = _count_sweeps(monkeypatch)
    new = _quiet(back.rescore, metrics=["ARI"], verbose=True)
    assert len(calls) == 1
    assert len(new.results[0]["label_order_candidates"]) == 6
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ("[rescore] totalVI: ranking 6 label orders on 150 cells with "
                        "one Leiden sweep ...")
    assert lines[1].startswith("[rescore] totalVI -> CHAIN_OK")


def test_a_stored_order_that_no_longer_fits_is_ranked_again(cross, tmp_path, monkeypatch):
    _cross_run(cross, tmp_path, "totalVI")
    p = tmp_path / "totalVI" / "batch_result.json"
    blob = json.loads(p.read_text())
    blob["records"][0]["labels_used"] = ["cty1.csv", "cty2.csv"]     # 90 of 150 cells
    p.write_text(json.dumps(blob))
    calls = _count_sweeps(monkeypatch)
    new = _quiet(mtb.load_batch(tmp_path / "totalVI").rescore, metrics=["ASW"])
    assert len(calls) == 1
    assert new.results[0]["labels_used"] == ["cty1.csv", "cty2.csv", "cty3.csv"]


def test_rescore_does_not_warn_about_its_own_file_of_origin_batch(cross, tmp_path):
    res = _cross_run(cross, tmp_path, "UINMF")
    rec = res.results[0]
    assert len(rec["label_order_candidates"]) == 2 and rec["n_batches"] == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        new = res.rescore(metrics=["ARI", "NMI"])
    out = new.results[0]
    assert out["metrics"]["ARI"] == rec["metrics"]["ARI"]
    assert (out["batch_source"], out["n_batches"]) == ("file_of_origin", 2)
    # a batch the caller gives still warns, in short sentences
    with pytest.warns(UserWarning) as caught:
        res.rescore(batch=np.repeat([1, 2], [40, 50]), metrics=["ARI"])
    assert ("batch= changes nothing here, because metrics=['ARI'] has no batch metric. "
            "Add ASW_batch, GC or iLISI, or pass metrics='all'.") in \
        [str(w.message) for w in caught]


def test_the_batch_warning_on_the_command_line(monkeypatch):
    rng = np.random.default_rng(0)
    emb, ct = rng.normal(size=(60, 4)), np.array(["A", "B"] * 30)
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.warns(UserWarning) as caught:
        pipeline.evaluate(emb, labels=ct, batch=np.array([1, 2] * 30),
                          metrics=["ARI", "NMI"])
    assert ("--batch changes nothing here, because --metrics ARI,NMI has no batch "
            "metric. Add ASW_batch, GC or iLISI to --metrics, or pass --metrics all.") in \
        [str(w.message) for w in caught]


# ======================================================================= R6-06
def test_label_order_confidence_stays_within_0_and_1():
    assert W._order_confidence([{"ARI": 0.6335}, {"ARI": -0.0009}]) == 1.0
    assert W._order_confidence([{"ARI": 0.688}, {"ARI": 0.1101}]) == 0.84
    rec = {"method": "UINMF", "status": "CHAIN_OK", "labels_used": ["cty1.csv", "cty2.csv"],
           "metrics": {"ARI": 0.6335},
           "label_order_candidates": [{"order": ["cty1.csv", "cty2.csv"], "ARI": 0.6335},
                                      {"order": ["cty2.csv", "cty1.csv"], "ARI": -0.0009}]}
    sm = W.BatchResult([rec], "D52", "cross").summary
    assert sm["label_order_confidence"].iloc[0] == 1.0


# ================================================================ docstrings
def test_docstrings_state_the_new_rules():
    flat = {k: " ".join(inspect.getdoc(o).split()) for k, o in {
        "rescore": W.BatchResult.rescore, "load_batch": mtb.load_batch,
        "summary": W.BatchResult.summary, "results": W.BatchResult.results,
        "save": W.BatchResult.save}.items()}
    assert ("With ``labels=None`` and several label files, ``rescore`` ranks the file "
            "orders by ARI, which needs one Leiden sweep. It keeps the stored order and "
            "skips the sweep when ``metrics=`` has no ARI, NMI or iF1.") in flat["rescore"]
    assert "is handed to ``evaluate(metrics=)``" not in flat["rescore"]
    assert ("``None`` = the batch ``run_all`` was given, else each cell's label "
            "file.") in flat["rescore"]
    assert "the batch that ``run_all(batch=)`` saved is reused" in flat["rescore"]
    assert "data_path : path-like | None Data root for re-scoring; ``None`` = each " \
        "record's own." in flat["load_batch"]
    assert "``(best - max(runner_up, 0)) / best``" in flat["summary"]
    assert "data_root" in flat["results"] and "batch_file" in flat["results"]
    assert "``batch_<hash>.csv``" in flat["save"]
    assert "data_path" in inspect.signature(mtb.load_batch).parameters
