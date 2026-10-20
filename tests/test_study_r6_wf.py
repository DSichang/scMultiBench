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
# 0.4.0: load_batch(data_path=) is deprecated; the fix names config.DEFAULT.data_path
MOVED = ("Set mtb.config.DEFAULT.data_path to the folder that holds it, or run rescore "
         "from the folder where run_all ran.")


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
    rec = res.records[0]
    assert rec["data_path"] == "data"
    assert Path(rec["data_root"]).is_absolute()
    assert Path(rec["data_root"]).samefile(root / "data")
    saved = json.loads((root / "out" / "batch_result.json").read_text())["records"][0]
    assert saved["data_root"] == rec["data_root"]


def test_rescore_from_another_directory_aligns_the_labels_by_barcode(rel, monkeypatch):
    res, root = rel
    at_root = _quiet(mtb.load_batch("out").rescore, labels=_shuffled_labels())
    assert "+".join(at_root.records[0]["labels_used"]) == "cty.csv"
    assert _ari(at_root) == _ari(res) > 0.3
    monkeypatch.chdir(root / "nb")
    back = mtb.load_batch("../out")
    assert Path(back.records[0]["data_path"]).samefile(root / "data")
    new, caught = _rescore(back, labels=_shuffled_labels())
    assert not _by_position(caught), caught
    assert "+".join(new.records[0]["labels_used"]) == "cty.csv"
    assert _ari(new) == _ari(at_root)


def test_rescore_of_an_in_memory_result_after_a_chdir(rel, monkeypatch):
    _, root = rel
    res = _quiet(mtb.run_all, "MYCITE", "vertical", root / "abs", data_path="data", **KW)
    monkeypatch.chdir(root / "nb")
    new, caught = _rescore(res, labels=_shuffled_labels())
    assert not _by_position(caught), caught
    assert "+".join(new.records[0]["labels_used"]) == "cty.csv" and _ari(new) == _ari(res)


def test_a_copied_result_is_pointed_at_its_data_with_data_path(rel, monkeypatch):
    res, root = rel
    copy = root / "elsewhere" / "copy"
    shutil.copytree(root / "out", copy)
    monkeypatch.chdir(root / "nb")
    new = _quiet(mtb.load_batch(copy, data_path=root / "data").rescore,
                 labels=_shuffled_labels())
    assert "+".join(new.records[0]["labels_used"]) == "cty.csv" and _ari(new) == _ari(res)
    # the data moved too: only the argument finds it
    shutil.move(str(root / "data"), str(root / "moved"))
    back = mtb.load_batch(copy, data_path=root / "moved")
    assert back.records[0]["data_path"] == str(root / "moved")
    assert back.records[0]["out_dir"] == str(copy / "Matilda_MYCITE")
    new, caught = _rescore(back, labels=_shuffled_labels())
    assert not _by_position(caught), caught
    assert "+".join(new.records[0]["labels_used"]) == "cty.csv" and _ari(new) == _ari(res)
    assert _ari(_quiet(back.rescore)) == _ari(res)


def test_a_folder_that_is_not_found_is_named(rel, monkeypatch):
    res, root = rel
    shutil.move(str(root / "data"), str(root / "moved"))
    monkeypatch.chdir(root / "nb")
    back = mtb.load_batch("../out")
    assert back.records[0]["data_path"] == "data"         # left as recorded
    # the roots tried: the recorded data_path from here, then data_root
    where = f"{Path.cwd() / 'data'} or {back.records[0]['data_root']}"
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
    rec = new.records[0]
    assert rec["status"] == "RUN_OK_NO_LABEL_MATCH"
    assert rec["note"] == f"The dataset folder MYCITE is not found in {where}. {MOVED}"
    for k in ("metrics", "batch_source", "n_batches", "labels_used"):
        assert k not in rec, k
    assert new.summary.iloc[0]["status"] == "ran"


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
    assert again.records[0]["data_path"] == again.records[0]["data_root"] == \
        str((root / "data").resolve())
    assert "+".join(again.records[0]["labels_used"]) == "cty.csv" and _ari(again) == _ari(res)
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
    assert back.records[0]["data_path"] is None
    new = _quiet(back.rescore)
    assert new.records[0]["data_path"] is None
    assert "+".join(new.records[0]["labels_used"]) == "cty.csv" and _ari(new) == _ari(res)


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
    rec = res.records[0]
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
        rec = new.records[0]
        assert (rec["batch_source"], rec["n_batches"]) == ("user", 2), kw
        assert _batch_metrics(new) == before, kw
        assert not [m for m in caught if "batch vector" in m], caught
    # the reference example: the re-scored result keeps the batch on disk
    _quiet(back.rescore, labels=_shuffled_labels(2)).save(tmp / "rescored")
    assert (tmp / "rescored" / res.records[0]["batch_file"]).is_file()
    again = _quiet(mtb.load_batch(tmp / "rescored").rescore)
    assert _batch_metrics(again) == before
    # a batch that is given wins, and is saved in its turn
    ones = _quiet(back.rescore, batch=np.ones(N, dtype=int))
    assert ones.records[0]["n_batches"] == 1
    assert ones.records[0]["batch_file"] != res.records[0]["batch_file"]
    ones.save(tmp / "ones")
    assert _quiet(mtb.load_batch(tmp / "ones").rescore).records[0]["n_batches"] == 1


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
    assert new.records[0]["n_batches"] == 1
    failed, caught = _rescore(old, metrics="all")
    assert caught.count(text) == 1, caught
    rec = failed.records[0]
    assert rec["status"] == "RUN_OK_EVAL_FAILED"
    assert ("metrics='all' needs batch labels for ASW_batch, ARI_batch, NMI_batch, GC and iLISI. "
            "Pass batch=<vector>, or metrics='clustering'.") in rec["error"]
    for k in ("batch_source", "n_batches", "metrics"):
        assert k not in rec, k
    assert failed.summary.iloc[0]["status"] == "ran"


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


def _cross_run(data, tmp, method, **kw):
    return _quiet(mtb.run_all, "MYCROSS", "cross", tmp / method, data_path=data,
                  methods=[method], verbose=False, **kw)


def _count_sweeps(monkeypatch, fail=False):
    calls = []
    real = W._escib.leiden_n_clusters

    def counted(*a, **k):
        calls.append(1)
        if fail:
            raise AssertionError("leiden_n_clusters was called")
        return real(*a, **k)
    monkeypatch.setattr(W._escib, "leiden_n_clusters", counted)
    return calls


def test_rescore_keeps_the_recorded_label_order(cross, tmp_path, monkeypatch):
    # c2 (D28): the order the record was scored with is not ranked again
    res = _cross_run(cross, tmp_path, "totalVI")
    rec = res.records[0]
    assert rec["labels_used"] == ["cty1.csv", "cty2.csv", "cty3.csv"]
    assert len(rec["label_order_candidates"]) == 6
    back = mtb.load_batch(tmp_path / "totalVI")
    calls = _count_sweeps(monkeypatch)
    new = back.rescore(metrics=["ASW"])
    assert len(calls) == 0
    out = new.records[0]
    assert out["status"] == "CHAIN_OK"
    assert out["metrics"]["ASW"] == rec["metrics"]["ASW"]
    assert out["labels_used"] == rec["labels_used"]
    assert out["label_order_candidates"] == rec["label_order_candidates"]
    assert [(r["batch_source"], r["n_batches"]) for r in new.records] == \
        [(r["batch_source"], r["n_batches"]) for r in res.records]


def test_rescore_keeps_the_recorded_candidates(cross, tmp_path, monkeypatch, capsys):
    # c2 (D28): no ranking line; the record keeps its candidates
    _cross_run(cross, tmp_path, "totalVI")
    p = tmp_path / "totalVI" / "batch_result.json"
    blob = json.loads(p.read_text())
    old = [{"order": ["cty1.csv", "cty2.csv", "cty3.csv"], "ARI": 0.99},
           {"order": ["cty2.csv", "cty1.csv", "cty3.csv"], "ARI": 0.01}]
    blob["records"][0]["label_order_candidates"] = old
    p.write_text(json.dumps(blob))
    new = _quiet(mtb.load_batch(tmp_path / "totalVI").rescore, metrics=["ARI"],
                 verbose=True)
    out = new.records[0]
    assert out["label_order_candidates"] == old
    assert out["labels_used"] == ["cty1.csv", "cty2.csv", "cty3.csv"]
    assert "ranking" not in capsys.readouterr().out


def test_rescore_ranks_the_orders_of_a_record_without_one(cross, tmp_path, monkeypatch,
                                                          capsys):
    _cross_run(cross, tmp_path, "totalVI")
    p = tmp_path / "totalVI" / "batch_result.json"
    blob = json.loads(p.read_text())
    blob["records"][0].pop("labels_used")
    p.write_text(json.dumps(blob))
    calls = _count_sweeps(monkeypatch)
    _quiet(mtb.load_batch(tmp_path / "totalVI").rescore, metrics=["ARI"], verbose=True)
    assert len(calls) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ("[rescore] totalVI: ranking 6 label orders on 150 cells with "
                        "one Leiden clustering ...")


def test_a_stored_order_that_no_longer_fits_is_ranked_again(cross, tmp_path, monkeypatch):
    _cross_run(cross, tmp_path, "totalVI")
    p = tmp_path / "totalVI" / "batch_result.json"
    blob = json.loads(p.read_text())
    blob["records"][0]["labels_used"] = ["cty1.csv", "cty2.csv"]     # 90 of 150 cells
    p.write_text(json.dumps(blob))
    calls = _count_sweeps(monkeypatch)
    new = _quiet(mtb.load_batch(tmp_path / "totalVI").rescore, metrics=["ASW"])
    assert len(calls) == 1
    assert new.records[0]["labels_used"] == ["cty1.csv", "cty2.csv", "cty3.csv"]


def test_rescore_does_not_warn_about_its_own_file_of_origin_batch(cross, tmp_path):
    # UINMF's two-batch variant: a run that reads two of the folder's three batches
    res = _cross_run(cross, tmp_path, "UINMF", modalities=["rna1", "rna2", "adt1", "adt2"])
    rec = res.records[0]
    assert len(rec["label_order_candidates"]) == 2 and rec["n_batches"] == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        new = res.rescore(metrics=["ARI", "NMI"])
    out = new.records[0]
    assert out["metrics"]["ARI"] == rec["metrics"]["ARI"]
    assert (out["batch_source"], out["n_batches"]) == ("file_of_origin", 2)
    # a batch the caller gives still warns, in short sentences
    with pytest.warns(UserWarning) as caught:
        res.rescore(batch=np.repeat([1, 2], [40, 50]), metrics=["ARI"])
    assert ("batch= changes nothing here, because metrics=['ARI'] has no batch metric. "
            "Add ASW_batch, ARI_batch, NMI_batch, GC or iLISI, or pass metrics='all'.") in \
        [str(w.message) for w in caught]


# ================================================================ docstrings
def test_docstrings_state_the_new_rules():
    flat = {k: " ".join(inspect.getdoc(o).split()) for k, o in {
        "rescore": W.BatchResult.rescore, "load_batch": mtb.load_batch,
        "save": W.BatchResult.save}.items()}
    assert "is handed to ``evaluate(metrics=)``" not in flat["rescore"]
    assert ("``None`` = the batch ``run_all`` was given, else each cell's label "
            "file.") in flat["rescore"]
    # R7-12 (b); 0.4.0: data_path= is deprecated, the Notes give the search order
    assert "data_path :" not in flat["load_batch"]
    assert ("the dataset folder under the recorded data root, then "
            "``mtb.config.DEFAULT.data_path``.") in flat["load_batch"]
    assert "``batch_<hash>.csv``" in flat["save"]
    assert "data_path" not in inspect.signature(mtb.load_batch).parameters
