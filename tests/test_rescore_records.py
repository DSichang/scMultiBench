"""What a rescored or saved record says about itself: the label orders ranked on
the current scoring, one ordering for identical label files, the plan's caveat
as this version words it, the version that ran next to the scorer, and an
``out_dir`` relative to the saved folder."""
import os
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs
from tests.test_study_r4_wf import ALL_ENVS, _h5, _quiet, _Res

pytest.importorskip("scib")


def _tree(tmp_path, monkeypatch, same_labels=False):
    """A three-batch RNA+ADT folder and one faked totalVI run of it."""
    d = tmp_path / "data" / "MYCROSS"
    d.mkdir(parents=True)
    rng = np.random.default_rng(3)
    first = rng.choice(["A", "B", "C"], 50)
    labs = {}
    for b in (1, 2, 3):
        bars = [f"b{b}c{i}" for i in range(50)]
        _h5(d / f"rna{b}.h5", [f"g{i}" for i in range(30)], bars)
        _h5(d / f"adt{b}.h5", [f"p{i}" for i in range(6)], bars)
        labs[b] = first if same_labels else rng.choice(["A", "B", "C"], 50)
        pd.DataFrame({"x": labs[b]}).to_csv(d / f"cty{b}.csv", index=False)
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)

    def fake_run(method, category, inputs, out_dir, params=None):
        lab = np.concatenate([labs[b] for b in (1, 2, 3)])
        emb = np.random.default_rng(7).normal(size=(len(lab), 4))
        emb[:, 0] += 8.0 * (lab == "B")
        emb[:, 1] += 8.0 * (lab == "C")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        return _Res(emb)
    monkeypatch.setattr(W, "_run", fake_run)
    out = tmp_path / "out"
    mtb.config.DEFAULT.data_path = tmp_path / "data"
    res = _quiet(mtb.run_all, 'MYCROSS', 'cross', out, methods=['totalVI'], verbose=False)
    return res, out


def test_identical_label_files_are_one_ordering(tmp_path, monkeypatch):
    # Y12: six orders of three identical files are one ordering: no 0.0 confidence
    res, _ = _tree(tmp_path, monkeypatch, same_labels=True)
    rec = res.records[0]
    assert rec["labels_used"] == ["cty1.csv", "cty2.csv", "cty3.csv"]
    assert "label_order_candidates" not in rec


def test_rescore_words_the_caveat_as_this_version_does(tmp_path, monkeypatch):
    # X81: a caveat of the version that ran is not carried over
    _, out = _tree(tmp_path, monkeypatch)
    p = out / "batch_result.json"
    blob = json.loads(p.read_text())
    blob["records"][0]["caveat"] = "totalVI needs about 5,000 cells in total."
    p.write_text(json.dumps(blob))
    new = _quiet(mtb.load_batch(out).rescore, metrics=["ARI"], verbose=False)
    plan = mtb.run_all(os.path.join(tmp_path / 'data', 'MYCROSS'), 'cross', methods=['totalVI'], dry_run=True, verbose=False)
    assert new.records[0]["caveat"] == W._run_caveat(plan["caveat"].iloc[0])
    assert "5,000" not in new.records[0]["caveat"]


def test_a_rescored_record_keeps_the_version_that_ran(tmp_path, monkeypatch):
    # Y31: multibench_version is the run's; scored_with names the scorer
    _, out = _tree(tmp_path, monkeypatch)
    p = out / "batch_result.json"
    blob = json.loads(p.read_text())
    blob["records"][0]["multibench_version"] = "0.3.9"
    p.write_text(json.dumps(blob))
    new = _quiet(mtb.load_batch(out).rescore, metrics=["ARI"], verbose=False)
    rec = new.records[0]
    assert rec["multibench_version"] == "0.3.9"
    assert rec["scored_with"] == new.long["scored_with"].iloc[0]
    assert rec["scored_with"].endswith(f"/{mtb.__version__}")


def test_save_writes_out_dir_relative_to_the_folder(tmp_path, monkeypatch):
    # Z02: a saved or rescored tree holds no host path of its own outputs
    _, out = _tree(tmp_path, monkeypatch)
    back = mtb.load_batch(out)
    _quiet(back.rescore, metrics=["ARI"], verbose=False).save(out)
    saved = json.loads((out / "batch_result.json").read_text())["records"][0]
    assert saved["out_dir"] == "totalVI_MYCROSS"
    moved = tmp_path / "moved"
    shutil.move(str(out), str(moved))
    again = mtb.load_batch(moved)
    assert again.records[0]["out_dir"] == str((moved / "totalVI_MYCROSS").resolve())
