"""Journal review, round 1: run, scan and RunResult."""
import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


def test_engine_and_eval_never_import_workflow():
    """workflow is built on engine and eval; neither imports it back."""
    pkg = Path(mtb.__file__).parent
    for path in sorted([*pkg.glob("engine/*.py"), *pkg.glob("eval/*.py")]):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] if node.module else [a.name for a in node.names]
                top = [n.split(".")[0] for n in names]
                assert not (node.level == 2 and "workflow" in top), (path.name, node.lineno)
                assert not (node.level == 0 and str(node.module).startswith(
                    "multibench.workflow")), (path.name, node.lineno)
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith("multibench.workflow") for a in node.names), \
                    (path.name, node.lineno)


def test_embedding_names_a_missing_dataset_folder(tmp_path, monkeypatch):
    """u7 D3: a run folder copied to a laptop that does not have the dataset
    folder; the label files are not the problem."""
    out = tmp_path / "out"
    (out / "Matilda_MYD").mkdir(parents=True)
    with h5py.File(out / "Matilda_MYD" / "embedding.h5", "w") as f:
        f.create_dataset("data", data=np.zeros((30, 4)))
    rec = {"method": "Matilda", "category": "vertical", "dataset": "MYD", "status": "scored",
           "modalities": ["rna", "adt"], "out_dir": "Matilda_MYD", "labels_used": ["cty.csv"],
           "data_path": str(tmp_path / "server")}
    (out / "batch_result.json").write_text(json.dumps(
        {"dataset": "MYD", "category": "vertical", "records": [rec]}))
    monkeypatch.setattr(mtb.config, "data_path", tmp_path / "laptop")
    res = mtb.load_run(out)
    with pytest.warns(UserWarning) as caught:
        obs = res.embedding("Matilda").obs
    text = " ".join(str(w.message) for w in caught)
    assert f"MYD is not found in {tmp_path / 'server'}" in text
    assert "mtb.config.data_path" in text and "label files" not in text
    assert len(obs) == 30


def test_scan_says_a_file_problem_once(tmp_path, capsys):
    """u8: a transposed rna.h5 was printed once per method (14 identical lines);
    the reason column keeps it for each method."""
    d = tmp_path / "TRANSPOSED"
    d.mkdir()
    rng = np.random.default_rng(0)
    for name, n_feat in (("rna.h5", 40), ("adt.h5", 8)):
        with h5py.File(d / name, "w") as f:
            g = f.create_group("matrix")
            g.create_dataset("data", data=rng.poisson(3.0, (120, n_feat)).astype(float))
            g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(120)], dtype="S8"))
            g.create_dataset("features", data=np.array([f"f{i}" for i in range(n_feat)],
                                                       dtype="S8"))
    pd.DataFrame({"x": ["A", "B"] * 60}).to_csv(d / "cty.csv", index=False)
    plan = mtb.scan(str(d), "vertical")
    lines = [line for line in capsys.readouterr().out.splitlines() if "cells x features" in line]
    assert len(lines) == 1 and lines[0].startswith("[scan] rna.h5 stores"), lines
    assert len(plan) > 1 and plan["reason"].str.contains("rna.h5 stores").all()


def test_run_refuses_an_unknown_params_key_before_any_method_runs(tmp_path, monkeypatch):
    """u4 D5: a typo in the params of the second method of a sweep was found
    only after the first method ran; run raises scan's error first."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    started = []
    monkeypatch.setattr(W, "_run", lambda **k: started.append(k["method"]))
    params = {"scMM": {"epoch": 3}}
    with pytest.raises(KeyError) as scanned:
        mtb.scan("D11", "vertical", methods=["Matilda", "scMM"], params=params, verbose=False)
    with pytest.raises(KeyError) as ran:
        mtb.run("D11", "vertical", tmp_path / "out", methods=["Matilda", "scMM"],
                params=params, verbose=False)
    assert ran.value.args == scanned.value.args
    assert started == [] and not (tmp_path / "out").exists()


def _fake_method(calls, *, stop_after=None):
    """A method that writes an embedding.h5 whose rows separate the cell types
    of its label files; with ``stop_after``, the job is killed when it starts
    the method after that many (KeyboardInterrupt: not a method failure)."""
    from multibench.engine import resolve
    from multibench.eval import io as eio

    def method(method, category, inputs, out_dir, params=None):
        if stop_after is not None and len(calls) >= stop_after:
            raise KeyboardInterrupt
        calls.append(method)
        folder = Path(str(next(p for p in inputs.values() if str(p).endswith(".h5")))).parent
        files = resolve.labels_for(folder.name, category, method, data_path=folder.parent)
        lab = np.concatenate([eio.read_labels(p) for p in files.values()]).astype(str)
        emb = np.random.default_rng(0).normal(scale=0.3, size=(len(lab), 6))
        emb[np.arange(len(lab)), pd.factorize(lab)[0] % 6] += 4.0
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        return emb
    return method


def test_a_run_killed_after_one_method_keeps_its_scores(tmp_path, monkeypatch):
    """u7 D1: the records were written once, after the last method, so a job
    killed at its wall-time limit kept no record of the methods that finished."""
    if not (Path(mtb.config.data_path) / "D11").is_dir():
        pytest.skip("D11 is not on disk")
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    out, calls = tmp_path / "out", []
    monkeypatch.setattr(W, "_run", _fake_method(calls, stop_after=1))
    with pytest.raises(KeyboardInterrupt):
        mtb.run("D11", "vertical", out, methods=["Matilda", "scMM"], verbose=False)
    assert calls == ["Matilda"]
    saved = mtb.load_run(out)
    assert [(r["method"], r["status"]) for r in saved.records] == [("Matilda", "scored")]
    assert (out / "summary.csv").is_file() and set(saved.long["method"]) == {"Matilda"}
    # the job again: Matilda's record is reused, only scMM runs
    calls.clear()
    monkeypatch.setattr(W, "_run", _fake_method(calls))
    res = mtb.run("D11", "vertical", out, methods=["Matilda", "scMM"], skip_existing=True,
                  verbose=False)
    assert calls == ["scMM"]
    first = {r["method"]: r for r in res.records}["Matilda"]
    assert first["reused"] is True and first["run_sec"] == saved.records[0]["run_sec"]
    assert list(mtb.load_run(out).summary["status"]) == ["scored", "scored"]
