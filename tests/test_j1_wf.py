"""Journal review, round 1: run, scan and RunResult."""
import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb


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
