"""Journal review, round 1: run, scan and RunResult."""
import ast
import json
from pathlib import Path

import h5py
import numpy as np
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
