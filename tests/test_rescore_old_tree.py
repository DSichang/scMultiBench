"""A saved tree whose records have no ``out_dir`` (written by an older multibench):
``load_batch`` points each record at ``<tree>/<method>_<dataset>``, and ``rescore``
raises when it finds no output to score instead of returning an empty table."""
import json

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb

N = 60
CELLS = [f"c{i}" for i in range(N)]
TYPES = np.repeat(["T", "B", "NK"], N // 3)


def _h5(path, features, data):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=data)
        f.create_dataset("matrix/features", data=np.array(features, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(CELLS, dtype="S"))


def _tree(tmp_path, *, with_outputs=True):
    """``data/MYCITE`` (vertical RNA + ADT) and a run tree of totalVI and
    Matilda whose records carry no ``out_dir``."""
    rng = np.random.default_rng(0)
    ds = tmp_path / "data" / "MYCITE"
    ds.mkdir(parents=True)
    _h5(ds / "rna.h5", [f"g{i}" for i in range(20)], rng.poisson(2, (20, N)).astype(float))
    _h5(ds / "adt.h5", [f"p{i}" for i in range(5)], rng.poisson(5, (5, N)).astype(float))
    pd.DataFrame({"x": TYPES}).to_csv(ds / "cty.csv", index=False)
    tree = tmp_path / "tree"
    tree.mkdir()
    records = []
    for method in ("totalVI", "Matilda"):
        records.append({"method": method, "category": "vertical", "dataset": "MYCITE",
                        "modalities": ["rna", "adt"], "status": "CHAIN_OK",
                        "data_path": None, "metrics": {"ARI": 0.5}})
        if with_outputs:
            out = tree / f"{method}_MYCITE"
            out.mkdir()
            emb = rng.normal(size=(N, 4)) + 6 * pd.factorize(TYPES)[0][:, None]
            with h5py.File(out / "embedding.h5", "w") as f:
                f.create_dataset("data", data=emb)
    (tree / "batch_result.json").write_text(json.dumps(
        {"dataset": "MYCITE", "category": "vertical", "records": records}))
    return tree


def test_records_without_out_dir_score_from_the_folders_next_to_the_json(tmp_path, monkeypatch):
    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", tmp_path / "data")
    tree = _tree(tmp_path)
    res = mtb.load_batch(tree)
    assert [r["out_dir"] for r in res.records] == [
        str((tree / "totalVI_MYCITE").resolve()), str((tree / "Matilda_MYCITE").resolve())]
    again = res.rescore(metrics=["ARI", "NMI"], verbose=False)
    assert [r["status"] for r in again.records] == ["CHAIN_OK", "CHAIN_OK"]
    assert set(again.long["metric"]) == {"ARI", "NMI"}


def test_rescore_raises_when_it_finds_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", tmp_path / "data")
    tree = _tree(tmp_path, with_outputs=False)
    res = mtb.load_batch(tree)
    with pytest.raises(FileNotFoundError) as e:
        res.rescore(metrics=["ARI"], verbose=False)
    assert str(e.value) == (
        f"No output of totalVI and Matilda in {tree}: load the folder that holds their "
        f"method folders, or run them again.")


def test_a_method_without_output_is_recorded_when_another_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", tmp_path / "data")
    tree = _tree(tmp_path)
    (tree / "Matilda_MYCITE" / "embedding.h5").unlink()
    again = mtb.load_batch(tree).rescore(metrics=["ARI"], verbose=False)
    assert [r["status"] for r in again.records] == ["CHAIN_OK", "RUN_OK_EVAL_FAILED"]
    assert "embedding.h5 is missing: run the method again." in again.records[1]["error"]
