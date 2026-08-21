"""P15: run_all(batch=) override, BatchResult.rescore(), summary batch_source/n_batches.

A synthetic 120-cell CITE-seq folder and a fake `_run` that WRITES embedding.h5
(so rescore can read it back) keep these independent of any conda env.
"""
import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
N = 120


def _h5(path, n_feat, n_cells, prefix):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=rng.poisson(2.0, size=(n_feat, n_cells)).astype(float))
        g.create_dataset("features", data=np.array([f"{prefix}{i}" for i in range(n_feat)], dtype="S12"))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(n_cells)], dtype="S12"))


@pytest.fixture
def cite(tmp_path, monkeypatch):
    """data_path with MYCITE (rna/adt/cty), envs mocked present, _run faked."""
    d = tmp_path / "data" / "MYCITE"
    d.mkdir(parents=True)
    _h5(d / "rna.h5", 30, N, "g")
    _h5(d / "adt.h5", 6, N, "p")
    labels = np.array(["A"] * (N // 2) + ["B"] * (N // 2))
    pd.DataFrame({"x": labels}).to_csv(d / "cty.csv", index=False)
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)

    class _Res:
        def __init__(self, out):
            self.output = out

    def _fake_run(method, category, inputs, out_dir, params=None):
        rng = np.random.default_rng(7)
        emb = rng.normal(size=(N, 4))
        emb[N // 2:, 0] += 8.0                       # two clean clusters = the labels
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)   # run() creates it
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)     # dims x cells, like many methods
        return _Res(emb)

    monkeypatch.setattr(W, "_run", _fake_run)
    return tmp_path / "data", labels


def test_run_all_default_batch_is_file_of_origin_or_none(cite, tmp_path):
    data, labels = cite
    res = mtb.run_all("MYCITE", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                      out_dir=tmp_path / "out", data_path=data, verbose=False)
    sm = res.summary
    assert sm.loc[0, "status"] == "CHAIN_OK"
    assert "batch_source" in sm.columns and "n_batches" in sm.columns
    assert sm.loc[0, "batch_source"] is None and sm.loc[0, "n_batches"] == 1
    assert sm.loc[0, "ARI"] > 0.9
    assert "ASW_batch" not in sm.columns           # single label file: clustering only
    # .long comes from the attached tidy frame, and the derived fallback agrees
    lng = res.long
    assert set(lng["metric"]) >= {"ARI", "NMI"}
    for r in res.records:
        r["_long"] = None
    derived = res.long
    assert set(derived["metric"]) == set(lng["metric"])
    np.testing.assert_allclose(derived.set_index("metric").loc["ARI", "value"],
                               round(float(lng.set_index("metric").loc["ARI", "value"]), 4))


def test_run_all_batch_override_scores_batch_metrics(cite, tmp_path):
    data, labels = cite
    batch = np.tile([1, 2], N // 2)                 # two batches, orthogonal to the labels
    res = mtb.run_all("MYCITE", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                      out_dir=tmp_path / "out", data_path=data, verbose=False, batch=batch)
    sm = res.summary
    assert sm.loc[0, "status"] == "CHAIN_OK"
    assert sm.loc[0, "batch_source"] == "user" and sm.loc[0, "n_batches"] == 2
    assert "ASW_batch" in sm.columns and pd.notna(sm.loc[0, "ASW_batch"])
    assert sm.loc[0, "ARI"] > 0.9
    # survives save/load_batch
    back = mtb.load_batch(tmp_path / "out")
    assert back.summary.loc[0, "batch_source"] == "user"


def test_run_all_batch_wrong_length_is_eval_failed_not_fatal(cite, tmp_path):
    data, _ = cite
    res = mtb.run_all("MYCITE", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                      out_dir=tmp_path / "out", data_path=data, verbose=False,
                      batch=np.ones(N - 1))
    sm = res.summary
    assert sm.loc[0, "status"] == "RUN_OK_EVAL_FAILED"
    err = res.failures.iloc[0]["error"]
    assert f"batch has {N - 1} entries, embedding has {N} cells" in err


def test_rescore_with_batch_and_labels(cite, tmp_path):
    data, labels = cite
    res = mtb.run_all("MYCITE", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                      out_dir=tmp_path / "out", data_path=data, verbose=False)
    assert res.summary.loc[0, "batch_source"] is None
    batch = np.tile([1, 2], N // 2)
    re = res.rescore(batch=batch)
    assert isinstance(re, mtb.BatchResult) and re is not res
    assert re.summary.loc[0, "batch_source"] == "user" and re.summary.loc[0, "n_batches"] == 2
    assert "ASW_batch" in re.summary.columns
    assert re.summary.loc[0, "ARI"] == res.summary.loc[0, "ARI"]   # same labels, same ARI
    # the original is untouched
    assert res.summary.loc[0, "batch_source"] is None
    # user labels bypass the label-order search
    shuffled = labels[::-1]
    re2 = res.rescore(labels=shuffled, only={"ARI", "NMI"})
    assert re2.summary.loc[0, "label_order"] == "(user labels)"
    assert set(re2.long["metric"]) == {"ARI", "NMI"}
    # a batch CSV path is accepted too (read like a label file)
    bp = tmp_path / "batch.csv"
    pd.DataFrame({"x": batch}).to_csv(bp, index=False)
    assert res.rescore(batch=bp).summary.loc[0, "n_batches"] == 2
    # wrong length -> RUN_OK_EVAL_FAILED with the precise message, not an exception
    bad = res.rescore(batch=np.ones(3))
    assert bad.summary.loc[0, "status"] == "RUN_OK_EVAL_FAILED"
    assert "batch has 3 entries" in bad.records[0]["error"]
    # not persisted until asked; .save() writes the new numbers
    assert mtb.load_batch(tmp_path / "out").summary.loc[0, "batch_source"] is None
    re.save(tmp_path / "out2")
    assert mtb.load_batch(tmp_path / "out2").summary.loc[0, "batch_source"] == "user"


def test_rescore_handles_missing_output(cite, tmp_path):
    data, _ = cite
    res = mtb.run_all("MYCITE", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                      out_dir=tmp_path / "out", data_path=data, verbose=False)
    (tmp_path / "out" / "Matilda_MYCITE" / "embedding.h5").unlink()
    re = res.rescore()
    assert re.summary.loc[0, "status"] == "RUN_OK_EVAL_FAILED"
    assert "no output" in re.records[0]["error"]


def test_rescore_is_documented():
    import inspect
    doc = inspect.getdoc(mtb.BatchResult.rescore)
    for w in ("batch", "labels", "only", "save", "RUN_OK_EVAL_FAILED"):
        assert w in doc
    doc = inspect.getdoc(mtb.BatchResult.summary.fget)
    assert "batch_source" in doc and "n_batches" in doc and "file_of_origin" in doc
