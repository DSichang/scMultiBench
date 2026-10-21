"""Round d1 of the 0.4.0 review: one check (scan) and one run, RunResult, cell
names in embedding.h5, the record of a run folder, one status vocabulary, and
the messages the users could not act on."""
import json
import os
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import _settings, config, workflow as W
from multibench.engine import envs, registry, resolve, scripts
from multibench.eval import io as eio

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


class _Res:
    def __init__(self, out):
        self.output = out


@pytest.fixture
def host(monkeypatch, tmp_path):
    """Linux with every environment; the fake method writes an embedding.h5 in
    the cell order of the method's label files (dims x cells, as R writes it),
    whose rows separate the cell types."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(_settings, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(scripts.SCRIPTS_REF_VAR, raising=False)
    calls = []

    def fake_run(method, category, inputs, out_dir, params=None):
        calls.append(method)
        folder = Path(str(next(p for r, p in inputs.items() if str(p).endswith(".h5")))).parent
        files = resolve.labels_for(folder.name, category, method, data_path=folder.parent)
        lab = np.concatenate([eio.read_labels(p) for p in files.values()]).astype(str)
        codes = pd.factorize(lab)[0]
        emb = np.random.default_rng(0).normal(scale=0.3, size=(len(lab), 6))
        emb[np.arange(len(lab)), codes % 6] += 4.0
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        return _Res(emb)
    monkeypatch.setattr(W, "_run", fake_run)
    return calls


def _on_disk(ds):
    if not (Path(config.data_path) / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")


def _names(path):
    return list(mtb.io.read_canonical(path).obs_names)


# ------------------------------------------------------------------ one check, one run
def test_scan_returns_the_plan_and_run_has_no_dry_run():
    plan = mtb.scan("D11", "vertical", methods=["Matilda"],
                    params={"Matilda": {"epochs": 7}}, verbose=False)
    assert list(plan.columns) == ["method", "modalities", "runnable", "reason", "env",
                                  "caveat", "command"]
    assert "--epochs 7" in plan["command"].iloc[0]
    assert f"{W.OUT_DIR_PLACEHOLDER}/Matilda_D11" in plan["command"].iloc[0]
    with pytest.raises(TypeError):
        mtb.run("D11", "vertical", methods=["Matilda"], dry_run=True)
    with pytest.raises(KeyError, match="does not accept"):
        mtb.scan("D11", "vertical", methods=["Matilda"], params={"Matilda": {"nope": 1}},
                 verbose=False)


def test_the_run_result_names():
    assert mtb.RunResult.__name__ == "RunResult" and callable(mtb.load_run)
    assert not hasattr(mtb, "BatchResult") and not hasattr(mtb, "load_batch")


# ------------------------------------------------------------------ cell names in embedding.h5
def test_run_writes_the_cell_names_into_embedding_h5(host, tmp_path):
    _on_disk("D52mini")
    res = _quiet(mtb.run, "D52mini", "cross", tmp_path / "out", methods=["StabMap"],
                 verbose=False)
    (rec,) = res.records
    assert rec["status"] == "scored"
    path = Path(rec["out_dir"]) / "embedding.h5"
    with h5py.File(path) as f:
        names = [b.decode() for b in f[W.CELLS_DATASET][:]]
    emb = res.embedding("StabMap")
    assert names == list(emb.obs_names) == _names(path)
    # StabMap puts its reference batch first: the names follow it
    assert names[0].startswith("cty3")


def test_evaluate_of_the_file_does_not_depend_on_method(host, tmp_path):
    """u5: a free name gave cLISI 0.7775 against 0.9939; with the cell names in
    the file, every name scores the same."""
    _on_disk("D52mini")
    res = _quiet(mtb.run, "D52mini", "cross", tmp_path / "out", methods=["StabMap"],
                 verbose=False)
    src = Path(res.records[0]["out_dir"]) / "embedding.h5"
    alone = tmp_path / "elsewhere" / "emb.h5"          # out of its run folder
    alone.parent.mkdir()
    shutil.copy(src, alone)
    kw = dict(dataset="D52mini", category="cross", metrics=["cLISI", "ARI"], verbose=False)
    want = _quiet(mtb.evaluate, src, method="StabMap", **kw).set_index("metric")["value"]
    for name in ("StabMap (this run)", "my_stabmap", "embedding"):
        got = _quiet(mtb.evaluate, alone, method=name, **kw).set_index("metric")["value"]
        assert got.to_dict() == pytest.approx(want.to_dict(), abs=1e-9), name


def test_a_file_without_names_takes_the_method_of_its_run_record(host, tmp_path):
    """An embedding.h5 of a run folder written before d1: the order is that of
    the method the folder's run record names, not of method=."""
    _on_disk("D52mini")
    res = _quiet(mtb.run, "D52mini", "cross", tmp_path / "out", methods=["StabMap"],
                 verbose=False)
    path = Path(res.records[0]["out_dir"]) / "embedding.h5"
    with h5py.File(path, "a") as f:
        del f[W.CELLS_DATASET]
    kw = dict(dataset="D52mini", category="cross", metrics=["cLISI"], verbose=False)
    want = _quiet(mtb.evaluate, path, method="StabMap", **kw)["value"].iloc[0]
    assert _quiet(mtb.evaluate, path, method="my_stabmap", **kw)["value"].iloc[0] == \
        pytest.approx(want, abs=1e-9)
    # out of the folder, without names, a free name reads the folder's order
    alone = tmp_path / "alone.h5"
    shutil.copy(path, alone)
    other = _quiet(mtb.evaluate, alone, method="my_stabmap", **kw)["value"].iloc[0]
    assert other < want - 0.05


# ------------------------------------------------------------------ RunResult.embedding obs
def _cross_folder(tmp_path, shared: bool):
    """A cross RNA + ADT folder of three batches; ``shared``: one barcode is in
    two batches."""
    import anndata as ad
    rng = np.random.default_rng(1)
    name = "SHARED" if shared else "UNIQUE"
    for b in (1, 2, 3):
        n = 30
        a = ad.AnnData(rng.poisson(1.0, (n, 30)).astype(float))
        a.obs_names = [f"b{b}c{i}" for i in range(n)]
        if shared and b == 2:
            a.obs_names = ["b1c0"] + [f"b{b}c{i}" for i in range(1, n)]
        a.var_names = [f"g{i}" for i in range(30)]
        a.obsm["protein"] = rng.poisson(3.0, (n, 8)).astype(float)
        a.obs["celltype"] = np.tile(["T", "B", "NK"], n // 3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mtb.io.export_dataset(a, tmp_path / "data" / name, adt="obsm:protein",
                                  labels="obs:celltype", batch_index=b, category="cross")
    return tmp_path / "data" / name


@pytest.mark.parametrize("shared", [False, True])
def test_embedding_obs_names_are_raw_barcodes_unless_they_repeat(host, tmp_path, shared):
    folder = _cross_folder(tmp_path, shared)
    res = _quiet(mtb.run, str(folder), "cross", tmp_path / "out", methods=["sciPENN"],
                 verbose=False)
    obs = res.embedding("sciPENN").obs
    assert obs.index.is_unique and list(obs.columns) == ["label", "batch"]
    if shared:
        assert all(":" in c for c in obs.index)
    else:
        assert not any(":" in c for c in obs.index)
    assert obs["batch"].nunique() == 3


def test_a_batch_column_for_the_batches_of_run(host, tmp_path):
    """One label file, two donors passed as batch=: obs gets the batch column."""
    _on_disk("D11")
    n = len(pd.read_csv(Path(config.data_path) / "D11" / "cty.csv"))
    donors = np.where(np.arange(n) % 2, "P01", "P02")
    res = _quiet(mtb.run, "D11", "vertical", tmp_path / "out", methods=["Matilda"],
                 batch=donors, verbose=False)
    obs = res.embedding("Matilda").obs
    assert obs["batch"].tolist() == donors.tolist()
    # one batch: no batch column
    res = _quiet(mtb.run, "D11", "vertical", tmp_path / "out1", methods=["Matilda"],
                 verbose=False)
    assert list(res.embedding("Matilda").obs.columns) == ["label"]


# ------------------------------------------------------------------ records
def test_records_hold_the_five_statuses_and_old_codes_are_mapped(tmp_path):
    recs = [{"method": "Matilda", "status": "CHAIN_OK", "metrics": {"ARI": 0.5}},
            {"method": "Seurat_WNN", "status": "RUN_OK_NO_EMBEDDING"},
            {"method": "totalVI", "status": "FAIL", "error": "boom"},
            {"method": "MIRA", "status": "SKIPPED", "error": "no atac"},
            {"method": "scMM", "status": "TIMEOUT_3600", "error": "late"}]
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "batch_result.json").write_text(json.dumps(
        {"dataset": "D11", "category": "vertical", "records": recs}))
    res = mtb.load_run(folder)
    assert [r["status"] for r in res.records] == ["scored", "ran", "failed", "skipped",
                                                  "timeout"]
    assert res.summary.set_index("method").loc["Seurat_WNN", "reason"] == \
        "It writes a graph, not an embedding."


def test_reused_is_true_after_skip_existing(host, tmp_path):
    _on_disk("D11")
    first = _quiet(mtb.run, "D11", "vertical", tmp_path / "out", methods=["Matilda"],
                   verbose=False)
    assert first.records[0]["reused"] is False
    again = _quiet(mtb.run, "D11", "vertical", tmp_path / "out", methods=["Matilda"],
                   skip_existing=True, verbose=False)
    assert again.records[0]["reused"] is True and host == ["Matilda"]
    assert mtb.load_run(tmp_path / "out").records[0]["reused"] is True


def test_skip_existing_leaves_no_skipped_row_in_the_saved_folder(host, tmp_path,
                                                                  monkeypatch):
    """u4: resuming a lab's folder with a method that cannot run here."""
    _on_disk("D11")
    _quiet(mtb.run, "D11", "vertical", tmp_path / "out", methods=["Matilda"], verbose=False)
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS - {envs.group_for("totalVI")})
    res = _quiet(mtb.run, "D11", "vertical", tmp_path / "out",
                 methods=["Matilda", "totalVI"], skip_existing=True, verbose=False)
    assert {r["method"]: r["status"] for r in res.records} == {"Matilda": "scored",
                                                               "totalVI": "skipped"}
    saved = mtb.load_run(tmp_path / "out")
    assert [r["method"] for r in saved.records] == ["Matilda"]


def test_save_keeps_the_order_of_the_saved_records(tmp_path, capsys):
    def res(method, ari):
        return mtb.RunResult([{"method": method, "status": "scored", "dataset": "D11",
                               "category": "vertical", "metrics": {"ARI": ari}}],
                             "D11", "vertical")
    res("totalVI", 0.1).save(tmp_path)
    res("Matilda", 0.2).save(tmp_path)
    res("totalVI", 0.3).save(tmp_path)
    assert [r["method"] for r in mtb.load_run(tmp_path).records] == ["totalVI", "Matilda"]
    assert "[save] Merged with 1 earlier record" in capsys.readouterr().out


# ------------------------------------------------------------------ messages
def test_the_diagonal_numbered_batches_message_names_the_missing_files(tmp_path):
    import anndata as ad
    rng = np.random.default_rng(0)
    d = tmp_path / "DIAG2"
    for b in (1, 2):
        rna = ad.AnnData(rng.poisson(1, (30, 20)).astype(float))
        rna.obs_names = [f"r{b}_{i}" for i in range(30)]
        rna.var_names = [f"G{i}" for i in range(20)]
        rna.obs["celltype"] = ["T", "B"] * 15
        atac = rna.copy()
        atac.obs_names = [f"a{b}_{i}" for i in range(30)]
        mtb.io.export_dataset(rna, d, atac=atac, atac_kind="gene_activity",
                              labels="obs:celltype", category="diagonal", batch_index=b)
    hint = resolve._per_batch_hint(d, "diagonal")
    assert "rna3.h5 and atac_gas3.h5 are missing" in hint
    assert "batch=" not in hint.split("run(batch=")[0]       # one action: run(batch=...)
    assert hint.count("run(batch=") == 1 and "Export without" not in hint


def test_no_pca_says_to_make_one():
    import anndata as ad
    a = ad.AnnData(np.zeros((10, 3)))
    a.obsm["protein"] = np.zeros((10, 2))
    with pytest.raises(ValueError, match=r"has no obsm\['X_pca'\]\. Make the PCA first, as "
                                         r"sc\.pp\.pca\(adata\)\.$"):
        mtb.evaluate(a, labels=["a", "b"] * 5, obsm="X_pca")


def test_a_modality_string_gets_a_list_that_works():
    with pytest.raises(TypeError) as e:
        mtb.find_methods("vertical", modalities="rna+adt")
    hint = str(e.value).split("pass modalities=")[1].rstrip(".")
    assert mtb.find_methods("vertical", modalities=eval(hint)) == \
        mtb.find_methods("vertical", modalities=["rna", "adt"])


def test_the_missing_label_code_of_export_works(tmp_path):
    import anndata as ad
    adata = ad.AnnData(np.random.default_rng(0).poisson(1, (40, 10)).astype(float))
    for gap in (None, "NA"):
        adata.obs["cell_type"] = ["T"] * 36 + [gap] * 4
        with pytest.raises(ValueError) as e:
            mtb.io.export_dataset(adata, tmp_path / "X", labels="obs:cell_type")
        code = str(e.value).split("keep the labelled cells first: ")[1]
        ns = {"adata": adata}
        exec(code, ns)
        assert ns["adata"].n_obs == 36
        mtb.io.export_dataset(ns["adata"], tmp_path / "X", labels="obs:cell_type",
                              overwrite=True)


def test_recommend_says_which_methods_have_the_scores():
    one = pd.DataFrame({"metric": ["OCA", "ACA"], "value": [0.5, 0.6],
                        "method": "Mine", "dataset": "D28", "category": "diagonal",
                        "source": "user"})
    with pytest.raises(ValueError, match=r"^Only Mine has classification scores on D28: a "
                                         r"ranking needs 2 or more methods\. Add the scores "
                                         r"of more methods to long_df\.$"):
        mtb.recommend("diagonal", long_df=one, metrics="classification")


def test_the_install_preview_off_linux_has_no_gpu_node_note(monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "this host is darwin/arm64")
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    mtb.env.install(["SCALEX"], dry_run=True)
    assert "GPU nodes" not in capsys.readouterr().err


def test_a_nan_in_the_embedding_is_one_sentence():
    x = np.random.default_rng(0).normal(size=(40, 3))
    x[7, 1] = np.nan
    with pytest.raises(ValueError, match=r"^1 of the 40 rows of the embedding hold NaN or "
                                         r"infinite values, such as row 7: drop those cells "
                                         r"from the embedding and the labels\.$"):
        mtb.evaluate(x, labels=["a", "b"] * 20, metrics=["ARI"])


def test_read_canonical_of_another_file_is_one_sentence(tmp_path):
    (tmp_path / "cty.csv").write_text("x\nA\n")
    with pytest.raises(ValueError, match=r"^cty\.csv is not an \.h5 file\."):
        mtb.io.read_canonical(tmp_path / "cty.csv")
    with pytest.raises(ValueError, match=r"is a folder: pass one of its \.h5 files"):
        mtb.io.read_canonical(tmp_path)


def test_a_tilde_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=f"The folder {tmp_path}/nothere does not"):
        mtb.scan("~/nothere", "vertical", verbose=False)
    with pytest.raises(FileNotFoundError, match="holds no batch_result.json"):
        mtb.load_run("~/")
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        mtb.evaluate("~/emb.h5", labels=["a"])


# ------------------------------------------------------------------ cleanup
def test_the_published_table_leftovers_are_gone():
    from multibench._data import results
    from multibench.plot import style
    assert "clustering" not in results.COLUMNS and "published" not in style.STORED_SOURCES
    assert not (Path(_settings._FILES_PATH) / "dataset.csv").exists()
    assert list(mtb.catalog.datasets()["dataset"])[:5] == ["D11", "D28", "D45", "D46", "D52"]
