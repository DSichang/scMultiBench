"""Core round c1 (0.4.0) blockers: the label order of evaluate(dataset=, method=),
obsm=None means X_emb, and one dtype in every route."""
import importlib
import json
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, workflow
from multibench.engine import registry, resolve
from multibench.eval import io as eio, pipeline, scib as escib
from multibench.plot import style
from multibench.plot.bubble import _build_table

bar_module = importlib.import_module("multibench.plot.bar")


# --------------------------------------------------------------- C01 label order
@pytest.mark.parametrize("name, want", [
    ("StabMap", "StabMap"), ("stabmap", "StabMap"), ("StabMap (this run)", "StabMap"),
    ("StabMap epochs=10", "StabMap"), ("MOFA+", "MOFA2"), ("MOFA+ (this run)", "MOFA2"),
    ("Mine", None), ("StabMap_mine", None), ("embedding", None)])
def test_a_row_name_stands_for_a_package_method_or_none(name, want):
    assert registry.method_of(name) == want


@pytest.mark.parametrize("name", ["StabMap", "stabmap", "StabMap (this run)"])
def test_a_package_method_name_reads_its_own_cell_order(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = pipeline._dataset_labels("D52mini", "cross", name, None)
    assert list(got) == ["cty3", "cty1", "cty2"]


@pytest.mark.parametrize("name", ["Mine", "embedding", "my run"])
def test_any_other_name_reads_the_folder_order(name):
    assert list(pipeline._dataset_labels("D52mini", "cross", name, None)) == \
        ["cty1", "cty2", "cty3"]


@pytest.mark.parametrize("dataset, category, method", [
    ("D28", "diagonal", "uniPort"), ("D45mini", "mosaic", "SMILE"),
    ("D45mini", "mosaic", "MultiVI")])
def test_the_rule_on_diagonal_and_mosaic_folders(dataset, category, method):
    own = list(resolve.labels_for(dataset, category, method))
    for name in (method, method.lower(), f"{method} (this run)"):
        assert list(pipeline._dataset_labels(dataset, category, name, None)) == own
    assert list(pipeline._dataset_labels(dataset, category, "Mine", None)) != own


def test_a_package_method_without_category_is_refused_before_scoring():
    with pytest.raises(ValueError, match=r"^evaluate\(dataset='D52mini', "
                       r"method='StabMap \(this run\)'\) needs category= to read StabMap's "
                       r"cell order\.$"):
        mtb.evaluate(np.zeros((3000, 2)), dataset="D52mini", method="StabMap (this run)")


def test_the_cli_refuses_it_too(tmp_path, capsys):
    emb = tmp_path / "e.npy"
    np.save(emb, np.zeros((3000, 2)))
    rc = cli.main(["evaluate", str(emb), "--dataset", "D52mini", "--method", "StabMap"])
    assert rc != 0
    assert "--method 'StabMap' needs --category to read StabMap's cell order." in \
        capsys.readouterr().err


def _ordered_embedding(dataset, category, method):
    """One-hot cell types in ``method``'s cell order, so its order scores ARI 1."""
    files = resolve.labels_for(dataset, category, method)
    labels = np.concatenate([pd.read_csv(f)["x"].astype(str).to_numpy()
                             for f in files.values()])
    types = sorted(set(labels))
    rng = np.random.default_rng(0)
    return (np.eye(len(types))[[types.index(x) for x in labels]] * 10
            + rng.normal(scale=0.01, size=(len(labels), len(types))))


def test_every_documented_call_form_scores_in_the_right_order():
    emb = _ordered_embedding("D52mini", "cross", "StabMap")
    ari = {name: mtb.evaluate(emb, dataset="D52mini", category="cross", method=name,
                              metrics="ARI", verbose=False)["value"].iloc[0]
           for name in ("StabMap", "stabmap", "StabMap (this run)", "Mine")}
    assert ari["StabMap"] == ari["stabmap"] == ari["StabMap (this run)"] > 0.99
    assert ari["Mine"] < 0.5
    out = mtb.evaluate(emb, dataset="D52mini", category="cross", method="StabMap (this run)",
                       metrics="ARI", verbose=False)
    assert out["method"].tolist() == ["StabMap (this run)"]


# --------------------------------------------------------------- C02 obsm=None
def test_obsm_none_means_x_emb_only():
    import anndata as ad
    a = ad.AnnData(np.zeros((6, 3)))
    a.obsm["protein"] = np.arange(12.0).reshape(6, 2)
    with pytest.raises(ValueError, match=r"^The AnnData has no obsm\['X_emb'\]; its obsm "
                                         r"keys are protein: pass obsm=, or obsm='X' for \.X\.$"):
        eio.as_matrix(a)
    assert eio.as_matrix(a, obsm="protein").shape == (6, 2)
    a.obsm["X_emb"] = np.ones((6, 4))
    assert eio.as_matrix(a).shape == (6, 4)


# --------------------------------------------------------------- C03 one dtype
def test_every_route_reads_the_embedding_at_its_own_dtype(tmp_path):
    import anndata as ad
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 60)).astype(np.float32)
    with h5py.File(tmp_path / "embedding.h5", "w") as f:
        f["data"] = x
    a = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(300)]), obsm={"X_emb": x})
    for route in (tmp_path / "embedding.h5", a, x, pd.DataFrame(x)):
        assert eio.as_matrix(route).dtype == np.float32
    assert escib._build_adata(x, ["a"] * 300, None, [0] * 300).X.dtype == np.float32
    labels = list(rng.choice(["T", "B", "NK"], 300))
    ari = {type(r).__name__: mtb.evaluate(r, labels, metrics=["ARI", "ASW"],
                                          verbose=False)["value"].tolist()
           for r in (str(tmp_path / "embedding.h5"), a, x)}
    assert len({tuple(v) for v in ari.values()}) == 1, ari


def test_an_integer_matrix_is_read_as_float():
    assert eio.as_matrix(np.arange(6).reshape(3, 2)).dtype.kind == "i"   # arrays as given
    assert eio._own_dtype(np.arange(6).reshape(3, 2)).dtype == np.float64
