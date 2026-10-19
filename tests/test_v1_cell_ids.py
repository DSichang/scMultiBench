"""v1 NEW-1: evaluate(dataset=, method=) names the folder's cells by the files the
method read, as res.embedding(method) names them, so an embedding in any cell
order scores as in the method's order; ids that are none of the folder's cells
are refused, not matched by position."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, workflow
from multibench.eval import pipeline

ad = pytest.importorskip("anndata")

#: cLISI, GC and ASW_batch read the kNN graph or the distances: the cell order
#: moves them by ties only
CLOSE = ["cLISI", "GC", "ASW_batch"]


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _values(long) -> dict:
    return long.set_index("metric")["value"].to_dict()


def _embedding(ds, category, method):
    """An embedding of ``ds`` in ``method``'s cell order, its cells named as
    ``res.embedding(method)`` names them: one block per cell type, plus noise."""
    root = mtb.config.DEFAULT.data_path
    files = _quiet(pipeline._dataset_labels, ds, category, method, None)
    files = list(files.values()) if isinstance(files, dict) else list(files)
    n = sum(len(pd.read_csv(f)) for f in files)
    obs = _quiet(workflow._embedding_obs, {"method": method, "modalities": None}, ds,
                 category, root, n)
    codes = pd.factorize(obs["label"])[0]
    rng = np.random.default_rng(0)
    x = np.eye(codes.max() + 1)[codes] * 4 + rng.normal(0, 0.5, (n, codes.max() + 1))
    return ad.AnnData(obs=pd.DataFrame(index=obs.index), obsm={"X_emb": x.astype("float32")})


CASES = [("D11", "vertical", "Matilda", ["cLISI", "ARI"]),
         ("D28", "diagonal", "scJoint", CLOSE),
         ("D45", "mosaic", "Cobolt", CLOSE),
         ("D52", "cross", "StabMap", CLOSE)]


@pytest.mark.parametrize("ds,category,method,metrics", CASES)
def test_a_shuffled_embedding_scores_as_in_order(ds, category, method, metrics):
    e = _embedding(ds, category, method)
    kw = dict(dataset=ds, category=category, method=method, metrics=metrics, verbose=False)
    want = _values(_quiet(mtb.evaluate, e, **kw))
    perm = np.random.default_rng(1).permutation(e.n_obs)
    got = _values(_quiet(mtb.evaluate, e[perm].copy(), **kw))
    for m in metrics:
        # evaluate.md: Leiden-based scores move by up to about 0.05 with cell order
        tol = 0.05 if m == "ARI" else 0.005
        assert got[m] == pytest.approx(want[m], abs=tol), (m, got, want)
    if ds == "D28":
        # scJoint read atac_gas.h5: its cells repeat the RNA barcodes, so each is
        # named by its label file
        assert e.obs_names[0].startswith("rna_cty:")


def test_the_stored_d28_scjoint_embedding_scores_shuffled_as_in_order():
    """The verification's case: cLISI 0.82 / GC 0.18 shuffled against 0.99 / 0.87."""
    stored = Path(mtb.config.DEFAULT.data_path) / "outputs" / "D28"
    repo = Path(__file__).resolve().parent.parent / "data" / "outputs" / "D28"
    folder = stored if (stored / "batch_result.json").is_file() else repo
    if not (folder / "batch_result.json").is_file():
        pytest.skip("no stored D28 outputs")
    e = _quiet(mtb.load_batch, folder).embedding("scJoint")
    kw = dict(dataset="D28", category="diagonal", method="scJoint",
              metrics=["cLISI", "GC"], verbose=False)
    want = _values(_quiet(mtb.evaluate, e, **kw))
    perm = np.random.default_rng(0).permutation(e.n_obs)
    got = _values(_quiet(mtb.evaluate, e[perm].copy(), **kw))
    assert got == pytest.approx(want, abs=0.005)


def test_your_own_diagonal_embedding_with_the_atac_cells_first():
    """No method: the folder's default files name the cells (atac_peak.h5 on
    D28); the ATAC cells may come first."""
    files = _quiet(pipeline._dataset_labels, "D28", "diagonal", None, None)
    files = list(files.values()) if isinstance(files, dict) else list(files)
    assert [Path(f).name for f in files] == ["rna_cty.csv", "atac_cty.csv"]
    keys = pipeline._file_barcodes([Path(f) for f in files])
    labels = np.concatenate([pd.read_csv(f).iloc[:, 0].to_numpy() for f in files])
    n_rna = len(pd.read_csv(files[0]))
    codes = pd.factorize(labels)[0]
    x = np.eye(codes.max() + 1)[codes] * 4 + \
        np.random.default_rng(0).normal(0, 0.5, (len(codes), codes.max() + 1))
    e = ad.AnnData(obs=pd.DataFrame(index=keys.to_numpy()), obsm={"X_emb": x.astype("float32")})
    kw = dict(dataset="D28", category="diagonal", metrics=CLOSE, verbose=False)
    want = _values(_quiet(mtb.evaluate, e, **kw))
    atac_first = np.r_[np.arange(n_rna, e.n_obs), np.arange(n_rna)]
    got = _values(_quiet(mtb.evaluate, e[atac_first].copy(), **kw))
    assert got == pytest.approx(want, abs=0.005)
    # your own method's name reads the same files
    own = _values(_quiet(mtb.evaluate, e[atac_first].copy(), method="mine", **kw))
    assert own == pytest.approx(want, abs=0.005)


def test_cell_ids_that_are_none_of_the_folders_are_refused():
    """scJoint's names (atac_gas.h5) are none of the folder's default cells: an
    embedding scored without method= is refused, not matched by position."""
    e = _embedding("D28", "diagonal", "scJoint")
    keys = pipeline._file_barcodes(
        [Path(f) for f in _quiet(pipeline._dataset_labels, "D28", "diagonal", None,
                                 None).values()])
    with pytest.raises(ValueError) as err:
        _quiet(mtb.evaluate, e, dataset="D28", category="diagonal", metrics=["cLISI"],
               verbose=False)
    assert str(err.value) == (
        f"None of the embedding's cell ids is a cell of rna_cty.csv and atac_cty.csv: "
        f"the embedding names a cell {e.obs_names[0]!r}, the dataset folder "
        f"{keys.iloc[0]!r}. Name the embedding's cells as the folder does.")


def test_the_command_line_scores_a_shuffled_embedding_as_in_order(tmp_path):
    e = _embedding("D28", "diagonal", "scJoint")
    perm = np.random.default_rng(2).permutation(e.n_obs)
    e.write_h5ad(tmp_path / "in_order.h5ad")
    e[perm].copy().write_h5ad(tmp_path / "shuffled.h5ad")
    got = {}
    for name in ("in_order", "shuffled"):
        rc = _quiet(cli.main, ["evaluate", str(tmp_path / f"{name}.h5ad"), "--dataset", "D28",
                               "--category", "diagonal", "--method", "scJoint", "--metrics",
                               "cLISI,GC", "--out", str(tmp_path / f"{name}.csv")])
        assert rc == 0
        got[name] = _values(pd.read_csv(tmp_path / f"{name}.csv"))
    assert got["shuffled"] == pytest.approx(got["in_order"], abs=0.005)
