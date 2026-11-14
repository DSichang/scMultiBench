"""scanpy 1.13.0 (2026-11-13) names the representation ``sc.pp.neighbors`` works
on ``use`` and deprecates ``use_rep``: with it every score printed a
``FutureWarning`` that pointed at ``multibench/eval/scib.py``, and the release
that drops ``use_rep`` would stop the scoring. ``_neighbors`` hands the
representation over under the name the installed scanpy has.
"""
import warnings

import anndata as ad
import numpy as np
import pytest

from multibench.eval import scib as escib

#: 40 cells: ``X`` and the embedding are unrelated, so the graph tells which was read
N = 40


def _adata():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(rng.normal(size=(N, 5)).astype("float32"))
    adata.obsm["X_emb"] = rng.normal(size=(N, 3)).astype("float32")
    return adata


def _nearest(points, k=14):
    """The ``k`` smallest distances from the first row of ``points`` to another row."""
    return np.sort(np.linalg.norm(points[1:] - points[0], axis=1))[:k]


@pytest.mark.parametrize("rep", ["X", "X_emb"])
def test_the_installed_scanpy_builds_the_graph_of_the_named_representation(rep):
    adata = _adata()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        escib._neighbors(adata, rep)
    # scanpy's default of 15 neighbours counts the cell itself: 14 others
    points = adata.X if rep == "X" else adata.obsm[rep]
    np.testing.assert_allclose(np.sort(adata.obsp["distances"][[0]].data), _nearest(points),
                               rtol=1e-5)
    # a deprecation scanpy raises for an argument of ours points at our file
    ours = [str(w.message) for w in seen if w.filename == escib.__file__]
    assert ours == []


class _Recorded(dict):
    """What a stand-in for ``sc.pp.neighbors`` was called with."""


def test_a_scanpy_that_has_use_is_handed_the_entry_itself(monkeypatch):
    acc = pytest.importorskip("anndata.acc")
    import scanpy as sc
    seen = _Recorded()

    def neighbors(adata, n_neighbors=15, *, use=None, use_rep=None):
        seen.update(use=use, use_rep=use_rep)

    monkeypatch.setattr(sc.pp, "neighbors", neighbors)
    escib._neighbors(_adata(), "X_emb")
    # the accessor: scanpy's text form "obsm.X_emb" splits a name at its periods
    assert seen == {"use": acc.A.obsm["X_emb"], "use_rep": None}
    escib._neighbors(_adata(), "X")
    assert seen == {"use": acc.A.X, "use_rep": None}


def test_a_scanpy_that_only_has_use_rep_is_handed_the_name(monkeypatch):
    import scanpy as sc
    seen = _Recorded()

    def neighbors(adata, n_neighbors=15, *, use_rep=None):
        seen.update(use_rep=use_rep)

    monkeypatch.setattr(sc.pp, "neighbors", neighbors)
    escib._neighbors(_adata(), "X_emb")
    assert seen == {"use_rep": "X_emb"}
