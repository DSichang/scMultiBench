"""umap-learn 0.6.0 (2026-11-10) cannot build scanpy's neighbour graph on 8,192
cells or more with scanpy 1.12.5: numba refuses the non-contiguous ``knn_dists``
scanpy hands over above that size. Every score starts from that graph.

The package leaves that one release out of its dependencies, and a run that
meets the failure all the same says what to install.
"""
import re
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from packaging.requirements import Requirement

from multibench.eval import scib as escib

ROOT = Path(__file__).resolve().parents[1]

#: numba's words when umap-learn 0.6.0 is handed scanpy's distances
KNN_DISTS = ("Failed in nopython mode pipeline (step: nopython frontend)\n"
             "No conversion from array(float32, 2d, A) to array(float32, 2d, C) "
             "for 'knn_dists', defined at None")


def _declared():
    """``{name: Requirement}`` of the ``dependencies`` line of pyproject.toml."""
    line = re.search(r"^dependencies\s*=\s*\[(.*)\]\s*$",
                     (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    reqs = [Requirement(r) for r in re.findall(r'"([^"]+)"', line)]
    return {r.name: r for r in reqs}


def test_the_dependencies_leave_out_the_one_broken_umap_release():
    umap = _declared()["umap-learn"]
    assert not umap.specifier.contains("0.6.0")
    # no bound on the releases around it: the one before, and the fix when it comes
    assert umap.specifier.contains("0.5.12") and umap.specifier.contains("0.6.1")


def _adata(n=30):
    return ad.AnnData(np.random.default_rng(0).normal(size=(n, 5)))


def test_the_umap_failure_says_which_release_and_what_to_install(monkeypatch):
    import scanpy as sc
    from numba.core.errors import TypingError

    def refuse(adata, **kwargs):
        raise TypingError(KNN_DISTS)

    monkeypatch.setattr(sc.pp, "neighbors", refuse)
    with pytest.raises(RuntimeError) as e:
        escib._neighbors(_adata(), "X")
    # the release named is the one installed: the failure is known by numba's words
    here = version("umap-learn")
    assert str(e.value) == (f'umap-learn {here} cannot build the neighbour graph of these 30 '
                            f'cells. Run pip install "umap-learn!={here}", then score again.')
    assert isinstance(e.value.__cause__, TypingError)


def test_another_failure_of_the_neighbour_graph_is_not_reworded(monkeypatch):
    import scanpy as sc

    def refuse(adata, **kwargs):
        raise ValueError("n_neighbors must be at least 2")

    monkeypatch.setattr(sc.pp, "neighbors", refuse)
    with pytest.raises(ValueError, match="^n_neighbors must be at least 2$"):
        escib._neighbors(_adata(), "X")
