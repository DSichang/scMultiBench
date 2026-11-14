"""Every score starts from scanpy's neighbour graph, and two releases broke it
on 8,192 cells or more, the size from which scanpy hands the search to
pynndescent and umap-learn.

umap-learn 0.6.0 (2026-11-10), with scanpy 1.12.5: numba refuses the
non-contiguous ``knn_dists``. The package leaves that one release out of its
dependencies, and a run that meets the failure all the same says what to
install.

pynndescent 0.7.0 (2026-10-27) returns sparse arrays, which have no ``getnnz``:
scanpy 1.10, 1.11 and 1.12.0 cannot read them. Which scanpy pip installs does
not follow from the Python version alone: on Python 3.13 with pandas held at
2.2.3 (Colab, through the tutorials' install cell) it is scanpy 1.12.0. So
the package keeps pynndescent below 0.7 on every Python, and a run that meets
the failure all the same says what to install.
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


def test_every_python_keeps_a_pynndescent_that_every_scanpy_reads():
    pynn = _declared()["pynndescent"]
    # no marker: scanpy 1.12.0, which Python 3.13 gets beside an older pandas,
    # reads pynndescent 0.7 no better than the scanpy of Python 3.9
    assert pynn.marker is None
    assert not pynn.specifier.contains("0.7.0") and pynn.specifier.contains("0.6.0")


#: scanpy's failure on what pynndescent 0.7 returns (Python 3.10 and later)
GETNNZ = "'csr_array' object has no attribute 'getnnz'"


def test_the_pynndescent_failure_says_which_release_and_what_to_install(monkeypatch):
    import scanpy as sc

    def refuse(adata, **kwargs):
        raise AttributeError(GETNNZ)

    monkeypatch.setattr(sc.pp, "neighbors", refuse)
    with pytest.raises(RuntimeError) as e:
        escib._neighbors(_adata(), "X")
    assert str(e.value) == (
        f'scanpy {version("scanpy")} cannot read the neighbours pynndescent '
        f'{version("pynndescent")} found for these 30 cells. Run pip install '
        f'"pynndescent<0.7", then score again.')
    assert isinstance(e.value.__cause__, AttributeError)


def test_another_attribute_error_of_the_neighbour_graph_is_not_reworded(monkeypatch):
    import scanpy as sc

    def refuse(adata, **kwargs):
        raise AttributeError("'NoneType' object has no attribute 'shape'")

    monkeypatch.setattr(sc.pp, "neighbors", refuse)
    with pytest.raises(AttributeError, match="has no attribute 'shape'"):
        escib._neighbors(_adata(), "X")


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
