"""evaluate() records how its scores were computed; to_long() keeps it.

A CSV of a user's rows must say which Leiden backend ran, whether the
clusters behind ARI/NMI came from the sweep or from the user, and the package
version: those decide whether two rows are comparable (S4-11).
"""
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.eval import pipeline

pytest.importorskip("scib")


def _blobs(n_per=40, n_labels=3, dims=5, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 8, size=(n_labels, dims))
    emb = np.vstack([centres[i] + rng.normal(0, 1, size=(n_per, dims))
                     for i in range(n_labels)])
    lab = np.array([f"t{i}" for i in range(n_labels) for _ in range(n_per)])
    return emb, lab


def _flavor():
    from multibench.eval.scib import _resolve_flavor
    return _resolve_flavor(None)


def test_evaluate_records_a_search():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ARI", "ASW"])
    assert wide.attrs == {"leiden_flavor": _flavor(), "clustering": "search",
                          "multibench_version": mtb.__version__,
                          "scib_version": version("scib")}


def test_evaluate_without_clustering_metrics_records_none():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["cLISI"])
    assert wide.attrs == {"leiden_flavor": None, "clustering": None,
                          "multibench_version": mtb.__version__,
                          "scib_version": version("scib")}


def test_evaluate_records_the_configured_backend(monkeypatch):
    pytest.importorskip("leidenalg")
    monkeypatch.setattr(config.DEFAULT, "leiden_flavor", "leidenalg")
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ARI"])
    assert wide.attrs["leiden_flavor"] == "leidenalg"


def test_run_all_scoring_labels_its_internal_search_as_search():
    """run hands the screening search's clusters to evaluate(clustering=);
    they are not the user's clusters and must not be recorded as such."""
    rng = np.random.default_rng(1)
    emb, lab = _blobs(n_per=50)
    bat = np.array([f"b{i % 2}" for i in range(len(lab))])
    cands = [(["good.csv"], lab, bat), (["scrambled.csv"], rng.permutation(lab), bat)]
    names, val, _ = W._evaluate_best_order(emb, "vertical", cands)
    assert names == ["good.csv"]
    assert val.attrs["clustering"] == "search"
    assert val.attrs["leiden_flavor"] == _flavor()
