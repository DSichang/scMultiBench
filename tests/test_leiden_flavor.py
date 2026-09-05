"""The Leiden backend knob: ``leiden_sweep(flavor=)`` / ``config.DEFAULT.leiden_flavor``.

scanpy's igraph backend (``flavor="igraph"``, ``n_iterations=2``,
``directed=False``) is several times faster than leidenalg on the 10-resolution
scIB sweep; ``evaluate`` picks it up through the config field. These tests pin
that both backends produce one clustering per resolution, that the default
reads the config field, and that the sweep notice names the flavor.
"""
import re

import numpy as np
import pytest

pytest.importorskip("scanpy")
pytest.importorskip("scib")

import multibench as mtb
from multibench import config
from multibench.eval import scib as S


def _blobs(n_per=60, k=3, seed=0):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(6 * i, 0.4, size=(n_per, 5)) for i in range(k)]
    labels = np.repeat(np.arange(k), n_per)
    return np.vstack(parts), labels


def _spy_leiden(monkeypatch):
    """Record the kwargs of every ``sc.tl.leiden`` call, then run the real thing."""
    import scanpy as sc
    real = sc.tl.leiden
    seen = []

    def spy(adata, *a, **kw):
        seen.append(dict(kw))
        return real(adata, *a, **kw)

    monkeypatch.setattr(sc.tl, "leiden", spy)
    return seen


@pytest.mark.parametrize("flavor", ["igraph", "leidenalg"])
def test_both_flavors_give_one_clustering_per_resolution(flavor, monkeypatch):
    from scib.metrics.clustering import get_resolutions
    seen = _spy_leiden(monkeypatch)
    emb, _ = _blobs()
    adata, keys = S.leiden_sweep(emb, flavor=flavor)
    assert keys == [f"_mb_res_{r}" for r in get_resolutions(n=10, max=2)]
    for k in keys:
        assert k in adata.obs and adata.obs[k].nunique() >= 1
    assert adata.obs[keys[-1]].nunique() >= 3            # three well-separated blobs
    assert len(seen) == 10 and {kw["flavor"] for kw in seen} == {flavor}
    if flavor == "igraph":
        assert all(kw["n_iterations"] == 2 and kw["directed"] is False for kw in seen)
    else:
        assert all("n_iterations" not in kw for kw in seen)


def test_default_reads_the_config_field(monkeypatch):
    seen = _spy_leiden(monkeypatch)
    emb, _ = _blobs(n_per=30)
    monkeypatch.setattr(config.DEFAULT, "leiden_flavor", "leidenalg", raising=False)
    S.leiden_sweep(emb)
    assert {kw["flavor"] for kw in seen} == {"leidenalg"}
    seen.clear()
    monkeypatch.setattr(config.DEFAULT, "leiden_flavor", "igraph", raising=False)
    S.leiden_sweep(emb)
    assert {kw["flavor"] for kw in seen} == {"igraph"}


def test_shipped_default_is_igraph():
    """``config.DEFAULT.leiden_flavor`` defaults to ``"igraph"``; a Config
    without the field (older checkouts) resolves to the same."""
    assert getattr(config.DEFAULT, "leiden_flavor", "igraph") == "igraph"
    assert S._resolve_flavor(None) == "igraph"


def test_unknown_flavor_raises():
    emb, _ = _blobs(n_per=10)
    with pytest.raises(ValueError) as e:
        S.leiden_sweep(emb, flavor="Igraph")
    assert str(e.value) == "unknown leiden flavor 'Igraph'; valid: igraph|leidenalg"
    with pytest.raises(ValueError, match="unknown leiden flavor 'louvain'"):
        S.compute(emb, np.zeros(len(emb)), None, np.zeros(len(emb)), flavor="louvain")


def test_leiden_sweep_flavor_is_keyword_only():
    import inspect
    p = inspect.signature(S.leiden_sweep).parameters["flavor"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None


@pytest.mark.parametrize("flavor", ["igraph", "leidenalg"])
def test_evaluate_recovers_the_blobs_on_either_flavor(flavor, monkeypatch):
    monkeypatch.setattr(config.DEFAULT, "leiden_flavor", flavor, raising=False)
    seen = _spy_leiden(monkeypatch)
    emb, lab = _blobs()
    df = mtb.evaluate(emb, labels=lab, metrics=["ARI", "NMI", "iF1"])
    assert float(df.loc["ARI", "Value"]) > 0.95
    assert float(df.loc["iF1", "Value"]) > 0.95
    assert {kw["flavor"] for kw in seen} == {flavor}
    assert len(seen) == 10                       # ONE sweep serves ARI/NMI and iF1


def test_sweep_notice_names_the_flavor(monkeypatch, capsys):
    monkeypatch.setattr(S, "_SWEEP_NOTICE_CELLS", 0)
    emb, lab = _blobs(n_per=20)
    for flavor in ("igraph", "leidenalg"):
        monkeypatch.setattr(config.DEFAULT, "leiden_flavor", flavor, raising=False)
        mtb.evaluate(emb, labels=lab, metrics=["ARI"])
        err = capsys.readouterr().err
        assert re.search(rf"Leiden resolution sweep \(10 resolutions, flavor={flavor}\)", err)


def test_leidenalg_path_does_not_nag_about_igraph(recwarn):
    """scanpy warns on every leidenalg call that igraph is faster; with the
    backend an explicit choice that is noise, and only that message is muted."""
    import warnings
    emb, _ = _blobs(n_per=15)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        S.leiden_sweep(emb, flavor="leidenalg")
    assert not [x for x in w if "igraph" in str(x.message)]


def test_isolated_f1_fallback_sweep_uses_the_flavor(monkeypatch):
    """Without precomputed keys ``_isolated_labels_f1`` runs its own sweep, on
    the same backend."""
    import anndata as ad
    import pandas as pd
    seen = _spy_leiden(monkeypatch)
    emb, lab = _blobs(n_per=25)
    adata = ad.AnnData(emb.astype(float))
    adata.obsm["X_emb"] = adata.X
    adata.obs["celltype"] = pd.Categorical(lab.astype(str))
    adata.obs["batch"] = pd.Categorical(["b"] * len(lab))
    v = S._isolated_labels_f1(adata, "celltype", "batch", "X_emb", iso_threshold=2,
                              flavor="leidenalg")
    assert 0.0 <= v <= 1.0
    assert len(seen) == 10 and {kw["flavor"] for kw in seen} == {"leidenalg"}


def test_igraph_falls_back_to_leidenalg_on_old_scanpy(monkeypatch):
    """scanpy < 1.10 has no flavor= (it forwards the keyword to leidenalg and
    fails with TypeError); the resolver downgrades to leidenalg, warning once."""
    monkeypatch.setattr(S, "_igraph_support", False)
    monkeypatch.setattr(S, "_fallback_warned", False)
    with pytest.warns(UserWarning, match=r"cannot run flavor='igraph' \(needs scanpy>=1.10"):
        assert S._resolve_flavor("igraph") == "leidenalg"
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # second call: no repeat
        assert S._resolve_flavor(None) == "leidenalg"
    assert S._resolve_flavor("leidenalg") == "leidenalg"


def test_igraph_probe_reads_scanpy_signature(monkeypatch):
    monkeypatch.setattr(S, "_igraph_support", None)
    assert S.igraph_flavor_available() is True      # this venv: scanpy >= 1.10 + igraph
