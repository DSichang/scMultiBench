"""The Leiden backend knob: ``leiden_sweep(flavor=)`` / ``config.DEFAULT.leiden_flavor``.

scanpy's igraph backend (``flavor="igraph"``, ``n_iterations=2``,
``directed=False``) is several times faster than leidenalg on the 10-resolution
scIB sweep; ``evaluate`` picks it up through the config field. These tests pin
that both backends produce one clustering per resolution, that the default
reads the config field, and that the sweep notice names the flavor.
"""
import re

import numpy as np
import warnings

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



@pytest.fixture(autouse=True)
def _pretend_modern_scanpy(monkeypatch):
    """Each test starts with the one-per-process fallback warning re-armed."""
    monkeypatch.setattr(S, "_fallback_warned", False)


def _eff(flavor):
    """The backend the package will actually run for a request: igraph only
    where this scanpy has it (>= 1.10 + igraph), leidenalg otherwise."""
    return flavor if flavor == "leidenalg" or S.igraph_flavor_available() else "leidenalg"


def _asked(seen):
    """Backends the spied calls asked scanpy for; an absent flavor= means
    leidenalg (the keyword is only sent to a scanpy that understands it)."""
    return {kw.get("flavor", "leidenalg") for kw in seen}

@pytest.mark.parametrize("flavor", ["igraph", "leidenalg"])
def test_evaluate_recovers_the_blobs_on_either_flavor(flavor, monkeypatch):
    monkeypatch.setattr(config.DEFAULT, "leiden_flavor", flavor, raising=False)
    seen = _spy_leiden(monkeypatch)
    emb, lab = _blobs()
    df = mtb.evaluate(emb, labels=lab, metrics=["ARI", "NMI", "iF1"]).set_index("metric")
    assert float(df.loc["ARI", "value"]) > 0.95
    assert float(df.loc["iF1", "value"]) > 0.95
    assert _asked(seen) == {_eff(flavor)}
    steps = len(seen)
    S.leiden_n_clusters(emb, len(set(lab)))
    assert len(seen) == 2 * steps                # ONE search serves ARI, NMI and iF1


def test_sweep_notice_names_the_flavor(monkeypatch, capsys):
    monkeypatch.setattr(S, "_SWEEP_NOTICE_CELLS", 0)
    emb, lab = _blobs(n_per=20)
    for flavor in ("igraph", "leidenalg"):
        monkeypatch.setattr(config.DEFAULT, "leiden_flavor", flavor, raising=False)
        mtb.evaluate(emb, labels=lab, metrics=["ARI"])
        err = capsys.readouterr().err
        assert re.search(rf"cells for ARI with the {_eff(flavor)} Leiden "
                         rf"backend\.", err)


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


def test_igraph_probe_reads_scanpy_version(monkeypatch):
    monkeypatch.setattr(S, "_igraph_support", None)
    from importlib.metadata import version
    from packaging.version import Version
    assert S.igraph_flavor_available() is (Version(version("scanpy")) >= Version("1.10"))


def test_leidenalg_path_passes_no_flavor_keyword(monkeypatch):
    """scanpy < 1.10 has no flavor= parameter; the leidenalg branch must not
    send it (the benchmark host's scanpy 1.9.8 raised TypeError 'flavor')."""
    import scanpy as sc
    calls = []
    monkeypatch.setattr(sc.tl, "leiden", lambda adata, **kw: calls.append(kw))
    monkeypatch.setattr(S, "_flavor_kw", False)            # an old scanpy
    S._leiden(object(), 0.5, "k", "leidenalg")
    assert calls and "flavor" not in calls[0] and calls[0]["resolution"] == 0.5
    calls.clear()
    monkeypatch.setattr(S, "_flavor_kw", True)             # scanpy >= 1.10
    S._leiden(object(), 0.5, "k", "leidenalg")
    assert calls[0]["flavor"] == "leidenalg"
    calls.clear()
    S._leiden(object(), 0.5, "k", "igraph")
    assert calls[0]["flavor"] == "igraph"
