"""evaluate() records how its scores were computed; to_long() keeps it.

A CSV of a user's rows must say which Leiden backend ran, whether the
clusters behind ARI/NMI came from the sweep or from the user, and the package
version: those decide whether two rows are comparable (S4-11).
"""
import warnings

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


def test_evaluate_records_a_sweep():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ARI", "ASW"])
    assert wide.attrs == {"leiden_flavor": _flavor(), "clustering": "sweep",
                          "multibench_version": mtb.__version__}


def test_evaluate_records_user_clusters_and_no_sweep():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, clustering=lab, metrics=["ARI", "NMI"])
    assert wide.attrs["clustering"] == "user"
    assert wide.attrs["leiden_flavor"] is None          # nothing needed a sweep


def test_evaluate_without_clustering_metrics_records_none():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ASW"])
    assert wide.attrs == {"leiden_flavor": None, "clustering": None,
                          "multibench_version": mtb.__version__}


def test_evaluate_records_the_configured_backend(monkeypatch):
    pytest.importorskip("leidenalg")
    monkeypatch.setattr(config.DEFAULT, "leiden_flavor", "leidenalg")
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ARI"])
    assert wide.attrs["leiden_flavor"] == "leidenalg"


def test_to_long_writes_scored_with_and_keeps_the_attrs():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ARI", "ASW"])
    long = mtb.to_long(wide, method="Mine", dataset="D11", category="vertical")
    assert long.columns.tolist() == pipeline.LONG_COLUMNS + ["scored_with"]
    assert set(long["scored_with"]) == {f"{_flavor()}/sweep/{mtb.__version__}"}
    assert long.attrs == wide.attrs
    # a clustering-free score fills the parts that did not apply with 'none'
    asw = mtb.to_long(mtb.evaluate(emb, labels=lab, metrics=["ASW"]), method="M")
    assert set(asw["scored_with"]) == {f"none/none/{mtb.__version__}"}


def test_to_long_of_a_frame_without_provenance_keeps_seven_columns(tmp_path):
    w = pd.DataFrame({"Value": [0.5, 0.6]}, index=["ARI", "NMI"])
    assert mtb.to_long(w, method="M").columns.tolist() == pipeline.LONG_COLUMNS
    # a CSV read back carries no attrs either
    emb, lab = _blobs()
    mtb.evaluate(emb, labels=lab, metrics=["ASW"]).to_csv(tmp_path / "w.csv")
    back = pd.read_csv(tmp_path / "w.csv", index_col=0)
    assert mtb.to_long(back, method="M").columns.tolist() == pipeline.LONG_COLUMNS


def _mine():
    emb, lab = _blobs()
    wide = mtb.evaluate(emb, labels=lab, metrics=["ARI", "NMI", "ASW"])
    return mtb.to_long(wide, method="Mine", dataset="D11", category="vertical")


def test_scored_with_column_passes_through_the_consumers(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    mine = _mine()
    stored = mtb.load_results("vertical", dataset="D11", source="rerun")
    both = pd.concat([stored, mine], ignore_index=True)
    assert both["scored_with"].notna().sum() == len(mine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tbl = mtb.plot.build_table(both)
        assert "Mine" in tbl.methods
        mtb.plot.bubble(both)
        mtb.plot.bar(both)
        rec = mtb.recommend("vertical", long_df=both, metrics=["ARI", "NMI", "ASW"])
    assert "Mine" in set(rec["method"])
    path = tmp_path / "all.csv"
    both.to_csv(path, index=False)
    back = mtb.load_results(result_path=path, source="user")
    assert set(back["method"]) == {"Mine"} and len(back) == len(mine)
    # the provenance survives the documented CSV round trip
    assert back.columns.tolist() == pipeline.LONG_COLUMNS + ["scored_with"]
    assert back["scored_with"].tolist() == mine["scored_with"].tolist()
    every = mtb.load_results(result_path=path, source="both")
    assert every["scored_with"].isna().sum() == len(stored)
    # a file without the column still loads with the seven columns
    stored.to_csv(tmp_path / "stored.csv", index=False)
    assert mtb.load_results(result_path=tmp_path / "stored.csv").columns.tolist() == \
        pipeline.LONG_COLUMNS


def test_run_all_scoring_labels_its_internal_sweep_as_sweep():
    """run_all hands the screening sweep's clusters to evaluate(clustering=);
    they are not the user's clusters and must not be recorded as such."""
    rng = np.random.default_rng(1)
    emb, lab = _blobs(n_per=50)
    bat = np.array([f"b{i % 2}" for i in range(len(lab))])
    cands = [(["good.csv"], lab, bat), (["scrambled.csv"], rng.permutation(lab), bat)]
    names, val, _ = W._evaluate_best_order(emb, "vertical", cands)
    assert names == ["good.csv"]
    assert val.attrs["clustering"] == "sweep"
    assert val.attrs["leiden_flavor"] == _flavor()
