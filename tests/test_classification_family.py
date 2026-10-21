"""The classification family (OCA, ACA, Spec, Sens, F1) in evaluate, to_long,
run / rescore, the plots and the command line.

``evaluate`` reports the ``mean`` row of
``classification_metrics(classify(...))``; by default only for two or more
batches outside ``vertical``, as the benchmark scores it.
"""
import importlib
import warnings
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
import multibench.eval.classify  # noqa: F401  (mtb.eval.classify)
from multibench import workflow as W
from multibench._data import catalog
B = importlib.import_module("multibench.plot.bubble")   # the module

CODES = ["OCA", "ACA", "Spec", "Sens", "F1"]
SCIB = ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI", "ASW_batch", "ARI_batch", "NMI_batch",
        "GC", "iLISI"]
CENTRES = {"A": [40, 0, 0, 0], "B": [0, 40, 0, 0], "C": [0, 0, 40, 0]}


def _block(types, n, seed):
    rng = np.random.default_rng(seed)
    labels = [types[i % len(types)] for i in range(n)]
    return np.array([CENTRES[t] for t in labels], float) + rng.normal(0, 0.5, (n, 4)), labels


def _write(folder, blocks):
    """``{stem: (types, n)}`` -> label files in ``folder``, the embedding in
    that block order, and the label paths."""
    folder.mkdir(parents=True, exist_ok=True)
    embs, paths = [], []
    for i, (stem, (types, n)) in enumerate(blocks.items()):
        e, lab = _block(types, n, seed=i)
        embs.append(e)
        paths.append(folder / f"{stem}.csv")
        pd.DataFrame({"x": lab}).to_csv(paths[-1], index=False)
    return np.vstack(embs), paths


@pytest.fixture
def three(tmp_path):
    """Three batches of 30 cells, types A, B and C in each."""
    return _write(tmp_path / "MB", {"cty1": ("ABC", 30), "cty2": ("ABC", 30),
                                    "cty3": ("ABC", 30)})


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _caught(fn, *a, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [str(x.message) for x in w]


def _reference(emb, paths):
    return _quiet(mtb.eval.classify.classification_metrics, _quiet(mtb.eval.classify.classify, emb, labels=paths)).loc["mean"]


def _values(df):
    """evaluate's long table as a Series of values indexed by metric."""
    return df.set_index("metric")["value"]


def test_metrics_selects_the_family():
    sel = catalog.metric_selection("classification")
    assert (sel.family, sel.codes, sel.explicit) == ("classification", CODES, False)
    assert catalog.metric_selection(["oca", "F1"]).family == "classification"
    assert catalog.metric_selection(["ARI", "OCA"]).family == "all"
    assert catalog.metric_selection(["ARI", "NMI"]).family == "clustering"
    with pytest.raises(ValueError) as e:
        catalog.metric_selection("classifcation")
    assert str(e.value) == (
        "Unknown metrics= token classifcation. Pass 'all', 'clustering', 'batch' or "
        "'classification', or metric codes such as metrics=['ARI', 'NMI'].")


# ---------------------------------------------------------------- evaluate
def test_explicit_classification_is_the_mean_row_of_classify(three):
    emb, paths = three
    df = mtb.evaluate(emb, labels=paths, metrics="classification")
    assert df["metric"].tolist() == CODES
    ref = _reference(emb, paths)
    assert all(_values(df)[c] == ref[c] for c in CODES)
    assert df.attrs["leiden_flavor"] is None and df.attrs["clustering"] is None
    sub = mtb.evaluate(emb, labels=paths, metrics=["f1", "OCA"])
    assert sub["metric"].tolist() == ["OCA", "F1"]
    assert _values(sub)["F1"] == ref["F1"]


@pytest.mark.parametrize("category", [None, "mosaic"])
def test_default_adds_the_family_for_several_batches(three, category):
    emb, paths = three
    df, said = _caught(mtb.evaluate, emb, labels=paths, category=category, verbose=False)
    assert df["metric"].tolist() == SCIB + CODES
    ref = _reference(emb, paths)
    assert all(_values(df)[c] == ref[c] for c in CODES)
    assert not [m for m in said if "OCA" in m]


def test_default_leaves_it_out_for_vertical_and_one_batch(three):
    emb, paths = three
    df, said = _caught(mtb.evaluate, emb, labels=paths, category="vertical",
                       verbose=False)
    assert df["metric"].tolist() == SCIB
    one, said1 = _caught(mtb.evaluate, emb[:30], labels=paths[0], verbose=False)
    assert one["metric"].tolist() == SCIB[:6]
    assert not [m for m in said + said1 if "OCA" in m]


def test_explicit_without_two_batches_is_an_error(three):
    emb, paths = three
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb[:30], labels=paths[0], metrics="classification")
    assert str(e.value) == (
        "metrics='classification' needs two or more batches. Pass batch=<vector>, or "
        "labels as a list of two or more files.")
    with pytest.raises(ValueError) as e:
        mtb.evaluate(emb[:30], labels=paths[0], batch=["d1"] * 30, metrics=["OCA"])
    assert str(e.value) == (
        "metrics=['OCA'] needs two or more batches; every cell is in one batch. Pass "
        "batch=<vector>, or labels as a list of two or more files.")


def test_all_with_one_batch_is_refused(three):
    # c1 (C11): the batch metrics on one batch are refused, as classification is
    emb, paths = three
    with pytest.raises(ValueError, match=r"^metrics='all' needs two or more batches; every "
                                         r"cell is in one batch\."):
        mtb.evaluate(emb[:30], labels=paths[0], batch=["d1"] * 30, metrics="all",
                     verbose=False)


def test_a_graph_output(three, tmp_path):
    emb, paths = three
    graph = tmp_path / "knn_indices.h5"
    with h5py.File(graph, "w") as f:
        f["data"] = np.tile(np.arange(20), (90, 1)).T
    with pytest.raises(ValueError) as e:
        mtb.evaluate(graph, labels=paths, metrics="classification")
    assert str(e.value) == ("scMoMaT writes a graph, not an embedding, so it has no "
                            "classification scores. Pass metrics='clustering'.")
    df, said = _caught(mtb.evaluate, graph, labels=paths, metrics="all", verbose=False)
    assert _values(df)[CODES].isna().all()
    assert [m for m in said if "OCA" in m] == [
        "scMoMaT writes a graph, not an embedding, so OCA, ACA, Spec, Sens and F1 are "
        "NaN. Pass metrics='clustering' to leave them out."]
    # cells x cells, by default
    df, said = _caught(mtb.evaluate, np.eye(90), labels=paths, verbose=False)
    assert _values(df)[CODES].isna().all()
    assert [m for m in said if "OCA" in m] == [
        "output is a graph, not an embedding, so OCA, ACA, Spec, Sens and F1 are NaN. "
        "Pass metrics='clustering' to leave them out."]


def _knn(n=90, k=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n)[:k] for _ in range(n)])


def test_every_form_of_a_graph_output_is_treated_alike(three, tmp_path):
    """The kNN index matrix mtb.run returns for scMoMaT, the same matrix in
    its file and scMoMaT's UMAP file: NaN and one warning by default, an
    error when classification is asked for."""
    emb, paths = three
    umap = tmp_path / "umap_embedding.h5"
    with h5py.File(umap, "w") as f:
        f["data"] = emb[:, :2].T
    knn = _knn()
    for output, who in ((knn, "output is a graph"), (knn + 1.0, "output is a graph"),
                        (umap, "scMoMaT writes a graph")):
        df, said = _caught(mtb.evaluate, output, labels=paths, verbose=False)
        assert _values(df)[CODES].isna().all()
        assert [m for m in said if "OCA" in m] == [
            f"{who}, not an embedding, so OCA, ACA, Spec, Sens and F1 are NaN. "
            f"Pass metrics='clustering' to leave them out."]
        with pytest.raises(ValueError) as e:
            mtb.evaluate(output, labels=paths, metrics=["OCA", "ARI"])
        assert str(e.value) == (f"{who}, not an embedding, so it has no classification "
                                f"scores. Pass metrics='clustering'.")
    # the scIB rows of the array are what 0.3.7 gave
    scib = _quiet(mtb.evaluate, knn, labels=paths, metrics="clustering", verbose=False)
    assert scib["metric"].tolist() == SCIB[:6]


def test_an_embedding_is_never_taken_for_a_graph(three):
    emb, paths = three
    df, said = _caught(mtb.evaluate, emb, labels=paths, verbose=False)
    assert not _values(df)[CODES].isna().any()
    assert not [m for m in said if "graph" in m]


def test_spec_nan_for_a_query_of_one_cell_type_is_said(tmp_path):
    # with cty1 as the reference only A is shared; with cty2, only A too
    emb, paths = _write(tmp_path, {"cty1": ("AB", 40), "cty2": ("AC", 40)})
    df, said = _caught(mtb.evaluate, emb, labels=paths, verbose=False)
    assert np.isnan(_values(df)["Spec"]) and _values(df)["OCA"] == 1.0
    assert [m for m in said if "Spec" in m] == [
        "Spec is NaN: when batch cty1 or cty2 is the reference, the cells it predicts "
        "are all of one cell type. Compare methods on the other metrics."]
    _, said = _caught(mtb.evaluate, emb, labels=paths, metrics=["OCA"], verbose=False)
    assert not [m for m in said if "Spec" in m]


def test_no_batch_to_learn_from(tmp_path):
    emb, paths = _write(tmp_path, {"cty1": ("A", 20), "cty2": ("B", 20)})
    with pytest.raises(ValueError) as e:
        _quiet(mtb.evaluate, emb, labels=paths, metrics="classification")
    assert str(e.value) == (
        "metrics='classification' found no batch to learn from: each needs two cell "
        "types that another batch has. Check labels and batch.")
    df, said = _caught(mtb.evaluate, emb, labels=paths, metrics="all", verbose=False)
    assert _values(df)[CODES].isna().all()
    # one warning, not classify's one per skipped batch
    assert [m for m in said if "OCA" in m or "skipped" in m] == [
        "OCA, ACA, Spec, Sens and F1 are NaN: no batch has two cell types that another "
        "batch has. Check labels and batch."]


def test_batch_with_classification_only_is_used_not_warned(three):
    emb, paths = three
    batch = ["b1"] * 30 + ["b2"] * 30 + ["b3"] * 30
    labels = sum((pd.read_csv(p)["x"].tolist() for p in paths), [])
    df, said = _caught(mtb.evaluate, emb, labels=labels, batch=batch,
                       metrics="classification")
    assert df["metric"].tolist() == CODES
    assert not [m for m in said if "changes nothing" in m]


# ---------------------------------------------------------------- run / rescore
def _record(tmp_path, emb, method="StabMap"):
    out = tmp_path / "out" / f"{method}_MB"
    out.mkdir(parents=True)
    with h5py.File(out / "embedding.h5", "w") as f:
        f.create_dataset("data", data=emb.T)
    return {"method": method, "category": "cross", "dataset": "MB", "status": "CHAIN_OK",
            "modalities": ["rna1", "rna2", "rna3", "adt1", "adt2", "adt3"],
            "output_kind": "embedding", "out_dir": str(out), "data_path": str(tmp_path)}


def test_score_record_carries_the_codes(three, tmp_path):
    emb, paths = three
    rec = _record(tmp_path, emb)
    embedding = SimpleNamespace(output=SimpleNamespace(kind="embedding"))
    _quiet(W._score_record, rec, emb, "MB", "cross", tmp_path, embedding)
    assert rec["status"] == "scored"
    assert list(rec["metrics"])[-5:] == CODES
    res = W.RunResult([rec], "MB", "cross")
    assert list(res.summary.columns[res.summary.columns.isin(CODES)]) == CODES
    assert set(CODES) <= set(res.long["metric"])
    # the UMAP of a graph method has no classification scores
    graph = SimpleNamespace(output=SimpleNamespace(kind="graph"))
    g = dict(rec)
    _quiet(W._score_record, g, emb, "MB", "cross", tmp_path, graph)
    assert g["status"] == "scored" and not set(CODES) & set(g["metrics"])
    with pytest.raises(ValueError) as e:
        W._score_record(dict(rec), emb, "MB", "cross", tmp_path, graph,
                        metrics="classification")
    assert str(e.value) == ("StabMap writes a graph, not an embedding, so it has no "
                            "classification scores. Pass metrics='clustering'.")
    # classification and another code: the graph row is scored on the other
    # code, and nothing is said of a batch the caller never passed
    h = dict(rec)
    _, said = _caught(W._score_record, h, emb, "MB", "cross", tmp_path, graph,
                      metrics=["OCA", "ARI"])
    assert list(h["metrics"]) == ["ARI"]
    assert not [m for m in said if "batch" in m]


def test_rescore_carries_the_codes(three, tmp_path):
    emb, paths = three
    res = W.RunResult([_record(tmp_path, emb)], "MB", "cross")
    new = _quiet(res.rescore, metrics="classification")
    assert new.records[0]["status"] == "scored", new.records[0].get("error")
    assert list(new.records[0]["metrics"]) == CODES
    ref = _reference(emb, paths)
    assert new.records[0]["metrics"]["OCA"] == round(float(ref["OCA"]), 4)
    assert new.long["metric"].tolist() == CODES


def test_recommend_without_classification_scores():
    with pytest.raises(ValueError) as e:
        _quiet(mtb.recommend, "cross", metrics="classification")
    assert str(e.value) == ("The stored results have no classification scores. Score "
                            "your runs with mtb.evaluate and pass them as long_df=.")


def test_stored_tables_have_no_classification():
    df, said = _caught(mtb.load_results, "vertical", dataset="D11",
                       metrics="classification")
    # one line says so since 0.4, as for metrics="batch" on a single-batch dataset
    assert df.empty and said == [
        "The stored results of vertical, D11 have no classification scores, so no rows "
        "are left. Leave out metrics= to get the scores they have (ARI, ASW, NMI, cLISI, "
        "iASW, iF1)."]


