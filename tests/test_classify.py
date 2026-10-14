"""mtb.classify: the predicted cell type of each query cell, one batch as
the reference (the benchmark's MLP classification step)."""
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb

COLUMNS = ["reference", "batch", "cell", "label", "predicted"]
#: far apart, so 10 epochs on a few cells already tell the types apart
CENTRES = {"A": [40, 0, 0, 0], "B": [0, 40, 0, 0], "C": [0, 0, 40, 0]}


def _batch(types, n, seed):
    rng = np.random.default_rng(seed)
    labels = [types[i % len(types)] for i in range(n)]
    emb = np.array([CENTRES[t] for t in labels], float) + rng.normal(0, 0.5, (n, 4))
    return emb, labels


def _write(tmp_path, blocks):
    """``{stem: (types, n)}`` -> label files cty<i>.csv, the embedding in the
    given block order and the labels in that order."""
    embs, labels, paths = [], [], {}
    for i, (stem, (types, n)) in enumerate(blocks.items()):
        e, lab = _batch(types, n, seed=i)
        embs.append(e)
        labels += lab
        paths[stem] = tmp_path / f"{stem}.csv"
        pd.DataFrame({"x": lab}).to_csv(paths[stem], index=False)
    return np.vstack(embs), labels, paths


@pytest.fixture
def three(tmp_path):
    """Three batches of 30 cells; cty1 has no cell of type C."""
    return _write(tmp_path, {"cty1": ("AB", 30), "cty2": ("ABC", 30), "cty3": ("ABC", 30)})


def test_return_frame(three):
    emb, labels, paths = three
    out = mtb.classify(emb, labels=list(paths.values()))
    assert list(out.columns) == COLUMNS
    assert isinstance(out.index, pd.RangeIndex)
    first = out[out["reference"] == "cty1"]
    # the query of cty1 is cty2 then cty3, in the rows of the output
    assert first["batch"].tolist() == ["cty2"] * 20 + ["cty3"] * 20
    assert first["cell"].tolist() == [i for i in range(30, 90) if labels[i] != "C"]
    assert first["label"].tolist() == [labels[i] for i in first["cell"]]
    assert (out["label"] == out["predicted"]).all()        # well separated types


def test_every_batch_is_the_reference_in_turn(three):
    emb, labels, paths = three
    out = mtb.classify(emb, labels=list(paths.values()))
    assert out["reference"].unique().tolist() == ["cty1", "cty2", "cty3"]
    assert out.groupby("reference").size().to_dict() == {"cty1": 40, "cty2": 60, "cty3": 60}
    assert not (out["reference"] == out["batch"]).any()


def test_one_reference(three):
    emb, labels, paths = three
    out = mtb.classify(emb, labels=list(paths.values()), reference="cty2")
    assert set(out["reference"]) == {"cty2"} and len(out) == 60
    assert out["batch"].tolist() == ["cty1"] * 30 + ["cty3"] * 30


def test_query_cells_of_a_type_not_in_the_reference_are_left_out(three):
    emb, labels, paths = three
    out = mtb.classify(emb, labels=list(paths.values()), reference="cty1")
    assert "C" not in set(out["label"]) and "C" not in set(out["predicted"])
    assert len(out) == 40


def test_cells_without_a_label_are_left_out(three, tmp_path):
    emb, labels, paths = three
    d = pd.read_csv(paths["cty2"])
    d.loc[[0, 1], "x"] = np.nan
    d.to_csv(paths["cty2"], index=False)
    with pytest.warns(UserWarning) as rec:
        out = mtb.classify(emb, labels=list(paths.values()))
    assert [str(w.message) for w in rec] == [
        "2 of the 90 cells have no label; they are left out."]
    assert not out["cell"].isin([30, 31]).any()
    assert out.groupby("reference").size().to_dict() == {"cty1": 38, "cty2": 60, "cty3": 58}


def test_labels_and_batch_as_obs_columns(three):
    emb, labels, paths = three
    a = ad.AnnData(np.zeros((90, 2)), obsm={"X_emb": emb})
    a.obs_names = [f"c{i}" for i in range(90)]
    a.obs["celltype"] = labels
    a.obs["donor"] = ["d1"] * 30 + ["d2"] * 30 + ["d3"] * 30
    out = mtb.classify(a, labels="celltype", batch="donor", reference="d3")
    assert out["cell"].tolist() == [f"c{i}" for i in range(60)]
    assert out["label"].tolist() == labels[:60]
    assert set(out["batch"]) == {"d1", "d2"}


def test_one_batch_is_an_error(three):
    emb, labels, paths = three
    for labels in ({"cty1": paths["cty1"]}, [paths["cty1"]]):
        with pytest.raises(ValueError) as e:
            mtb.classify(emb[:30], labels=labels)
        assert str(e.value) == ("classify needs at least two batches; every cell is in "
                                "batch cty1. Pass batch= with two or more batches.")


def test_no_batch_is_an_error(three):
    emb, labels, paths = three
    with pytest.raises(ValueError) as e:
        mtb.classify(emb, labels=labels)
    with pytest.raises(ValueError) as e2:
        mtb.classify(emb[:30], labels=paths["cty1"])
    assert str(e.value) == str(e2.value) == (
        "classify needs the batch of each cell. Pass batch=, or "
        "labels=mtb.labels_for(dataset, category, method).")


def test_a_graph_or_label_output_is_an_error(three, tmp_path):
    emb, labels, paths = three
    graph = tmp_path / "knn_indices.h5"
    with h5py.File(graph, "w") as f:
        f["data"] = np.tile(np.arange(20), (90, 1))
    with pytest.raises(ValueError) as e:
        mtb.classify(graph, labels=list(paths.values()))
    assert str(e.value) == ("classify needs an embedding; knn_indices.h5 is not one. "
                            "Pass the method's embedding file.")
    for output in (labels, np.eye(90)):           # predicted labels; cells x cells
        with pytest.raises(ValueError) as e:
            mtb.classify(output, labels=list(paths.values()))
        assert str(e.value) == ("classify needs an embedding; output is not one. "
                                "Pass the method's embedding file.")


def test_unknown_reference_is_an_error(three):
    emb, labels, paths = three
    with pytest.raises(ValueError) as e:
        mtb.classify(emb, labels=list(paths.values()), reference="cty4")
    assert str(e.value) == "reference='cty4' is not a batch. Pass one of cty1, cty2 or cty3."


def test_a_reference_with_one_cell_type_is_skipped(tmp_path):
    emb, labels, paths = _write(tmp_path, {"cty1": ("A", 20), "cty2": ("AB", 20)})
    with pytest.warns(UserWarning) as rec:
        out = mtb.classify(emb, labels=list(paths.values()))
    assert [str(w.message) for w in rec] == ["Reference batch cty1 has one cell type; skipped."]
    assert set(out["reference"]) == {"cty2"} and len(out) == 20


def test_same_seed_same_predictions(tmp_path):
    # overlapping types, so the predictions depend on the fit
    emb, labels, paths = _write(tmp_path, {"cty1": ("ABC", 60), "cty2": ("ABC", 60)})
    emb = emb * 0.05
    a = mtb.classify(emb, labels=list(paths.values()), seed=3)
    b = mtb.classify(emb, labels=list(paths.values()), seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_labels_for_dict_in_a_method_order(tmp_path):
    """A labels_for dict whose order is not the default one (StabMap: its
    reference batch first) names the batches and lines up the cells."""
    ds = tmp_path / "DS"
    ds.mkdir()
    emb, labels, paths = _write(ds, {"cty3": ("ABC", 30), "cty1": ("AB", 20),
                                     "cty2": ("ABC", 25)})
    d = mtb.labels_for("DS", "cross", "StabMap", data_path=tmp_path)
    assert list(d) == ["cty3", "cty1", "cty2"]
    out = mtb.classify(emb, labels=d, reference="cty3")
    assert out["batch"].tolist() == ["cty1"] * 20 + ["cty2"] * 25
    assert out["label"].tolist() == labels[30:]
    assert (out["label"] == out["predicted"]).all()
    # the same files in another order need label_order=, as for evaluate
    plain = {k: d[k] for k in ("cty1", "cty3", "cty2")}
    with pytest.raises(ValueError) as e:
        mtb.classify(emb, labels=plain)
    with pytest.raises(ValueError) as e2, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mtb.evaluate(emb, labels=plain)
    assert str(e.value) == str(e2.value)
    out2 = mtb.classify(emb, labels=plain, label_order=["cty3", "cty1", "cty2"],
                        reference="cty3")
    pd.testing.assert_frame_equal(out, out2)


# --- classification_metrics -------------------------------------------------

METRICS = ["OCA", "ACA", "Spec", "Sens", "F1"]
#: truth and prediction of three queries, and the five numbers the benchmark's
#: R code (compute_metrics in classification_metrics.Rmd) gives for each
R_CASES = {
    # D is predicted but never true: left out of ACA and Sens, F1 0 in the mean
    "r1": ("AABBC", "ABBDC",
           [0.59999999999999998, 0.66666666666666663, 0.8666666666666667,
            0.66666666666666663, 0.54166666666666663]),
    # C is true but never predicted: Sens 0, Spec 1 for C
    "r2": ("AABBCC", "AABBBB",
           [0.66666666666666663, 0.66666666666666663, 0.83333333333333337,
            0.66666666666666663, 0.55555555555555558]),
    # one true cell type: no true negative for A, so Spec is NaN
    "r3": ("AAAAA", "AAABA",
           [0.80000000000000004, 0.80000000000000004, np.nan,
            0.80000000000000004, 0.44444444444444442]),
}


def _pred(refs):
    """A classify-shaped frame of the queries ``refs`` of R_CASES."""
    parts = []
    for r in refs:
        truth, predicted, _ = R_CASES[r]
        parts.append(pd.DataFrame({"reference": r, "batch": "q", "cell": range(len(truth)),
                                   "label": list(truth), "predicted": list(predicted)}))
    return pd.concat(parts, ignore_index=True)


def test_metrics_equal_the_r_code():
    out = mtb.classification_metrics(_pred(["r1", "r2", "r3"]))
    assert list(out.columns) == METRICS and out.index.name == "reference"
    assert out.index.tolist() == ["r1", "r2", "r3", "mean"]
    for r in ("r1", "r2", "r3"):
        np.testing.assert_allclose(out.loc[r].to_numpy(float), R_CASES[r][2],
                                   rtol=0, atol=1e-12)


def test_mean_row_is_the_mean_over_references():
    out = mtb.classification_metrics(_pred(["r2", "r1"]))
    assert out.index.tolist() == ["r2", "r1", "mean"]    # input order
    np.testing.assert_allclose(
        out.loc["mean"].to_numpy(float),
        np.mean([R_CASES["r1"][2], R_CASES["r2"][2]], axis=0), rtol=0, atol=1e-12)
    # a NaN of one reference is not averaged away
    out = mtb.classification_metrics(_pred(["r1", "r3"]))
    assert np.isnan(out.loc["mean", "Spec"]) and not out.loc["mean"].drop("Spec").isna().any()


def test_a_frame_without_reference_gives_one_row():
    # labels that are numbers, as the R code reads them: 1, 2, 3, 10
    pred = pd.DataFrame({"label": [1, 2, 10, 2, 1, 10, 3],
                         "predicted": [1, 10, 10, 2, 2, 10, 1]})
    out = mtb.classification_metrics(pred)
    assert list(out.columns) == METRICS and len(out) == 1
    np.testing.assert_allclose(out.iloc[0].to_numpy(float),
                               [0.5714285714285714, 0.5, 0.85000000000000009, 0.5,
                                0.45000000000000001], rtol=0, atol=1e-12)


def test_metrics_of_classify(three):
    emb, labels, paths = three
    out = mtb.classification_metrics(mtb.classify(emb, labels=list(paths.values())))
    assert out.index.tolist() == ["cty1", "cty2", "cty3", "mean"]
    assert (out == 1.0).all().all()                      # well separated types


def test_missing_column_is_an_error():
    with pytest.raises(ValueError) as e:
        mtb.classification_metrics(pd.DataFrame({"label": ["A"]}))
    assert str(e.value) == ("predictions has no predicted column. "
                            "Pass the frame mtb.classify returns.")
    with pytest.raises(ValueError) as e:
        mtb.classification_metrics(pd.DataFrame({"cell": [0]}))
    assert str(e.value) == ("predictions has no label and no predicted column. "
                            "Pass the frame mtb.classify returns.")
