"""classify: the predicted cell type of each query cell, each batch in turn the
reference (the benchmark's MLP classification step), on the arrays evaluate has
checked; classification_metrics scores its frame. evaluate reports the mean row."""

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.eval.classify import classification_metrics, classify

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
    given block order, the labels in that order and the batch (stem) of each cell."""
    embs, labels, batch, paths = [], [], [], {}
    for i, (stem, (types, n)) in enumerate(blocks.items()):
        e, lab = _batch(types, n, seed=i)
        embs.append(e)
        labels += lab
        batch += [stem] * n
        paths[stem] = tmp_path / f"{stem}.csv"
        pd.DataFrame({"x": lab}).to_csv(paths[stem], index=False)
    return np.vstack(embs), labels, batch, paths


@pytest.fixture
def three(tmp_path):
    """Three batches of 30 cells; cty1 has no cell of type C."""
    return _write(tmp_path, {"cty1": ("AB", 30), "cty2": ("ABC", 30), "cty3": ("ABC", 30)})


def test_return_frame(three):
    emb, labels, batch, _ = three
    out = classify(emb, labels, batch)
    assert list(out.columns) == COLUMNS
    assert isinstance(out.index, pd.RangeIndex)
    first = out[out["reference"] == "cty1"]
    # the query of cty1 is cty2 then cty3, in the rows of the output
    assert first["batch"].tolist() == ["cty2"] * 20 + ["cty3"] * 20
    assert first["cell"].tolist() == [i for i in range(30, 90) if labels[i] != "C"]
    assert first["label"].tolist() == [labels[i] for i in first["cell"]]
    assert (out["label"] == out["predicted"]).all()        # well separated types


def test_every_batch_is_the_reference_in_turn(three):
    emb, labels, batch, _ = three
    out = classify(emb, labels, batch)
    assert out["reference"].unique().tolist() == ["cty1", "cty2", "cty3"]
    assert out.groupby("reference").size().to_dict() == {"cty1": 40, "cty2": 60, "cty3": 60}
    assert not (out["reference"] == out["batch"]).any()


def test_query_cells_of_a_type_not_in_the_reference_are_left_out(three):
    emb, labels, batch, _ = three
    out = classify(emb, labels, batch)
    first = out[out["reference"] == "cty1"]
    assert "C" not in set(first["label"]) and "C" not in set(first["predicted"])
    assert len(first) == 40 and out.attrs["left_out"] == {"cty1": 20, "cty2": 0, "cty3": 0}


def test_a_graph_or_kNN_output_is_an_error(three, tmp_path):
    emb, labels, batch, paths = three
    files = list(paths.values())
    # scMoMaT's umap_embedding.h5 is written only by a graph variant
    umap = tmp_path / "umap_embedding.h5"
    with h5py.File(umap, "w") as f:
        f["data"] = emb[:, :2]
    with pytest.raises(ValueError) as e:
        mtb.evaluate(umap, labels=files, metrics="classification")
    assert str(e.value).startswith("scMoMaT writes a graph, not an embedding")
    # cells x cells, and a kNN index matrix as mtb.run returns it for scMoMaT
    knn = np.stack([np.roll(np.arange(90), -i)[:20] for i in range(90)])
    for output in (np.eye(90), knn, knn.astype(float), knn + 1):
        with pytest.raises(ValueError) as e:
            mtb.evaluate(output, labels=files, metrics="classification")
        assert str(e.value) == ("output is a graph, not an embedding, so it has no "
                                "classification scores. Pass metrics='clustering'.")


def test_the_neighbour_index_rule():
    from multibench.eval.classify import _is_neighbour_index
    rng = np.random.default_rng(0)
    knn = np.stack([rng.permutation(200)[:15] for _ in range(200)])
    assert _is_neighbour_index(knn) and _is_neighbour_index(knn + 1)
    assert _is_neighbour_index(knn.astype(np.float32))
    emb = rng.normal(size=(200, 15))
    assert not _is_neighbour_index(emb)
    assert not _is_neighbour_index(np.round(emb, 3))
    assert not _is_neighbour_index(np.zeros((200, 15)))     # a value twice in a row
    assert not _is_neighbour_index(knn[:, :1])              # one column
    assert not _is_neighbour_index(knn + 2)                 # beyond the cells
    assert not _is_neighbour_index(knn[:10])                # more columns than cells


def test_a_reference_with_one_cell_type_is_skipped(tmp_path):
    emb, labels, batch, _ = _write(tmp_path, {"cty1": ("A", 20), "cty2": ("AB", 20)})
    with pytest.warns(UserWarning) as rec:
        out = classify(emb, labels, batch)
    assert [str(w.message) for w in rec] == ["Reference batch cty1 has one cell type; skipped."]
    assert set(out["reference"]) == {"cty2"} and len(out) == 20


def test_same_seed_same_predictions(tmp_path):
    # overlapping types, so the predictions depend on the fit
    emb, labels, batch, _ = _write(tmp_path, {"cty1": ("ABC", 60), "cty2": ("ABC", 60)})
    emb = emb * 0.05
    a = classify(emb, labels, batch, seed=3)
    b = classify(emb, labels, batch, seed=3)
    pd.testing.assert_frame_equal(a, b)


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
    out = classification_metrics(_pred(["r1", "r2", "r3"]))
    assert list(out.columns) == METRICS and out.index.name == "reference"
    assert out.index.tolist() == ["r1", "r2", "r3", "mean"]
    for r in ("r1", "r2", "r3"):
        np.testing.assert_allclose(out.loc[r].to_numpy(float), R_CASES[r][2],
                                   rtol=0, atol=1e-12)


def test_mean_row_is_the_mean_over_references():
    out = classification_metrics(_pred(["r2", "r1"]))
    assert out.index.tolist() == ["r2", "r1", "mean"]    # input order
    np.testing.assert_allclose(
        out.loc["mean"].to_numpy(float),
        np.mean([R_CASES["r1"][2], R_CASES["r2"][2]], axis=0), rtol=0, atol=1e-12)
    # a NaN of one reference is not averaged away
    out = classification_metrics(_pred(["r1", "r3"]))
    assert np.isnan(out.loc["mean", "Spec"]) and not out.loc["mean"].drop("Spec").isna().any()


def test_metrics_of_classify(three):
    emb, labels, batch, _ = three
    out = classification_metrics(classify(emb, labels, batch))
    assert out.index.tolist() == ["cty1", "cty2", "cty3", "mean"]
    assert (out == 1.0).all().all()                      # well separated types
