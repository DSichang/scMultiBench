import numpy as np
import pandas as pd
import h5py
from multibench.eval import io


def test_read_embedding_autoorients_to_cells_by_dims(tmp_path):
    # store as dims x cells (4 x 10) -> reader returns 10 x 4.
    # Use distinct values so we verify the data is correctly transposed,
    # not merely reshaped: stored[d, c] must land at result[c, d].
    p = tmp_path / "embedding.h5"
    stored = np.arange(40, dtype=float).reshape(4, 10)  # (dims=4, cells=10)
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=stored)
    X = io.read_embedding(p)
    assert X.shape == (10, 4)
    np.testing.assert_array_equal(X, stored.T)
    # cell 0 should hold the dim-0..dim-3 values from column 0 of the source
    np.testing.assert_array_equal(X[0], stored[:, 0])
    np.testing.assert_array_equal(X[9], stored[:, 9])


def test_read_embedding_keeps_cells_by_dims(tmp_path):
    p = tmp_path / "embedding.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.zeros((100, 8)))
    assert io.read_embedding(p).shape == (100, 8)


def test_read_labels_skips_header_and_codes(tmp_path):
    p = tmp_path / "cty.csv"
    pd.DataFrame(["celltype", "B", "T", "B", "NK"]).to_csv(p, index=False, header=False)
    labels = io.read_labels(p)
    assert len(labels) == 4              # header row dropped
    assert labels.dtype.kind in "iu"     # integer codes
    assert labels[0] == labels[2]        # both "B"


def test_read_clustering_decodes_bytes(tmp_path):
    p = tmp_path / "cluster.h5"
    with h5py.File(p, "w") as f:
        grp = f.create_group("obs")
        grp.create_dataset("cluster_leiden", data=np.array([b"0", b"1", b"0"]))
    c = io.read_clustering(p)
    assert list(c) == [0, 1, 0]


def test_evaluate_rejects_non_scib():
    import pytest
    from multibench.eval import pipeline
    with pytest.raises(NotImplementedError) as exc:
        pipeline.evaluate(output="x.h5", category="mosaic", task="imputation",
                          labels="c.csv")
    assert "imputation" in str(exc.value)


def test_evaluate_requires_labels_and_clustering(tmp_path):
    import pytest
    from multibench.eval import pipeline
    with pytest.raises(ValueError):
        pipeline.evaluate(output="x.h5", category="vertical", task="clustering",
                          labels=None, clustering=None)


def test_evaluate_requires_batch_for_batch_task():
    import pytest
    from multibench.eval import pipeline
    emb = np.zeros((6, 3))
    cl = np.array([0, 0, 1, 1, 2, 2])
    ct = cl.copy()
    with pytest.raises(ValueError) as exc:
        pipeline.evaluate(output=emb, category="x", task="batch",
                          labels=ct, clustering=cl, batch=None)
    assert "batch labels required" in str(exc.value)


def test_compute_rejects_length_mismatch():
    import pytest
    anndata = pytest.importorskip("anndata")
    from multibench.eval import scib as escib
    emb = np.zeros((6, 3))
    ct = np.array([0, 0, 1, 1, 2])      # one short -> mismatch
    cl = np.array([0, 0, 1, 1, 2, 2])
    ba = np.array([0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError) as exc:
        escib.compute(emb, ct, cl, ba, group="clustering")
    assert "length mismatch" in str(exc.value)


def test_compute_rejects_unknown_group():
    import pytest
    anndata = pytest.importorskip("anndata")
    from multibench.eval import scib as escib
    emb = np.zeros((4, 2))
    lab = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError) as exc:
        escib.compute(emb, lab, lab, lab, group="bogus")
    assert "unknown group" in str(exc.value)


def test_top_level_exposes_evaluate():
    import multibench as mtb
    assert hasattr(mtb, "evaluate")
