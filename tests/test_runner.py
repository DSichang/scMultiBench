import numpy as np
import h5py
from multibench.engine import io as rio
from multibench.engine.schema import OutputSpec


def test_load_embedding_output(tmp_path):
    p = tmp_path / "embedding.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.zeros((10, 5)))
    spec = OutputSpec(kind="embedding", file="embedding.h5", dataset="data")
    arr = rio.load_output(tmp_path, spec)
    assert arr.shape == (10, 5)


def test_load_graph_output(tmp_path):
    p = tmp_path / "knn_indices.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.arange(12).reshape(4, 3))
    spec = OutputSpec(kind="graph", file="knn_indices.h5", dataset="data")
    arr = rio.load_output(tmp_path, spec)
    assert arr.shape == (4, 3)
    assert int(arr[0, 0]) == 0 and int(arr[3, 2]) == 11


def test_load_labels_output(tmp_path):
    p = tmp_path / "Prediction.csv"
    p.write_text("B\nT\nNK\n")
    spec = OutputSpec(kind="labels", file="Prediction.csv")
    labels = rio.load_output(tmp_path, spec)
    assert list(labels) == ["B", "T", "NK"]


