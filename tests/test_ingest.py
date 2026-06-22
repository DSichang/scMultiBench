import numpy as np
import h5py
import pytest
from multibench.run import ingest


def test_canonical_passthrough(tmp_path):
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("matrix/data", data=np.zeros((3, 5)))
    out = ingest.to_canonical(p, out=tmp_path / "y.h5", convert=False)
    assert str(out) == str(p)  # untouched


def test_from_anndata_writes_features_by_cells(tmp_path):
    ad = pytest.importorskip("anndata")
    import numpy as np
    a = ad.AnnData(np.arange(6, dtype=float).reshape(2, 3))  # 2 cells x 3 genes
    a.var_names = ["g0", "g1", "g2"]
    a.obs_names = ["c0", "c1"]
    out = ingest.to_canonical(a, out=tmp_path / "c.h5")
    with h5py.File(out, "r") as f:
        data = np.array(f["matrix/data"])
        feats = [x.decode() if isinstance(x, bytes) else x for x in np.array(f["matrix/features"])]
        bars = [x.decode() if isinstance(x, bytes) else x for x in np.array(f["matrix/barcodes"])]
    assert data.shape == (3, 2)              # features x cells
    assert feats == ["g0", "g1", "g2"]
    assert bars == ["c0", "c1"]


def test_roundtrip_read_canonical(tmp_path):
    ad = pytest.importorskip("anndata")
    import numpy as np
    a = ad.AnnData(np.arange(6, dtype=float).reshape(2, 3))
    out = ingest.to_canonical(a, out=tmp_path / "c.h5")
    back = ingest.read_canonical(out)
    assert back.shape == (2, 3)              # cells x genes restored


def test_from_csv(tmp_path):
    import pandas as pd, numpy as np
    p = tmp_path / "m.csv"  # rows=cells, cols=genes
    pd.DataFrame(np.ones((4, 2))).to_csv(p, index=False)
    out = ingest.to_canonical(p, out=tmp_path / "c.h5")
    with h5py.File(out, "r") as f:
        assert np.array(f["matrix/data"]).shape == (2, 4)  # genes x cells
