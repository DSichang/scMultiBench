"""``evaluate`` on a MuData takes ``"<mod>:<col>"`` labels and batch, as
``export_dataset`` does (``mdata[mod].obs[col]``); before, it raised "neither
an obs column of the AnnData (obs columns: [])"."""
import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")
md = pytest.importorskip("mudata")

import multibench as mtb


@pytest.fixture
def mdata():
    rng = np.random.default_rng(0)
    n = 120
    obs = pd.DataFrame({"celltype": np.repeat(["A", "B"], n // 2),
                        "donor": np.tile(["d1", "d2"], n // 2)},
                       index=[f"c{i}" for i in range(n)])
    m = md.MuData({"rna": ad.AnnData(rng.normal(size=(n, 5)), obs=obs),
                   "adt": ad.AnnData(rng.normal(size=(n, 4)),
                                     obs=pd.DataFrame(index=obs.index))})
    m.obsm["X_emb"] = np.vstack([rng.normal(0, 1, (n // 2, 3)),
                                 rng.normal(4, 1, (n // 2, 3))])
    return m


def _values(df):
    """evaluate's long table as {metric: value}."""
    return df.set_index("metric")["value"].round(6).to_dict()


def test_mod_column_labels_and_batch(mdata):
    got = mtb.evaluate(mdata, labels="rna:celltype", batch="rna:donor",
                       metrics=["ARI", "iLISI"], verbose=False)
    ref = mtb.evaluate(np.asarray(mdata.obsm["X_emb"]),
                       labels=mdata["rna"].obs["celltype"].to_numpy(),
                       batch=mdata["rna"].obs["donor"].to_numpy(),
                       metrics=["ARI", "iLISI"], verbose=False)
    assert _values(got) == _values(ref)


def test_an_unknown_column_lists_the_mod_columns(mdata):
    with pytest.raises(ValueError) as e:
        mtb.evaluate(mdata, labels="rna:cell_type", metrics=["ARI"], verbose=False)
    assert str(e.value).startswith("labels='rna:cell_type' is neither an obs column of "
                                   "the MuData (obs columns: [")
    assert "'rna:celltype'" in str(e.value)
