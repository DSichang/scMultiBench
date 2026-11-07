"""export_dataset names an obs column with or without the ``obs:`` prefix.

``"obs:<column>"`` and ``"<column>"`` name a column of the object passed.
0.3.4 took the first only and answered the second with a line about
selector syntax. A name that is no column now names the columns there are.
"""

import numpy as np
import pandas as pd
import pytest

from multibench.engine import ingest

ad = pytest.importorskip("anndata")


def _cells(n=60, tag="c", seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, (n, 20)).astype(float))
    a.var_names = [f"GENE{i}" for i in range(20)]
    a.obs_names = [f"{tag}{i}" for i in range(n)]
    a.obs["celltype"] = np.array(["T", "B", "NK"])[rng.integers(0, 3, n)]
    a.obs["donor"] = np.array(["d1", "d2"])[np.arange(n) % 2]
    return a


def _files(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def _key_error(tmp_path, data, **kw):
    out = tmp_path / "OUT"
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(data, out, **kw)
    assert not out.exists()
    return e.value.args[0]


def test_a_column_whose_name_looks_like_a_file_is_a_column(tmp_path):
    a = _cells()
    a.obs["annotation.csv"] = a.obs["celltype"]
    out = ingest.export_dataset(a, tmp_path / "COL", rna="X", labels="annotation.csv")
    assert pd.read_csv(out / "cty.csv")["x"].tolist() == list(a.obs["celltype"])


def test_a_name_that_is_no_column(tmp_path):
    a = _cells()
    assert _key_error(tmp_path, a, rna="X", labels="nope") == (
        "labels='nope': the AnnData's obs has no column 'nope'. Columns: celltype, donor.")
    assert _key_error(tmp_path, a, rna="X", batch="sample", category="cross") == (
        "batch='sample': the AnnData's obs has no column 'sample'. Columns: celltype, donor.")
    # a column of the list, as it is written there
    out = ingest.export_dataset(a, tmp_path / "LISTED", rna="X", labels="celltype")
    assert pd.read_csv(out / "cty.csv")["x"].tolist() == list(a.obs["celltype"])
    # with the prefix: the same text
    assert _key_error(tmp_path, a, rna="X", labels="obs:nope") == (
        "labels='obs:nope': the AnnData's obs has no column 'nope'. Columns: celltype, donor.")


def test_at_most_five_columns_are_listed(tmp_path):
    a = _cells()
    for i in range(3):
        a.obs[f"qc{i}"] = i
    assert _key_error(tmp_path, a, rna="X", labels="nope") == (
        "labels='nope': the AnnData's obs has no column 'nope'. Columns: celltype, donor, qc0, "
        "qc1, qc2.")
    for i in range(3, 1203):
        a.obs[f"qc{i}"] = i
    assert _key_error(tmp_path, a, rna="X", labels="nope") == (
        "labels='nope': the AnnData's obs has no column 'nope'. Columns: celltype, donor, qc0, "
        "qc1, qc2 and 1,200 more.")
    a.obs = pd.DataFrame(index=a.obs_names)
    assert _key_error(tmp_path, a, rna="X", labels="nope") == (
        "labels='nope': the AnnData's obs has no column 'nope'. Columns: none.")


def test_diagonal_takes_the_bare_name(tmp_path):
    rna, atac = _cells(60, "r"), _cells(40, "a", seed=1)
    kw = dict(rna="X", atac=atac, atac_kind="gene_activity", category="diagonal")
    bare = ingest.export_dataset(rna, tmp_path / "BARE", labels="celltype", **kw)
    prefixed = ingest.export_dataset(rna, tmp_path / "PREFIXED", labels="obs:celltype", **kw)
    assert sorted(_files(bare)) == ["atac_cty.csv", "atac_gas.h5", "rna.h5", "rna_cty.csv"]
    for name in ("rna_cty.csv", "atac_cty.csv"):
        assert (bare / name).read_bytes() == (prefixed / name).read_bytes()
    assert pd.read_csv(bare / "atac_cty.csv")["x"].tolist() == list(atac.obs["celltype"])
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, tmp_path / "OUT", labels="annotation.csv", **kw)
    assert e.value.args[0] == ("labels='annotation.csv': the obs of the RNA object (data) has "
                               "no column 'annotation.csv'. Columns: celltype, donor.")


@pytest.mark.parametrize("spec", ["obs:celltype", "celltype"])
def test_diagonal_side_without_the_column(tmp_path, spec):
    """Cell types for one side only: the error names the side and its columns."""
    rna, atac = _cells(60, "r"), _cells(40, "a", seed=1)
    del atac.obs["celltype"]
    out = tmp_path / "OUT"
    kw = dict(rna="X", atac_kind="gene_activity", labels=spec, category="diagonal")
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, out, atac=atac, **kw)
    assert e.value.args[0] == (
        f"labels={spec!r}: the obs of the ATAC object (atac) has no column 'celltype'. "
        f"Columns: donor.")
    del rna.obs["celltype"]
    atac.obs["celltype"] = "T"
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, out, atac=atac, **kw)
    assert e.value.args[0] == (
        f"labels={spec!r}: the obs of the RNA object (data) has no column 'celltype'. "
        f"Columns: donor.")
    assert not out.exists()


def test_mudata_spellings_are_unchanged(tmp_path):
    """The spellings a MuData had in 0.3.4 read the same columns and raise
    the same errors. The bare name of a column of mdata.obs is new."""
    mu = pytest.importorskip("mudata")
    rna = _cells()
    adt = ad.AnnData(np.ones((rna.n_obs, 4)))
    adt.obs_names, adt.var_names = rna.obs_names, ["CD3", "CD4", "CD8", "CD19"]
    m = mu.MuData({"rna": rna, "adt": adt})
    m.obs["site"] = np.array(["s1", "s2", "s3"])[np.arange(rna.n_obs) % 3]
    truth = list(rna.obs["celltype"])
    for i, spec in enumerate(("rna:celltype", "mod:rna.obs:celltype")):
        d = ingest.export_dataset(m, tmp_path / f"M{i}", rna="rna", labels=spec)
        assert pd.read_csv(d / "cty.csv")["x"].tolist() == truth
    d = ingest.export_dataset(m, tmp_path / "G", rna="rna", labels="obs:site")
    e = ingest.export_dataset(m, tmp_path / "H", rna="rna", labels="site")       # new
    assert (d / "cty.csv").read_bytes() == (e / "cty.csv").read_bytes()
    assert pd.read_csv(d / "cty.csv")["x"].tolist() == list(m.obs["site"])
    out = tmp_path / "OUT"
    with pytest.raises(KeyError) as err:
        ingest.export_dataset(m, out, rna="rna", labels="rna:nope")
    assert err.value.args[0] == (
        "labels='rna:nope': mdata['rna'].obs has no column 'nope'. Columns: celltype, donor.")
    with pytest.raises(KeyError) as err:
        ingest.export_dataset(m, out, rna="rna", labels="obs:nope")
    assert err.value.args[0] == (
        "labels='obs:nope': mdata.obs has no column 'nope'. Columns: site.")
    with pytest.raises(KeyError) as err:
        ingest.export_dataset(m, out, rna="rna", labels="mod:rna.obs:nope")
    assert err.value.args[0] == (
        "labels='mod:rna.obs:nope': mdata['rna'].obs has no column 'nope'. Columns: celltype, "
        "donor.")
    with pytest.raises(ValueError, match=r"use 'obs:<col>' \(or 'mod:<name>.obs:<col>'\)"):
        ingest.export_dataset(m, out, rna="rna", labels="mod:rna.celltype")
    with pytest.raises(KeyError) as err:
        ingest.export_dataset(m, out, rna="rna", labels="nope")
    assert err.value.args[0] == "labels='nope': mdata.obs has no column 'nope'. Columns: site."
    assert not out.exists()
    # a missing label of a column read by its bare name
    site = m.obs["site"].astype(object).to_numpy()
    site[0] = np.nan
    m.obs["site"] = site
    with pytest.raises(ValueError) as err:
        ingest.export_dataset(m, out, rna="rna", labels="site")
    assert str(err.value).startswith(
        "labels='site': 1 of the 60 labels for the cells are missing")


def test_a_modality_prefix_on_an_anndata_is_a_column_name(tmp_path, capsys):
    """``'<mod>:<col>'`` reads a column of one modality of a MuData. Two
    AnnData objects have no modalities: the name is a column name, and the
    error lists the columns there are."""
    rna, atac = (_cells(60, 'r'), _cells(40, 'a', seed=1))
    kw = dict(rna='X', atac=atac, atac_kind='gene_activity', category='diagonal')
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, tmp_path / 'OUT', labels='rna:celltype', **kw)
    assert e.value.args[0] == ("labels='rna:celltype': the obs of the RNA object (data) has "
                               "no column 'rna:celltype'. Columns: celltype, donor.")
    out = ingest.export_dataset(rna, tmp_path / 'OBS', labels='obs:celltype', **kw)
    assert sorted(_files(out)) == ['atac_cty.csv', 'atac_gas.h5', 'rna.h5', 'rna_cty.csv']
    for a in (rna, atac):
        a.obs['rna:celltype'] = a.obs['celltype'].to_numpy()
    named = ingest.export_dataset(rna, tmp_path / 'NAMED', labels='rna:celltype', **kw)
    for name in ('rna_cty.csv', 'atac_cty.csv'):
        assert (named / name).read_bytes() == (out / name).read_bytes()
    for a in (rna, atac):
        del a.obs['rna:celltype']
    assert not (tmp_path / 'OUT').exists()
