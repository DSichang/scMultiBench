"""export_dataset names an obs column with or without the ``obs:`` prefix.

``"obs:<column>"`` and ``"<column>"`` name a column of the object passed.
0.3.4 took the first only and answered the second with a line about
selector syntax. A name that is no column now says which columns there are,
and a file name says that ``labels`` and ``batch`` take a column.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from multibench import cli
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


def _value_error(tmp_path, data, **kw):
    out = tmp_path / "OUT"
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(data, out, **kw)
    assert not out.exists()
    return str(e.value)


def test_bare_and_prefixed_names_write_the_same_labels(tmp_path):
    a = _cells()
    bare = ingest.export_dataset(a, tmp_path / "BARE", rna="X", labels="celltype")
    prefixed = ingest.export_dataset(a, tmp_path / "PREFIXED", rna="X", labels="obs:celltype")
    assert (bare / "cty.csv").read_bytes() == (prefixed / "cty.csv").read_bytes()
    assert pd.read_csv(bare / "cty.csv")["x"].tolist() == list(a.obs["celltype"])
    kw = dict(rna="X", category="cross")
    bare = ingest.export_dataset(a, tmp_path / "B1", labels="celltype", batch="donor", **kw)
    prefixed = ingest.export_dataset(a, tmp_path / "B2", labels="obs:celltype",
                                     batch="obs:donor", **kw)
    assert sorted(_files(bare)) == ["cty1.csv", "cty2.csv", "rna1.h5", "rna2.h5"]
    for name in ("cty1.csv", "cty2.csv"):
        assert (bare / name).read_bytes() == (prefixed / name).read_bytes()
    for name in ("rna1.h5", "rna2.h5"):
        x, y = (ingest.read_canonical(d / name, sparse=False) for d in (bare, prefixed))
        assert np.array_equal(x.X, y.X) and list(x.obs_names) == list(y.obs_names)
    # the column itself, as before
    series = ingest.export_dataset(a, tmp_path / "B3", labels=a.obs["celltype"],
                                   batch=a.obs["donor"], **kw)
    assert (series / "cty2.csv").read_bytes() == (bare / "cty2.csv").read_bytes()


def test_a_column_whose_name_looks_like_a_file_is_a_column(tmp_path):
    a = _cells()
    a.obs["annotation.csv"] = a.obs["celltype"]
    out = ingest.export_dataset(a, tmp_path / "COL", rna="X", labels="annotation.csv")
    assert pd.read_csv(out / "cty.csv")["x"].tolist() == list(a.obs["celltype"])


def test_a_name_that_is_no_column(tmp_path):
    a = _cells()
    assert _value_error(tmp_path, a, rna="X", labels="nope") == (
        "labels='nope' is not a column of data.obs. Its columns are ['celltype', 'donor']. "
        "Name a column, as labels='obs:<col>' or labels='<col>', or pass the values as a "
        "Series indexed by barcode.")
    assert _value_error(tmp_path, a, rna="X", batch="sample", category="cross") == (
        "batch='sample' is not a column of data.obs. Its columns are ['celltype', 'donor']. "
        "Name a column, as batch='obs:<col>' or batch='<col>', or pass the values as a "
        "Series indexed by barcode.")
    # with the prefix: the error of 0.3.4, word for word
    out = tmp_path / "OUT"
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(a, out, rna="X", labels="obs:nope")
    assert e.value.args[0] == ("labels='obs:nope': column 'nope' not in obs: searched "
                               "data.obs; available: ['celltype', 'donor']")
    assert not out.exists()


def _cli_error(argv, capsys):
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli.main(argv)
    err = capsys.readouterr().err
    return rc, next(ln for ln in err.splitlines() if ln.startswith("error: "))


def test_a_file_name_gets_its_own_message(tmp_path, capsys, monkeypatch):
    a = _cells()
    assert _value_error(tmp_path, a, rna="X", labels="annotation.csv") == (
        "labels='annotation.csv' is a file name, and labels= takes a column of the data. "
        "Read the file and add the column first, for example adata.obs['celltype'] = "
        "pd.read_csv('annotation.csv', index_col=0)['<column>'], then pass "
        "labels='obs:celltype'.")
    assert _value_error(tmp_path, a, rna="X", batch="donors.TSV", category="cross") == (
        "batch='donors.TSV' is a file name, and batch= takes a column of the data. Read the "
        "file and add the column first, for example adata.obs['batch'] = "
        "pd.read_csv('donors.TSV', index_col=0)['<column>'], then pass batch='obs:batch'.")
    # a file that exists, whatever its name
    table = tmp_path / "cell_types"
    table.write_text("barcode,celltype\nc0,T\n")
    assert _value_error(tmp_path, a, rna="X", labels=str(table)).startswith(
        f"labels={str(table)!r} is a file name, and labels= takes a column of the data.")
    # the command line names its option and no Python keyword
    a.write_h5ad(tmp_path / "a.h5ad")
    argv = ["convert", str(tmp_path / "a.h5ad"), str(tmp_path / "OUT"), "--rna", "X"]
    rc, line = _cli_error(argv + ["--labels", "annotation.csv"], capsys)
    assert rc == 1 and line == (
        "error: --labels annotation.csv is a file name, and --labels takes a column of SRC, "
        "as obs:<col>. Add the column to the .h5ad file first.")
    assert "labels=" not in line
    rc, line = _cli_error(argv + ["--labels", "nope"], capsys)
    assert rc == 1 and line == (
        "error: --labels nope is not a column of the obs of SRC. Its columns are "
        "['celltype', 'donor']. Name a column, as --labels obs:<col>.")
    assert not (tmp_path / "OUT").exists()
    rc = cli.main(argv + ["--labels", "celltype"])
    assert rc == 0 and (tmp_path / "OUT" / "cty.csv").is_file()


def test_diagonal_takes_the_bare_name(tmp_path):
    rna, atac = _cells(60, "r"), _cells(40, "a", seed=1)
    kw = dict(rna="X", atac=atac, atac_kind="gene_activity", category="diagonal")
    bare = ingest.export_dataset(rna, tmp_path / "BARE", labels="celltype", **kw)
    prefixed = ingest.export_dataset(rna, tmp_path / "PREFIXED", labels="obs:celltype", **kw)
    assert sorted(_files(bare)) == ["atac_cty.csv", "atac_gas.h5", "rna.h5", "rna_cty.csv"]
    for name in ("rna_cty.csv", "atac_cty.csv"):
        assert (bare / name).read_bytes() == (prefixed / name).read_bytes()
    assert pd.read_csv(bare / "atac_cty.csv")["x"].tolist() == list(atac.obs["celltype"])
    assert _value_error(tmp_path, rna, labels="annotation.csv", **kw).startswith(
        "labels='annotation.csv' is a file name, and labels= takes a column of the data.")


@pytest.mark.parametrize("spec", ["obs:celltype", "celltype"])
def test_diagonal_side_without_the_column(tmp_path, spec):
    """Cell types for one side only: the old error names the side, and now
    also what to do without them."""
    rna, atac = _cells(60, "r"), _cells(40, "a", seed=1)
    del atac.obs["celltype"]
    out = tmp_path / "OUT"
    kw = dict(rna="X", atac_kind="gene_activity", labels=spec, category="diagonal")
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, out, atac=atac, **kw)
    assert e.value.args[0] == (
        f"labels={spec!r}: column 'celltype' not in obs of the ATAC object (atac); "
        f"available: ['donor'] Without cell types for the ATAC cells, leave labels= out and "
        f"run with evaluate=False.")
    del rna.obs["celltype"]
    atac.obs["celltype"] = "T"
    with pytest.raises(KeyError) as e:
        ingest.export_dataset(rna, out, atac=atac, **kw)
    assert e.value.args[0] == (
        f"labels={spec!r}: column 'celltype' not in obs of the RNA object (data); "
        f"available: ['donor'] Without cell types for the RNA cells, leave labels= out and "
        f"run with evaluate=False.")
    assert not out.exists()


def test_diagonal_side_without_the_column_on_the_command_line(tmp_path, capsys):
    rna, atac = _cells(60, "r"), _cells(40, "a", seed=1)
    del atac.obs["celltype"]
    rna.write_h5ad(tmp_path / "rna.h5ad")
    atac.write_h5ad(tmp_path / "atac.h5ad")
    rc, line = _cli_error(["convert", str(tmp_path / "rna.h5ad"), str(tmp_path / "OUT"),
                           "--rna", "X", "--atac-from", str(tmp_path / "atac.h5ad"),
                           "--atac-kind", "gene_activity", "--labels", "obs:celltype",
                           "--category", "diagonal"], capsys)
    assert rc == 1 and not (tmp_path / "OUT").exists()
    assert line.endswith("available: ['donor'] Without cell types for the ATAC cells, leave "
                         "--labels out and run with --no-evaluate.")


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
    assert err.value.args[0].startswith(
        "labels='rna:nope': column 'nope' not in obs: searched mdata['rna'].obs and "
        "mdata.obs['rna:nope']; mdata['rna'].obs has ['celltype', 'donor']")
    with pytest.raises(KeyError) as err:
        ingest.export_dataset(m, out, rna="rna", labels="obs:nope")
    assert err.value.args[0].startswith(
        "labels='obs:nope': column 'nope' not in obs: searched mdata.obs; available: ")
    with pytest.raises(KeyError) as err:
        ingest.export_dataset(m, out, rna="rna", labels="mod:rna.obs:nope")
    assert err.value.args[0] == (
        "labels='mod:rna.obs:nope': column 'nope' not in obs: searched mdata['rna'].obs; "
        "available: ['celltype', 'donor']")
    with pytest.raises(ValueError, match=r"use 'obs:<col>' \(or 'mod:<name>.obs:<col>'\)"):
        ingest.export_dataset(m, out, rna="rna", labels="mod:rna.celltype")
    with pytest.raises(ValueError) as err:
        ingest.export_dataset(m, out, rna="rna", labels="nope")
    assert str(err.value).startswith("labels='nope' is not a column of mdata.obs. Its "
                                     "columns are [")
    assert str(err.value).endswith("Name a column, as labels='obs:<col>' or labels='<col>', "
                                   "or pass the values as a Series indexed by barcode.")
    assert not out.exists()
    # a missing label of a column read by its bare name names mdata.obs
    site = m.obs["site"].astype(object).to_numpy()
    site[0] = np.nan
    m.obs["site"] = site
    with pytest.raises(ValueError) as err:
        ingest.export_dataset(m, out, rna="rna", labels="site")
    assert str(err.value).startswith(
        "labels='site': 1 of the 60 labels for the cells (mdata.obs) are missing")
