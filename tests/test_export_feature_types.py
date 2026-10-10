"""One file holds one modality: features of several types are not written together.

A 10x Genomics file read with ``gex_only=False`` holds genes and peaks (or
genes and antibodies) in one ``X``, told apart by ``var['feature_types']``.
0.3.4 wrote such an ``X`` as ``rna.h5`` when no filter was given, and
``scan`` reported the folder as complete. ``export_dataset``,
``to_canonical`` and the conversion inside ``mtb.run`` now raise, and the
10x ``.h5`` file itself is named as what it is instead of being taken for a
dataset file.
"""
import re

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench.engine import ingest, runner

ad = pytest.importorskip("anndata")

GENES, PEAKS, PROTEINS = 800, 1200, 12
LIST = "1,200 'Peaks' and 800 'Gene Expression'"
HEAD = (f"2,000 features of 2 types: {LIST} (var['feature_types']). One file holds one "
        f"modality.")
RECIPE = 'The documentation has the steps as the recipe "A 10x .h5 file".'


def _exit(kind, obj="adata"):
    return (f"If all of them are {kind} features, delete the column from the object first, "
            f"for example del {obj}.var['feature_types'].")


def _mixed(types, n=20, seed=0):
    """An AnnData whose ``X`` holds features of several types:
    ``types`` is ``[(type name, number of features, name stem), ...]``."""
    rng = np.random.default_rng(seed)
    names = [f"{stem}{i}" for _, k, stem in types for i in range(k)]
    a = ad.AnnData(rng.poisson(1.0, (n, len(names))).astype(float))
    a.var_names = names
    a.obs_names = [f"cell{i}-1" for i in range(n)]
    a.var["feature_types"] = pd.Categorical([t for t, k, _ in types for _ in range(k)])
    a.var["genome"] = "GRCh38"
    a.obs["celltype"] = ["T", "B"] * (n // 2)
    a.layers["counts"] = a.X.copy()
    return a


def _arc():
    """A 10x Multiome object: genes and peaks in one X."""
    a = _mixed([("Gene Expression", GENES, "GENE"), ("Peaks", PEAKS, "chr1:1-")])
    a.var_names = [f"GENE{i}" for i in range(GENES)] \
        + [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(PEAKS)]
    return a


def _cite():
    """A 10x CITE-seq object: genes and antibodies in one X."""
    return _mixed([("Gene Expression", GENES, "GENE"), ("Antibody Capture", PROTEINS, "CD")])


def _raises(tmp_path, *args, **kw):
    """The message of a ``ValueError`` from export_dataset; the folder was not made."""
    out = tmp_path / "OUT"
    with pytest.raises(ValueError) as e:
        ingest.export_dataset(*args[:1], out, *args[1:], **kw)
    assert not out.exists()
    return str(e.value)


# ---------------------------------------------------------------- F1: a selector
def test_the_default_rna_selector_next_to_an_atac_filter(tmp_path):
    """The classic call: a filter for the ATAC, and rna left at its default."""
    msg = _raises(tmp_path, _arc(), atac="X[feature_types=Peaks]", atac_kind="peak")
    assert msg == (f"rna='X' selects {HEAD} Add a filter, for example "
                   f"rna='X[feature_types=Gene Expression]'. {_exit('RNA')}")


def test_a_layer_selector(tmp_path):
    msg = _raises(tmp_path, _arc(), rna="layer:counts")
    assert msg == (f"rna='layer:counts' selects {HEAD} Add a filter, for example "
                   f"rna='layer:counts[feature_types=Gene Expression]'. {_exit('RNA')}")


def test_adt_from_x_of_a_cite_seq_object(tmp_path):
    msg = _raises(tmp_path, _cite(), rna="X[feature_types=Gene Expression]", adt="X")
    assert msg == (
        "adt='X' selects 812 features of 2 types: 800 'Gene Expression' and 12 'Antibody "
        "Capture' (var['feature_types']). One file holds one modality. Add a filter, for "
        f"example adt='X[feature_types=Antibody Capture]'. {_exit('ADT')}")


def test_a_mudata_modality_name(tmp_path):
    mudata = pytest.importorskip("mudata")
    mdata = mudata.MuData({"rna": _arc()})
    msg = _raises(tmp_path, mdata, rna="rna")
    assert msg == (f"rna='rna' selects {HEAD} Add a filter, for example "
                   f"rna='rna[feature_types=Gene Expression]'. {_exit('RNA')}")
    out = ingest.export_dataset(mdata, tmp_path / "MU",
                                rna="rna[feature_types=Gene Expression]")
    assert ingest.read_canonical(out / "rna.h5").n_vars == GENES


# ------------------------------------------- F2: another filter, or an object
def test_a_selector_with_another_filter(tmp_path):
    msg = _raises(tmp_path, _arc(), rna="X[genome=GRCh38]")
    assert msg == (f"rna='X[genome=GRCh38]' holds {HEAD} Pass the features of one type, for "
                   f"example rna=adata[:, adata.var['feature_types'] == 'Gene Expression']. "
                   f"{_exit('RNA')}")


def test_an_object(tmp_path):
    arc = _arc()
    msg = _raises(tmp_path, None, rna=arc)
    assert msg == (f"rna=AnnData holds {HEAD} Pass the features of one type, for example "
                   f"rna=adata[:, adata.var['feature_types'] == 'Gene Expression']. "
                   f"{_exit('RNA', obj='rna')}")
    genes = arc[:, arc.var["feature_types"] == "Gene Expression"]
    msg = _raises(tmp_path, genes, atac=arc, atac_kind="peak")
    assert msg == (f"atac=AnnData holds {HEAD} Pass the features of one type, for example "
                   f"atac=adata[:, adata.var['feature_types'] == 'Peaks']. "
                   f"{_exit('ATAC', obj='atac')}")


# ----------------------------------------------- the example, or none at all
def test_the_example_in_the_message_works(tmp_path):
    """The selector the message shows is taken as it is, and the file then
    holds that type only."""
    arc, cite = _arc(), _cite()
    for data, kw, role, n_feat in (
            (arc, dict(atac="X[feature_types=Peaks]", atac_kind="peak"), "rna", GENES),
            (arc, dict(rna="X[feature_types=Gene Expression]", atac="layer:counts",
                       atac_kind="peak"), "atac", PEAKS),
            (cite, dict(rna="X[feature_types=Gene Expression]", adt="X"), "adt", PROTEINS)):
        msg = _raises(tmp_path, data, **kw)
        arg, selector = re.search(r"for example (\w+)='([^']+)'\.", msg).groups()
        assert arg == role
        out = ingest.export_dataset(data, tmp_path / f"OK_{role}", **{**kw, arg: selector},
                                    category="vertical")
        written = ingest.read_canonical(out / f"{role}.h5")
        wanted = data.var_names[data.var["feature_types"] == ingest._TENX_TYPE[role]]
        assert written.n_vars == n_feat and list(written.var_names) == list(wanted)


@pytest.mark.parametrize("types", [("GEX", "ATAC"), ("GEX", "ADT"),
                                   ("protein_coding", "lncRNA")])
def test_no_example_is_guessed(tmp_path, types):
    """Type names that are not the 10x ones: the message does not pick one
    of them, because the larger type need not be the modality asked for."""
    a = _mixed([(types[0], 30, "A"), (types[1], 50, "B")])
    msg = _raises(tmp_path, a, rna="X")
    assert msg == (
        f"rna='X' selects 80 features of 2 types: 50 {types[1]!r} and 30 {types[0]!r} "
        f"(var['feature_types']). One file holds one modality. Add a filter that names the "
        f"type holding the RNA features, as rna='X[feature_types=<type>]'. {_exit('RNA')}")
    assert re.findall(r"feature_types=([^\]]*)\]", msg) == ["<type>"]
    msg = _raises(tmp_path, None, rna=a)
    assert msg.endswith("Pass the features of one type, as rna=adata[:, "
                        "adata.var['feature_types'] == '<type>']. " + _exit("RNA", obj="rna"))


def test_a_10x_name_of_another_modality_is_no_example(tmp_path):
    """Genes and peaks, and adt= asked from X: neither type is the ADT one."""
    msg = _raises(tmp_path, _arc(), rna=None, adt="X")
    assert msg == (f"adt='X' selects {HEAD} Add a filter that names the type holding the ADT "
                   f"features, as adt='X[feature_types=<type>]'. {_exit('ADT')}")


def test_peaks_are_no_example_for_gene_activity(tmp_path):
    """A 10x file has no type for gene-activity scores, so the filter for
    atac_kind='gene_activity' is not filled in with the peaks."""
    msg = _raises(tmp_path, _arc(), rna="X[feature_types=Gene Expression]", atac="X",
                  atac_kind="gene_activity")
    assert msg == (f"atac='X' selects {HEAD} Add a filter that names the type holding the "
                   f"ATAC features, as atac='X[feature_types=<type>]'. {_exit('ATAC')}")
    msg = _raises(tmp_path, _arc(), rna="X[feature_types=Gene Expression]", atac="X",
                  atac_kind="peak")
    assert "for example atac='X[feature_types=Peaks]'." in msg


def test_many_types_are_counted(tmp_path):
    a = _mixed([(f"type{i}", 10 - i, f"T{i}_") for i in range(6)])
    msg = _raises(tmp_path, a, rna="X")
    assert ("45 features of 6 types: 10 'type0', 9 'type1', 8 'type2', 7 'type3' and 2 more "
            "(var['feature_types']).") in msg


# ----------------------------------------------------------- calls that pass
def test_calls_that_pass(tmp_path):
    arc = _arc()
    genes = arc[:, arc.var["feature_types"] == "Gene Expression"]
    peaks = arc[:, arc.var["feature_types"] == "Peaks"]
    kw = dict(atac_kind="peak", labels="obs:celltype", category="vertical")
    # a filter for each modality
    one = ingest.export_dataset(arc, tmp_path / "FILTERS", rna="X[feature_types=Gene Expression]",
                                atac="X[feature_types=Peaks]", **kw)
    # two objects; the values that occur count, not the categories of the column
    genes = genes.copy()
    genes.var["feature_types"] = pd.Categorical(["Gene Expression"] * GENES,
                                                categories=["Gene Expression", "Peaks"])
    two = ingest.export_dataset(genes, tmp_path / "SPLIT", atac=peaks, **kw)
    for d in (one, two):
        assert ingest.read_canonical(d / "rna.h5").n_vars == GENES
        assert ingest.read_canonical(d / "atac.h5").n_vars == PEAKS
    # one value in the column, and no column
    plain = genes.copy()
    ingest.export_dataset(plain, tmp_path / "ONE", rna="X")
    del plain.var["feature_types"]
    ingest.export_dataset(plain, tmp_path / "NONE", rna="X")
    ingest.to_canonical(genes, tmp_path / "rna.h5", modality="rna")
    # an obsm matrix has no var axis, so the column says nothing about it
    arc.obsm["protein"] = pd.DataFrame(np.ones((arc.n_obs, 3)), index=arc.obs_names,
                                       columns=["CD3", "CD4", "CD8"])
    ingest.export_dataset(arc, tmp_path / "OBSM", rna=None, adt="obsm:protein")
    ingest.to_canonical(arc, tmp_path / "adt.h5", modality="adt", obsm="protein")


def test_a_missing_type_is_a_type(tmp_path):
    a = _mixed([("Gene Expression", 30, "GENE")])
    a.var["feature_types"] = ["Gene Expression"] * 29 + [np.nan]
    msg = _raises(tmp_path, a, rna="X")
    assert msg.startswith("rna='X' selects 30 features of 2 types: 29 'Gene Expression' and "
                          "1 'nan' (var['feature_types']).")


# ------------------------------------------------- F3: to_canonical and mtb.run
def _f3(example, kind):
    return (f"The matrix holds {HEAD} Pass the features of one type, for example adata[:, "
            f"adata.var['feature_types'] == {example!r}], or use mtb.io.export_dataset, "
            f"whose selectors take a filter such as 'X[feature_types={example}]'. "
            + _exit(kind))


F3 = _f3("Gene Expression", "RNA")


def test_to_canonical(tmp_path):
    arc = _arc()
    for kw in (dict(modality="rna"), dict(modality="rna", layer="counts")):
        with pytest.raises(ValueError) as e:
            ingest.to_canonical(arc, f"{tmp_path}/DIR/", **kw)
        assert str(e.value) == F3
    assert not (tmp_path / "DIR").exists()
    # the file name says the modality when modality= does not
    with pytest.raises(ValueError) as e:
        ingest.to_canonical(arc, tmp_path / "DIR" / "rna2.h5")
    assert str(e.value) == F3
    with pytest.raises(ValueError) as e:
        ingest.to_canonical(arc, tmp_path / "DIR" / "atac_peak.h5")
    assert str(e.value) == _f3("Peaks", "ATAC")
    # neither: no type is offered
    with pytest.raises(ValueError) as e:
        ingest.to_canonical(arc, tmp_path / "DIR" / "matrix.h5")
    assert str(e.value) == (
        f"The matrix holds {HEAD} Pass the features of one type, as adata[:, "
        f"adata.var['feature_types'] == '<type>'], or use mtb.io.export_dataset, whose "
        f"selectors take a filter such as 'X[feature_types=<type>]'. If all of them are "
        f"features of one modality, delete the column from the object first, for example "
        f"del adata.var['feature_types'].")
    assert not (tmp_path / "DIR").exists()


def test_the_conversion_inside_run(tmp_path, monkeypatch):
    """mtb.run converts an .h5ad input to inputs/<role>.h5: the same check."""
    src = tmp_path / "arc.h5ad"
    _arc().write_h5ad(src)
    adt = ingest.to_canonical(_mixed([("Antibody Capture", 5, "CD")]), tmp_path / "adt.h5")
    (tmp_path / "repo" / "tools_scripts").mkdir(parents=True)
    monkeypatch.setattr(runner, "provide_script_files", lambda *a, **k: None)
    monkeypatch.setattr(runner.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("the method was started"))
    with pytest.raises(ValueError) as e:
        mtb.run("scMDC", "vertical", inputs={"rna": str(src), "adt": str(adt)},
                out_dir=str(tmp_path / "o"), cmd_template="{cmd}",
                repo_path=str(tmp_path / "repo"))
    assert str(e.value) == F3
    assert not (tmp_path / "o" / "inputs" / "rna.h5").exists()


def _cli(argv, capsys):
    import warnings
    capsys.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli.main(argv)
    err = capsys.readouterr().err
    return rc, next((ln for ln in err.splitlines() if ln.startswith("error: ")), err)


def test_convert_one_file_from_the_command_line(tmp_path, capsys):
    src = tmp_path / "arc.h5ad"
    _arc().write_h5ad(src)
    rc, line = _cli(["convert", str(src), f"{tmp_path}/DIR/", "--modality", "rna"], capsys)
    assert rc == 1 and not (tmp_path / "DIR").exists()
    assert line == (f"error: The matrix holds {HEAD} Write the dataset folder instead, for "
                    f"example multibench convert SRC OUT --rna "
                    f'"X[feature_types=Gene Expression]".')
    rc, line = _cli(["convert", str(src), f"{tmp_path}/DIR/", "--modality", "peak"], capsys)
    assert rc == 1 and line.endswith(
        'for example multibench convert SRC OUT --atac "X[feature_types=Peaks]" '
        "--atac-kind peak.")
    # the command the message shows writes the folder
    rc, _ = _cli(["convert", str(src), str(tmp_path / "OUT"), "--atac",
                  "X[feature_types=Peaks]", "--atac-kind", "peak"], capsys)
    assert rc == 0
    assert ingest.read_canonical(tmp_path / "OUT" / "atac_peak.h5").n_vars == PEAKS


def test_selectors_on_the_command_line(tmp_path, capsys):
    """F1 and F2 as the command line spells them: the option, a quoted
    selector, and no Python."""
    src = tmp_path / "arc.h5ad"
    _arc().write_h5ad(src)
    out = tmp_path / "OUT"
    cli_exit = ("If all of them are RNA features, delete the var column feature_types in "
                "Python and save the file again.")
    rc, line = _cli(["convert", str(src), str(out), "--rna", "X"], capsys)
    assert rc == 1 and line == (
        f"error: --rna X selects {HEAD} Add a filter, for example --rna "
        f'"X[feature_types=Gene Expression]". {cli_exit}')
    rc, line = _cli(["convert", str(src), str(out), "--rna", "X[genome=GRCh38]"], capsys)
    assert rc == 1 and line == (
        f"error: --rna X[genome=GRCh38] holds {HEAD} A selector takes one filter. Keep the "
        f"features of one type in the file first. In Python: adata[:, "
        f"adata.var['feature_types'] == 'Gene Expression'].write_h5ad(\"one_type.h5ad\").")
    assert not out.exists()


@pytest.mark.parametrize("selector", [None, "layer:counts"], ids=["default", "layer"])
def test_atac_from_is_checked(tmp_path, capsys, selector):
    """--atac-from reads the ATAC of unpaired data from a second file. Its
    default selector X never reached the matrix check in 0.3.4."""
    arc = _arc()
    arc.write_h5ad(tmp_path / "arc.h5ad")
    rna = arc[:, arc.var["feature_types"] == "Gene Expression"].copy()
    rna.obs_names = [f"other{i}" for i in range(rna.n_obs)]
    rna.write_h5ad(tmp_path / "rna.h5ad")
    out = tmp_path / "OUT"
    argv = ["convert", str(tmp_path / "rna.h5ad"), str(out), "--rna", "X", "--atac-from",
            str(tmp_path / "arc.h5ad"), "--atac-kind", "peak", "--category", "diagonal"]
    spec = selector or "X"
    rc, line = _cli(argv + (["--atac", selector] if selector else []), capsys)
    assert rc == 1 and not out.exists()
    assert line == (
        f"error: --atac {spec} selects {HEAD} Add a filter, for example --atac "
        f'"{spec}[feature_types=Peaks]". If all of them are ATAC features, delete the var '
        f"column feature_types in Python and save the file again.")
    rc, _ = _cli(argv + ["--atac", f"{spec}[feature_types=Peaks]"], capsys)
    assert rc == 0 and ingest.read_canonical(out / "atac_peak.h5").n_vars == PEAKS


# --------------------------------------------------------- the 10x .h5 file
def _tenx_h5(path, n_cells=15):
    """A 10x Genomics feature-barcode matrix as cellranger-arc writes it: a
    sparse matrix (data, indices, indptr, shape) with its features."""
    import scipy.sparse as sp
    arc = _arc()[:n_cells]
    m = sp.csc_matrix(arc.X.T)                       # features x barcodes
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=m.data.astype("int32"))
        g.create_dataset("indices", data=m.indices)
        g.create_dataset("indptr", data=m.indptr)
        g.create_dataset("shape", data=np.array(m.shape, dtype="int32"))
        g.create_dataset("barcodes", data=np.array(arc.obs_names, dtype="S"))
        ft = g.create_group("features")
        ft.create_dataset("id", data=np.array(arc.var_names, dtype="S"))
        ft.create_dataset("name", data=np.array(arc.var_names, dtype="S"))
        ft.create_dataset("feature_type", data=np.array(arc.var["feature_types"], dtype="S"))
        ft.create_dataset("genome", data=np.array(arc.var["genome"], dtype="S"))
    return path


def test_a_10x_file_is_named(tmp_path, capsys):
    src = _tenx_h5(tmp_path / "filtered_feature_bc_matrix.h5")
    with h5py.File(src, "r") as f:
        assert f["matrix/data"].ndim == 1 and "matrix/indptr" in f
    assert ingest._is_canonical_h5(src) is False
    told = (f"{src} is a 10x Genomics file, not a multibench dataset file. Read it with "
            f"scanpy, sc.read_10x_h5(path, gex_only=False), split it by "
            f"var['feature_types'] and write a dataset folder with mtb.io.export_dataset. "
            f"{RECIPE}")
    for call in (lambda: ingest.to_canonical(src, f"{tmp_path}/DIR/", modality="rna"),
                 lambda: ingest.to_canonical(str(src), tmp_path / "DIR" / "rna.h5"),
                 lambda: mtb.run("scMDC", "vertical", inputs={"rna": str(src), "adt": str(src)},
                                 out_dir=str(tmp_path / "o"), dry_run=True),
                 # a real run, with the command given so that no environment is asked for
                 lambda: mtb.run("scMDC", "vertical", inputs={"rna": str(src), "adt": str(src)},
                                 out_dir=str(tmp_path / "o"), cmd_template="{cmd}")):
        with pytest.raises(ValueError) as e:
            call()
        assert str(e.value) == told
    # 0.3.4 passed the file through as its own result, whatever the arguments
    for kw in ({}, {"convert": False}, {"modality": "rna"}):
        with pytest.raises(ValueError) as e:
            ingest.to_canonical(str(src), **kw)
        assert str(e.value) == told
    cli_told = (f"error: {src} is a 10x Genomics file. multibench convert reads .h5ad files, "
                f"not this file. Read it in Python with scanpy, save one .h5ad file per "
                f"modality and convert those. {RECIPE}")
    for argv in (["convert", str(src), f"{tmp_path}/DIR/", "--modality", "rna"],
                 ["convert", str(src), str(tmp_path / "DIR"), "--rna", "X"]):
        rc, line = _cli(argv, capsys)
        assert rc == 1 and line == cli_told
    assert not (tmp_path / "DIR").exists() and not (tmp_path / "o").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["filtered_feature_bc_matrix.h5"]


def test_read_canonical_names_a_10x_file(tmp_path):
    """The reader of dataset files said "not enough values to unpack" for a
    10x file. Whatever it read back before, it reads back."""
    src = _tenx_h5(tmp_path / "filtered_feature_bc_matrix.h5")
    for path in (src, str(src)):
        with pytest.raises(ValueError) as e:
            mtb.io.read_canonical(path)
        assert str(e.value) == ingest._tenx_message(path) == (
            f"{src} is a 10x Genomics file, not a multibench dataset file. Read it with "
            f"scanpy, sc.read_10x_h5(path, gex_only=False), split it by "
            f"var['feature_types'] and write a dataset folder with mtb.io.export_dataset. "
            f"{RECIPE}")
    # a dataset file, with its shape and names
    arc = _arc()
    made = ingest.to_canonical(arc[:, :GENES].copy(), tmp_path / "rna.h5")
    back = mtb.io.read_canonical(made)
    assert back.shape == (arc.n_obs, GENES) and list(back.obs_names) == list(arc.obs_names) \
        and list(back.var_names) == list(arc.var_names[:GENES])
    # a 2-D matrix/data is read as before, whatever lies next to it
    odd = tmp_path / "odd.h5"
    with h5py.File(odd, "w") as f:
        f.create_dataset("matrix/data", data=np.arange(12.0).reshape(3, 4))
        f.create_dataset("matrix/indptr", data=np.arange(3))
    assert ingest._is_tenx_h5(odd) and mtb.io.read_canonical(odd, sparse=False).shape == (4, 3)
    # and a file without matrix/data keeps h5py's KeyError
    bare = tmp_path / "bare.h5"
    with h5py.File(bare, "w") as f:
        f.create_dataset("data", data=np.zeros((3, 4)))
    with pytest.raises(KeyError):
        mtb.io.read_canonical(bare)


def test_other_h5_files_keep_their_message(tmp_path):
    """Not canonical and not 10x: the keys are listed, as before."""
    p = tmp_path / "embedding.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=np.zeros((3, 4)))
    with pytest.raises(ValueError) as e:
        ingest.to_canonical(p, tmp_path / "x.h5")
    assert str(e.value) == (
        f"{p} has no dataset 'matrix/data'; found keys ['data'] - a canonical input .h5 "
        f"holds matrix/data (features x cells), matrix/features and matrix/barcodes (a "
        f"top-level 'data' dataset is a method output such as out/<method>/embedding.h5, "
        f"which mtb.evaluate reads); pass an AnnData / .h5ad / .csv to convert instead")
    # a matrix group whose data is not a dataset is no dataset file either
    q = tmp_path / "group.h5"
    with h5py.File(q, "w") as f:
        f.create_group("matrix/data")
    assert ingest._is_canonical_h5(q) is False
    # nor is one whose matrix/data has one dimension and no indptr next to
    # it. The message says what is missed: matrix/data is there
    r = tmp_path / "flat.h5"
    with h5py.File(r, "w") as f:
        f.create_dataset("matrix/data", data=np.ones(4))
    assert ingest._is_canonical_h5(r) is False
    for path in (q, r):
        with pytest.raises(ValueError) as e:
            ingest.to_canonical(path, tmp_path / "OUT" / "rna.h5", modality="rna")
        assert str(e.value) == (
            f"{path} has no 2-D dataset 'matrix/data'; found keys ['matrix'] - a canonical "
            f"input .h5 holds matrix/data (features x cells), matrix/features and "
            f"matrix/barcodes; pass an AnnData / .h5ad / .csv to convert instead")
    assert not (tmp_path / "OUT").exists()


def _demo_h5_files():
    """The .h5 files of the package's demo datasets that are on disk."""
    from multibench.data.fetch import AVAILABLE
    root = config.DEFAULT.data_path
    return sorted(p for ds in AVAILABLE for p in (root / ds).glob("*.h5"))


def test_canonical_files_still_pass(tmp_path):
    """Every .h5 of the demo datasets on disk is a dataset file as before, and
    so is what to_canonical writes."""
    files = _demo_h5_files()
    if not files:
        pytest.skip("no demo dataset on disk")
    not_canonical = [str(p) for p in files if not ingest._is_canonical_h5(p)]
    assert not not_canonical
    for p in files[:3]:
        assert ingest.to_canonical(p, tmp_path / "copy.h5") == p         # passed through
    assert not (tmp_path / "copy.h5").exists()
    made = ingest.to_canonical(_mixed([("Gene Expression", 5, "G")]), tmp_path / "rna.h5")
    assert ingest._is_canonical_h5(made) and ingest.to_canonical(made) == made
