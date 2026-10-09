"""mtb.data.write_example: the example files of the tutorials' own-data section.

The section starts from ``.h5ad`` files, as a reader's own data does. So
that it runs before the reader has put a file in place, ``write_example``
writes a part of the tutorial's dataset under the names the section reads.
The objects in the files are the ones the 0.3.4 tutorials built in a code
cell (``tests/_own_cells_034.py``), so a tutorial exports the folder it
exported before.

A file of the reader with one of those names is never replaced.
"""
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench.data import examples
from multibench.engine import ingest

from _own_cells_034 import OLD

ad = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

ROOT = Path(__file__).resolve().parent.parent
MARK = "multibench_example"

#: dataset -> task of the 0.3.4 tutorials
TASK = {re.search(r'data_path / "(\w+)"', cell["data"]).group(1): key
        for key, cell in OLD.items()}
#: dataset -> [(file, the variable of the 0.3.4 data cell that it holds,
#: index into the cell's ``labels`` list)]: the file table of the spec
HOLDS = {
    "D11": [("my_file.h5ad", "adata", None)],
    "D52mini": [("my_file.h5ad", "adata", None)],
    "D22mini": [("my_file.h5ad", "adata", None), ("my_atac.h5ad", "atac", None)],
    "D59mini": [("my_file.h5ad", "adata", None), ("my_atac.h5ad", "atac", None)],
    "D27mini_vertical": [("my_rna.h5ad", "rna", None), ("my_atac.h5ad", "atac", None)],
    "D56mini": [("my_rna.h5ad", "rna", None), ("my_atac.h5ad", "atac", None)],
    "D27mini": [("my_rna.h5ad", "rna", None), ("my_atac.h5ad", "atac", None)],
    "D58mini": [("my_adt.h5ad", "adt", None), ("my_atac.h5ad", "atac", None)],
    "D37mini": [(f"my_rna{b}.h5ad", f"rna[{b - 1}]", None) for b in (1, 2, 3)]
               + [(f"my_atac{b}.h5ad", f"atac[{b - 1}]", None) for b in (1, 2, 3)],
    "D38mini": [("my_rna1.h5ad", "rna1", 0), ("my_rna2.h5ad", "rna2", 1),
                ("my_adt2.h5ad", "adt2", 1), ("my_adt3.h5ad", "adt3", 2)],
    "D45mini": [("my_rna1.h5ad", "rna1", 0), ("my_rna2.h5ad", "rna2", 1),
                ("my_atac2.h5ad", "atac2", 1), ("my_atac3.h5ad", "atac3", 2)],
    "D46mini": [("my_rna1.h5ad", "rna[0]", 0), ("my_adt1.h5ad", "adt1", 0),
                ("my_rna2.h5ad", "rna[1]", 1), ("my_atac2.h5ad", "atac2", 1),
                ("my_rna3.h5ad", "rna[2]", 2)],
    "D49mini": [("my_rna1.h5ad", "rna1", 0), ("my_adt1.h5ad", "adt1", 0),
                ("my_rna2.h5ad", "rna2", 1), ("my_atac2.h5ad", "atac2", 1),
                ("my_adt3.h5ad", "adt3", 2)],
}
CROSS = ("D52mini", "D56mini", "D58mini", "D59mini")


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _mark(path):
    with h5py.File(path, "r") as f:
        return f.attrs.get(MARK)


def test_the_table_of_examples():
    assert set(examples.EXAMPLES) == set(HOLDS) == set(TASK) and len(TASK) == 13
    for ds, files in examples.EXAMPLES.items():
        assert isinstance(files, tuple) and list(files) == [f for f, _, _ in HOLDS[ds]]
        assert all(re.fullmatch(r"my_[a-z]+\d?\.h5ad", f) for f in files)
    assert "write_example" in mtb.data.__all__ and "EXAMPLES" not in mtb.data.__all__
    assert mtb.data.write_example is examples.write_example
    assert examples.__all__ == ["write_example"]


@pytest.mark.parametrize("ds", sorted(HOLDS))
def test_files_and_contents(ds, tmp_path, monkeypatch):
    """Each file holds the object of the 0.3.4 data cell: the same matrix,
    cells, features, cell types, batches and proteins."""
    if not (mtb.config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")
    monkeypatch.chdir(tmp_path)
    ns = {"mtb": mtb, "pd": pd}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(OLD[TASK[ds]]["data"], f"0.3.4 {TASK[ds]}", "exec"), ns)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert mtb.data.write_example(ds, quiet=True) is None
    # the cell of the tutorial shows the reader no warning of the package or of anndata
    ours = [str(w.message) for w in seen if "multibench" in (w.filename or "")
            or issubclass(w.category, UserWarning)]
    assert not ours, ours
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(examples.EXAMPLES[ds])
    main = eval(HOLDS[ds][0][1], ns)
    for name, expr, label in HOLDS[ds]:
        old, new = eval(expr, ns), ad.read_h5ad(name)
        assert _mark(name) == f"{ds}@{mtb.__version__}"
        assert list(new.obs_names) == list(old.obs_names), name
        assert list(new.var_names) == list(old.var_names), name
        assert np.array_equal(_dense(new.X), _dense(old.X)), name
        if label is not None:                       # mosaic: the labels were a list of arrays
            celltype = ns["labels"][label]
        elif "celltype" in old.obs:
            celltype = old.obs["celltype"]
        else:                                       # the second file of the same cells
            assert list(old.obs_names) == list(main.obs_names)
            celltype = main.obs["celltype"]
        assert [str(x) for x in new.obs["celltype"]] == [str(x) for x in celltype], name
        assert ("batch" in new.obs) == (ds in CROSS), name
        if ds in CROSS:
            assert [str(x) for x in new.obs["batch"]] == [str(x) for x in main.obs["batch"]]
        assert "donor" not in new.obs
        if "protein" in old.obsm:
            df = old.obsm["protein"]
            assert isinstance(new.obsm["protein"], np.ndarray)
            assert np.array_equal(new.obsm["protein"], df.to_numpy())
            assert list(new.uns["protein_names"]) == list(df.columns)
        else:
            assert not len(new.obsm) and "protein_names" not in new.uns


# ----------------------------------------------- a small dataset in a data root
def _canonical(path, n_cells, feats, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, (n_cells, len(feats))).astype(float))
    a.var_names, a.obs_names = feats, [f"cell{i}-1" for i in range(n_cells)]
    ingest.to_canonical(a, path)


@pytest.fixture
def root(tmp_path):
    """A data root with small D11, D52mini, D27mini and D58mini folders."""
    r = tmp_path / "data"
    genes, prots = [f"GENE{i}" for i in range(12)], ["CD3", "CD4", "CD8"]
    peaks = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(9)]
    _canonical(r / "D11" / "rna.h5", 40, genes)
    _canonical(r / "D11" / "adt.h5", 40, prots, seed=1)
    ingest._write_labels(["T", "B"] * 20, r / "D11" / "cty.csv")
    for b in ("1", "2", "3"):
        _canonical(r / "D52mini" / f"rna{b}.h5", 20, genes, seed=int(b))
        _canonical(r / "D52mini" / f"adt{b}.h5", 20, prots, seed=10 + int(b))
        ingest._write_labels(["T", "B"] * 10, r / "D52mini" / f"cty{b}.csv")
    for b in ("1", "2"):
        _canonical(r / "D58mini" / f"adt{b}.h5", 20, prots, seed=int(b))
        _canonical(r / "D58mini" / f"atac{b}.h5", 20, peaks, seed=20 + int(b))
        ingest._write_labels(["T", "B"] * 10, r / "D58mini" / f"cty{b}.csv")
    _canonical(r / "D27mini" / "rna.h5", 30, genes)
    _canonical(r / "D27mini" / "atac_gas.h5", 20, genes, seed=3)
    ingest._write_labels(["T", "B", "NK"] * 10, r / "D27mini" / "rna_cty.csv")
    ingest._write_labels(["T", "B"] * 10, r / "D27mini" / "atac_cty.csv")
    return r


def _bytes(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_printed_lines(root, tmp_path, capsys, monkeypatch):
    here = tmp_path / "work"
    here.mkdir()
    monkeypatch.chdir(here)
    mtb.data.write_example("D11", data_path=root)
    assert capsys.readouterr().out == (
        'wrote my_file.h5ad: 24 cells. RNA counts in X, protein counts in obsm["protein"], '
        'cell types in obs["celltype"].\n')
    mtb.data.write_example("D58mini", "files", data_path=root)
    assert capsys.readouterr().out == (
        'wrote files/my_adt.h5ad: 24 cells. protein counts in X, cell types in '
        'obs["celltype"], batches in obs["batch"].\n'
        'wrote files/my_atac.h5ad: 24 cells. ATAC peak counts in X, cell types in '
        'obs["celltype"], batches in obs["batch"].\n')
    mtb.data.write_example("D27mini", "unpaired", data_path=root)
    assert capsys.readouterr().out == (
        'wrote unpaired/my_rna.h5ad: 27 cells. RNA counts in X, cell types in '
        'obs["celltype"].\n'
        'wrote unpaired/my_atac.h5ad: 18 cells. ATAC gene-activity scores in X, cell types '
        'in obs["celltype"].\n')
    mtb.data.write_example("D52mini", "quiet", data_path=root, quiet=True)
    mtb.data.write_example("D52mini", "quiet", data_path=root, quiet=True)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert (here / "quiet" / "my_file.h5ad").is_file()


def test_the_first_line_of_the_first_tutorial(tmp_path, capsys, monkeypatch):
    if not (mtb.config.DEFAULT.data_path / "D11").is_dir():
        pytest.skip("D11 is not on disk")
    monkeypatch.chdir(tmp_path)
    mtb.data.write_example("D11")
    assert capsys.readouterr().out == (
        'wrote my_file.h5ad: 1,718 cells. RNA counts in X, protein counts in '
        'obsm["protein"], cell types in obs["celltype"].\n')


def test_a_file_of_the_reader_is_kept(root, tmp_path, capsys):
    """A file with the example's name that multibench did not write: an
    .h5ad of the reader, or anything else."""
    folder = tmp_path / "mine"
    folder.mkdir()
    own = ad.AnnData(np.ones((3, 2)))
    own.write_h5ad(folder / "my_file.h5ad")
    (folder / "my_atac.h5ad").write_text("not an HDF5 file")
    (folder / "my_adt.h5ad").mkdir()
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    mtb.data.write_example("D11", folder, data_path=root)
    mtb.data.write_example("D58mini", folder, data_path=root)
    assert capsys.readouterr().out == "".join(
        f"{folder / name} is already here and is not an example file of multibench. It was "
        f"left as it is.\n" for name in ("my_file.h5ad", "my_adt.h5ad", "my_atac.h5ad"))
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == before
    assert (folder / "my_adt.h5ad").is_dir() and len(list(folder.iterdir())) == 3


def test_a_saved_copy_has_no_marker(root, tmp_path, capsys):
    """An example that the reader read and saved again is the reader's file:
    write_h5ad does not carry the marker over."""
    folder = tmp_path / "f"
    mtb.data.write_example("D11", folder, data_path=root)
    target = folder / "my_file.h5ad"
    assert _mark(target) == f"D11@{mtb.__version__}"
    mine = ad.read_h5ad(target)
    mine.obs["celltype"] = "edited"
    mine.write_h5ad(target)
    assert _mark(target) is None
    before = target.read_bytes()
    capsys.readouterr()
    for ds in ("D11", "D52mini"):
        mtb.data.write_example(ds, folder, data_path=root)
        assert capsys.readouterr().out == (
            f"{target} is already here and is not an example file of multibench. It was "
            f"left as it is.\n")
    assert target.read_bytes() == before
    assert set(ad.read_h5ad(target).obs["celltype"]) == {"edited"}


def test_the_same_example_is_kept(root, tmp_path, capsys):
    """The second call writes nothing. Nor does a call after the file was
    changed in place, which keeps the marker."""
    folder = tmp_path / "f"
    mtb.data.write_example("D11", folder, data_path=root)
    target = folder / "my_file.h5ad"
    before = target.read_bytes()
    capsys.readouterr()
    kept = f"{target} is already here (the example of D11). Delete the file to write it again.\n"
    mtb.data.write_example("D11", folder, data_path=root)
    assert capsys.readouterr().out == kept and target.read_bytes() == before
    backed = ad.read_h5ad(target, backed="r+")
    backed.obs["mine"] = 1
    backed.write()
    backed.file.close()
    changed = target.read_bytes()
    assert changed != before and _mark(target) == f"D11@{mtb.__version__}"
    mtb.data.write_example("D11", folder, data_path=root)
    assert capsys.readouterr().out == kept and target.read_bytes() == changed
    # deleting it, as the line says, writes it again
    target.unlink()
    mtb.data.write_example("D11", folder, data_path=root)
    assert capsys.readouterr().out.startswith(f"wrote {target}: 24 cells.")
    assert "mine" not in ad.read_h5ad(target).obs


def test_another_example_is_replaced(root, tmp_path, capsys):
    """Two tutorials use the name my_file.h5ad. Run in one folder, the second
    must not read the example of the first; nor must a newer multibench read
    the example an older one wrote."""
    folder = tmp_path / "f"
    target = folder / "my_file.h5ad"
    mtb.data.write_example("D11", folder, data_path=root, quiet=True)
    assert "batch" not in ad.read_h5ad(target).obs
    mtb.data.write_example("D52mini", folder, data_path=root)
    assert capsys.readouterr().out.startswith(f"wrote {target}: 36 cells.")
    assert _mark(target) == f"D52mini@{mtb.__version__}"
    assert "batch" in ad.read_h5ad(target).obs
    # the example of another version
    with h5py.File(target, "a") as f:
        f.attrs[MARK] = "D52mini@0.0.1"
    old = target.read_bytes()
    mtb.data.write_example("D52mini", folder, data_path=root)
    assert capsys.readouterr().out.startswith(f"wrote {target}: 36 cells.")
    assert target.read_bytes() != old and _mark(target) == f"D52mini@{mtb.__version__}"
    assert sorted(p.name for p in folder.iterdir()) == ["my_file.h5ad"]


def test_unknown_dataset_and_missing_folder(root, tmp_path):
    folder = tmp_path / "f"
    with pytest.raises(ValueError) as e:
        mtb.data.write_example("D99", folder, data_path=root)
    assert str(e.value) == (
        "No example is defined for 'D99'. Examples exist for D11, D22mini, D27mini, "
        "D27mini_vertical, D37mini, D38mini, D45mini, D46mini, D49mini, D52mini, D56mini, "
        "D58mini, D59mini.")
    with pytest.raises(ValueError, match="No example is defined for 'D28'"):
        mtb.data.write_example("D28", folder, data_path=root)        # a dataset, but no example
    with pytest.raises(FileNotFoundError) as e:
        mtb.data.write_example("D22mini", folder, data_path=root)
    assert str(e.value) == (f"{root / 'D22mini'} does not exist. Run "
                            f"mtb.data.fetch('D22mini') first.")
    assert not folder.exists()                       # no folder for a call that wrote nothing


def test_the_data_root_is_read_at_call_time(root, tmp_path, monkeypatch):
    folder = tmp_path / "f"
    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError) as e:
        mtb.data.write_example("D11", folder)
    assert str(e.value) == (f"{tmp_path / 'nowhere' / 'D11'} does not exist. Run "
                            f"mtb.data.fetch('D11') first.")
    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", root)
    mtb.data.write_example("D11", folder, quiet=True)
    assert ad.read_h5ad(folder / "my_file.h5ad").n_obs == 24
    # data_path= wins over the configured root
    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", tmp_path / "nowhere")
    mtb.data.write_example("D27mini", folder, data_path=str(root), quiet=True)
    assert (folder / "my_rna.h5ad").is_file()


def test_a_failed_write_leaves_no_file(root, tmp_path, monkeypatch):
    """A file appears under its name only when it is complete."""
    folder = tmp_path / "f"
    real = ad.AnnData.write_h5ad

    def half_written(self, path, *a, **kw):
        real(self, path, *a, **kw)
        if "my_atac" in str(path) or self.n_vars == 3:       # the second file of D58mini
            raise OSError("disk full")
    monkeypatch.setattr(ad.AnnData, "write_h5ad", half_written)
    with pytest.raises(OSError, match="disk full"):
        mtb.data.write_example("D58mini", folder, data_path=root, quiet=True)
    assert list(folder.iterdir()) == []              # neither the file nor a temporary one
    monkeypatch.setattr(ad.AnnData, "write_h5ad", real)
    # an example that was to be replaced is still whole
    mtb.data.write_example("D11", folder, data_path=root, quiet=True)
    before = _bytes(folder)

    def fails(self, path, *a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(ad.AnnData, "write_h5ad", fails)
    with pytest.raises(OSError, match="disk full"):
        mtb.data.write_example("D52mini", folder, data_path=root, quiet=True)
    assert _bytes(folder) == before


def test_the_files_are_what_the_export_reads(root, tmp_path):
    """Read back, the example goes through export_dataset as the tutorial
    writes it: the protein names come from uns, with no warning."""
    folder = tmp_path / "f"
    mtb.data.write_example("D52mini", folder, data_path=root, quiet=True)
    adata = ad.read_h5ad(folder / "my_file.h5ad")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = mtb.io.export_dataset(adata, tmp_path / "mydata" / "MYCROSS", rna="X",
                                    adt="obsm:protein", labels="obs:celltype",
                                    batch="obs:batch", category="cross")
    assert sorted(p.name for p in out.iterdir()) == [
        f"{stem}{b}{ext}" for stem, ext in (("adt", ".h5"), ("cty", ".csv"), ("rna", ".h5"))
        for b in (1, 2, 3)]
    assert list(mtb.io.read_canonical(out / "adt1.h5").var_names) == ["CD3", "CD4", "CD8"]


def test_the_docstring():
    import inspect
    doc = inspect.getdoc(mtb.data.write_example)
    assert doc.splitlines()[0] == "Write the example files of a tutorial dataset as .h5ad files."
    for section in ("Parameters", "Returns", "Raises", "Examples", "Notes", "See Also"):
        assert f"\n{section}\n{'-' * len(section)}\n" in doc
    for word in ("stand-in", "stand in", "tidy"):
        assert word not in doc.lower()
    assert not re.search(r"<[a-z_]+>", re.sub(r"``.*?``", "", doc, flags=re.S))
    assert "Save your own data under another name" in doc
    for case in ("it is the example of this dataset, written by this version",
                 "it is a file of your own with that name",
                 "saved again with ``write_h5ad``",
                 "replaced only when it is an example of another dataset or of another"):
        assert case in " ".join(doc.split()), case


def test_import_multibench_stays_light():
    """examples.py is loaded with the package. It must not load anndata and
    scanpy with it: that would triple the time of ``import multibench``."""
    code = ("import sys, multibench, multibench.data.examples\n"
            "print('scanpy' in sys.modules, 'anndata' in sys.modules)")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=ROOT, check=True).stdout
    assert out.split() == ["False", "False"]
    src = (ROOT / "multibench" / "data" / "examples.py").read_text()
    top = [ln for ln in src.splitlines() if re.match(r"(import|from) ", ln)]
    assert top == ["from __future__ import annotations", "import os", "from pathlib import Path"]
