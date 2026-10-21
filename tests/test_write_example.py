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
from multibench._data import examples
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
    if not (mtb.config.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")
    monkeypatch.chdir(tmp_path)
    ns = {"mtb": mtb, "pd": pd}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(OLD[TASK[ds]]["data"], f"0.3.4 {TASK[ds]}", "exec"), ns)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert mtb.data.write_example(ds, verbose=False) is None    # c3 (E15): no repr in a cell
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
    ingest._to_canonical(a, path)


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


def test_the_first_line_of_the_first_tutorial(tmp_path, capsys, monkeypatch):
    if not (mtb.config.data_path / "D11").is_dir():
        pytest.skip("D11 is not on disk")
    monkeypatch.chdir(tmp_path)
    mtb.data.write_example("D11")
    assert capsys.readouterr().out == (
        'wrote my_file.h5ad: 1,718 cells. RNA counts in X, protein counts in '
        'obsm["protein"], cell types in obs["celltype"].\n')


def test_the_docstring():
    import inspect
    doc = inspect.getdoc(mtb.data.write_example)
    assert doc.splitlines()[0] == "Write the example files of a tutorial dataset as .h5ad files."
    for section in ("Parameters", "Raises", "Examples", "See Also"):
        assert f"\n{section}\n{'-' * len(section)}\n" in doc
    assert "\nReturns\n" not in doc                        # c3 (E15): None
    for word in ("stand-in", "stand in", "tidy"):
        assert word not in doc.lower()
    assert not re.search(r"<[a-z_]+>", re.sub(r"``.*?``", "", doc, flags=re.S))
    assert "A file of your own with such a name is kept." in " ".join(doc.split())


def test_import_multibench_stays_light():
    """examples.py is loaded with the package. It must not load anndata and
    scanpy with it: that would triple the time of ``import multibench``."""
    code = ("import sys, multibench, multibench._data.examples\n"
            "print('scanpy' in sys.modules, 'anndata' in sys.modules)")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=ROOT, check=True).stdout
    assert out.split() == ["False", "False"]
    src = (ROOT / "multibench" / "_data" / "examples.py").read_text()
    top = [ln for ln in src.splitlines() if re.match(r"(import|from) ", ln)]
    # 0.4.0: the deprecation helper and DATA_KEYWORDS come from the package itself
    # (light: the subprocess above shows that neither loads anndata or scanpy)
    assert [ln for ln in top if not ln.startswith("from .")] == [
        "from __future__ import annotations", "import os", "from pathlib import Path"]


def test_write_example_returns_nothing(tmp_path):
    # c3 (E15): the printed line names the file; the tutorial cells show no PosixPath('.')
    import multibench as mtb
    if not (mtb.config.data_path / "D11").is_dir():
        pytest.skip("D11 is not in the data root")
    assert mtb.data.write_example('D11', tmp_path / 'ex', verbose=False) is None
    assert (tmp_path / "ex" / "my_file.h5ad").is_file()
