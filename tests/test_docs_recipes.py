"""The recipes for your own data on the docs page tutorials/run (0.3.5).

The section "Your own data" of a task tutorial starts from ``.h5ad`` files
that hold the cell types in ``obs``. The page has four recipes for other
starting points: a 10x ``.h5`` file, labels in a CSV file, no labels, and
another number of donors. A reader runs a recipe in the tutorial notebook of
their task, so a recipe sets ``NAME`` and the check cell and the run cell of
the tutorial then work on the folder of the recipe.

These tests read the page and run its code fences in page order, on
synthetic files, with the names the page uses (``annotation.csv``,
``mydata/``, ``celltype``). A value that the page quotes next to a fence (a
status, a file name, a line of the check, the name of an error, a feature
type) is read from the page and compared with the run, so an edit of the
page fails here as a change of the package does. A sentence that says what a
call does, without a value to compare, is looked up on the page next to the
run that shows it. No method runs: the environments are pinned as present,
and ``_run`` writes a random embedding.

Needs the docs source (SCMULTIBENCH_DOCS=<docs dir>); skipped without it.
"""
import ast
import json
import re
import shlex
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry, resolve, schema
from multibench.eval import io as eio

from tests.test_docs_r4 import _read, _visible, needs_docs
from tests.test_tutorial_own_data import (CHECK, EXAMPLE, EXPORT, GEN, NAMES, READ, RUN,
                                          VERTICAL, _bytes, _content, _namespace, _on_disk,
                                          _run_cell, _sources)

ad = pytest.importorskip("anndata")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("scanpy")

pytestmark = needs_docs

ROOT = Path(__file__).resolve().parent.parent
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())

#: the anchors of the section and of its four recipes
ANCHORS = ["own-data-recipes", "recipe-10x-h5", "recipe-labels-csv", "recipe-no-labels",
           "recipe-donors"]
#: the python fences of each recipe, by their first line. A fence that is
#: added to the page has no test until it is listed here.
FENCES = {
    "recipe-10x-h5": ["import scanpy as sc", "import pandas as pd",
                      'ann.index = ann.index + "-1"',
                      'adata = adata[adata.obs["celltype"].notna()].copy()',
                      'kind = adata.var["feature_types"]', 'kind = adata.var["feature_types"]',
                      'mtb.scan("mydata/" + NAME, "vertical", methods=METHODS)'],
    "recipe-labels-csv": ["import pandas as pd",
                          'ann.index = ann.index + "-1"',
                          'adata = adata[adata.obs["celltype"].notna()].copy()'],
    "recipe-no-labels": ['NAME = "MYDATA"   # <- a name for your data',
                         'mine = mtb.run_all("mydata/" + NAME, "vertical", '
                         'methods=METHODS,',
                         'mtb.io.export_dataset(adata, "mydata/" + NAME, rna=None, '
                         'labels="obs:celltype",'],
    "recipe-donors": ['NAME = "MYDONORS"   # <- a name for your data',
                      'print(adata.obs["donor"].value_counts())   # the names in your file',
                      'adata = adata[adata.obs["donor"].isin(["A", "B", "C"])].copy()'
                      '   # <- three donors'],
}
#: the names in the list of the block that keeps three donors, as the page
#: writes them: they stand for the names of the reader's file
PLACEHOLDERS = '["A", "B", "C"]'
#: donor names of a file, which are not those of the page
DONORS = ["P01", "P02", "P03", "P04", "P05"]
# the fences of the 10x recipe, in page order
READ_10X, LABELS, ENDING, KEEP, MULTIOME, CITE, PAGE_CHECK = range(7)

GENES = [f"GENE{i}" for i in range(120)]
GENES[1] = GENES[0]                       # 10x files repeat gene names
PEAKS = [f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(60)]
ANTIBODIES = [f"CD{i}" for i in range(20)]
ANTIBODIES[3], ANTIBODIES[7] = "GENE5", "GENE9"     # two antibodies named as a gene
#: the type that Cell Ranger gives the features of each modality
TENX_TYPES = {"rna": "Gene Expression", "adt": "Antibody Capture", "atac": "Peaks"}

ZERO_CELLS = (
    "The data holds 0 cells, so there is nothing to write. Check the filter that selects the "
    "cells.")
#: where the line that adds the ending goes, as both recipes say it
ENDING_SENTENCE = ("When the barcodes of the CSV lack the `-1` ending that the cell names "
                   "have, put the next line into the block above, after the line that reads "
                   "the CSV, and run the block again.")


# ------------------------------------------------------------------ the page
def _section():
    """The recipes on the page: from their heading to the rule that ends them."""
    return _read("tutorials/run.md").split("### Recipes for your own data", 1)[1] \
        .split("\n---\n", 1)[0]


def _recipe(anchor):
    """The text of one recipe: from its heading to the next recipe."""
    parts = re.split(r"^#### .*? \{ #([a-z0-9-]+) \}$", _section(), flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[anchor]


def _fences(anchor):
    """The python fences of one recipe, in page order."""
    return re.findall(r"^```python\n(.*?)^```", _recipe(anchor), flags=re.S | re.M)


def _said(anchor):
    """One recipe on one line, to look a sentence up in."""
    return " ".join(_recipe(anchor).split())


def _says(anchor, sentence):
    """The recipe has ``sentence``, word for word."""
    assert sentence in _said(anchor), sentence


def _quoted(anchor, pattern):
    """What the recipe quotes in its one sentence that matches ``pattern``:
    the group, or the groups as a tuple. A sentence that is reworded on the
    page, or that is there twice, fails here."""
    found = re.findall(pattern, _said(anchor))
    assert len(found) == 1, (anchor, pattern, found)
    return found[0]


def _feature_types():
    """``{modality: feature type}`` as the recipe names them for the two
    kinds of 10x file: the names that Cell Ranger writes, and that the
    package knows such a file by."""
    rna, atac, rna_again, adt = _quoted(
        "recipe-10x-h5", r"A 10x Multiome file holds `([^`]+)` and `([^`]+)`, and a CITE-seq "
                         r"file holds `([^`]+)` and `([^`]+)`\.")
    assert rna == rna_again
    types = {"rna": rna, "adt": adt, "atac": atac}
    assert types == TENX_TYPES == ingest._TENX_TYPE
    return types


def _scanpy_warning():
    """The warning of scanpy that the recipe names, as scanpy words it."""
    what = _quoted("recipe-10x-h5", r"scanpy warns that ([a-z ]+) are not unique, here and in "
                                    r"the next steps\.")
    return f"{what.capitalize()} are not unique"


# ------------------------------------------------------------- synthetic data
def _tenx(path, features, other, modality, n_cells=300):
    """A Cell Ranger ``filtered_feature_bc_matrix.h5`` as scanpy reads it:
    features x barcodes, sparse, with a type per feature. The genes and the
    features of ``modality`` (``"atac"`` or ``"adt"``) get the types that the
    recipe names. Returns the barcodes."""
    rng = np.random.default_rng(0)
    names = list(features) + list(other)
    kinds = _feature_types()
    types = [kinds["rna"]] * len(features) + [kinds[modality]] * len(other)
    m = sp.csc_matrix(rng.poisson(0.6, (len(names), n_cells)).astype(np.int32))
    bars = ["".join(rng.choice(list("ACGT"), 16)) + "-1" for _ in range(n_cells)]
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("barcodes", data=np.array(bars, dtype="S"))
        for key in ("data", "indices", "indptr"):
            g.create_dataset(key, data=getattr(m, key))
        g.create_dataset("shape", data=np.array(m.shape, dtype=np.int32))
        ft = g.create_group("features")
        ft.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))
        ft.create_dataset("feature_type", data=np.array(types, dtype="S"))
        ft.create_dataset("genome", data=np.array(["GRCh38"] * len(names), dtype="S"))
        ft.create_dataset("id", data=np.array([f"ID{i}" for i in range(len(names))], dtype="S"))
        ft.create_dataset("name", data=np.array(names, dtype="S"))
    return bars


def _annotation(path, barcodes, labels=None, keep=None):
    """The CSV of the page: barcodes first, the cell types in
    ``cluster_annotation``, the rows in another order than the cells. ``keep``
    leaves that many rows in. Returns ``{barcode: label}`` of the rows written."""
    if labels is None:
        labels = np.random.default_rng(1).choice(["T", "B", "NK"], len(barcodes))
    ann = pd.DataFrame({"cluster_annotation": labels}, index=list(barcodes))
    ann = ann.sample(frac=1, random_state=0).iloc[:keep]
    ann.to_csv(path)
    return ann["cluster_annotation"].to_dict()


def _cells(n=300, genes=40, seed=0):
    """An AnnData of RNA counts whose cell names end in ``-1``."""
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, (n, genes)).astype(float))
    a.obs_names = [f"C{i:04d}-1" for i in range(n)]
    a.var_names = [f"GENE{i}" for i in range(genes)]
    return a


def _cite(n=120, donors=None):
    """The object of the vertical RNA + ADT tutorial: RNA in ``X``, proteins
    in ``obsm["protein"]``, cell types in ``obs["celltype"]``; with
    ``donors``, a donor per cell."""
    rng = np.random.default_rng(0)
    a = _cells(n)
    a.obs["celltype"] = rng.choice(["T", "B", "NK"], n)
    if donors:
        a.obs["donor"] = np.repeat(list(donors), n // len(donors))
    a.obsm["protein"] = pd.DataFrame(rng.poisson(3.0, (n, 8)).astype(float),
                                     index=a.obs_names, columns=[f"p{i}" for i in range(8)])
    return a


def _h5(path, key):
    with h5py.File(path, "r") as f:
        v = f[f"matrix/{key}"]
        return v.shape if key == "data" else [x.decode() for x in v[()]]


def _with_ending(add, ending):
    """The block that adds the column, with the line that adds the ending
    where the recipes say to put it: after the line that reads the CSV."""
    lines = add.splitlines()
    (read,) = [i for i, line in enumerate(lines) if line.startswith("ann = pd.read_csv(")]
    return "\n".join(lines[:read + 1] + [ending.strip()] + lines[read + 1:]) + "\n"


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _run_recipe_cell(src, ns):
    """Run a fence that works on the 10x object. scanpy warns that its
    variable names are not unique, as the page says; a warning of the
    package's own class with another text would be new."""
    named = _scanpy_warning()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        out = _run_cell(src, ns)
    told = [str(w.message) for w in seen if issubclass(w.category, UserWarning)]
    assert [t for t in told if named not in t] == []
    return out, len(told)


class _Res:
    def __init__(self, out):
        self.output = out


@pytest.fixture
def host(monkeypatch, tmp_path):
    """A Linux host with every environment and a GPU, no method scripts. The
    fake ``_run`` writes ``embedding.h5`` with one row per cell of the
    matrices it is given, and records the method and its files. The files of
    one batch hold the same cells. Two batches hold different cells, and so
    do the two sides of a diagonal folder, also where they share barcodes."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    calls = []

    def fake_run(method, category, inputs, out_dir, params=None):
        files = {role: Path(str(f)) for role, f in inputs.items()}
        calls.append((method, files))
        cells = {}
        for f in files.values():
            if f.suffix == ".h5":
                side = category == "diagonal" and f.stem.startswith("rna")
                batch = re.search(r"\d*$", f.stem).group()
                cells.setdefault((side, batch), set()).update(resolve._barcodes_of(f))
        n = sum(map(len, cells.values()))
        emb = np.random.default_rng(0).normal(size=(n, 4))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(Path(out_dir) / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        return _Res(emb)
    monkeypatch.setattr(W, "_run", fake_run)
    return calls


# ------------------------------------------------------------- the section
def test_the_anchors_exist_once(tmp_path):
    """The tutorials link the recipes by anchor, and the error for a 10x file
    names its recipe by title. Each anchor is set once, in the heading of its
    recipe, and every link to a recipe has one."""
    page = _read("tutorials/run.md")
    for anchor in ANCHORS:
        assert page.count(f"{{ #{anchor} }}") == 1, anchor
    assert re.findall(r"^#{3,4} .*? \{ #([a-z0-9-]+) \}$", page, flags=re.M) == ANCHORS
    assert set(re.findall(r"\(#(recipe-[a-z0-9-]+)\)", page)) <= set(ANCHORS)
    # the section sits in "Inputs", after the ATAC details and before the rule
    before, after = page.split("### Recipes for your own data { #own-data-recipes }")
    assert before.rstrip().endswith("export the ATAC form it reads.")
    assert after.split("\n---\n", 1)[1].lstrip().startswith("## GPU or CPU?")
    linked = set()
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        text = "\n".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"])
        linked |= set(re.findall(r"#(recipe-[a-z0-9-]+)", text))
    assert linked == set(ANCHORS[1:])
    # so does the Changes page
    listed = re.findall(r"tutorials/run\.md#((?:own-data|recipe)[a-z0-9-]*)", _read("changes.md"))
    assert set(listed) == set(ANCHORS)
    # the title that the error for a 10x file gives
    assert re.search(r"^#### A 10x `\.h5` file \{ #recipe-10x-h5 \}$", page, flags=re.M)
    src = tmp_path / "filtered_feature_bc_matrix.h5"
    _tenx(src, GENES, PEAKS, "atac", n_cells=20)
    with pytest.raises(ValueError) as e:
        mtb.io.read_canonical(src)
    assert str(e.value) == f'{src} is a 10x Genomics file. See the recipe "A 10x .h5 file" in the docs.'
    assert not (tmp_path / "out").exists()


def test_a_10x_file_is_not_read_as_a_dataset_file(tmp_path, monkeypatch, capsys):
    """The recipe opens by saying that the package does not read the file,
    and the command-line block says so for ``multibench export`` (0.4.0: was
    convert). A run and an export that are given the file stop and write
    nothing."""
    from multibench.engine import runner
    _says("recipe-10x-h5", "`scan`, `run_all` and `multibench export` do not read this file. "
                           "Read it with scanpy first.")
    _says("recipe-donors", "`multibench export` reads `.h5ad` files, not a 10x `.h5` file.")
    monkeypatch.chdir(tmp_path)
    src = "filtered_feature_bc_matrix.h5"
    _tenx(src, GENES, PEAKS, "atac", n_cells=20)
    # the call that run_all makes for each method
    with pytest.raises(ValueError, match=rf'{re.escape(src)} is a 10x Genomics file\. See the '
                                         r'recipe "A 10x \.h5 file" in the docs\.$'):
        runner.run("scMDC", "vertical", inputs={"rna": src, "adt": src}, out_dir="out",
                   dry_run=True)
    for argv in (["export", src, "OUT/"],
                 ["export", src, "OUT", "--rna", "X"]):
        assert cli.main(argv) == 1
        assert capsys.readouterr().err.splitlines()[0] == (
            f'error: {src} is a 10x Genomics file. See the recipe "A 10x .h5 file" in the '
            f'docs.')
    assert sorted(p.name for p in Path(".").iterdir()) == [src]


def test_the_fences_are_the_ones_these_tests_run():
    """The tests below take a fence by its position in its recipe. A fence
    that is added, removed or moved on the page fails here first."""
    for anchor, first_lines in FENCES.items():
        assert [f.splitlines()[0] for f in _fences(anchor)] == first_lines, anchor
    # the section opens by saying where a recipe runs and which names it uses
    opening = " ".join(_section().split("####", 1)[0].split())
    assert ("Run them in the tutorial notebook of your task: `mtb` is imported, `METHODS` is "
            "the list of methods from its section 2, and its check cell and `run_all` cell "
            "use `NAME`, the folder name that a recipe sets.") in opening
    assert ("The recipes write to `mydata/` and call the cell-type column `celltype`, as the "
            "tutorials do.") in opening


def test_the_dataset_folder_section_points_to_the_recipes():
    """The section on dataset folders has tabs for data in an AnnData or a
    MuData. A reader who starts from another kind of file finds the recipes
    from its visible text, above the tabs."""
    page = _read("tutorials/run.md")
    section = page.split("\n### Your own data as a dataset folder\n", 1)[1] \
        .split("\n### Recipes for your own data", 1)[0]
    pointer = ("For a Cell Ranger `.h5` file, labels in a CSV file, data without labels or "
               "another number of donors, see [Recipes for your own data](#own-data-recipes).")
    paragraphs = [" ".join(par.split()) for par in section.strip().split("\n\n")]
    assert paragraphs[0] == ("`mtb.io.export_dataset` writes an AnnData or MuData as a dataset "
                             "folder, which `scan` and `run_all` read.")
    assert paragraphs[1] == pointer and " ".join(page.split()).count(pointer) == 1
    # before the first tab, and outside every collapsed block
    link = "[Recipes for your own data](#own-data-recipes)"
    assert section.index(link) < section.index('\n=== "') < section.index("\n??? note ")
    assert pointer in " ".join(_visible(section).split())
    # the link leads to the heading of the recipes, which follow in that order
    assert re.search(r"^### Recipes for your own data \{ #own-data-recipes \}$", page, flags=re.M)
    titles = re.findall(r"^#### (.+?) \{ #recipe-[a-z0-9-]+ \}$", page, flags=re.M)
    assert titles == ["A 10x `.h5` file", "Labels in a CSV file", "No labels",
                      "Another number of donors"]


# ------------------------------------------------------------ a 10x .h5 file
def test_10x_recipe_runs(host, tmp_path, monkeypatch, capsys):
    """A 10x Multiome file, in the notebook of the vertical RNA + ATAC
    tutorial. The recipe writes its own folder, with the labels of a shuffled
    CSV on the cells of their rows, and leaves the folder of the tutorial's
    example as it is. The check cell and the run cell of the tutorial then
    work on the folder of the recipe. A CSV with other barcodes shows in the
    number that the label block prints, and the block runs again on the
    corrected table."""
    key = "vertical_rna_atac"
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    for i in (EXAMPLE, READ, EXPORT):
        _run_cell(src[i], ns)
    example = Path("mydata") / NAMES[key]
    before = _bytes(example)
    bars = _tenx("filtered_feature_bc_matrix.h5", GENES, PEAKS, "atac")
    fences = _fences("recipe-10x-h5")
    kinds = _feature_types()
    capsys.readouterr()

    # "scanpy warns that variable names are not unique, here and in the next steps"
    _, warned = _run_recipe_cell(fences[READ_10X], ns)
    assert warned
    # "The printout lists the feature types of the file."
    printed = capsys.readouterr().out
    assert re.search(rf"^{re.escape(kinds['rna'])} +120$", printed, flags=re.M)
    assert re.search(rf"^{re.escape(kinds['atac'])} +60$", printed, flags=re.M)
    # "The lines are those of Labels in a CSV file": the block that adds the
    # labels, and the block that keeps the labelled cells
    assert fences[LABELS:KEEP + 1] == _fences("recipe-labels-csv")
    assert fences[LABELS].splitlines()[2].endswith("# <- your file, barcodes first")
    # a CSV whose barcodes lack the -1 ending: the label block prints 0 and
    # leaves the cells in, so the reader can stop there, as the recipe says
    compare, theirs = _quoted(
        "recipe-10x-h5", r"When it prints 0, or fewer cells than you expect, stop here and "
                         r"compare `([^`]+)` with `([^`]+)`\. " + re.escape(ENDING_SENTENCE)
                         + r" ```python [^`]+``` Then keep the labelled cells\.")
    short = _annotation("annotation.csv", [b[:-2] for b in bars])
    _run_recipe_cell(fences[LABELS], ns)
    assert capsys.readouterr().out == "0 of 300 cells have a label\n"
    assert ns["adata"].n_obs == 300
    assert not [c for c in eval(compare, ns) if c.endswith("-1")]
    assert all(c.endswith("-1") for c in eval(theirs, ns))
    # the line of the recipe, put where the recipe says: every cell has the
    # label of its row of the CSV
    _run_recipe_cell(_with_ending(fences[LABELS], fences[ENDING]), ns)
    assert capsys.readouterr().out == "300 of 300 cells have a label\n"
    assert ns["adata"].obs["celltype"].to_dict() == {b + "-1": v for b, v in short.items()}
    # the block again, with the corrected table
    labels = _annotation("annotation.csv", bars, keep=270)
    _run_recipe_cell(fences[LABELS], ns)
    assert capsys.readouterr().out == "270 of 300 cells have a label\n"
    assert ns["adata"].n_obs == 300
    _, warned = _run_recipe_cell(fences[KEEP], ns)
    assert warned
    assert ns["adata"].n_obs == 270
    # the error that the recipe names for the object before the split
    error = _quoted("recipe-10x-h5", r"`export_dataset` raises `(\w+)` when a modality would "
                                     r"hold two feature types, for example `rna=\"X\"` on the "
                                     r"object before the split\.")
    with pytest.raises(ValueError, match=r"^rna='X' selects features of 2 types, ") as e:
        mtb.io.export_dataset(ns["adata"], "mydata/TWO_TYPES", rna="X")
    assert type(e.value).__name__ == error
    assert not Path("mydata/TWO_TYPES").exists()
    # the split takes each part by the type that the recipe names for it
    assert f'kind == "{kinds["rna"]}"' in fences[MULTIOME]
    assert f'kind == "{kinds["atac"]}"' in fences[MULTIOME]
    _, warned = _run_recipe_cell(fences[MULTIOME], ns)
    assert warned

    mine = Path("mydata") / "MY10X"
    assert ns["NAME"] == "MY10X"
    assert sorted(p.name for p in mine.iterdir()) == ["atac.h5", "atac_peak.h5", "cty.csv",
                                                      "rna.h5"]
    assert _h5(mine / "rna.h5", "data") == (len(GENES), 270)
    assert _h5(mine / "atac_peak.h5", "data") == (len(PEAKS), 270)
    assert _h5(mine / "atac.h5", "data") == (len(PEAKS), 270)
    # the gene names only were made unique
    genes = _h5(mine / "rna.h5", "features")
    assert genes[:3] == ["GENE0", "GENE0-1", "GENE2"] and len(set(genes)) == len(GENES)
    assert _h5(mine / "atac_peak.h5", "features") == PEAKS
    # pandas matched the rows of the CSV by barcode
    cells = _h5(mine / "rna.h5", "barcodes")
    assert set(cells) == set(labels) and len(cells) == 270
    assert pd.read_csv(mine / "cty.csv")["x"].tolist() == [labels[c] for c in cells]
    # "features x cells x 8 bytes": the matrix is stored dense, as float64
    with h5py.File(mine / "rna.h5", "r") as f:
        assert f["matrix/data"].dtype == np.float64
    assert "150,000 peaks and 10,000 cells take 12 GB." in _said("recipe-10x-h5")
    assert 150_000 * 10_000 * 8 == 12 * 10**9
    assert _bytes(example) == before

    # the check cell and the run cell of the tutorial use NAME
    capsys.readouterr()
    table = _run_cell(src[CHECK], ns)
    assert capsys.readouterr().out.splitlines() == [
        "[scan] 2 of 2 methods have their input files in MY10X. 2 of 2 can run on this "
        "computer. MY10X holds RNA and ATAC.",
        "[scan] Cells per file: rna.h5 270, atac.h5 270, atac_peak.h5 270."]
    assert sorted(table["method"]) == sorted(ns["METHODS"]) and table["runnable"].all()
    # the check that the page shows is that cell
    assert "".join(fences[PAGE_CHECK].split()) == "".join(src[CHECK].split())
    assert _run_cell(fences[PAGE_CHECK], ns).equals(table)
    # the two ATAC names of its second line: one matrix, the same cells
    first, second = _quoted(
        "recipe-10x-h5", r"For 10x Multiome the check lists `([^`]+)` and `([^`]+)`: one "
                         r"matrix under two names, one copy on disk\.")   # Z21 (r3): no reader claim
    assert (first, second) == ("atac.h5", "atac_peak.h5")
    assert (mine / first).samefile(mine / second)          # "one copy on disk"
    assert capsys.readouterr().out.splitlines()[1] == (
        f"[scan] Cells per file: rna.h5 270, {first} 270, {second} 270.")
    content = _content(mine)
    assert content[first] == content[second] != content["rna.h5"]
    # the diagonal methods read the second name; none reads a file named atac.h5
    roles = {r for m in mtb.list_methods(category="diagonal")
             for v in mtb.method_info(m)["supports"] if v["category"] == "diagonal"
             for r in v["modalities"] if r.startswith("atac")}
    assert Path(second).stem in roles and Path(first).stem not in roles
    _quiet(_run_cell, src[RUN], ns)
    res = ns["mine"]
    assert (res.dataset, res.category) == ("MY10X", "vertical")
    assert sorted(m for m, _ in host) == sorted(ns["METHODS"])
    for _m, files in host:
        assert {f.parent.resolve() for f in files.values()} == {mine.resolve()}
    assert dict(zip(res.summary["method"], res.summary["status"])) == {
        m: "scored" for m in ns["METHODS"]}
    assert sorted(p.name for p in Path("out").iterdir()) == ["MY10X"]
    assert _bytes(example) == before


def test_10x_recipe_for_cite_seq(host, tmp_path, monkeypatch, capsys):
    """A CITE-seq file, in the notebook of the vertical RNA + ADT tutorial.
    Two antibodies have the name of a gene. The split makes the gene names
    unique and leaves the antibodies their names."""
    key = "vertical_rna_adt"
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    for i in (EXAMPLE, READ, EXPORT):
        _run_cell(src[i], ns)
    example = Path("mydata") / NAMES[key]
    before = _bytes(example)
    bars = _tenx("filtered_feature_bc_matrix.h5", GENES, ANTIBODIES, "adt")
    labels = _annotation("annotation.csv", bars, keep=270)
    fences = _fences("recipe-10x-h5")
    kinds = _feature_types()
    _run_recipe_cell(fences[READ_10X], ns)
    assert re.search(rf"^{re.escape(kinds['adt'])} +20$", capsys.readouterr().out, flags=re.M)
    assert f'kind == "{kinds["adt"]}"' in fences[CITE]
    _run_recipe_cell(fences[LABELS], ns)
    _run_recipe_cell(fences[KEEP], ns)
    # "on the whole object it also renames an antibody that has the name of a gene"
    whole = _quiet(ns["adata"].copy)
    _quiet(whole.var_names_make_unique)
    renamed = [new for old, new in zip(ANTIBODIES, whole.var_names[len(GENES):]) if old != new]
    assert renamed == ["GENE5-1", "GENE9-1"]
    _run_recipe_cell(fences[CITE], ns)

    assert list(ns["adt"].var_names) == ANTIBODIES
    assert list(ns["rna"].var_names[:3]) == ["GENE0", "GENE0-1", "GENE2"]
    assert ns["rna"].var_names.is_unique
    mine = Path("mydata") / "MY10X"
    assert sorted(p.name for p in mine.iterdir()) == ["adt.h5", "cty.csv", "rna.h5"]
    assert _h5(mine / "adt.h5", "features") == ANTIBODIES
    assert _h5(mine / "adt.h5", "data") == (len(ANTIBODIES), 270)
    assert _h5(mine / "rna.h5", "data") == (len(GENES), 270)
    cells = _h5(mine / "rna.h5", "barcodes")
    assert _h5(mine / "adt.h5", "barcodes") == cells
    assert pd.read_csv(mine / "cty.csv")["x"].tolist() == [labels[c] for c in cells]
    assert _bytes(example) == before
    # the check cell of this tutorial works on the folder of the recipe
    capsys.readouterr()
    table = _run_cell(src[CHECK], ns)
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "[scan] 2 of 2 methods have their input files in MY10X. 2 of 2 can run on this "
        "computer. MY10X holds RNA and ADT.",
        "[scan] Cells per file: rna.h5 270, adt.h5 270."]
    assert table["runnable"].all()
    # "For 10x Multiome the check lists atac.h5 and atac_peak.h5": this folder
    # has neither
    for name in _quoted("recipe-10x-h5", r"For 10x Multiome the check lists `([^`]+)` and "
                                         r"`([^`]+)`: one matrix under two names, one copy on disk\."):
        assert not (mine / name).exists() and name not in printed[1]


@pytest.mark.parametrize("other,modality,fence,files", [
    (PEAKS, "atac", MULTIOME, ["atac.h5", "atac_peak.h5", "rna.h5"]),
    (ANTIBODIES, "adt", CITE, ["adt.h5", "rna.h5"]),
], ids=["multiome", "cite_seq"])
def test_10x_recipe_without_labels(other, modality, fence, files, tmp_path, monkeypatch,
                                   capsys):
    """The recipe says to skip the two label blocks without cell types, and
    to leave the ``labels=`` line out of the export. The folder then has no
    label file."""
    _says("recipe-10x-h5", "Without cell types, skip the next two blocks and leave the "
                           "`labels=` line out of the export below, as in "
                           "[No labels](#recipe-no-labels).")
    monkeypatch.chdir(tmp_path)
    _tenx("filtered_feature_bc_matrix.h5", GENES, other, modality)
    fences, ns = _fences("recipe-10x-h5"), {"mtb": mtb}
    _run_recipe_cell(fences[READ_10X], ns)
    _says("recipe-10x-h5", "`gex_only=False` keeps the peaks or the antibody counts. Without "
                           "it, scanpy returns the genes alone.")
    assert ns["adata"].n_vars == len(GENES) + len(other)
    assert _quiet(ns["sc"].read_10x_h5, "filtered_feature_bc_matrix.h5").n_vars == len(GENES)
    # the second of the two blocks needs the column that the first one adds
    with pytest.raises(KeyError, match="celltype"):
        _run_recipe_cell(fences[KEEP], ns)
    line = '                      labels="obs:celltype",\n'
    assert fences[fence].count(line) == 1
    _run_recipe_cell(fences[fence].replace(line, ""), ns)
    mine = Path("mydata") / "MY10X"
    assert sorted(p.name for p in mine.iterdir()) == files
    assert _h5(mine / "rna.h5", "data") == (len(GENES), 300)
    assert capsys.readouterr().err == ""


# ------------------------------------------------------- labels in a CSV file
def _label_forms():
    rng = np.random.default_rng(2)
    base = np.array(rng.choice(["T", "B", "NK"], 300), dtype=object)
    blank = base.copy()
    blank[:10], blank[10:20] = " ", ""
    spaced = base.copy()
    spaced[:10], spaced[10:15] = "NA ", "NA"
    gaps = base.copy()
    gaps[:40] = None
    return [("blank", blank, 280), ("NA_with_a_space", spaced, 285),
            ("integer_clusters", rng.integers(0, 5, 300), 300),
            ("categorical", pd.Categorical(base), 300), ("missing_values", gaps, 260)]


@pytest.mark.parametrize("labels,labelled", [f[1:] for f in _label_forms()],
                         ids=[f[0] for f in _label_forms()])
def test_csv_recipe_label_forms(labels, labelled, tmp_path, monkeypatch, capsys):
    """The number that the recipe prints is the number of labels that the
    export then writes, whatever a missing label looks like in the CSV."""
    monkeypatch.chdir(tmp_path)
    adata = _cells()
    written = _annotation("annotation.csv", adata.obs_names, labels)
    add, _ending, keep = _fences("recipe-labels-csv")
    ns = {"mtb": mtb, "adata": adata}
    _run_cell(add, ns)
    assert capsys.readouterr().out == f"{labelled} of 300 cells have a label\n"
    _run_cell(keep, ns)
    assert ns["adata"].n_obs == labelled
    # "Then go on with the export cell and labels="obs:celltype""
    out = mtb.io.export_dataset(ns["adata"], "mydata/MYCSV", rna="X", labels="obs:celltype")
    got = pd.read_csv(out / "cty.csv")["x"].astype(str).tolist()
    assert len(got) == labelled
    # each label sits on its cell, without the blanks around it
    assert got == [str(written[c]).strip() for c in _h5(out / "rna.h5", "barcodes")]


def test_csv_recipe_names_another_missing_text(tmp_path, monkeypatch, capsys):
    """A text that the list of the recipe does not have is named by
    ``export_dataset``, and the recipe says to add it to the list."""
    _says("recipe-labels-csv", "The third line turns a blank label and the text `NA` into a "
                               "missing value, because a label file reads them as missing.")
    _says("recipe-labels-csv", "When `export_dataset` names another such text, add it to that "
                               "list.")
    monkeypatch.chdir(tmp_path)
    adata = _cells()
    labels = np.array(["T", "B", "NK"] * 100, dtype=object)
    labels[:12] = "N/A "                        # read as text, for the blank after it
    _annotation("annotation.csv", adata.obs_names, labels)
    add, _ending, keep = _fences("recipe-labels-csv")
    # a label file reads each text of the list in the third line as missing
    listed = '["", "NA", "nan"]'
    assert add.count(listed) == 1 and listed in add.splitlines()[4]
    texts = json.loads(listed)
    assert eio.missing_mask(texts).all()
    Path("labels.csv").write_text("cell,label\n" + "".join(
        f"c{i},{t}\n" for i, t in enumerate(texts + ["T"])))
    assert pd.isna(eio.read_labels("labels.csv")).tolist() == [True] * len(texts) + [False]
    ns = {"mtb": mtb, "adata": adata.copy()}
    _run_cell(add, ns)
    _run_cell(keep, ns)
    assert capsys.readouterr().out == "300 of 300 cells have a label\n"
    with pytest.raises(ValueError) as e:
        mtb.io.export_dataset(ns["adata"], "mydata/MYCSV", rna="X", labels="obs:celltype")
    assert str(e.value) == (
        "labels='obs:celltype': 12 of the 300 labels for the cells (the AnnData's obs) are missing "
        "(NaN, None, '' or text such as 'N/A'); give every cell a label, or subset the data "
        "to the labelled cells first")
    assert not Path("mydata").exists()
    ns = {"mtb": mtb, "adata": adata.copy()}
    _run_cell(add.replace(listed, '["", "NA", "nan", "N/A"]'), ns)
    _run_cell(keep, ns)
    assert capsys.readouterr().out == "288 of 300 cells have a label\n"
    out = mtb.io.export_dataset(ns["adata"], "mydata/MYCSV", rna="X", labels="obs:celltype")
    assert len(pd.read_csv(out / "cty.csv")) == 288


def test_csv_with_other_barcodes_stops(tmp_path, monkeypatch, capsys):
    """The barcodes of the CSV lack the ``-1`` of the cell names. The recipe
    prints 0 and says to stop there. A reader who goes on is stopped by the
    export, which writes nothing."""
    monkeypatch.chdir(tmp_path)
    adata = _cells()
    _annotation("annotation.csv", [c[:-2] for c in adata.obs_names])
    add, _ending, keep = _fences("recipe-labels-csv")
    ns = {"mtb": mtb, "adata": adata}
    _run_cell(add, ns)
    assert capsys.readouterr().out == "0 of 300 cells have a label\n"
    _says("recipe-labels-csv", "When it prints 0, or fewer cells than you expect, stop here.")
    # "Compare ann.index[:3] with adata.obs_names[:3]": the ending shows
    _says("recipe-labels-csv", "Compare `ann.index[:3]` with `adata.obs_names[:3]`")
    # the ending that the recipe names as the usual difference, in the two
    # expressions it says to compare
    ending, mine, theirs = _quoted(
        "recipe-labels-csv", r"often by an ending such as `([^`]+)`\. Compare `([^`]+)` with "
                             r"`([^`]+)`, and do not go on until the number is right\.")
    assert ending == "-1"
    assert not [c for c in eval(mine, ns) if c.endswith(ending)]
    assert all(c.endswith(ending) for c in eval(theirs, ns))
    _run_cell(keep, ns)
    with pytest.raises(ValueError) as e:
        mtb.io.export_dataset(ns["adata"], "mydata/MYCSV", rna="X", labels="obs:celltype")
    assert str(e.value) == ZERO_CELLS
    assert not Path("mydata").exists()


def test_csv_without_the_ending_gets_it_from_the_line_of_the_recipe(tmp_path, monkeypatch,
                                                                   capsys):
    """The barcodes of the CSV lack the ``-1`` of the cell names. The reader
    copies the line of the recipe into the block that adds the column, after
    the line that reads the CSV, and runs the block again: every cell gets
    the label of its row. A second run adds the ending once, because the
    block reads the CSV first."""
    monkeypatch.chdir(tmp_path)
    adata = _cells()
    short = _annotation("annotation.csv", [c[:-2] for c in adata.obs_names])
    add, ending, keep = _fences("recipe-labels-csv")
    assert ending == 'ann.index = ann.index + "-1"\n'
    _says("recipe-labels-csv", ENDING_SENTENCE + " The line that reads the CSV comes first, "
                               "so that a second run does not add the ending twice.")
    ns = {"mtb": mtb, "adata": adata}
    _run_cell(add, ns)
    assert capsys.readouterr().out == "0 of 300 cells have a label\n"
    block = _with_ending(add, ending)
    for _ in range(2):
        _run_cell(block, ns)
        assert capsys.readouterr().out == "300 of 300 cells have a label\n"
        assert ns["adata"].obs["celltype"].to_dict() == {b + "-1": v for b, v in short.items()}
    # the line alone, run twice, would add the ending twice: the recipe does
    # not say to run it alone
    assert list(ns["ann"].index) == [b + "-1" for b in short]
    _run_cell(keep, ns)
    out = mtb.io.export_dataset(ns["adata"], "mydata/MYCSV", rna="X", labels="obs:celltype")
    assert pd.read_csv(out / "cty.csv")["x"].tolist() \
        == [short[b[:-2]] for b in _h5(out / "rna.h5", "barcodes")]


def test_csv_with_half_the_cells(tmp_path, monkeypatch, capsys):
    """A CSV for half of the cells: the recipe says so, keeps the labelled
    cells, and a second object of the same cells is cut to them."""
    monkeypatch.chdir(tmp_path)
    adata = _cells()
    atac = _cells(genes=30, seed=1)
    atac.var_names = PEAKS[:30]
    _annotation("annotation.csv", adata.obs_names, keep=150)
    add, _ending, keep = _fences("recipe-labels-csv")
    ns = {"mtb": mtb, "adata": adata, "atac": atac}
    _run_cell(add, ns)
    assert capsys.readouterr().out == "150 of 300 cells have a label\n"
    _says("recipe-labels-csv", "A cell without a label cannot be scored, and `export_dataset` "
                               "refuses it. When some cells have no label, keep the labelled "
                               "cells.")
    with pytest.raises(ValueError, match=r"^labels='obs:celltype': 150 of the 300 labels "):
        mtb.io.export_dataset(ns["adata"], "mydata/MYCSV", rna="X", labels="obs:celltype")
    assert not Path("mydata").exists()
    _run_cell(keep, ns)
    again = "atac = atac[adata.obs_names].copy()"
    assert f"`{again}`" in _said("recipe-labels-csv")
    _run_cell(again, ns)
    out = mtb.io.export_dataset(ns["adata"], "mydata/MYCSV", rna="X", atac=ns["atac"],
                                atac_kind="peak", labels="obs:celltype")
    assert _h5(out / "rna.h5", "barcodes") == _h5(out / "atac.h5", "barcodes")
    assert len(pd.read_csv(out / "cty.csv")) == 150 == len(_h5(out / "rna.h5", "barcodes"))


def test_csv_recipe_names_the_object_above_its_code(tmp_path, monkeypatch, capsys):
    """The code of the recipe writes ``adata``. The paragraph directly above
    the first block says which object that is, and in the notebook of an
    RNA + ATAC tutorial, which has no ``adata``, the blocks run with ``rna``
    written for it in every line."""
    above = " ".join(_recipe("recipe-labels-csv").split("```python", 1)[0]
                     .strip().split("\n\n")[-1].split())
    assert above == (
        "In the code, `adata` is the object that holds your cells. In a tutorial that reads "
        "one object, it is `adata`. In the RNA + ATAC tutorials it is `rna`, and the name "
        "`adata` does not exist there: write `rna` for `adata` in every line of this recipe. "
        "With another name, such as `adt` or `rna1`, write the object whose cells the CSV "
        "lists.")
    _says("recipe-labels-csv", "For diagonal data, do the recipe once for `rna` and once for "
                               "`atac`.")
    # the objects that the read cells of the tutorials define
    reads = {key: re.findall(r"^(\w+) = ad\.read_h5ad\(", _sources(key)[READ], flags=re.M)
             for key in GEN.TASKS}
    assert reads["vertical_rna_adt"] == reads["cross_rna_adt"] == ["adata"]
    for key in ("vertical_rna_atac", "diagonal_rna_atac", "cross_rna_atac"):
        assert reads[key] == ["rna", "atac"], key
    assert "adt" in reads["cross_adt_atac"] and "rna1" in reads["mosaic_rna_adt"]

    monkeypatch.chdir(tmp_path)
    rna = _cells()
    atac = _cells(genes=30, seed=1)
    atac.var_names = PEAKS[:30]
    _annotation("annotation.csv", rna.obs_names, keep=150)
    add, _ending, keep = _fences("recipe-labels-csv")
    again = "atac = atac[adata.obs_names].copy()"
    assert f"`{again}`" in _said("recipe-labels-csv")
    # as pasted, the first block stops on the name
    ns = {"mtb": mtb, "rna": rna.copy(), "atac": atac.copy()}
    with pytest.raises(NameError, match="name 'adata' is not defined"):
        _run_cell(add, ns)
    assert "celltype" not in ns["rna"].obs
    # with rna for adata in every line
    for block in (add, keep, again):
        assert "adata" in block
        _run_cell(re.sub(r"\badata\b", "rna", block), ns)
    assert capsys.readouterr().out == "150 of 300 cells have a label\n"
    assert "adata" not in ns and ns["rna"].n_obs == 150
    assert list(ns["atac"].obs_names) == list(ns["rna"].obs_names)
    out = mtb.io.export_dataset(ns["rna"], "mydata/MYCSV", rna="X", atac=ns["atac"],
                                atac_kind="peak", labels="obs:celltype")
    assert _h5(out / "rna.h5", "barcodes") == _h5(out / "atac.h5", "barcodes")
    assert len(pd.read_csv(out / "cty.csv")) == 150


def test_csv_with_a_barcode_twice_stops(tmp_path, monkeypatch, capsys):
    """The recipe says that the barcodes of the CSV must be unique, and
    quotes the error of pandas for one that is there twice. The object is
    left as it was."""
    stops = _quoted("recipe-labels-csv", r"The barcodes in the CSV must be unique: a duplicated "
                                         r"barcode stops the block with the pandas error "
                                         r"`([^`]+)`\.")
    monkeypatch.chdir(tmp_path)
    adata = _cells()
    labels = np.random.default_rng(1).choice(["T", "B", "NK"], adata.n_obs)
    ann = pd.DataFrame({"cluster_annotation": labels}, index=list(adata.obs_names))
    pd.concat([ann, ann.iloc[:1]]).to_csv("annotation.csv")       # the first barcode again
    add, _ending, _keep = _fences("recipe-labels-csv")
    ns = {"mtb": mtb, "adata": adata}
    with pytest.raises(ValueError) as e:
        _run_cell(add, ns)
    assert str(e.value) == stops
    assert "celltype" not in ns["adata"].obs and ns["adata"].n_obs == 300
    assert capsys.readouterr().out == ""
    # the same rows without the repeat
    ann.to_csv("annotation.csv")
    _run_cell(add, ns)
    assert capsys.readouterr().out == "300 of 300 cells have a label\n"


# ------------------------------------------------------------------ no labels
def test_no_labels_recipe(host, tmp_path, monkeypatch, capsys):
    """A folder without labels, in the notebook of the vertical RNA + ADT
    tutorial: the check names the method that reads the cell types, the
    others run without scores, and labels that come later are written on the
    cells of their rows and score the saved embeddings."""
    monkeypatch.chdir(tmp_path)
    adata = _cite()
    methods = list(GEN.TASKS["vertical_rna_adt"]["methods"])
    assert methods == ["Matilda", "sciPENN"]
    ns = {"mtb": mtb, "METHODS": methods, "adata": adata}
    export, run, later = _fences("recipe-no-labels")

    table = _run_cell(export, ns)
    folder = Path("mydata") / "MYDATA"
    assert sorted(p.name for p in folder.iterdir()) == ["adt.h5", "rna.h5"]
    out = capsys.readouterr().out.splitlines()
    assert out[0] == ("[scan] 1 of 2 methods has its input files in MYDATA. 1 of 2 can run on "
                      "this computer. MYDATA holds RNA and ADT.")
    # the reason that the recipe quotes for a method that reads the cell types
    missing = _quoted("recipe-no-labels", r"A few methods also read them as an input\. The "
                                          r"check names those methods: their `reason` says "
                                          r"`([^`]+)`")
    assert missing == "cty.csv is missing."
    reason = dict(zip(table["method"], table["reason"]))
    assert reason["Matilda"] == missing
    # runnable: empty, or the note that the method scripts are not fetched yet
    assert reason["sciPENN"] == "" or reason["sciPENN"].startswith(
        "The method scripts are not in ")
    assert f"[scan] Matilda: {missing}" in out
    # the two blocks are those of this tutorial
    _says("recipe-no-labels", "The two blocks below are those of the vertical RNA + ADT "
                              "tutorial. In another tutorial, do the same by removing the "
                              "`labels=` line of that tutorial's export cell, and pass "
                              "`evaluate=False` in its `run_all` cell.")
    assert _recipe("recipe-no-labels").index("The two blocks below") < \
        _recipe("recipe-no-labels").index("```python")
    tutorial = _without_labels(_sources("vertical_rna_adt")[EXPORT])
    assert [line.split("#")[0].split() for line in tutorial.splitlines()[1:]] == [
        line.split("#")[0].split() for line in export.splitlines()[1:-1]]

    # the two statuses that the recipe explains, and the third without evaluate=False
    ran, skipped = _quoted(
        "recipe-no-labels", r"The status `(\w+)` means that the method ran and nothing was "
                            r"scored\. `(\w+)` marks a method that needs cell types as an "
                            r"input\.")
    unmatched = _quoted("recipe-no-labels", r"Without `evaluate=False` the methods run as "
                                            r"well, and the status `(\w+)` says that nothing "
                                            r"was scored\.")
    assert (ran, skipped, unmatched) == ("ran", "skipped", "ran")        # c1 (C38)
    status = _run_cell(run, ns)
    assert list(status.columns) == ["method", "status"]
    assert dict(zip(status["method"], status["status"])) == {"Matilda": skipped, "sciPENN": ran}
    assert [m for m, _ in host] == ["sciPENN"]
    assert not set(ns["mine"].summary.columns) & set(mtb.catalog.known_metrics())
    # the folder of the run, and the file in it, as the recipe names them: a
    # folder for the method that ran, none for the one that was skipped
    out_dir, embedding = _quoted(
        "recipe-no-labels", r"Each method that ran has a folder in `([^`]+)`\. For most methods "
                            r"the embedding is the file `([^`]+)` in it\.")
    assert (out_dir, embedding) == ("out/MYDATA/", "embedding.h5")
    assert Path(out_dir) == Path("out") / ns["NAME"]
    (run_dir,) = [p for p in Path(out_dir).iterdir() if p.is_dir()]
    assert run_dir.name.startswith("sciPENN") and (run_dir / embedding).is_file()
    unscored = mtb.run_all("MYDATA", "vertical", methods=methods, data_path="mydata",
                           out_dir="out/UNSCORED", verbose=False)
    assert dict(zip(unscored.summary["method"], unscored.summary["status"])) == {
        "Matilda": skipped, "sciPENN": unmatched}
    assert [m for m, _ in host] == ["sciPENN"] * 2
    host.clear()

    # cell types later: by then the object is in another row order
    _says("recipe-no-labels", "When you have cell types later, add them to the folder and "
                              "score the saved embeddings. No method runs again.")
    _says("recipe-no-labels", "`export_dataset` writes the labels in the cell order of the "
                              "files already in the folder, whatever the row order of `adata` "
                              "is by then. It refuses labels for other cells.")
    ns["adata"] = adata[np.random.default_rng(3).permutation(adata.n_obs)].copy()
    assert list(ns["adata"].obs_names) != list(adata.obs_names)
    capsys.readouterr()
    for _ in (1, 2):                             # the cell runs a second time
        summary = _quiet(_run_cell, later, ns)
        assert capsys.readouterr().err == (
            "export_dataset: wrote cty.csv in the cell order of rna.h5.\n")
        cells = _h5(folder / "rna.h5", "barcodes")
        assert cells == list(adata.obs_names)
        assert pd.read_csv(folder / "cty.csv")["x"].tolist() == \
            adata.obs["celltype"].reindex(cells).tolist()
        scored = summary.set_index("method")
        assert scored.loc["sciPENN", "status"] == "scored"
        assert np.isfinite(scored.loc["sciPENN", "ARI"])
    assert host == []                            # no method ran again
    # labels for other cells
    other = adata.copy()
    other.obs_names = [f"other{i}" for i in range(other.n_obs)]
    before = _bytes(folder)
    with pytest.raises(FileExistsError, match=r"^mydata/MYDATA holds rna\.h5 and 1 more file "
                                              r"with other cells\. Export to a new folder\.$"):
        mtb.io.export_dataset(other, "mydata/MYDATA", rna=None, labels="obs:celltype",
                              overwrite=True)
    assert _bytes(folder) == before


def test_the_label_file_that_the_check_names_depends_on_the_layout(host, tmp_path, monkeypatch,
                                                                  capsys):
    """The reason that the recipe quotes first is the one of a vertical
    folder. For a diagonal folder without labels the recipe quotes two
    other sentences, and the reasons of the check hold each of them, for the
    folder that the export cell of the diagonal tutorial writes without its
    ``labels=`` line, and for one with ATAC peaks."""
    one, both = _quoted(
        "recipe-no-labels", r"The label file that it names depends on the layout\. For diagonal "
                            r"data it is `rna_cty\.csv`, for some methods together with "
                            r"`atac_cty\.csv`: the reason then holds `([^`]+)` or `([^`]+)` Without cell types, leave "
                            r"`labels=` out of the export and pass `evaluate=False` to "
                            r"`run_all`\.")
    assert (one, both) == ("rna_cty.csv is missing.", "rna_cty.csv, atac_cty.csv are missing.")
    monkeypatch.chdir(tmp_path)
    rna = _cells(70)
    atac = _cells(60, seed=1)
    atac.obs_names = [f"A{i:04d}-1" for i in range(60)]
    peaks = _cells(60, genes=30, seed=2)
    peaks.obs_names, peaks.var_names = atac.obs_names, PEAKS[:30]
    # the export cell of the diagonal tutorial, without its labels= line
    cell = _without_labels(_sources("diagonal_rna_atac")[EXPORT])
    _run_cell(cell, {"mtb": mtb, "rna": rna, "atac": atac})
    name = NAMES["diagonal_rna_atac"]
    assert sorted(p.name for p in (Path("mydata") / name).iterdir()) == ["atac_gas.h5", "rna.h5"]
    mtb.io.export_dataset(rna, "mydata/PEAKS", atac=peaks, atac_kind="peak",
                          category="diagonal")
    assert sorted(p.name for p in Path("mydata/PEAKS").iterdir()) == ["atac_peak.h5", "rna.h5"]
    capsys.readouterr()
    seen = {}
    for folder in (name, "PEAKS"):
        table = mtb.scan(folder, "diagonal", data_path="mydata")
        out = capsys.readouterr().out.splitlines()
        reason = dict(zip(table["method"], table["reason"]))
        seen[folder] = {m: r for m, r in reason.items() if "cty" in r}
        for m, r in seen[folder].items():
            assert f"[scan] {m}: {r}" in out
        # no reason names the label file of a vertical folder
        assert not [r for r in reason.values() if re.search(r"(?<!_)cty\.csv", r)]
    # alone where the method has its other files, and after the sentence
    # about the ATAC file where it has not
    assert seen[name]["scJoint"] == one
    assert seen["PEAKS"]["scJoint"] == (
        "scJoint needs gene-activity ATAC (atac_gas.h5), and the folder has peaks "
        "(atac_peak.h5). " + one)
    assert seen[name]["Seurat_v3"] == (
        "Seurat_v3 needs peak ATAC (atac_peak.h5), and the folder has gene activity "
        "(atac_gas.h5). " + both)
    assert seen["PEAKS"]["Seurat_v3"] == (
        "Seurat_v3 needs gene-activity ATAC (atac_gas.h5), and the folder has peaks "
        "(atac_peak.h5). " + both)
    assert {m for rs in seen.values() for m in rs} == {"scJoint", "Seurat_v3", "scBridge"}
    # every reason about labels names the file of the RNA side
    assert all("rna_cty.csv" in r for rs in seen.values() for r in rs.values())
    # and the methods of the tutorial are not among them
    assert not set(seen[name]) & set(GEN.TASKS["diagonal_rna_atac"]["methods"])


def test_a_diagonal_folder_with_labels_for_one_side_is_not_scored(host, tmp_path, monkeypatch):
    """Diagonal data with cell types for the RNA cells only is the case of
    this recipe: nothing scores it."""
    monkeypatch.chdir(tmp_path)
    assert ("`run_all` scores a diagonal folder only when both sides have labels. Leave "
            "`labels=` out and pass `evaluate=False`.") in _said("recipe-no-labels")
    rna = _cells(70)
    rna.obs["celltype"] = np.random.default_rng(0).choice(["T", "B"], 70)
    atac = _cells(60, seed=1)
    atac.obs_names = [f"A{i:04d}-1" for i in range(60)]
    # export_dataset does not write the labels of one side alone
    with pytest.raises(KeyError, match=r"Without cell types for the ATAC cells, leave labels= "
                                       r"out\.\"?$"):
        mtb.io.export_dataset(rna, "mydata/ONESIDE", atac=atac, atac_kind="gene_activity",
                              labels="obs:celltype", category="diagonal")
    assert not Path("mydata").exists()
    out = mtb.io.export_dataset(rna, "mydata/ONESIDE", atac=atac, atac_kind="gene_activity",
                                category="diagonal")
    # nor does a label file of one side, put there by hand, give scores
    rna.obs["celltype"].rename("x").to_csv(out / "rna_cty.csv", index=False)
    res = mtb.run_all("ONESIDE", "diagonal", methods=["SCALEX"], data_path="mydata",
                      out_dir="out/ONESIDE", verbose=False)
    assert [r["status"] for r in res.records] == ["RUN_OK_NO_LABEL_MATCH"]
    res = mtb.run_all("ONESIDE", "diagonal", methods=["SCALEX"], data_path="mydata",
                      out_dir="out/UNSCORED", evaluate=False, verbose=False)
    assert [r["status"] for r in res.records] == ["RUN_OK"]


def _labelled_call():
    """The page's call for an embedding with labels for some of the cells
    (0.4.0: a subset of the AnnData, not the deprecated missing_labels=)."""
    call = 'mtb.evaluate(emb[emb.obs["label"].notna()], labels="label")'
    assert f"`{call}`" in _said("recipe-no-labels")
    return call


def _emb(matrix, labels):
    """An embedding as res.embedding(m) returns it: obsm["X_emb"], obs["label"]."""
    emb = ad.AnnData(np.zeros((len(labels), 1)))
    emb.obsm["X_emb"] = matrix
    emb.obs["label"] = np.asarray(labels, dtype=object)
    return emb


def test_the_evaluate_call_of_the_page_scores_the_labelled_cells():
    """The call for an embedding with labels for some of the cells."""
    call = _labelled_call()
    _says("recipe-no-labels", "`evaluate` raises when a label is missing")
    labels = _cite().obs["celltype"].astype(object).to_numpy()
    labels[:12] = None
    emb = _emb(np.random.default_rng(0).normal(size=(len(labels), 4)), labels)
    scores = eval(call, {"mtb": mtb, "emb": emb})
    assert scores["value"].notna().all()
    assert scores.equals(mtb.evaluate(emb[12:].copy(), labels="label"))
    # "evaluate raises when a label is missing."
    with pytest.raises(ValueError, match=r"^labels: 12 of the 120 cells have no label\. "):
        mtb.evaluate(emb, labels="label")


def test_scores_of_one_labelled_side_do_not_see_the_other_side():
    """Cell types for the RNA cells only. The scores of the labelled cells
    are those of the RNA cells alone: embeddings that put the ATAC cells in
    different places get the same scores, so the recipe says that no score
    compares methods on such data."""
    _says("recipe-no-labels", "No score compares methods on such data: scores of the RNA "
                              "cells alone do not change with where a method puts the ATAC "
                              "cells.")
    call = _labelled_call()
    rng = np.random.default_rng(0)
    n = 120
    types = rng.choice(["T", "B", "NK"], n)
    centre = {"T": 0.0, "B": 6.0, "NK": 12.0}
    rna = np.array([[centre[t]] * 4 for t in types]) + rng.normal(size=(n, 4))
    # the ATAC cells on the RNA cells, far from all of them, and at one point
    scores = []
    for atac in (rna, rna + 100.0, np.zeros((n, 4))):
        emb = _emb(np.vstack([rna, atac]), [*types, *[None] * n])   # RNA, then ATAC
        scores.append(eval(call, {"mtb": mtb, "emb": emb}))
    assert scores[0]["value"].notna().all()
    assert scores[1].equals(scores[0]) and scores[2].equals(scores[0])
    # they are the scores of the RNA cells alone
    assert mtb.evaluate(_emb(rna, types), labels="label").equals(scores[0])


# ------------------------------------------------ cell types later, per layout
def _without_labels(export):
    """The export cell of a tutorial without its ``labels=`` argument."""
    out = re.sub(r"^ *labels=[^\n]*\n", "", export, flags=re.M)   # on a line of its own
    out = re.sub(r" labels=\w+,", "", out)                         # next to other arguments
    assert "labels=" not in out and out != export
    return out


def _objects(ns):
    """The names of the AnnData objects that the read cell defined."""
    return [name for name, value in ns.items() if isinstance(value, ad.AnnData)]


def _shuffled(ns, names):
    """Puts the objects ``names`` of the notebook in another row order.
    Returns ``{name: {barcode: cell type}}``."""
    rng = np.random.default_rng(5)
    truth = {}
    for name in names:
        a = ns[name]
        truth[name] = a.obs["celltype"].astype(str).to_dict()
        ns[name] = a[rng.permutation(a.n_obs)].copy()
        assert list(ns[name].obs_names) != list(a.obs_names)
    return truth


def _unlabelled_run(key, ns, host):
    """The read cell of the tutorial ``key``, its export cell without
    ``labels=``, and a run without scores. Returns the folder, the cell order
    of its matrix files, and the methods that ran."""
    src = _sources(key)
    for i in (EXAMPLE, READ):
        _run_cell(src[i], ns)
    _run_cell(_without_labels(src[EXPORT]), ns)
    folder = Path("mydata") / NAMES[key]
    assert ns["NAME"] == NAMES[key]
    assert {p.suffix for p in folder.iterdir()} == {".h5"}          # no label file
    first = mtb.run_all(ns["NAME"], GEN.TASKS[key]["cat"], methods=ns["METHODS"],
                        data_path="mydata", out_dir="out/" + ns["NAME"], evaluate=False,
                        verbose=False)
    status = {r["method"]: r["status"] for r in first.records}
    ran = sorted(m for m, _ in host)
    assert ran == sorted(m for m in status if status[m] == "RUN_OK") and ran
    host.clear()
    return folder, {p.name: _h5(p, "barcodes") for p in sorted(folder.glob("*.h5"))}, ran


def _later():
    """The fence for cell types that come later: its export call, and its
    two lines that score the saved embeddings."""
    lines = _fences("recipe-no-labels")[2].splitlines()
    assert lines[0].startswith("mtb.io.export_dataset(adata, ") and len(lines) == 5
    # c1 (C10): the new scores are saved into the folder
    assert lines[2:] == ['mine = mtb.load_batch("out/" + NAME).rescore()',
                         "mine.save()                        # the new scores into out/MYDATA",
                         "mine.summary"]
    return "\n".join(lines[:2]), "\n".join(lines[2:])


@pytest.mark.parametrize("key", VERTICAL)
def test_labels_later_in_a_vertical_tutorial(key, host, tmp_path, monkeypatch):
    """The line for cell types that come later is written for the folder of
    a vertical tutorial: no batch numbers, one set of cells. ``adata`` is the
    object of the read cell, and in the tutorial that reads ``rna`` and
    ``atac`` the recipe says which name to write."""
    _says("recipe-no-labels", "No method runs again. Here `adata` is the object whose cells "
                              "the folder holds. In the vertical RNA + ATAC tutorial it is "
                              "`rna`.")
    _says("recipe-no-labels", "This line is for a folder without batch numbers that holds one "
                              "set of cells, as the vertical tutorials write it.")
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    ns = _namespace(key)
    folder, order, ran = _quiet(_unlabelled_run, key, ns, host)
    assert not [name for name in order if re.search(r"\d", name[:-len(".h5")])]
    (cells,) = {tuple(c) for c in order.values()}
    # the name that the recipe gives for the object
    name = "rna" if key == "vertical_rna_atac" else "adata"
    assert name == _objects(ns)[0] and ("adata" in ns) == (name == "adata")
    truth = _shuffled(ns, [name])[name]
    export, rescore = _later()
    assert export.count("(adata,") == 1
    _quiet(_run_cell, export.replace("(adata,", f"({name},"), ns)
    assert {p.name: _h5(p, "barcodes") for p in folder.glob("*.h5")} == order
    assert pd.read_csv(folder / "cty.csv")["x"].astype(str).tolist() == [
        truth[c] for c in cells]
    summary = _quiet(_run_cell, rescore, ns).set_index("method")
    assert summary.loc[ran, "status"].tolist() == ["scored"] * len(ran)
    assert summary.loc[ran, "ARI"].notna().all() and host == []


@pytest.mark.parametrize("key", ["diagonal_rna_atac", "diagonal_multi"])
def test_labels_later_on_a_diagonal_folder(key, host, tmp_path, monkeypatch):
    """Diagonal data whose cell types come after the run. The line of the
    recipe is refused on such a folder, and the folder keeps its files. The
    export cell of the tutorial, run again with its ``labels=`` line from
    objects in another row order, writes the label files of both sides on
    the rows of the matrix files, and the saved embeddings are scored."""
    _says("recipe-no-labels", "For cross, mosaic and diagonal data it does not apply: run the "
                              "export cell of your tutorial again with its `labels=` line, "
                              "and then the `rescore` line above. The files keep their cell "
                              "order. Diagonal data needs the column in both files.")
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    ns = _namespace(key)
    folder, order, ran = _quiet(_unlabelled_run, key, ns, host)
    assert ran == sorted(ns["METHODS"])
    before = _bytes(folder)
    names = _objects(ns)
    rna, atac = names[0], next(n for n in names if n.startswith("atac"))
    export_cell = _sources(key)[EXPORT]
    export, rescore = _later()

    # the line of the recipe, with the RNA object of the tutorial for adata
    with pytest.raises(FileExistsError, match=r" with other cells\. To add labels, run the "
                                              r"export call that wrote the folder again with "
                                              r"labels= and overwrite=True\.$"):
        _run_cell(export.replace("(adata,", f"({rna},"), dict(ns))
    assert _bytes(folder) == before
    # "Diagonal data needs the column in both files."
    one_side = dict(ns)
    one_side[atac] = ns[atac].copy()
    del one_side[atac].obs["celltype"]
    with pytest.raises(KeyError, match=r"Without cell types for the ATAC cells, leave labels= "
                                       r"out\.\"?$"):
        _run_cell(export_cell, one_side)
    assert _bytes(folder) == before

    # the export cell again, by which time the objects are in another row order
    truth = _shuffled(ns, names)
    _quiet(_run_cell, export_cell, ns)
    assert {p.name: _h5(p, "barcodes") for p in folder.glob("*.h5")} == order
    for matrix, cells in order.items():
        side, batch = re.fullmatch(r"(rna|atac)(?:_gas)?(\d*)\.h5", matrix).groups()
        labels = pd.read_csv(folder / f"{side}_cty{batch}.csv")["x"].astype(str).tolist()
        assert labels == [truth[side + batch][c] for c in cells], matrix
    assert len(list(folder.glob("*.csv"))) == len(order)
    summary = _quiet(_run_cell, rescore, ns)
    assert dict(zip(summary["method"], summary["status"])) == {m: "scored" for m in ran}
    assert summary["ARI"].notna().all()
    assert host == []                            # no method ran again


@pytest.mark.parametrize("key,batches", [("cross_rna_adt", 3), ("mosaic_shared", 3)])
def test_labels_later_on_a_folder_with_batches(key, batches, host, tmp_path, monkeypatch):
    """Cross and mosaic data whose cell types come after the run: the export
    cell of the tutorial again, with its ``labels=`` line. From objects in
    another row order it writes one label file per batch on the rows of the
    files of that batch, and the saved embeddings are scored over the
    batches."""
    _says("recipe-no-labels", "For cross, mosaic and diagonal data it does not apply: run the "
                              "export cell of your tutorial again with its `labels=` line, "
                              "and then the `rescore` line above. The files keep their cell "
                              "order.")
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    ns = _namespace(key)
    folder, order, ran = _quiet(_unlabelled_run, key, ns, host)
    assert ran == sorted(ns["METHODS"])
    assert {int(re.search(r"\d+$", name[:-len(".h5")]).group()) for name in order} \
        == set(range(1, batches + 1))
    names = _objects(ns)
    truth = _shuffled(ns, names)
    _quiet(_run_cell, _sources(key)[EXPORT], ns)
    assert {p.name: _h5(p, "barcodes") for p in folder.glob("*.h5")} == order
    assert sorted(p.name for p in folder.glob("*.csv")) == [
        f"cty{b}.csv" for b in range(1, batches + 1)]
    for matrix, cells in order.items():
        # the object of the file: the one of its name, or the one object of all batches
        stem = matrix[:-len(".h5")]
        source = truth[stem] if stem in truth else truth[names[0]]
        assert (stem in truth) == (len(names) > 1)
        batch = re.search(r"\d+$", stem).group()
        labels = pd.read_csv(folder / f"cty{batch}.csv")["x"].astype(str).tolist()
        assert labels == [source[c] for c in cells], matrix
    summary = _quiet(_run_cell, _later()[1], ns)
    assert dict(zip(summary["method"], summary["status"])) == {m: "scored" for m in ran}
    assert summary["ARI"].notna().all()
    assert {r["n_batches"] for r in ns["mine"].records} == {batches}
    assert host == []                            # no method ran again


# ---------------------------------------------------- another number of donors
def test_donor_recipe(host, tmp_path, monkeypatch, capsys):
    """Five donors, treated as a vertical task: one folder with all cells,
    and the donor of each cell given to ``run_all``. The recipe runs in each
    vertical tutorial, with the methods of that tutorial. The line that keeps
    names of the file are printed by a block of its own, before the block that
    keeps some donors for the cross tutorial."""
    adata = _cite(150, donors=DONORS)
    run, show, keep_three = _fences("recipe-donors")
    # the batch metrics that the recipe names
    metrics = list(_quoted("recipe-donors", r"The summary then has the batch metrics `(\w+)`, "
                                            r"`(\w+)`, `(\w+)`, `(\w+)` and `(\w+)`, which "
                                            r"say how well the donors mix\."))
    assert metrics == ["ASW_batch", "ARI_batch", "NMI_batch", "GC", "iLISI"]
    _says("recipe-donors", "The methods do not read `batch=`.")
    for key in VERTICAL:
        (tmp_path / key).mkdir()
        monkeypatch.chdir(tmp_path / key)
        ns = {"mtb": mtb, "METHODS": list(GEN.TASKS[key]["methods"]), "adata": adata}
        host.clear()
        summary = _quiet(_run_cell, run, ns)
        assert sorted(p.name for p in Path("mydata/MYDONORS").iterdir()) == [
            "adt.h5", "cty.csv", "rna.h5"]                       # "no batch= here"
        assert sorted(summary["method"]) == sorted(ns["METHODS"])
        assert set(summary["status"]) == {"scored"}, key
        # they are the batch metrics of the summary, with a value for each method
        batch = set(mtb.catalog.metrics().query("family == 'batch'").metric)
        assert [c for c in summary.columns if c in batch] == metrics
        assert summary[metrics].notna().all().all()
        assert summary[metrics].notna().all().all()
        # the methods were given the files of the folder, and no donor
        assert sorted(m for m, _ in host) == sorted(ns["METHODS"])
        for _m, files in host:
            assert {f.name for f in files.values()} <= {"rna.h5", "adt.h5", "cty.csv"}

    # the column itself, not its name; matched to the cells by barcode
    _says("recipe-donors", "`batch=` takes the column itself, not its name. `run_all` reads "
                           "the folder, and the folder holds no `obs` columns. The values are "
                           "matched to the cells by barcode. Every cell needs a donor: "
                           "`run_all` stops when one is missing.")
    column = 'batch=adata.obs["donor"]'
    assert run.count(column) == 1
    host.clear()
    for name in ('batch="donor"', 'batch="obs:donor"'):
        with pytest.raises(FileNotFoundError, match="Pass the column itself"):
            _run_cell(run.replace(column, name), dict(ns))
    assert host == []
    # the donors in another row order give the same batch metrics; taken row
    # by row, without the barcodes, they give others
    mixed = 'batch=adata.obs["donor"].sample(frac=1, random_state=3)'
    by_barcode = _quiet(_run_cell, run.replace(column, mixed), dict(ns))
    assert by_barcode[metrics].equals(summary[metrics])
    by_row = _quiet(_run_cell, run.replace(column, mixed + ".to_numpy()"), dict(ns))
    assert not by_row[metrics].equals(summary[metrics])

    # the cross method that the recipe names does not run on vertical data
    monkeypatch.chdir(tmp_path)
    named = _quoted("recipe-donors", r"(\w+), a cross method, does not run on vertical data\.")
    assert named == "StabMap"
    cross = list(GEN.TASKS["cross_rna_adt"]["methods"])
    assert named in cross
    host.clear()
    with pytest.raises(ValueError) as e:
        _run_cell(run, {"mtb": mtb, "METHODS": cross, "adata": adata})
    assert str(e.value).startswith(f"{named} does not run on vertical data.")
    assert host == [] and not Path("out").exists()

    # "Every cell needs a donor: run_all stops when one is missing."
    gap = adata.copy()
    gap.obs["donor"] = gap.obs["donor"].astype(object)
    gap.obs.loc[gap.obs_names[:11], "donor"] = np.nan
    with pytest.raises(ValueError) as e:
        _run_cell(run, {"mtb": mtb, "METHODS": ["Matilda", "sciPENN"], "adata": gap})
    assert str(e.value) == (
        "batch: 11 of the 150 cells of MYDONORS have no batch. Give every cell a batch.")
    assert host == [] and not Path("out").exists()

    # "To use the cross tutorial instead, keep as many donors as its methods read."
    _says("recipe-donors", "To use the cross tutorial instead, keep as many donors as its "
                           "methods read.")
    # the page: the block that prints, the sentence that says what to write
    # into the list, the block that keeps the donors
    text = _recipe("recipe-donors")
    said = ("Write three of the printed names into the list of the next block, in place of "
            "`A`, `B` and `C`. Then run it.")
    _says("recipe-donors", "First print the donor names of your file. This block changes "
                           "nothing.")
    _says("recipe-donors", said)
    assert text.index("```python\n" + show) < text.index("Write three of the printed names") \
        < text.index("```python\n" + keep_three)
    assert show.splitlines() == [FENCES["recipe-donors"][1]]
    assert keep_three.splitlines() == [FENCES["recipe-donors"][2]]
    names = PLACEHOLDERS
    assert keep_three.count(names) == 1 and keep_three.rstrip().endswith("# <- three donors")
    assert json.loads(names) == re.findall(r"`(\w)`", said)
    assert not set(json.loads(names)) & set(DONORS)
    # in page order, as the reader runs them: print, read three of the
    # printed names, write them into the list, run the second block
    capsys.readouterr()
    ns = {"adata": adata}
    _run_cell(show, ns)
    assert ns["adata"] is adata and adata.n_obs == 150
    printed = re.findall(r"^(\w+) +30$", capsys.readouterr().out, flags=re.M)
    assert sorted(printed) == DONORS
    kept = printed[:3]
    _run_cell(keep_three.replace(names, json.dumps(kept)), ns)
    assert capsys.readouterr().out == ""
    assert sorted(set(ns["adata"].obs["donor"])) == sorted(kept)
    assert ns["adata"].obs["donor"].nunique() == 3 and ns["adata"].n_obs == 90
    # three is what the methods of the cross RNA + ADT tutorial read. For
    # ADT + ATAC the recipe says two, with two names in the list of that line
    reads = _cross_reads()
    assert set(reads[("rna", "adt")]) == {3, 2} and set(reads[("adt", "atac")]) == {2}
    _says("recipe-donors", "ADT + ATAC has no vertical task: for that data, keep two donors, "
                           "as shown further down, with two names in the list.")
    ns = {"adata": adata}
    _run_cell(keep_three.replace(names, json.dumps(DONORS[:2])), ns)
    assert ns["adata"].obs["donor"].nunique() == 2 and ns["adata"].n_obs == 60

    # what the recipe says about the names in the list: the line that it
    # quotes is the first sentence of the error for data without a cell
    stops = _quoted(
        "recipe-donors", r"Write the names of the donors you keep in the list, as printed\. "
                         r"When none of them is in the file, no cell is kept, and the export "
                         r"stops: `([^`]+)` When one name is wrong, the cells of the other "
                         r"donors are kept, and the check names the files of the batch that "
                         r"is not there\.")
    assert stops == ingest._NO_CELLS.split(". ", 1)[0] + "." and ingest._NO_CELLS == ZERO_CELLS
    # the export cell of the cross tutorial, with the donor column of the recipe
    export = _sources("cross_rna_adt")[EXPORT]
    assert export.count('"obs:batch"') == 1
    export = export.replace('"obs:batch"', '"obs:donor"')
    folder = Path("mydata") / NAMES["cross_rna_adt"]
    # none of the names is in the file, as when the second block is run with
    # the names of the page: no cell is kept, and the export stops
    ns = {"mtb": mtb, "METHODS": cross, "adata": adata}
    _run_cell(keep_three, ns)
    assert ns["adata"].n_obs == 0
    with pytest.raises(ValueError) as e:
        _run_cell(export, ns)
    assert str(e.value) == ZERO_CELLS
    assert not folder.exists()
    # one name is wrong: the cells of the other two donors are kept, and the
    # check names the files of the third batch
    ns = {"mtb": mtb, "METHODS": cross, "adata": adata}
    _run_cell(keep_three.replace(names, json.dumps(DONORS[:2] + ["X"])), ns)
    assert sorted(set(ns["adata"].obs["donor"])) == DONORS[:2] and ns["adata"].n_obs == 60
    _run_cell(export, ns)
    assert sorted(p.name for p in folder.iterdir()) == [
        "adt1.h5", "adt2.h5", "cty1.csv", "cty2.csv", "rna1.h5", "rna2.h5"]
    capsys.readouterr()
    table = _run_cell(_sources("cross_rna_adt")[CHECK], ns)
    lines = capsys.readouterr().out.splitlines()
    missing = "rna3.h5, adt3.h5 are missing."
    assert lines[2:] == [f"[scan] {m}: {missing}" for m in table["method"]]
    assert table["reason"].tolist() == [missing] * len(cross) and not table["runnable"].any()


def test_the_block_that_prints_the_donors_changes_nothing(capsys):
    """The block that prints the donor names is run before the reader knows
    them. It holds the print alone, and leaves ``adata`` as it was, so the
    block that keeps three donors starts from every cell."""
    _run, show, keep_three = _fences("recipe-donors")
    assert "adata =" not in show and "isin" not in show
    assert [type(node) for node in ast.parse(show).body] == [ast.Expr]
    adata = _cite(150, donors=DONORS)
    before = adata.copy()
    ns = {"adata": adata}
    for _ in (1, 2):
        _run_cell(show, ns)
        assert capsys.readouterr().out.splitlines()[1:-1] == [f"{d}    30" for d in DONORS]
    assert ns["adata"] is adata and list(ns) == ["adata", "__builtins__"]
    assert adata.shape == before.shape and list(adata.obs_names) == list(before.obs_names)
    assert adata.obs.equals(before.obs) and (adata.X == before.X).all()
    assert adata.obsm["protein"].equals(before.obsm["protein"])


def test_a_wrong_donor_name_is_undone_by_the_read_cell(tmp_path, monkeypatch, capsys):
    """The block that keeps three donors, run with a name that is not in the
    file, leaves fewer cells or none, and the corrected list then filters
    what is left. The recipe says so, and that the read cell of the tutorial
    brings the data back: after it, the block with the right names keeps
    three donors."""
    _says("recipe-donors", "When the list holds a name that is not in the file, `adata` ends "
                           "with fewer cells or with none, and a corrected list does not bring "
                           "them back. Run the read cell of the tutorial again to get the data "
                           "back, and then this block.")
    monkeypatch.chdir(tmp_path)
    _cite(150, donors=DONORS).write_h5ad("my_file.h5ad")
    read = _sources("cross_rna_adt")[READ]
    assert 'adata = ad.read_h5ad("my_file.h5ad")' in read
    _run, show, keep_three = _fences("recipe-donors")
    right = keep_three.replace(PLACEHOLDERS, json.dumps(DONORS[1:4]))
    for wrong, left in ((PLACEHOLDERS, 0), (json.dumps(DONORS[1:3] + ["X"]), 60)):
        ns = {"mtb": mtb}
        _quiet(_run_cell, read, ns)
        assert ns["adata"].n_obs == 150
        _run_cell(show, ns)
        _run_cell(keep_three.replace(PLACEHOLDERS, wrong), ns)
        assert ns["adata"].n_obs == left
        # the corrected list alone works on what is left
        _run_cell(right, ns)
        assert ns["adata"].n_obs == left and ns["adata"].obs["donor"].nunique() == left // 30
        # the read cell again, and then the block
        _quiet(_run_cell, read, ns)
        assert ns["adata"].n_obs == 150
        _run_cell(right, ns)
        assert sorted(set(ns["adata"].obs["donor"])) == DONORS[1:4] and ns["adata"].n_obs == 90
    capsys.readouterr()


def test_quoted_check_lines_are_live(host, tmp_path, monkeypatch, capsys):
    """The two lines that the recipe quotes from the check are what the check
    prints on a cross folder with five donors and on one with two."""
    monkeypatch.chdir(tmp_path)
    said = _said("recipe-donors")
    (more,) = re.findall(r"and the check says so: `([^`]+)`", said)
    (fewer,) = re.findall(r"names the files that are not there: `([^`]+)`", said)
    assert more == "StabMap reads batches 1-3 of 5. Batches 4-5 are not used."
    assert fewer == "rna3.h5, adt3.h5 are missing."
    adata = _cite(150, donors="ABCDE")
    methods = list(GEN.TASKS["cross_rna_adt"]["methods"])
    for name, donors in (("FIVE", "ABCDE"), ("TWO", "AB")):
        part = adata[adata.obs["donor"].isin(list(donors))].copy()
        mtb.io.export_dataset(part, "mydata/" + name, rna="X", adt="obsm:protein",
                              labels="obs:celltype", batch="obs:donor", category="cross")
    capsys.readouterr()
    table = mtb.scan("FIVE", "cross", methods=methods, data_path="mydata")
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("FIVE holds RNA and ADT, in batches 1-5.")
    assert f"[scan] {more}" in out and table["runnable"].all()
    table = mtb.scan("TWO", "cross", methods=methods, data_path="mydata")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == ("[scan] 0 of 2 methods have their input files in TWO. 0 of 2 can run on "
                      "this computer. TWO holds RNA and ADT, in batches 1-2.")
    assert f"[scan] StabMap: {fewer}" in out
    assert dict(zip(table["method"], table["reason"]))["StabMap"] == fewer
    assert not table["runnable"].any()


def test_batches_are_numbered_in_the_text_order_of_the_donor_names(tmp_path):
    """Donors 1, 2 and 10 become batches 1, 3 and 2."""
    assert ("The batches are numbered in the sorted order of the donor names, read as text, "
            "so `10` comes before `2`.") in _said("recipe-donors")
    adata = _cite(150)
    adata.obs["donor"] = np.repeat([1, 2, 10], 50)
    out = mtb.io.export_dataset(adata, tmp_path / "TEXT", rna="X", adt="obsm:protein",
                                labels="obs:celltype", batch="obs:donor", category="cross")
    donor_of = adata.obs["donor"].to_dict()
    assert [{donor_of[c] for c in _h5(out / f"rna{b}.h5", "barcodes")} for b in (1, 2, 3)] == [
        {1}, {10}, {2}]


def _cross_patterns():
    """``{modalities: {number of batches: methods}}`` of the batch patterns
    that ``describe_layout("cross")`` lists."""
    out = {}
    for line in mtb.describe_layout("cross").splitlines():
        m = re.match(r"  batch (\d+ = .+?): (.+?)(?: \(demo \w+\))?$", line)
        if m:
            batches = [b.split(" = ")[1] for b in m.group(1).split(", ")]
            assert len(set(batches)) == 1, line          # every batch, every modality
            out.setdefault(tuple(batches[0].split("+")), {})[len(batches)] = \
                set(m.group(2).split(", "))
    return out


def _cross_reads():
    """``{modalities: {number of batches: methods}}`` of the cross variants
    in the registry."""
    reads = {}
    for method in mtb.list_methods("cross"):
        for v in registry.get(method).variants:
            if v.when.get("category") != "cross":
                continue
            roles = v.when["modalities"]
            mods = tuple(dict.fromkeys(schema.base_modality(r) for r in roles))
            n = max(int(re.search(r"\d+$", r).group()) for r in roles)
            reads.setdefault(mods, {}).setdefault(n, set()).add(method)
    return reads


def test_batch_counts_follow_the_registry():
    """The recipe says how many batches a cross method reads, per pair of
    modalities, and that ``describe_layout("cross")`` lists them. Both
    follow the registry, and so do the methods of the four cross tutorials.
    UINMF's two-batch variant stays in the registry, but run_all leaves it
    out next to its three-batch one (0.4.0), so the page no longer names it."""
    assert ("A cross method reads a fixed number of batches: three for RNA + ADT and for "
            "RNA + ATAC, and two for ADT + ATAC and for RNA + ADT + ATAC. "
            "`mtb.describe_layout(\"cross\")` lists them.") in _said("recipe-donors")
    assert "UINMF" not in _said("recipe-donors")
    said = {("rna", "adt"): (3, {"UINMF": 2}), ("rna", "atac"): (3, {"UINMF": 2}),
            ("adt", "atac"): (2, {}), ("rna", "adt", "atac"): (2, {})}
    reads = _cross_reads()
    assert reads == _cross_patterns()
    assert set(reads) == set(said)
    for mods, (n, also) in said.items():
        assert set(reads[mods]) == {n, *also.values()}, mods
        for method, fewer in also.items():
            assert reads[mods][fewer] == {method} and method in reads[mods][n]
    for key, t in GEN.TASKS.items():
        if t["cat"] == "cross":
            mods = tuple(key.split("_")[1:])
            assert set(t["methods"]) <= reads[mods][said[mods][0]], key
    # "ADT + ATAC has no vertical task"
    assert "ADT + ATAC has no vertical task" in _said("recipe-donors")
    vertical = {frozenset(schema.base_modality(r) for r in v.when["modalities"])
                for m in mtb.list_methods("vertical") for v in registry.get(m).variants
                if v.when.get("category") == "vertical"}
    assert frozenset({"adt", "atac"}) not in vertical
    assert {frozenset(k.split("_")[1:]) for k in VERTICAL} == vertical


# ------------------------------------------------------------ the command line
def _commands():
    """The command lines of the collapsed block, one list of arguments each."""
    block = _recipe("recipe-donors").split('??? note "From the command line"', 1)[1]
    (code,) = re.findall(r"^    ```\n(.*?)^    ```", block, flags=re.S | re.M)
    return [shlex.split(c) for c in code.replace("\\\n", " ").splitlines()]


def test_the_command_lines_parse(host, tmp_path, monkeypatch, capsys):
    """The two commands of the block, on two files of other cells: the first
    writes a diagonal folder without label files, the second runs the methods
    on it without scores."""
    monkeypatch.chdir(tmp_path)
    convert, run_all = _commands()
    assert convert[:2] == ["multibench", "export"] and run_all[:2] == ["multibench", "run-all"]
    _says("recipe-donors", "Without cell types, leave `--labels` out and run without scores.")
    assert "--labels" not in convert and "--no-evaluate" in run_all
    parser = cli.build_parser()
    for argv in (convert, run_all):
        parser.parse_args(argv[1:])
    rna, atac = _cells(60), _cells(50, seed=1)
    atac.obs_names = [f"A{i:04d}-1" for i in range(50)]
    # "Save the objects first, for example with rna.write_h5ad("my_rna.h5ad")."
    assert '`rna.write_h5ad("my_rna.h5ad")`' in _said("recipe-donors")
    rna.write_h5ad("my_rna.h5ad")
    atac.write_h5ad("my_atac.h5ad")
    assert cli.main(convert[1:]) == 0
    folder = Path("mydata") / "MYDATA"
    assert sorted(p.name for p in folder.iterdir()) == ["atac_gas.h5", "rna.h5"]
    assert len(_h5(folder / "rna.h5", "barcodes")) == 60
    assert len(_h5(folder / "atac_gas.h5", "barcodes")) == 50
    capsys.readouterr()
    assert cli.main(run_all[1:]) == 0
    summary = pd.read_csv("out/MYDATA/summary.csv")
    assert len(host) == len(summary) > 0 and set(summary["status"]) == {"ran"}


def test_the_donor_file_of_the_command_line(host, tmp_path, monkeypatch, capsys):
    """The last sentence of the block: the file that ``--batch`` takes is
    the donor column written with its barcodes."""
    monkeypatch.chdir(tmp_path)
    write = 'adata.obs[["donor"]].to_csv("donor.csv")'
    assert f"`multibench run-all --batch donor.csv` takes the file that `{write}` writes." \
        in " ".join(_recipe("recipe-donors").split())
    adata = _cite(150, donors="ABCDE")
    mtb.io.export_dataset(adata, "mydata/MYDONORS", rna="X", adt="obsm:protein",
                          labels="obs:celltype")
    eval(write, {"adata": adata})
    argv = ["run-all", "MYDONORS", "--category", "vertical", "--data-path", "mydata",
            "--methods", "sciPENN", "--batch", "donor.csv"]
    cli.build_parser().parse_args(argv)
    assert _quiet(cli.main, argv + ["--out-dir", "out/MYDONORS"]) == 0
    summary = pd.read_csv("out/MYDONORS/summary.csv")
    assert summary["status"].tolist() == ["scored"]
    assert mtb.load_batch("out/MYDONORS").records[0]["n_batches"] == 5
    assert summary["ASW_batch"].notna().all()


# ------------------------------------------------- the two Details paragraphs
def test_details_name_the_filter_error_and_the_files_of_other_data(tmp_path):
    """The block "Details: files and labels" says what an export without a
    feature filter does on 10x data, and what it does next to files of other
    data. Both follow the package."""
    block = _read("tutorials/run.md").split('??? note "Details: files and labels"', 1)[1] \
        .split('??? note "Details: ATAC"', 1)[0]
    said = " ".join(block.split())
    assert ("Without a filter on such data, `export_dataset` raises `ValueError`. The recipe "
            "[A 10x `.h5` file](#recipe-10x-h5) reads the file and splits it.") in said
    assert ("An export also raises `FileExistsError` when the folder holds files of other "
            "data that it does not write: files with other cells, and after `batch=` files "
            "of other batch numbers. With `batch_index=`, a call checks its own batch "
            "number. The message names the first file. `overwrite=True` does not remove "
            "them. Write data with fewer batches into a new folder.") in said
    assert "n_batches=" not in said
    # the two filters of the paragraph select the two parts
    both = _cells(60, genes=50)
    both.var_names = GENES[2:32] + PEAKS[:20]
    both.var["feature_types"] = ["Gene Expression"] * 30 + ["Peaks"] * 20
    with pytest.raises(ValueError, match=r"^rna='X' selects features of 2 types, 'Gene Expression' and "
                                         r"'Peaks'\. Add a filter, as "
                                         r"rna='X\[feature_types=Gene Expression\]'\.$"):
        mtb.io.export_dataset(both, tmp_path / "ARC")
    assert not (tmp_path / "ARC").exists()
    filters = dict(re.findall(r'`(rna|atac)="(X\[feature_types=[^"]+\])"`', said))
    assert set(filters) == {"rna", "atac"}
    out = mtb.io.export_dataset(both, tmp_path / "ARC", atac_kind="peak", **filters)
    assert _h5(out / "rna.h5", "data") == (30, 60) and _h5(out / "atac.h5", "data") == (20, 60)
    # other cells into the same folder: refused with or without overwrite=True
    other = _cells(40, seed=5)
    other.obs_names = [f"other{i}" for i in range(40)]
    before = _bytes(out)
    for overwrite in (False, True):
        with pytest.raises(FileExistsError) as left:
            mtb.io.export_dataset(other, tmp_path / "ARC", overwrite=overwrite)
        assert str(left.value) == (f"{out} holds atac.h5 and 1 more file with other cells. "
                                   f"Export to a new folder.")
    assert _bytes(out) == before
