"""Section 5 of the task tutorials, "Your own data", starts from the reader's files.

Since 0.3.5 the section reads ``.h5ad`` files. So that the notebook runs as
it is, ``mtb.data.write_example`` first writes example files under the names
the read cell reads. A reader changes the names marked ``# <-``: the file
names in the read cell, and in the export cell the folder name ``NAME`` and
the names of the parts of the file. The check cell and the run cell take the
folder from ``NAME``.

These tests read the committed notebooks and run the cells of the section.
The folder the cells write is the one the 0.3.4 cells wrote
(``tests/_own_cells_034.py``), and the check and the run act on the folder
that ``NAME`` names. No method runs: the environments are pinned as present,
and ``_run`` returns a random embedding.
"""
import ast
import hashlib
import importlib.util
import json
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.data.examples import EXAMPLES
from multibench.engine import envs, registry

from _own_cells_034 import OLD

ad = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

ROOT = Path(__file__).resolve().parent.parent
ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())



def _ready(reason) -> bool:
    """A runnable row's reason: empty, or (0.4) the note that the method scripts
    are not fetched yet, which the test hosts lack."""
    return reason == "" or reason.startswith("The method scripts are not in ")

def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
KEYS = list(GEN.TASKS)
VERTICAL = [k for k in KEYS if GEN.TASKS[k]["cat"] == "vertical"]

#: the folder each tutorial writes. Three keep the name they had in 0.3.4,
#: where one tutorial each wrote that layout; ten are new, because 0.3.4
#: wrote several layouts under MYDIAG, MYMOSAIC and MYCROSS.
NAMES = {
    "vertical_rna_adt": "MYCITE", "vertical_rna_atac": "MYMULTIOME",
    "vertical_rna_adt_atac": "MYTRI",
    "diagonal_rna_atac": "MYDIAG_RNA_ATAC", "diagonal_multi": "MYDIAG_BATCHES",
    "mosaic_rna_adt": "MYMOSAIC_RNA_ADT", "mosaic_rna_atac": "MYMOSAIC_RNA_ATAC",
    "mosaic_shared": "MYMOSAIC_SHARED", "mosaic_unshared": "MYMOSAIC_UNSHARED",
    "cross_rna_adt": "MYCROSS_RNA_ADT", "cross_rna_atac": "MYCROSS_RNA_ATAC",
    "cross_adt_atac": "MYCROSS_ADT_ATAC", "cross_rna_adt_atac": "MYCROSS_TRI",
}
SHARED_IN_0_3_4 = {"MYDIAG", "MYMOSAIC", "MYCROSS"}
NAME_MARK = "# <- a new name for your own data"

KINDS = ["markdown", "markdown", "code", "markdown", "code", "markdown", "code",
         "markdown", "code", "markdown", "code", "code"]
# positions in the section, counted from 0
EXAMPLE, READ, EXPORT, CHECK, RUN = 2, 4, 6, 8, 10

RECIPES = "https://dsichang.github.io/scMultiBench/tutorials/run/#recipe-"
LINKS = {"10x-h5": "a 10x `.h5` file", "labels-csv": "labels in a CSV file",
         "no-labels": "no labels"}
#: the donors recipe is linked by what the reader of the category comes with:
#: a donor or sample column, or another number of batches than the methods read
DONORS_LINK = {"vertical": "several donors or samples", "cross": "another number of donors"}
LINKS_OF = {"vertical": ["10x-h5", "labels-csv", "no-labels", "donors"],
            "diagonal": ["labels-csv", "no-labels"],
            "mosaic": ["10x-h5", "labels-csv", "no-labels"],
            "cross": ["10x-h5", "labels-csv", "no-labels", "donors"]}

#: what a vertical tutorial says on donors: in the visible text of the first
#: cell, and in its collapsed block
BATCH_ADVICE = (
    "Several donors or samples in the file? Keep them in one folder, without `batch=` in "
    "`export_dataset`, and give `run_all` the column itself: "
    "`mtb.run_all(..., batch={obj}.obs[\"donor\"])`. The summary then has the scores for how "
    "well they mix.")
BATCH_ADVICE_COLLAPSED = (
    "The donors that `run_all` gets as `batch=` are matched to the cells by barcode, and "
    "every cell needs a donor. A column name alone is not enough there, because `run_all` "
    "reads the folder and the folder holds no `obs` columns.")


def _cells(key):
    nb = json.loads((ROOT / "notebooks" / f"tutorial_{key}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _section(key):
    """The cells of section 5: from its heading to the next heading."""
    cells = _cells(key)
    i = next(i for i, (kind, src) in enumerate(cells)
             if kind == "markdown" and src.startswith("## 5. Your own data"))
    j = next(j for j in range(i + 1, len(cells))
             if cells[j][0] == "markdown" and cells[j][1].startswith("## "))
    return cells[i:j]


def _sources(key):
    section = _section(key)
    assert [kind for kind, _ in section] == KINDS, key
    return [src for _, src in section]


def _visible(src):
    """A markdown cell without its collapsed block."""
    return re.sub(r"<details>.*?</details>", "", src, flags=re.S)


def _collapsed(src):
    """The collapsed block of a markdown cell that has one."""
    (block,) = re.findall(r"<details>.*?</details>", src, flags=re.S)
    return block


def _run_cell(src, ns):
    """Run a code cell as a notebook does, and return the value of its last
    expression, which a notebook displays."""
    tree = ast.parse(src)
    last = tree.body.pop() if isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "cell", "exec"), ns)
    if last is not None:
        return eval(compile(ast.Expression(last.value), "cell", "eval"), ns)


def _namespace(key):
    """What the notebook has defined when section 5 starts."""
    return {"mtb": mtb, "pd": pd, "METHODS": list(GEN.TASKS[key]["methods"])}


def _on_disk(key):
    ds = GEN.TASKS[key]["ds"]
    if not (config.DEFAULT.data_path / ds).is_dir():
        pytest.skip(f"{ds} is not on disk")
    return ds


def _renamed(export, key, name):
    """The export cell with another value of ``NAME``."""
    old = f'NAME = "{NAMES[key]}"'
    assert export.count(old) == 1 and export.count(NAMES[key]) == 1, export
    return export.replace(old, f'NAME = "{name}"')


def _bytes(folder):
    """``{file name: digest of its bytes}``."""
    return {p.name: hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).iterdir())}


def _content(folder):
    """``{file name: digest of its content}``: every dataset of a matrix file
    with its shape and type, the bytes of a label file. Two matrix files with
    the same content can differ in their bytes."""
    out = {}
    for p in sorted(Path(folder).iterdir()):
        h = hashlib.sha1()
        if p.suffix == ".h5":
            with h5py.File(p, "r") as f:
                names = []
                f.visit(lambda n: names.append(n) if isinstance(f[n], h5py.Dataset) else None)
                for n in sorted(names):
                    a = f[n][()]
                    a = a.astype(bytes) if getattr(a, "dtype", None) == object else a
                    h.update(f"{n} {np.shape(a)} {getattr(a, 'dtype', type(a))}".encode())
                    h.update(np.ascontiguousarray(a).tobytes())
        else:
            h.update(p.read_bytes())
        out[p.name] = h.hexdigest()
    return out


@pytest.fixture
def linux(monkeypatch, tmp_path):
    """A Linux host with every environment and a GPU; no method scripts on
    this machine, no ref. ``_run`` returns a random embedding with one row
    per labelled cell of the folder, and records the files it was given."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    calls = []

    class _Res:
        def __init__(self, out):
            self.output = out

    def fake_run(method, category, inputs, out_dir, params=None):
        files = {role: Path(str(f)) for role, f in inputs.items()}
        calls.append((method, files))
        folder = next(iter(files.values())).parent
        labels = mtb.labels_for(folder.name, category, method, data_path=folder.parent)
        n = sum(len(pd.read_csv(f)) for f in labels.values())
        return _Res(np.random.default_rng(0).normal(size=(n, 4)))
    monkeypatch.setattr(W, "_run", fake_run)
    return calls


# ------------------------------------------------------------- the twelve cells
@pytest.mark.parametrize("key", KEYS)
def test_section_5_has_the_twelve_cells(key):
    t = GEN.TASKS[key]
    section = _section(key)
    assert [kind for kind, _ in section] == KINDS
    src = [s for _, s in section]
    assert src[EXAMPLE] == f'mtb.data.write_example("{t["ds"]}")'
    # diagonal_multi lines the mark up with the marks of the lines below it
    first = src[EXPORT].splitlines()[0]
    assert re.sub(r" {4,}", "   ", first) == f'NAME = "{NAMES[key]}"   {NAME_MARK}'
    assert (first != f'NAME = "{NAMES[key]}"   {NAME_MARK}') == (key == "diagonal_multi")
    # 0.4: the folder path, as the export cell writes it; no data_path=
    assert src[CHECK] == f'mtb.scan("mydata/" + NAME, "{t["cat"]}", methods=METHODS)'
    assert src[RUN].startswith('mine = mtb.run_all("mydata/" + NAME,')
    assert src[RUN] == (f'mine = mtb.run_all("mydata/" + NAME, "{t["cat"]}", methods=METHODS,\n'
                        f'                   out_dir="out/" + NAME)\n'
                        f'mine.summary')
    assert src[11] == "mine.plot()"
    # the folder is named once, in NAME: no later cell spells it out
    for later in src[EXPORT + 1:]:
        assert NAMES[key] not in later
    assert src[EXPORT].count(NAMES[key]) == 1


def test_folder_names_are_unique_and_new():
    names = {key: re.match(r'NAME = "(\w+)"', _sources(key)[EXPORT]).group(1) for key in KEYS}
    assert names == NAMES == {key: GEN.OWN[key]["name"] for key in KEYS}
    assert len(set(names.values())) == 13
    assert not set(names.values()) & SHARED_IN_0_3_4
    # a name of 0.3.4 is kept only where one tutorial alone wrote that folder
    was = [OLD[key]["name"] for key in KEYS]
    for key in KEYS:
        assert (names[key] == OLD[key]["name"]) == (was.count(OLD[key]["name"]) == 1), key


@pytest.mark.parametrize("key", KEYS)
def test_recipe_links(key):
    """The first cell links the recipes that fit the category, in the visible
    text: a 10x file holds cells measured together, and the donors recipe is
    for the categories that take samples from several donors in one object."""
    cat = GEN.TASKS[key]["cat"]
    first = _sources(key)[0]
    visible = re.sub(r"<details>.*?</details>", "", first, flags=re.S)
    text = {**LINKS, "donors": DONORS_LINK.get(cat)}
    links = [f"[{text[a]}]({RECIPES}{a})" for a in LINKS_OF[cat]]
    assert (f"Starting from something else? See the recipes for "
            f"{GEN.and_list(links)}.") in visible
    assert re.findall(r"#(recipe-[a-z0-9-]+)", first) == [f"recipe-{a}" for a in LINKS_OF[cat]]
    assert re.findall(r"#recipe-[a-z0-9-]+", "\n".join(s for _, s in _cells(key))) == \
        [f"#recipe-{a}" for a in LINKS_OF[cat]]


def test_install_cell_asks_for_this_version():
    """The section calls what 0.3.4 does not have, so the install cell asks
    for this version or a later one. The requirement is in quotes: the line
    goes through a shell, where a bare ``>`` redirects the output."""
    assert f'"multibench-sc>={mtb.__version__}"' in GEN.INSTALL_CELL
    assert GEN.INSTALL_CELL.splitlines() == [
        "import numpy, pandas",
        'pins = f"numpy=={numpy.__version__} pandas=={pandas.__version__}"',
        f'%pip install -q "multibench-sc>={mtb.__version__}" {{pins}}']
    for name in [f"tutorial_{k}" for k in KEYS] + ["tutorial_end_to_end", "colab_quickstart"]:
        nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        assert code[0] == GEN.INSTALL_CELL, name


def test_the_install_cell_hands_pip_the_three_requirements():
    """IPython fills ``{pins}`` in and hands the line to a shell. The shell
    takes the quotes off and leaves the ``>=`` in the requirement."""
    import shlex
    shell = pytest.importorskip("IPython.core.interactiveshell")
    import numpy
    ip = shell.InteractiveShell.instance()
    seen = []
    ip.system = seen.append                  # nothing is installed
    assert ip.run_cell(GEN.INSTALL_CELL).success
    (line,) = seen
    assert shlex.split(line)[1:] == [
        "-m", "pip", "install", "-q", f"multibench-sc>={mtb.__version__}",
        f"numpy=={numpy.__version__}", f"pandas=={pd.__version__}"]


# ------------------------------------------------------ the marks a reader follows
@pytest.mark.parametrize("key", KEYS)
def test_the_read_cell_reads_what_the_example_cell_writes(key):
    src = _sources(key)
    files = re.findall(r'"(my_\w+\.h5ad)"', src[READ])
    assert sorted(files) == sorted(EXAMPLES[GEN.TASKS[key]["ds"]])
    assert GEN.OWN[key]["n_files"] == len(files)
    # the sentences count the files as the example does
    one = len(files) == 1
    assert ("with that name as it is." in src[1]) == one
    assert ("with one of those names as it is." in src[1]) == (not one)
    assert src[3].startswith("Read your file. " if one else "Read your files. ")
    assert ("A name that you leave is read from the example." in src[3]) == (not one)


@pytest.mark.parametrize("key", KEYS)
def test_names_to_change_are_marked(key):
    """Every name a reader has to change ends its line with ``# <-``."""
    cat = GEN.TASKS[key]["cat"]
    src = _sources(key)
    reads = [ln for ln in src[READ].splitlines() if "read_h5ad" in ln]
    assert len(reads) == len(EXAMPLES[GEN.TASKS[key]["ds"]])
    for ln in reads:
        # one file per line, and its name is the one thing to change
        assert re.fullmatch(r'\w+ = ad\.read_h5ad\("my_\w+\.h5ad"\) +# <- \S.*', ln), ln
    export = src[EXPORT].splitlines()
    assert NAME_MARK in export[0]
    own_line = [ln for ln in export if re.match(r'\s*(rna|adt|labels|batch)="[^"]*",', ln)]
    for ln in own_line:
        assert "# <- " in ln, ln
    if cat == "mosaic":
        # one call per line pair: the column is named once, in a variable
        assert 'celltype = "obs:celltype"   # <- the cell-type column of every file' in export
        assert not own_line
    else:
        assert any(ln.lstrip().startswith("labels=") for ln in own_line)
        assert any(ln.lstrip().startswith("rna=") for ln in own_line) == (key != "cross_adt_atac")
    kind = [ln for ln in export if re.match(r'\s*atac_kind="\w+",', ln)]
    if cat == "diagonal":
        # these methods read gene activity, and other diagonal methods read peaks
        assert len(kind) == 1 and kind[0].rstrip().endswith('# <- or "peak"')
    elif key in ("vertical_rna_atac", "vertical_rna_adt_atac", "cross_rna_atac",
                 "cross_adt_atac", "cross_rna_adt_atac"):
        # every method of the task reads peaks: nothing to change
        assert len(kind) == 1 and "# <-" not in kind[0]
        assert kind[0].rstrip().endswith("# the methods here read peaks")
    else:
        assert not kind
    # a mark stands only on a line that holds a name in quotes or a list of objects
    for ln in src[READ].splitlines() + export:
        if "# <-" in ln:
            code = ln.split("#")[0]
            assert '"' in code or re.match(r"\w+ = \[", code), ln


# ------------------------------------------------------------ the folder
@pytest.mark.parametrize("key", KEYS)
def test_the_three_cells_write_the_folder_of_0_3_4(key, tmp_path, monkeypatch, capsys):
    """The example cell, the read cell and the export cell write the folder
    that the data cell and the export cell of 0.3.4 wrote, file by file. The
    export cell shows no warning, and it runs a second time."""
    ds = _on_disk(key)
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    monkeypatch.chdir(tmp_path / "old")
    ns = {"mtb": mtb, "pd": pd}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(OLD[key]["data"], f"0.3.4 {key}", "exec"), ns)
        exec(compile(OLD[key]["export"], f"0.3.4 {key}", "exec"), ns)
    was = _content(Path("mydata") / OLD[key]["name"])
    assert was

    monkeypatch.chdir(tmp_path / "new")
    src, ns = _sources(key), _namespace(key)
    capsys.readouterr()
    _run_cell(src[EXAMPLE], ns)
    cap = capsys.readouterr()
    assert [ln.split(":")[0] for ln in cap.out.splitlines()] == [
        f"wrote {f}" for f in EXAMPLES[ds]] and cap.err == ""
    assert sorted(p.name for p in Path(".").iterdir()) == sorted(EXAMPLES[ds])
    _run_cell(src[READ], ns)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _run_cell(src[EXPORT], ns)
        first = _content(Path("mydata") / NAMES[key])
        _run_cell(src[EXPORT], ns)
    assert [f"{w.category.__name__}: {w.message}" for w in seen] == []
    assert capsys.readouterr().err == ""
    assert first == was
    assert _content(Path("mydata") / NAMES[key]) == was
    assert sorted(p.name for p in Path("mydata").iterdir()) == [NAMES[key]]


@pytest.mark.parametrize("key", KEYS)
def test_the_section_runs_twice_and_keeps_a_file_of_the_reader(key, tmp_path, monkeypatch,
                                                              capsys):
    """A reader saves their own file under a name of the example. The
    example cell leaves it byte for byte and says so, on the first run and on
    the second, and the three cells run both times."""
    ds = _on_disk(key)
    mine, *others = EXAMPLES[ds]
    mtb.data.write_example(ds, tmp_path / "side", quiet=True)
    work = tmp_path / "work"
    work.mkdir()
    # saved again with write_h5ad: the file carries no mark of an example
    ad.read_h5ad(tmp_path / "side" / mine).write_h5ad(work / mine)
    before = (work / mine).read_bytes()
    monkeypatch.chdir(work)
    src, ns = _sources(key), _namespace(key)
    kept = (f"{mine} is already here and is not an example file of multibench. It was left "
            f"as it is.")
    for run in (1, 2):
        capsys.readouterr()
        _run_cell(src[EXAMPLE], ns)
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == kept
        assert len(lines) == len(EXAMPLES[ds])
        for f, line in zip(others, lines[1:]):
            if run == 1:
                assert line.startswith(f"wrote {f}: "), line
            else:
                assert line == (f"{f} is already here (the example of {ds}). Delete the "
                                f"file to write it again.")
        assert (work / mine).read_bytes() == before
        _run_cell(src[READ], ns)
        _run_cell(src[EXPORT], ns)
        assert (work / mine).read_bytes() == before
    assert (work / "mydata" / NAMES[key]).is_dir()


# ------------------------------------------------------- NAME drives the cells
#: the folders that are scored for real below, one without batches and one
#: with. Elsewhere the scoring is a recorder: it takes 5 to 15 seconds on a
#: folder of a tutorial, and the cells under test do not differ by task.
SCORED = ("vertical_rna_adt", "cross_rna_adt")
#: what the first line of the check says each folder holds
HOLDS = {
    "vertical_rna_adt": "RNA and ADT", "vertical_rna_atac": "RNA and ATAC",
    "vertical_rna_adt_atac": "RNA, ADT and ATAC",
    "diagonal_rna_atac": "RNA and ATAC", "diagonal_multi": "RNA and ATAC, in batches 1-3",
    "mosaic_rna_adt": "RNA and ADT, in batches 1-3",
    "mosaic_rna_atac": "RNA and ATAC, in batches 1-3",
    "mosaic_shared": "RNA, ADT and ATAC, in batches 1-3",
    "mosaic_unshared": "RNA, ADT and ATAC, in batches 1-3",
    "cross_rna_adt": "RNA and ADT, in batches 1-3",
    "cross_rna_atac": "RNA and ATAC, in batches 1-3",
    "cross_adt_atac": "ADT and ATAC, in batches 1-2",
    "cross_rna_adt_atac": "RNA, ADT and ATAC, in batches 1-2",
}
#: the tutorials whose run still says that rows of other variants lack files,
#: with the number of rows: the folder holds the modalities of those variants
# diagonal_rna_atac: its one-batch folder lists no row for numbered batches (X80)
MORE_ROWS = {"diagonal_multi": 2, "mosaic_shared": 3,
             "mosaic_unshared": 6, "cross_rna_adt_atac": 4}


@pytest.mark.parametrize("key", KEYS)
def test_one_name_drives_the_three_cells(key, linux, tmp_path, monkeypatch, capsys):
    """With another value of ``NAME`` in the export cell, the export, the
    check and the run act on the new folder, and the folder written before
    stays as it was."""
    _on_disk(key)
    t = GEN.TASKS[key]
    scored = []
    if key not in SCORED:
        def record(rec, emb, dataset, category, data_path, variant, **kw):
            scored.append((dataset, str(data_path)))
            rec["status"] = "CHAIN_OK"
        monkeypatch.setattr(W, "_score_record", record)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    for i in (EXAMPLE, READ, EXPORT):
        _run_cell(src[i], ns)
    old, new = Path("mydata") / NAMES[key], Path("mydata") / "MINE"
    before = _bytes(old)
    capsys.readouterr()

    _run_cell(_renamed(src[EXPORT], key, "MINE"), ns)
    assert ns["NAME"] == "MINE"
    assert _content(new) == _content(old)
    table = _run_cell(src[CHECK], ns)
    out = capsys.readouterr().out.splitlines()
    # the two lines of the check and nothing else: what the folder holds, and
    # the cells of each matrix file
    assert out[0] == (f"[scan] 2 of 2 methods have their input files in MINE. 2 of 2 can run "
                      f"on this computer. MINE holds {HOLDS[key]}.")
    assert out[1].startswith("[scan] Cells per file: ") and len(out) == 2
    assert set(table.attrs["cells"]) == {p.name for p in new.glob("*.h5")}
    assert sorted(table["method"]) == sorted(t["methods"])
    assert table["runnable"].all() and table["reason"].map(_ready).all()

    _run_cell(src[RUN], ns)
    log = capsys.readouterr().out.splitlines()
    more = [ln for ln in log if "more rows" in ln]
    assert more == ([f"[run_all] {MORE_ROWS[key]} more rows need files this folder does not "
                     f"have. mtb.scan shows them."] if key in MORE_ROWS else [])
    mine = ns["mine"]
    assert (mine.dataset, mine.category) == ("MINE", t["cat"])
    assert Path(mine.out_dir).resolve() == Path("out/MINE").resolve()
    assert sorted(p.name for p in Path("out").iterdir()) == ["MINE"]
    # every file a method was given is a file of the new folder
    assert sorted(m for m, _ in linux) == sorted(t["methods"])
    for _m, files in linux:
        assert files and {f.parent.resolve() for f in files.values()} == {new.resolve()}
    status = dict(zip(mine.summary["method"], mine.summary["status"]))
    assert set(status) == set(t["methods"])
    # scMoMaT writes a graph and a UMAP, which the pretend run does not write
    assert set(status.values()) <= {"CHAIN_OK", "RUN_OK_NO_EMBEDDING"}, status
    if key in SCORED:
        assert set(status.values()) == {"CHAIN_OK"} and mine.summary["ARI"].notna().all()
        assert ("ASW_batch" in mine.summary) == (t["cat"] == "cross")
    else:
        assert scored and {d for d, _ in scored} == {"MINE"}
    assert _bytes(old) == before


@pytest.mark.parametrize("key,changed,cells", [
    ("diagonal_rna_atac", "my_rna.h5ad", "rna.h5 700, atac_gas.h5 2,700"),
    ("mosaic_rna_adt", "my_rna1.h5ad", "rna1.h5 700, rna2.h5 1,500, adt2.h5 1,500, adt3.h5 1,500"),
], ids=["diagonal_rna_atac", "mosaic_rna_adt"])
def test_a_file_name_left_unchanged_is_visible(key, changed, cells, linux, tmp_path,
                                               monkeypatch, capsys):
    """A reader changes one file name of the read cell and leaves the others.
    The export then writes the reader's cells next to cells of the example.
    The second line of the check gives the cells of each file, so the
    reader's number and the example's number stand side by side."""
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    _run_cell(src[EXAMPLE], ns)
    ad.read_h5ad(changed)[:700].copy().write_h5ad("theirs.h5ad")
    assert src[READ].count(f'"{changed}"') == 1
    _run_cell(src[READ].replace(f'"{changed}"', '"theirs.h5ad"'), ns)
    _run_cell(_renamed(src[EXPORT], key, "MINE"), ns)
    capsys.readouterr()
    table = _run_cell(src[CHECK], ns)
    out = capsys.readouterr().out.splitlines()
    assert out[1] == f"[scan] Cells per file: {cells}."
    assert sorted(set(table.attrs["cells"].values())) == [700, int(cells.rsplit(" ", 1)[1]
                                                               .replace(",", ""))]
    assert "A name that you leave is read from the example." in src[3]


@pytest.mark.parametrize("key,files", [
    ("diagonal_multi", "rna3.h5 and 3 more files of batch 3, above n_batches=2."),
    ("cross_rna_adt", "rna3.h5 and 2 more files of batch 3."),
])
def test_fewer_batches_than_the_folder_holds_are_refused(key, files, tmp_path, monkeypatch):
    """A reader with two batches leaves ``NAME`` as it is. The folder holds
    the three batches of the example, and a method would read the third as
    part of the reader's data. The export stops and names the files, as the
    sentence above the cell says. A new value of ``NAME`` avoids it."""
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    for i in (EXAMPLE, READ, EXPORT):
        _run_cell(src[i], ns)
    folder = Path("mydata") / NAMES[key]
    before = _bytes(folder)
    export = src[EXPORT]
    if key == "diagonal_multi":
        # the loop takes its batches from the two lists
        assert "`n_batches` says how many batches the data has." in src[5]
        for three, two in (("rna = [rna1, rna2, rna3]", "rna = [rna1, rna2]"),
                           ("atac = [atac1, atac2, atac3]", "atac = [atac1, atac2]")):
            assert export.count(three) == 1
            export = export.replace(three, two)
    else:
        assert ("When the folder holds batches that your data does not have, "
                "`export_dataset` stops and names the files. A new value of `NAME` avoids "
                "that.") in src[5]
        ns["adata"] = ns["adata"][ns["adata"].obs["batch"].astype(str) != "3"].copy()
    with pytest.raises(FileExistsError) as e:
        _run_cell(export, ns)
    assert str(e.value) == f"{folder} holds {files} Export to a new folder."
    assert _bytes(folder) == before
    _run_cell(_renamed(export, key, "MINE"), ns)
    assert not [p.name for p in (Path("mydata") / "MINE").iterdir() if "3" in p.name]
    assert _bytes(folder) == before


# ------------------------------------------------------ the batch sentence
@pytest.mark.parametrize("key", KEYS)
def test_the_batch_sentence_uses_a_variable_of_the_notebook(key):
    """The vertical tutorials say how to score samples or donors, with a call
    a reader can copy: the object in it is one the read cell assigns. The
    call is in the visible text, in a paragraph of its own before the
    recipes, and the collapsed block says how the values are matched. The
    other tutorials do not bring a donor column up."""
    src = _sources(key)
    everything = "\n".join(s for _, s in _cells(key))
    if key not in VERTICAL:
        assert '.obs["donor"]' not in everything
        return
    (obj,) = re.findall(r'batch=(\w+)\.obs\["donor"\]', everything)
    assert (f"\n\n{BATCH_ADVICE.format(obj=obj)}\n\nStarting from something else?"
            in _visible(src[0]))
    assert f"\n\n{BATCH_ADVICE_COLLAPSED}\n\n" in _collapsed(src[0])
    assert "Keep several samples or donors" not in everything
    assigned = [n.targets[0].id for n in ast.walk(ast.parse(src[READ]))
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)]
    assert obj in assigned
    # it is the object the export cell writes, whose obs the reader looks at
    assert f"mtb.io.export_dataset({obj}," in src[EXPORT]


@pytest.mark.parametrize("key", VERTICAL)
def test_the_documented_batch_call_works(key, linux, tmp_path, monkeypatch):
    """The call of the batch sentence, on the tutorial's folder. The example
    has no donor column, so the test adds one with two donors."""
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    (obj,) = re.findall(r'batch=(\w+)\.obs\["donor"\]', src[0])
    _run_cell(src[EXAMPLE], ns)
    _run_cell(src[READ], ns)
    assert "donor" not in ns[obj].obs
    ns[obj].obs["donor"] = np.where(np.arange(ns[obj].n_obs) % 2, "donor1", "donor2")
    _run_cell(src[EXPORT], ns)                 # without batch=, as the sentence says
    assert not list((Path("mydata") / NAMES[key]).glob("*[0-9].*"))
    call = (f'mtb.run_all("mydata/" + NAME, "vertical", methods=METHODS, '
            f'batch={obj}.obs["donor"], dry_run=True, verbose=False)')
    plan = eval(call, ns)
    assert set(plan["method"][plan["runnable"]]) == set(GEN.TASKS[key]["methods"])
    # "a column name alone is not enough there"
    with pytest.raises(FileNotFoundError, match="batch='donor' is not a file"):
        eval(call.replace(f'{obj}.obs["donor"]', '"donor"'), ns)
    # "every cell needs a donor"
    ns[obj].obs.loc[ns[obj].obs_names[0], "donor"] = np.nan
    with pytest.raises(ValueError, match=rf"batch: 1 of the [\d,]+ cells of {NAMES[key]} have "
                                         rf"no batch\."):
        eval(call, ns)


@pytest.mark.parametrize("key", VERTICAL)
def test_the_donor_sentence_is_visible_and_its_call_adds_the_batch_scores(
        key, linux, tmp_path, monkeypatch):
    """A reader whose file has a donor column finds what to do with it before
    the cells, and the link to the recipe names that case. The run cell with
    the argument of the sentence gives the scores that say how well the
    donors mix, as the sentence says. The run cell as it is gives none
    (``test_one_name_drives_the_three_cells``, on the RNA + ADT folder)."""
    from multibench.eval import scib as escib
    _on_disk(key)
    # the probe of scib's helper binary itself, not an answer in its cache
    monkeypatch.setattr(escib, "_lisi_helper_problem", escib._lisi_helper_problem.__wrapped__)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    visible = _visible(src[0])
    (arg,) = re.findall(r'`mtb\.run_all\(\.\.\., (batch=\w+\.obs\["donor"\])\)`', visible)
    obj = arg[len("batch="):].split(".")[0]
    assert BATCH_ADVICE.format(obj=obj) in visible
    assert visible.rstrip().endswith(f"and [several donors or samples]({RECIPES}donors).")
    _run_cell(src[EXAMPLE], ns)
    _run_cell(src[READ], ns)
    ns[obj].obs["donor"] = np.where(np.arange(ns[obj].n_obs) % 2, "donor1", "donor2")
    _run_cell(src[EXPORT], ns)                 # without batch=, as the sentence says
    # the "..." of the sentence are the arguments of the run cell
    last = '                   out_dir="out/" + NAME)\n'
    assert src[RUN].count(last) == 1 and "batch=" not in src[RUN]
    _run_cell(src[RUN].replace(last, last.replace(")\n", f",\n                   {arg})\n")), ns)
    summary = ns["mine"].summary
    assert sorted(summary["method"]) == sorted(GEN.TASKS[key]["methods"])
    assert set(summary["status"]) == {"CHAIN_OK"}
    assert summary[["ARI", "ASW_batch", "GC"]].notna().all().all()
    # iLISI is a number wherever scib's helper binary runs
    assert summary["iLISI"].notna().all() == (escib._lisi_helper_problem() is None)


def test_the_command_line_of_the_diagonal_tutorial_writes_the_same_folder(tmp_path, monkeypatch):
    """The collapsed block of the diagonal tutorial gives a command line and
    says that it writes the same folder from the two files."""
    import shlex
    import shutil
    from multibench import cli
    key = "diagonal_rna_atac"
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    assert "The command line writes the same folder from the two files:" in src[0]
    (command,) = re.findall(r"```\n(multibench convert .*?)\n```", src[0], re.S)
    argv = shlex.split(command.replace("\\\n", " "))
    assert argv[:4] == ["multibench", "convert", "my_rna.h5ad", f"mydata/{NAMES[key]}"]
    for i in (EXAMPLE, READ, EXPORT):
        _run_cell(src[i], ns)
    by_cell = _content(Path("mydata") / NAMES[key])
    shutil.rmtree("mydata")
    assert cli.main(argv[1:]) == 0
    assert _content(Path("mydata") / NAMES[key]) == by_cell


# ------------------------------------------------- the sentences above the cells
READ_ONE = ("Read your file. For your own data, change the file name marked `# <-`. On Colab, "
            "upload the file first: click the folder icon on the left, then the upload icon.")
READ_SEVERAL = ("Read your files. For your own data, change every file name marked `# <-`. A "
                "name that you leave is read from the example. On Colab, upload the files "
                "first: click the folder icon on the left, then the upload icon.")
READS_H5AD = (" The cell reads `.h5ad` files. A 10x `.h5` file is read another way: see the "
              "recipe for a 10x `.h5` file, linked at the top of this section.")

WRITE = ("Write the dataset folder. Change each name in quotes marked `# <-`: give `NAME` a new "
         "value, and write the names that your {files} for the other parts. The printout above "
         "lists the names in {printed}. Write a name listed after `obs:` with that prefix, as "
         "`\"obs:celltype\"`{obsm}.")
WRITE_OBSM = ", and one after `obsm:` as `\"obsm:protein\"`"
#: what the sentence adds where the cell writes batches, by category or by tutorial
WRITE_BATCHES = {
    "mosaic": " Each call writes one batch.",
    "diagonal_multi": (" The loop writes one RNA batch and one ATAC batch per call, and "
                       "`n_batches` says how many batches the data has."),
    "cross": (" When the folder holds batches that your data does not have, `export_dataset` "
              "stops and names the files. A new value of `NAME` avoids that."),
}

CHECK_TEXT = ("Check the folder before you run anything. `runnable` says whether a method can "
              "run on your data on this computer, and `reason` says why not. The first two "
              "lines above the table say what the folder holds and how many cells each file "
              "has. Compare them with your data.")
# Z21 (r3): the vertical methods read atac.h5 too, so no sentence on which reads which
CHECK_TWO_NAMES = " `atac.h5` and `atac_peak.h5` are one matrix under two names."
CHECK_FURTHER = (" A further line is about one method: why it cannot run, or what it does not "
                 "read or needs from your data.")
CHECK_CROSS = (" With fewer batches than the methods read, such a line names the files that "
               "are not there. With more, it says which batches are not used.")
#: the tutorials whose folder holds the ATAC peaks under two names
TWO_ATAC_NAMES = {"vertical_rna_atac", "vertical_rna_adt_atac"}

SMALL_DATASET_ROW = ("| a method fails on a small dataset | `mtb.run_all(..., dry_run=True)` "
                     "names what the method needs from the data's size in its `caveat` column |")
WRONG_NAME = ("A wrong name after `obsm:`, `layer:` or `obs:` raises an error that lists the "
              "names the file has.")


def _tenx_file(path, n_cells=30):
    """A file in the layout of Cell Ranger's ``filtered_feature_bc_matrix.h5``:
    features x barcodes, sparse, with a type per feature."""
    import scipy.sparse as sp
    types = ["Gene Expression"] * 30 + ["Peaks"] * 10
    m = sp.csc_matrix(np.random.default_rng(0).poisson(0.6, (len(types), n_cells))
                      .astype(np.int32))
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("barcodes", data=np.array([f"BC{i}-1" for i in range(n_cells)],
                                                   dtype="S"))
        for name in ("data", "indices", "indptr"):
            g.create_dataset(name, data=getattr(m, name))
        g.create_dataset("shape", data=np.array(m.shape, dtype=np.int32))
        ft = g.create_group("features")
        ft.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))
        ft.create_dataset("feature_type", data=np.array(types, dtype="S"))
        ft.create_dataset("genome", data=np.array(["GRCh38"] * len(types), dtype="S"))
        ft.create_dataset("id", data=np.array([f"ID{i}" for i in range(len(types))], dtype="S"))
        ft.create_dataset("name", data=np.array([f"F{i}" for i in range(len(types))], dtype="S"))


def test_the_read_sentence_sends_a_10x_file_to_its_recipe(tmp_path):
    """The read cell calls ``read_h5ad``, which does not read a 10x ``.h5``
    file. Where the first cell links the recipe for that file, the sentence
    above the read cell says what the cell reads and points to the link. The
    sentence itself links nothing: each recipe is linked once."""
    import scanpy as sc
    for key in KEYS:
        cat = GEN.TASKS[key]["cat"]
        src = _sources(key)
        tenx = "10x-h5" in LINKS_OF[cat]
        assert tenx == ("10x" in GEN.RECIPES_OF[cat]) == (cat != "diagonal")
        n = len(EXAMPLES[GEN.TASKS[key]["ds"]])
        assert src[READ - 1] == (READ_ONE if n == 1 else READ_SEVERAL) + (READS_H5AD if tenx
                                                                         else ""), key
        assert (f"[a 10x `.h5` file]({RECIPES}10x-h5)" in _visible(src[0])) == tenx
        assert "](" not in src[READ - 1]
        assert re.findall(r'read_h5ad\("\w+(\.\w+)"\)', src[READ]) == [".h5ad"] * n
    path = tmp_path / "filtered_feature_bc_matrix.h5"
    _tenx_file(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # scanpy reads the file as the recipe does, so it is a 10x file
        assert sc.read_10x_h5(path, gex_only=False).shape == (30, 40)
        # the class and the text of the error are anndata's
        with pytest.raises(Exception) as e:
            ad.read_h5ad(path)
    assert not isinstance(e.value, (FileNotFoundError, ImportError)), e.value


@pytest.mark.parametrize("key", KEYS)
def test_the_write_sentence_speaks_of_the_file_that_was_read(key, tmp_path, monkeypatch, capsys):
    """The sentence above the export cell says where the names to write are:
    in the printout of the read cell, which lists the names of the files
    that were read. It counts the files that the cell shows, and it names
    ``obsm:`` where the export cell takes a matrix from there. The names it
    quotes are those the printout shows for the example."""
    ds = _on_disk(key)
    assert "the names of the example" not in "\n".join(s for _, s in _cells(key))
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    _run_cell(src[EXAMPLE], ns)
    capsys.readouterr()
    shown = _run_cell(src[READ], ns)
    printout = capsys.readouterr().out + ("" if shown is None else repr(shown))
    n_read, n_shown = len(EXAMPLES[ds]), printout.count("AnnData object with n_obs")
    assert n_shown == min(n_read, 2)
    printed = ("your file" if n_read == 1 else "your files" if n_shown == n_read else
               "two of your files. The other files need the same names")
    obsm = '"obsm:' in src[EXPORT]
    assert src[EXPORT - 1] == WRITE.format(
        files="file uses" if n_read == 1 else "files use", printed=printed,
        obsm=WRITE_OBSM if obsm else "") + WRITE_BATCHES.get(
            key, WRITE_BATCHES.get(GEN.TASKS[key]["cat"], "")), key
    lines = [ln.strip() for ln in printout.splitlines()]
    listed = [ln for ln in lines if ln.startswith("obs: ")]
    assert len(listed) == n_shown and all(ln.startswith("obs: 'celltype'") for ln in listed)
    assert [ln for ln in lines if ln.startswith("obsm: ")] == (["obsm: 'protein'"] if obsm
                                                                else [])
    assert ('"obs:celltype"' in src[EXPORT]) and (obsm == ('"obsm:protein"' in src[EXPORT]))
    if key != "vertical_rna_adt":
        return
    # a file of the reader: the printout is that of the file, with its names
    rng = np.random.default_rng(0)
    theirs = ad.AnnData(X=rng.poisson(1.0, (50, 30)).astype(np.float32))
    theirs.obs["celltype.l1"] = rng.choice(["T", "B"], 50)
    theirs.obsm["protein_expression"] = pd.DataFrame(
        rng.poisson(3.0, (50, 6)).astype(np.float32), index=theirs.obs_names,
        columns=[f"CD{i}" for i in range(6)])
    theirs.write_h5ad("my_citeseq.h5ad")
    assert src[READ].count('"my_file.h5ad"') == 1
    shown = _run_cell(src[READ].replace('"my_file.h5ad"', '"my_citeseq.h5ad"'), ns)
    lines = [ln.strip() for ln in repr(shown).splitlines()]
    assert "obs: 'celltype.l1'" in lines and "obsm: 'protein_expression'" in lines
    # written as the sentence says, the two names give the folder
    export = _renamed(src[EXPORT], key, "THEIRS")
    for old, new in (('"obs:celltype"', '"obs:celltype.l1"'),
                     ('"obsm:protein"', '"obsm:protein_expression"')):
        assert export.count(old) == 1
        export = export.replace(old, new)
    _run_cell(export, ns)
    assert sorted(p.name for p in (Path("mydata") / "THEIRS").iterdir()) == [
        "adt.h5", "cty.csv", "rna.h5"]


def test_the_check_sentence_covers_the_further_lines(linux, tmp_path, monkeypatch, capsys):
    """The check prints two lines on the folder and then a line for each
    method that cannot run or that leaves part of the folder out. The
    sentence above the cell says what the first two are and what a further
    line is. In a cross tutorial it says what such a line holds for fewer
    and for more batches than the methods read, the two cases that the first
    cell of the section announces."""
    for key in KEYS:
        cat = GEN.TASKS[key]["cat"]
        assert _sources(key)[CHECK - 1] == (
            CHECK_TEXT + (CHECK_TWO_NAMES if key in TWO_ATAC_NAMES else "") + CHECK_FURTHER
            + (CHECK_CROSS if cat == "cross" else "")), key
        assert "The two lines above" not in "\n".join(s for _, s in _cells(key))
    key = "cross_rna_adt"
    _on_disk(key)
    assert CROSS_BATCHES[key].startswith("The methods of this task read three batches")
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    _run_cell(src[EXAMPLE], ns)
    _run_cell(src[READ], ns)
    every = ns["adata"]
    donor = every.obs["batch"].astype(str).to_numpy()
    assert sorted(set(donor)) == ["1", "2", "3"]

    # two donors: the methods lack the files of the third batch
    ns["adata"] = every[donor != "3"].copy()
    _run_cell(_renamed(src[EXPORT], key, "TWO"), ns)
    capsys.readouterr()
    table = _run_cell(src[CHECK], ns)
    n1, n2 = (format(int((donor == d).sum()), ",") for d in ("1", "2"))
    assert capsys.readouterr().out.splitlines() == [
        "[scan] 0 of 2 methods have their input files in TWO. 0 of 2 can run on this computer. "
        "TWO holds RNA and ADT, in batches 1-2.",
        f"[scan] Cells per file: rna1.h5 {n1}, adt1.h5 {n1}, rna2.h5 {n2}, adt2.h5 {n2}.",
        "[scan] StabMap: rna3.h5, adt3.h5 are missing.",
        "[scan] sciPENN: rna3.h5, adt3.h5 are missing."]
    assert table.drop(columns="modalities").to_dict("list") == {"method": ["StabMap", "sciPENN"], "runnable": [False] * 2,
                                     "reason": ["rna3.h5, adt3.h5 are missing."] * 2}

    # five donors: the methods run, and leave two batches out
    five = every.copy()
    donor = donor.copy()
    donor[3::5], donor[4::5] = "4", "5"
    five.obs["batch"] = donor
    ns["adata"] = five
    _run_cell(_renamed(src[EXPORT], key, "FIVE"), ns)
    capsys.readouterr()
    table = _run_cell(src[CHECK], ns)
    n = [format(int((donor == d).sum()), ",") for d in "1234"]
    assert capsys.readouterr().out.splitlines() == [
        "[scan] 2 of 2 methods have their input files in FIVE. 2 of 2 can run on this computer. "
        "FIVE holds RNA and ADT, in batches 1-5.",
        "[scan] Cells per file: " + ", ".join(f"rna{i}.h5 {c}, adt{i}.h5 {c}"
                                              for i, c in enumerate(n, 1)) + " and 2 more.",
        "[scan] StabMap reads batches 1-3 of 5. Batches 4-5 are not used.",
        "[scan] sciPENN reads batches 1-3 of 5. Batches 4-5 are not used."]
    assert table["reason"].map(_ready).all()
    assert table.drop(columns=["modalities", "reason"]).to_dict("list") == {
        "method": ["StabMap", "sciPENN"], "runnable": [True] * 2}


@pytest.mark.parametrize("key", KEYS)
def test_the_check_names_the_two_atac_files(key, tmp_path, monkeypatch, capsys):
    """Without a category, the export writes the ATAC peaks as ``atac.h5``
    and as ``atac_peak.h5``. The second line of the check lists both, under a
    sentence that asks the reader to compare the files with their data, so
    the sentence says that the two are one matrix. It is there exactly where
    the folder holds both."""
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    for i in (EXAMPLE, READ, EXPORT):
        _run_cell(src[i], ns)
    table = _run_cell(src[CHECK], ns)
    capsys.readouterr()
    cells = table.attrs["cells"]
    both = {"atac.h5", "atac_peak.h5"} <= set(cells)
    assert both == (key in TWO_ATAC_NAMES) == (CHECK_TWO_NAMES in src[CHECK - 1])
    assert ("atac_peak.h5" in src[CHECK - 1]) == both
    if both:
        content = _content(Path("mydata") / NAMES[key])
        assert cells["atac.h5"] == cells["atac_peak.h5"] > 0
        assert content["atac.h5"] == content["atac_peak.h5"]
        # the diagonal methods read atac_peak.h5; none reads a file named atac.h5
        roles = {r for m in mtb.list_methods(category="diagonal")
                 for v in mtb.method_info(m)["supports"] if v["category"] == "diagonal"
                 for r in v["modalities"] if r.startswith("atac")}
        assert "atac_peak" in roles and not [r for r in roles if re.fullmatch(r"atac\d*", r)]


def test_the_small_dataset_row_names_the_scan_that_has_the_column(tmp_path, monkeypatch,
                                                                  capsys):
    """The Troubleshooting row sends a reader to the ``caveat`` column of the
    full plan. The check cell of the tutorial has no such column, so the row
    names the call that has it: ``run_all(..., dry_run=True)``."""
    for key in KEYS:
        rows = [ln for _, s in _cells(key) for ln in s.splitlines()
                if ln.startswith("| a method fails on a small dataset |")]
        assert rows == [SMALL_DATASET_ROW], key
    key = "diagonal_rna_atac"
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    for i in (EXAMPLE, READ, EXPORT):
        _run_cell(src[i], ns)
    assert list(_run_cell(src[CHECK], ns).columns) == ["method", "modalities", "runnable",
                                                       "reason"]
    full = _run_cell(src[CHECK].replace("mtb.scan(", "mtb.run_all(").replace(
        ")", ", dry_run=True)"), ns)
    capsys.readouterr()
    needs = "online_iNMF needs about 5,000 cells in total. On fewer it stops with std::exception."
    assert needs in [v.size_hint for v in registry.get("online_iNMF").variants]
    row = full[(full["method"] == "online_iNMF") & (full["modalities"] == "rna+atac_gas")]
    # the example's 2,700 RNA + 2,700 ATAC cells are enough: no size note (X81)
    assert len(row) == 1 and needs not in row["caveat"].iloc[0]


def test_a_wrong_name_after_a_prefix_lists_the_names(tmp_path, monkeypatch):
    """The collapsed block says that a wrong name after one of the three
    prefixes raises an error that lists the names the file has. Each of the
    three does, and the refused call writes nothing."""
    for key in KEYS:
        everything = "\n".join(s for _, s in _cells(key))
        assert _collapsed(_sources(key)[0]).count(WRONG_NAME) == 1, key
        assert everything.count("A wrong name") == 1, key
    key = "vertical_rna_adt"
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    src, ns = _sources(key), _namespace(key)
    _run_cell(src[EXAMPLE], ns)
    _run_cell(src[READ], ns)
    ns["adata"].layers["counts"] = ns["adata"].X.copy()
    for old, new, message in [
            ('adt="obsm:protein"', 'adt="obsm:nope"',
             "obsm key 'nope' not found; available: ['protein']"),
            ('rna="X"', 'rna="layer:nope"', "layer 'nope' not found; available: ['counts']"),
            ('labels="obs:celltype"', 'labels="obs:nope"',
             "labels='obs:nope': column 'nope' not in obs: searched data.obs; available: "
             "['celltype']")]:
        assert src[EXPORT].count(old) == 1
        with pytest.raises(KeyError) as e:
            _run_cell(src[EXPORT].replace(old, new), ns)
        assert e.value.args[0] == message
        assert not Path("mydata").exists()


# ------------------------------------------------------ the generator's checks
CROSS_BATCHES = {
    "cross_rna_adt": "The methods of this task read three batches, and UINMF also reads two.",
    "cross_rna_atac": "The methods of this task read three batches, and UINMF also reads two.",
    "cross_adt_atac": "The methods of this task read two batches.",
    "cross_rna_adt_atac": "The methods of this task read two batches.",
}


def test_the_cross_sentence_counts_the_batches_from_the_registry(monkeypatch):
    """Each cross tutorial says how many batches its methods read, in the
    visible text: a reader with two or five donors sees the number before
    the cells. The generator counts the batches in the modalities of the
    task's variants."""
    assert set(CROSS_BATCHES) == {k for k in KEYS if GEN.TASKS[k]["cat"] == "cross"}
    for key in KEYS:
        first = _sources(key)[0]
        if key not in CROSS_BATCHES:
            assert "The methods of this task read" not in first
            continue
        sentence = (f"{CROSS_BATCHES[key]} The recipe for another number of donors covers "
                    f"the other cases.")
        assert GEN.cross_batches_sentence(key) == sentence and sentence in first
        # a paragraph of its own before the recipes, and not in the collapsed block
        assert f"\n\n{sentence}\n\nStarting from something else?" in _visible(first)
        assert "The methods of this task read" not in _collapsed(first)
        counts = {m: set() for m in GEN.task_methods(key)}
        for m, v in GEN.task_variants(key):
            counts[m].add(max(int(r[-1]) for r in v.when["modalities"]))
        most = max(max(c) for c in counts.values())
        assert all(most in c for c in counts.values())
        assert ("three" in sentence) == (most == 3)
        assert ("UINMF also reads two" in sentence) == (counts["UINMF"] == {2, 3})
        assert all(c == {most} for m, c in counts.items() if m != "UINMF")
    # a set of variants the sentence has no words for stops the generation
    variants = GEN.task_variants("cross_rna_adt")
    monkeypatch.setattr(GEN, "task_variants", lambda key: [
        (m, v) for m, v in variants if m != "UINMF" or len(v.when["modalities"]) == 4])
    with pytest.raises(SystemExit, match="reword cross_batches_sentence"):
        GEN.cross_batches_sentence("cross_rna_adt")


def test_check_task_checks_section_5(monkeypatch):
    seen = []
    monkeypatch.setattr(GEN, "check_own", seen.append)
    GEN.check_task("vertical_rna_adt")
    assert seen == ["vertical_rna_adt"]


def test_the_generator_refuses_a_section_whose_texts_do_not_fit(monkeypatch):
    """What the read cell reads, the number of files the sentences speak of
    and the folder name are each written in two places."""
    key = "vertical_rna_atac"
    own = GEN.OWN[key]
    monkeypatch.setitem(own, "read", own["read"].replace('"my_atac.h5ad"', '"my_peaks.h5ad"'))
    with pytest.raises(SystemExit) as e:
        GEN.check_own(key)
    assert str(e.value) == (
        "vertical_rna_atac: the read cell reads ['my_rna.h5ad', 'my_peaks.h5ad'] and n_files "
        "is 2, and write_example writes ['my_rna.h5ad', 'my_atac.h5ad'] for D27mini_vertical")
    monkeypatch.undo()
    monkeypatch.setitem(own, "n_files", 1)
    with pytest.raises(SystemExit, match="n_files is 1, and write_example writes"):
        GEN.check_own(key)
    monkeypatch.undo()
    monkeypatch.setitem(own, "example", own["example"].replace("two small", "three small"))
    with pytest.raises(SystemExit, match="writes 2 files for D27mini_vertical: reword"):
        GEN.check_own(key)
    monkeypatch.undo()
    monkeypatch.setitem(own, "name", "MYDATA")
    with pytest.raises(SystemExit, match='does not start with NAME = "MYDATA"'):
        GEN.check_own(key)


def test_the_generator_refuses_cells_that_warn_or_leave_a_method_without_files(
        monkeypatch, tmp_path):
    """The cells of section 5 run in a temporary folder before a notebook is
    written. A warning of the export stops the generation, and so does a
    folder that lacks an input file of a method of the notebook."""
    import os
    key = "vertical_rna_adt"
    _on_disk(key)
    monkeypatch.chdir(tmp_path)
    GEN.check_own(key)
    assert os.getcwd() == str(tmp_path.resolve()) and not list(tmp_path.iterdir())
    own = GEN.OWN[key]
    monkeypatch.setitem(own, "read", own["read"] + "\nadata.X = adata.X * 0.5")
    with pytest.raises(SystemExit, match="vertical_rna_adt: the cells of section 5 warn: .*"
                                         "whole numbers"):
        GEN.check_own(key)
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(GEN.TASKS[key], "methods", ["Matilda", "MIRA"])   # MIRA reads ATAC
    with pytest.raises(SystemExit) as e:
        GEN.check_own(key)
    assert str(e.value) == ("vertical_rna_adt: the folder section 5 writes lacks the input "
                            "files of MIRA")
    assert os.getcwd() == str(tmp_path.resolve()) and not list(tmp_path.iterdir())
