"""The tutorials after the round-1 student study (fix round f1).

The package changed underneath the notebooks (export_dataset writes diagonal
and per-batch mosaic folders itself, find_methods selects by the ATAC form a
method reads), and the prose contract was revised: facts that decide a
correct result are visible, the Details blocks carry plain paragraphs, and
the vocabulary is plain ('replacement' / 'stored outputs', not 'stand-in').
These tests pin both, on the committed notebooks (the generated four plus the
quickstart, and the hand-maintained end-to-end tutorial).
"""
import ast
import importlib.util
import io
import json
import re
import tokenize
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
CATS = list(GEN.SCEN)
ALL = [f"tutorial_{c}" for c in CATS] + ["colab_quickstart", "tutorial_end_to_end"]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _visible(md):
    """Markdown with the collapsed blocks removed: what a reader sees first."""
    return re.sub(r"<details>.*?</details>", "", md, flags=re.S)


def _details_paragraphs(md):
    """``[[paragraph, ...], ...]``: the paragraphs of each collapsed block."""
    blocks = re.findall(r"<details>\n<summary>[^<]*</summary>\n\n(.*?)\n\n</details>", md, re.S)
    return [[p for p in b.split("\n\n") if p.strip()] for b in blocks]


# ------------------------------------------------------------- vocabulary
RETIRED_WORDS = {
    r"(?i)stand[-_ ]in": "'stored outputs' / 'replacement' (revised contract)",
    r"TL;DR": "'Summary'",
    r"(?i)get going": "'Next steps'",
    r"\biff\b": "'only when'",
    r"(?i)tidy frame": "'long table'",
    r"from multibench\.engine|multibench\.engine\.": "a public call (method_info, scan) instead of an internal module",
}


@pytest.mark.parametrize("name", ALL)
def test_notebooks_use_plain_vocabulary_and_the_public_api(name):
    """Markdown, code and comments: a reader copies the code as well."""
    for kind, src in _cells(name):
        for pattern, fix in RETIRED_WORDS.items():
            hit = re.search(pattern, src)
            assert hit is None, f"{name}: {kind} cell says {hit.group(0)!r}: use {fix}"


@pytest.mark.parametrize("name", ALL)
def test_details_blocks_carry_plain_paragraphs(name):
    """No '**Label.** sentence' template: each paragraph of a collapsed block
    opens with a plain sentence (the contract allows bold lead-ins only in a
    block of 4 or more parallel entries; no block here needs them)."""
    md = "\n\n".join(src for kind, src in _cells(name) if kind == "markdown")
    blocks = _details_paragraphs(md)
    assert blocks, f"{name}: no collapsed block found"
    for paras in blocks:
        for p in paras:
            assert not p.lstrip().startswith("**"), f"{name}: bold lead-in label: {p[:70]!r}"


@pytest.mark.parametrize("name", ALL)
def test_no_trailing_code_comment_runs_past_80_columns(name):
    """A long explanation sits on its own comment line above the code, so a
    code box of the site's width does not cut it off."""
    ipy = pytest.importorskip("IPython.core.inputtransformer2")
    tm = ipy.TransformerManager()
    for kind, src in _cells(name):
        if kind != "code":
            continue
        lines = tm.transform_cell(src).splitlines()
        for tok in tokenize.generate_tokens(io.StringIO("\n".join(lines) + "\n").readline):
            if tok.type != tokenize.COMMENT:
                continue
            line = lines[tok.start[0] - 1]
            if line[:tok.start[1]].strip():                  # code before the comment
                assert len(line) <= 80, f"{name}: trailing comment past 80 columns: {line!r}"


# ------------------------------------------- facts that decide a correct result
@pytest.mark.parametrize("cat", CATS)
def test_correctness_facts_are_visible(cat):
    """Raw counts (section 5) and 'Methods run on Linux' (the title) are in
    the visible text, not only in a collapsed block."""
    cells = _cells(f"tutorial_{cat}")
    assert "Methods run on Linux." in _visible(cells[0][1])
    own = next(s for kind, s in cells if kind == "markdown" and s.startswith("## 5. Your own data"))
    assert re.search(r"Your data needs raw [A-Za-z ]*counts", _visible(own)), own


def test_diagonal_title_names_the_atac_form_of_each_method():
    """Which ATAC form a diagonal method reads is visible in the first cell,
    read from the registry: every peak-reading method is named there."""
    import multibench as mtb
    title = _visible(_cells("tutorial_diagonal")[0][1])
    # the lookup call: describe_layout lists each method under the ATAC files
    # it needs (round 7; method_info(m)["atac"] lists MultiMAP under peak)
    assert "gene-activity scores" in title and 'mtb.describe_layout("diagonal")' in title
    for m in mtb.find_methods("diagonal", atac="peak"):
        assert m in title, f"diagonal title does not name {m}, which reads peaks"


def test_mosaic_and_cross_titles_say_what_the_methods_read():
    mosaic = _visible(_cells("tutorial_mosaic")[0][1])
    assert "every mosaic method reads ATAC as peaks" in mosaic
    cross = _visible(_cells("tutorial_cross")[0][1])
    assert "Every cross method here reads RNA and ADT" in cross


# ------------------------------------------------ own-data demos (section 5)
EXPECTED_FILES = {
    "vertical": ["adt.h5", "cty.csv", "rna.h5"],
    "diagonal": ["atac_cty.csv", "atac_gas.h5", "rna.h5", "rna_cty.csv"],
    "mosaic": ["adt1.h5", "atac2.h5", "cty1.csv", "cty2.csv", "cty3.csv",
               "rna1.h5", "rna2.h5", "rna3.h5"],
    "cross": ["adt1.h5", "adt2.h5", "adt3.h5", "cty1.csv", "cty2.csv", "cty3.csv",
              "rna1.h5", "rna2.h5", "rna3.h5"],
}


def _own_data_cells(cat):
    """Section 5's code: the cell that builds the demo data, and the export
    calls of the cell that then runs run_all on the folder."""
    code = [src for kind, src in _cells(f"tutorial_{cat}") if kind == "code"]
    i = next(i for i, src in enumerate(code) if "mtb.io.export_dataset(" in src)
    export = code[i].split("\n\nmine = mtb.run_all(")[0]
    assert "run_all" not in export, code[i]
    return code[i - 1], export


@pytest.mark.parametrize("cat", CATS)
def test_own_data_demo_writes_the_folder_with_export_dataset(cat, root, tmp_path, monkeypatch):
    """Each demo writes its folder through export_dataset alone - diagonal
    with category='diagonal', mosaic one call per batch with batch_index= -
    with no hand-written file, no multibench warning, and a folder that scan
    reads for the tutorial's methods."""
    import multibench as mtb
    import pandas as pd
    data, export = _own_data_cells(cat)
    calls = [n for n in ast.walk(ast.parse(export)) if isinstance(n, ast.Call)]
    names = {ast.unparse(n.func) for n in calls}
    assert "mtb.io.to_canonical" not in names and not any(n.endswith("to_csv") for n in names), \
        f"{cat}: the demo writes files by hand instead of through export_dataset"
    exports = [n for n in calls if ast.unparse(n.func) == "mtb.io.export_dataset"]
    kws = [{k.arg: ast.unparse(k.value) for k in n.keywords} for n in exports]
    if cat == "diagonal":
        assert len(exports) == 1 and kws[0]["category"] == "'diagonal'" and "atac" in kws[0]
    if cat == "mosaic":
        assert [k.get("batch_index") for k in kws] == ["1", "2", "3"]
    monkeypatch.setattr(mtb.config.DEFAULT, "data_path", root / "data")
    monkeypatch.chdir(tmp_path)
    ns = {"mtb": mtb, "pd": pd}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(data, f"tutorial_{cat}", "exec"), ns)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        exec(compile(export, f"tutorial_{cat}", "exec"), ns)
    ours = [str(w.message) for w in seen if "multibench" in (w.filename or "")
            or issubclass(w.category, UserWarning)]
    assert not ours, f"{cat}: the demo warns: {ours}"
    own = GEN.OWN_NAME[cat]
    folder = tmp_path / "mydata" / own
    assert sorted(f.name for f in folder.iterdir()) == EXPECTED_FILES[cat]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = mtb.scan(own, cat, data_path=tmp_path / "mydata", verbose=False)
    ok = set(frame[frame.files_ok].method)
    assert set(GEN.SCEN[cat]["methods"]) <= ok
    if cat == "vertical":
        assert ok == set(mtb.find_methods("vertical", modalities=["rna", "adt"]))
    elif cat == "diagonal":
        assert ok == set(mtb.find_methods("diagonal", atac="gene_activity"))
    elif cat == "mosaic":
        assert ok == {"StabMap", "scMoMaT"}
    else:
        assert ok == set(frame.method) and len(ok) >= 7


# ------------------------------------------------------------- the generator
def test_generator_gives_stable_cell_ids():
    """Cell ids follow the notebook name and the cell position, so a
    regeneration that only changes text leaves every id alone."""
    import nbformat as nbf
    cells = [nbf.v4.new_markdown_cell("a"), nbf.v4.new_code_cell("b")]
    first = [c["id"] for c in GEN._notebook(cells, "tutorial_x").cells]
    again = [c["id"] for c in GEN._notebook([nbf.v4.new_markdown_cell("c"),
                                             nbf.v4.new_code_cell("d")], "tutorial_x").cells]
    assert first == again and len(set(first)) == 2
    for name in [f"tutorial_{c}" for c in CATS] + ["colab_quickstart"]:
        nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
        n = len(nb["cells"])
        expected = [c["id"] for c in GEN._notebook([nbf.v4.new_raw_cell("") for _ in range(n)],
                                                    name).cells]
        assert [c["id"] for c in nb["cells"]] == expected, f"{name}: regenerate with tools/gen_tut.py"
