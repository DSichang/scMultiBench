"""The tutorials after fix round 8 of the student study.

Round 8 split the ';' joins in the prose of the docs tutorial pages (R8-09)
and took ';' out of the package messages (R8-07). The notebook tutorials
render on the same site, so their prose and the lines their cells print
follow the same rule. Code and code comments are left as they are.

R8-04 makes ``scan`` and ``run_all`` raise ``ValueError`` when ``methods=``
names a method with no variant in the category, where they used to drop it.
The run cells of the category tutorials pass a fixed method list to both,
and the end-to-end notebook passes ``find_methods``' result to ``scan``. The
guards below check those lists against the live package, so a list that
would now raise fails here instead of in a Run all.
"""
import ast
import importlib.util
import json
import re
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
NOTEBOOKS = [f"tutorial_{c}" for c in CATS] + ["colab_quickstart", "tutorial_end_to_end"]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _prose(md):
    """Markdown without fenced code blocks and with each code span blanked."""
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    return re.sub(r"`[^`\n]*`", "`x`", md)


def _printed_strings(src):
    """The string parts of every ``print(...)`` argument in a code cell."""
    ipy = pytest.importorskip("IPython.core.inputtransformer2")
    tree = ast.parse(ipy.TransformerManager().transform_cell(src))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        out.append(sub.value)
    return out


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ------------------------------------------------ R8-09 style: no ';' joins
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_prose_joins_no_two_facts_with_a_semicolon(name):
    hits = [ln for kind, src in _cells(name) if kind == "markdown"
            for ln in _prose(src).splitlines() if ";" in ln]
    assert not hits, f"{name}: split these sentences at the ';': {hits}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_printed_lines_hold_no_semicolon(name):
    hits = [s for kind, src in _cells(name) if kind == "code"
            for s in _printed_strings(src) if ";" in s]
    assert not hits, f"{name}: a printed line joins two facts with ';': {hits}"


def test_split_sentences_keep_their_facts():
    """The sentences split at the ';' keep both halves (and the pinned
    phrases of earlier rounds)."""
    for cat in CATS:
        md = "\n".join(src for kind, src in _cells(f"tutorial_{cat}") if kind == "markdown")
        assert "Circle size shows the rank within a column. Bigger is better." in md
        assert ("Give raw counts for every modality, as in the demo data. The methods "
                "normalise the data themselves.") in md
        assert "ARI can fall slightly below 0. About 0 means a random clustering." in md
        assert (f"`multibench env plan --category {cat}` lists the size of each.") in md
        assert ("run it on a GPU machine. On a computer without one, "
                "`mtb.scan(..., assume_gpu=True)` checks everything else") in md
    cross = _cells("tutorial_cross")[0][1]
    assert ("Every cross method here reads RNA and ADT. For several 10x Multiome samples, "
            "use the vertical tutorial.") in cross
    mosaic = "\n".join(src for _, src in _cells("tutorial_mosaic"))
    assert ("writes one batch per call. Number the batches to match a pattern that "
            "`describe_layout` lists.") in mosaic
    e2e = "\n".join(src for _, src in _cells("tutorial_end_to_end"))
    assert "| mosaic | several batches that share only some modalities |" in e2e
    assert "`DegenerateRerunWarning`. Leave it out when ranking." in e2e
    assert "uniPort on D28 puts ATAC before RNA. `labels_for` with the method returns both orders." in e2e


# ------------------------------------------ R8-04 guards: the named methods
@pytest.mark.parametrize("cat", CATS)
def test_run_cells_name_only_methods_with_a_variant_on_their_dataset(cat):
    """Both run cells pass the tutorial's method list to scan and run_all:
    section 2 on the run dataset, section 3 on a subsample of own_src. Since
    R8-04 a listed method with no variant there raises ValueError, so every
    listed method must get a row."""
    import multibench as mtb
    s = GEN.SCEN[cat]
    trio = s["own_trio"]
    for ds in {s["live_ds"] or s["ds"], s["own_src"]}:
        if not (mtb.config.DEFAULT.data_path / ds).is_dir():
            pytest.skip(f"{ds} is not on disk")
        sc = _quiet(mtb.scan, ds, cat, methods=trio, verbose=False)
        assert set(sc.method) == set(trio), (cat, ds)
    # the check the guard protects against: an off-category name raises
    other = next(m for m in mtb.list_methods() if cat not in mtb.method_info(m)["categories"])
    with pytest.raises(ValueError, match=rf"{re.escape(other)} does not run on {cat} data"):
        _quiet(mtb.scan, s["own_src"], cat, methods=trio + [other], verbose=False)


def test_end_to_end_scenario_lists_raise_nothing_in_scan():
    """The end-to-end method-count cell passes find_methods' result for each
    category to scan without modalities=. Every method it names has a
    variant in the category, so scan returns a row for each of them."""
    import multibench as mtb
    src = next(s for kind, s in _cells("tutorial_end_to_end")
               if kind == "code" and "SCENARIOS = {" in s)
    ns = {}
    exec(src.split("\nfor cat, s in SCENARIOS.items():", 1)[0], ns)
    assert set(ns["SCENARIOS"]) == set(CATS)
    for cat, s in ns["SCENARIOS"].items():
        if not (mtb.config.DEFAULT.data_path / s["dataset"]).is_dir():
            continue
        got = mtb.find_methods(category=cat, modalities=[m.rstrip("123") for m in s["modalities"]])
        sc = _quiet(mtb.scan, s["dataset"], cat, methods=got, verbose=False)
        assert set(sc.method) == set(got), cat
