"""The tutorials are one straight path that installs the method environments
and runs the methods, on the demo data and on data exported from an AnnData.

The owner asked for tutorials that let a reader install the environments with
one call and run the methods themselves, also on a new dataset, in code that
is short and plain. These tests pin that shape: one shared install cell, no
flags, fallbacks or helper functions, the environment install and both
``run_all`` calls in every category tutorial, methods that read every batch of
their dataset, and download sizes that match the package's own plans.
"""
import ast
import importlib.util
import json
import os
import re
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
RUNNING = [f"tutorial_{c}" for c in CATS] + ["tutorial_end_to_end"]
ALL = RUNNING + ["colab_quickstart"]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _code(name):
    return [src for kind, src in _cells(name) if kind == "code"]


def _markdown(name):
    return "\n".join(src for kind, src in _cells(name) if kind == "markdown")


def _tree(src):
    ipy = pytest.importorskip("IPython.core.inputtransformer2")
    return ast.parse(ipy.TransformerManager().transform_cell(src))


@pytest.mark.parametrize("name", ALL)
def test_one_shared_install_cell(name):
    """The first code cell is the shared two-line install, and no other cell
    installs the package."""
    code = _code(name)
    assert code[0] == GEN.INSTALL_CELL
    assert not any("pip" in c for c in code[1:]), name
    assert "git+https" not in "\n".join(code)


def test_install_cell_keeps_numpy_and_pandas():
    """The pins are what keep Colab from upgrading its numpy and pandas (a
    plain install upgrades both, and the session must restart)."""
    assert "numpy=={numpy.__version__}" in GEN.INSTALL_CELL
    assert "pandas=={pandas.__version__}" in GEN.INSTALL_CELL


@pytest.mark.parametrize("name", ALL)
def test_no_flags_fallbacks_or_helpers(name):
    """No function definitions, no try/except, no if/else in any code cell."""
    for src in _code(name):
        for node in ast.walk(_tree(src)):
            assert not isinstance(node, (ast.FunctionDef, ast.Lambda, ast.Try, ast.If)), (
                name, type(node).__name__, src[:80])
    text = "\n".join(_code(name))
    for gone in ("INSTALL_ENVS", "stored_sweep", "replacement", "subsample_dataset",
                 "fetch_outputs", "env_ok"):
        assert gone not in text, (name, gone)


@pytest.mark.parametrize("name", ALL)
def test_code_cells_are_short(name):
    for src in _code(name):
        assert len(src.splitlines()) <= 15, (name, src[:80])
        assert max(len(line) for line in src.splitlines()) <= 90, (name, src[:80])


@pytest.mark.parametrize("cat", CATS)
def test_category_tutorial_installs_and_runs(cat):
    s = GEN.SCEN[cat]
    code = "\n".join(_code(f"tutorial_{cat}"))
    assert f"METHODS = {json.dumps(s['methods'])}" in code
    assert "mtb.env.install(METHODS, dry_run=False)" in code
    assert f'mtb.run_all("{s["ds"]}", "{cat}", methods=METHODS' in code
    own = GEN.OWN_NAME[cat]
    assert f'"mydata/{own}"' in code
    assert f'mtb.run_all("{own}", "{cat}", methods=METHODS, data_path="mydata"' in code
    assert code.index("mtb.env.install") < code.index("mtb.run_all")


def test_end_to_end_installs_and_runs_matilda():
    code = "\n".join(_code("tutorial_end_to_end"))
    assert 'mtb.env.install(["Matilda"], dry_run=False)' in code
    assert 'mtb.run("Matilda", "vertical"' in code
    assert "mtb.evaluate(emb" in code


@pytest.mark.parametrize("name", RUNNING)
def test_title_says_where_methods_run(name):
    first = _cells(name)[0][1]
    assert "Methods run on Linux." in first
    assert f"{GEN.COLAB}{name}.ipynb" in first


def test_colab_quickstart_runs_nothing():
    code = "\n".join(_code("colab_quickstart"))
    for call in ("env.install", "run_all", "mtb.run(", "data.fetch"):
        assert call not in code


@pytest.mark.parametrize("cat", CATS)
def test_methods_read_every_batch(cat):
    """A method that reads only some batches (UINMF on D52) would make the
    comparison unequal; the generator refuses, and so does this test."""
    import multibench as mtb
    s = GEN.SCEN[cat]
    if not (mtb.config.DEFAULT.data_path / s["ds"]).is_dir():
        pytest.skip(f"{s['ds']} not downloaded")
    n = len(mtb.labels_for(s["ds"]))
    for m in s["methods"]:
        assert len(mtb.labels_for(s["ds"], cat, m)) == n, m


@pytest.mark.parametrize("cat", CATS)
def test_methods_are_fast(cat):
    import multibench as mtb
    for m in GEN.SCEN[cat]["methods"]:
        assert mtb.method_info(m)["runtime"]["tier"] in ("fast", "medium"), m


@pytest.mark.parametrize("cat", CATS)
def test_download_sentence_matches_the_plans(cat):
    assert GEN.env_download_sentence(GEN.SCEN[cat]["methods"]) in _markdown(f"tutorial_{cat}")


def test_end_to_end_download_sentence_matches_the_plan():
    assert GEN.env_download_sentence(["Matilda"]) in _markdown("tutorial_end_to_end")


def test_installation_page_sizes_match_the_plans():
    docs = os.environ.get("SCMULTIBENCH_DOCS")
    if not docs:
        pytest.skip("SCMULTIBENCH_DOCS not set")
    import multibench as mtb
    text = (Path(docs) / "installation.md").read_text()
    rows = {c: s["methods"] for c, s in GEN.SCEN.items()}
    rows["end-to-end"] = ["Matilda"]
    for label, methods in rows.items():
        gb = [GEN._gb(mtb.env.install(methods, flavor=f)) for f in ("gpu", "cpu")]
        sizes = gb[0] if gb[0] == gb[1] else f"{gb[0]}, {gb[1]}"
        line = f"- {label} {sizes}: {GEN.and_list(methods)}"
        assert line in text, line


def test_notebooks_match_the_generator(tmp_path, monkeypatch):
    """Regenerating writes the committed notebooks byte for byte."""
    import nbformat as nbf
    for cat, s in GEN.SCEN.items():
        name = f"tutorial_{cat}"
        fresh = nbf.writes(GEN._notebook(GEN.build_tutorial(cat, s), name))
        assert fresh.strip() == (ROOT / "notebooks" / f"{name}.ipynb").read_text().strip(), name
    fresh = nbf.writes(GEN._notebook(GEN.build_colab_quickstart(), "colab_quickstart"))
    assert fresh.strip() == (ROOT / "notebooks" / "colab_quickstart.ipynb").read_text().strip()


def test_visible_text_stays_short():
    """Layer-1 prose (outside <details>) of each section stays near the
    two-layer budget; the collapsed blocks carry the rest."""
    for name in RUNNING:
        for kind, src in _cells(name):
            if kind != "markdown":
                continue
            visible = re.sub(r"<details>.*?</details>", "", src, flags=re.S)
            visible = re.sub(r"```.*?```", "", visible, flags=re.S)
            visible = re.sub(r"^#.*$", "", visible, flags=re.M)
            visible = re.sub(r"\]\(https?://[^)]+\)", "]", visible)
            words = len(re.findall(r"[A-Za-z0-9_']+", visible))
            # the diagonal title carries the ATAC-form sentence, the longest cell
            assert words <= 125, (name, words, visible[:80])
