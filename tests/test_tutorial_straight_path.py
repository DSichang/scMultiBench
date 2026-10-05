"""The tutorials are one straight path that installs the method environments
and runs the methods, on the demo data and on data exported from an AnnData.

The owner asked for tutorials that let a reader install the environments with
one call and run the methods themselves, also on a new dataset, in code that
is short and plain. These tests pin that shape: one shared install cell, no
flags, fallbacks or helper functions, the environment install and both
``run_all`` calls in every task tutorial, methods that read every batch of
their dataset, and download sizes that match the package's own plans. One
dataset serves every method of a task, and the ``_all`` notebook of the task
runs them all.
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
KEYS = list(GEN.TASKS)
TUTORIALS = [f"tutorial_{k}" for k in KEYS]
EVERY = [f"tutorial_{k}_all" for k in KEYS]        # every method of a task
RUNNING = TUTORIALS + EVERY + ["tutorial_end_to_end"]
ALL = RUNNING + ["colab_quickstart"]
CATEGORIES = ("vertical", "diagonal", "mosaic", "cross")
TIERS = ["fast", "medium", "slow", "very_slow"]


def _scan_kwargs(key):
    """The modalities the task's ``run_all`` passes (``GEN.run_kwargs``)."""
    return {"modalities": GEN.TASKS[key]["variants"][0]} if GEN.run_kwargs(key) else {}


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


@pytest.mark.parametrize("key", KEYS)
def test_task_tutorial_installs_and_runs(key):
    t = GEN.TASKS[key]
    cat, ds, extra = t["cat"], t["ds"], GEN.run_kwargs(key)
    code = "\n".join(_code(f"tutorial_{key}"))
    assert f"METHODS = {json.dumps(t['methods'])}" in code
    assert "mtb.env.install(METHODS, dry_run=False)" in code
    assert f'mtb.data.fetch("{ds}")' in code
    # one line or wrapped: the call is compared with its whitespace collapsed
    assert (f'res = mtb.run_all("{ds}", "{cat}", methods=METHODS, out_dir="out/{ds}"{extra})'
            in " ".join(code.split()))
    own = GEN.OWN[key]["name"]
    assert f'"mydata/{own}"' in code
    mine = [c for c in _code(f"tutorial_{key}") if c.startswith("mine = mtb.run_all(")]
    assert len(mine) == 1
    call = " ".join(mine[0].split())
    assert call.startswith(f'mine = mtb.run_all("{own}", "{cat}", methods=METHODS, '
                           f'data_path="mydata", out_dir="out/{own}"{extra})'), call
    assert code.index("mtb.env.install") < code.index("mtb.run_all")


@pytest.mark.parametrize("key", KEYS)
def test_all_methods_notebook_installs_and_runs_every_method(key):
    """The ``_all`` notebook: install, ``run_all`` and plot for every method
    of the task, on the task's dataset."""
    t = GEN.TASKS[key]
    cat, ds, extra = t["cat"], t["ds"], GEN.run_kwargs(key)
    cells = _code(f"tutorial_{key}_all")
    code = "\n".join(cells)
    assign = next(n for c in cells for n in ast.walk(_tree(c))
                  if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "METHODS")
    assert ast.literal_eval(assign.value) == GEN.task_methods(key)
    assert "mtb.env.install(METHODS, dry_run=False)" in code
    assert f'mtb.data.fetch("{ds}")' in code
    assert (f'res = mtb.run_all("{ds}", "{cat}", methods=METHODS, out_dir="out/{ds}_all"{extra})'
            in " ".join(code.split()))
    assert code.index("mtb.env.install") < code.index("mtb.run_all") < code.index("res.plot()")
    assert code.count("mtb.run_all(") == 1 and "export_dataset" not in code


def test_only_the_three_modality_vertical_task_names_its_modalities():
    """scMoMaT has an RNA + ADT variant that D22mini also satisfies, so that
    run names the modalities. No other task passes ``modalities=``."""
    assert {k for k in KEYS if GEN.run_kwargs(k)} == {"vertical_rna_adt_atac"}
    assert GEN.run_kwargs("vertical_rna_adt_atac") == ', modalities=["rna", "adt", "atac"]'


@pytest.mark.parametrize("key", KEYS)
def test_every_method_section_lists_the_task_and_links_the_all_page(key):
    """Markdown only: the section names every method of the task, says to set
    METHODS, and links the page that shows that run."""
    t = GEN.TASKS[key]
    cells = _cells(f"tutorial_{key}")
    i = next(i for i, (kind, src) in enumerate(cells)
             if kind == "markdown" and re.match(r"## \d\. Every method of this task\n", src))
    md = cells[i][1]
    everyone = GEN.task_methods(key)
    n = len(everyone)
    assert (f"{n} method{'s have' if n > 1 else ' has'} a variant for {t['cat']} "
            f"{t['label']}: {GEN.and_list(everyone)}. Each runs on `{t['ds']}`.") in md
    assert "set `METHODS` to that list in section 2" in md
    assert f"That downloads {GEN.env_size_text(methods=everyone)}." in md
    assert f"]({GEN.SITE}tutorials/{key}_all/)" in md
    assert (ROOT / "notebooks" / f"tutorial_{key}_all.ipynb").is_file()
    # markdown only: the next cell is Troubleshooting, not code
    assert cells[i + 1][0] == "markdown" and cells[i + 1][1].startswith("## Troubleshooting")
    assert cells[i + 2][1].startswith("## Next steps")


@pytest.mark.parametrize("key", KEYS)
def test_every_method_of_a_task_has_its_input_files_on_the_dataset(key):
    """One dataset serves the task: scan finds the input files of every
    method of the task, in the variant the task names."""
    import warnings
    import multibench as mtb
    t = GEN.TASKS[key]
    if not (mtb.config.DEFAULT.data_path / t["ds"]).is_dir():
        pytest.skip(f"{t['ds']} is not on disk")
    everyone = GEN.task_methods(key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan(t["ds"], t["cat"], methods=everyone, assume_gpu=True, verbose=False,
                      **_scan_kwargs(key))
    assert sorted(set(sc[sc.files_ok].method)) == everyone, \
        (key, sorted(set(everyone) - set(sc[sc.files_ok].method)))


def test_the_tasks_cover_every_method_of_the_four_categories():
    import multibench as mtb
    assert {t["cat"] for t in GEN.TASKS.values()} == set(CATEGORIES)
    for cat in CATEGORIES:
        covered = {m for k, t in GEN.TASKS.items() if t["cat"] == cat
                   for m in GEN.task_methods(k)}
        assert covered == set(mtb.list_methods(category=cat)), cat
    assert set(GEN.OWN) == set(GEN.TASKS)


def test_end_to_end_installs_and_runs_matilda():
    code = "\n".join(_code("tutorial_end_to_end"))
    assert 'mtb.env.install(["Matilda"], dry_run=False)' in code
    assert 'mtb.run("Matilda", "vertical"' in code
    assert "mtb.evaluate(emb" in code


@pytest.mark.parametrize("name", RUNNING)
def test_title_says_where_methods_run(name):
    assert "Methods run on Linux." in _cells(name)[0][1]


@pytest.mark.parametrize("name", TUTORIALS + ["tutorial_end_to_end"])
def test_title_links_its_own_colab_notebook(name):
    assert f"{GEN.COLAB}{name}.ipynb" in _cells(name)[0][1]


@pytest.mark.parametrize("key", KEYS)
def test_all_methods_title_names_the_methods_and_links_the_tutorial(key):
    """The ``_all`` title links the tutorial that explains the steps, and
    states the environments' size from the dry-run plans."""
    t = GEN.TASKS[key]
    first = _cells(f"tutorial_{key}_all")[0][1]
    everyone = GEN.task_methods(key)
    assert first.startswith(f"# {t['cat'].capitalize()} integration: {t['label']}, every method\n")
    assert f"on `{t['ds']}`: {GEN.and_list(everyone)}." in first
    assert f"[tutorial]({GEN.SITE}tutorials/{key}/)" in first
    assert f"runs {GEN.and_list(t['methods'])} only." in first
    assert f"The environments are {GEN.env_size_text(methods=everyone)}." in first


def test_quickstart_and_next_steps_link_every_task_tutorial():
    quick = _markdown("colab_quickstart")
    for key in KEYS:
        assert f"]({GEN.COLAB}tutorial_{key}.ipynb)" in quick, key
        last = _cells(f"tutorial_{key}")[-1][1]
        assert last.startswith("## Next steps")
        for other in KEYS:
            assert (f"]({GEN.SITE}tutorials/{other}/)" in last) == (other != key), (key, other)


def test_colab_quickstart_runs_nothing():
    code = "\n".join(_code("colab_quickstart"))
    for call in ("env.install", "run_all", "mtb.run(", "data.fetch"):
        assert call not in code


@pytest.mark.parametrize("key", KEYS)
def test_methods_read_every_batch(key):
    """A method that reads only some batches (UINMF on D52) would make the
    comparison unequal; the generator refuses, and so does this test."""
    import multibench as mtb
    t = GEN.TASKS[key]
    if not (mtb.config.DEFAULT.data_path / t["ds"]).is_dir():
        pytest.skip(f"{t['ds']} is not on disk")
    n = len(mtb.labels_for(t["ds"]))
    for m in t["methods"]:
        assert len(mtb.labels_for(t["ds"], t["cat"], m)) == n, m


@pytest.mark.parametrize("key", KEYS)
def test_methods_are_fast(key):
    """The default methods are tier fast or medium. Where a task has fewer
    fast or medium methods than the tutorial runs (mosaic RNA + ATAC has
    none, mosaic RNA + ADT has one method in all), the defaults are the
    fastest methods the task has."""
    import multibench as mtb
    rank = {m: TIERS.index(mtb.method_info(m)["runtime"]["tier"])
            for m in GEN.task_methods(key)}
    defaults = GEN.TASKS[key]["methods"]
    quick = [m for m, r in rank.items() if r <= TIERS.index("medium")]
    if len(quick) >= len(defaults):
        assert set(defaults) <= set(quick), {m: TIERS[rank[m]] for m in defaults}
    else:
        allowed = sorted(rank.values())[len(defaults) - 1]
        assert all(rank[m] <= allowed for m in defaults), {m: TIERS[r] for m, r in rank.items()}


@pytest.mark.parametrize("key", KEYS)
def test_download_sentence_matches_the_plans(key):
    t = GEN.TASKS[key]
    md = next(src for kind, src in _cells(f"tutorial_{key}")
              if kind == "markdown" and src.startswith("## 2. Download the data"))
    assert GEN.env_download_sentence(t["methods"]) in md
    assert f"`mtb.data.fetch` downloads `{t['ds']}` ({GEN.download_size([t['ds']])}) once." in md
    assert f"for {GEN.and_list(t['methods'])}, with no conda needed." in md


def test_end_to_end_download_sentence_matches_the_plan():
    assert GEN.env_download_sentence(["Matilda"]) in _markdown("tutorial_end_to_end")


def test_installation_page_sizes_match_the_plans():
    docs = os.environ.get("SCMULTIBENCH_DOCS")
    if not docs:
        pytest.skip("SCMULTIBENCH_DOCS not set")
    import multibench as mtb
    text = (Path(docs) / "installation.md").read_text()
    # one row per task, named as the tutorials' links name it ("vertical RNA + ADT")
    rows = {f"{t['cat']} {t['label']}": t["methods"] for t in GEN.TASKS.values()}
    rows["end-to-end"] = ["Matilda"]
    for label, methods in rows.items():
        gb = [GEN._gb(mtb.env.install(methods, flavor=f)) for f in ("gpu", "cpu")]
        sizes = gb[0] if gb[0] == gb[1] else f"{gb[0]}, {gb[1]}"
        line = f"- {label} {sizes}: {GEN.and_list(methods)}"
        assert line in text, line


def test_notebooks_match_the_generator(tmp_path, monkeypatch):
    """Regenerating writes the committed notebooks byte for byte."""
    import nbformat as nbf
    built = {"colab_quickstart": GEN.build_colab_quickstart}
    for key in KEYS:
        built[f"tutorial_{key}"] = lambda key=key: GEN.build_tutorial(key)
        built[f"tutorial_{key}_all"] = lambda key=key: GEN.build_all_methods(key)
    for name, build in built.items():
        fresh = nbf.writes(GEN._notebook(build(), name))
        assert fresh.strip() == (ROOT / "notebooks" / f"{name}.ipynb").read_text().strip(), name
    # nothing else is in the folder but the hand-maintained end-to-end tutorial
    assert sorted(p.stem for p in (ROOT / "notebooks").glob("*.ipynb")) == \
        sorted([*built, "tutorial_end_to_end"])


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
