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
ALL_KEYS = [k for k in KEYS if GEN.has_all(k)]   # tasks with more methods than the defaults
EVERY = [f"tutorial_{k}_all" for k in KEYS if GEN.has_all(k)]        # every method of a task
RUNNING = TUTORIALS + EVERY + ["tutorial_end_to_end"]
ALL = RUNNING + ["colab_quickstart"]
CATEGORIES = ("vertical", "diagonal", "mosaic", "cross")
TIERS = ["fast", "medium", "slow", "very_slow"]


# The 13 integration tasks of the benchmark article, in its order, with the
# article's name of each: the tab labels of the site.
ARTICLE_TASKS = {
    "vertical_rna_adt": ("vertical", "RNA + ADT"),
    "vertical_rna_atac": ("vertical", "RNA + ATAC"),
    "vertical_rna_adt_atac": ("vertical", "RNA + ADT + ATAC"),
    "diagonal_rna_atac": ("diagonal", "[RNA, ATAC]"),
    "diagonal_multi": ("diagonal", "[multiple RNA, multiple ATAC]"),
    "mosaic_rna_adt": ("mosaic", "[RNA, RNA + ADT, ADT]"),
    "mosaic_rna_atac": ("mosaic", "[RNA, RNA + ATAC, ATAC]"),
    "mosaic_shared": ("mosaic", "Mixed, with shared modality"),
    "mosaic_unshared": ("mosaic", "Mixed, without shared modality"),
    "cross_rna_adt": ("cross", "Multiple RNA + ADT"),
    "cross_rna_atac": ("cross", "Multiple RNA + ATAC"),
    "cross_adt_atac": ("cross", "Multiple ADT + ATAC"),
    "cross_rna_adt_atac": ("cross", "Multiple RNA + ADT + ATAC"),
}
# the methods the article evaluates on a task and the package does not run
NOT_WRAPPED = {"vertical_rna_adt_atac": ["UINMF"], "diagonal_multi": ["Conos"],
               "mosaic_shared": ["UINMF", "Multigrate"], "mosaic_unshared": ["Multigrate"],
               "cross_rna_adt_atac": ["UINMF"]}


def _flat(text):
    """``text`` with every run of whitespace as one space: a call compares
    the same whether it sits on one line or is wrapped."""
    return " ".join(text.split())


def _would_run(key, methods, every=False, **where):
    """The rows ``run_all`` runs for ``methods`` with the notebook's
    arguments, on a computer that has every environment: the rows with their
    input files that nothing but the environment blocks, less the rows inside
    a larger one of the same method when the call names no modalities."""
    import warnings
    import multibench as mtb
    from multibench import workflow as W
    t = GEN.TASKS[key]
    args = GEN.run_args(key, every)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = mtb.run_all(where.pop("dataset", t["ds"]), t["cat"], methods=methods,
                           dry_run=True, verbose=False, assume_gpu=True, **args, **where)
    other = [str(r).replace(str(e), "").strip() for r, e in zip(plan.reason, plan.env_reason)]
    rows = plan[[bool(f) and o == "" for f, o in zip(plan.files_ok, other)]]
    if "modalities" not in args:
        rows = rows.drop(index=W._nested_rows(rows)[0])
    return rows


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
    assert (_flat(f'res = mtb.run_all("{ds}", "{cat}", methods=METHODS, '
                  f'out_dir="out/{ds}"{extra})') in _flat(code))
    own = GEN.OWN[key]["name"]
    assert f'"mydata/{own}"' in code
    mine = [c for c in _code(f"tutorial_{key}") if c.startswith("mine = mtb.run_all(")]
    assert len(mine) == 1
    call = _flat(mine[0])
    assert call.startswith(_flat(f'mine = mtb.run_all("{own}", "{cat}", methods=METHODS, '
                                 f'data_path="mydata", out_dir="out/{own}"{extra})')), call
    assert code.index("mtb.env.install") < code.index("mtb.run_all")


@pytest.mark.parametrize("key", ALL_KEYS)
def test_all_methods_notebook_installs_and_runs_every_method(key):
    """The ``_all`` notebook: install, ``run_all`` and plot for every method
    of the task, on the task's dataset."""
    t = GEN.TASKS[key]
    cat, ds, extra = t["cat"], t["ds"], GEN.run_kwargs(key, every=True)
    cells = _code(f"tutorial_{key}_all")
    code = "\n".join(cells)
    assign = next(n for c in cells for n in ast.walk(_tree(c))
                  if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "METHODS")
    assert ast.literal_eval(assign.value) == GEN.task_methods(key)
    assert "mtb.env.install(METHODS, dry_run=False)" in code
    assert f'mtb.data.fetch("{ds}")' in code
    assert (_flat(f'res = mtb.run_all("{ds}", "{cat}", methods=METHODS, '
                  f'out_dir="out/{ds}_all"{extra})') in _flat(code))
    assert code.index("mtb.env.install") < code.index("mtb.run_all") < code.index("res.plot()")
    assert code.count("mtb.run_all(") == 1 and "export_dataset" not in code


def test_only_the_three_modality_tasks_name_their_modalities():
    """The methods of the two tasks with RNA, ADT and ATAC have variants with
    fewer modalities that the task's dataset also satisfies, so those runs
    name the modalities. No other task passes ``modalities=``: the cross
    tasks with RNA + ADT and RNA + ATAC run UINMF's two-batch variant next to
    the three-batch variants of the other methods."""
    named = {"vertical_rna_adt_atac": ["rna", "adt", "atac"],
             "cross_rna_adt_atac": ["rna1", "rna2", "adt1", "adt2", "atac1", "atac2"]}
    for every in (False, True):
        assert {k: GEN.run_args(k, every)["modalities"] for k in KEYS
                if "modalities" in GEN.run_args(k, every)} == named
    assert _flat(GEN.run_kwargs("vertical_rna_adt_atac")) == \
        ', modalities=["rna", "adt", "atac"]'
    assert _flat(GEN.run_kwargs("cross_rna_adt_atac")) == \
        ', modalities=["rna1", "rna2", "adt1", "adt2", "atac1", "atac2"]'
    for key in KEYS:
        for name in [f"tutorial_{key}"] + [f"tutorial_{key}_all"] * GEN.has_all(key):
            assert ("modalities=" in "\n".join(_code(name))) == (key in named), name


def test_allow_atac_mismatch_is_passed_only_where_the_atac_check_stops_a_method():
    """Matilda (vertical RNA + ADT + ATAC), UnitedNet and scMDC (cross RNA +
    ATAC) are listed as reading gene activity, and the task's dataset holds
    peaks, as in the benchmark. Only the every-method notebooks of those two
    tasks pass the flag, and they say why in one sentence. No other notebook
    names the flag, in code or text."""
    blocked = {"vertical_rna_adt_atac": ["Matilda"], "cross_rna_atac": ["UnitedNet", "scMDC"]}
    assert {k: GEN.atac_blocked(k) for k in KEYS if GEN.atac_blocked(k)} == blocked
    flagged = {f"tutorial_{k}_all" for k in blocked}
    assert flagged <= set(EVERY)
    for name in ALL:
        text = "\n".join(src for _, src in _cells(name))
        assert ("allow_atac_mismatch" in text) == (name in flagged), name
    for key, names in blocked.items():
        t = GEN.TASKS[key]
        assert GEN.run_args(key).get("allow_atac_mismatch") is None
        assert GEN.run_args(key, every=True)["allow_atac_mismatch"] is True
        assert not set(names) & set(t["methods"]), "a default needs the flag"
        code = "\n".join(_code(f"tutorial_{key}_all"))
        assert code.count("allow_atac_mismatch=True") == 1
        md = _markdown(f"tutorial_{key}_all")
        it = "it" if len(names) == 1 else "them"
        assert (f"`allow_atac_mismatch=True` lets {GEN.and_list(names)} run. The package "
                f"lists {it} as reading gene-activity ATAC, and `{t['ds']}` holds peaks, as "
                f"in the benchmark.") in md
        # the tutorial points to that page for the argument, without naming it
        section = next(src for kind, src in _cells(f"tutorial_{key}")
                       if kind == "markdown" and "Every method of this task" in src)
        assert (f"{GEN.and_list(names)} run{'s' if len(names) == 1 else ''} only with one "
                f"more `run_all` argument, which that page shows.") in section


def test_the_generator_refuses_a_default_the_atac_check_stops(monkeypatch):
    monkeypatch.setitem(GEN.TASKS["vertical_rna_adt_atac"], "methods", ["Matilda", "scMoMaT"])
    with pytest.raises(SystemExit, match="Matilda needs allow_atac_mismatch on D22mini"):
        GEN.check_task("vertical_rna_adt_atac")


def test_the_generator_refuses_a_default_that_reads_only_some_batches(monkeypatch):
    """UINMF's script takes two batches, so it cannot be a default on the
    cross datasets with three."""
    for key in ("cross_rna_adt", "cross_rna_atac"):
        ds = GEN.TASKS[key]["ds"]
        monkeypatch.setitem(GEN.TASKS[key], "methods", ["StabMap", "UINMF"])
        with pytest.raises(SystemExit, match=f"UINMF reads only some batches of {ds}"):
            GEN.check_task(key)


# ------------------------------------------------- the article's 13 tasks
def test_the_tasks_are_the_13_of_the_article_with_its_labels():
    assert {k: (t["cat"], t["label"]) for k, t in GEN.TASKS.items()} == ARTICLE_TASKS
    assert KEYS == list(ARTICLE_TASKS)             # the article's order
    assert [sum(c == cat for c, _ in ARTICLE_TASKS.values()) for cat in CATEGORIES] == \
        [3, 2, 4, 4]
    for key, t in GEN.TASKS.items():
        assert key.startswith(t["cat"] + "_")
        first = _cells(f"tutorial_{key}")[0][1]
        assert first.startswith(f"# {t['cat'].capitalize()} integration: {t['label']}\n"), key
        assert f"`{t['ds']}`" in first
    assert len({t["ds"] for t in GEN.TASKS.values()}) == 13    # one dataset per task


def test_each_task_has_the_methods_the_registry_gives_its_modality_sets():
    """A method belongs to a task when the registry gives it a variant of the
    task's category with exactly one of the task's modality sets."""
    from multibench.engine import registry
    for key, t in GEN.TASKS.items():
        want = [frozenset(v) for v in t["variants"]]
        assert len(set(want)) == len(want), key
        methods = sorted(
            m for m in registry.list_methods()
            if any(v.when.get("category") == t["cat"]
                   and frozenset(v.when.get("modalities", [])) in want
                   for v in registry.get(m).variants))
        assert GEN.task_methods(key) == methods and methods, key
        assert set(t["methods"]) <= set(methods), key


def test_the_method_counts_of_the_tasks():
    assert {k: len(GEN.task_methods(k)) for k in KEYS} == {
        "vertical_rna_adt": 14, "vertical_rna_atac": 14, "vertical_rna_adt_atac": 4,
        "diagonal_rna_atac": 14, "diagonal_multi": 4,
        "mosaic_rna_adt": 3, "mosaic_rna_atac": 6, "mosaic_shared": 2, "mosaic_unshared": 2,
        "cross_rna_adt": 10, "cross_rna_atac": 8, "cross_adt_atac": 5,
        "cross_rna_adt_atac": 4}
    # the single-batch diagonal task holds none of the several-batch variants
    for _m, v in GEN.task_variants("diagonal_rna_atac"):
        assert not any(r[-1].isdigit() for r in v.when.get("modalities", []))


def test_every_method_variant_of_the_four_categories_is_in_exactly_one_task():
    """No variant unassigned and none in two tasks."""
    from multibench.engine import registry
    every = [(m, i) for m in registry.list_methods()
             for i, v in enumerate(registry.get(m).variants)
             if v.when.get("category") in CATEGORIES]
    assigned = []
    for key in KEYS:
        for m, v in GEN.task_variants(key):
            i = next(i for i, x in enumerate(registry.get(m).variants) if x is v)
            assigned.append((m, i))
    assert len(assigned) == len(set(assigned)), \
        sorted(x for x in set(assigned) if assigned.count(x) > 1)
    assert set(assigned) == set(every), sorted(set(every) ^ set(assigned))
    # within a task a method has one variant: one row of the run per method
    for key in KEYS:
        names = [m for m, _ in GEN.task_variants(key)]
        assert len(names) == len(set(names)), key


@pytest.mark.parametrize("key", KEYS)
def test_not_wrapped_sentence(key):
    """One sentence, in the tutorial's "Every method of this task" section,
    names the methods the article evaluates on the task and the package does
    not run. They are no methods of the task in the registry."""
    names = GEN.TASKS[key].get("not_wrapped", [])
    assert names == NOT_WRAPPED.get(key, [])
    assert not set(names) & set(GEN.task_methods(key))
    section = next(src for kind, src in _cells(f"tutorial_{key}")
                   if kind == "markdown" and "Every method of this task" in src)
    everything = "\n".join(src for _, src in _cells(f"tutorial_{key}"))
    if not names:
        assert "The article also evaluates" not in everything
        return
    if len(names) == 1:
        sentence = (f"The article also evaluates {names[0]} on this task. Its published "
                    "script does not take this layout, so the package does not run it.")
    else:
        sentence = (f"The article also evaluates {GEN.and_list(names)} on this task. Their "
                    "published scripts do not take this layout, so the package does not "
                    "run them.")
    assert sentence in section
    assert everything.count("The article also evaluates") == 1


@pytest.mark.parametrize("key", KEYS)
def test_two_batch_sentence(key):
    """UINMF's script takes two batches. On the cross datasets with three it
    reads batches 1 and 2, and the tutorial and the every-method notebook
    say so. No other method of a task reads fewer batches than its dataset
    holds."""
    import multibench as mtb
    t = GEN.TASKS[key]
    if not (mtb.config.DEFAULT.data_path / t["ds"]).is_dir():
        pytest.skip(f"{t['ds']} is not on disk")
    n = len(mtb.labels_for(t["ds"]))
    short = [m for m in GEN.task_methods(key)
             if len(mtb.labels_for(t["ds"], t["cat"], m)) < n]
    assert short == (["UINMF"] if key in ("cross_rna_adt", "cross_rna_atac") else []), key
    sentence = "UINMF's script takes two batches, so it reads batches 1 and 2."
    for name in [f"tutorial_{key}"] + [f"tutorial_{key}_all"] * GEN.has_all(key):
        assert (sentence in _markdown(name)) == bool(short), name
    if short:
        assert [p.stem if hasattr(p, "stem") else str(p)
                for p in mtb.labels_for(t["ds"], t["cat"], "UINMF")] == ["cty1", "cty2"]


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
    if GEN.has_all(key):
        assert (f"{n} method{'s have' if n > 1 else ' has'} a variant for {t['cat']} "
                f"{t['label']}: {GEN.and_list(everyone)}. Each runs on `{t['ds']}`.") in md
        assert "set `METHODS` to that list in section 2" in md
        assert f"That downloads {GEN.env_size_text(methods=everyone)}." in md
        assert f"]({GEN.SITE}tutorials/{key}_all/)" in md
    else:
        # the tutorial already runs every method: no second notebook, no link
        assert everyone == sorted(t["methods"])
        assert f"with a variant for {t['cat']} {t['label']}, so this tutorial runs every method" in md
        assert "_all" not in md
    assert (ROOT / "notebooks" / f"tutorial_{key}_all.ipynb").is_file() == GEN.has_all(key)
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
    args = {k: v for k, v in GEN.run_args(key).items() if k == "modalities"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan(t["ds"], t["cat"], methods=everyone, assume_gpu=True, verbose=False,
                      **args)
    assert sorted(set(sc[sc.files_ok].method)) == everyone, \
        (key, sorted(set(everyone) - set(sc[sc.files_ok].method)))


@pytest.mark.parametrize("key", KEYS)
def test_the_notebook_arguments_give_every_method_one_row_with_its_files(key):
    """With the exact arguments a notebook passes to ``run_all``, each of its
    methods runs once, on the variant of the task, with its input files on
    the task's dataset: the default methods in the tutorial, every method of
    the task in the every-method notebook."""
    import multibench as mtb
    t = GEN.TASKS[key]
    if not (mtb.config.DEFAULT.data_path / t["ds"]).is_dir():
        pytest.skip(f"{t['ds']} is not on disk")
    of_task = {m: "+".join(v.when.get("modalities") or ["(data_dir)"])
               for m, v in GEN.task_variants(key)}
    runs = [(False, t["methods"])] + [(True, GEN.task_methods(key))] * GEN.has_all(key)
    for every, methods in runs:
        rows = _would_run(key, methods, every)
        assert bool(rows.files_ok.all())
        assert dict(zip(rows.method, rows.modalities)) == {m: of_task[m] for m in methods}, \
            (key, every)
        assert len(rows) == len(methods), (key, every, list(rows.method))


def test_without_the_flag_the_atac_check_leaves_its_methods_out():
    """The reason the two every-method notebooks pass the flag."""
    for key, names in (("vertical_rna_adt_atac", ["Matilda"]),
                       ("cross_rna_atac", ["UnitedNet", "scMDC"])):
        import multibench as mtb
        if not (mtb.config.DEFAULT.data_path / GEN.TASKS[key]["ds"]).is_dir():
            pytest.skip(f"{GEN.TASKS[key]['ds']} is not on disk")
        everyone = GEN.task_methods(key)
        rows = _would_run(key, everyone, every=False)
        assert sorted(set(everyone) - set(rows.method)) == sorted(names), key


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


@pytest.mark.parametrize("key", ALL_KEYS)
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
    fast or medium methods than the tutorial runs, the defaults are the
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
        if GEN.has_all(key):
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
