"""Scope guard: the package describes only the tasks it benchmarks.

The owner's rule: three benchmark tasks and four removed methods are never
mentioned where a user can see them - API return values, CLI help, and every
text file the package ships (docstrings, YAML, CSV, drivers). This file pins
that on the live objects, on the shipped files and on the docs surfaces
(notebooks, SETUP.md, the notebook generator, and the site's source when it is
reachable), with three exceptions that are spelled out below and fail loudly
when they go stale:

- frozen lockfile pins of real envs (``engine/env_locks/*.yml``), by package name;
- an upstream API keyword in a driver (Matilda's own ``task()`` argument);
- a paper title, which is a citation fact (``engine/references.yaml``).

Everything else that matches FORBIDDEN is a finding: reword it, do not add an
exception for it.
"""
import argparse
import json
import os
import re
import warnings
from pathlib import Path

import pytest
import yaml

import multibench as mtb
from multibench import cli

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "multibench"

FORBIDDEN = re.compile(r"(?i)classif|imput|registration|spatial|\bGPSA\b|\bPASTE2?\b|\bSPIRAL\b")

#: the tasks methods declare; ``find_methods(task=)`` accepts exactly these
TASKS = ["batch", "clustering", "dimension_reduction"]

# ---- the three exceptions ---------------------------------------------------
#: lockfile pins (``- name==version`` / ``- name=version``) that freeze a real
#: env's contents; exempt by package name only, and only in env_locks/*.yml
LOCK_PINS = {"gpsa", "paste-bio", "paste2", "spatial-eggplant"}
_PIN_LINE = re.compile(r"^\s*-\s*(?P<name>[A-Za-z0-9_.\-]+)\s*(?:==|=)\s*\S+\s*$")

#: upstream API keywords a driver must pass verbatim, per driver file
DRIVER_KEYWORDS = {
    # matilda.task(..., classification=True). The embedding comes from the same
    # call; the keyword goes once a benchmark-host run shows that dim_reduce=True
    # alone writes the same embedding.h5 (then drop predict.csv from the driver
    # and from the Matilda variants' extra_outputs too).
    "engine/drivers/run_matilda.py": ["classification=True"],
}

#: paper titles (Crossref-checked citation facts), per method id in references.yaml
CITATION_TITLES = {
    "sciPENN": "A multi-use deep learning method for CITE-seq and single-cell RNA-seq "
               "data integration with cell surface protein prediction and imputation",
}


def _hits(text):
    return sorted({m.group(0) for m in FORBIDDEN.finditer(str(text))})


def _strip_titles(text):
    for title in CITATION_TITLES.values():
        text = text.replace(title, "<title>")
    return text


def _walk_parsers(parser):
    yield parser.prog, parser
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for sub in a.choices.values():
                yield from _walk_parsers(sub)


# ---- tasks ------------------------------------------------------------------
def test_list_tasks_names_only_the_benchmarked_tasks():
    assert mtb.list_tasks() == TASKS


def test_find_methods_by_task_still_works():
    everything = set(mtb.list_methods())
    for task in TASKS:
        found = mtb.find_methods(task=task)
        assert found, f"find_methods(task={task!r}) found nothing"
        for m in found:
            assert task in mtb.method_info(m)["tasks"], (task, m)
    # every method keeps clustering, so dropping the other tags lost no method
    assert set(mtb.find_methods(task="clustering")) == everything
    with pytest.raises(ValueError, match="valid: \\['batch', 'clustering', 'dimension_reduction'\\]"):
        mtb.find_methods(task="no_such_task")


def test_cli_task_help_names_the_live_tasks():
    p = dict(_walk_parsers(cli.build_parser()))["multibench list"]
    text = " ".join(p.format_help().split())
    for task in TASKS:
        assert task in text, task


# ---- user-visible return values and help ------------------------------------
def test_api_return_values_stay_in_scope():
    found = []

    def check(where, value):
        hits = _hits(_strip_titles(repr(value) if not isinstance(value, str) else value))
        if hits:
            found.append(f"{where}: {hits}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check("list_tasks()", mtb.list_tasks())
        check("list_categories()", mtb.list_categories())
        for m in mtb.list_methods():
            info = mtb.method_info(m, verbose=True)
            check(f"method_info({m!r}, verbose=True)", info)
            check(f"env.recipe({m!r})", mtb.env.recipe(m))
            for v in info["supports"]:
                check(f"params_for({m!r}, {v['category']!r}, {v['modalities']})",
                      mtb.params_for(m, v["category"], v["modalities"]))
        check("cite(<all methods>)", mtb.cite(*mtb.list_methods()))
        for c in [None, *mtb.list_categories()]:
            check(f"describe_layout({c!r})", mtb.describe_layout(c))
        for name in ("methods", "datasets", "metrics"):
            check(f"catalog.{name}()", getattr(mtb.catalog, name)().to_string())
    assert not found, "\n".join(found)


def test_cli_help_stays_in_scope():
    found = [f"{name}: {_hits(p.format_help())}"
             for name, p in _walk_parsers(cli.build_parser()) if _hits(p.format_help())]
    assert not found, "\n".join(found)


# ---- shipped text files -----------------------------------------------------
def _package_text_files():
    """Every text file under multibench/ (skips __pycache__ and hidden build dirs)."""
    for path in sorted(PKG.rglob("*")):
        rel = path.relative_to(PKG)
        if not path.is_file() or any(p == "__pycache__" or p.startswith(".") for p in rel.parts[:-1]):
            continue
        try:
            yield rel.as_posix(), path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue                                   # binary (h5, png, ...)


def _exempt(rel, text):
    """``text`` with the spelled-out exceptions of ``rel`` removed."""
    if rel.startswith("engine/env_locks/") and rel.endswith(".yml"):
        lines = []
        for line in text.splitlines():
            pin = _PIN_LINE.match(line)
            lines.append("" if pin and pin.group("name").lower() in LOCK_PINS else line)
        text = "\n".join(lines)
    for keyword in DRIVER_KEYWORDS.get(rel, []):
        text = text.replace(keyword, "<upstream keyword>")
    if rel == "engine/references.yaml":
        text = _strip_titles(text)
    return text


def test_shipped_text_files_stay_in_scope():
    found = []
    for rel, text in _package_text_files():
        for i, line in enumerate(_exempt(rel, text).splitlines(), 1):
            if _hits(line):
                found.append(f"multibench/{rel}:{i}: {line.strip()[:120]}")
    assert not found, "\n".join(found)


def test_pypi_page_and_metadata_stay_in_scope():
    for name in ("README.md", "pyproject.toml"):
        assert not _hits((ROOT / name).read_text()), name


# ---- docs surfaces outside the package ---------------------------------------
def _docs_surfaces():
    """Text a user reads that lives outside the package tree: the notebooks,
    SETUP.md and tools/gen_tut.py (which writes the notebooks); with
    SCMULTIBENCH_DOCS=<docs dir> (the site's source is a separate repository)
    also every page, script and stylesheet of the site and its theme overrides."""
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    paths += [ROOT / "SETUP.md", ROOT / "tools" / "gen_tut.py"]
    docs = os.environ.get("SCMULTIBENCH_DOCS")
    if docs and Path(docs).is_dir():
        docs = Path(docs).resolve()
        paths += sorted(docs.rglob("*.md"))
        paths += sorted((docs / "javascripts").glob("*.js"))
        paths += sorted((docs / "stylesheets").glob("*.css"))
        paths += sorted((docs.parent / "overrides").rglob("*.html"))
    return paths


def _surface_id(path):
    """``path`` relative to the package root, or to the docs repository root."""
    for base in (ROOT, Path(os.environ.get("SCMULTIBENCH_DOCS") or ROOT).resolve().parent):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            pass
    return path.name


@pytest.mark.parametrize("path", _docs_surfaces(), ids=_surface_id)
def test_docs_surfaces_stay_in_scope(path):
    """A notebook is checked by its cell sources: its outputs come from an
    execution, and the API values they print are pinned above."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        text = "\n".join("".join(c["source"]) for c in json.loads(text)["cells"])
    assert not _hits(text), f"{path}: {_hits(text)}"


def test_exceptions_are_still_needed():
    """Each exception names something that is still there, so a stale one fails
    here instead of silently widening the guard."""
    for rel, keywords in DRIVER_KEYWORDS.items():
        text = (PKG / rel).read_text()
        for keyword in keywords:
            assert text.count(keyword) == 1, (rel, keyword)
    refs = yaml.safe_load((PKG / "engine" / "references.yaml").read_text())
    for method, title in CITATION_TITLES.items():
        assert refs[method]["reference"]["title"] == title, method
    pinned = set()
    for lock in (PKG / "engine" / "env_locks").glob("*.yml"):
        for line in lock.read_text().splitlines():
            pin = _PIN_LINE.match(line)
            if pin and _hits(pin.group("name")):
                pinned.add(pin.group("name").lower())
    assert pinned == LOCK_PINS


def test_every_output_kind_a_variant_declares_loads(tmp_path):
    """io.load_output accepts every kind the registry's variants produce (the
    unused kind it dropped was produced by no variant)."""
    import h5py
    import numpy as np
    from multibench.engine import io, registry
    from multibench.engine.schema import OutputSpec

    kinds = {o.kind for s in registry.load() for v in s.variants
             for o in [v.output, *v.extra_outputs]}
    assert kinds == {"embedding", "graph", "labels"}
    for kind in kinds:
        if kind == "labels":
            (tmp_path / "l.txt").write_text("a\nb\n")
            assert io.load_output(tmp_path, OutputSpec(kind=kind, file="l.txt")) == ["a", "b"]
        else:
            with h5py.File(tmp_path / f"{kind}.h5", "w") as f:
                f.create_dataset("data", data=np.zeros((2, 3)))
            out = io.load_output(tmp_path, OutputSpec(kind=kind, file=f"{kind}.h5", dataset="data"))
            assert out.shape == (2, 3)
