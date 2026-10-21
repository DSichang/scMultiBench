"""The public surface of 0.4.0, pinned: ``mtb.__all__`` and each namespace's
``__all__``, no command line, and no FutureWarning from any public call."""
import importlib
import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

import multibench as mtb

ROOT = Path(__file__).resolve().parents[1]

TOP = ["config", "plot", "catalog", "io", "env", "data",
       "find_methods", "method_info", "params_for", "cite",
       "scan", "run", "load_batch", "BatchResult",
       "evaluate", "load_results", "recommend",
       "__version__"]

NAMESPACES = {"config": ["DEFAULT"], "plot": ["bubble", "bar"],
              "catalog": ["datasets", "metrics"], "io": ["export_dataset", "read_canonical"],
              "env": ["install"], "data": ["fetch", "write_example"]}

#: every public function and its parameters, in order
SIGNATURES = {
    "find_methods": ["category", "modalities", "needs_labels"],
    "method_info": ["method"],
    "params_for": ["method", "category", "modalities"],
    "cite": ["methods", "format"],
    "scan": ["dataset", "category", "methods", "modalities", "assume_gpu", "verbose"],
    "run": ["dataset", "category", "out_dir", "methods", "modalities", "params", "batch",
            "evaluate", "dry_run", "timeout", "skip_existing", "verbose"],
    "load_batch": ["out_dir", "methods"],
    "evaluate": ["embedding", "labels", "dataset", "method", "category", "batch",
                 "metrics", "obsm", "verbose"],
    "load_results": ["category", "dataset", "methods", "metrics"],
    "recommend": ["category", "modalities", "methods", "metrics", "long_df"],
    "plot.bubble": ["long_df", "metrics", "methods", "title", "save"],
    "plot.bar": ["long_df", "metrics", "methods", "title", "save"],
    "catalog.datasets": ["category"],
    "catalog.metrics": [],
    "io.export_dataset": ["data", "dataset_dir", "rna", "adt", "atac", "atac_kind",
                          "labels", "batch", "category", "adt_names", "batch_index",
                          "overwrite"],
    "io.read_canonical": ["path"],
    "env.install": ["methods", "category", "dry_run", "flavor", "build"],
    "data.fetch": ["datasets", "verbose"],
    "data.write_example": ["dataset", "folder", "verbose"],
}

REMOVED = ["run_all", "inputs_for", "labels_for", "to_long", "sweep", "classify",
           "classification_metrics", "results_coverage", "list_methods",
           "list_categories", "list_tasks", "available_datasets",
           "AmbiguousVariantError", "DegenerateRerunWarning", "plan", "plan_commands",
           "runtime_hint"]


def test_top_level_all_is_pinned():
    assert mtb.__all__ == TOP
    assert [n for n in dir(mtb) if not n.startswith("__")] == sorted(
        n for n in TOP if not n.startswith("__"))


@pytest.mark.parametrize("ns", sorted(NAMESPACES))
def test_namespace_all_is_pinned(ns):
    module = getattr(mtb, ns)
    assert module.__all__ == NAMESPACES[ns]
    assert [n for n in dir(module) if not n.startswith("__")] == sorted(NAMESPACES[ns])


@pytest.mark.parametrize("name", sorted(SIGNATURES))
def test_signatures_are_pinned(name):
    obj = mtb
    for part in name.split("."):
        obj = getattr(obj, part)
    assert list(inspect.signature(obj).parameters) == SIGNATURES[name]


@pytest.mark.parametrize("name", REMOVED)
def test_removed_names_are_gone(name):
    assert not hasattr(mtb, name)


def test_removed_namespace_names_are_gone():
    for ns, name in [("data", "fetchable"), ("data", "fetch_outputs"),
                     ("catalog", "methods"), ("io", "to_canonical"), ("config", "Config"),
                     ("plot", "build_table"), ("plot", "BubbleTable"), ("env", "doctor"),
                     ("env", "status"), ("env", "plan"), ("eval", "to_long")]:
        assert not hasattr(getattr(mtb, ns), name), f"{ns}.{name}"
    assert not hasattr(mtb.BatchResult([], "D11", "vertical"), "results")


def test_there_is_no_command_line():
    assert not (ROOT / "multibench" / "cli.py").exists()
    assert not (ROOT / "multibench" / "__main__.py").exists()
    with pytest.raises(ImportError):
        importlib.import_module("multibench.cli")
    with pytest.raises(ImportError):
        importlib.import_module("multibench.__main__")
    text = (ROOT / "pyproject.toml").read_text()
    assert "[project.scripts]" not in text and "console_scripts" not in text
    assert not hasattr(mtb.config, "hint") and not hasattr(mtb.config, "_CLI")


def test_no_deprecation_machinery_is_left():
    from multibench import _compat
    for name in ("deprecated", "renamed", "keywords", "module_getattr", "use", "note",
                 "removed_text", "NO_EFFECT"):
        assert not hasattr(_compat, name), name


def test_public_calls_emit_no_future_warning(tmp_path):
    """A smoke run of the public calls that need no method environment."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(120, 5))
    lab = np.array(["a", "b", "c"] * 40)
    emb[lab == "a", 0] += 5
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mtb.find_methods("vertical", modalities=["rna", "adt"])
        mtb.method_info("Matilda")
        mtb.params_for("Matilda", "vertical", ["rna", "adt"])
        mtb.cite("Matilda", format="bibtex")
        mtb.catalog.datasets()
        mtb.catalog.metrics()
        long = mtb.load_results("vertical", dataset="D11")
        mtb.recommend("vertical")
        mtb.recommend("vertical", long_df=long)
        mtb.plot.bubble(long, save=tmp_path / "b.png")
        mtb.plot.bar(long, save=tmp_path / "a.png")
        mtb.evaluate(emb, labels=lab, metrics=["ARI", "NMI"], verbose=False)
        mtb.scan("D11", "vertical", verbose=False)
        mtb.run("D11", "vertical", methods=["Matilda"], dry_run=True, verbose=False)
        mtb.env.install(["Matilda"], dry_run=True)
        print(mtb.config.DEFAULT)
    future = [str(w.message) for w in rec
              if issubclass(w.category, (FutureWarning, DeprecationWarning))
              and ("/multibench/" in str(w.filename)
                   or Path(w.filename).resolve() == Path(__file__).resolve())]
    assert future == []
