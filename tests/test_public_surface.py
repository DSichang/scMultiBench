"""The public surface of 0.4.0, pinned: ``mtb.__all__`` and each namespace's
``__all__``, every public signature, the removed names and keywords, no command
line, and no FutureWarning from any public call."""
import importlib
import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

import multibench as mtb
from multibench.engine import ingest

ROOT = Path(__file__).resolve().parents[1]

TOP = ["config", "plot", "catalog", "io", "env", "data",
       "find_methods", "method_info", "params_for",
       "scan", "run", "load_run", "RunResult",
       "evaluate", "load_results", "rank",
       "__version__"]

NAMESPACES = {"config": ["data_path", "repo_path", "envs_dir", "leiden_flavor"], "plot": ["bubble", "bar"],
              "catalog": ["datasets", "metrics"], "io": ["export_dataset", "read"],
              "env": ["install"], "data": ["fetch", "write_example"]}

#: every public function and its parameters, in order
SIGNATURES = {
    "find_methods": ["category", "modalities", "needs_labels"],
    "method_info": ["method"],
    "params_for": ["method", "category", "modalities"],
    "scan": ["dataset", "category", "methods", "modalities", "params", "assume_gpu",
             "verbose", "command"],
    "run": ["dataset", "category", "out_dir", "methods", "modalities", "params", "batch",
            "evaluate", "timeout", "skip_existing", "verbose"],
    "load_run": ["out_dir", "methods"],
    "evaluate": ["embedding", "labels", "dataset", "method", "category", "batch",
                 "metrics", "obsm", "verbose"],
    "load_results": ["category", "dataset", "methods", "metrics"],
    "rank": ["category", "modalities", "methods", "metrics", "long_df"],
    "plot.bubble": ["long_df", "metrics", "methods", "title", "save"],
    "plot.bar": ["long_df", "metrics", "methods", "title", "save"],
    "catalog.datasets": ["category"],
    "catalog.metrics": [],
    "io.export_dataset": ["data", "dataset_dir", "rna", "adt", "atac", "atac_kind",
                          "labels", "batch", "category", "adt_names", "batch_index",
                          "overwrite"],
    "io.read": ["path"],
    "env.install": ["methods", "category", "dry_run", "flavor", "build"],
    "data.fetch": ["datasets", "verbose"],
    "data.write_example": ["dataset", "folder", "verbose"],
}

REMOVED = ["run_all", "inputs_for", "labels_for", "to_long", "sweep", "classify",
           "classification_metrics", "results_coverage", "list_methods",
           "list_categories", "list_tasks", "available_datasets",
           "AmbiguousVariantError", "DegenerateRerunWarning", "plan", "plan_commands",
           "runtime_hint", "BatchResult", "load_batch"]


def test_top_level_all_is_pinned():
    assert mtb.__all__ == TOP
    assert all(hasattr(mtb, n) for n in TOP)
    assert [n for n in dir(mtb) if not n.startswith("__")] == sorted(
        n for n in TOP if not n.startswith("__"))


@pytest.mark.parametrize("ns", sorted(set(NAMESPACES) - {"config"}))
def test_namespace_all_is_pinned(ns):
    module = getattr(mtb, ns)
    assert module.__all__ == NAMESPACES[ns]
    assert [n for n in dir(module) if not n.startswith("__")] == sorted(NAMESPACES[ns])


def test_config_fields_are_pinned():
    assert list(mtb.config.FIELDS) == NAMESPACES["config"]
    public = [n for n in dir(mtb.config) if not n.startswith("_")]
    assert public == sorted(["FIELDS", *NAMESPACES["config"]])


def test_config_is_the_settings_object():
    """``mtb.config`` holds the settings itself (0.4.0): no ``DEFAULT``, no module."""
    from multibench._settings import Settings
    assert isinstance(mtb.config, Settings)
    assert not (Path(mtb.__file__).parent / "config.py").exists()
    assert not hasattr(mtb.config, "DEFAULT") and not hasattr(mtb.config, "Config")
    text = repr(mtb.config)
    assert text.startswith("mtb.config(data_path=")
    assert all(f"{k}=" in text for k in NAMESPACES["config"])
    with pytest.raises(AttributeError, match="no attribute 'data_dir'"):
        mtb.config.data_dir = "x"
    with pytest.raises(AttributeError, match="no attribute 'ensure_repo'"):
        mtb.config.ensure_repo = 3


def test_io_export_dataset_is_the_engine_function():
    assert mtb.io.export_dataset is ingest.export_dataset


def test_plot_internals_stay_in_their_module():
    B = importlib.import_module("multibench.plot.bubble")
    assert callable(B.render) and B.FamilyBlock is not None


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
                     ("env", "status"), ("env", "plan"), ("eval", "to_long"),
                     # internals are not reachable through the public namespaces
                     ("env", "recipe"), ("env", "group_for"), ("io", "export_canonical"),
                     ("io", "suffix_fix"), ("catalog", "FAMILY_METRICS"),
                     ("data", "catalog"), ("data", "results"), ("config", "ensure_repo"),
                     # the scripts and git helpers live in engine.scripts
                     ("config", "category_folder"), ("config", "scripts_commit"),
                     ("config", "_and_list")]:
        assert not hasattr(getattr(mtb, ns), name), f"{ns}.{name}"
    assert not hasattr(mtb.RunResult([], "D11", "vertical"), "results")


def test_there_is_no_command_line():
    assert not (ROOT / "multibench" / "cli.py").exists()
    assert not (ROOT / "multibench" / "__main__.py").exists()
    text = (ROOT / "pyproject.toml").read_text()
    assert "[project.scripts]" not in text and "console_scripts" not in text
    assert not hasattr(mtb.config, "hint") and not hasattr(mtb.config, "_CLI")


def test_no_deprecation_machinery_is_left():
    from multibench import _text
    for name in ("deprecated", "renamed", "keywords", "module_getattr", "use", "note",
                 "removed_text", "NO_EFFECT"):
        assert not hasattr(_text, name), name


def test_command_preview_is_gone_use_run_dry_run():
    assert not hasattr(mtb, "command_preview")
    from multibench import workflow
    assert not hasattr(workflow, "command_preview")
    inp = mtb.engine.resolve.inputs_for("D11", "vertical", "Matilda", modalities=["rna", "adt"])
    argv = mtb.engine.runner.preview("Matilda", "vertical", inputs=inp, out_dir="o")[0]
    assert isinstance(argv, list) and "--save_path" in argv


def test_io_from_mudata_and_write_labels_are_gone():
    with pytest.raises(AttributeError):
        mtb.io.from_mudata
    with pytest.raises(AttributeError):
        mtb.io.write_labels
    assert callable(ingest._write_labels)


@pytest.mark.parametrize("kw", [{"available": True}, {"foo": 1}])
def test_find_methods_other_keywords_raise_pythons_own_typeerror(kw):
    with pytest.raises(TypeError) as e:
        mtb.find_methods(**kw)
    assert str(e.value).endswith(f"got an unexpected keyword argument {next(iter(kw))!r}")


def test_find_methods_filters_are_keyword_only():
    with pytest.raises(TypeError):
        mtb.find_methods("vertical", "clustering")


def test_method_info_files_dir_is_gone_and_runtime_is_there():
    with pytest.raises(TypeError, match="files_dir"):
        mtb.method_info("Matilda", files_dir="x")
    with pytest.raises(TypeError):
        mtb.method_info("Matilda", True)                  # verbose is keyword-only
    info = mtb.method_info("Matilda")
    assert set(info["runtime"]) == {"measured_sec", "tier", "worst_sec", "observed", "host",
                                    "note"}
    assert "deep_learning" not in info and "output" not in info


def test_scan_is_the_plan_with_its_command():
    """d1: one check, one run - scan returns the plan, run has no dry_run."""
    with pytest.raises(TypeError):
        mtb.scan("D11", "vertical", "data")
    df = mtb.scan("D11", "vertical", methods=["Matilda"], verbose=False, command=True)
    assert list(df.columns) == ["method", "modalities", "runnable", "reason", "env",
                                "caveat", "command"]
    assert df["command"].iloc[0].startswith("conda run -n matilda ")
    assert "command" not in mtb.scan("D11", "vertical", methods=["Matilda"], verbose=False).columns
    assert "dry_run" not in inspect.signature(mtb.run).parameters


def test_out_dir_is_required_for_a_run():
    params = inspect.signature(mtb.run).parameters
    assert params["out_dir"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="out_dir"):
        mtb.run("D11", "vertical", methods=["Matilda"], verbose=False)
    with pytest.raises(TypeError):
        mtb.run("D11", "vertical", None, methods=["Matilda"], verbose=False)


def test_rescore_only_is_gone():
    res = mtb.RunResult([], "D11", "vertical")
    with pytest.raises(TypeError, match="only"):
        res.rescore(only={"ARI"})
    assert "metrics" in inspect.signature(mtb.RunResult.rescore).parameters


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
        mtb.catalog.datasets()
        mtb.catalog.metrics()
        long = mtb.load_results("vertical", dataset="D11")
        mtb.rank("vertical")
        mtb.rank("vertical", long_df=long)
        mtb.plot.bubble(long, save=tmp_path / "b.png")
        mtb.plot.bar(long, save=tmp_path / "a.png")
        mtb.evaluate(emb, labels=lab, metrics=["ARI", "NMI"], verbose=False)
        mtb.scan("D11", "vertical", verbose=False)
        mtb.scan("D11", "vertical", methods=["Matilda"], verbose=False)
        mtb.env.install(["Matilda"], dry_run=True)
        print(mtb.config)
    future = [str(w.message) for w in rec
              if issubclass(w.category, (FutureWarning, DeprecationWarning))
              and ("/multibench/" in str(w.filename)
                   or Path(w.filename).resolve() == Path(__file__).resolve())]
    assert future == []
