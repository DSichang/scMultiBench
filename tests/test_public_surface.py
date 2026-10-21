"""The public surface (the 0.3.0 cut and the 0.4.0 core API), pinned.

``__all__`` of ``mtb`` (19 names), ``mtb.plot``, ``mtb.io``, ``mtb.env``,
``mtb.config``, ``mtb.catalog`` and ``mtb.data`` are EXACTLY the contract's sets; ``dir()`` of those namespaces
leaks no import; every deprecated alias warns and forwards; every removed
name / keyword fails loudly with the contract's message; every ``--help``
line that names a Python function names one that exists.
"""
import inspect
import re
import warnings

import pytest

import multibench as mtb
from multibench.engine import envs, ingest

MTB_ALL = {                                       # the 0.4.0 core API (TARGET.txt)
    "config", "plot", "catalog", "io", "env", "data",
    "find_methods", "method_info", "params_for", "cite",
    "scan", "run", "load_batch", "BatchResult",
    "evaluate", "load_results", "recommend",
    "describe_layout", "__version__",
}
PLOT_ALL = {"bubble", "bar"}
IO_ALL = {"export_dataset", "read_canonical"}
ENV_ALL = {"install"}                    # recipe left __all__ in 0.4.0 (X34)
CONFIG_ALL = {"DEFAULT"}
CATALOG_ALL = {"datasets", "metrics"}
DATA_ALL = {"fetch", "write_example"}


def _leaks(ns) -> set:
    """Names ``dir(ns)`` advertises beyond ``__all__`` (dunders and private
    names excluded)."""
    return {n for n in dir(ns) if not n.startswith("_") and n not in set(ns.__all__)}


# ------------------------------------------------------------------ __all__ sets
def test_top_level_all_is_exactly_the_contract():
    assert set(mtb.__all__) == MTB_ALL and len(MTB_ALL) == 19
    assert len(mtb.__all__) == len(MTB_ALL)                 # no duplicates
    assert all(hasattr(mtb, n) for n in MTB_ALL)
    assert _leaks(mtb) == set()


def test_io_all_is_exactly_the_contract():
    assert set(mtb.io.__all__) == IO_ALL and mtb.io is ingest
    assert _leaks(mtb.io) == set()


def test_config_all_is_exactly_the_contract():
    assert set(mtb.config.__all__) == CONFIG_ALL
    assert _leaks(mtb.config) == set()
    # kept as functions for internal imports, hidden from dir()
    assert mtb.config.category_folder("vertical") == "vertical integration"
    assert "category_folder" not in dir(mtb.config) and "metric_set_dir" not in dir(mtb.config)


def test_catalog_and_data_all_are_exactly_the_contract():
    assert set(mtb.catalog.__all__) == CATALOG_ALL and _leaks(mtb.catalog) == set()
    assert set(mtb.data.__all__) == DATA_ALL and _leaks(mtb.data) == set()


def test_plot_all_is_exactly_the_contract():
    assert set(mtb.plot.__all__) == PLOT_ALL
    assert _leaks(mtb.plot) == set()
    # the drawing internals stay in their module
    import importlib
    B = importlib.import_module("multibench.plot.bubble")
    assert callable(B.render) and B.FamilyBlock is not None


# ------------------------------------------------------------------ removed names / kwargs
def test_command_preview_is_gone_use_run_dry_run():
    assert not hasattr(mtb, "command_preview")
    from multibench import workflow
    assert not hasattr(workflow, "command_preview")
    inp = mtb.engine.resolve.inputs_for("D11", "vertical", "Matilda", modalities=["rna", "adt"])
    argv = mtb.engine.runner.run("Matilda", "vertical", inputs=inp, out_dir="o", dry_run=True)
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
    assert set(info["runtime"]) == {"tier", "worst_sec", "observed", "host", "note"}
    assert "deep_learning" not in info and "output" not in info


def test_cite_methods_keyword_is_gone():
    with pytest.raises(TypeError, match="methods"):
        mtb.cite(methods=["Matilda"])
    assert inspect.signature(mtb.cite).parameters["format"].default == "text"


def test_scan_data_path_is_keyword_only_and_the_plan_has_command():
    with pytest.raises(TypeError):
        mtb.scan("D11", "vertical", "data")
    df = mtb.run("D11", "vertical", methods=["Matilda"], dry_run=True, verbose=False)
    assert df.columns[-1] == "command" and len(df.columns) == 7          # c1 (C32)
    assert list(mtb.scan("D11", "vertical", methods=["Matilda"], verbose=False).columns) == \
        ["method", "modalities", "runnable", "reason"]


def test_run_all_dry_run_is_the_full_plan_and_out_dir_is_required_for_a_real_run():
    import pandas as pd
    from tests.conftest import full_scan
    a = mtb.run("D11", "vertical", methods=["Matilda"], dry_run=True, verbose=False)
    b = full_scan("D11", "vertical", methods=["Matilda"], verbose=False)
    from multibench import workflow as W
    pd.testing.assert_frame_equal(a, b[W.PLAN_COLUMNS])                 # c1 (C32)
    with pytest.raises(TypeError, match="out_dir"):
        mtb.run("D11", "vertical", methods=["Matilda"], verbose=False)


def test_rescore_only_is_gone():
    res = mtb.BatchResult([], "D11", "vertical")
    with pytest.raises(TypeError, match="only"):
        res.rescore(only={"ARI"})
    assert "metrics" in inspect.signature(mtb.BatchResult.rescore).parameters


# ------------------------------------------------------------------ CLI help names real functions
_MTB_REF = re.compile(r"mtb\.([A-Za-z_][\w.]*)")


import argparse as _argparse  # noqa: E402
argparse_SUPPRESS = _argparse.SUPPRESS
argparse_SubParsersAction = _argparse._SubParsersAction


