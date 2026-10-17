"""The public surface (the 0.3.0 cut and the 0.4 deprecations), pinned.

``__all__`` of ``mtb``, ``mtb.plot``, ``mtb.io``, ``mtb.env`` and
``mtb.config`` are EXACTLY the contract's sets; ``dir()`` of those namespaces
leaks no import; every deprecated alias warns and forwards; every removed
name / keyword fails loudly with the contract's message; every ``--help``
line that names a Python function names one that exists.
"""
import inspect
import re
import warnings

import pytest

import multibench as mtb
from multibench import cli
from multibench.engine import envs, ingest

MTB_ALL = {
    "config", "plot", "catalog", "io", "env", "data",
    "find_methods", "method_info", "params_for", "describe_layout", "cite",
    "inputs_for", "labels_for",
    "scan", "run", "run_all", "sweep", "load_batch", "BatchResult",
    "evaluate", "to_long", "classify", "classification_metrics",
    "load_results", "results_coverage", "recommend",
    "AmbiguousVariantError", "DegenerateRerunWarning", "__version__",
}
PLOT_ALL = {"bubble", "bar", "build_table", "BubbleTable"}
IO_ALL = {"export_dataset", "to_canonical", "read_canonical"}
ENV_ALL = {"install", "doctor", "recipe"}
CONFIG_ALL = {"Config", "DEFAULT"}


def _leaks(ns) -> set:
    """Names ``dir(ns)`` advertises beyond ``__all__`` (dunders and private
    names excluded)."""
    return {n for n in dir(ns) if not n.startswith("_") and n not in set(ns.__all__)}


# ------------------------------------------------------------------ __all__ sets
def test_top_level_all_is_exactly_the_contract():
    assert set(mtb.__all__) == MTB_ALL
    assert len(mtb.__all__) == len(MTB_ALL)                 # no duplicates
    assert all(hasattr(mtb, n) for n in MTB_ALL)
    assert _leaks(mtb) == set()


def test_io_all_is_exactly_the_contract():
    assert set(mtb.io.__all__) == IO_ALL and mtb.io is ingest
    assert _leaks(mtb.io) == set()


def test_env_all_is_exactly_the_contract():
    assert set(mtb.env.__all__) == ENV_ALL and mtb.env is envs
    # the two helpers the reference documents are in dir() too (0.4)
    assert _leaks(mtb.env) == {"env_prefix", "host_has_gpu"}
    # the rest of engine/envs.py stays importable for the CLI and the tests
    for name in ("group_for", "create_all", "install_packed", "installed_envs",
                 "default_env_name", "packed_sizes", "packed_manifest"):
        assert callable(getattr(envs, name)) and name not in dir(mtb.env)


def test_config_all_is_exactly_the_contract():
    assert set(mtb.config.__all__) == CONFIG_ALL
    assert _leaks(mtb.config) == set()
    # kept as functions for internal imports, hidden from dir()
    assert mtb.config.category_folder("vertical") == "vertical integration"
    assert mtb.config.metric_set_dir("scib") == "scib_metric"
    assert "category_folder" not in dir(mtb.config) and "metric_set_dir" not in dir(mtb.config)


def test_plot_all_is_exactly_the_contract():
    assert set(mtb.plot.__all__) == PLOT_ALL
    assert _leaks(mtb.plot) == set()
    # render / FamilyBlock stay importable but are not advertised
    assert callable(mtb.plot.render) and mtb.plot.FamilyBlock is not None


# ------------------------------------------------------------------ deprecated aliases
@pytest.mark.parametrize("name, new", [
    ("plan", "mtb.run_all(..., dry_run=True)"),
    ("plan_commands", "mtb.run_all(..., dry_run=True)"),
    ("runtime_hint", "mtb.method_info(method)['runtime']"),
])
def test_the_02_names_are_removed_naming_the_replacement(name, new):
    """0.4 removes them, as their 0.3 warnings said; the error names what replaces them."""
    with pytest.raises(AttributeError) as e:
        getattr(mtb, name)
    assert str(e.value) == f"mtb.{name} was removed in 0.4. Use {new}."
    assert name not in mtb.__all__ and name not in dir(mtb)


# ------------------------------------------------------------------ removed names / kwargs
def test_command_preview_is_gone_use_run_dry_run():
    assert not hasattr(mtb, "command_preview")
    from multibench import workflow
    assert not hasattr(workflow, "command_preview")
    inp = mtb.inputs_for("D11", "vertical", "Matilda", modalities=["rna", "adt"])
    argv = mtb.run("Matilda", "vertical", inputs=inp, out_dir="o", dry_run=True)
    assert isinstance(argv, list) and "--save_path" in argv


def test_run_task_is_keyword_only():
    inp = mtb.inputs_for("D11", "vertical", "Matilda", modalities=["rna", "adt"])
    with pytest.raises(TypeError):
        mtb.run("Matilda", "vertical", "clustering", inputs=inp, out_dir="o", dry_run=True)
    assert mtb.run("Matilda", "vertical", task="clustering", inputs=inp, out_dir="o",
                   dry_run=True)


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


def test_inputs_for_old_order_raises_the_contract_message():
    with pytest.raises(TypeError) as e:
        mtb.inputs_for("D11", "Matilda", "vertical")
    assert str(e.value) == ("inputs_for argument order is (dataset, category, method) "
                            "since 0.3.0; you passed (dataset, method, category)")
    with pytest.raises(TypeError) as e:
        mtb.labels_for("D11", "Matilda", "vertical")
    assert str(e.value) == ("labels_for argument order is (dataset, category, method) "
                            "since 0.3.0; you passed (dataset, method, category)")
    # modalities / data_path / check are keyword-only in both
    with pytest.raises(TypeError):
        mtb.inputs_for("D11", "vertical", "Matilda", ["rna", "adt"])
    with pytest.raises(TypeError):
        mtb.labels_for("D11", "vertical", "Matilda", ["rna", "adt"])


def test_scan_data_path_is_keyword_only_and_the_plan_has_command():
    with pytest.raises(TypeError):
        mtb.scan("D11", "vertical", "data")
    df = mtb.run_all("D11", "vertical", methods=["Matilda"], dry_run=True, verbose=False)
    assert df.columns[-1] == "command" and len(df.columns) == 18
    assert list(mtb.scan("D11", "vertical", methods=["Matilda"], verbose=False).columns) == \
        ["method", "modalities", "runnable", "reason"]


def test_run_all_dry_run_is_the_full_plan_and_out_dir_is_required_for_a_real_run():
    import pandas as pd
    from tests.conftest import full_scan
    a = mtb.run_all("D11", "vertical", methods=["Matilda"], dry_run=True, verbose=False)
    b = full_scan("D11", "vertical", methods=["Matilda"], verbose=False)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(TypeError, match="out_dir"):
        mtb.run_all("D11", "vertical", methods=["Matilda"], verbose=False)


def test_rescore_only_is_gone():
    res = mtb.BatchResult([], "D11", "vertical")
    with pytest.raises(TypeError, match="only"):
        res.rescore(only={"ARI"})
    assert "metrics" in inspect.signature(mtb.BatchResult.rescore).parameters


# ------------------------------------------------------------------ env.install
def test_env_install_is_the_cli_code_path(monkeypatch):
    rows = [{"env": "scmb_torch", "methods": ["SCALEX"], "exists": False, "has_lock": True,
             "cmds": []},
            {"env": "have_it", "methods": ["Z"], "exists": True, "has_lock": True, "cmds": []}]
    seen = {}

    def fake_create_all(**kw):
        seen.update(kw)
        return rows
    monkeypatch.setattr(envs, "create_all", fake_create_all)
    monkeypatch.setattr(envs, "install_packed", lambda env, **kw: pytest.fail("no download"))
    monkeypatch.setattr(envs, "packed_manifest", lambda: {"scmb_torch": "https://x/t.tar.gz"})
    out = mtb.env.install(["SCALEX"], category=None, packed=True, dry_run=True)
    assert seen == {"category": None, "methods": ["SCALEX"], "conda": None,
                    "dry_run": True, "force": False}
    assert [r["state"] for r in out] == ["packed archive published", "have"]
    assert out[0]["packed_url"] == "https://x/t.tar.gz"
    out = mtb.env.install(["SCALEX"], packed=False)
    assert [r["state"] for r in out] == ["build(dry-run)", "have"]
    with pytest.raises(KeyError, match="Did you mean"):
        mtb.env.install(["StabMapp"])
    sig = inspect.signature(mtb.env.install)
    assert list(sig.parameters) == ["methods", "category", "packed", "dry_run", "conda",
                                    "force", "flavor"]
    assert sig.parameters["dry_run"].default is True and sig.parameters["packed"].default is True


# ------------------------------------------------------------------ CLI help names real functions
_MTB_REF = re.compile(r"mtb\.([A-Za-z_][\w.]*)")


def _help_texts(parser):
    yield parser.description or ""
    yield parser.epilog or ""
    for a in parser._actions:
        if a.help and a.help is not argparse_SUPPRESS:
            yield a.help
        if isinstance(a, argparse_SubParsersAction):
            for sub in a.choices.values():
                yield from _help_texts(sub)


import argparse as _argparse  # noqa: E402
argparse_SUPPRESS = _argparse.SUPPRESS
argparse_SubParsersAction = _argparse._SubParsersAction


def test_every_help_line_names_an_existing_function():
    missing = []
    for text in _help_texts(cli.build_parser()):
        for ref in _MTB_REF.findall(text):
            obj = mtb
            for part in ref.rstrip(".").split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    missing.append(ref)
                    break
    assert missing == []
    for gone in ("command_preview", "plan_commands", "runtime_hint", "--only"):
        assert gone not in "\n".join(_help_texts(cli.build_parser()))
