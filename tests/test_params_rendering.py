"""Hyperparameters: how a value is rendered, what is tunable, argument mistakes."""
from __future__ import annotations
import warnings

import pytest

import multibench as mtb
from multibench.engine import builder, envs, registry

MATILDA = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
SCJOINT = {"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _dry(method, category, inputs, params, tmp_path):
    return mtb.engine.runner.run(method, category, inputs=inputs, out_dir=str(tmp_path / "o"),
                   params=params, dry_run=True, cmd_template="conda run -n x {cmd}")


# X63: a param is rendered by the type the script declares
def test_a_bool_argument_gets_an_explicit_value(tmp_path, gpu):
    # a type=bool argument: bool('') is False (scJoint's use_cuda left tunable in r3)
    argv = _dry("Matilda", "vertical", MATILDA, {"augmentation": False}, tmp_path)
    assert argv[argv.index("--augmentation") + 1] == ""
    argv = _dry("Matilda", "vertical", MATILDA, {"augmentation": True}, tmp_path)
    assert argv[argv.index("--augmentation") + 1] == "True"


def test_matilda_augmentation_false_reaches_the_driver(tmp_path):
    argv = _dry("Matilda", "vertical", MATILDA, {"augmentation": False}, tmp_path)
    i = argv.index("--augmentation")
    assert argv[i + 1] == ""
    src = (registry._YAML.parent / "drivers" / "run_matilda.py").read_text()
    assert 'a.augmentation not in ("", "False")' in src


def test_a_switch_is_present_only_for_true(tmp_path):
    inputs = {"rna": "/d/rna.h5", "adt": "/d/adt.h5"}
    argv = _dry("scMDC", "vertical", inputs, {"filter1": True}, tmp_path)
    assert argv[-1] == "--filter1"
    argv = _dry("scMDC", "vertical", inputs, {"filter1": False}, tmp_path)
    assert "--filter1" not in argv


def test_a_switch_or_bool_refuses_another_value(tmp_path):
    with pytest.raises(TypeError, match=r"scMDC: filter1 takes True or False, got 1\."):
        _dry("scMDC", "vertical", {"rna": "/d/rna.h5", "adt": "/d/adt.h5"},
             {"filter1": 1}, tmp_path)
    with pytest.raises(TypeError, match=r"augmentation takes True or False, got 'False'"):
        _dry("Matilda", "vertical", MATILDA, {"augmentation": "False"}, tmp_path)


def test_a_store_false_switch_is_present_for_false():
    decl = {"type": "flag", "default": True}
    assert builder._param_tokens("umap_plot", False, decl) == ["--umap_plot"]
    assert builder._param_tokens("umap_plot", True, decl) == []


# X64: params_for leaves out what the package sets
@pytest.mark.parametrize("method, category, mods, gone", [
    ("scMDC", "vertical", ["rna", "adt"], {"nbatch", "data_file"}),
    ("scMM", "vertical", ["rna", "atac"], {"model", "experiment", "analytics"}),
    ("MIRA", "vertical", ["rna", "atac"], {"save_name", "load_model",
                                           "rna_topic_modal_path"}),
])
def test_tunable_leaves_out_what_the_package_sets(method, category, mods, gone):
    assert not gone & set(mtb.params_for(method, category, mods)["tunable"])


def test_a_package_set_key_is_refused():
    with pytest.raises(KeyError, match=r"scMDC does not accept \['nbatch'\]"):
        mtb.run("D52mini", "cross", methods=["scMDC"], params={"scMDC": {"nbatch": 5}},
                    dry_run=True, verbose=False)


# Y38: one default, the one a run uses
def test_tunable_default_is_what_a_run_uses():
    # r3 (X64): no_cuda is the package's switch and left tunable; 0.4.0: one
    # default per key, no script_default copy
    t = mtb.params_for("scMM", "vertical", ["rna", "atac"])["tunable"]
    assert "no_cuda" not in t and set(t["epochs"]) == {"type", "default"}


def test_run_all_params_for_a_method_not_run():
    with pytest.raises(ValueError, match=r"^params names scMM, which is not in methods=\.$"):
        mtb.run("D52mini", "cross", methods=["Multigrate"],
                    params={"scMM": {"epochs": 3}}, out_dir="/o", dry_run=True,
                    verbose=False)


