"""Hyperparameters: how a value is rendered, what is tunable, argument mistakes."""
from __future__ import annotations

import pytest

import multibench as mtb
from multibench.engine import builder, envs, registry

SCJOINT = {"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _dry(method, category, inputs, params, tmp_path):
    return mtb.engine.runner.preview(method, category, inputs=inputs, out_dir=str(tmp_path / "o"), params=params)[0]


# X63: a param is rendered by the type the script declares
def test_a_bool_argument_gets_an_explicit_value():
    # a type=bool argument: bool('') is False, bool('False') is True
    decl = {"type": "bool", "default": True}
    assert builder._param_tokens("x", False, decl) == ["--x", ""]
    assert builder._param_tokens("x", True, decl) == ["--x", "True"]


def test_matilda_augmentation_is_not_offered():
    # matilda's main_train fails whenever augmentation is False (new_data unset)
    assert "augmentation" not in mtb.params_for("Matilda", "vertical", ["rna", "adt"])["tunable"]
    with pytest.raises(KeyError, match=r"Matilda does not accept \['augmentation'\]"):
        mtb.scan("D11", "vertical", methods=["Matilda"],
                 params={"Matilda": {"augmentation": False}}, verbose=False)


def test_scmdc_settings_without_effect_are_not_offered():
    # the script only pretrains and encodes; filter1/filter2 fail on its np.matrix data
    tunable = mtb.params_for("scMDC", "vertical")["tunable"]
    dead = {"tau", "gamma", "phi2", "n_clusters", "resolution", "n_neighbors",
            "filter1", "filter2", "f1", "f2"}
    assert not dead & set(tunable)
    assert {"pretrain_epochs", "batch_size", "sigma1", "sigma2", "cutoff", "phi1"} <= set(tunable)
    with pytest.raises(KeyError, match=r"scMDC does not accept \['filter1'\]"):
        mtb.scan("D11", "vertical", methods=["scMDC"],
                 params={"scMDC": {"filter1": True}}, verbose=False)


def test_a_switch_is_present_only_for_true(tmp_path):
    inputs = {"rna": "/d/rna.h5", "atac": "/d/atac.h5"}
    argv = _dry("MIRA", "vertical", inputs, {"filter_atac_cells": True}, tmp_path)
    assert argv[-1] == "--filter_atac_cells"
    argv = _dry("MIRA", "vertical", inputs, {"filter_atac_cells": False}, tmp_path)
    assert "--filter_atac_cells" not in argv


def test_a_switch_or_bool_refuses_another_value(tmp_path):
    with pytest.raises(TypeError, match=r"MIRA: filter_atac_cells takes True or False, got 1\."):
        _dry("MIRA", "vertical", {"rna": "/d/rna.h5", "atac": "/d/atac.h5"},
             {"filter_atac_cells": 1}, tmp_path)
    with pytest.raises(TypeError, match=r"early_stopping takes True or False, got 'False'"):
        _dry("totalVI", "vertical", {"rna": "/d/rna.h5", "adt": "/d/adt.h5"},
             {"early_stopping": "False"}, tmp_path)


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
        mtb.scan("D52mini", "cross", methods=["scMDC"], params={"scMDC": {"nbatch": 5}}, verbose=False)


# Y38: one default, the one a run uses
def test_tunable_default_is_what_a_run_uses():
    # r3 (X64): no_cuda is the package's switch and left tunable; 0.4.0: one
    # default per key, no script_default copy
    t = mtb.params_for("scMM", "vertical", ["rna", "atac"])["tunable"]
    assert "no_cuda" not in t and set(t["epochs"]) == {"type", "default"}
