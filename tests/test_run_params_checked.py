"""``run(params=)`` and ``run --param`` refuse a key the variant does not accept.

``run(params={"epoch": 3}, dry_run=True)`` returned a command ending
``--epoch 3`` (the script would stop on it) and ``run --param nonsense=1``
did the same, though ``run --help`` says such a key is rejected. Both raise
now, with the message ``run`` gives.
"""
from __future__ import annotations

import pytest

import multibench as mtb

INPUTS = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
# r3: device is the package's (X64); the list alone, without the empty-set note (Z16)
ACCEPTS = ("it accepts ['augmentation', 'batch_size', 'epochs', 'hidden_adt', "
           "'hidden_atac', 'hidden_rna', 'lr', 'seed', 'z_dim'].")


def test_run_dry_run_refuses_an_unknown_key():
    with pytest.raises(KeyError) as e:
        mtb.engine.runner.preview("Matilda", "vertical", inputs=INPUTS, out_dir="/o", params={"epoch": 3})[0]
    assert e.value.args[0] == f"Matilda does not accept ['epoch']; {ACCEPTS}"


def test_a_real_run_refuses_before_anything_is_written(tmp_path):
    with pytest.raises(KeyError, match=r"does not accept \['epoch'\]"):
        mtb.engine.runner.run("Matilda", "vertical", inputs=INPUTS, out_dir=str(tmp_path / "o"),
                params={"epoch": 3})
    assert not (tmp_path / "o").exists()


def test_a_known_key_still_reaches_the_command():
    argv = mtb.engine.runner.preview("Matilda", "vertical", inputs=INPUTS, out_dir="/o", params={"epochs": 3})[0]
    assert argv[-2:] == ["--epochs", "3"] or "--epochs" in argv


def test_run_all_says_the_same():
    with pytest.raises(KeyError) as e:
        mtb.scan("D11", "vertical", methods=["Matilda"], modalities=["rna", "adt"], params={"Matilda": {"epoch": 3}}, verbose=False)
    assert e.value.args[0] == f"Matilda does not accept ['epoch']; {ACCEPTS}"


def test_a_method_without_hyperparameters_says_so():
    # Z16 (r3): the hardcoded-upstream sentence only where the set is empty
    with pytest.raises(KeyError) as e:
        mtb.scan("D11", "vertical", methods=["totalVI"], modalities=["rna", "adt"], params={"totalVI": {"epochs": 3}}, verbose=False)
    assert e.value.args[0] == ("totalVI does not accept ['epochs']; totalVI takes no "
                               "hyperparameters (its script hardcodes them).")
