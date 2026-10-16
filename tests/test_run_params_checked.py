"""``run(params=)`` and ``run --param`` refuse a key the variant does not accept.

``run(params={"epoch": 3}, dry_run=True)`` returned a command ending
``--epoch 3`` (the script would stop on it) and ``run --param nonsense=1``
did the same, though ``run --help`` says such a key is rejected. Both raise
now, with the message ``run_all`` gives.
"""
from __future__ import annotations

import pytest

import multibench as mtb
from multibench import cli

INPUTS = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
ACCEPTS = ("it accepts ['augmentation', 'batch_size', 'device', 'epochs', 'hidden_adt', "
           "'hidden_atac', 'hidden_rna', 'lr', 'seed', 'z_dim']. An empty set means it "
           "hardcodes its hyperparameters upstream.")


def test_run_dry_run_refuses_an_unknown_key():
    with pytest.raises(KeyError) as e:
        mtb.run("Matilda", "vertical", inputs=INPUTS, out_dir="/o", params={"epoch": 3},
                dry_run=True)
    assert e.value.args[0] == f"Matilda does not accept ['epoch']; {ACCEPTS}"


def test_a_real_run_refuses_before_anything_is_written(tmp_path):
    with pytest.raises(KeyError, match=r"does not accept \['epoch'\]"):
        mtb.run("Matilda", "vertical", inputs=INPUTS, out_dir=str(tmp_path / "o"),
                params={"epoch": 3})
    assert not (tmp_path / "o").exists()


def test_a_known_key_still_reaches_the_command():
    argv = mtb.run("Matilda", "vertical", inputs=INPUTS, out_dir="/o", params={"epochs": 3},
                   dry_run=True)
    assert argv[-2:] == ["--epochs", "3"] or "--epochs" in argv


def test_cli_param_refuses_an_unknown_key(capsys):
    rc = cli.main(["run", "--method", "Matilda", "--category", "vertical", "--out-dir", "/o",
                   "--input", "rna=/d/rna.h5", "--input", "adt=/d/adt.h5",
                   "--input", "cty=/d/cty.csv", "--param", "nonsense=1", "--dry-run"])
    assert rc == 1
    assert f"error: Matilda does not accept ['nonsense']; {ACCEPTS}" in capsys.readouterr().err


def test_run_all_says_the_same():
    with pytest.raises(KeyError) as e:
        mtb.run_all("D11", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                    params={"Matilda": {"epoch": 3}}, dry_run=True, verbose=False)
    assert e.value.args[0] == f"Matilda does not accept ['epoch']; {ACCEPTS}"
