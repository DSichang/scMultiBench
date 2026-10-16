"""``sweep`` checks the parameter name before anything runs and has ``dry_run``.

A misspelled parameter of a method with several variants (Matilda
``learning_rate``) became a FAIL row per value after each setting was
launched; sciPENN's got its own message. Both now raise ``run_all``'s
KeyError first, and ``sweep(dry_run=True)`` returns the planned settings.
"""
import inspect

import pytest

import multibench as mtb
from multibench import workflow


def test_dry_run_is_a_keyword_with_default_false():
    p = inspect.signature(mtb.sweep).parameters["dry_run"]
    assert p.kind is p.KEYWORD_ONLY and p.default is False


def test_an_unknown_param_raises_before_any_setting_runs(monkeypatch, tmp_path):
    real = workflow.run_all

    def no_real_run(*a, dry_run=False, **kw):
        assert dry_run, "a setting was launched"
        return real(*a, dry_run=dry_run, **kw)
    monkeypatch.setattr(workflow, "run_all", no_real_run)
    with pytest.raises(KeyError) as e:
        mtb.sweep("D11", "vertical", "Matilda", "learning_rate", [1e-3, 1e-2],
                  out_dir=tmp_path / "sw", verbose=False)
    assert e.value.args[0].startswith("Matilda does not accept ['learning_rate']; it "
                                      "accepts ['augmentation', ")
    with pytest.raises(KeyError) as e:
        mtb.sweep("D11", "vertical", "sciPENN", "epochz", [1], out_dir=tmp_path / "sw",
                  verbose=False)
    assert e.value.args[0].startswith("sciPENN does not accept ['epochz']; it accepts "
                                      "['seed']. An empty set")
    assert not (tmp_path / "sw").exists()


def test_dry_run_returns_one_plan_per_setting(tmp_path, capsys):
    df = mtb.sweep("D11", "vertical", "Matilda", "lr", [1e-3, 1e-2],
                   modalities=["rna", "adt"], out_dir=tmp_path / "sw", dry_run=True)
    assert list(df.columns[:2]) == ["lr", "method"]
    assert df["lr"].tolist() == [1e-3, 1e-2]
    assert df.command.iloc[0].endswith("--lr 0.001") and "lr_0p001" in df.command.iloc[0]
    assert df.command.iloc[1].endswith("--lr 0.01") and "lr_0p01" in df.command.iloc[1]
    assert "[sweep] Dry run: 2 settings of lr for Matilda on D11 (vertical). Nothing was " \
           "run." in capsys.readouterr().out
    assert not (tmp_path / "sw").exists()
