"""``sweep`` checks the parameter name before anything runs.

A misspelled parameter of a method with several variants (Matilda
``learning_rate``) became a FAIL row per value after each setting was
launched; sciPENN's got its own message. Both now raise ``run_all``'s
KeyError first.

0.4.0: ``sweep`` is deprecated (a loop over ``mtb.run_all(params=...)``), and
its ``dry_run=``, new on the 0.4 branch, never shipped.
"""
import inspect

import pytest

import multibench as mtb
from multibench import workflow

pytestmark = pytest.mark.filterwarnings("ignore:sweep is deprecated:FutureWarning")


def test_sweep_is_deprecated_without_dry_run():
    assert "sweep" not in mtb.__all__
    assert "dry_run" not in inspect.signature(mtb.sweep).parameters
    with pytest.warns(FutureWarning, match=r"^sweep is deprecated and will be removed "
                      r"in 0\.5\. Use a loop over mtb\.run_all\(params=\.\.\.\)\.$") as rec:
        with pytest.raises(TypeError, match=r"^values takes a list"):
            mtb.sweep("D11", "vertical", "Matilda", "epochs", 5, out_dir="x")
    assert len([w for w in rec if issubclass(w.category, FutureWarning)]) == 1


def test_an_unknown_param_raises_before_any_setting_runs(monkeypatch, tmp_path):
    real = workflow._run_all

    def no_real_run(*a, dry_run=False, **kw):
        assert dry_run, "a setting was launched"
        return real(*a, dry_run=dry_run, **kw)
    monkeypatch.setattr(workflow, "_run_all", no_real_run)
    with pytest.raises(KeyError) as e:
        mtb.sweep("D11", "vertical", "Matilda", "learning_rate", [1e-3, 1e-2],
                  out_dir=tmp_path / "sw", verbose=False)
    assert e.value.args[0].startswith("Matilda does not accept ['learning_rate']; it "
                                      "accepts ['augmentation', ")
    with pytest.raises(KeyError) as e:
        mtb.sweep("D11", "vertical", "sciPENN", "epochz", [1], out_dir=tmp_path / "sw",
                  verbose=False)
    assert e.value.args[0].startswith("sciPENN does not accept ['epochz']; it accepts "
                                      "['seed'].")
    assert not (tmp_path / "sw").exists()


def test_sweep_argument_mistakes_get_one_sentence():
    # Y16 (r3): run's argument order, and a scalar for values
    with pytest.raises(TypeError, match=r"^sweep takes \(dataset, category, method, "
                                        r"param, values\)\.$"):
        mtb.sweep("Matilda", "vertical", "D11", "epochs", [5], out_dir="x")
    with pytest.raises(TypeError, match=r"^values takes a list, e\.g\. \[5\]\.$"):
        mtb.sweep("D11", "vertical", "Matilda", "epochs", 5, out_dir="x")
