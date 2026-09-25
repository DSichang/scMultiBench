"""The tutorials after fix round 5 of the student study.

- R5-11e: ``evaluate`` refuses a labels dict out of the default order with
  "the keys [...] are not in the default order". The tutorials said such a
  dict "is read in the default order", which reads as if it were re-sorted;
  no tutorial says so.

R5-03 added a ``reason`` column to ``res.summary``; the Troubleshooting
sentence on ``res.failures`` stays true, and the guard below checks both on
the live package. (R5-05, the download error of ``fetch_outputs``, is pinned
in tests/test_study_r5_cfg.py.)
"""
import importlib.util
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tut()
E2E = "tutorial_end_to_end"
# the tutorials whose dataset has several label files (vertical has one, cty)
MULTI_FILE = [f"tutorial_{c}" for c in GEN.SCEN if c != "vertical"]


def _cells(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def _markdown(name):
    return "\n".join(src for kind, src in _cells(name) if kind == "markdown")


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ------------------------------------- R5-11e: a dict in the default order
@pytest.mark.parametrize("name", MULTI_FILE + [E2E])
def test_no_tutorial_says_a_label_dict_is_read_in_the_default_order(name):
    assert "is read in the default order" not in _markdown(name)


def _two_files(tmp_path):
    rng = np.random.default_rng(0)
    paths = {}
    for k, n in (("cty1", 30), ("cty2", 20)):
        paths[k] = tmp_path / f"{k}.csv"
        pd.DataFrame({"x": rng.choice(["T", "B"], n)}).to_csv(paths[k], index=False)
    return paths, rng.normal(size=(50, 4))


def test_evaluate_refuses_a_dict_out_of_the_default_order(tmp_path):
    """The sentence on the live package: a dict built in the default order is
    scored, the same dict reordered raises, and ``label_order=`` names any
    other order."""
    import multibench as mtb
    paths, emb = _two_files(tmp_path)
    built = {"cty1": paths["cty1"], "cty2": paths["cty2"]}
    assert "ASW" in _quiet(mtb.evaluate, emb, labels=built, metrics=["ASW"], verbose=False).index
    reordered = {"cty2": paths["cty2"], "cty1": paths["cty1"]}
    with pytest.raises(ValueError, match=re.escape(
            "The label keys cty2 and cty1 are not in the default order.")):
        _quiet(mtb.evaluate, emb, labels=reordered, metrics=["ASW"], verbose=False)
    got = _quiet(mtb.evaluate, emb, labels=reordered, label_order=["cty2", "cty1"],
                 metrics=["ASW"], verbose=False)
    assert "ASW" in got.index


# --------------------------------------- R5-03: guard for Troubleshooting
def test_a_named_skipped_method_has_its_reason_in_summary_and_failures(tmp_path, monkeypatch):
    """The vertical run cell's call with one environment missing: the method
    is SKIPPED, ``res.failures`` holds the reason (the Troubleshooting
    sentence), and the summary's last column, ``reason``, holds the same
    text."""
    import multibench as mtb
    from multibench import config
    from multibench import workflow as W
    from multibench.engine import envs, registry
    if not (config.DEFAULT.data_path / "D11").is_dir():
        pytest.skip("D11 is not on disk")
    trio = GEN.SCEN["vertical"]["methods"]
    missing = envs.group_for(trio[-1])
    every = frozenset(envs.group_for(m) for m in registry.list_methods())
    monkeypatch.setattr(W, "_installed_envs", lambda: every - {missing})
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)

    class _Res:
        output = np.zeros((2864, 5))
    monkeypatch.setattr(W, "_run", lambda *a, **k: _Res())
    res = _quiet(mtb.run_all, "D11", "vertical", tmp_path / "out", methods=trio,
                 evaluate=False, verbose=False)
    sm = res.summary
    assert list(sm.columns)[-1] == "reason"
    sm = sm.set_index("method")
    assert sm.loc[trio[-1], "status"] == "SKIPPED"
    assert f"multibench env install --methods {trio[-1]}" in sm.loc[trio[-1], "reason"]
    assert (sm.drop(index=trio[-1])["reason"] == "").all()
    fails = res.failures.set_index("method")
    assert fails.loc[trio[-1], "error"] == sm.loc[trio[-1], "reason"]
