"""The tutorials after fix round 5 of the student study.

- R5-11e: ``evaluate`` refuses a labels dict out of the default order with
  "the keys [...] are not in the default order". The tutorials said such a
  dict "is read in the default order", which reads as if it were re-sorted;
  no tutorial says so.

R5-03 added a ``reason`` column to ``res.summary``; the guard below checks
the Troubleshooting sentence on the live package. (R5-05, the download error of ``fetch_outputs``, is pinned
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
GENERATED = [f"tutorial_{k}{suffix}" for k in GEN.TASKS for suffix in ("", "_all")
             if not suffix or GEN.has_all(k)]


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
@pytest.mark.parametrize("name", GENERATED + [E2E])
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
    scored, the same dict reordered raises, and a list of paths gives any
    other order (0.4.0: the message names the list, not the deprecated
    ``label_order=``)."""
    import multibench as mtb
    paths, emb = _two_files(tmp_path)
    built = {"cty1": paths["cty1"], "cty2": paths["cty2"]}
    scored = mtb.evaluate(emb, labels=built, metrics=["ASW"], verbose=False)
    assert "ASW" in set(scored.metric)
    reordered = {"cty2": paths["cty2"], "cty1": paths["cty1"]}
    with pytest.raises(ValueError, match=re.escape(
            "The label keys cty2 and cty1 are not in the default order. Pass a list of "
            "paths in cell order, or dataset=, category= and method=.")):
        mtb.evaluate(emb, labels=reordered, metrics=["ASW"], verbose=False)
    got = mtb.evaluate(emb, labels=[paths["cty2"], paths["cty1"]], metrics=["ASW"],
                       verbose=False)
    assert "ASW" in set(got.metric)


# --------------------------------------- R5-03: guard for Troubleshooting
def test_a_named_skipped_method_has_its_reason_in_summary(tmp_path, monkeypatch):
    """The vertical run cell's call with one environment missing: the method
    is SKIPPED, and the summary's last column, ``reason``, holds why (the
    Troubleshooting sentence)."""
    import multibench as mtb
    from multibench import config
    from multibench import workflow as W
    from multibench.engine import envs, registry
    if not (config.data_path / "D11").is_dir():
        pytest.skip("D11 is not on disk")
    trio = GEN.TASKS["vertical_rna_adt"]["methods"]
    missing = envs.group_for(trio[-1])
    every = frozenset(envs.group_for(m) for m in registry.list_methods())
    monkeypatch.setattr(W, "_installed_envs", lambda: every - {missing})
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    monkeypatch.setattr(config, "repo_path", tmp_path / "no_scripts")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)

    class _Res:
        output = np.zeros((2864, 5))
    monkeypatch.setattr(W, "_run", lambda *a, **k: _Res())
    res = _quiet(mtb.run, "D11", "vertical", tmp_path / "out", methods=trio,
                 evaluate=False, verbose=False)
    sm = res.summary
    assert list(sm.columns)[-1] == "reason"
    sm = sm.set_index("method")
    assert sm.loc[trio[-1], "status"] == "skipped"
    assert f"mtb.env.install([{trio[-1]!r}])" in sm.loc[trio[-1], "reason"]
    assert (sm.drop(index=trio[-1])["reason"] == "").all()
