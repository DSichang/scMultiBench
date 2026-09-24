"""The tutorials after fix round 5 of the student study.

Round 5 changed two things the notebooks show a reader:

- R5-05: a failed download in ``mtb.data.fetch_outputs`` raises ``OSError``
  whose message is several sentences: the URL, the HTTP status, and what to
  do next. The replacement helpers of the run cells printed that message
  inside parentheses, which left a three-sentence parenthetical ending in
  ``.)``. They now print it as its own sentence after the one-line summary.
- R5-11e: ``evaluate`` refuses a labels dict out of the default order with
  "the keys [...] are not in the default order". The tutorials said such a
  dict "is read in the default order", which reads as if it were re-sorted.
  They now say it must be in the default order.

R5-03 added a ``reason`` column to ``res.summary``; the Troubleshooting
sentence on ``res.failures`` stays true, and the guard below checks both on
the live package.

Each prose test reads the committed notebooks and checks the same fact on the
live package, so a later package change that makes the sentence untrue fails
here too.
"""
import ast
import importlib.util
import json
import re
import urllib.error
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


def _e2e_replacement():
    """Source of the end-to-end notebook's ``def replacement``."""
    src = next(s for kind, s in _cells(E2E) if kind == "code" and "def replacement(" in s)
    found = [ast.unparse(n) for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.FunctionDef) and n.name == "replacement"]
    assert len(found) == 1
    return found[0]


# ------------------------------------------- R5-05: the download error line
@pytest.fixture
def download_fails(tmp_path, monkeypatch):
    """No stored outputs on disk, and every download answers HTTP 404."""
    from multibench import config
    # the module, not the function multibench.data.fetch that shadows it
    F = importlib.import_module("multibench.data.fetch")

    def _404(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
    monkeypatch.setattr(F, "_download", _404)
    monkeypatch.setattr(config.DEFAULT, "data_path", tmp_path / "data")
    return tmp_path


def _replacement_line(out):
    lines = [ln for ln in out.splitlines() if ln.startswith("replacement:")]
    assert len(lines) == 1, out
    return lines[0]


def test_category_replacement_prints_the_download_error_as_a_sentence(download_fails, capsys):
    """The run cells of the four category tutorials: the stored table stands
    in, and the OSError message follows the summary as its own sentences."""
    import multibench as mtb
    trio = GEN.SCEN["vertical"]["own_trio"]
    ns = {"mtb": mtb, "CATEGORY": "vertical"}
    exec(GEN.STORED_SWEEP_FN, ns)
    exec(GEN.REPLACEMENT_FN, ns)
    res = _quiet(ns["replacement"], "D11", trio, stored=("D11", trio))
    assert isinstance(res, mtb.BatchResult) and set(res.summary.status) == {"STORED"}
    line = _replacement_line(capsys.readouterr().out)
    assert line.startswith(
        "replacement: the package's stored metric table. OSError from fetch_outputs: "
        "Could not download the stored outputs of D11 from https://"), line
    assert "(HTTP 404)" in line
    # the package's message ends the line; no parenthesis wraps it
    dest = download_fails / "data" / "outputs" / "D11"
    assert line.endswith(f"unpack it into {dest}, so that "
                         f"{dest / 'batch_result.json'} exists."), line
    assert not line.endswith(")"), line


def test_end_to_end_replacement_prints_the_download_error_as_a_sentence(download_fails, capsys):
    import h5py
    import multibench as mtb
    ns = {"mtb": mtb, "Path": Path, "h5py": h5py}
    exec(_e2e_replacement(), ns)
    assert _quiet(ns["replacement"], "D11", "Matilda") is None
    line = _replacement_line(capsys.readouterr().out)
    assert line.startswith(
        "replacement: the package's stored scores for Matilda. OSError from fetch_outputs: "
        "Could not download the stored outputs of D11 from https://"), line
    assert line.endswith("batch_result.json exists."), line


@pytest.mark.parametrize("src", [GEN.REPLACEMENT_FN, "E2E"], ids=["category", "end_to_end"])
def test_the_replacement_line_wraps_no_exception_in_parentheses(src):
    src = _e2e_replacement() if src == "E2E" else src
    printed = [ast.unparse(n) for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Call) and ast.unparse(n.func) == "print"
               and "type(e).__name__" in ast.unparse(n)]
    assert len(printed) == 1
    assert "({type(e).__name__}" not in printed[0] and "{e})" not in printed[0], printed[0]


# ------------------------------------- R5-11e: a dict in the default order
DICT_SENTENCE = "A dict you build or reorder yourself must be in the default order"


@pytest.mark.parametrize("name", MULTI_FILE + [E2E])
def test_the_label_dict_sentence_says_the_default_order_is_required(name):
    md = _markdown(name)
    assert DICT_SENTENCE in md
    assert "is read in the default order" not in md


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
            "labels: the keys ['cty2', 'cty1'] are not in the default order.")):
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
    trio = GEN.SCEN["vertical"]["own_trio"]
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
