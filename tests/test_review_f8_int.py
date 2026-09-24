"""Review of round 8 (integration): the findings applied in the package.

- R8-01: a sweep saved before the fix keeps ARI -0.0 in batch_result.json;
  ``summary`` shows 0.0 for it.
- R8-07: the grey-column warning under ``aggregate="summary"`` gives each
  mean rank its own sentence, with no serial comma.
- R8-05: off Linux, the "No method can run" list counts only the rows whose
  files are in the folder, as for SKIPPED records; one line counts the rest.
  A variant that reads a folder is listed as ``scBridge``, not
  ``scBridge ((data_dir))``.
- describe_layout and list_categories join no two facts with ';'.
- labels_for names a missing dataset folder as scan does; load_results
  names missing tables in plain sentences without the package path.
- Old-style messages: an unknown modality, an unknown metrics= token, no
  conda, an empty bar table, the empty CLI table and the deprecated
  evaluate flags.
- The fetch error with a ref keeps its subject in run_all's FAIL line.
- The ``env plan`` and ``env install`` descriptions are short sentences
  (the option help of ``scan`` and ``convert``: tests/test_study_r8_other.py);
  'preflight' is gone.
- tools/gen_tut.py picks the label-order example also without the method
  scripts, and fails loudly when no method fits.
"""
import importlib.util
import inspect
import json
import math
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench import workflow as W
from multibench.data import results as RS
from multibench.engine import envs
from multibench.engine import runner as R
from tests.test_docs_r4 import _flat, needs_docs
from tests.test_study_r8_other import _depth, _fetch_error, _plain, _subparser

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
MACOS = "Methods run only on Linux, and this computer runs macOS."


def _doc(obj) -> str:
    return " ".join((inspect.getdoc(obj) or "").split())


# ======================================================================= R8-01
def test_a_stored_minus_zero_ari_reads_zero_in_summary(tmp_path):
    rec = {"method": "VIMCCA", "status": "CHAIN_OK", "run_sec": 1.0,
           "metrics": {"ARI": -0.0, "NMI": 0.0101}, "out_dir": str(tmp_path / "VIMCCA_X")}
    (tmp_path / "batch_result.json").write_text(json.dumps(
        {"dataset": "X", "category": "vertical", "records": [rec]}))
    assert '"ARI": -0.0' in (tmp_path / "batch_result.json").read_text()
    sm = mtb.load_batch(tmp_path).summary
    ari = sm.loc[0, "ARI"]
    assert ari == 0.0 and math.copysign(1, ari) == 1
    assert "-0.0" not in sm.to_csv(index=False)
    assert sm.loc[0, "NMI"] == 0.0101


# ======================================================================= R8-07
def _two_tie_groups():
    vals = {"ARI": [.9, .1, .2, .8, .5, .4], "NMI": [.9, .1, .1, .9, .5, .5],
            "GC": [.5] * 6}
    keys = [("A", "D1"), ("B", "D1"), ("A", "D2"), ("B", "D2"), ("A", "D3"), ("B", "D3")]
    return pd.DataFrame([{"method": m, "dataset": d, "metric": k, "value": v,
                          "category": "cross"}
                         for k, vs in vals.items() for (m, d), v in zip(keys, vs)])


def test_each_mean_rank_gets_its_own_sentence():
    B = importlib.import_module("multibench.plot.bubble")
    msg = B._no_comparison_message(["A", "B"], {"NMI": 1.6667, "GC": 2.0, "ARI": 1.6667},
                                   aggregate="summary")
    assert msg == ("Every method has mean rank 1.67 in NMI and ARI. Every method has "
                   "mean rank 2 in GC. Those columns are grey.")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mtb.plot.bubble(_two_tie_groups(), aggregate="summary")
    (grey,) = [str(w.message) for w in rec if "grey" in str(w.message)]
    assert grey == ("Every method has mean rank 1.67 in NMI. Every method has mean "
                    "rank 2 in GC. Those columns are grey.")
    for text in (msg, grey):
        assert ", and" not in text and ";" not in text


# ======================================================================= R8-05
@pytest.fixture
def macos_no_gpu(monkeypatch):
    monkeypatch.setattr(R, "linux_only_sentence", lambda: MACOS)
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)


def _nothing_runnable(dataset, category, **kw):
    with pytest.raises(ValueError) as e:
        mtb.run_all(dataset, category, out_dir="out", verbose=False, **kw)
    return str(e.value)


def test_the_macos_list_counts_only_rows_whose_files_are_present(macos_no_gpu):
    msg = _nothing_runnable("D11", "vertical")
    plan = mtb.scan("D11", "vertical", verbose=False)
    lacking = int((~plan["files_ok"]).sum())
    have = len(plan) - lacking
    lines = msg.splitlines()
    assert lines[2] == f"1 of {have} methods is also blocked by something else:"
    assert lines[3].startswith("  moETM (rna+adt): moETM needs an NVIDIA GPU")
    assert lines[4] == f"{lacking} rows need files that D11 does not have."
    assert lines[5].startswith("mtb.scan('D11', 'vertical') shows every row.")
    assert "atac" not in msg
    # with modalities= the rows the folder cannot have are not in the scan
    msg = _nothing_runnable("D11", "vertical", modalities=["rna", "adt"])
    assert "need files" not in msg and msg.splitlines()[2] == lines[2]


def test_a_mosaic_folder_lists_no_file_it_cannot_have(macos_no_gpu):
    msg = _nothing_runnable("D46", "mosaic")
    listed = [l for l in msg.splitlines() if l.startswith("  ")]
    assert not [l for l in listed if "atac3.h5" in l or "adt2.h5" in l], listed
    assert re.search(r"^\d+ rows? needs? files that D46 does not have\.$", msg, re.M), msg


def test_a_folder_variant_is_listed_by_its_method_name(macos_no_gpu):
    for kw in ({}, {"methods": ["scBridge", "SCALEX"]}):
        msg = _nothing_runnable("D28", "diagonal", **kw)
        assert "((" not in msg
        assert "\n  scBridge: scBridge needs an NVIDIA GPU" in msg, msg


# ============================================================ describe_layout
@pytest.mark.parametrize("cli_mode", [False, True])
@pytest.mark.parametrize("category", [None, "vertical", "diagonal", "mosaic", "cross"])
def test_the_layout_text_joins_no_two_facts_with_a_semicolon(monkeypatch, category,
                                                             cli_mode):
    monkeypatch.setattr(config, "_CLI", cli_mode)
    joined = [l for l in mtb.describe_layout(category).splitlines() if ";" in l]
    assert not joined, joined


def test_the_category_descriptions_have_no_semicolon():
    assert not [c for c, d in mtb.list_categories().items() if ";" in d]


# ================================================= missing folders and tables
@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "D11").symlink_to(ROOT / "data" / "D11")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_labels_for_names_a_missing_folder_as_scan_does(in_tmp):
    with pytest.raises(FileNotFoundError) as e:
        mtb.labels_for("MYCITE", "vertical", "Matilda", data_path="data")
    with pytest.raises(FileNotFoundError) as s:
        mtb.scan("MYCITE", "vertical", data_path="data", verbose=False)
    assert str(e.value) == str(s.value)
    assert str(e.value).startswith("The folder data/MYCITE does not exist. data holds D11.")


@pytest.mark.parametrize("kw,start", [
    (dict(category="vertical", clustering="louvain"),
     "The published tables of vertical have no louvain variant. They have default only."),
    (dict(category="cross", dataset="D52s", source="published"),
     "The published tables of cross hold D52, not D52s. Pass result_path= to read "
     "another results folder."),
    (dict(category="cross", dataset=["D52", "D52s"]),
     "The published tables of cross hold D52, not D52s."),
    (dict(category="cross", dataset="D99", source="rerun"),
     "The re-run tables of cross hold D52 and D52s, not D99."),
    (dict(category="cross", dataset="D99", source="both"),
     "The published tables of cross hold D52, not D99. The re-run tables of cross hold "
     "D52 and D52s, not D99. Pass result_path= to read another results folder."),
    (dict(dataset="D99", source="rerun"), "The re-run tables hold D11, D11s, "),
])
def test_missing_tables_are_named_in_plain_sentences(kw, start):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FileNotFoundError) as e:
            mtb.load_results(**kw)
    msg = str(e.value)
    assert msg.startswith(start), msg
    assert msg.count("Pass result_path=") <= 1
    _plain(msg)
    assert "scib_metric" not in msg and str(RS._SHIPPED_BASE) not in msg


def test_the_cli_names_the_result_path_flag(capsys, tmp_path):
    rc = cli.main(["plot", "bubble", "--category", "cross", "--dataset", "D53",
                   "--out", str(tmp_path / "x.png")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "Pass --result-path to read another results folder." in err, err


# ====================================================== old-style messages
def _raised(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fn(*args, **kw)
        except Exception as e:  # noqa: BLE001 - the text is checked
            return e
    raise AssertionError("no error")


def test_an_unknown_modality_is_named_in_sentences():
    e = _raised(mtb.scan, "D11", "vertical", modalities=["rna", "adtt"], verbose=False)
    assert isinstance(e, ValueError)
    assert str(e) == ("Unknown modality adtt. The tokens are rna, adt (or protein) and "
                      "atac. atac_peak (or peak, peaks) and atac_gas (or gas, "
                      "gene_activity) name the ATAC form a method reads. rna1, adt1, "
                      "atac2 ... name numbered batches.")
    _plain(str(e))
    e = _raised(mtb.io.to_canonical, None, "x.h5", modality="bogus")
    assert str(e) == ("Unknown modality bogus. The modalities are rna, adt (or protein), "
                      "atac, atac_peak (or peak) and atac_gas (or gas, gene_activity).")


def test_a_metrics_token_error_is_plain_outside_its_examples():
    emb, lab = np.random.default_rng(0).normal(size=(6, 2)), list("aabbcc")
    e = _raised(mtb.evaluate, emb, labels=lab, metrics="ARI")
    assert isinstance(e, ValueError)
    assert str(e) == ("Unknown metrics= token ARI. Pass 'all', 'clustering' or 'batch', "
                      "or a list of metric codes such as metrics=['ARI', 'NMI']. A "
                      "single code goes in a list: metrics=['ARI'].")
    _plain(str(e), "metrics=['ARI', 'NMI']", "metrics=['ARI']")
    e = _raised(mtb.evaluate, emb, labels=lab, metrics="dimension_reduction")
    _plain(str(e), "metrics=['ARI', 'NMI']")
    e = _raised(mtb.evaluate, emb, labels=lab, metrics=3)
    assert isinstance(e, TypeError)
    assert str(e) == ("metrics= takes None, 'all', 'clustering', 'batch' or a list of "
                      "metric codes. It got an int.")
    _plain(str(e), lead="metrics=")
    e = _raised(mtb.evaluate, emb, labels=lab, metrics=[])
    _plain(str(e), "metrics=[]", lead=" selects nothing")


def test_the_no_conda_error_is_plain(monkeypatch):
    monkeypatch.setattr(envs.shutil, "which", lambda _: None)
    e = _raised(envs._run_all, [["mamba", "create"]])
    assert isinstance(e, RuntimeError)
    assert str(e).startswith("Neither conda nor mamba is on this machine. Method "
                             "environments cannot be built here.")
    _plain(str(e))


def test_an_empty_bar_table_is_plain():
    e = _raised(mtb.plot.bar, pd.DataFrame(columns=["method", "metric", "value"]))
    assert isinstance(e, ValueError)
    assert str(e).startswith("long_df has no rows to plot. ")
    _plain(str(e), lead="long_df")


def test_the_empty_cli_table_is_a_sentence(capsys):
    cli._print_frame(pd.DataFrame(columns=["method", "modalities", "files_ok"]))
    out = capsys.readouterr().out.strip()
    assert out == "The table has no rows. Its columns are method, modalities, files_ok."
    _plain(out)


def test_the_deprecated_evaluate_flags_warn_in_sentences(monkeypatch, capsys):
    from tests.test_cli_parity import _wide
    monkeypatch.setattr(mtb, "evaluate", lambda **kw: _wide())
    err = ""
    for flag in (["--only", "ARI"], ["--task", "clustering"]):
        assert cli.main(["evaluate", "--output", "e.h5", "--labels", "l.csv", *flag]) == 0
        err += capsys.readouterr().err
    assert "warning: --only is deprecated. Use --metrics.\n" in err, err
    assert "warning: --task is deprecated. Use --metrics clustering.\n" in err, err
    assert "deprecated;" not in err


# ================================================================ fetch error
def test_a_ref_fetch_error_keeps_its_subject_in_the_fail_line(monkeypatch, tmp_path):
    sha = "0123456789abcdef0123456789abcdef01234567"
    e = _fetch_error(monkeypatch, tmp_path, ref=sha)
    assert str(e) == (f"Could not fetch the method scripts at '{sha}'. There is no "
                      "network, or no such commit or tag. Offline, copy a fetched "
                      "scripts folder and set MULTIBENCH_REPO_PATH.")
    shown = W._error_tail(f"{type(e).__name__}: {e}")
    assert shown.startswith("Could not fetch") or "Could not fetch" in shown[:20], shown
    assert sha in shown
    _plain(str(e))


# ============================================================ CLI help
def test_env_plan_and_install_descriptions_are_short_sentences():
    env = _subparser("env")
    sub = next(a for a in env._actions if a.__class__.__name__ == "_SubParsersAction")
    for name in ("plan", "install"):
        text = sub.choices[name].description
        assert ";" not in text and _depth(text) <= 1, (name, text)
        assert not re.search(r"\w/\w", text), (name, text)
    assert "--run refuses unless --force is given." in sub.choices["install"].description


def test_preflight_is_gone_from_what_users_read():
    env = _subparser("env")
    sub = next(a for a in env._actions if a.__class__.__name__ == "_SubParsersAction")
    helps = [a.help for a in sub._choices_actions]
    assert "check which environments are installed or need building" in helps
    texts = helps + [_doc(W.run_all), _doc(mtb.inputs_for), _doc(W.scan)]
    assert not [t for t in texts if "preflight" in t.lower()]


# ============================================================ gen_tut
def _gen_tut():
    spec = importlib.util.spec_from_file_location("gen_tut", ROOT / "tools" / "gen_tut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _label_example(cells):
    src = "\n".join(c.source for c in cells)
    return re.findall(r'print\("(\w+):", list\(mtb\.labels_for', src)


@pytest.fixture(scope="module")
def gen():
    return _gen_tut()


def test_the_label_order_example_ignores_a_missing_scripts_checkout(gen, monkeypatch,
                                                                   tmp_path):
    monkeypatch.setattr(R, "_repo_root_no_fetch", lambda: tmp_path / "no_scripts")
    plan = mtb.scan("D28", "diagonal", methods=["uniPort"], verbose=False)
    assert plan.loc[0, "caveat"].startswith(R.SCRIPTS_NOT_HERE)
    cells = gen.build_tutorial("diagonal", gen.SCEN["diagonal"])
    assert _label_example(cells) == ["uniPort"]


def test_the_label_order_example_fails_loudly_when_no_method_fits(gen, monkeypatch):
    real = mtb.scan

    def every_row_has_a_caveat(*args, **kw):
        df = real(*args, **kw)
        if kw.get("methods") is not None:
            df = df.assign(caveat="The method needs something else.")
        return df
    monkeypatch.setattr(mtb, "scan", every_row_has_a_caveat)
    with pytest.raises(SystemExit, match="pick the label-order example by hand"):
        gen.build_tutorial("diagonal", gen.SCEN["diagonal"])


# ============================================================ Changes page

@needs_docs
def test_changes_page_follows_the_review_of_round_8(capsys, monkeypatch):
    text = _flat("changes.md")
    assert text.count("- Reworded:") == 1
    for phrase in (
            "- `bubble` widens a family pill to fit its header, and a figure of one to "
            "three metrics to fit its row labels and key. The height is unchanged.",
            "0.3.1 dropped that method. `multibench run-all` exits with `1`.",
            "- For `batch=`, `labels=` and these CSVs, ids that are missing, repeated or "
            "not cells of the dataset raise `ValueError`.",
            "copy a fetched scripts folder and set `MULTIBENCH_REPO_PATH`.",
            "- Missing batch labels get `metrics='all' needs batch labels"):
        assert phrase in text, phrase
    for gone in ("The commands exit with `1`.", "In these three cases",
                 "family headers and key fit", "a scripts checkout"):
        assert gone not in text, gone
    # the deprecation line the page quotes is the one the command prints
    from tests.test_cli_parity import _wide
    monkeypatch.setattr(mtb, "evaluate", lambda **kw: _wide())
    cli.main(["evaluate", "--output", "e.h5", "--labels", "l.csv", "--only", "ARI"])
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert f"(stderr: `{line}`)" in text, line
