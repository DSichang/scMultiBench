"""Round 8 of the virtual-student study, package work package 'wf' (workflow.py).

R8-01  the ``run_all`` / ``rescore`` result line ended with a bare number, the
       ARI, and printed ``-0.0`` for a tiny negative ARI. It now reads
       ``ARI 0.000``; the records round negative zero to ``0.0``.
R8-02  a peak token warned that scBridge was left out, although scBridge
       reads gene activity and the token excludes it anyway.
R8-04  a ``methods`` list that mixed a method of the category with one that has
       no variant there dropped the second one silently.
R8-05  off Linux, the "No method can run" error repeated the platform
       sentence under every method and hid the other blocks.
R8-06  template-like docstring passages; ``label_order_confidence`` was an
       object column of ``None`` when every row was blank.
"""
import inspect
import math
import warnings
from pathlib import Path

import pandas as pd
import pytest

import multibench as mtb
from multibench import cli
from multibench import workflow as W
from multibench.engine import envs, runner as R
from tests.test_f4_cli import working  # noqa: F401 - the stand-in Matilda env fixture


def _doc(obj) -> str:
    return " ".join((inspect.getdoc(obj) or "").split())


# ============================================================ R8-06
def _record(method, ari, cands=None):
    rec = {"method": method, "status": "CHAIN_OK", "run_sec": 1.0,
           "output_kind": "embedding", "emb_shape": [10, 2], "n_tunable": 0,
           "metrics": {"ARI": ari, "NMI": 0.5}, "labels_used": ["cty.csv"]}
    if cands:
        rec["label_order_candidates"] = cands
    return rec


def test_an_all_blank_confidence_column_is_float():
    res = W.BatchResult([_record("A", 0.4), _record("B", 0.6)], "D11", "vertical")
    col = res.summary["label_order_confidence"]
    assert col.dtype == "float64", col.dtype
    assert col.isna().all()
    assert not (col > 0.5).any()
    assert list(res.summary["label_order_note"]) == ["single ordering"] * 2


def test_a_mixed_confidence_column_stays_float():
    cands = [{"order": ["a", "b"], "ARI": 0.6}, {"order": ["b", "a"], "ARI": 0.1}]
    res = W.BatchResult([_record("A", 0.6, cands), _record("B", 0.6)], "D11", "vertical")
    col = res.summary["label_order_confidence"]
    assert col.dtype == "float64"
    assert col.isna().tolist() == [False, True]


def test_the_empty_summary_has_a_float_confidence_column():
    res = W.BatchResult([], "D11", "vertical")
    assert res.summary["label_order_confidence"].dtype == "float64"


@pytest.mark.parametrize("obj,gone", [
    (W.BatchResult.rescore, ("Re-evaluate", "Typical uses", " / batch / ")),
    (W.load_batch, ("inspect, re-plot or re-score",)),
    (W.BatchResult, ("not by hand",)),
    (W.BatchResult.summary, ("behave", "stays numeric")),
])
def test_the_template_passages_are_gone(obj, gone):
    doc = _doc(obj.fget if isinstance(obj, property) else obj)
    assert not [g for g in gone if g in doc]


def test_the_rewritten_passages_keep_their_facts():
    assert inspect.getdoc(W.BatchResult.rescore).splitlines()[0] == (
        "Score the saved outputs again with new labels, batch or metrics.")
    assert "No method is re-run. Each record's embedding is read back from its " \
           "``out_dir``." in _doc(W.BatchResult.rescore)
    assert "``mtb.run_all`` and ``mtb.load_batch`` build it. It keeps one record per " \
           "method, which ``rescore`` and ``plot`` read." in _doc(W.BatchResult)
    assert ("The column is numeric, and a blank is ``NaN``. So ``> 0.5`` is ``False`` "
            "for it and ``.isna()`` finds it. It is blank in three cases, named by "
            "``label_order_note``:") in _doc(W.BatchResult.summary.fget)


# ============================================================ R8-01
def test_the_ari_tail_names_its_metric_and_never_prints_minus_zero():
    assert W._ari_tail({"metrics": {"ARI": 0.62894}}) == "ARI 0.629"
    assert W._ari_tail({"metrics": {"ARI": -0.00004}}) == "ARI 0.000"
    assert W._ari_tail({"metrics": {"ARI": -0.0}}) == "ARI 0.000"
    assert W._ari_tail({"metrics": {"ARI": -0.0123}}) == "ARI -0.012"
    for rec in ({}, {"metrics": None}, {"metrics": {"ASW": 0.5}},
                {"metrics": {"ARI": None}}, {"metrics": {"ARI": float("nan")}}):
        assert W._ari_tail(rec) == "", rec


@pytest.fixture
def near_zero_ari(monkeypatch):
    """Every evaluation that computes ARI reports -0.00004 (ARI at chance)."""
    real = W._evaluate

    def fake(*a, **k):
        val = real(*a, **k)
        if "ARI" in val.index:
            val.loc["ARI", "Value"] = -0.00004
        return val
    monkeypatch.setattr(W, "_evaluate", fake)


def _result_lines(text, prefix):
    return [line for line in text.splitlines() if line.startswith(prefix)]


@pytest.fixture
def swept(working, near_zero_ari, tmp_path, capsys):
    """A one-method sweep (Matilda stand-in) whose ARI is -0.00004."""
    out = tmp_path / "py"
    res = mtb.run_all("MYCITE", "vertical", out_dir=out, methods=["Matilda"],
                      modalities=["rna", "adt"], data_path=working)
    return res, out, capsys.readouterr().out


def test_the_run_all_line_ends_with_ari_0(swept):
    pytest.importorskip("scib")
    res, out, printed = swept
    (line,) = _result_lines(printed, "[run_all]   -> ")
    assert line.startswith("[run_all]   -> CHAIN_OK (") and line.endswith("s) ARI 0.000"), line
    # the record, the summary and summary.csv hold 0.0 with a positive sign
    for ari in (res.records[0]["metrics"]["ARI"], res.summary.loc[0, "ARI"],
                pd.read_csv(out / "summary.csv").loc[0, "ARI"]):
        assert ari == 0.0 and math.copysign(1, ari) == 1


def test_the_rescore_lines_name_the_ari_or_end_with_the_status(swept, capsys):
    pytest.importorskip("scib")
    _, out, _ = swept
    new = mtb.load_batch(out).rescore(verbose=True)
    (line,) = _result_lines(capsys.readouterr().out, "[rescore] Matilda -> ")
    assert line == "[rescore] Matilda -> CHAIN_OK ARI 0.000"
    assert math.copysign(1, new.summary.loc[0, "ARI"]) == 1
    # no ARI among the metrics: the line ends with the status
    mtb.load_batch(out).rescore(metrics=["ASW"], verbose=True)
    (line,) = _result_lines(capsys.readouterr().out, "[rescore] Matilda -> ")
    assert line == "[rescore] Matilda -> CHAIN_OK"


# ============================================================ R8-02
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def lung(tmp_path, monkeypatch):
    """``<tmp>/data/LUNG``: diagonal RNA and peaks with the two label files, no
    gene-activity file; the working directory at ``<tmp>``."""
    folder = tmp_path / "data" / "LUNG"
    folder.mkdir(parents=True)
    for name in ("rna.h5", "atac_peak.h5", "rna_cty.csv", "atac_cty.csv"):
        (folder / name).symlink_to(ROOT / "data" / "D28" / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _scan_warnings(**kw):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        mtb.scan("LUNG", "diagonal", data_path="data", verbose=False, **kw)
    return [str(w.message) for w in seen if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize("mods", [["rna", "atac_peak"], ["rna", "peak"]])
def test_a_peak_token_does_not_name_scbridge(lung, mods):
    assert mtb.method_info("scBridge")["atac"] == "gene_activity"
    assert not [m for m in _scan_warnings(modalities=mods) if "scBridge" in m]


@pytest.mark.parametrize("mods", [["rna", "atac_gas"], ["rna", "atac"],
                                  ["rna", "atac_peak", "atac_gas"]])
def test_other_tokens_still_name_scbridge(lung, mods):
    (msg,) = [m for m in _scan_warnings(modalities=mods) if "scBridge" in m]
    assert msg.startswith("The modalities rna"), msg
    assert " leave out scBridge, which reads a folder instead of modality files." in msg


def test_one_token_is_singular(lung):
    (msg,) = [m for m in _scan_warnings(modalities=["rna"]) if "scBridge" in m]
    assert msg.startswith("The modality rna leaves out scBridge, which reads a folder "
                          "instead of modality files. Pass modalities=[] to select it"), msg


def test_the_cli_scan_with_a_peak_token_does_not_name_scbridge(lung, capsys):
    rc = cli.main(["scan", "LUNG", "--category", "diagonal", "--data-path", "data",
                   "--modalities", "rna,atac_peak"])
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "scBridge" not in err, err
    rc = cli.main(["scan", "LUNG", "--category", "diagonal", "--data-path", "data",
                   "--modalities", "rna,atac_gas"])
    assert "warning: The modalities rna and atac_gas leave out scBridge" in \
        capsys.readouterr().err


def test_the_docstrings_say_which_folder_fed_method_is_named():
    for fn in (W.scan, W.run_all):
        assert ("``modalities`` leaves out scBridge, which reads a folder, without "
                "excluding its ATAC form.") in _doc(fn)
    assert ("Other lists drop them, with a ``UserWarning`` unless the tokens already "
            "exclude their ATAC representation.") in _doc(W.scan)


# ============================================================ R8-04
@pytest.fixture
def no_envs(monkeypatch):
    """No method environment is installed (the laptop situation)."""
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())


@pytest.fixture
def labmos(tmp_path, monkeypatch):
    """``<tmp>/data/LABMOS``: the D46 mosaic files (StabMap / scMoMaT layout)."""
    folder = tmp_path / "data" / "LABMOS"
    folder.mkdir(parents=True)
    for f in (ROOT / "data" / "D46").iterdir():
        (folder / f.name).symlink_to(f)
    monkeypatch.chdir(tmp_path)
    return tmp_path


_TOTALVI = "totalVI does not run on mosaic data. Its categories: vertical, cross."


def test_scan_raises_for_a_named_method_of_another_category(no_envs, labmos):
    with pytest.raises(ValueError) as e:
        mtb.scan("LABMOS", "mosaic", data_path="data", methods=["StabMap", "totalVI"],
                 verbose=False)
    assert str(e.value) == _TOTALVI
    # one sentence per method
    with pytest.raises(ValueError) as e:
        mtb.scan("LABMOS", "mosaic", data_path="data",
                 methods=["totalVI", "StabMap", "SCALEX"], verbose=False)
    assert str(e.value) == (f"{_TOTALVI} SCALEX does not run on mosaic data. Its "
                            "categories: diagonal.")


def test_run_all_raises_before_anything_runs(no_envs, labmos, capsys):
    for kw in ({"dry_run": True}, {"out_dir": "out"}):
        with pytest.raises(ValueError) as e:
            mtb.run_all("LABMOS", "mosaic", data_path="data",
                        methods=["StabMap", "scMoMaT", "totalVI"], **kw)
        assert str(e.value) == _TOTALVI
    assert not (labmos / "out").exists()
    assert capsys.readouterr().out == ""


def test_a_selection_with_every_method_present_is_unchanged(no_envs, labmos):
    one = mtb.scan("LABMOS", "mosaic", data_path="data", methods=["StabMap"],
                   verbose=False)
    assert list(one["method"]) == ["StabMap"]
    every = mtb.scan("LABMOS", "mosaic", data_path="data", verbose=False)
    assert {"StabMap", "scMoMaT"} <= set(every["method"])


def test_a_method_the_modalities_drop_raises_with_the_representation_note(lung):
    with pytest.raises(ValueError) as e:
        mtb.scan("LUNG", "diagonal", data_path="data", methods=["GLUE", "SCALEX"],
                 modalities=["rna", "atac_peak"], verbose=False)
    assert str(e.value) == ("SCALEX does not read rna+atac_peak in diagonal data. SCALEX "
                            "reads gene activity. Pass modalities=['rna', 'atac_gas'] or "
                            "['rna', 'atac'].")
    # another category and the modalities, in one message
    with pytest.raises(ValueError) as e:
        mtb.scan("LUNG", "diagonal", data_path="data", methods=["GLUE", "Matilda", "SCALEX"],
                 modalities=["rna", "atac_peak"], verbose=False)
    assert str(e.value).startswith("Matilda does not run on diagonal data. Its categories: "
                                   "vertical. SCALEX does not read rna+atac_peak in "
                                   "diagonal data. SCALEX reads gene activity."), e.value


def test_the_cli_run_all_dry_run_exits_1_with_the_sentence(no_envs, labmos, capsys):
    rc = cli.main(["run-all", "LABMOS", "--category", "mosaic", "--data-path", "data",
                   "--methods", "StabMap,scMoMaT,totalVI", "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 1
    assert f"error: {_TOTALVI}\n" in captured.err, captured.err
    assert captured.out == ""


def test_the_cli_scan_gives_the_same_sentence(no_envs, labmos, capsys):
    for methods in ("StabMap,SCALEX", "SCALEX"):
        rc = cli.main(["scan", "LABMOS", "--category", "mosaic", "--data-path", "data",
                       "--methods", methods])
        err = capsys.readouterr().err
        assert rc == 1
        (line,) = [l for l in err.splitlines() if l.startswith("error: ")]
        assert line == "error: SCALEX does not run on mosaic data. Its categories: diagonal."
        assert "[" not in err and ";" not in err, err


def test_the_docstrings_say_a_method_without_a_variant_raises():
    assert "A method in ``methods`` has no variant in ``category`` or ``modalities``." \
        in _doc(W.scan)
    assert ("A named method with no variant under ``category`` or ``modalities`` raises "
            "``ValueError``, even when other named methods have one.") in _doc(W.scan)
    assert ("A named method or a selection with no variant: ``ValueError`` before "
            "anything runs, dry run included") in _doc(W.run_all)


# ============================================================ R8-05
MACOS = "Methods run only on Linux, and this computer runs macOS."


@pytest.fixture
def macos(monkeypatch, no_envs):
    """This computer runs macOS, no environment is installed, a GPU is present
    (so only the platform and the files block a row)."""
    monkeypatch.setattr(R, "linux_only_sentence", lambda: MACOS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


@pytest.fixture
def mydata(tmp_path, monkeypatch):
    """``<tmp>/data/MYDATA``: the D11 files (CITE-seq) under a relative data root."""
    folder = tmp_path / "data" / "MYDATA"
    folder.mkdir(parents=True)
    for name in ("rna.h5", "adt.h5", "cty.csv"):
        (folder / name).symlink_to(ROOT / "data" / "D11" / name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _nothing_runnable(**kw):
    with pytest.raises(ValueError) as e:
        mtb.run_all("MYDATA", "vertical", out_dir="out", verbose=False, data_path="data",
                    **kw)
    return str(e.value)


def _plan(**kw):
    return mtb.scan("MYDATA", "vertical", data_path="data", verbose=False, **kw)


def test_rows_blocked_only_by_the_platform_are_not_listed(macos, mydata):
    msg = _nothing_runnable(modalities=["rna", "adt"])
    n = len(_plan(modalities=["rna", "adt"]))
    lines = msg.splitlines()
    assert lines[0] == "No method can run on MYDATA (vertical)."
    assert lines[1].startswith(MACOS)
    assert lines[2] == f"Nothing else blocks these {n} methods."
    assert lines[3].startswith("mtb.scan('MYDATA', 'vertical', data_path='data', "
                               "modalities=['rna', 'adt']) shows every row.")
    assert len(lines) == 4 and "not on this computer" not in msg
    msg = _nothing_runnable(modalities=["rna", "adt"], methods=["Matilda", "totalVI"])
    assert msg.splitlines()[2] == "Nothing else blocks these 2 requested methods."
    msg = _nothing_runnable(modalities=["rna", "adt"], methods=["totalVI"])
    assert msg.splitlines()[2] == "Nothing else blocks this requested method."


def test_a_row_that_also_lacks_a_file_is_listed_with_that_reason(macos, mydata):
    msg = _nothing_runnable(methods=["Matilda", "totalVI"])
    lines = msg.splitlines()
    assert lines[0] == ("None of the requested methods (Matilda, totalVI) can run on "
                        "MYDATA (vertical).")
    assert lines[2] == "1 of 3 requested rows is also blocked by something else:"
    assert lines[3] == ("  Matilda (rna+atac): Matilda needs gene-activity ATAC (atac.h5), "
                        "which is not in the folder.")
    assert lines[4].startswith("mtb.scan('MYDATA', 'vertical', data_path='data', "
                               "methods=['Matilda', 'totalVI']) shows these rows.")
    assert "not on this computer" not in msg


def test_without_methods_the_first_3_other_blocks_are_shown(macos, mydata, monkeypatch):
    # the review of round 8: rows whose files MYDATA lacks are counted, not listed
    msg = _nothing_runnable()
    plan = _plan()
    lacking = int((~plan["files_ok"]).sum())
    have = plan[plan["files_ok"]]
    lines = msg.splitlines()
    assert lines[2] == f"Nothing else blocks these {len(have)} methods."
    assert lines[3] == f"{lacking} rows need files that MYDATA does not have."
    # scripts at another commit block every row: the first 3 are listed
    from multibench import config
    monkeypatch.setattr(config, "scripts_ref_problem", lambda repo=None: "Wrong scripts.")
    msg = _nothing_runnable()
    lines = msg.splitlines()
    n = len(have)
    assert lines[2] == (f"{n} of {n} methods are also blocked by something else. "
                        "The first 3:")
    listed = lines[3:6]
    assert [l.split(" (")[0].strip() for l in listed] == sorted(have["method"])[:3]
    assert all(l.startswith("  ") and "not on this computer" not in l for l in listed)
    assert lines[6] == f"{lacking} rows need files that MYDATA does not have."
    assert lines[7].startswith("mtb.scan('MYDATA', 'vertical', data_path='data') shows "
                               "every row.")


def test_a_gpu_block_is_listed_without_the_linux_sentence(macos, mydata, monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    msg = _nothing_runnable(modalities=["rna", "adt"])
    plan = _plan(modalities=["rna", "adt"])
    gpu = sorted(m for m in plan["method"] if mtb.method_info(m)["requires_gpu"])
    assert "moETM" in gpu
    lines = msg.splitlines()
    verb = "is" if len(gpu) == 1 else "are"
    assert lines[2] == (f"{len(gpu)} of {len(plan)} methods {verb} also blocked by "
                        "something else:")
    assert [l.split(" (")[0].strip() for l in lines[3:3 + len(gpu)]] == gpu
    assert lines[3].endswith("needs an NVIDIA GPU, and this computer has none. See "
                             f"method_info(\"{gpu[0]}\")[\"requires_gpu\"].")
    assert "not on this computer" not in msg


def test_on_linux_the_message_is_todays(no_envs, mydata, monkeypatch):
    monkeypatch.setattr(R, "linux_only_sentence", lambda: None)
    msg = _nothing_runnable(modalities=["rna", "adt"])
    plan = _plan(modalities=["rna", "adt"])
    order = plan.assign(_f=~plan["files_ok"]).sort_values(["_f", "method"], kind="stable")
    want = ("No method can run on MYDATA (vertical).\n"
            f"The first 3 of {len(plan)} blocked methods:\n"
            + "\n".join(f"  {r.method} ({r.modalities}): {r.reason}"
                        for r in order.head(3).itertuples())
            + "\nmtb.scan('MYDATA', 'vertical', data_path='data', modalities=['rna', "
            "'adt']) shows every row. Its files_ok and env_ok columns say which check "
            "failed. mtb.env.doctor() checks the environments.")
    assert msg == want
    assert "is not installed. Run multibench env install --methods" in msg
    msg = _nothing_runnable(modalities=["rna", "adt"], methods=["totalVI"])
    assert "\nBlocked, one line per requested method:\n  totalVI (rna+adt): Environment " \
           in msg


def test_the_scan_frame_keeps_the_linux_sentence(macos, mydata):
    plan = _plan(modalities=["rna", "adt"])
    assert plan["env_reason"].str.endswith("runs only on Linux, not on this computer.").all()
    assert list(plan.columns) == W.SCAN_COLUMNS


def test_the_dry_run_header_names_no_column(no_envs, mydata, capsys):
    rc = cli.main(["run-all", "MYDATA", "--category", "vertical", "--data-path", "data",
                   "--modalities", "rna,adt", "--methods", "Matilda,totalVI",
                   "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 0, err
    head = err.splitlines()[0]
    assert "files_ok False" not in head
    assert head.endswith(" A method whose input files are missing has no command."), head
    rc = cli.main(["run-all", "MYDATA", "--category", "vertical", "--data-path", "data",
                   "--methods", "Matilda", "--dry-run"])
    head = capsys.readouterr().err.splitlines()[0]
    assert head.endswith(" A row whose input files are missing has no command."), head


def test_the_run_all_notes_describe_the_off_linux_list():
    doc = _doc(W.run_all)
    assert ("On macOS or Windows, its second line says that methods run only on Linux, "
            "and it lists only the rows something else also blocks.") in doc
