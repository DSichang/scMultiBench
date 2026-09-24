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
        assert ("``modalities`` drops a folder-fed method whose ATAC representation it "
                "allows.") in _doc(fn)
    assert ("Other lists drop them with a ``UserWarning``, unless the tokens exclude "
            "their ATAC representation.") in _doc(W.scan)
