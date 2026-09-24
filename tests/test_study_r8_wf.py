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

import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W


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
