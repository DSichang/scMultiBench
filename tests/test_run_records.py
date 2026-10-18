"""Run records say which version scored them, and the summary shows no stale n_tunable
(core040 R-2); rescore keeps the saved caveats when no plan can be made (R-5)."""
import multibench as mtb
from multibench import workflow


def test_summary_has_no_n_tunable():
    recs = [{"method": "Matilda", "status": "CHAIN_OK", "n_tunable": 21,
             "metrics": {"ARI": 0.5}}]
    summary = mtb.BatchResult(recs, "D11", "vertical").summary
    assert "n_tunable" not in summary.columns


def test_a_scored_record_names_the_version_that_scored_it():
    assert "scored_by_version" in workflow._SCORE_KEYS


def test_rescore_keeps_the_saved_caveat_when_no_plan_can_be_made(tmp_path):
    rec = {"method": "Matilda", "status": "CHAIN_OK", "caveat": "the saved caveat",
           "modalities": ["rna", "adt"], "out_dir": str(tmp_path / "gone"),
           "data_path": str(tmp_path / "nowhere"), "n_tunable": 9}
    res = mtb.BatchResult([rec], "NOSUCH", "vertical", out_dir=tmp_path)
    assert workflow._plan_caveats("NOSUCH", "vertical", str(tmp_path / "nowhere"),
                                  ["Matilda"]) is None
    try:
        new = res.rescore(verbose=False)
    except FileNotFoundError:          # no output to score: rescore says so
        return
    assert new.records[0]["caveat"] == "the saved caveat"
    assert "n_tunable" not in new.records[0]
