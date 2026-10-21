"""cite takes a RunResult: the methods that ran (core040 M-4)."""
import multibench as mtb


def test_cite_a_batch_result():
    recs = [{"method": "Matilda", "status": "CHAIN_OK"},
            {"method": "totalVI", "status": "FAIL"},
            {"method": "scMM", "status": "SKIPPED"}]
    res = mtb.RunResult(recs, "D11", "vertical")
    assert mtb.cite(res) == mtb.cite("Matilda")
