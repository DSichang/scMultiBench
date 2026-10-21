"""Review of the round-5 integration (wp/f5_int).

R5-02: a first column of numbers is read as row numbers unless it holds
exactly the target's cell ids, so R's write.csv files keep scoring as before
on an AnnData or a folder whose cell ids are '0'..'n-1'. Several label files
and ``evaluate --column`` are aligned by their barcode column too. A
repeated sample column is not taken for cell ids, a repeated barcode is
named, and the command line names the label files instead of a Python call.
Also: ``_error_tail`` keeps a whole first word, ``load_run`` finds the
method folders of a moved tree, ``rescore(metrics=[...])`` without ARI
scores a one-order dataset, CLI help has no backticks, and the scan reasons
fit the table.
"""
import argparse
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import envs, registry
from tests.test_study_r4_wf import N, _batch_metrics, _h5, _quiet, cite  # noqa: F401
from tests.test_study_r5_wf import (BARS, CELLTYPE, KW, SAMPLES, _adata,  # noqa: F401
                                    _barcode_csv, _one_column, _order_warnings,
                                    _r_row_numbers, swept)

pytest.importorskip("scib")

METRICS = ["ASW", "ASW_batch", "iLISI"]


def _recorded(fn, *a, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, _order_warnings(rec)


def test_a_repeated_first_column_is_not_taken_for_cell_ids(tmp_path):
    a = _adata(tmp_path)
    p = tmp_path / "sample_x.csv"
    pd.DataFrame({"sample": SAMPLES, "x": SAMPLES}).to_csv(p, index=False)
    # c3 (E05): a file without cell ids next to barcodes is refused
    with pytest.raises(ValueError, match=r"^sample_x\.csv has no cell ids"):
        mtb.evaluate(a, labels="celltype", batch=str(p), metrics=["iLISI"])
    got, warned = _recorded(mtb.evaluate, a.obsm["X_emb"], labels=CELLTYPE, batch=str(p),
                            metrics=["iLISI"])
    assert warned == []
    pd.testing.assert_frame_equal(
        got, mtb.evaluate(a.obsm["X_emb"], labels=CELLTYPE, batch=SAMPLES,
                          metrics=["iLISI"]))


def test_a_repeated_barcode_is_named(cite, tmp_path):
    a = _adata(tmp_path)
    idx = list(a.obs_names)
    idx[5] = idx[4]
    rep = tmp_path / "b_rep.csv"
    pd.DataFrame({"sample": SAMPLES}, index=idx).to_csv(rep)
    with pytest.raises(ValueError, match=r"^batch: the first column of b_rep\.csv repeats "
                                         r"1 id \(first: \['CELL4-1'\]\)\. Give each cell "
                                         r"one row\.$"):
        mtb.evaluate(a, labels="celltype", batch=str(rep), metrics=["iLISI"])
    # a named id column next to x: the same count, without '(s)'
    named = tmp_path / "b_named.csv"
    pd.DataFrame({"cell": idx, "x": SAMPLES}).to_csv(named, index=False)
    with pytest.raises(ValueError, match=r"repeats 1 id \(first: \['CELL4-1'\]\)"):
        mtb.evaluate(a, labels="celltype", batch=str(named), metrics=["iLISI"])
    # run reads the file before any method runs
    data, batch, _ = cite
    bars = list(BARS)
    bars[5] = bars[4]
    drep = tmp_path / "d_rep.csv"
    pd.DataFrame({"sample": batch}, index=bars).to_csv(drep)
    with pytest.raises(ValueError, match=r"^batch: the first column of d_rep\.csv repeats "
                                         r"1 id"):
        mtb.run(str(data / "MYCITE"), "vertical", tmp_path / "out", batch=drep, **KW)
    assert not (tmp_path / "out").exists()
    # a file with several columns: the fix names no column= argument (R6-03:
    # it suggests a Series)
    wide = tmp_path / "wide.csv"
    pd.DataFrame({"sample": SAMPLES, "donor": SAMPLES, "day": SAMPLES}).to_csv(wide,
                                                                              index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels="celltype", batch=str(wide), metrics=["iLISI"])
    assert str(e.value) == ("The batch file wide.csv has several columns: sample, donor, "
                            "day. "
                            "Pass one column, for example "
                            f'pd.read_csv("{wide}")["sample"].to_numpy().')
    assert "column=" not in str(e.value)


# ================================================================ run records
def test_error_tail_keeps_a_word_the_cut_starts_with():
    t = "could not reach github.com because the proxy refused it"
    assert W._error_tail("x" * 50 + " " + t, width=len(t) + 3) == "..." + t
    assert W._error_tail("x" * 50 + " " + t, width=len(t) + 5) == "..." + t
    assert W._error_tail("x" * 50 + " " + t, width=len(t) + 1) == \
        "..." + t[t.index(" ") + 1:]


def test_a_moved_run_all_folder_rescores(swept, tmp_path):
    res, _, _ = swept
    moved = tmp_path / "moved"
    shutil.copytree(res.out_dir, moved)
    shutil.rmtree(res.out_dir)
    again = mtb.load_run(moved)
    assert [r["out_dir"] for r in again.records] == [str(moved / "Matilda_MYCITE")]
    assert _quiet(again.rescore).summary["status"].tolist() == ["scored"]


def test_rescore_with_metrics_that_leave_out_ari(swept):
    res, _, _ = swept
    new = _quiet(res.rescore, metrics=["ASW"])
    assert new.summary["status"].tolist() == ["scored"]
    assert new.summary["ASW"].notna().all()


# ====================================================================== text
def _walk(parser):
    yield parser.prog, parser
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for sub in a.choices.values():
                yield from _walk(sub)


def test_the_mira_and_bridge_reasons_are_sentences(tmp_path, monkeypatch):
    from types import SimpleNamespace
    script = tmp_path / "tools_scripts" / "MIRA" / "main_MIRA.py"
    script.parent.mkdir(parents=True)
    script.write_text("")
    monkeypatch.setattr(W.config, "repo_path", tmp_path)
    # a helper the package does not ship (MIRA's logger.py is shipped since 0.3.4)
    monkeypatch.setattr("multibench.engine.runner.shipped_helper", lambda m, h: None)
    v = SimpleNamespace(entrypoint="tools_scripts/MIRA/main_MIRA.py",
                        helpers=["logger.py"])
    why = W._missing_script(v, method="MIRA")
    assert why.startswith("MIRA's script imports logger.py. The public scMultiBench "
                          "repository does not include it. Put a logger.py next to "
                          "main_MIRA.py.")
    assert ";" not in why and "(" not in why.replace("('MIRA')", "")
    from multibench.engine.resolve import SAME_CELLS_REASON
    text = SAME_CELLS_REASON.format(method="Seurat_v5", dataset="D28", a="rna.h5",
                                    b="atac_peak.h5", n_a=6408, n_b=4606, shared=0)
    assert text == ("Seurat_v5 needs RNA and ATAC from the same cells as its bridge. "
                    "In D28, rna.h5 and atac_peak.h5 share 0 of 6,408 and 4,606 cells.")


def test_run_all_reads_r_row_numbers_on_a_folder_with_numbered_barcodes(cite, tmp_path, capsys):
    data, batch, _ = cite
    d = data / 'NUMBC'
    d.mkdir()
    bars = [str(i) for i in range(N)]
    _h5(d / 'rna.h5', [f'g{i}' for i in range(30)], bars)
    _h5(d / 'adt.h5', [f'p{i}' for i in range(6)], bars)
    pd.DataFrame({'x': CELLTYPE}).to_csv(d / 'cty.csv', index=False)
    rb = _r_row_numbers(tmp_path / 'b_r.csv', batch)
    (vec, aligned), warned = _recorded(W._cell_vector, rb, 'NUMBC', data)
    assert not aligned and list(vec) == list(batch) and (warned == [])
    by_r = _quiet(mtb.run, str(d), 'vertical', tmp_path / 'a', batch=rb, **KW)
    by_one = _quiet(mtb.run, str(d), 'vertical', tmp_path / 'b', batch=_one_column(tmp_path / 'b1.csv', batch), **KW)
    assert _batch_metrics(by_r) == _batch_metrics(by_one)
    assert _batch_metrics(by_r)[0] == 'scored'


def test_the_linux_only_reason_fits_the_table(monkeypatch):
    monkeypatch.setattr(envs, 'host_platform_problem', lambda: 'Method environments are Linux-only; this host is darwin/arm64')
    longest = max((envs.group_for(m) for m in registry.list_methods()), key=len)
    reason = W._env_hint(longest, 'X', 'vertical')
    assert reason == f'Environment {longest} runs only on Linux, not on this computer.'
