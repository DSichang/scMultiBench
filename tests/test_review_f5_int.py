"""Review of the round-5 integration (wp/f5_int).

R5-02: a first column of numbers is read as row numbers unless it holds
exactly the target's cell ids, so R's write.csv files keep scoring as before
on an AnnData or a folder whose cell ids are '0'..'n-1'. Several label files
and ``evaluate --column`` are aligned by their barcode column too. A
repeated sample column is not taken for cell ids, a repeated barcode is
named, and the command line names the label files instead of a Python call.
Also: ``_error_tail`` keeps a whole first word, ``load_batch`` finds the
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
from multibench import cli, workflow as W
from multibench.engine import envs, registry
from tests.test_study_r4_wf import N, _batch_metrics, _h5, _quiet, cite  # noqa: F401
from tests.test_study_r5_wf import (BARS, CELLTYPE, KW, SAMPLES, _adata,  # noqa: F401
                                    _barcode_csv, _one_column, _order_warnings,
                                    _r_row_numbers, swept)

pytest.importorskip("scib")

METRICS = ["ASW", "ASW_batch", "iLISI"]


def _numbered_adata():
    """An AnnData with the default obs_names '0'..'n-1'."""
    a = _adata(None)
    a.obs_names = [str(i) for i in range(N)]
    return a


def _recorded(fn, *a, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, _order_warnings(rec)


# ============================================================ numbers as ids
def test_r_row_numbers_score_as_before_on_numbered_obs_names(tmp_path):
    a = _numbered_adata()
    rb = _r_row_numbers(tmp_path / "b_r.csv", SAMPLES)
    rl = _r_row_numbers(tmp_path / "ct_r.csv", CELLTYPE)
    want = mtb.evaluate(a, labels="celltype", batch="sample", metrics=METRICS)
    assert want.loc["ASW_batch", "Value"] < 0.5
    got, warned = _recorded(mtb.evaluate, a, labels=str(rl), batch=str(rb), metrics=METRICS)
    pd.testing.assert_frame_equal(got, want)
    assert warned == []
    # pandas' own index, shuffled: the numbers are the obs_names, so they align
    shuf = tmp_path / "b_pd.csv"
    pd.DataFrame({"sample": SAMPLES}, index=range(N)).sample(frac=1.0, random_state=2) \
        .to_csv(shuf)
    pd.testing.assert_frame_equal(
        mtb.evaluate(a, labels="celltype", batch=str(shuf), metrics=METRICS), want)
    # the command line, on the .h5ad
    a.write_h5ad(tmp_path / "num.h5ad")
    for tag, (lab, bat) in {"r": (rl, rb), "one": (
            _one_column(tmp_path / "l1.csv", CELLTYPE),
            _one_column(tmp_path / "b1.csv", SAMPLES))}.items():
        assert _quiet(cli.main, ["evaluate", "--output", str(tmp_path / "num.h5ad"),
                                 "--labels", str(lab), "--batch", str(bat), "--metrics",
                                 ",".join(METRICS), "--out", str(tmp_path / tag)]) == 0
    assert pd.read_csv(tmp_path / "r").equals(pd.read_csv(tmp_path / "one"))


def test_run_all_reads_r_row_numbers_on_a_folder_with_numbered_barcodes(
        cite, tmp_path, capsys):
    data, batch, _ = cite
    d = data / "NUMBC"
    d.mkdir()
    bars = [str(i) for i in range(N)]
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], bars)
    _h5(d / "adt.h5", [f"p{i}" for i in range(6)], bars)
    pd.DataFrame({"x": CELLTYPE}).to_csv(d / "cty.csv", index=False)
    rb = _r_row_numbers(tmp_path / "b_r.csv", batch)
    (vec, aligned), warned = _recorded(W._cell_vector, rb, "NUMBC", data)
    assert not aligned and list(vec) == list(batch) and warned == []
    kw = dict(data_path=data, **KW)
    by_r = _quiet(mtb.run_all, "NUMBC", "vertical", tmp_path / "a", batch=rb, **kw)
    by_one = _quiet(mtb.run_all, "NUMBC", "vertical", tmp_path / "b",
                    batch=_one_column(tmp_path / "b1.csv", batch), **kw)
    assert _batch_metrics(by_r) == _batch_metrics(by_one)
    assert _batch_metrics(by_r)[0] == "CHAIN_OK"
    rc = _quiet(cli.main, ["run-all", "NUMBC", "--category", "vertical", "--methods",
                           "Matilda", "--modalities", "rna,adt", "--data-path", str(data),
                           "--batch", str(rb), "--dry-run"])
    assert rc == 0, capsys.readouterr().err


# ================================================ several files, --column
def _per_sample_files(tmp_path, a, extra=False):
    """One barcode CSV per sample, rows shuffled within each file."""
    files = []
    for i, s in enumerate(("s1", "s2"), 1):
        keep = SAMPLES == s
        cols = {"celltype": CELLTYPE[keep]}
        if extra:
            cols["donor"] = np.full(keep.sum(), f"d{i}")
        p = tmp_path / f"{s}.csv"
        pd.DataFrame(cols, index=np.asarray(a.obs_names)[keep]).sample(
            frac=1.0, random_state=i).to_csv(p)
        files.append(str(p))
    return files


def test_several_barcode_label_files_are_aligned_with_their_batch(tmp_path):
    a = _adata(tmp_path)
    # the file of origin is the sample: 1 for s1, 2 for s2
    want = mtb.evaluate(a, labels="celltype", batch=np.where(SAMPLES == "s1", 1, 2),
                        metrics=METRICS)
    assert want.loc["ASW_batch", "Value"] < 0.5
    files = _per_sample_files(tmp_path, a)
    got, warned = _recorded(mtb.evaluate, a, labels=files, metrics=METRICS)
    pd.testing.assert_frame_equal(got, want)
    assert warned == []
    # the command line: repeated --labels, and --column on files with more columns
    a.write_h5ad(tmp_path / "mine.h5ad")
    (tmp_path / "wide").mkdir()
    wide = _per_sample_files(tmp_path / "wide", a, extra=True)
    runs = {"labels": [x for f in files for x in ("--labels", f)],
            "column": [x for f in wide for x in ("--labels", f)] + ["--column", "celltype"]}
    for tag, flags in runs.items():
        assert _quiet(cli.main, ["evaluate", "--output", str(tmp_path / "mine.h5ad"),
                                 *flags, "--metrics", ",".join(METRICS),
                                 "--out", str(tmp_path / f"{tag}.csv")]) == 0, tag
        out = pd.read_csv(tmp_path / f"{tag}.csv", index_col=0)["Value"]
        # iLISI is NaN where the LISI backend is unavailable
        assert out.round(6).fillna(-1).to_dict() == \
            want["Value"].round(6).fillna(-1).to_dict(), tag


def test_a_repeated_first_column_is_not_taken_for_cell_ids(tmp_path):
    a = _adata(tmp_path)
    p = tmp_path / "sample_x.csv"
    pd.DataFrame({"sample": SAMPLES, "x": SAMPLES}).to_csv(p, index=False)
    got, warned = _recorded(mtb.evaluate, a, labels="celltype", batch=str(p),
                            metrics=["iLISI"])
    assert warned == []
    pd.testing.assert_frame_equal(
        got, mtb.evaluate(a, labels="celltype", batch="sample", metrics=["iLISI"]))


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
    # run_all reads the file before any method runs
    data, batch, _ = cite
    bars = list(BARS)
    bars[5] = bars[4]
    drep = tmp_path / "d_rep.csv"
    pd.DataFrame({"sample": batch}, index=bars).to_csv(drep)
    with pytest.raises(ValueError, match=r"^batch: the first column of d_rep\.csv repeats "
                                         r"1 id"):
        mtb.run_all("MYCITE", "vertical", dry_run=True, batch=drep, data_path=data, **KW)
    # a file with several columns: the fix names no column= argument (R6-03:
    # it suggests a Series)
    wide = tmp_path / "wide.csv"
    pd.DataFrame({"sample": SAMPLES, "donor": SAMPLES, "day": SAMPLES}).to_csv(wide,
                                                                              index=False)
    with pytest.raises(ValueError) as e:
        mtb.evaluate(a, labels="celltype", batch=str(wide), metrics=["iLISI"])
    assert str(e.value) == ("The batch file wide.csv has several columns: sample, donor, "
                            "day. "
                            "Pass one column as a Series, for example "
                            f'pd.read_csv("{wide}")["sample"].')
    assert "column=" not in str(e.value)


def test_run_all_cli_names_the_label_files_not_a_python_call(cite, tmp_path, capsys):
    data, batch, _ = cite
    half = _barcode_csv(tmp_path / "half.csv", batch,
                        index=BARS[: N // 2] + [f"x{i}" for i in range(N // 2)])
    rc = _quiet(cli.main, ["run-all", "MYCITE", "--category", "vertical", "--methods",
                           "Matilda", "--modalities", "rna,adt", "--data-path", str(data),
                           "--batch", str(half), "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "drop that column when the rows follow the order of the label files of " \
           "MYCITE." in err
    assert "mtb." not in err


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
    again = mtb.load_batch(moved)
    assert [r["out_dir"] for r in again.records] == [str(moved / "Matilda_MYCITE")]
    assert _quiet(again.rescore).summary["status"].tolist() == ["CHAIN_OK"]


def test_rescore_with_metrics_that_leave_out_ari(swept):
    res, _, _ = swept
    new = _quiet(res.rescore, metrics=["ASW"])
    assert new.summary["status"].tolist() == ["CHAIN_OK"]
    assert new.summary["ASW"].notna().all()


# ====================================================================== text
def _walk(parser):
    yield parser.prog, parser
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for sub in a.choices.values():
                yield from _walk(sub)


def test_the_top_help_gives_one_sentence_per_exit_code(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    top = " ".join(capsys.readouterr().out.split())
    assert ("Exit codes: 0 means success. 1 is a runtime error, printed as error: ... "
            "on stderr. MULTIBENCH_DEBUG=1 shows the traceback. 2 is a usage error. 3 "
            "means run-all") in top


def test_the_linux_only_reason_fits_the_table(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem",
                        lambda: "Method environments are Linux-only; this host is "
                                "darwin/arm64")
    longest = max((envs.group_for(m) for m in registry.list_methods()), key=len)
    reason = W._env_hint(longest, "X", "vertical")
    assert reason == f"Environment {longest} runs only on Linux, not on this computer."
    assert len(reason) <= cli._TRUNCATE_WIDTH


def test_the_mira_and_bridge_reasons_are_sentences(tmp_path, monkeypatch):
    from types import SimpleNamespace
    script = tmp_path / "tools_scripts" / "MIRA" / "main_MIRA.py"
    script.parent.mkdir(parents=True)
    script.write_text("")
    monkeypatch.setattr(W.config.DEFAULT, "repo_path", tmp_path)
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


def test_dry_run_commands_do_not_double_the_parentheses(capsys):
    root = mtb.config.DEFAULT.data_path
    if not (root / "D28").is_dir():
        pytest.skip("D28 is not in the data path")
    rc = _quiet(cli.main, ["run-all", "D28", "--category", "diagonal", "--methods",
                           "scBridge,SCALEX", "--dry-run"])
    out = capsys.readouterr().out
    assert rc in (0, 1)
    lines = [l for l in out.splitlines() if l.startswith(("scBridge ", "SCALEX "))]
    assert any(l.startswith("scBridge (data_dir)") for l in lines), out
    assert any(l.startswith("SCALEX (rna+atac_gas)") for l in lines), out
    assert "((" not in out
