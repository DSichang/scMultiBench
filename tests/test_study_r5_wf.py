"""Study round 5, work package wf: R5-01, R5-02, R5-03, R5-12 and R5-13.

R5-01: rescore(labels=) matched a barcode-indexed Series by position, and a
batch Series given with it was forced to the embedding rows.

R5-02: a batch or labels CSV written with its index (barcodes in the first
column) was matched by position wherever the target carries cell ids.

R5-03: summary.csv had no reason for a SKIPPED row; the run-all stderr line
called a skip a failure; the exit-code texts did not say that only a skip of
a method named in --methods sets exit 3.

R5-12: reference Notes without dash tails.

R5-13: no docstring list item has a continuation line that starts with
'- ' (Markdown renders it as a nested bullet).
"""
import ast
from pathlib import Path

import multibench

PKG = Path(multibench.__file__).parent


# ======================================================================= R5-13
def _dash_continuations(doc: str) -> list[str]:
    """Lines of ``doc`` that continue a list item but start with ``- ``.

    A continuation sits two spaces right of its item's ``- ``; starting it
    with ``- `` makes it a nested bullet. A nested list introduced by a line
    ending in ``:`` is left alone.
    """
    lines, found, item = doc.splitlines(), [], None
    for i, line in enumerate(lines):
        text = line.lstrip()
        indent = len(line) - len(text)
        if not text:
            item = None
            continue
        if (text.startswith("- ") and item is not None and indent == item + 2
                and not lines[i - 1].rstrip().endswith(":")):
            found.append(line)
        if text.startswith(("- ", "* ")):
            item = indent
        elif item is not None and indent <= item:
            item = None
    return found


def test_no_list_item_continues_on_a_line_starting_with_a_dash():
    bad = []
    for path in sorted(PKG.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                doc = ast.get_docstring(node, clean=True) or ""
                bad += [f"{path.relative_to(PKG)} {getattr(node, 'name', '<module>')}: "
                        f"{line.strip()}" for line in _dash_continuations(doc)]
    assert not bad, "\n".join(bad)


def test_the_check_sees_a_dash_continuation():
    doc = "- An item that runs on\n  - ``ValueError``, before any method runs.\n"
    assert _dash_continuations(doc) == ["  - ``ValueError``, before any method runs."]
    nested = "- Three kinds:\n  - one\n  - two\n"
    assert _dash_continuations(nested) == []


# ================================================================ R5-01 fixtures
import json  # noqa: E402
import warnings  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import multibench as mtb  # noqa: E402
from multibench import cli  # noqa: E402
from multibench import workflow as W  # noqa: E402
from tests.test_study_r4_wf import N, _batch_metrics, _h5, _quiet, cite  # noqa: E402,F401

pytest.importorskip("scib")

KW = dict(methods=["Matilda"], modalities=["rna", "adt"], verbose=False)
BARS = [f"c{i}" for i in range(N)]
CELLTYPE = np.array(["A"] * (N // 2) + ["B"] * (N // 2))


def _ari(res, method=None):
    sm = res.summary.set_index("method")
    return round(float(sm["ARI"].iloc[0] if method is None else sm.loc[method, "ARI"]), 4)


def _order_warnings(rec) -> list:
    """The recorded warnings about matching a vector or a file by position."""
    return [str(w.message) for w in rec
            if "by position" in str(w.message) or "looks like cell ids" in str(w.message)]


def _meta(batch):
    """The dataset's cells as one table, barcodes as index, rows shuffled."""
    return pd.DataFrame({"celltype": CELLTYPE, "sample": batch},
                        index=BARS).sample(frac=1.0, random_state=11)


@pytest.fixture
def swept(cite, tmp_path):
    """A saved one-method vertical sweep of the ``cite`` folder."""
    data, batch, shuffled = cite
    _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "a", data_path=data, **KW)
    return mtb.load_batch(tmp_path / "a"), data, batch


# ======================================================================= R5-01
def test_rescore_aligns_a_barcode_indexed_labels_series(swept):
    res, _, _ = swept
    ordered = _quiet(res.rescore, labels=pd.Series(CELLTYPE, index=BARS))
    shuffled = _quiet(res.rescore, labels=_meta(np.zeros(N))["celltype"])
    # the labels are those of cty.csv: the file labels' ARI, far from chance
    assert _ari(ordered) == _ari(res) > 0.3
    assert _ari(shuffled) == _ari(ordered)
    # the search ran over the file order, and the cells form one batch
    row = shuffled.summary.iloc[0]
    assert row["label_order"] == "cty.csv" and row["n_batches"] == 1
    # a one-column DataFrame is aligned the same way
    frame = _quiet(res.rescore, labels=_meta(np.zeros(N))[["celltype"]])
    assert _ari(frame) == _ari(ordered)


def test_rescore_labels_and_batch_as_two_shuffled_series(swept):
    res, _, batch = swept
    meta = _meta(batch)
    both = _quiet(res.rescore, labels=meta["celltype"], batch=meta["sample"])
    ordered = _quiet(res.rescore, labels=CELLTYPE, batch=batch)
    assert _batch_metrics(ordered)[1] is not None
    assert _batch_metrics(both) == _batch_metrics(ordered)
    assert _ari(both) == _ari(ordered)
    # no warning about the batch: both are aligned by barcode
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res.rescore(labels=meta["celltype"], batch=meta["sample"], metrics=["ARI"])
    assert not _order_warnings(rec)


def test_a_labels_series_with_foreign_ids_raises_before_scoring(swept, monkeypatch):
    res, _, _ = swept
    monkeypatch.setattr(W, "_score_record",
                        lambda *a, **k: pytest.fail("a record was scored"))
    bad = pd.Series(CELLTYPE, index=[f"x{b}" for b in BARS])
    with pytest.raises(ValueError, match=r"^labels: 120 ids are not cells of MYCITE "
                                         r"\(first: \['xc0'"):
        res.rescore(labels=bad)
    with pytest.raises(ValueError, match=r"have no id in labels .* Give a label for "
                                         r"every cell\.$"):
        res.rescore(labels=pd.Series(CELLTYPE, index=BARS).iloc[:-4])


def test_rescore_labels_as_an_array_keeps_todays_numbers(swept):
    res, _, batch = swept
    arr = _quiet(res.rescore, labels=CELLTYPE)
    # an array follows the embedding rows and bypasses the label-order search
    row = arr.summary.iloc[0]
    assert row["label_order"] == "(user labels)" and row["n_batches"] == 1
    assert _ari(arr) == _ari(res)
    # a batch array next to it still follows the embedding rows
    assert _batch_metrics(_quiet(res.rescore, labels=CELLTYPE, batch=batch)) == \
        _batch_metrics(_quiet(res.rescore, batch=batch))


def test_a_labels_series_without_usable_barcodes_is_positional_with_advice(swept):
    res, data, _ = swept
    for f in ("rna.h5", "adt.h5"):
        with h5py.File(data / "MYCITE" / f, "a") as h:
            del h["matrix/barcodes"]
            h["matrix/barcodes"] = np.array(["same"] * N, dtype="S8")
    with pytest.warns(UserWarning) as rec:
        got = res.rescore(labels=pd.Series(CELLTYPE, index=BARS), metrics=["ARI"])
    msgs = [str(w.message) for w in rec]
    assert ("The labels Series is matched by position, because the files of MYCITE "
            "have no usable cell ids (missing or repeated barcodes). Check that it "
            "follows the embedding rows.") in msgs
    assert not any("silence" in m for m in msgs)
    assert got.summary.iloc[0]["label_order"] == "(user labels)"


PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(30)]
GENES = [f"GENE{i}" for i in range(30)]
N_RNA, N_ATAC = 40, 40


def _diagonal_sweep(root):
    """A diagonal folder and a saved GLUE + uniPort sweep: GLUE stacks the RNA
    cells first, uniPort the ATAC cells; each embedding separates the types."""
    d = root / "data" / "MYDIAG"
    d.mkdir(parents=True)
    rna_bars = [f"r{i}" for i in range(N_RNA)]
    atac_bars = [f"a{i}" for i in range(N_ATAC)]
    _h5(d / "rna.h5", GENES, rna_bars)
    _h5(d / "atac_peak.h5", PEAKS, atac_bars)
    _h5(d / "atac_gas.h5", GENES, atac_bars)
    rna_lab = np.array(["A"] * 30 + ["B"] * 10)
    atac_lab = np.array(["A"] * 10 + ["B"] * 30)
    pd.DataFrame({"x": rna_lab}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": atac_lab}).to_csv(d / "atac_cty.csv", index=False)
    rng = np.random.default_rng(3)
    records = []
    for method, mods, first in (("GLUE", ["rna", "atac_peak"], "rna"),
                                ("uniPort", ["rna", "atac_gas"], "atac")):
        lab = (np.concatenate([rna_lab, atac_lab]) if first == "rna"
               else np.concatenate([atac_lab, rna_lab]))
        emb = rng.normal(size=(N_RNA + N_ATAC, 4))
        emb[lab == "B", 0] += 8.0
        out = root / "out" / f"{method}_MYDIAG"
        out.mkdir(parents=True)
        with h5py.File(out / "embedding.h5", "w") as f:
            f.create_dataset("data", data=emb.T)
        records.append({"method": method, "category": "diagonal", "dataset": "MYDIAG",
                        "modalities": mods, "status": "RUN_OK", "out_dir": str(out),
                        "data_path": str(root / "data"), "output_kind": "embedding"})
    labels = pd.Series(np.concatenate([rna_lab, atac_lab]), index=rna_bars + atac_bars)
    return W.BatchResult(records, "MYDIAG", "diagonal"), labels


def test_a_diagonal_labels_series_is_put_in_each_methods_cell_order(tmp_path):
    res, labels = _diagonal_sweep(tmp_path)
    files = _quiet(res.rescore, metrics=["ARI"]).summary.set_index("method")
    mine = _quiet(res.rescore, labels=labels.sample(frac=1.0, random_state=5),
                  metrics=["ARI"]).summary.set_index("method")
    assert mine.loc["GLUE", "label_order"] == "rna_cty.csv+atac_cty.csv"
    assert mine.loc["uniPort", "label_order"] == "atac_cty.csv+rna_cty.csv"
    for m in ("GLUE", "uniPort"):
        assert mine.loc[m, "ARI"] > 0.9
        assert mine.loc[m, "ARI"] == files.loc[m, "ARI"]


# ======================================================================= R5-02
import textwrap  # noqa: E402

from multibench.eval import io as eio  # noqa: E402
from tests.test_f3_cli import _clear_env_caches  # noqa: E402
from tests.test_f4_cli import _run_all, _standin  # noqa: E402

SAMPLES = np.tile(["s1", "s2"], N // 2)
# a stand-in method whose embedding separates the cell types and shifts sample s2
_BATCHY = textwrap.dedent('''\
    import sys
    from pathlib import Path
    import h5py, numpy as np
    out = Path(sys.argv[sys.argv.index("--save_path") + 1])
    out.mkdir(parents=True, exist_ok=True)
    emb = np.random.default_rng(7).normal(size=(%d, 4))
    emb[%d:, 0] += 8.0
    emb[1::2, 1] += 8.0
    with h5py.File(out / "embedding.h5", "w") as f:
        f.create_dataset("data", data=emb.T)
    (out / "predict.csv").write_text("x\\n" + "A\\n" * %d + "B\\n" * %d)
    ''') % (N, N // 2, N // 2, N // 2)


@pytest.fixture
def batchy(tmp_path, monkeypatch):
    yield _standin(tmp_path, monkeypatch, _BATCHY)
    _clear_env_caches()


def _one_column(path, values):
    pd.DataFrame({"x": values}).to_csv(path, index=False)
    return path


def _barcode_csv(path, values, index=BARS, seed=2):
    """``obs[["sample"]].to_csv(path)``: barcodes first, rows shuffled."""
    pd.DataFrame({"sample": values}, index=index).sample(
        frac=1.0, random_state=seed).to_csv(path)
    return path


def _r_row_numbers(path, values):
    """R's ``write.csv(data.frame(x = values))``: row numbers 1..n first."""
    lines = ['"","x"'] + [f'"{i}","{v}"' for i, v in enumerate(values, 1)]
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def test_run_all_cli_aligns_a_barcode_csv(batchy, tmp_path, capsys):
    one = _one_column(tmp_path / "one.csv", SAMPLES)
    shuf = _barcode_csv(tmp_path / "shuf.csv", SAMPLES)
    assert _run_all(batchy, tmp_path / "o1", "--batch", str(one)) == 0
    assert _run_all(batchy, tmp_path / "o2", "--batch", str(shuf)) == 0
    capsys.readouterr()
    a = mtb.load_batch(tmp_path / "o1").summary.iloc[0]
    b = mtb.load_batch(tmp_path / "o2").summary.iloc[0]
    assert a["status"] == b["status"] == "CHAIN_OK" and a["n_batches"] == 2
    assert a["ASW_batch"] < 0.5                  # the ordered file sees the batch shift
    for k in ("ASW_batch", "GC", "iLISI"):
        assert (pd.isna(a[k]) and pd.isna(b[k])) or a[k] == b[k], k


def test_rescore_aligns_a_barcode_csv(swept, tmp_path):
    res, _, batch = swept
    shuf = _barcode_csv(tmp_path / "b.csv", batch)
    assert _batch_metrics(_quiet(res.rescore, batch=shuf)) == \
        _batch_metrics(_quiet(res.rescore, batch=batch))
    labels = _barcode_csv(tmp_path / "l.csv", CELLTYPE)
    assert _ari(_quiet(res.rescore, labels=labels)) == _ari(res)


def test_a_csv_with_foreign_barcodes_raises(cite, tmp_path, monkeypatch, capsys):
    data, batch, _ = cite
    half = _barcode_csv(tmp_path / "half.csv", batch,
                        index=BARS[: N // 2] + [f"x{i}" for i in range(N // 2)])
    monkeypatch.setattr(W, "_run", lambda *a, **k: pytest.fail("a method was started"))
    with pytest.raises(ValueError, match=r"^batch: 60 of the 120 ids in the first column "
                                         r"of half\.csv are not cells of MYCITE "
                                         r"\(first: \['x"):
        mtb.run_all("MYCITE", "vertical", tmp_path / "out", batch=half, data_path=data,
                    **KW)
    assert not (tmp_path / "out").exists()
    # the command line's dry run reads the file as the real run does
    rc = _quiet(cli.main, ["run-all", "MYCITE", "--category", "vertical", "--methods",
                           "Matilda", "--modalities", "rna,adt", "--data-path", str(data),
                           "--batch", str(half), "--dry-run"])
    assert rc == 1
    assert "error: batch: 60 of the 120 ids in the first column of half.csv" in \
        capsys.readouterr().err


def _adata(tmp_path):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(7)
    emb = rng.normal(size=(N, 4))
    emb[N // 2:, 0] += 8.0
    emb[SAMPLES == "s2", 1] += 8.0
    obs = pd.DataFrame({"celltype": CELLTYPE, "sample": SAMPLES},
                       index=[f"CELL{i}-1" for i in range(N)])
    a = ad.AnnData(np.zeros((N, 2)), obs=obs)
    a.obsm["X_emb"] = emb
    return a


def test_evaluate_aligns_a_barcode_csv_to_the_obs_names(tmp_path):
    a = _adata(tmp_path)
    shuf = _barcode_csv(tmp_path / "b.csv", SAMPLES, index=list(a.obs_names))
    kw = dict(labels="celltype", metrics=["ASW_batch", "iLISI"])
    by_obs = mtb.evaluate(a, batch="sample", **kw)
    by_csv = mtb.evaluate(a, batch=str(shuf), **kw)
    assert by_obs.loc["ASW_batch", "Value"] < 0.5
    pd.testing.assert_frame_equal(by_obs, by_csv)
    # the command line reads the same file against an .h5ad
    a.write_h5ad(tmp_path / "mine.h5ad")
    _one_column(tmp_path / "ct.csv", CELLTYPE)
    for name, batch in (("obs", "o.csv"), ("csv", str(shuf))):
        if name == "obs":
            _one_column(tmp_path / batch, SAMPLES)
            batch = str(tmp_path / batch)
        assert cli.main(["evaluate", "--output", str(tmp_path / "mine.h5ad"), "--labels",
                         str(tmp_path / "ct.csv"), "--batch", batch, "--metrics",
                         "ASW_batch,iLISI", "--out", str(tmp_path / f"{name}.out")]) == 0
    assert pd.read_csv(tmp_path / "obs.out").equals(pd.read_csv(tmp_path / "csv.out"))


def test_evaluate_raises_on_a_csv_with_some_foreign_ids(tmp_path):
    a = _adata(tmp_path)
    half = _barcode_csv(tmp_path / "half.csv", SAMPLES,
                        index=list(a.obs_names[: N // 2]) + [f"x{i}" for i in range(N // 2)])
    with pytest.raises(ValueError, match=r"^batch: 60 of the 120 ids in the first column "
                                         r"of half\.csv are not cells of the output"):
        mtb.evaluate(a, labels="celltype", batch=str(half), metrics=["iLISI"])


def test_a_text_first_column_with_no_cell_warns_and_stays_positional(tmp_path):
    a = _adata(tmp_path)
    other = _barcode_csv(tmp_path / "other.csv", SAMPLES,
                         index=[f"z{i}" for i in range(N)], seed=None)
    with pytest.warns(UserWarning, match=r"^The first column of other\.csv looks like "
                                         r"cell ids, but none is a cell of the output\. "
                                         r"The file is matched by position\."):
        got = mtb.evaluate(a, labels="celltype", batch=str(other), metrics=["iLISI"])
    pd.testing.assert_frame_equal(
        got, mtb.evaluate(a, labels="celltype", batch="sample", metrics=["iLISI"]))
    # a bare array has no ids: positional, with the same kind of warning
    with pytest.warns(UserWarning, match=r"looks like cell ids, but the output has no "
                                         r"cell ids"):
        mtb.evaluate(a.obsm["X_emb"], labels=CELLTYPE, batch=str(other),
                     metrics=["iLISI"])


def test_r_row_numbers_and_one_column_files_score_as_before(tmp_path, cite):
    data, batch, _ = cite
    a = _adata(tmp_path)
    one = _one_column(tmp_path / "one.csv", SAMPLES)
    rnum = _r_row_numbers(tmp_path / "r.csv", SAMPLES)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        # R's row numbers: silent, positional, the same scores as the one-column file
        for target in (a, a.obsm["X_emb"]):
            kw = dict(labels=CELLTYPE, metrics=["ASW_batch", "iLISI"])
            pd.testing.assert_frame_equal(mtb.evaluate(target, batch=str(rnum), **kw),
                                          mtb.evaluate(target, batch=str(one), **kw))
    assert not _order_warnings(rec)
    kw = dict(data_path=data, **KW)
    by_one = _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "a",
                    batch=_one_column(tmp_path / "b1.csv", batch), **kw)
    by_r = _quiet(mtb.run_all, "MYCITE", "vertical", tmp_path / "b",
                  batch=_r_row_numbers(tmp_path / "b2.csv", batch), **kw)
    assert _batch_metrics(by_r) == _batch_metrics(by_one)
    assert _batch_metrics(by_one)[1] is not None


def test_files_without_an_id_column_are_read_as_before(tmp_path, cite):
    data, _, _ = cite
    one = _one_column(tmp_path / "one.csv", SAMPLES)
    rnum = _r_row_numbers(tmp_path / "r.csv", SAMPLES)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for f in (one, rnum):
            vec, aligned = W._cell_vector(f, "MYCITE", data)
            assert not aligned and list(vec) == list(SAMPLES)
        # the demo label files are single-column files, read as before
        root = Path(mtb.config.DEFAULT.data_path)
        for ds, f in (("D11", "cty.csv"), ("D28", "rna_cty.csv"), ("D28", "atac_cty.csv")):
            p = root / ds / f
            if not p.exists():
                pytest.skip(f"demo file {p} not present")
            vals, first = eio.read_labels_ids(p)
            assert first is None and list(vals) == list(eio.read_labels(p))
            vec, aligned = W._cell_vector(p, ds, root)
            assert not aligned and list(vec) == list(vals)
    assert not _order_warnings(rec)


# ======================================================================= R5-03
from tests.test_f3_cli import _WRITER  # noqa: E402


def _rec(method, status, **kw):
    return {"method": method, "status": status, "category": "vertical",
            "dataset": "MYCITE", "modalities": ["rna", "adt"], **kw}


def test_the_failure_line_keeps_failures_and_skips_apart():
    res = W.BatchResult([_rec("Matilda", "CHAIN_OK"),
                         _rec("Seurat_WNN", "FAIL", error="boom"),
                         _rec("totalVI", "SKIPPED", error="no env", requested=True),
                         _rec("MIRA", "SKIPPED", error="no script", requested=False)],
                        "MYCITE", "vertical")
    assert cli._failed_line(res, "out/failures.csv") == (
        "# 1 failed (Seurat_WNN), 1 skipped (totalVI). See out/failures.csv.")
    assert cli._not_run_line(res, "out/summary.csv") == (
        "# Not run: MIRA (SKIPPED). The reason column of summary.csv says why.")
    only = W.BatchResult([_rec("MIRA", "SKIPPED", error="x", requested=True)],
                         "MYCITE", "vertical")
    assert cli._failed_line(only, "f.csv") == "# 1 skipped (MIRA). See f.csv."
    assert cli._not_run_line(only, "s.csv") is None


@pytest.fixture
def standin(tmp_path, monkeypatch):
    yield _standin(tmp_path, monkeypatch, _WRITER)
    _clear_env_caches()


def test_a_run_whose_only_problems_are_unnamed_skips_exits_0_and_names_them(
        standin, tmp_path, capsys):
    out = tmp_path / "out"
    rc = cli.main(["run-all", "MYCITE", "--category", "vertical", "--modalities", "rna,adt",
                   "--data-path", str(standin), "--out-dir", str(out), "--format", "csv"])
    err = capsys.readouterr().err
    assert rc == 0, err
    sm = pd.read_csv(out / "summary.csv").set_index("method")
    skipped = sorted(sm.index[sm["status"] == "SKIPPED"])
    assert "totalVI" in skipped and "Matilda" not in skipped
    line = next(l for l in err.splitlines() if l.startswith("# Not run: "))
    assert line == (f"# Not run: {', '.join(skipped)} (SKIPPED). The reason column of "
                    f"summary.csv says why.")
    assert "failed" not in err
    # the reason column says why each SKIPPED row did not run, as the log does
    assert sm.columns[-1] == "reason"
    assert f"[run_all] skipping totalVI: {sm.loc['totalVI', 'reason']}" in err
    assert sm.loc[skipped, "reason"].str.len().gt(0).all()
    assert pd.isna(sm.loc["Matilda", "reason"])
    assert pd.read_csv(out / "failures.csv").empty


def test_summary_reason_comes_from_the_record_also_for_older_folders(tmp_path):
    out = tmp_path / "old"
    out.mkdir()
    recs = [_rec("Matilda", "CHAIN_OK", metrics={"ARI": 0.5}),
            _rec("MIRA", "SKIPPED", error="MIRA's script imports logger.py",
                 requested=False)]
    # a folder saved before the column: batch_result.json only has error
    (out / "batch_result.json").write_text(json.dumps(
        {"dataset": "MYCITE", "category": "vertical", "records": recs}))
    sm = mtb.load_batch(out).summary.set_index("method")
    assert sm.loc["MIRA", "reason"] == "MIRA's script imports logger.py"
    assert sm.loc["Matilda", "reason"] == ""
    assert W.BatchResult([], "MYCITE", "vertical").summary.columns[-1] == "reason"


def test_exit_code_texts_name_the_methods_rule(capsys):
    rule = ("a method failed, or a method named in --methods was skipped. Without "
            "--methods, a skipped method is only logged and marked SKIPPED in "
            "summary.csv.")
    with pytest.raises(SystemExit):
        cli.main(["run-all", "--help"])
    assert rule in " ".join(capsys.readouterr().out.split())
    doc = " ".join(cli.__doc__.split())
    assert ("but a method failed, or a method named in ``--methods`` was skipped. "
            "Without ``--methods``, a skipped method is only logged and marked "
            "``SKIPPED`` in ``summary.csv``.") in doc
    assert "failed or was skipped" not in doc
