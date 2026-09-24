"""Findings of the review of the round-3 integration (wp/f3_int).

- The wrong-ATAC rule also blocks GLUE and Seurat_v3 when their peak file
  holds names the rewrite cannot turn into chr:start-end; the peak-name
  caveat has no subject, like the others (R3-02 / R3-04).
- A method named in methods= keeps the row its roles spell, as scan and
  method_info show them; a representation token that drops a named method
  says why (R3-01).
- Dry runs print the caveat of each row the sweep would run, without the
  notes on starting a method, and the scripts note once (R3-02).
- A scripts-ref mismatch is its own blocker: files_ok stays True, --strict
  counts it apart, the dry run keeps its commands (R3-10).
- run-all --dry-run --batch checks the length; a batch in the dataset's cell
  order is put in each method's cell order (R3-09).
- The runnable sets of the demo datasets are pinned, D28's vertical change
  included (R3-02).
- Smaller: the empty summary's columns, the evaluate count messages per
  flag, the scan error's list of methods, the unpacked-size check, the
  describe_layout and export_dataset wording, load_results on a missing file.

The env probe is pinned (every env installed, a GPU present) where a verdict
depends on it.
"""
import inspect
import json
import re
import shutil
import subprocess
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest

import multibench as mtb
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, ingest, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """Every env installed, a GPU, no method scripts on this machine, no ref."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _h5(path, feats, bars):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.ones((len(feats), len(bars))))
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))


def _diagonal(root, name, peaks, n=20):
    d = root / name
    d.mkdir(parents=True)
    rna, atac = [f"r{i}" for i in range(n)], [f"a{i}" for i in range(n)]
    genes = [f"GENE{i}" for i in range(60)]
    _h5(d / "rna.h5", genes, rna)
    _h5(d / "atac_peak.h5", peaks, atac)
    _h5(d / "atac_gas.h5", genes, atac)
    pd.DataFrame({"x": ["A"] * n}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": ["A"] * n}).to_csv(d / "atac_cty.csv", index=False)
    return root


def _vertical_peaks(root, name="MU_PEAK", n=60):
    d = root / name
    d.mkdir(parents=True)
    bars = [f"c{i}" for i in range(n)]
    _h5(d / "rna.h5", [f"g{i}" for i in range(30)], bars)
    _h5(d / "atac.h5", [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)], bars)
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(d / "cty.csv", index=False)
    return root


# ---------------------------------------------------------- wrong peak names block
def test_gene_names_in_the_peak_file_block_glue_seurat_v3_and_multimap(tmp_path, pinned):
    root = _diagonal(tmp_path, "LUNG_ga", [f"GENE{i}" for i in range(60)])
    sc = _quiet(mtb.scan, "LUNG_ga", "diagonal", data_path=root,
                verbose=False).set_index("method")
    for m in ("GLUE", "Seurat_v3", "MultiMAP"):
        assert not sc.loc[m, "runnable"] and sc.loc[m, "files_ok"], m
        assert sc.loc[m, "reason"].endswith(
            f"pass allow_atac_mismatch=True to run {m} anyway."), m
    assert sc.loc["GLUE", "reason"].startswith(
        "GLUE reads peak names such as chr1:100-200. atac_peak.h5 holds other names, for "
        "example GENE0. Rename them to chr:start-end, or pass ")
    # R4-01: naming the method keeps it blocked; allow_atac_mismatch runs it
    for m in ("GLUE", "Seurat_v3", "MultiMAP"):
        named = _quiet(mtb.scan, "LUNG_ga", "diagonal", methods=[m], data_path=root,
                       verbose=False).iloc[0]
        assert not named["runnable"], m
        allowed = _quiet(mtb.scan, "LUNG_ga", "diagonal", methods=[m], data_path=root,
                         verbose=False, allow_atac_mismatch=True).iloc[0]
        assert allowed["runnable"] and allowed["reason"] == "", m


def test_strict_counts_unreadable_peak_names_apart(tmp_path, pinned, capsys):
    root = _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(60)])
    rc = _quiet(cli.main, ["scan", "LUNG_ids", "--category", "diagonal", "--data-path",
                           str(root), "--modalities", "rna,atac_peak", "--strict"])
    err = capsys.readouterr().err
    # GLUE and Seurat_v3. MultiMAP never reads the names: peak_0 names that
    # are not the folder's genes are not gene activity (review of wp/f4_int)
    assert rc == 1 and "Rows whose peak names the method cannot read: 2." in err, err
    assert "wrong ATAC kind" not in err


def test_peak_name_caveat_has_no_subject_and_no_line_names_the_method_twice(
        tmp_path, pinned, capsys):
    root = _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(60)])
    _quiet(mtb.run_all, "LUNG_ids", "diagonal", methods=["GLUE", "Seurat_v3"],
           data_path=root, dry_run=True, allow_atac_mismatch=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[run_all] ")]
    assert "[run_all] GLUE reads peak names such as chr1:100-200. atac_peak.h5 holds " \
           "other names, for example peak_0. Rename them to chr:start-end" in "\n".join(lines)
    for line in lines:
        for m in ("GLUE", "Seurat_v3"):
            assert not re.search(rf"\b{m} {m}\b", line), line


# ---------------------------------------------------------- named methods, tokens
def test_a_named_method_keeps_the_row_its_roles_spell(pinned, capsys):
    """moETM's own spelling, as scan and method_info show it."""
    assert ["rna", "atac_gas"] in [s["modalities"] for s in
                                   mtb.method_info("moETM")["supports"]]
    for call in (mtb.scan, lambda *a, **k: mtb.run_all(*a, dry_run=True, **k)):
        df = _quiet(call, "D28", "vertical", methods=["moETM"],
                    modalities=["rna", "atac_gas"], verbose=False)
        assert list(df["modalities"]) == ["rna+atac_gas"]
    # without methods= the token still keeps only the gene-activity methods
    df = _quiet(mtb.scan, "D28", "vertical", modalities=["rna", "atac_gas"], verbose=False)
    assert "moETM" not in set(df["method"])
    rc = _quiet(cli.main, ["run-all", "D28", "--category", "vertical", "--methods", "moETM",
                           "--modalities", "rna,atac_gas", "--dry-run"])
    assert rc == 0, capsys.readouterr().err


def test_a_representation_token_that_drops_a_named_method_says_why(pinned, capsys):
    for call in (mtb.scan, lambda *a, **k: mtb.run_all(*a, dry_run=True, **k)):
        with pytest.raises(ValueError) as e:
            _quiet(call, "D28", "vertical", methods=["moETM"],
                   modalities=["rna", "gene_activity"], verbose=False)
        assert str(e.value).endswith(
            "moETM reads peaks through its atac_gas input; pass modalities=['rna', "
            "'atac_peak'] (or 'atac')"), str(e.value)
        assert "method_info(m)['supports']" not in str(e.value)
    with pytest.raises(ValueError, match=r"Matilda reads gene activity; pass "
                                         r"modalities=\['rna', 'atac_gas'\] \(or 'atac'\)$"):
        _quiet(mtb.scan, "D11", "vertical", methods=["Matilda"],
               modalities=["rna", "atac_peak"], verbose=False)
    # the command line keeps the message and gives its own spelling
    for sub in ("scan", "run-all"):
        argv = [sub, "D28", "--category", "vertical", "--methods", "moETM",
                "--modalities", "rna,gas"] + (["--dry-run"] if sub == "run-all" else [])
        assert _quiet(cli.main, argv) == 1
        err = capsys.readouterr().err
        assert "moETM reads peaks through its atac_gas input; pass --modalities " \
               "rna,atac_peak (or rna,atac)" in err, err


def test_scan_notes_list_gas_among_the_tokens():
    flat = " ".join(inspect.getdoc(mtb.scan).split())
    assert "``atac_gas`` / ``gas`` / ``gene_activity``" in flat


def test_modalities_help_says_the_representation_tokens_select_by_what_is_read():
    for sub in ("scan", "run-all"):
        parser = cli.build_parser()
        action = next(a for a in parser._actions if hasattr(a, "choices") and a.choices
                      and sub in a.choices)
        h = next(a.help for a in action.choices[sub]._actions
                 if "--modalities" in a.option_strings)
        assert "atac_peak and atac_gas select by what the method reads" in h, sub


def test_scan_error_lists_the_categorys_methods_for_a_mixed_list(capsys):
    rc = _quiet(cli.main, ["scan", "D11", "--category", "diagonal", "--methods",
                           "GLUE,Matilda"])
    err = capsys.readouterr().err
    assert rc == 1
    present = sorted(set(mtb.list_methods("diagonal")))
    assert "method(s) ['Matilda'] are not in the scan table for D11/diagonal" in err
    assert f"methods present: {present}" in err, err


# ---------------------------------------------------------- dry-run caveat lines
def test_dry_run_prints_the_scripts_note_once_and_no_start_notes(pinned, capsys):
    _quiet(mtb.run_all, "D11", "vertical", dry_run=True)
    out = capsys.readouterr().out
    assert out.count("The method scripts are not in") == 1, out
    assert "[run_all] The method scripts are not in" in out
    _quiet(mtb.run_all, "D28", "diagonal", dry_run=True)
    out = capsys.readouterr().out
    assert "the command reads" not in out and "start the method with" not in out
    assert out.count("The method scripts are not in") == 1
    rc = _quiet(cli.main, ["run-all", "D11", "--category", "vertical", "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 0 and err.count("# The method scripts are not in") == 1, err
    assert not re.search(r"^# \w+ The method scripts are not in", err, re.M)


def test_dry_run_prints_no_caveat_for_a_row_it_would_skip(tmp_path, pinned, capsys):
    root = _vertical_peaks(tmp_path)
    _quiet(mtb.run_all, "MU_PEAK", "vertical", modalities=["rna", "atac"],
           data_path=root, dry_run=True)
    out = capsys.readouterr().out
    assert "[run_all] Matilda" not in out           # blocked: the wrong ATAC kind
    assert "[run_all] moETM" not in out or "moETM needs" not in out
    _quiet(mtb.run_all, "MU_PEAK", "vertical", methods=["Matilda"],
           modalities=["rna", "atac"], data_path=root, dry_run=True)
    assert "[run_all] Matilda" not in capsys.readouterr().out    # named: still skipped
    _quiet(mtb.run_all, "MU_PEAK", "vertical", methods=["Matilda"],
           modalities=["rna", "atac"], data_path=root, dry_run=True,
           allow_atac_mismatch=True)
    assert "[run_all] Matilda needs gene-activity ATAC. atac.h5 holds peaks, because" in \
        capsys.readouterr().out


# ---------------------------------------------------------- scripts ref
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@pytest.fixture
def scripts_at_head(tmp_path, monkeypatch):
    repo = tmp_path / "scripts"
    ep = repo / registry.get("totalVI").variants[0].entrypoint
    ep.parent.mkdir(parents=True)
    ep.write_text("print(1)\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(repo), "PATH": "/usr/bin:/bin"}
    for argv in (["git", "init", "-q"], ["git", "add", "."],
                 ["git", "commit", "-q", "-m", "x"]):
        subprocess.run(argv, cwd=repo, check=True, env=env)
    monkeypatch.setattr(config.DEFAULT, "repo_path", repo)
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, "deadbeef")
    return repo


@needs_git
def test_a_scripts_ref_mismatch_is_its_own_blocker(scripts_at_head, capsys):
    sc = mtb.scan("D11", "vertical", methods=["totalVI"], verbose=False)
    r = sc.iloc[0]
    assert not r["runnable"] and r["files_ok"] and r["command"]
    assert "(MULTIBENCH_SCRIPTS_REF)" in r["reason"]
    rc = cli.main(["scan", "D11", "--category", "vertical", "--methods", "totalVI",
                   "--strict"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "Rows whose scripts are not at MULTIBENCH_SCRIPTS_REF: 1." in err \
        and "input files" not in err
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--methods", "totalVI",
                   "--dry-run"])
    cap = capsys.readouterr()
    assert rc == 0
    assert "# Commands of the 1 row whose input files are in place." in cap.out
    assert "totalVI (rna+adt): " in cap.out


# ---------------------------------------------------------- batch
def test_run_all_dry_run_checks_the_batch_length(tmp_path, pinned, capsys):
    short = tmp_path / "short.csv"
    pd.DataFrame({"b": ["s1", "s2", "s1"]}).to_csv(short, index=False)
    argv = ["run-all", "D11", "--category", "vertical", "--methods", "totalVI",
            "--dry-run", "--batch"]
    rc = _quiet(cli.main, argv + [str(short)])
    err = capsys.readouterr().err
    assert rc == 1
    assert ("error: --batch gave 3 batch ids for the 2,864 cells of the label files of "
            "D11. The batch metrics of totalVI would fail.") in err, err
    full = tmp_path / "full.csv"
    pd.DataFrame({"b": np.tile(["s1", "s2"], 1432)}).to_csv(full, index=False)
    assert _quiet(cli.main, argv + [str(full)]) == 0
    _quiet(mtb.run_all, "D11", "vertical", methods=["totalVI"], dry_run=True,
           batch=str(short))
    assert "[run_all] batch has 3 entries for the 2,864 cells" in capsys.readouterr().out


def _cty(path, labels):
    pd.DataFrame({"x": labels}).to_csv(path, index=False)


def test_a_dataset_order_batch_follows_each_methods_cell_order(tmp_path, monkeypatch):
    """uniPort puts its ATAC cells first: the batch ids go with them."""
    d = tmp_path / "DIAG"
    d.mkdir()
    _cty(d / "rna_cty.csv", ["A", "B"] * 5)
    _cty(d / "atac_cty.csv", ["A", "B"] * 4)
    batch = np.array(["rna_s"] * 10 + ["atac_s"] * 8)          # labels_for order
    seen = {}

    def fake(emb, category, cands, batch=None, metrics=None, **kw):
        names, lab, bat = next(c for c in cands if c[0][0] == "atac_cty.csv")
        seen["bat"] = np.asarray(batch if batch is not None else bat)
        val = pd.DataFrame({"Value": {"ARI": 0.5}})
        return names, val, [{"order": names, "ARI": 0.5}]
    monkeypatch.setattr(W, "_evaluate_best_order", fake)
    rec = {"method": "uniPort"}
    W._score_record(rec, np.zeros((18, 3)), "DIAG", "diagonal", tmp_path,
                    registry.get("uniPort").variants[0], batch=batch)
    assert list(seen["bat"]) == ["atac_s"] * 8 + ["rna_s"] * 10
    assert rec["batch_source"] == "user" and rec["n_batches"] == 2


def test_a_dataset_order_batch_is_cut_to_the_batches_a_method_reads(tmp_path, monkeypatch):
    """UINMF reads batches 1-2 of 3: it gets the ids of those cells."""
    d = tmp_path / "CROSS"
    d.mkdir()
    for i, n in ((1, 4), (2, 6), (3, 5)):
        _cty(d / f"cty{i}.csv", ["A", "B"] * (n // 2) + ["A"] * (n % 2))
    batch = np.array(["d1"] * 4 + ["d2"] * 6 + ["d3"] * 5)
    seen = {}

    def fake(emb, category, cands, batch=None, metrics=None, **kw):
        names, lab, bat = next(c for c in cands if c[0] == ["cty1.csv", "cty2.csv"])
        seen["bat"] = np.asarray(batch if batch is not None else bat)
        return names, pd.DataFrame({"Value": {"ARI": 0.5}}), [{"order": names, "ARI": 0.5}]
    monkeypatch.setattr(W, "_evaluate_best_order", fake)
    rec = {"method": "UINMF"}
    W._score_record(rec, np.zeros((10, 3)), "CROSS", "cross", tmp_path,
                    registry.get("UINMF").variants[0], batch=batch)
    assert list(seen["bat"]) == ["d1"] * 4 + ["d2"] * 6
    with pytest.raises(ValueError, match="batch has 7 entries, embedding has 10 cells"):
        W._score_record({"method": "UINMF"}, np.zeros((10, 3)), "CROSS", "cross",
                        tmp_path, registry.get("UINMF").variants[0], batch=batch[:7])


def test_batch_docs_name_the_label_file_order():
    p = " ".join(inspect.getdoc(mtb.run_all).split("Parameters", 1)[1]
                 .split("Returns", 1)[0].split())
    assert "cells in the order of ``mtb.labels_for(dataset)``" in p


# ---------------------------------------------------------- pinned demo sets
PINNED = {
    "D11": ["Concerto|vertical|rna+adt", "MOFA2|vertical|rna+adt",
            "Matilda|vertical|rna+adt", "Multigrate|vertical|rna+adt",
            "Seurat_WNN|vertical|rna+adt", "UINMF|vertical|rna+adt",
            "VIMCCA|vertical|rna+adt", "moETM|vertical|rna+adt", "scMDC|vertical|rna+adt",
            "scMM|vertical|rna+adt", "scMSI|vertical|rna+adt", "scMoMaT|vertical|rna+adt",
            "sciPENN|vertical|rna+adt", "totalVI|vertical|rna+adt"],
    # R3-02: moETM, scMM and iPOLNG read peaks; D28's atac_gas.h5 holds gene
    # activity, so their vertical rna+atac_gas rows are no longer runnable
    "D28": ["Conos|diagonal|rna+atac_gas", "GLUE|diagonal|rna+atac_peak",
            "MultiMAP|diagonal|rna+atac_peak+atac_gas", "Portal|diagonal|rna+atac_gas",
            "SCALEX|diagonal|rna+atac_gas", "Seurat_v3|diagonal|rna+atac_peak+atac_gas",
            "VIPCCA|diagonal|rna+atac_gas", "iNMF|diagonal|rna+atac_gas",
            "online_iNMF|diagonal|rna+atac_gas", "scBridge|diagonal|(data_dir)",
            "scJoint|diagonal|rna+atac_gas", "scMDC|vertical|rna+atac_gas",
            "sciCAN|diagonal|rna+atac_gas", "uniPort|diagonal|rna+atac_gas"],
    "D45": ["Cobolt|mosaic|rna1+rna2+atac2+atac3", "MultiVI|mosaic|rna1+atac3+rna2+atac2",
            "Multigrate|mosaic|rna1+rna2+atac2+atac3", "SMILE|mosaic|rna2+atac2+rna1+atac3"],
    "D46": ["StabMap|mosaic|rna1+rna2+rna3+adt1+atac2",
            "scMoMaT|mosaic|rna1+rna2+rna3+adt1+atac2"],
    "D52": ["Concerto|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "Multigrate|mosaic|rna1+rna2+adt2+adt3",
            "StabMap|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "UINMF|cross|rna1+rna2+adt1+adt2", "scMDC|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "scMM|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "scMoMaT|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "sciPENN|cross|rna1+rna2+rna3+adt1+adt2+adt3",
            "totalVI|cross|rna1+rna2+rna3+adt1+adt2+adt3"],
}


@pytest.mark.parametrize("dataset", sorted(PINNED))
def test_demo_datasets_keep_their_pinned_runnable_rows(dataset, pinned):
    df = _quiet(mtb.scan, dataset, verbose=False)
    got = sorted(f"{r.method}|{r.category}|{r.modalities}" for r in df[df.runnable].itertuples())
    assert got == sorted(PINNED[dataset])
    blocked = df[W._is_wrong_atac(df["reason"])]
    want = ({"iPOLNG", "moETM", "scMM"} if dataset == "D28" else set())
    assert set(blocked["method"]) == want
    assert set(blocked["category"]) <= {"vertical"}


# ---------------------------------------------------------- smaller findings
def test_the_empty_summary_ends_like_a_full_one():
    empty = W.BatchResult([], "D11", "vertical").summary
    full = W.BatchResult([{"method": "Matilda", "status": "RUN_OK"}], "D11",
                         "vertical").summary
    # R5-03: reason is the last column
    assert list(empty.columns)[-3:] == ["label_order_note", "caveat", "reason"]
    assert list(full.columns)[-3:] == ["label_order_note", "caveat", "reason"]


def test_evaluate_count_errors_name_where_the_labels_came_from(tmp_path, capsys):
    emb = tmp_path / "emb_small.npy"
    np.save(emb, np.random.default_rng(0).normal(size=(500, 4)))
    rc = _quiet(cli.main, ["evaluate", "--output", str(emb), "--dataset", "D46",
                           "--category", "mosaic", "--method", "scMoMaT"])
    err = capsys.readouterr().err
    assert rc == 1
    assert ("error: the label files of D46 (cty1.csv, cty2.csv, cty3.csv) hold 21,416 "
            "labels for 500 cells in --output.") in err, err
    assert "--labels" not in err
    lab = tmp_path / "cty.csv"
    pd.DataFrame({"x": ["A", "B"] * 250}).to_csv(lab, index=False)
    cl = tmp_path / "cl.csv"
    pd.DataFrame({"x": [1, 2] * 15}).to_csv(cl, index=False)
    rc = _quiet(cli.main, ["evaluate", "--output", str(emb), "--labels", str(lab),
                           "--clustering", str(cl), "--metrics", "ARI"])
    err = capsys.readouterr().err
    assert rc == 1 and "--clustering gave 30 cluster ids (cl.csv) for 500 cells" in err, err


def test_an_unpacked_size_below_the_archive_reads_unknown(tmp_path, monkeypatch):
    shipped = envs._SIZES_JSON
    table = tmp_path / "packed_sizes.json"
    table.write_text(json.dumps({"a": {"archive_bytes": 5, "unpacked_bytes": 2},
                                 "b": {"archive_bytes": 5, "unpacked_bytes": 9}}))
    monkeypatch.setattr(envs, "_SIZES_JSON", table)
    envs.packed_sizes.cache_clear()
    try:
        sizes = envs.packed_sizes()
    finally:
        envs.packed_sizes.cache_clear()
    assert sizes["a"]["unpacked_bytes"] is None and sizes["b"]["unpacked_bytes"] == 9
    # the shipped table: no new entry of that kind (scmb_scmm2-cpu awaits a
    # measurement on the Linux host)
    raw = json.loads(shipped.read_text())
    bad = {k for k, v in raw.items() if isinstance(v, dict)
           and isinstance(v.get("unpacked_bytes"), int)
           and v["unpacked_bytes"] < (v.get("archive_bytes") or 0)}
    assert bad <= {"scmb_scmm2-cpu"}


def test_layout_and_export_notes_say_scan_skips_the_other_representation():
    txt = mtb.describe_layout("vertical")
    assert "still runs" not in txt
    assert "A method whose ATAC file holds the other representation gives a wrong " \
           "embedding." in txt
    assert "mtb.scan and mtb.run_all skip such a method unless allow_atac_mismatch=True." \
        in txt
    notes = " ".join(inspect.getdoc(ingest.export_dataset).split())
    assert "``mtb.scan`` and ``run_all`` skip a method whose file holds the other " \
           "representation" in notes
    assert "reports the mismatch in its ``caveat`` column" not in notes


def test_load_results_on_a_missing_file_names_the_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"result_path 'nope\.csv' does not exist "
                                                r"\(working directory: "):
        mtb.load_results(result_path="nope.csv", source="user")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mtb.load_results(result_path="nope.csv")
    mine = tmp_path / "mine.csv"
    pd.DataFrame({"metric": ["ARI"], "value": [0.5], "method": ["X"], "dataset": ["D1"],
                  "category": ["vertical"], "source": ["user"]}).to_csv(mine, index=False)
    assert len(_quiet(mtb.load_results, result_path=str(mine), source="user")) == 1


# ---------------------------------------------------------- docs pages
def _docs_root():
    import os
    from pathlib import Path
    root = os.environ.get("SCMULTIBENCH_DOCS")
    return Path(root) if root and Path(root).is_dir() else None


needs_docs = pytest.mark.skipif(_docs_root() is None, reason="SCMULTIBENCH_DOCS not set")


def _flat(page):
    return " ".join((_docs_root() / page).read_text().split())


@needs_docs
def test_pages_state_the_donor_count_cross_reads():
    """The Quickstart and the run guide say what describe_layout('cross') lists."""
    layout = [l for l in mtb.describe_layout("cross").splitlines() if "batch 1 =" in l]
    reads = {}
    for line in layout:
        pattern, methods = line.split(":", 1)
        top = max(int(n) for n in re.findall(r"(?:batch )?(\d+) =", pattern))
        for m in re.sub(r"\(demo \w+\)", "", methods).split(","):
            reads[m.strip()] = top
    assert reads.pop("UINMF") == 2 and set(reads.values()) == {3}
    # R4-16 moved the counts from the visible bullet to the Step 1 Details block
    step1 = _flat("quickstart.md").split("## Step 1:", 1)[1].split("## Step 2:", 1)[0]
    assert f"The `cross` category has {len(reads) + 1} methods, which integrate the donors" in step1
    assert "The cross methods read batches 1-3. UINMF reads only the first two." in step1
    assert "also fits three donors, one file each" in step1
    run = _flat("tutorials/run.md")
    assert ('category="cross") ``` The cross methods read batches 1-3. '
            '`describe_layout("cross")` lists them.') in run


@needs_docs
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_job_script_is_valid_bash(tmp_path):
    text = (_docs_root() / "installation.md").read_text()
    job = text.split('```bash title="job.sh (Slurm)"', 1)[1].split("```", 1)[0]
    script = tmp_path / "job.sh"
    script.write_text(job)
    res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "# export MULTIBENCH_SCRIPTS_REF=<commit>" in job


@needs_docs
def test_changes_page_follows_the_review_fixes():
    text = _flat("changes.md")
    for phrase in ("wrong ATAC kind, unreadable peak names and scripts not at "
                   "`MULTIBENCH_SCRIPTS_REF`",
                   "With `--dry-run`, a file of the wrong length exits with `1`.",
                   "iPOLNG, which read peaks, unless `methods=` names them.",
                   "GLUE and Seurat_v3 are not runnable in `scan` when the peak file",
                   "takes the ids in the cell order of `labels_for(dataset)`",
                   "the refusal of `mtb.run` starts with `Methods run only on Linux` and "
                   "names no install command.",
                   "`load_results` with a `result_path` that does not exist raises"):
        assert phrase in text, phrase
    assert "is one sentence that starts with" not in text
    assert 'With `"atac_gas"`, `scan(..., methods=["moETM"])` raises' not in text


@needs_docs
def test_discover_details_say_a_representation_token_keeps_the_readers():
    text = _flat("tutorials/discover.md")
    assert ("A representation token keeps every row whose method reads it, so "
            '`"atac_peak"` also keeps MultiMAP and Seurat_v3.') in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = set(mtb.scan("D28", "diagonal", modalities=["rna", "atac_peak"],
                           verbose=False)["method"])
    assert {"MultiMAP", "Seurat_v3"} <= got


@needs_docs
def test_deploy_gate_refuses_while_the_data_release_is_not_public(tmp_path, monkeypatch):
    """mtb.data.fetch reads RELEASE_URL; a private release answers 404."""
    import importlib
    import importlib.util
    fetch = importlib.import_module("multibench.data.fetch")     # the module, not mtb.data.fetch
    spec = importlib.util.spec_from_file_location(
        "deploy_gate", _docs_root().parent / "hooks" / "deploy_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    from pathlib import Path
    root = Path(mtb.__file__).resolve().parent.parent
    assert gate.checkout_release_url(root) == fetch.RELEASE_URL
    asked = []
    monkeypatch.setattr(gate, "asset_status", lambda url, timeout=15: asked.append(url) or 404)
    (msg,) = gate.data_problems(root)
    assert asked == [f"{fetch.RELEASE_URL}/D11.tar.gz"]
    assert msg.endswith("answers HTTP 404 to an anonymous request: make the data release "
                        "public before the deploy, or mtb.data.fetch and the site's "
                        "download steps fail")
    monkeypatch.setattr(gate, "asset_status", lambda url, timeout=15: 302)
    assert gate.data_problems(root) == []                    # GitHub redirects a public asset

    def offline(url, timeout=15):
        raise OSError("no network")
    monkeypatch.setattr(gate, "asset_status", offline)
    (msg,) = gate.data_problems(root)
    assert msg.startswith("could not reach ") and "cannot be checked" in msg
    assert gate.data_problems(tmp_path) == [
        f"no RELEASE_URL in {tmp_path / 'multibench' / 'data' / 'fetch.py'}"]
    assert "data_problems(checkout)" in (_docs_root().parent / "hooks" /
                                         "deploy_gate.py").read_text()
