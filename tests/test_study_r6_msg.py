"""Round 6, work package 'msg': caveats and messages read as plain sentences.

- R6-10: every scan caveat is one or more sentences, joined with a space,
  that start with the method name (the scripts note starts with "The method
  scripts"). A caller that prints the method name before a caveat prints the
  caveat alone. scBridge's data_dir reason is a sentence like its neighbours.
  The --strict counts do not change.
- R6-11: the scripts-ref reason starts with a capital, and scan --strict
  prints it in full.
- R6-12: the nothing-runnable head, the run-all dry-run header, the Linux-only
  summary, the inputs_for file error and its layout hints, the merge line,
  the mosaic gene-activity warning, the plural positional warning and the
  method counts.
- R6-14: five Notes passages without dash insertions.
"""
import inspect
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
from multibench.engine import runner as R

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
GENES = [f"GENE{i}" for i in range(60)]
PEAKS = [f"chr1:{i * 100}-{i * 100 + 50}" for i in range(40)]
SCRIPTS_NOTE = "The method scripts are not in "


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """Every env installed, a GPU, no method scripts on this machine, no ref."""
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    return tmp_path / "no_scripts"


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _h5(path, feats, bars, value=1.0):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.full((len(feats), len(bars)), value))
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))


def _labels(path, n):
    pd.DataFrame({"x": ["A", "B"] * (n // 2)}).to_csv(path, index=False)


def _lung_ids(root, n=20):
    """S3's LUNG_ids: diagonal, peaks named peak_0, no gene-activity file."""
    d = root / "LUNG_ids"
    d.mkdir(parents=True)
    _h5(d / "rna.h5", GENES, [f"r{i}" for i in range(n)])
    _h5(d / "atac_peak.h5", [f"peak_{i}" for i in range(60)], [f"a{i}" for i in range(n)])
    _labels(d / "rna_cty.csv", n)
    _labels(d / "atac_cty.csv", n)
    return root


def _log_cite(root, n=20):
    """A vertical RNA+ADT folder whose RNA holds log-normalised values."""
    d = root / "LOGCITE"
    d.mkdir(parents=True)
    bars = [f"c{i}" for i in range(n)]
    _h5(d / "rna.h5", GENES[:30], bars, value=0.5)
    _h5(d / "adt.h5", [f"p{i}" for i in range(6)], bars)
    _labels(d / "cty.csv", n)
    return root


def _mu_peak(root, n=60):
    d = root / "MU_PEAK"
    d.mkdir(parents=True)
    bars = [f"c{i}" for i in range(n)]
    _h5(d / "rna.h5", GENES[:30], bars)
    _h5(d / "atac.h5", PEAKS, bars)
    _labels(d / "cty.csv", n)
    return root


def _caveats(df):
    return [(m, c) for m, c in zip(df["method"], df["caveat"]) if c]


def _cli(argv, capsys):
    rc = _quiet(cli.main, argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


# ======================================================================= R6-10
def test_no_scan_caveat_has_a_semicolon_or_starts_in_lower_case(tmp_path, pinned, root,
                                                                monkeypatch):
    frames = [
        _quiet(mtb.scan, "LUNG_ids", "diagonal", data_path=_lung_ids(tmp_path / "a"),
               verbose=False, allow_atac_mismatch=True),
        _quiet(mtb.scan, "LOGCITE", "vertical", data_path=_log_cite(tmp_path / "b"),
               verbose=False),
        _quiet(mtb.scan, "MU_PEAK", "vertical", data_path=_mu_peak(tmp_path / "c"),
               verbose=False, allow_atac_mismatch=True),
        _quiet(mtb.scan, "D52", "cross", data_path=root / "data", verbose=False),
        _quiet(mtb.scan, "D28", "diagonal", data_path=root / "data", verbose=False),
    ]
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    frames.append(_quiet(mtb.scan, "D45", "mosaic", data_path=root / "data",
                         verbose=False, assume_gpu=True))
    seen = [mc for df in frames for mc in _caveats(df)]
    assert len(seen) > 20
    for m, cav in seen:
        assert "; " not in cav and ";" not in cav, (m, cav)
        # the method name leads (moETM, scMoMaT: as the registry spells it), or
        # the note about the scripts of every method
        assert cav.startswith((f"{m} ", SCRIPTS_NOTE)), (m, cav)
        # no sentence starts with a verb or a label that has lost its subject
        for s in re.split(r"(?<=\.) (?=\S)", cav):
            assert not re.match(r"(reads|expects|needs|assumes|setup|method)\b", s), \
                (m, s, cav)
        assert "setup:" not in cav, (m, cav)


def test_the_glue_uinmf_and_raw_count_caveats_read_as_sentences(tmp_path, pinned, root,
                                                               monkeypatch):
    glue = _quiet(mtb.scan, "LUNG_ids", "diagonal", methods=["GLUE"],
                  data_path=_lung_ids(tmp_path / "a"), verbose=False).iloc[0]
    assert glue["caveat"].startswith(
        "GLUE reads peak names such as chr1:100-200. atac_peak.h5 holds other names, for "
        "example peak_0. Rename them to chr:start-end. GLUE needs the GENCODE v43 human "
        "annotation (gencode.v43.chr_patch_hapl_scaff.annotation.gtf.gz) in "
        "<repo_path>/tools_scripts/GLUE/. "
        f"{SCRIPTS_NOTE}{pinned}. The first real run clones PYangLab/scMultiBench with "
        "git. On a host without network, run multibench fetch --scripts first. GLUE reads "
        "inputs/atac_peak_normpeaks.h5."), glue["caveat"]
    uinmf = _quiet(mtb.scan, "D52", "cross", methods=["UINMF"], data_path=root / "data",
                   verbose=False).iloc[0]
    assert uinmf["caveat"].startswith("UINMF reads batches 1-2 of 3. Batch 3 is not used. "
                                      f"{SCRIPTS_NOTE}{pinned}.")
    tv = _quiet(mtb.scan, "LOGCITE", "vertical", methods=["totalVI"],
                data_path=_log_cite(tmp_path / "b"), verbose=False).iloc[0]
    assert tv["caveat"].startswith("totalVI needs raw counts. rna.h5 holds non-integer "
                                   "values. ")
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    smile = _quiet(mtb.scan, "D45", "mosaic", methods=["SMILE"], data_path=root / "data",
                   verbose=False, assume_gpu=True).iloc[0]
    assert smile["caveat"].startswith("SMILE needs an NVIDIA GPU. This check assumes the "
                                      "job runs on a GPU node. ")


def test_the_atac_caveats_start_with_the_method(tmp_path, pinned):
    sc = _quiet(mtb.scan, "MU_PEAK", "vertical", methods=["Matilda", "scMVP"],
                modalities=["rna", "atac"], data_path=_mu_peak(tmp_path),
                verbose=False, allow_atac_mismatch=True).set_index("method")
    assert sc.loc["Matilda", "caveat"].startswith(
        "Matilda needs gene-activity ATAC. atac.h5 holds peaks, because its features look "
        "like chr:start-end. ")
    # the reason the row carries without the override is unchanged
    blocked = _quiet(mtb.scan, "MU_PEAK", "vertical", methods=["Matilda"],
                     modalities=["rna", "atac"], data_path=tmp_path,
                     verbose=False).iloc[0]
    assert blocked["reason"] == (
        "Matilda needs gene-activity ATAC, and atac.h5 holds peaks. Export the ATAC as gene "
        "activity, or pass allow_atac_mismatch=True to run Matilda anyway.")


def test_callers_print_the_method_name_once(tmp_path, pinned, root, monkeypatch, capsys):
    """run_all's log, both dry runs and run's warning print the caveat alone."""
    data = root / "data"
    _quiet(mtb.run_all, "D52", "cross", methods=["UINMF"], data_path=data, dry_run=True)
    out = capsys.readouterr().out
    assert "[run_all] UINMF reads batches 1-2 of 3. Batch 3 is not used.\n" in out
    rc, out, err = _cli(["run-all", "D52", "--category", "cross", "--methods", "UINMF",
                         "--data-path", str(data), "--dry-run"], capsys)
    assert rc == 0 and "# UINMF reads batches 1-2 of 3. Batch 3 is not used.\n" in err

    def fake_run(**kw):
        raise RuntimeError("stub: not run in tests")
    monkeypatch.setattr(W, "_run", fake_run)
    _quiet(mtb.run_all, "D52", "cross", tmp_path / "out", methods=["UINMF"],
           data_path=data, evaluate=False)
    log = capsys.readouterr().out
    assert "[run_all]   UINMF reads batches 1-2 of 3. Batch 3 is not used.\n" in log
    # run's own warning
    mu = _mu_peak(tmp_path / "mu")
    inp = mtb.inputs_for("MU_PEAK", "vertical", "Matilda", modalities=["rna", "atac"],
                         data_path=mu)

    class Stop(Exception):
        pass

    def stop(spec):
        raise Stop
    monkeypatch.setattr(R, "check_gpu_requirement", stop)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(Stop):
            mtb.run("Matilda", "vertical", inputs=inp, out_dir=str(tmp_path / "o"))
    msgs = [str(w.message) for w in rec if "ATAC" in str(w.message)]
    assert msgs and msgs[0].startswith("Matilda needs gene-activity ATAC. "), msgs
    for text in (out, err, log, *msgs):
        assert not re.search(r"\b(UINMF|Matilda) \1\b", text), text


def test_strict_counts_are_unchanged(tmp_path, pinned, root, capsys):
    data = _lung_ids(tmp_path)
    base = ["scan", "LUNG_ids", "--category", "diagonal", "--data-path", str(data),
            "--strict"]
    rc, _, err = _cli(base, capsys)
    assert rc == 1
    assert err.startswith(
        "error: --strict: 0 of 14 rows are runnable. Rows with missing input files: 13. "
        "Rows whose peak names the method cannot read: 1. Rows whose method scripts are "
        "not fetched: 14. The reason column says why.\n"), err
    rc, _, err = _cli(base + ["--methods", "GLUE,Seurat_v3,MultiMAP,scBridge"], capsys)
    assert rc == 1
    assert err.startswith(
        "error: --strict: 0 of 4 rows are runnable. Rows with missing input files: 3. Rows "
        "whose peak names the method cannot read: 1. Rows whose method scripts are not "
        "fetched: 4. No runnable row for GLUE, Seurat_v3, MultiMAP, scBridge:\n"), err
    assert ("  GLUE: GLUE reads peak names such as chr1:100-200. atac_peak.h5 holds other "
            "names, for example peak_0. Rename them to chr:start-end, or pass "
            "--allow-atac-mismatch to run GLUE anyway.\n") in err
    assert "  scBridge: scBridge needs atac_gas.h5, and LUNG_ids has no such file.\n" in err
    rc, _, err = _cli(["scan", "D52", "--category", "cross", "--data-path",
                       str(root / "data"), "--methods", "UINMF,StabMap", "--strict"], capsys)
    assert rc == 1 and err.startswith(
        "error: --strict: 0 of 2 rows are runnable. Rows whose method scripts are not "
        "fetched: 2. No runnable row for UINMF, StabMap:\n"), err


def test_strict_passes_on_the_uinmf_folder_with_the_scripts(root, monkeypatch, capsys):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config, "scripts_present", lambda cfg=None: True)
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    rc, out, err = _cli(["scan", "D52", "--category", "cross", "--data-path",
                         str(root / "data"), "--methods", "UINMF,StabMap", "--strict"],
                        capsys)
    assert rc == 0, err


def test_scbridge_reason_is_a_sentence(tmp_path, pinned):
    row = _quiet(mtb.scan, "LUNG_ids", "diagonal", methods=["scBridge"],
                 data_path=_lung_ids(tmp_path), verbose=False).iloc[0]
    assert row["reason"] == "scBridge needs atac_gas.h5, and LUNG_ids has no such file."
    assert f"{tmp_path / 'LUNG_ids'} has no such file." in row["files_reason"]


def test_every_setup_hint_starts_with_its_method():
    """scan finds the setup note by its text and shows it as a caveat: it must
    start with the method name like every other caveat."""
    for spec in registry.load():
        if spec.setup_hint:
            assert spec.setup_hint.startswith(f"{spec.id} "), (spec.id, spec.setup_hint)


# ======================================================================= R6-11
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _scripts_repo(path):
    ep = path / registry.get("Matilda").variants[0].entrypoint
    ep.parent.mkdir(parents=True)
    ep.write_text("print(1)\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(path), "PATH": "/usr/bin:/bin"}
    for argv in (["git", "init", "-q"], ["git", "add", "."],
                 ["git", "commit", "-q", "-m", "x"]):
        subprocess.run(argv, cwd=path, check=True, env=env)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def wrong_ref(tmp_path, monkeypatch):
    repo = tmp_path / "scripts"
    head = _scripts_repo(repo)
    monkeypatch.setattr(config.DEFAULT, "repo_path", repo)
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, "deadbeef")
    return repo, head[:7]


@needs_git
def test_strict_prints_the_scripts_ref_reason_in_full(wrong_ref, capsys):
    repo, head = wrong_ref
    rc, _, err = _cli(["scan", "D11", "--category", "vertical", "--methods",
                       "Matilda,scMoMaT", "--modalities", "rna,adt", "--strict"], capsys)
    assert rc == 1
    ref = (f"The method scripts are at {head}, not deadbeef (MULTIBENCH_SCRIPTS_REF). "
           f"Unset MULTIBENCH_SCRIPTS_REF, or set MULTIBENCH_REPO_PATH to a new folder and "
           f"run multibench fetch --scripts.")
    assert f"  Matilda: {ref}\n" in err, err
    # scMoMaT's script is not in this checkout either: that sentence follows, unclipped
    line = next(ln for ln in err.splitlines() if ln.startswith("  scMoMaT: "))
    assert line.startswith(f"  scMoMaT: {ref} ") and not line.endswith("..."), line


@needs_git
def test_the_scripts_ref_reason_is_sentences(wrong_ref):
    repo, head = wrong_ref
    assert config.scripts_ref_problem() == (
        f"The method scripts are at {head}, not deadbeef (MULTIBENCH_SCRIPTS_REF). Unset "
        f"the variable, or fetch that ref into a new repo_path.")
    reason = _quiet(mtb.scan, "D11", "vertical", methods=["Matilda"],
                    modalities=["rna", "adt"], verbose=False)["reason"].iloc[0]
    assert reason.startswith("The method scripts are at "), reason
    # the error of a real run names the folder; the --ref form reads the same way
    with pytest.raises(RuntimeError) as e:
        config.ensure_repo()
    assert str(e.value).startswith(f"The method scripts in {repo} are at {head}, not "
                                   f"deadbeef (MULTIBENCH_SCRIPTS_REF). Unset the variable")
    with pytest.raises(RuntimeError) as e:
        config.ensure_repo(repo, ref="deadbeef")
    assert str(e.value) == (
        f"The method scripts in {repo} are at {head}, not deadbeef (--ref). Set "
        f"MULTIBENCH_REPO_PATH to a new folder and run multibench fetch --scripts --ref "
        f"deadbeef.")


# ======================================================================= R6-12
def test_nothing_runnable_heads_are_sentences(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir=tmp_path, verbose=False)
    assert str(e.value).startswith("No method can run on D11 (vertical).\n")
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir=tmp_path, methods=["Matilda", "totalVI"],
                    verbose=False)
    assert str(e.value).startswith(
        "None of the requested methods (Matilda, totalVI) can run on D11 (vertical).\n")
    assert "dataset=" not in str(e.value) and "methods=[" not in str(e.value).split("\n")[0]


def test_run_all_dry_run_header_is_sentences(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    rc, out, err = _cli(["run-all", "D28", "--category", "diagonal", "--dry-run",
                         "--out-dir", str(tmp_path)], capsys)
    assert rc == 0
    assert re.search(r"^# Commands of the 13 rows whose input files are in place\. "
                     r"\[env missing\] marks a row whose environment is not installed\. "
                     r"\[use multibench run\] marks a command that reads a file multibench "
                     r"run writes first\.$", out, re.M), out
    # D28 has one row per method: the count line counts methods
    assert re.match(r"# Dry run\. Nothing was executed\. 0 of 14 methods can run on D28 "
                    r"\(diagonal\)\.", err), err


def test_count_lines_say_methods_only_when_each_has_one_row(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    df = mtb.scan("D28", "diagonal")
    assert df["method"].is_unique
    out = capsys.readouterr().out
    assert out.startswith(f"[scan] {int(df['files_ok'].sum())} of {len(df)} methods have "
                          f"their input files. 0 of {len(df)} have their environment "
                          f"installed."), out
    d11 = mtb.scan("D11", "vertical")
    assert not d11["method"].is_unique
    assert f"of {len(d11)} rows have their input files." in capsys.readouterr().out
    mtb.run_all("D52", "cross", out_dir=tmp_path, dry_run=True)
    assert "[run_all] Dry run: 0 of 8 requested methods can run on D52 (cross)." in \
        capsys.readouterr().out


def test_linux_only_summary_is_three_sentences(monkeypatch, capsys):
    assert W.LINUX_ONLY_SUMMARY == (
        "Method environments run only on Linux. On this computer you can check files, "
        "score embeddings and plot. The commands use this computer's paths, so run scan "
        "again on the Linux machine.")
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "not linux")
    monkeypatch.setattr(W, "_installed_envs", lambda: frozenset())
    mtb.scan("D11", "vertical", methods=["totalVI"])
    assert W.LINUX_ONLY_SUMMARY in capsys.readouterr().out


def _lung(root):
    d = root / "LUNG"
    d.mkdir(parents=True)
    _h5(d / "rna.h5", GENES, [f"r{i}" for i in range(20)])
    _h5(d / "atac_peak.h5", PEAKS, [f"a{i}" for i in range(20)])
    _labels(d / "rna_cty.csv", 20)
    _labels(d / "atac_cty.csv", 20)
    return d


def test_inputs_for_names_what_is_missing_in_sentences(tmp_path, monkeypatch):
    d = _lung(tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        mtb.inputs_for("LUNG", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    assert str(e.value) == (
        f"SCALEX (diagonal) needs atac_gas.h5 in {d}. The folder holds atac_cty.csv, "
        f"atac_peak.h5, rna.h5 and rna_cty.csv. Diagonal methods read atac_gas.h5 or "
        f'atac.h5. method_info("SCALEX")["atac"] says which ATAC SCALEX needs.')
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.raises(FileNotFoundError) as e:
        mtb.inputs_for("LUNG", "diagonal", "SCALEX", data_path=tmp_path, check=True)
    assert str(e.value).endswith("multibench info SCALEX says which ATAC SCALEX needs.")


def test_per_batch_hint_is_sentences(tmp_path):
    d = tmp_path / "MB"
    d.mkdir()
    for i in (1, 2):
        bars = [f"b{i}_{j}" for j in range(20)]
        _h5(d / f"rna{i}.h5", GENES[:30], bars)
        _h5(d / f"adt{i}.h5", [f"p{k}" for k in range(6)], bars)
        _labels(d / f"cty{i}.csv", 20)
    with pytest.raises(FileNotFoundError) as e:
        mtb.inputs_for("MB", "vertical", "totalVI", data_path=tmp_path, check=True)
    msg = str(e.value)
    assert msg.endswith(
        "This folder holds per-batch files (adt1.h5, adt2.h5, ...). Vertical methods read "
        "one adt.h5. Export without batch= and pass the batch column to "
        "run_all(batch=...) or evaluate(batch=...). For RNA+ADT batches, use "
        'category="cross".') or msg.endswith(
        "This folder holds per-batch files (rna1.h5, rna2.h5, ...). Vertical methods read "
        "one rna.h5. Export without batch= and pass the batch column to "
        "run_all(batch=...) or evaluate(batch=...). For RNA+ADT batches, use "
        'category="cross".'), msg
    assert " - " not in msg and "; " not in msg


def test_save_merge_line_is_a_sentence(tmp_path, capsys):
    def res(method):
        rec = {"method": method, "category": "vertical", "dataset": "D11",
               "modalities": ["rna", "adt"], "status": "CHAIN_OK", "out_dir": f"x/{method}",
               "metrics": {"ARI": 0.5, "NMI": 0.5}}
        return mtb.BatchResult([rec], "D11", "vertical")
    res("StabMap").save(tmp_path)
    res("scMoMaT").save(tmp_path)
    assert capsys.readouterr().out == (
        f"# Merged with 1 earlier record in {tmp_path} (StabMap).\n")
    res("totalVI").save(tmp_path)
    assert capsys.readouterr().out == (
        f"# Merged with 2 earlier records in {tmp_path} (StabMap, scMoMaT).\n")


def test_mosaic_gene_activity_warning_names_the_fix(tmp_path, monkeypatch):
    import anndata as ad
    a = ad.AnnData(X=np.ones((6, 4)))
    a.var_names = [f"g{i}" for i in range(4)]
    with pytest.warns(UserWarning) as rec:
        ingest.export_dataset(a, tmp_path / "PY", rna=None, atac="X",
                              atac_kind="gene_activity", category="mosaic", batch_index=1)
    msgs = [str(w.message) for w in rec if "mosaic" in str(w.message)]
    assert ("Every mosaic method reads peak ATAC. Export the ATAC as peaks "
            "(atac_kind='peak'), or pass allow_atac_mismatch=True to scan and run_all.") \
        in msgs, msgs
    monkeypatch.setattr(config, "_CLI", True)
    with pytest.warns(UserWarning) as rec:
        ingest.export_dataset(a, tmp_path / "CLI", rna=None, atac="X",
                              atac_kind="gene_activity", category="mosaic", batch_index=1)
    msgs = [str(w.message) for w in rec if "mosaic" in str(w.message)]
    assert ("Every mosaic method reads peak ATAC. Convert the ATAC as peaks (--atac-kind "
            "peak), or pass --allow-atac-mismatch to scan and run-all.") in msgs, msgs


def test_positional_warning_is_plural_for_several_files(tmp_path):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(40, 5))
    for name, prefix in (("lab_atac.csv", "a"), ("lab_rna.csv", "r")):
        pd.DataFrame({"barcode": [f"{prefix}{i}" for i in range(20)],
                      "cell_type": ["T", "B"] * 10}).to_csv(tmp_path / name, index=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mtb.evaluate(emb, labels=[str(tmp_path / "lab_atac.csv"),
                                  str(tmp_path / "lab_rna.csv")], metrics=["ARI"])
    msgs = [str(w.message) for w in rec if "matched by position" in str(w.message)]
    assert msgs == [
        "The first column of lab_atac.csv and lab_rna.csv looks like cell ids, but the "
        "output has no cell ids. These files are matched by position. Check that their "
        "rows follow the rows of the output."], msgs


# ======================================================================= R6-14
def test_notes_passages_have_no_dash_insertions():
    flat = {fn: " ".join(inspect.getdoc(fn).split()) for fn in (
        mtb.BatchResult.results.fget, mtb.run, mtb.io.export_dataset,
        mtb.BatchResult.rescore, mtb.recommend)}
    assert ("``label_order_candidates`` holds every ordering tried and its ARI. "
            "``summary``'s ``label_order_confidence`` is computed from them.") in \
        flat[mtb.BatchResult.results.fget]
    assert ("``cpu_params`` are the flags that turn CUDA off in a script that uses it by "
            "default") in flat[mtb.run]
    assert ("Every modality is re-indexed by barcode to one order: ``data.obs_names``, or "
            "the first modality's ``obs_names`` when ``data`` is not given.") in \
        flat[mtb.io.export_dataset]
    assert ("Without ``labels=``, a ``batch`` array follows the order of "
            "``mtb.labels_for(dataset)``.") in flat[mtb.BatchResult.rescore]
    assert 'The re-run tables may hold more of these methods (``source="rerun"``).' in \
        flat[mtb.recommend]
    for doc in flat.values():
        assert " - the evidence" not in doc and "- the flags" not in doc
        assert "written in one order -" not in doc and "need not score" not in doc
