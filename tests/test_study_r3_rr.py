"""Round-3 student study, package work 'rr': R3-04, R3-10."""
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
from multibench.engine import envs, ingest, registry, runner

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())


# ============================================================ R3-04 GLUE peak names
def _h5(path, feats, bars):
    with h5py.File(path, "w") as f:
        f.create_dataset("matrix/data", data=np.ones((len(feats), len(bars))))
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))


def _diagonal(root, name, peaks):
    """A diagonal folder whose atac_peak.h5 holds ``peaks``."""
    d = root / name
    d.mkdir()
    rna, atac = [f"r{i}" for i in range(20)], [f"a{i}" for i in range(20)]
    genes = [f"G{i}" for i in range(30)]
    _h5(d / "rna.h5", genes, rna)
    _h5(d / "atac_peak.h5", peaks, atac)
    _h5(d / "atac_gas.h5", genes, atac)
    pd.DataFrame({"x": ["A"] * 20}).to_csv(d / "rna_cty.csv", index=False)
    pd.DataFrame({"x": ["A"] * 20}).to_csv(d / "atac_cty.csv", index=False)
    return d


def _scan_row(root, name, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = mtb.scan(name, "diagonal", methods=[method], data_path=root, verbose=False)
    return sc.iloc[0]


@pytest.mark.parametrize("method", ["GLUE", "Seurat_v3"])
def test_underscore_peaks_are_passed_as_a_rewritten_copy(tmp_path, method):
    _diagonal(tmp_path, "LUNG_us", [f"chr4_{171325400 + 10 * i}_{171325903 + 10 * i}"
                                    for i in range(60)])
    inp = mtb.inputs_for("LUNG_us", "diagonal", method, data_path=tmp_path)
    argv = mtb.run(method, "diagonal", inputs=inp, out_dir=str(tmp_path / "o"),
                   dry_run=True)
    assert str(tmp_path / "o" / "inputs" / "atac_peak_normpeaks.h5") in argv
    assert inp["atac_peak"] not in argv
    row = _scan_row(tmp_path, "LUNG_us", method)
    assert row["files_ok"]
    assert "peak names" not in row["caveat"]
    assert "inputs/atac_peak_normpeaks.h5" in row["caveat"]      # the prepared-file note


def test_dash_peaks_need_no_caveat(tmp_path):
    _diagonal(tmp_path, "LUNG_dash", [f"chr1-{100 + i}-{200 + i}" for i in range(60)])
    assert "peak names" not in _scan_row(tmp_path, "LUNG_dash", "GLUE")["caveat"]


def test_peak_ids_without_coordinates_get_the_peak_names_caveat(tmp_path):
    _diagonal(tmp_path, "LUNG_ids", [f"peak_{i}" for i in range(1, 61)])
    cav = _scan_row(tmp_path, "LUNG_ids", "GLUE")["caveat"]
    assert cav.startswith("GLUE reads peak names such as chr1:100-200; atac_peak.h5 "
                          "holds other names (e.g. peak_1)")
    # one caveat for the file, not also the representation guess
    assert "holds gene activity" not in cav
    # the same content check for the other method whose peaks mtb.run renames
    cav = _scan_row(tmp_path, "LUNG_ids", "Seurat_v3")["caveat"]
    assert "Seurat_v3 reads peak names such as chr1:100-200" in cav


def test_d28_glue_caveat_does_not_name_the_dataset():
    r = mtb.scan("D28", "diagonal", methods=["GLUE"], verbose=False).iloc[0]
    assert "D28" not in r["caveat"] and "peak names" not in r["caveat"]


def test_normalize_peak_names_notes_name_the_methods_run_renames_for():
    """The Notes list exactly the methods with a renamed peak role."""
    renamed = sorted(m for m in registry.list_methods()
                     if any(v.normalize_peaks for v in registry.get(m).variants))
    assert renamed == ["GLUE", "Seurat_v3"]
    doc = " ".join(inspect.getdoc(ingest.normalize_peak_names).split())
    para = doc.split("**Inside ``mtb.run``.**", 1)[1].split("**", 1)[0]
    assert "applies this itself for GLUE and Seurat_v3," in para


# ============================================================ R3-10 scripts ref
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _scripts_repo(path):
    """A git checkout with Matilda's entrypoint and one commit; returns HEAD."""
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
def scripts(tmp_path, monkeypatch):
    """The method scripts at a git commit; every env installed; a GPU."""
    repo = tmp_path / "scripts"
    head = _scripts_repo(repo)
    monkeypatch.setattr(config.DEFAULT, "repo_path", repo)
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    return repo, head


def _mismatch(head):
    return (f"method scripts are at {head[:7]}, not deadbeef (MULTIBENCH_SCRIPTS_REF). "
            f"Fetch that ref into a new repo_path, or unset the variable")


@needs_git
def test_scan_blocks_every_row_under_another_scripts_ref(scripts, monkeypatch):
    repo, head = scripts
    before = mtb.scan("D11", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                      verbose=False)
    assert before["runnable"].all()
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, "deadbeef")
    sc = mtb.scan("D11", "vertical", verbose=False)
    assert not sc["runnable"].any()
    assert (sc["reason"].str.startswith(_mismatch(head))).all()
    # the matching ref changes nothing
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, head[:7])
    after = mtb.scan("D11", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                     verbose=False)
    pd.testing.assert_frame_equal(before, after)


@needs_git
def test_scan_strict_fails_under_another_scripts_ref(scripts, monkeypatch, capsys):
    repo, head = scripts
    argv = ["scan", "D11", "--category", "vertical", "--methods", "Matilda",
            "--modalities", "rna,adt", "--strict"]
    assert cli.main(argv) == 0
    capsys.readouterr()
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, "deadbeef")
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    # the CLI spelling of the fix
    assert f"method scripts are at {head[:7]}, not deadbeef (MULTIBENCH_SCRIPTS_REF)" in err
    assert "repo_path" not in err.split("MULTIBENCH_SCRIPTS_REF)", 1)[1]


@needs_git
def test_dry_runs_note_another_scripts_ref(scripts, monkeypatch, capsys, tmp_path):
    repo, head = scripts
    inp = mtb.inputs_for("D11", "vertical", "Matilda")
    mtb.run("Matilda", "vertical", inputs=inp, out_dir=str(tmp_path / "o"), dry_run=True)
    assert "MULTIBENCH_SCRIPTS_REF" not in capsys.readouterr().err
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, "deadbeef")
    mtb.run("Matilda", "vertical", inputs=inp, out_dir=str(tmp_path / "o"), dry_run=True)
    assert capsys.readouterr().err.count(f"# {_mismatch(head)}\n") == 1
    mtb.run_all("D11", "vertical", methods=["Matilda"], dry_run=True, verbose=False)
    assert capsys.readouterr().err.count(f"# {_mismatch(head)}\n") == 1
    # the CLI: one note, exit code unchanged
    pairs = [a for role, path in inp.items() for a in ("--input", f"{role}={path}")]
    rc = cli.main(["run", "--method", "Matilda", "--category", "vertical", *pairs,
                   "--out", str(tmp_path / "o"), "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 0
    assert err.count(f"# method scripts are at {head[:7]}, not deadbeef") == 1
    assert "`multibench fetch --scripts`" in err
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--methods", "Matilda",
                   "--dry-run"])
    assert rc == 0
    assert capsys.readouterr().err.count("# method scripts are at ") == 1
    # the real run refuses with the same sentence
    with pytest.raises(RuntimeError, match=re.escape(_mismatch(head))):
        config.ensure_repo()


@needs_git
def test_config_shows_the_scripts_ref_row(scripts, monkeypatch, capsys):
    repo, head = scripts
    assert "scripts_ref" not in [r["name"] for r in config._sources()]
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, "deadbeef")
    rows = {r["name"]: r for r in config._sources()}
    names = [r["name"] for r in config._sources()]
    assert names.index("scripts_ref") == names.index("scripts_commit") + 1
    assert rows["scripts_ref"]["value"] == "deadbeef"
    assert rows["scripts_ref"]["source"] == (
        f"environment variable MULTIBENCH_SCRIPTS_REF; does not match scripts_commit "
        f"{head[:7]}")
    assert cli.main(["config"]) == 0
    assert f"does not match scripts_commit {head[:7]}" in capsys.readouterr().out
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, head[:10])
    row = next(r for r in config._sources() if r["name"] == "scripts_ref")
    assert row["source"].endswith("; matches scripts_commit")


def test_scripts_ref_problem_is_none_without_scripts_or_variable(tmp_path, monkeypatch):
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "none")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setenv(config.SCRIPTS_REF_VAR, "deadbeef")
    assert config.scripts_ref_problem() is None          # the first fetch checks it out
    monkeypatch.delenv(config.SCRIPTS_REF_VAR)
    assert config.scripts_ref_problem(tmp_path) is None
