"""Seams between the round-4 package work packages (wp/f4_wf, _ing, _plot, _cli).

- run-all's failure line counts the methods of the run: a SKIPPED record of
  a method nobody named (R4-04) is not one of them (R4-03).
- scan --strict with --allow-atac-mismatch still needs fetched method
  scripts (R4-01 with R4-06).
"""
from __future__ import annotations

import warnings

from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs
from tests.test_study_r4_wf import ALL_ENVS, _gasmos


def _rec(method, status, **kw):
    return {"method": method, "status": status, "category": "vertical",
            "dataset": "MYCITE", "modalities": ["rna", "adt"], "error": kw.pop("error", None),
            **kw}


def test_failed_line_does_not_count_unnamed_skipped_methods():
    res = W.BatchResult([_rec("Matilda", "CHAIN_OK"),
                         _rec("totalVI", "FAIL", error="boom"),
                         _rec("scMDC", "SKIPPED", error="needs a GPU", requested=False)],
                        "MYCITE", "vertical")
    assert cli._failed_line(res, "out/failures.csv") == (
        "# 1 of 2 methods failed: totalVI (FAIL). See out/failures.csv.")


def test_failed_line_counts_a_named_skipped_method():
    res = W.BatchResult([_rec("Matilda", "CHAIN_OK"),
                         _rec("UnitedNet", "SKIPPED", error="needs a GPU", requested=True)],
                        "MYCITE", "vertical")
    assert cli._failed_line(res, "f.csv") == (
        "# 1 of 2 methods failed: UnitedNet (SKIPPED). See f.csv.")


def test_strict_with_allow_atac_mismatch_still_needs_fetched_scripts(tmp_path, monkeypatch,
                                                                     capsys):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", tmp_path / "no_scripts")
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    root = _gasmos(tmp_path)
    argv = ["scan", "GASMOS", "--category", "mosaic", "--data-path", str(root),
            "--methods", "StabMap,scMoMaT", "--strict", "--assume-gpu",
            "--allow-atac-mismatch"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli.main(argv)
    err = capsys.readouterr().err
    assert rc == 1 and "Method scripts are not fetched in 2." in err, err
    assert err.rstrip().endswith("Run multibench fetch --scripts first."), err
    monkeypatch.setattr(config, "scripts_present", lambda cfg=None: True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert cli.main(argv) == 0, capsys.readouterr().err


def test_run_all_dry_run_names_a_scripts_folder_without_scripts(tmp_path, monkeypatch,
                                                                capsys):
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    repo = tmp_path / "other"
    repo.mkdir()
    (repo / "notes.txt").write_text("x")
    monkeypatch.setattr(config.DEFAULT, "repo_path", repo)
    root = _gasmos(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = W.run_all("GASMOS", "mosaic", data_path=root, dry_run=True,
                       methods=["StabMap"], verbose=False)
    sentence = (f"{repo} holds other files and no method scripts. Set MULTIBENCH_REPO_PATH "
                f"or mtb.config.DEFAULT.repo_path to an empty or new folder, then run "
                f"multibench fetch --scripts. If the folder is left over from an earlier "
                f"fetch, you can remove it instead")
    assert not df["runnable"].any() and df["reason"].str.startswith(sentence).all()
    err = capsys.readouterr().err
    assert err.count(f"# {sentence}\n") == 1, err
