"""Seams between the round-4 package work packages (wp/f4_wf, _ing, _plot, _cli).

- run-all's failure line counts the methods of the run: a SKIPPED record of
  a method nobody named (R4-04) is not one of them (R4-03).
- scan --strict with --allow-atac-mismatch still needs fetched method
  scripts (R4-01 with R4-06).
"""
from __future__ import annotations

import warnings

from multibench import config
from multibench import workflow as W
from multibench.engine import envs
from tests.test_study_r4_wf import ALL_ENVS, _gasmos


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
    # R7-07: the note is printed with verbose=True only, after the count line
    assert sentence not in capsys.readouterr().err
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W.run_all("GASMOS", "mosaic", data_path=root, dry_run=True, methods=["StabMap"],
                  verbose=True)
    out = capsys.readouterr().out
    assert out.count(f"[run_all] {sentence}.\n") == 1, out
