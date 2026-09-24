"""Round-4 student study, package work 'ing': R4-06 (method scripts not fetched).

- An existing empty ``repo_path`` is filled in place by ``multibench fetch
  --scripts`` and by the first run; the folder itself is kept.
- A ``repo_path`` that holds other files and no ``tools_scripts/`` is refused
  with one sentence, which ``scan`` (reason, not runnable), the dry run and
  ``multibench config`` repeat.
- ``scan --strict`` fails while the method scripts are not fetched; the Python
  ``scan`` keeps such rows runnable, with the caveat.
"""
import os
import re
import shutil
import stat
import subprocess
import warnings

import pytest

import multibench as mtb
from multibench import cli, config
from multibench import workflow as W
from multibench.engine import envs, registry

ALL_ENVS = frozenset(envs.group_for(m) for m in registry.list_methods())
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
#: the fix of a folder with other files: the safe action first (review of wp/f4_int)
FIX = ("an empty or new folder, then run multibench fetch --scripts. If the folder is "
       "left over from an earlier fetch, you can remove it instead")
PY_FIX = f"Set MULTIBENCH_REPO_PATH or mtb.config.DEFAULT.repo_path to {FIX}"
CLI_FIX = f"Set MULTIBENCH_REPO_PATH to {FIX}"
OTHER = "holds other files and no method scripts."
MATILDA = ["scan", "D11", "--category", "vertical", "--methods", "Matilda",
           "--modalities", "rna,adt"]


def _upstream(path):
    """A local git repository standing in for PYangLab/scMultiBench; returns HEAD."""
    ep = path / registry.get("Matilda").variants[0].entrypoint
    ep.parent.mkdir(parents=True)
    ep.write_text("print(1)\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(path), "PATH": "/usr/bin:/bin"}
    for argv in (["git", "init", "-q"], ["git", "add", "."],
                 ["git", "commit", "-q", "-m", "x"], ["git", "tag", "v1"]):
        subprocess.run(argv, cwd=path, check=True, env=env)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def host(tmp_path, monkeypatch):
    """No scripts in the package root; every env installed; a GPU; no ref pinned."""
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")
    monkeypatch.setattr(W, "_installed_envs", lambda: ALL_ENVS)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    monkeypatch.delenv(config.SCRIPTS_REF_VAR, raising=False)
    return tmp_path


def _use(monkeypatch, repo):
    monkeypatch.setattr(config.DEFAULT, "repo_path", repo)


def _scan_row(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.scan("D11", "vertical", methods=["Matilda"], modalities=["rna", "adt"],
                        verbose=False, **kw).iloc[0]


# ================================================= (1) an empty folder is filled
@needs_git
def test_fetch_scripts_fills_an_existing_empty_folder(host, monkeypatch, capsys):
    head = _upstream(host / "upstream")
    monkeypatch.setattr(config, "SCRIPTS_URL", (host / "upstream").as_uri())
    shared = host / "shared"
    target = shared / "scripts"
    target.mkdir(parents=True)                       # mkdir -p before the fetch
    target.chmod(0o750)
    before = os.stat(target)
    _use(monkeypatch, target)
    # a parent the user cannot write to: the folder is filled from the inside
    shared.chmod(0o555)
    try:
        rc = cli.main(["fetch", "--scripts"])
    finally:
        shared.chmod(0o755)
    out = capsys.readouterr().out
    assert rc == 0
    assert out == f"method scripts fetched: {target / 'tools_scripts'} at {head}\n"
    after = os.stat(target)
    assert after.st_ino == before.st_ino                 # the same folder, kept
    assert stat.S_IMODE(after.st_mode) == 0o750
    assert (target / registry.get("Matilda").variants[0].entrypoint).is_file()
    assert config.scripts_commit(target) == head
    assert sorted(p.name for p in shared.iterdir()) == ["scripts"]   # no leftover
    assert config._PARTIAL not in os.listdir(target)
    # the second call finds them
    assert cli.main(["fetch", "--scripts"]) == 0
    assert capsys.readouterr().out.startswith("method scripts present: ")


@needs_git
def test_first_run_fills_an_empty_folder_at_a_ref(host, monkeypatch):
    head = _upstream(host / "upstream")
    monkeypatch.setattr(config, "SCRIPTS_URL", (host / "upstream").as_uri())
    target = host / "scripts"
    (target / config._PARTIAL / "junk").mkdir(parents=True)   # an interrupted fetch
    assert config.scripts_folder_problem(target) is None
    assert config.ensure_repo(target, ref="v1") == target
    assert config.scripts_commit(target) == head
    assert config.scripts_ref_problem(target) is None
    assert not (target / config._PARTIAL).exists()


def test_a_failed_fetch_leaves_the_empty_folder_as_it_was(host, monkeypatch):
    target = host / "scripts"
    target.mkdir()

    def dead_proxy(argv, **kw):
        raise subprocess.CalledProcessError(128, argv)
    monkeypatch.setattr(subprocess, "run", dead_proxy)
    with pytest.raises(RuntimeError, match="^could not reach github.com"):
        config.ensure_repo(target)
    assert target.is_dir() and not os.listdir(target)


# ============================== a folder with other files is refused, in one sentence
def test_a_folder_with_other_files_is_refused_plainly(host, monkeypatch, capsys):
    target = host / "scripts"
    target.mkdir()
    (target / "gencode.gtf").write_text("x\n")
    sentence = f"{target} {OTHER} {PY_FIX}"
    with pytest.raises(RuntimeError) as e:
        config.ensure_repo(target)
    assert str(e.value) == sentence + "."
    assert " - " not in str(e.value)
    _use(monkeypatch, target)
    assert cli.main(["fetch", "--scripts"]) == 1
    assert capsys.readouterr().err.startswith(f"error: {target} {OTHER} {CLI_FIX}.\n")
    assert os.listdir(target) == ["gencode.gtf"]                 # nothing deleted
    assert config.scripts_folder_problem() == sentence


def test_scan_and_config_name_a_folder_with_other_files(host, monkeypatch, capsys):
    target = host / "scripts"
    target.mkdir()
    (target / "gencode.gtf").write_text("x\n")
    _use(monkeypatch, target)
    sentence = f"{target} {OTHER} {PY_FIX}"
    cli_sentence = f"{target} {OTHER} {CLI_FIX}"
    row = _scan_row()
    assert not row["runnable"]
    assert row["reason"] == sentence + "."
    assert row["files_ok"] and row["env_ok"]          # its own blocker, like the ref
    assert "first real run" not in row["caveat"]
    # the dry run notes the same sentence once, not a clone on the first run
    inp = mtb.inputs_for("D11", "vertical", "Matilda")
    mtb.run("Matilda", "vertical", inputs=inp, out_dir=str(host / "o"), dry_run=True)
    err = capsys.readouterr().err
    assert err.count(f"# {sentence}\n") == 1 and "first real run" not in err
    # config
    row = next(r for r in config._sources() if r["name"] == "scripts_commit")
    assert (row["value"], row["source"]) == ("not fetched", sentence)
    assert cli.main(["config"]) == 0
    assert f"({cli_sentence})" in capsys.readouterr().out
    # --strict: counted as not fetched; the tail is the sentence, not a fetch
    # that would refuse
    assert cli.main(MATILDA + ["--strict"]) == 1
    err = capsys.readouterr().err
    assert "Rows whose method scripts are not fetched: 1." in err
    assert err.rstrip().endswith(cli_sentence + ".")
    assert "Run multibench fetch --scripts first." not in err


def test_config_keeps_the_fetch_hint_for_an_empty_folder(host, monkeypatch):
    target = host / "scripts"
    target.mkdir()
    _use(monkeypatch, target)
    row = next(r for r in config._sources() if r["name"] == "scripts_commit")
    assert (row["value"], row["source"]) == (
        "not fetched", "multibench fetch --scripts fetches the method scripts")


# ================================================= (2) --strict: scripts not fetched
@pytest.mark.parametrize("make", [True, False], ids=["empty folder", "no folder"])
def test_scan_strict_fails_while_the_scripts_are_not_fetched(host, monkeypatch, capsys,
                                                             make):
    target = host / "emptyrepo"
    if make:
        target.mkdir()
    _use(monkeypatch, target)
    # the Python scan: runnable, with the caveat (a host with network clones)
    row = _scan_row()
    assert row["runnable"] and row["reason"] == ""
    assert f"method scripts not found under {target}: the first real run" in row["caveat"]
    assert cli.main(MATILDA) == 0                       # without --strict: unchanged
    capsys.readouterr()
    assert cli.main(MATILDA + ["--strict"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: --strict: 0 of 1 row is runnable. Rows whose method "
                          "scripts are not fetched: 1.")
    assert "  Matilda: method scripts not fetched\n" in err
    assert err.rstrip().endswith("Run multibench fetch --scripts first.")
    assert "no row" not in err


def test_scan_strict_names_fetch_without_methods(host, monkeypatch, capsys):
    _use(monkeypatch, host / "none")
    argv = ["scan", "D11", "--category", "vertical", "--modalities", "rna,adt", "--strict"]
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    # the rows blocked only by the scripts have no reason: the head does not
    # send the reader to that column
    assert re.search(r"^error: --strict: 0 of (\d+) rows are runnable\. Rows whose method "
                     r"scripts are not fetched: \1\.\n", err), err
    assert "reason column" not in err
    assert err.rstrip().endswith("Run multibench fetch --scripts first.")


def test_the_student_gate_on_a_mosaic_dataset(host, monkeypatch, capsys):
    """S5-R4-03: StabMap and scMoMaT, --assume-gpu, an empty scripts folder."""
    target = host / "emptyrepo2"
    target.mkdir()
    _use(monkeypatch, target)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    argv = ["scan", "D46", "--category", "mosaic", "--methods", "StabMap,scMoMaT",
            "--strict", "--assume-gpu"]
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "Rows whose method scripts are not fetched: 2." in err
    assert "Run multibench fetch --scripts first." in err


@needs_git
def test_scan_strict_passes_once_the_scripts_are_fetched(host, monkeypatch, capsys):
    _upstream(host / "upstream")
    monkeypatch.setattr(config, "SCRIPTS_URL", (host / "upstream").as_uri())
    target = host / "scripts"
    target.mkdir()
    _use(monkeypatch, target)
    assert cli.main(MATILDA + ["--strict"]) == 1
    assert cli.main(["fetch", "--scripts"]) == 0
    capsys.readouterr()
    assert cli.main(MATILDA + ["--strict"]) == 0, capsys.readouterr().err


def test_scan_strict_help_names_the_scripts():
    act = next(a for a in cli.build_parser()._subparsers._group_actions[0]
               .choices["scan"]._actions if a.dest == "strict")
    text = " ".join(act.help.split())
    assert "; also when the method scripts are not fetched (for scripts: " in text
    assert text.startswith("exit 1 when no requested row is runnable; with --methods, "
                           "when any named method has none")
