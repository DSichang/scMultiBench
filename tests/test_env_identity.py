"""An env counts as a method's env only when multibench installed it.

A user's own ``conda create -n matilda`` env was taken as Matilda's env:
scan said runnable, doctor ``[x]``, install skipped it and run used it. Now
every install path writes ``.multibench_flavor`` into the env (packed
archives their flavour, lockfile builds ``build``, recipe builds
``recipe``); a lockfile build of an earlier multibench, which wrote no such
file, is known by the build command conda keeps in ``conda-meta/history``.
Any other env of that name is reported and never used. Off Linux, ``run``
refuses whether or not the env exists.

The tests run against a fake conda root: a ``conda`` script that lists,
creates and runs envs under ``<tmp>/root/envs``.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

import multibench as mtb
from multibench import config
from multibench.engine import envs, runner
from tests.conftest import full_scan

FAKE_CONDA = r"""#!/bin/sh
ROOT="__ROOT__"
case "$1" in
  info) printf '{"envs_dirs": ["%s/envs"], "root_prefix": "%s"}\n' "$ROOT" "$ROOT" ;;
  env)
    case "$2" in
      list)
        printf '{"envs": ['; sep=""
        for d in "$ROOT"/envs/*/; do
          [ -d "$d" ] || continue
          printf '%s"%s"' "$sep" "${d%/}"; sep=", "
        done
        printf ']}\n' ;;
      create)
        name=""; prev=""
        for a in "$@"; do [ "$prev" = "-n" ] && name="$a"; prev="$a"; done
        [ -e "$ROOT/envs/$name" ] && { echo "prefix already exists" >&2; exit 1; }
        mkdir -p "$ROOT/envs/$name/bin" "$ROOT/envs/$name/conda-meta"
        printf '==> 2026-10-16 <==\n# cmd: %s %s\n' "$0" "$*" > "$ROOT/envs/$name/conda-meta/history" ;;
    esac ;;
  create)
    name=""; prev=""
    for a in "$@"; do [ "$prev" = "-n" ] && name="$a"; prev="$a"; done
    mkdir -p "$ROOT/envs/$name/bin" "$ROOT/envs/$name/conda-meta"
    printf '==> 2026-10-16 <==\n# cmd: %s %s\n' "$0" "$*" > "$ROOT/envs/$name/conda-meta/history" ;;
  run) exit 0 ;;
esac
"""


def _env(root: Path, name: str, *, history: str | None = None, flavor: str | None = None):
    p = root / "envs" / name
    (p / "bin").mkdir(parents=True)
    (p / "conda-meta").mkdir()
    if history is not None:
        (p / "conda-meta" / "history").write_text(f"==> 2026-08-20 <==\n# cmd: {history}\n"
                                                  "+conda-forge/noarch::pip-24.0\n")
    if flavor is not None:
        (p / envs.FLAVOR_FILE).write_text(flavor + "\n")
    return p


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """A conda root with a user's own ``matilda`` env (same name as Matilda's),
    a packed ``env_sciPENN``, a ``scmb_torch`` built from a lockfile by 0.3.x
    (no marker file), and a conda executable that knows them."""
    root = tmp_path / "root"
    (root / "envs").mkdir(parents=True)
    exe = tmp_path / "bin" / "conda"
    exe.parent.mkdir()
    exe.write_text(FAKE_CONDA.replace("__ROOT__", str(root)))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{exe.parent}:/usr/bin:/bin")
    monkeypatch.setattr(config, "envs_dir", root / "envs")
    _env(root, "matilda", history=f"{exe} create -n matilda python=3.11")
    _env(root, "env_sciPENN", history="/home/u/conda create -n env_sciPENN", flavor="cpu")
    _env(root, "scmb_torch",
         history=f"{exe} env create -n scmb_torch -f "
                 "/tmp/multibench_envbuild/scmb_torch/conda.yml")
    envs._conda_prefixes.cache_clear()
    yield root
    envs._conda_prefixes.cache_clear()


def test_a_users_same_named_env_is_not_the_method_env(fake_root):
    assert envs.group_for("Matilda") == "matilda"
    assert envs.env_prefix("matilda") is None
    assert envs.other_env_prefix("matilda") == fake_root / "envs" / "matilda"
    assert "matilda" not in envs.installed_envs()


def test_a_packed_install_is_the_method_env(fake_root):
    assert envs.env_prefix("env_sciPENN") == fake_root / "envs" / "env_sciPENN"
    assert envs.other_env_prefix("env_sciPENN") is None


def test_a_lockfile_build_of_an_earlier_version_is_the_method_env(fake_root):
    assert envs.env_prefix("scmb_torch") == fake_root / "envs" / "scmb_torch"
    assert sorted(envs.installed_envs()) == ["env_sciPENN", "scmb_torch"]


@pytest.mark.parametrize("cmd", [
    "/opt/conda/bin/mamba env create -y -n scmb_r -f /x/env_locks/.build/scmb_r/conda.yml",
    "/opt/conda/bin/conda env create -n scmb_r -f /x/multibench/engine/env_locks/scmb_r.yml",
    "/opt/conda/bin/conda env create -n scmb_r -f /tmp/multibench_envbuild/scmb_r/conda.yml",
    # older conda records its env subcommand as conda-env
    "/opt/conda/bin/conda-env create -n scmb_r -f /tmp/multibench_envbuild/scmb_r/conda.yml",
])
def test_the_three_lockfile_build_commands_of_earlier_versions_count(tmp_path, cmd):
    p = _env(tmp_path, "scmb_r", history=cmd)
    assert envs.made_by_multibench(p, "scmb_r")


@pytest.mark.parametrize("cmd", [
    "conda create -n scmb_r r-base",                                   # by hand
    "conda env create -n scmb_r -f /home/u/scmb_r.yml",                # another file
    "conda env create -n scmb_r -f /tmp/multibench_envbuild/glue/conda.yml",  # other env
    "conda env create -n scmb_r2 -f /tmp/multibench_envbuild/scmb_r/conda.yml",
    "conda create -y -n scmb_r -c conda-forge r-base",                 # a recipe build
])
def test_other_creation_commands_do_not_count(tmp_path, cmd):
    p = _env(tmp_path, "scmb_r", history=cmd)
    assert not envs.made_by_multibench(p, "scmb_r")


def test_only_the_first_command_in_the_history_decides(tmp_path):
    p = _env(tmp_path, "glue", history="conda create -n glue python=3.9")
    with open(p / "conda-meta" / "history", "a") as fh:
        fh.write("# cmd: conda env create -n glue -f /tmp/multibench_envbuild/glue/conda.yml\n")
    assert not envs.made_by_multibench(p, "glue")


def test_scan_reports_the_users_env_and_does_not_run_it(fake_root):
    row = full_scan("D11", "vertical", methods=["Matilda"], verbose=False).iloc[0]
    assert not row.env_ok and not row.runnable
    assert row.env_reason == (
        f"Environment matilda was not installed by this version of multibench, so it is "
        f"not used. Rename or remove {fake_root / 'envs' / 'matilda'}, then run "
        f"mtb.env.install(['Matilda']).")


def test_install_does_not_take_the_users_env_as_installed(fake_root, monkeypatch):
    row = mtb.env.install(["Matilda"], dry_run=True).to_dict("records")[0]
    assert row["state"] == "name taken" and row["exists"] is False
    monkeypatch.setattr(envs, "install_packed",
                        lambda *a, **k: pytest.fail("downloaded despite the env in the way"))
    with pytest.raises(RuntimeError) as e:
        mtb.env.install(["Matilda"], dry_run=False).to_dict("records")
    assert str(e.value) == (
        f"Environment matilda was not installed by this version of multibench, so it is "
        f"not used. Rename or remove {fake_root / 'envs' / 'matilda'}, then install again.")
    assert not (fake_root / "envs" / "matilda" / envs.FLAVOR_FILE).exists()


def test_install_packed_refuses_a_users_env_at_its_target(fake_root):
    with pytest.raises(RuntimeError, match="not installed by this version of multibench"):
        envs.install_packed("matilda", envs_dir=fake_root / "envs")


def test_run_refuses_the_users_env_on_linux(fake_root, tmp_path):
    inputs = mtb.engine.resolve.inputs_for("D11", "vertical", "Matilda")
    with pytest.raises(OSError) as e:
        mtb.engine.runner.run("Matilda", "vertical", inputs=inputs, out_dir=str(tmp_path / "out"))
    assert str(e.value) == (
        f"Environment matilda was not installed by this version of multibench, so it is "
        f"not used. Rename or remove {fake_root / 'envs' / 'matilda'}, then call "
        f"mtb.env.install(['Matilda']).")
    assert not (tmp_path / "out").exists()


def test_run_refuses_off_linux_also_when_the_env_is_installed(fake_root, tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem",
                        lambda: "method environments are linux-64 conda envs (packed "
                                "archives + lockfiles); this host is darwin/arm64")
    assert envs.env_prefix("env_sciPENN") is not None
    inputs = mtb.engine.resolve.inputs_for("D11", "vertical", "sciPENN")
    with pytest.raises(OSError, match=r"^Methods run only on Linux, and this computer "
                                      r"runs macOS\. Run this call on a Linux machine\."):
        mtb.engine.runner.run("sciPENN", "vertical", inputs=inputs, out_dir=str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_a_lockfile_build_marks_its_env_and_refuses_a_users_env(fake_root):
    conda = str(Path(config.envs_dir).parent.parent / "bin" / "conda")
    envs.create_env("glue", conda=conda, dry_run=False, force=True)
    assert (fake_root / "envs" / "glue" / envs.FLAVOR_FILE).read_text() == "build\n"
    assert envs.env_prefix("glue") == fake_root / "envs" / "glue"
    assert envs.installed_flavor("glue") is None          # not a packed install
    with pytest.raises(RuntimeError, match="matilda was not installed by this version of multibench"):
        envs.create_env("matilda", conda=conda, dry_run=False, force=True)


