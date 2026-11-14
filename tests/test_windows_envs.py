"""Windows: a method runs in an env built on the computer or unpacked from the
archive published for Windows.

An env there is a conda prefix (``python.exe`` at its root) or an R
installation (``bin/Rscript.exe``), marked with ``FLAVOR_FILE`` like every
env; ``METHODS_FILE`` lists the methods it was built for. The archives are
listed in ``engine/packed_windows.json``, apart from the Linux ones. Every
test pins the platform through ``envs.host_system``, so it holds on any host;
nothing here starts a method, and an archive is a small file served from a
folder.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from multibench import config, workflow
from multibench.engine import envs, registry, runner


@pytest.fixture
def windows(monkeypatch, tmp_path):
    """Windows, no conda, no archive published, and ``config.envs_dir`` at a
    new folder (returned)."""
    monkeypatch.setattr(envs, "host_system", lambda: "Windows")
    monkeypatch.setattr(envs, "_find_conda", lambda: None)
    monkeypatch.setattr(envs, "_WINDOWS_JSON", tmp_path / "no_archives.json")
    monkeypatch.delenv(runner.RUN_MODE_VAR, raising=False)
    d = tmp_path / "envs"
    d.mkdir()
    monkeypatch.setattr(config, "envs_dir", d)
    return d


def conda_prefix(envs_dir: Path, env: str, methods=None) -> Path:
    """A fake conda prefix as Windows lays it out, marked as built here."""
    prefix = envs_dir / env
    prefix.mkdir(parents=True)
    (prefix / "python.exe").write_bytes(b"")
    (prefix / envs.FLAVOR_FILE).write_text("build\n")
    if methods is not None:
        (prefix / envs.METHODS_FILE).write_text("\n".join(methods) + "\n")
    return prefix


def r_installation(envs_dir: Path, env: str, methods=None) -> Path:
    """A fake R installation (``bin/Rscript.exe``), marked as built here."""
    prefix = envs_dir / env
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "Rscript.exe").write_bytes(b"")
    (prefix / envs.FLAVOR_FILE).write_text("build\n")
    if methods is not None:
        (prefix / envs.METHODS_FILE).write_text("\n".join(methods) + "\n")
    return prefix


# ----------------------------------------------------------------- the platform
def test_windows_is_not_refused_as_a_platform_and_an_intel_mac_still_is(monkeypatch):
    """A Mac with Apple silicon has archives of its own (tests/test_macos_envs.py)."""
    monkeypatch.setattr(envs, "host_system", lambda: "Windows")
    assert envs.on_windows() and envs.linux_only_text() is None
    monkeypatch.setattr(envs, "host_system", lambda: "macOS")
    monkeypatch.setattr(envs, "_machine", lambda: "x86_64")
    assert not envs.on_windows()
    assert envs.linux_only_text() == "Methods run only on Linux, and this computer runs macOS."
    monkeypatch.setattr(envs, "host_system", lambda: None)
    assert not envs.on_windows() and envs.linux_only_text() is None


# ----------------------------------------------------------------- which folder is an env
def test_an_env_on_windows_is_a_folder_with_an_interpreter(windows):
    assert envs.installed_envs() == []
    conda_prefix(windows, "matilda")
    r_installation(windows, "scmb_r")
    (windows / "linux_env" / "bin").mkdir(parents=True)        # a Linux prefix: no interpreter here
    (windows / "linux_env" / envs.FLAVOR_FILE).write_text("gpu\n")
    assert envs.installed_envs() == ["matilda", "scmb_r"]
    assert envs.env_prefix("matilda") == windows / "matilda"
    assert envs.env_prefix("linux_env") is None
    assert envs.windows_interpreter(windows / "matilda", "python") == windows / "matilda" / "python.exe"
    assert envs.windows_interpreter(windows / "scmb_r", "Rscript") == windows / "scmb_r" / "bin" / "Rscript.exe"
    assert envs.windows_interpreter(windows / "matilda", "Rscript") is None


def test_an_env_made_by_hand_is_not_used_on_windows(windows):
    prefix = conda_prefix(windows, "matilda")
    (prefix / envs.FLAVOR_FILE).unlink()
    assert envs.env_prefix("matilda") is None and not envs.serves("matilda", "Matilda")


def test_on_linux_an_env_is_still_a_folder_with_bin(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "host_system", lambda: None)
    monkeypatch.setattr(envs, "_find_conda", lambda: None)
    monkeypatch.setattr(config, "envs_dir", tmp_path)
    conda_prefix(tmp_path, "matilda")                           # python.exe, no bin/
    assert envs.installed_envs() == []
    (tmp_path / "scmb_torch" / "bin").mkdir(parents=True)
    (tmp_path / "scmb_torch" / envs.FLAVOR_FILE).write_text("gpu\n")
    assert envs.installed_envs() == ["scmb_torch"]


# ----------------------------------------------------------------- which methods an env runs
def test_an_env_built_for_some_methods_runs_only_those(windows):
    assert envs.group_for("sciPENN") == envs.group_for("SMILE") == "env_sciPENN"
    assert not envs.serves("env_sciPENN", "sciPENN")            # not installed
    conda_prefix(windows, "env_sciPENN", methods=["sciPENN"])
    assert envs.serves("env_sciPENN", "sciPENN")
    assert not envs.serves("env_sciPENN", "SMILE")
    conda_prefix(windows, "matilda")                            # no list: every method of the env
    assert envs.serves("matilda", "Matilda")


def test_the_refusal_text():
    one = envs.no_windows_env_text(["Matilda"])
    assert one == "Matilda has no Windows environment. Run it on a Linux machine, such as Colab."
    two = envs.no_windows_env_text(["Matilda", "sciPENN"])
    assert two.startswith("Matilda and sciPENN have no Windows environment. Run them on ")
    many = envs.no_windows_env_text(["A", "B", "C", "D"])
    assert many.startswith("4 of these methods have no Windows environment.")


# ----------------------------------------------------------------- install
def test_install_refuses_a_method_without_a_windows_env_before_any_download(windows, monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("a download started on Windows"))
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build started on Windows"))
    with pytest.raises(OSError, match="^Matilda has no Windows environment. Run it on a Linux"):
        envs.install(["Matilda"])
    conda_prefix(windows, "env_sciPENN", methods=["sciPENN"])
    with pytest.raises(OSError, match="^SMILE has no Windows environment."):
        envs.install(["sciPENN", "SMILE"])


def test_install_finds_the_envs_built_here(windows, monkeypatch):
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build started on Windows"))
    conda_prefix(windows, "matilda", methods=["Matilda"])
    conda_prefix(windows, "env_sciPENN", methods=["sciPENN"])
    table = envs.install(["Matilda", "sciPENN"])
    assert sorted(table["env"]) == ["env_sciPENN", "matilda"]
    assert set(table["state"]) == {"have"}


def test_the_dry_run_names_no_download_on_windows(windows, capsys):
    conda_prefix(windows, "matilda", methods=["Matilda"])
    table = envs.install(["Matilda", "StabMap"], dry_run=True)
    state = dict(zip(table["env"], table["state"]))
    assert state == {"matilda": "have", "scmb_r": envs.NO_WINDOWS_ENV}
    assert "to download" not in capsys.readouterr().out


# ----------------------------------------------------------------- the archives for Windows
def publish(tmp_path, monkeypatch, key, methods, *, r=False, sha256=None, longest=None):
    """A small archive as ``tools/pack_windows.py`` writes one, in a folder that
    ``file:`` URLs serve, and its entry in a ``packed_windows.json`` of the test."""
    import hashlib
    import io
    import json
    import tarfile

    folder = tmp_path / "published"
    folder.mkdir(exist_ok=True)
    files = {"bin/Rscript.exe" if r else "python.exe": b"",
             envs.METHODS_FILE: ("\n".join(methods) + "\n").encode(),
             envs.FLAVOR_FILE: b"build\n", "Lib/site-packages/pkg/__init__.py": b"x = 1\n"}
    name = f"{key}.tar.xz"
    with tarfile.open(folder / name, "w:xz") as tar:
        for rel, data in files.items():
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    blob = (folder / name).read_bytes()
    listing = tmp_path / "packed_windows.json"
    data = (json.loads(listing.read_text()) if listing.exists()
            else {"base": folder.as_uri(), "archives": {}})
    data["archives"][key] = {
        "files": [name], "sha256": sha256 or hashlib.sha256(blob).hexdigest(),
        "archive_bytes": len(blob), "unpacked_bytes": sum(map(len, files.values())),
        "longest_path": longest or max(map(len, files)), "methods": list(methods)}
    listing.write_text(json.dumps(data))
    monkeypatch.setattr(envs, "_WINDOWS_JSON", listing)


def test_install_unpacks_the_archive_published_for_windows(windows, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build started on Windows"))
    publish(tmp_path, monkeypatch, "matilda-cpu", ["Matilda"])
    publish(tmp_path, monkeypatch, "matilda", ["Matilda"])
    publish(tmp_path, monkeypatch, "scmb_r", ["StabMap"], r=True)
    plan = envs.install(["Matilda", "StabMap"], dry_run=True)
    assert dict(zip(plan["env"], plan["state"])) == {
        "matilda": "packed archive published", "scmb_r": "packed archive published"}
    assert dict(zip(plan["env"], plan["flavor"])) == {"matilda": "cpu", "scmb_r": "any"}
    assert "2 envs: " in capsys.readouterr().out and not (windows / "matilda").exists()

    table = envs.install(["Matilda", "StabMap"])
    assert set(table["state"]) == {"PACKED"}
    assert envs.env_prefix("matilda") == windows / "matilda"
    assert (windows / "matilda" / "Lib" / "site-packages" / "pkg" / "__init__.py").is_file()
    assert (windows / "matilda" / envs.FLAVOR_FILE).read_text() == "cpu\n"
    assert envs.installed_flavor("matilda") == "cpu" and envs.installed_flavor("scmb_r") == "single"
    assert envs.serves("matilda", "Matilda") and envs.serves("scmb_r", "StabMap")
    assert not list(windows.glob(".*download*")) and not list(windows.glob("*.partial*"))
    assert set(envs.install(["Matilda", "StabMap"])["state"]) == {"have"}        # nothing again


def test_the_gpu_build_is_unpacked_on_a_host_with_a_gpu(windows, tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    publish(tmp_path, monkeypatch, "env_sciPENN-cpu", ["sciPENN"])
    publish(tmp_path, monkeypatch, "env_sciPENN", ["sciPENN", "SMILE"])
    assert envs.windows_archive_runs("SMILE") and envs.windows_archive_runs("sciPENN")
    assert not envs.windows_archive_runs("SMILE", "cpu")       # the CPU build was not made for it
    envs.install(["sciPENN", "SMILE"])
    assert (windows / "env_sciPENN" / envs.FLAVOR_FILE).read_text() == "gpu\n"
    assert envs.serves("env_sciPENN", "SMILE")


def test_a_method_the_archive_was_not_built_for_is_refused_before_any_download(windows, tmp_path,
                                                                                 monkeypatch):
    import urllib.request
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    publish(tmp_path, monkeypatch, "env_sciPENN-cpu", ["sciPENN"])
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("a download started"))
    with pytest.raises(OSError, match="^SMILE has no Windows environment."):
        envs.install(["sciPENN", "SMILE"])
    plan = envs.install(["SMILE"], dry_run=True)
    assert list(plan["state"]) == [envs.NO_WINDOWS_ENV]
    with pytest.raises(OSError, match="not built on Windows. Leave out build=True"):
        envs.install(["sciPENN"], build=True)


def test_a_damaged_download_is_refused_and_leaves_no_folder(windows, tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    publish(tmp_path, monkeypatch, "matilda-cpu", ["Matilda"], sha256="0" * 64)
    with pytest.raises(RuntimeError, match="The environment matilda could not be unpacked .*damaged"):
        envs.install(["Matilda"])
    assert list(windows.iterdir()) == []


def test_an_envs_folder_with_too_long_a_path_is_refused_before_the_download(windows, tmp_path,
                                                                             monkeypatch):
    import urllib.request
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    publish(tmp_path, monkeypatch, "matilda-cpu", ["Matilda"], longest=250)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("a download started"))
    with pytest.raises(OSError, match="has too long a path for the environment matilda"):
        envs.install(["Matilda"])


def test_an_env_that_is_not_installed_names_the_install_when_an_archive_is_published(
        windows, tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    with pytest.raises(EnvironmentError, match="^Matilda has no Windows environment."):
        runner._check_env("Matilda", "matilda")
    publish(tmp_path, monkeypatch, "matilda-cpu", ["Matilda"])
    with pytest.raises(EnvironmentError, match=r"is not installed. Run mtb.env.install\(\['Matilda'\]\)"):
        runner._check_env("Matilda", "matilda")
    assert workflow._env_hint("matilda", "Matilda").startswith("Environment matilda is not installed.")
    conda_prefix(windows, "env_sciPENN", methods=["sciPENN"])       # there, and not built for SMILE
    assert workflow._env_hint("env_sciPENN", "SMILE") == envs.no_windows_env_text(["SMILE"])


def test_the_linux_archives_and_the_windows_archives_are_listed_apart(windows, tmp_path, monkeypatch):
    publish(tmp_path, monkeypatch, "matilda-cpu", ["Matilda"])
    assert list(envs._host_manifest()) == ["matilda-cpu"] == list(envs._host_sizes())
    assert envs._host_manifest()["matilda-cpu"][0].startswith("file:")
    monkeypatch.setattr(envs, "host_system", lambda: None)          # Linux: its own files, as before
    assert envs._host_manifest() == envs.packed_manifest()
    assert envs._host_sizes() == envs.packed_sizes()
    assert "matilda-cpu" in envs.packed_manifest()


# ----------------------------------------------------------------- the command
def test_the_command_names_the_env_interpreter_and_has_no_shell_wrapper(windows, tmp_path):
    prefix = conda_prefix(windows, "matilda", methods=["Matilda"])
    assert runner.run_mode("matilda") == ("prefix", prefix)
    inputs = {"rna": "rna.h5", "adt": "adt.h5", "cty": "cty.csv"}
    argv = runner.preview("Matilda", "vertical", inputs=inputs, out_dir=tmp_path / "o")[0]
    assert argv[0] == str(prefix / "python.exe")
    assert "bash" not in argv and argv[1].endswith("run_matilda.py")
    r = r_installation(windows, "scmb_r", methods=["StabMap"])
    assert runner.windows_prefix_argv(["Rscript", "x.R", "--a"], r) == [
        str(r / "bin" / "Rscript.exe"), "x.R", "--a"]


def test_the_activation_puts_the_env_first_on_path(windows, monkeypatch):
    prefix = conda_prefix(windows, "matilda")
    monkeypatch.setenv("PATH", "before")
    act = runner.windows_activation("matilda", prefix)
    assert act["CONDA_PREFIX"] == str(prefix) and act["CONDA_DEFAULT_ENV"] == "matilda"
    parts = act["PATH"].split(os.pathsep)
    assert parts[0] == str(prefix) and parts[-1] == "before"
    assert str(prefix / "Library" / "bin") in parts and str(prefix / "Scripts") in parts
    # an env with a Python names it for reticulate; an R installation alone names none
    assert act["RETICULATE_PYTHON"] == str(prefix / "python.exe")
    assert "RETICULATE_PYTHON" not in runner.windows_activation("scmb_r", r_installation(windows, "scmb_r"))
    # an env that holds an R under its Python names it for rpy2
    assert "R_HOME" not in act
    (prefix / "Lib" / "R" / "bin").mkdir(parents=True)
    assert runner.windows_activation("matilda", prefix)["R_HOME"] == str(prefix / "Lib" / "R")


def test_on_linux_the_command_is_still_the_bash_wrapper(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "host_system", lambda: None)
    monkeypatch.setattr(envs, "_find_conda", lambda: None)
    monkeypatch.delenv(runner.RUN_MODE_VAR, raising=False)
    monkeypatch.setattr(config, "envs_dir", tmp_path)
    (tmp_path / "matilda" / "bin").mkdir(parents=True)
    (tmp_path / "matilda" / envs.FLAVOR_FILE).write_text("cpu\n")
    inputs = {"rna": "rna.h5", "adt": "adt.h5", "cty": "cty.csv"}
    argv = runner.preview("Matilda", "vertical", inputs=inputs, out_dir=tmp_path / "o")[0]
    assert argv[:2] == ["bash", "-c"] and argv[4] == "python"


# ----------------------------------------------------------------- the checks before a run
def test_run_refuses_a_method_without_a_windows_env(windows):
    with pytest.raises(EnvironmentError, match="^Matilda has no Windows environment."):
        runner._check_env("Matilda", "matilda")
    conda_prefix(windows, "env_sciPENN", methods=["sciPENN"])
    runner._check_env("sciPENN", "env_sciPENN")
    with pytest.raises(EnvironmentError, match="^SMILE has no Windows environment."):
        runner._check_env("SMILE", "env_sciPENN")


def test_scan_blocks_a_method_its_env_was_not_built_for(windows):
    conda_prefix(windows, "env_sciPENN", methods=["sciPENN"])

    def env_check(method):
        rec = {"env": envs.group_for(method), "env_ok": True, "caveat": ""}
        ctx = workflow._ScanContext(None, None, None, frozenset(envs.installed_envs()), None, {},
                                    "", False, {}, None, {}, False)
        hint, _gpu = workflow._check_env(rec, registry.get(method), ctx)
        return rec["env_ok"], hint

    assert env_check("sciPENN") == (True, "")
    assert env_check("SMILE") == (False, envs.no_windows_env_text(["SMILE"]))
    assert env_check("Matilda") == (False, envs.no_windows_env_text(["Matilda"]))


def test_the_cpu_build_of_an_env_gets_the_cpu_parameters_on_a_gpu_host(windows, monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    spec = registry.get("scMDC")
    assert spec.cpu_params == {"device": "cpu"}
    prefix = conda_prefix(windows, envs.group_for("scMDC"), methods=["scMDC"])
    (prefix / envs.FLAVOR_FILE).write_text("gpu\n")
    assert runner.cpu_params_for(spec, None) == (None, {})              # the GPU build: the script's default
    (prefix / envs.FLAVOR_FILE).write_text("cpu\n")
    assert runner.cpu_params_for(spec, None) == ({"device": "cpu"}, {"device": "cpu"})
    assert runner.cpu_params_for(spec, {"device": "cuda"}) == ({"device": "cuda"}, {})   # the caller's value stays
    monkeypatch.setattr(envs, "host_system", lambda: None)              # Linux: the host alone decides, as before
    assert runner.cpu_params_for(spec, None, gpu=True) == (None, {})


def test_an_input_link_is_a_hard_link_or_a_copy_on_windows(windows, tmp_path, monkeypatch):
    src = tmp_path / "rna.h5"
    src.write_bytes(b"data")
    dest = tmp_path / "inputs" / "rna.h5"
    dest.parent.mkdir()
    monkeypatch.setattr(os, "symlink", lambda *a, **k: pytest.fail("a symbolic link was made on Windows"))
    runner._link_input(src, dest)
    assert dest.read_bytes() == b"data" and not dest.is_symlink()
    runner._link_input(src, dest)                           # a second run replaces it
    assert dest.read_bytes() == b"data"

    def no_link(*a, **k):
        raise OSError("another drive")
    monkeypatch.setattr(os, "link", no_link)
    runner._link_input(src, dest)                           # no hard link possible: a copy
    assert dest.read_bytes() == b"data" and src.exists()


def test_old_output_files_are_removed_before_a_run_on_windows(tmp_path):
    out = tmp_path / "StabMap_D52"
    (out / "inputs").mkdir(parents=True)
    for name in ("embedding.h5", "dist.h5", "time.csv", "inputs/rna.h5"):
        (out / name).write_bytes(b"old")
    runner._remove_old_outputs_windows(out)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "inputs/rna.h5", "time.csv"]
    runner._remove_old_outputs_windows(tmp_path / "absent")         # no folder yet: nothing to do


def test_scan_says_once_that_methods_have_no_windows_environment(windows, capsys):
    reasons = [(m, envs.no_windows_env_text([m]), "") for m in ("B", "A", "D", "C")]
    reasons.append(("E", "E needs adt.h5.", ""))
    workflow._print_reasons(reasons, "")
    assert capsys.readouterr().out.splitlines() == [
        "[scan] 4 of these methods have no Windows environment. Run them on a Linux machine, "
        "such as Colab.",
        "[scan] E: E needs adt.h5."]
    workflow._print_reasons(reasons[:1], "")                # one method: its own line
    assert capsys.readouterr().out.splitlines() == [f"[scan] B: {envs.no_windows_env_text(['B'])}"]


@pytest.mark.skipif(sys.platform != "win32",
                    reason="the timer's interrupt wakes the main thread's wait on Windows only")
def test_a_deadline_without_the_alarm_signal_raises_the_same_timeout():
    # On a POSIX system the deadline is the alarm signal and this path is never taken; forced there
    # (by removing signal.SIGALRM), the interrupt raised in the timer thread does not end the main
    # thread's time.sleep, and the TimeoutError comes when the sleep is over (seen on Linux and macOS).
    import time
    start = time.time()
    with pytest.raises(TimeoutError, match="exceeded timeout of 0.3s"):
        with workflow._deadline(0.3):
            time.sleep(20)
    assert time.time() - start < 10
    with workflow._deadline(30):                # a step that ends in time: no error, the timer is gone
        pass
    time.sleep(0.4)


def test_an_interrupt_the_deadline_did_not_send_passes_through():
    with pytest.raises(KeyboardInterrupt):
        with workflow._deadline_without_alarm(30):
            raise KeyboardInterrupt


def test_scan_reasons_show_the_last_part_of_a_windows_path(windows, monkeypatch):
    text = r"scBridge needs atac_gas.h5, and C:\Users\me\data sets\x\LUNG_ids has no such file."
    assert workflow._short_reason(r"scBridge needs atac_gas.h5, and C:\Users\me\data\LUNG_ids has no such file.",
                                  "scBridge", "LUNG_ids") == (
        "scBridge needs atac_gas.h5, and LUNG_ids has no such file.")
    assert workflow._short_reason("rna.h5 of D:/data/D11/rna.h5 is not canonical", "M", "D11") == (
        "rna.h5 of rna.h5 is not canonical")
    monkeypatch.setattr(envs, "host_system", lambda: None)      # Linux: a Windows path is left as it is
    assert "C:\\Users" in workflow._short_reason(text, "scBridge", "LUNG_ids")


# ----------------------------------------------------------------- the subprocess (Windows hosts)
@pytest.mark.skipif(sys.platform != "win32", reason="starts a Windows process")
def test_execute_passes_the_activation_and_reports_a_failure(windows, tmp_path):
    out = tmp_path / "seen.txt"
    code = ("import os, sys; open(sys.argv[1], 'w').write(os.environ['CONDA_DEFAULT_ENV'] + '|' "
            "+ os.environ['PYTHONNOUSERSITE'] + '|' + os.environ['MPLBACKEND'])")
    runner._execute("M", [sys.executable, "-c", code, str(out)], str(tmp_path), None,
                    activation={"CONDA_DEFAULT_ENV": "matilda"})
    assert out.read_text() == "matilda|1|Agg"
    with pytest.raises(RuntimeError, match=r"M failed \(exit 3\)"):
        runner._execute("M", [sys.executable, "-c", "import sys; sys.exit(3)"], str(tmp_path), None)


@pytest.mark.skipif(sys.platform != "win32", reason="starts a Windows process")
def test_output_outside_the_code_page_keeps_the_methods_error(windows, tmp_path):
    # 0x81 is undefined in cp1252 and a lone lead byte in cp936 and UTF-8
    code = ("import sys; sys.stdout.buffer.write(bytes([0x62, 0x81])); "
            "sys.stderr.write('no package called x'); sys.exit(3)")
    with pytest.raises(RuntimeError, match=r"M failed \(exit 3\)") as e:
        runner._execute("M", [sys.executable, "-c", code], str(tmp_path), None)
    assert "no package called x" in str(e.value)


@pytest.mark.skipif(sys.platform != "win32", reason="ends a Windows process tree")
def test_the_deadline_ends_a_running_method_on_windows(windows, tmp_path):
    import subprocess
    import time

    pid_file = tmp_path / "method.pid"
    code = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(120)"
    start = time.time()
    with pytest.raises(TimeoutError):
        with workflow._deadline_without_alarm(3):
            runner._execute("M", [sys.executable, "-c", code, str(pid_file)], str(tmp_path), None)
    assert time.time() - start < 30
    pid = pid_file.read_text()
    listed = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    assert pid not in listed


@pytest.mark.skipif(sys.platform != "win32", reason="ends a Windows process tree")
def test_an_interrupted_run_ends_the_method_and_its_children(windows, tmp_path, monkeypatch):
    import subprocess
    import time

    pid_file = tmp_path / "child.pid"
    child = "import time; time.sleep(120)"
    parent = ("import subprocess, sys, time; "
              f"p = subprocess.Popen([sys.executable, '-c', {child!r}]); "
              "open(sys.argv[1], 'w').write(str(p.pid)); time.sleep(120)")
    real = subprocess.Popen.communicate
    calls = []

    def interrupted(self, *a, **k):
        # the method's own communicate() is interrupted; taskkill's is left alone
        calls.append(self)
        if len(calls) > 1:
            return real(self, *a, **k)
        for _ in range(100):                    # until the child's pid is written
            if pid_file.exists() and pid_file.read_text():
                break
            time.sleep(0.1)
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner._execute("M", [sys.executable, "-c", parent, str(pid_file)], str(tmp_path), None)
    monkeypatch.setattr(subprocess.Popen, "communicate", real)
    child_pid = pid_file.read_text()
    listed = subprocess.run(["tasklist", "/FI", f"PID eq {child_pid}", "/NH"],
                            capture_output=True, text=True).stdout
    assert child_pid not in listed
