"""macOS on Apple silicon (pilot): a method runs in an env unpacked from an
archive packed on a Mac, when a folder of such archives is named.

An env there is a conda prefix as on Linux (``bin/``), run through the bash
wrapper; ``METHODS_FILE`` lists the methods it was built for. The archives and
their ``packed_macos.json`` sit in the folder ``MULTIBENCH_MACOS_ARCHIVES``
names; without it a Mac is refused as before. Every test pins the platform
through ``envs.host_system`` and ``envs._machine``, so it holds on any host;
nothing here starts a method.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from multibench import config, workflow
from multibench.engine import envs, runner


@pytest.fixture
def mac(monkeypatch, tmp_path):
    """A Mac with Apple silicon, no conda, an archive folder with no archive in
    it yet (``publish`` fills it), and ``config.envs_dir`` at a new folder
    (returned)."""
    monkeypatch.setattr(envs, "host_system", lambda: "macOS")
    monkeypatch.setattr(envs, "_machine", lambda: "arm64")
    monkeypatch.setattr(envs, "_find_conda", lambda: None)
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    monkeypatch.setenv(envs.MACOS_ARCHIVES_VAR, str(tmp_path / "archives"))
    monkeypatch.delenv(runner.RUN_MODE_VAR, raising=False)
    d = tmp_path / "envs"
    d.mkdir()
    monkeypatch.setattr(config, "envs_dir", d)
    return d


def publish(tmp_path, key, methods, *, sha256=None, unpack_script=None):
    """A small archive as conda-pack writes one (``bin/``), in the archive
    folder, and its entry in that folder's ``packed_macos.json``."""
    folder = tmp_path / "archives"
    folder.mkdir(exist_ok=True)
    files = {"bin/Rscript": b"", envs.METHODS_FILE: ("\n".join(methods) + "\n").encode(),
             "lib/pkg/__init__.py": b"x = 1\n"}
    if unpack_script is not None:
        files["bin/conda-unpack"] = unpack_script.encode()
    name = f"{key}.tar.gz"
    with tarfile.open(folder / name, "w:gz") as tar:
        for rel, data in files.items():
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    blob = (folder / name).read_bytes()
    listing = folder / "packed_macos.json"
    data = json.loads(listing.read_text()) if listing.exists() else {"archives": {}}
    data["archives"][key] = {
        "files": [name], "sha256": sha256 or hashlib.sha256(blob).hexdigest(),
        "archive_bytes": len(blob), "unpacked_bytes": sum(map(len, files.values())),
        "methods": list(methods)}
    listing.write_text(json.dumps(data))


# ----------------------------------------------------------------- the gate
def test_a_mac_without_an_archive_folder_is_refused_as_before(monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "host_system", lambda: "macOS")
    monkeypatch.setattr(envs, "_machine", lambda: "arm64")
    monkeypatch.delenv(envs.MACOS_ARCHIVES_VAR, raising=False)
    assert not envs.on_macos() and not envs.archive_only()
    assert envs.linux_only_text() == "Methods run only on Linux, and this computer runs macOS."
    with pytest.raises(OSError, match="^Methods run only on Linux"):
        envs.install(["Matilda"])
    with pytest.raises(OSError, match="^Methods run only on Linux"):
        runner._check_env("Matilda", "matilda")
    # a folder that lists nothing changes nothing
    monkeypatch.setenv(envs.MACOS_ARCHIVES_VAR, str(tmp_path))
    assert not envs.on_macos() and envs.linux_only_text()


def test_a_mac_with_another_processor_is_refused_as_before(mac, tmp_path, monkeypatch):
    publish(tmp_path, "matilda", ["Matilda"])
    assert envs.on_macos() and envs.linux_only_text() is None
    monkeypatch.setattr(envs, "_machine", lambda: "x86_64")      # an Intel Mac, or Rosetta
    assert not envs.on_macos()
    assert envs.linux_only_text() == "Methods run only on Linux, and this computer runs macOS."


def test_the_folder_changes_nothing_on_linux_or_windows(mac, tmp_path, monkeypatch):
    publish(tmp_path, "matilda", ["Matilda"])
    for system in (None, "Windows"):
        monkeypatch.setattr(envs, "host_system", lambda system=system: system)
        assert not envs.on_macos()
        assert envs.host_archives() == envs.windows_archives()
    monkeypatch.setattr(envs, "host_system", lambda: None)
    assert not envs.archive_only() and envs._host_manifest() == envs.packed_manifest()
    assert envs._host_sizes() == envs.packed_sizes()


# ----------------------------------------------------------------- the install
def test_install_unpacks_the_archive_from_the_folder(mac, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build started on macOS"))
    publish(tmp_path, "matilda", ["Matilda"])
    publish(tmp_path, "scmb_r", ["StabMap"])
    plan = envs.install(["Matilda", "StabMap"], dry_run=True)
    assert dict(zip(plan["env"], plan["state"])) == {
        "matilda": "packed archive published", "scmb_r": "packed archive published"}
    assert "2 envs: " in capsys.readouterr().out and not (mac / "matilda").exists()

    table = envs.install(["Matilda", "StabMap"])
    assert set(table["state"]) == {"PACKED"}
    assert envs.env_prefix("scmb_r") == mac / "scmb_r"
    assert (mac / "matilda" / "lib" / "pkg" / "__init__.py").is_file()
    assert envs.installed_flavor("matilda") == "single"
    assert envs.serves("matilda", "Matilda") and envs.serves("scmb_r", "StabMap")
    assert not envs.serves("scmb_r", "UINMF")
    assert not list(mac.glob(".*download*")) and not list(mac.glob("*.partial*"))
    assert set(envs.install(["Matilda", "StabMap"])["state"]) == {"have"}        # nothing again


def test_a_method_without_a_macos_archive_is_refused_before_any_download(mac, tmp_path,
                                                                          monkeypatch):
    import urllib.request
    publish(tmp_path, "scmb_r", ["StabMap"])
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("a download started"))
    with pytest.raises(OSError, match="^UINMF has no macOS environment. Run it on a Linux"):
        envs.install(["StabMap", "UINMF"])
    with pytest.raises(OSError, match="^totalVI has no macOS environment."):
        envs.install(["totalVI"])
    assert list(envs.install(["totalVI"], dry_run=True)["state"]) == [envs.NO_MACOS_ENV]
    with pytest.raises(OSError, match="not built on macOS. Leave out build=True"):
        envs.install(["StabMap"], build=True)


def test_a_damaged_archive_is_refused_and_leaves_no_folder(mac, tmp_path):
    publish(tmp_path, "matilda", ["Matilda"], sha256="0" * 64)
    with pytest.raises(RuntimeError, match="The environment matilda could not be unpacked .*damaged"):
        envs.install(["Matilda"])
    assert list(mac.iterdir()) == []


def test_conda_unpack_of_an_env_without_python_runs_with_this_python(mac, tmp_path):
    # as conda-pack writes it: `#!/usr/bin/env python`, and an R env has no bin/python
    script = ("#!/usr/bin/env python\nimport os, sys\n"
              "open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ran.txt'), 'w')"
              ".write(sys.executable)\n")
    publish(tmp_path, "scmb_r", ["StabMap"], unpack_script=script)
    envs.install(["StabMap"])
    assert (mac / "scmb_r" / "bin" / "ran.txt").read_text() == sys.executable


# ----------------------------------------------------------------- the run
def test_run_refuses_a_method_its_env_was_not_built_for(mac, tmp_path):
    publish(tmp_path, "matilda", ["Matilda"])
    with pytest.raises(EnvironmentError, match="^StabMap has no macOS environment."):
        runner._check_env("StabMap", "scmb_r")              # no archive listed for it yet
    publish(tmp_path, "scmb_r", ["StabMap"])
    with pytest.raises(EnvironmentError, match=r"is not installed. Run mtb.env.install\(\['StabMap'\]\)"):
        runner._check_env("StabMap", "scmb_r")
    envs.install(["StabMap"])
    runner._check_env("StabMap", "scmb_r")
    with pytest.raises(EnvironmentError, match="^UINMF has no macOS environment."):
        runner._check_env("UINMF", "scmb_r")
    assert workflow._env_hint("scmb_r", "UINMF") == (
        "UINMF has no macOS environment. Run it on a Linux machine, such as Colab.")


def test_the_command_on_macos_is_the_bash_wrapper(mac, tmp_path):
    publish(tmp_path, "matilda", ["Matilda"])
    envs.install(["Matilda"])
    inputs = {"rna": "rna.h5", "adt": "adt.h5", "cty": "cty.csv"}
    argv = runner.preview("Matilda", "vertical", inputs=inputs, out_dir=tmp_path / "o")[0]
    assert argv[:2] == ["bash", "-c"] and argv[4] == "python"
    assert str(mac / "matilda" / "bin") in argv[2]
    # a Python script runs through the fork driver, then as on Linux
    assert argv[5] == str(runner.MACOS_FORK_DRIVER) and argv[6].endswith(".py")
    assert runner.MACOS_FORK_DRIVER.is_file()


def test_an_r_method_on_macos_has_no_fork_driver(mac, tmp_path):
    publish(tmp_path, "scmb_seurat4b", ["Seurat_v3"])
    envs.install(["Seurat_v3"])
    inputs = {"rna": "rna.h5", "atac_peak": "peak.h5", "atac_gas": "gas.h5",
              "rna_cty": "c1.csv", "atac_cty": "c2.csv"}
    argv = runner.preview("Seurat_v3", "diagonal", inputs=inputs, out_dir=tmp_path / "o")[0]
    assert argv[:2] == ["bash", "-c"] and str(runner.MACOS_FORK_DRIVER) not in argv


def test_the_fork_driver_runs_a_script_as_python_would(tmp_path):
    import subprocess
    script = tmp_path / "s.py"
    (tmp_path / "beside.py").write_text("X = 7\n")
    script.write_text("import multiprocessing, sys, beside\n"
                      "print(multiprocessing.get_start_method(), sys.argv[1:], beside.X, __name__)\n")
    out = subprocess.run([sys.executable, str(runner.MACOS_FORK_DRIVER), str(script), "--a", "1"],
                         capture_output=True, text=True, cwd=tmp_path.parent)
    if sys.platform == "win32":                 # no fork there; the driver is used on macOS only
        assert out.returncode != 0
    else:
        assert out.stdout.split("\n")[0] == "fork ['--a', '1'] 7 __main__", out.stderr


def test_a_method_that_needs_a_terminal_gets_the_bsd_form_of_script(mac, tmp_path, monkeypatch):
    # scJoint reads the terminal size. Linux: `script -q -e -c "<command>" /dev/null`;
    # the `script` of macOS takes the file first and the command as an argv
    publish(tmp_path, "scmb_scjoint", ["scJoint"])
    envs.install(["scJoint"])
    inputs = {"rna": "rna.h5", "atac_gas": "atac.h5", "rna_cty": "c1.csv", "atac_cty": "c2.csv"}
    argv = runner.preview("scJoint", "diagonal", inputs=inputs, out_dir=tmp_path / "o")[0]
    assert argv[:2] == ["bash", "-c"]
    assert argv[4:9] == ["script", "-q", "-e", "/dev/null", "python"] and "-c" not in argv[4:]
    assert argv[9] == str(runner.MACOS_FORK_DRIVER)
    monkeypatch.setattr(envs, "host_system", lambda: None)           # Linux: as before
    monkeypatch.setattr(config, "envs_dir", tmp_path / "none")
    monkeypatch.setenv(runner.RUN_MODE_VAR, "conda")
    linux = runner.preview("scJoint", "diagonal", inputs=inputs, out_dir=tmp_path / "o")[0]
    assert linux[linux.index("script"):][:4] == ["script", "-q", "-e", "-c"] and linux[-1] == "/dev/null"
