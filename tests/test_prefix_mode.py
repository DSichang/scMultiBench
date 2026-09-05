"""Conda-free execution of packed envs (contract section E).

Nothing here needs a real env or a conda binary: a *fake prefix* under
``tmp_path`` (an executable ``bin/python`` shell stub plus an
``etc/conda/activate.d/x.sh`` exporting a marker) stands in for an unpacked
conda-pack archive, and ``shutil.which`` is pinned to ``None`` so the host's
conda - if any - is invisible. The cases pin:

* ``config.Config.envs_dir`` resolution order and settability;
* ``env_prefix`` / ``installed_envs`` seeing the fake prefix without conda;
* the runner's ``prefix`` mode: the argv is the ``bash -c`` wrapper, and
  RUNNING it executes the stub with the activate.d marker set;
* ``MULTIBENCH_RUN_MODE`` forcing and its errors;
* ``install_packed`` on a tiny tar.gz with conda absent;
* ``env.install``'s no-conda errors, raised before any download;
* the ``scan`` env gate flipping on the fake prefix.
"""
from __future__ import annotations

import io
import os
import shlex
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

import multibench as mtb
from multibench import config
from multibench import workflow as W
from multibench.engine import envs, runner

MARKER = "MTB_ACTIVATED_MARKER"


# ---------------------------------------------------------------- fixtures
def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def make_prefix(envs_dir: Path, env: str) -> Path:
    """A fake unpacked env: ``bin/python`` stub + one activate.d script."""
    prefix = envs_dir / env
    _executable(prefix / "bin" / "python",
                "#!/bin/sh\n"
                f'echo "stub-python marker=${MARKER} prefix=$CONDA_PREFIX '
                'env=$CONDA_DEFAULT_ENV args=$*"\n')
    (prefix / "etc" / "conda" / "activate.d").mkdir(parents=True)
    (prefix / "etc" / "conda" / "activate.d" / "x.sh").write_text(
        '[ -n "$CONDA_PREFIX" ] || exit 99\n'          # they assume it is set
        f'export {MARKER}="set-from-$CONDA_DEFAULT_ENV"\n')
    return prefix


def _clear_caches():
    """Drop the per-process conda probes (a test may have replaced one)."""
    for fn in (config._conda_envs_dir, envs._conda_prefixes):
        clear = getattr(fn, "cache_clear", None)
        if clear:
            clear()


@pytest.fixture
def no_conda(monkeypatch):
    """Hide any conda/mamba on this host from every discovery path."""
    monkeypatch.setattr(envs.shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(config._shutil, "which", lambda *a, **k: None)
    _clear_caches()
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.delenv(runner.RUN_MODE_VAR, raising=False)
    yield
    _clear_caches()


@pytest.fixture
def envs_dir(tmp_path, monkeypatch, no_conda):
    """``config.DEFAULT.envs_dir`` pointed at a fresh directory."""
    d = tmp_path / "envs"
    d.mkdir()
    monkeypatch.setattr(config.DEFAULT, "envs_dir", d)
    monkeypatch.delenv(config.ENVS_DIR_VAR, raising=False)
    yield d


SCALEX_INPUTS = {"rna": "a.h5", "atac_gas": "b.h5"}
SCALEX_ENV = envs.group_for("SCALEX")


# ---------------------------------------------------------------- Config.envs_dir
def test_envs_dir_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENVS_DIR_VAR, str(tmp_path / "mine"))
    assert config.Config().envs_dir == tmp_path / "mine"


def test_envs_dir_uses_conda_envs_dir_when_conda_found(monkeypatch, tmp_path):
    monkeypatch.delenv(config.ENVS_DIR_VAR, raising=False)
    monkeypatch.setattr(config._shutil, "which", lambda name: "/fake/bin/" + name)
    monkeypatch.setattr(envs, "_envs_dir", lambda exe: tmp_path / "conda-envs")
    config._conda_envs_dir.cache_clear()
    try:
        assert config.Config().envs_dir == tmp_path / "conda-envs"
    finally:
        config._conda_envs_dir.cache_clear()


def test_envs_dir_falls_back_to_the_user_cache(no_conda, monkeypatch):
    monkeypatch.delenv(config.ENVS_DIR_VAR, raising=False)
    d = config.Config().envs_dir
    assert d == config._CACHE / "envs"
    assert d.parts[-2:] == ("multibench", "envs")


def test_envs_dir_is_a_settable_field_and_lazy(monkeypatch, tmp_path):
    import dataclasses
    assert "envs_dir" in [f.name for f in dataclasses.fields(config.Config)]
    cfg = config.Config()
    assert cfg.__dict__["_envs_dir"] is None, "resolved on first read, not at construction"
    cfg.envs_dir = str(tmp_path / "x")
    assert cfg.envs_dir == tmp_path / "x" and isinstance(cfg.envs_dir, Path)
    assert config.Config(envs_dir=tmp_path / "y").envs_dir == tmp_path / "y"
    monkeypatch.setattr(config.DEFAULT, "envs_dir", tmp_path / "z")
    assert config.DEFAULT.envs_dir == tmp_path / "z"
    assert "envs_dir" in repr(config.DEFAULT)


def test_import_does_not_probe_conda():
    """``config.DEFAULT`` is built at import; the probe must not run then."""
    code = ("import multibench.config as c, multibench; "
            "print(c.DEFAULT.__dict__['_envs_dir'] is None, c._conda_envs_dir.cache_info().currsize)")
    out = subprocess.run([os.sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(Path(config.__file__).parents[1])},
                         check=True).stdout.split()
    assert out == ["True", "0"]


# ---------------------------------------------------------------- env_prefix / installed_envs
def test_env_prefix_and_installed_envs_without_conda(envs_dir):
    assert envs.env_prefix("matilda") is None
    assert envs.installed_envs() == []
    prefix = make_prefix(envs_dir, "matilda")
    (envs_dir / "not-an-env").mkdir()                   # no bin/ -> not an env
    (envs_dir / "stray.txt").write_text("")
    assert envs.env_prefix("matilda") == prefix
    assert envs.installed_envs() == ["matilda"]
    assert envs.env_prefix("scmb_torch") is None


def test_env_prefix_asks_conda_when_not_under_envs_dir(envs_dir, monkeypatch):
    monkeypatch.setattr(envs, "_find_conda", lambda: "/fake/conda")
    monkeypatch.setattr(envs, "_conda_prefixes",
                        lambda conda: ("/opt/conda/envs/scmb_r", "/opt/conda"))
    assert envs.env_prefix("scmb_r") == Path("/opt/conda/envs/scmb_r")
    assert envs.env_prefix("conda") == Path("/opt/conda")
    assert envs.env_prefix("matilda") is None
    assert envs.installed_envs() == ["scmb_r", "conda"]
    make_prefix(envs_dir, "matilda")                     # on disk wins, no conda call
    assert envs.env_prefix("matilda") == envs_dir / "matilda"
    assert envs.installed_envs() == ["matilda", "scmb_r", "conda"]


def test_doctor_and_status_count_the_prefix(envs_dir):
    make_prefix(envs_dir, SCALEX_ENV)
    row = next(r for r in envs.doctor(methods=["SCALEX"]) if r["env"] == SCALEX_ENV)
    assert row["exists"] is True
    st = next(r for r in envs.status() if r["method"] == "SCALEX")
    assert st["exists"] is True


# ---------------------------------------------------------------- runner: prefix mode
def test_runner_argv_is_the_bash_wrapper_and_it_runs(envs_dir, tmp_path):
    prefix = make_prefix(envs_dir, SCALEX_ENV)
    argv = mtb.run("SCALEX", "diagonal", inputs=SCALEX_INPUTS, out_dir=tmp_path / "o",
                   dry_run=True)
    assert argv[:2] == ["bash", "-c"]
    assert argv[3] == "--" and argv[4] == "python"
    act = argv[2]
    assert act.endswith('; exec "$@"')
    assert f"export CONDA_PREFIX={shlex.quote(str(prefix))}" in act
    assert f"export CONDA_DEFAULT_ENV={SCALEX_ENV}" in act
    assert f"export PATH={shlex.quote(str(prefix / 'bin'))}:\"$PATH\"" in act
    assert "etc/conda/activate.d" in act and "conda run" not in shlex.join(argv)
    # running the wrapper as-is executes the prefix's python stub with the
    # activate.d marker exported and CONDA_PREFIX visible to the script
    proc = subprocess.run(argv, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert f"marker=set-from-{SCALEX_ENV}" in proc.stdout
    assert f"prefix={prefix}" in proc.stdout and f"env={SCALEX_ENV}" in proc.stdout
    assert "main_SCALEX.py" in proc.stdout and "--path1" in proc.stdout


def test_prefix_mode_no_activate_d_is_fine(envs_dir):
    """An env without activate.d (most python envs) must not fail the glob."""
    prefix = envs_dir / "scmb_torch"
    _executable(prefix / "bin" / "python", '#!/bin/sh\necho "plain marker=${MTB_ACTIVATED_MARKER:-none} $*"\n')
    argv = runner.wrap_prefix(["python", "-c", "x"], "scmb_torch", prefix)
    proc = subprocess.run(argv, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "plain marker=none -c x"


def test_run_mode_defaults(envs_dir):
    assert runner.run_mode("matilda") == ("conda", None)
    prefix = make_prefix(envs_dir, "matilda")
    assert runner.run_mode("matilda") == ("prefix", prefix)


def test_conda_mode_when_no_prefix(envs_dir, tmp_path):
    argv = mtb.run("SCALEX", "diagonal", inputs=SCALEX_INPUTS, out_dir=tmp_path / "o",
                   dry_run=True)
    assert argv[:4] == ["conda", "run", "-n", SCALEX_ENV]


def test_cmd_template_overrides_the_prefix(envs_dir, tmp_path):
    make_prefix(envs_dir, SCALEX_ENV)
    argv = mtb.run("SCALEX", "diagonal", inputs=SCALEX_INPUTS, out_dir=tmp_path / "o",
                   dry_run=True, cmd_template="docker run img {cmd}")
    assert argv[:3] == ["docker", "run", "img"] and "bash" not in argv


def test_pty_wrap_stays_inside_the_prefix_wrapper(envs_dir, tmp_path, monkeypatch):
    make_prefix(envs_dir, SCALEX_ENV)
    from multibench.engine import registry
    variant = registry.get("SCALEX").select("diagonal", {"rna", "atac_gas"})
    monkeypatch.setattr(variant, "pty", True)
    argv = mtb.run("SCALEX", "diagonal", inputs=SCALEX_INPUTS, out_dir=tmp_path / "o",
                   dry_run=True)
    assert argv[:2] == ["bash", "-c"] and argv[4] == "script"


def test_forced_prefix_without_prefix_raises(envs_dir, monkeypatch, tmp_path):
    monkeypatch.setenv(runner.RUN_MODE_VAR, "prefix")
    with pytest.raises(OSError) as e:
        mtb.run("SCALEX", "diagonal", inputs=SCALEX_INPUTS, out_dir=tmp_path / "o",
                dry_run=True)
    msg = str(e.value)
    assert str(envs_dir) in msg and "mtb.env.install" in msg and SCALEX_ENV in msg


def test_forced_conda_ignores_the_prefix(envs_dir, monkeypatch, tmp_path):
    make_prefix(envs_dir, SCALEX_ENV)
    monkeypatch.setenv(runner.RUN_MODE_VAR, "conda")
    argv = mtb.run("SCALEX", "diagonal", inputs=SCALEX_INPUTS, out_dir=tmp_path / "o",
                   dry_run=True)
    assert argv[:4] == ["conda", "run", "-n", SCALEX_ENV]


def test_forced_prefix_with_prefix_and_bad_value(envs_dir, monkeypatch):
    make_prefix(envs_dir, "matilda")
    monkeypatch.setenv(runner.RUN_MODE_VAR, "prefix")
    assert runner.run_mode("matilda")[0] == "prefix"
    monkeypatch.setenv(runner.RUN_MODE_VAR, "podman")
    with pytest.raises(ValueError, match="MULTIBENCH_RUN_MODE='podman'"):
        runner.run_mode("matilda")


def test_real_run_uses_the_wrapper_and_preflight_passes(envs_dir, tmp_path, monkeypatch):
    """``run()`` (not dry) spawns the bash wrapper: the preflight sees the
    prefix as installed without conda and Popen receives the wrapper argv."""
    import h5py
    import numpy as np
    make_prefix(envs_dir, SCALEX_ENV)
    calls = {}

    class FakePopen:
        def __init__(self, cmd, cwd, stdout, stderr, text, env=None,
                     start_new_session=False):
            calls["cmd"], calls["session"] = cmd, start_new_session
            self.pid, self.returncode = 4242, 0
            with h5py.File(Path(cwd) / "embedding.h5", "w") as f:
                f.create_dataset("data", data=np.zeros((6, 3)))

        def communicate(self):
            return "", ""

        def kill(self):
            pass

        def wait(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    res = mtb.run("SCALEX", "diagonal",
                  inputs={"rna": str(tmp_path / "a.h5"), "atac_gas": str(tmp_path / "b.h5")},
                  out_dir=str(tmp_path / "out"), convert=False)
    assert calls["cmd"][:2] == ["bash", "-c"] and calls["session"] is True
    assert res.cmd == calls["cmd"] and res.output.shape == (6, 3)


def test_preflight_reprobes_conda_before_refusing(envs_dir, tmp_path, monkeypatch):
    """An env conda created in another terminal after the first probe must
    not be refused from the stale per-process cache."""
    monkeypatch.setattr(envs, "_find_conda", lambda: "/fake/conda")
    listing = {"envs": ("/opt/conda/envs/base",)}
    calls = []

    def fake_list(conda):
        calls.append(conda)
        return listing["envs"]
    fake_list.cache_clear = lambda: listing.update(envs=("/opt/conda/envs/base",
                                                          f"/opt/conda/envs/{SCALEX_ENV}"))
    monkeypatch.setattr(envs, "_conda_prefixes", fake_list)
    monkeypatch.setattr(runner.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("must not spawn in this test") )
    # first probe: not there -> cache cleared -> second probe sees it -> the
    # run proceeds to the (failing-on-purpose) spawn
    with pytest.raises(pytest.fail.Exception):
        mtb.run("SCALEX", "diagonal",
                inputs={"rna": str(tmp_path / "a.h5"), "atac_gas": str(tmp_path / "b.h5")},
                out_dir=str(tmp_path / "out"), convert=False)
    assert len(calls) >= 2


# ---------------------------------------------------------------- install_packed
def _tiny_archive(path: Path, *, unpack_ok: bool = True) -> Path:
    """A conda-pack-shaped tar.gz: bin/python + bin/conda-unpack stubs."""
    def add(name, body, mode=0o755):
        data = body.encode()
        ti = tarfile.TarInfo(name)
        ti.size, ti.mode = len(data), mode
        t.addfile(ti, io.BytesIO(data))
    with tarfile.open(path, "w:gz") as t:
        add("bin/python", "#!/bin/sh\necho stub\n")
        add("bin/conda-unpack",
            "#!/bin/sh\n"
            + ("" if unpack_ok else "exit 3\n")
            + 'here=$(cd "$(dirname "$0")/.." && pwd)\n'
            'echo "PATH=$PATH" > "$here/unpacked.txt"\n')
        add("lib/keep", "", mode=0o644)
    return path


def test_install_packed_without_conda_creates_the_prefix(envs_dir, tmp_path, monkeypatch):
    tgz = _tiny_archive(tmp_path / "matilda.tar.gz")
    fetched = []
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda url: (fetched.append(url), (str(tgz), None))[1])
    assert envs.install_packed("matilda") is True
    prefix = envs_dir / "matilda"
    assert fetched and "matilda" in fetched[0]
    assert (prefix / "bin" / "python").exists() and (prefix / "lib" / "keep").exists()
    assert not (envs_dir / "matilda.partial").exists()
    # conda-unpack ran, with <prefix>/bin FIRST on PATH
    recorded = (prefix / "unpacked.txt").read_text()
    assert recorded.startswith(f"PATH={prefix / 'bin'}{os.pathsep}")
    # now installed, without conda
    assert envs.env_prefix("matilda") == prefix
    assert envs.installed_envs() == ["matilda"]
    # idempotent: an existing prefix returns True without downloading again
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda url: pytest.fail("must not download twice"))
    assert envs.install_packed("matilda") is True


def test_install_packed_explicit_envs_dir(envs_dir, tmp_path, monkeypatch):
    tgz = _tiny_archive(tmp_path / "x.tar.gz")
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url: (str(tgz), None))
    other = tmp_path / "elsewhere"
    assert envs.install_packed("scmb_r", envs_dir=other) is True
    assert (other / "scmb_r" / "bin" / "python").exists()
    assert not (envs_dir / "scmb_r").exists()


def test_install_packed_unpack_failure_cleans_up(envs_dir, tmp_path, monkeypatch, capsys):
    tgz = _tiny_archive(tmp_path / "bad.tar.gz", unpack_ok=False)
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url: (str(tgz), None))
    assert envs.install_packed("matilda") is False
    assert not (envs_dir / "matilda").exists() and not (envs_dir / "matilda.partial").exists()
    assert "falling back to the lockfile build" in capsys.readouterr().out
    assert envs.installed_envs() == []


def test_install_packed_no_archive_returns_false(envs_dir, monkeypatch):
    import urllib.error
    import urllib.request
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)

    def gone(url):
        raise urllib.error.HTTPError(url, 404, "nope", {}, None)
    monkeypatch.setattr(urllib.request, "urlretrieve", gone)
    assert envs.install_packed("matilda") is False
    assert not (envs_dir / "matilda").exists()


def test_install_packed_platform_guard_precedes_download(envs_dir, monkeypatch):
    import urllib.request
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "this host is darwin/arm64")
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda url: pytest.fail("downloaded on a non-linux host"))
    with pytest.raises(RuntimeError, match="darwin/arm64"):
        envs.install_packed("matilda")


def test_install_packed_signature():
    import inspect
    sig = inspect.signature(envs.install_packed)
    assert list(sig.parameters) == ["env", "envs_dir", "conda", "force"]
    assert all(sig.parameters[k].kind is inspect.Parameter.KEYWORD_ONLY
               for k in ("envs_dir", "conda", "force"))


# ---------------------------------------------------------------- env.install without conda
@pytest.fixture
def linux_no_conda(envs_dir, monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "packed_manifest",
                        lambda: {"scmb_torch": "https://x/scmb_torch.tar.gz"})
    monkeypatch.setattr(envs, "install_packed",
                        lambda env, **kw: pytest.fail(f"download of {env} started"))
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build started"))
    yield


def test_install_no_conda_lockfile_build_refused_with_archive(linux_no_conda):
    # SCALEX -> scmb_torch: archive published, but packed=False asked for a build
    with pytest.raises(RuntimeError) as e:
        mtb.env.install(["SCALEX"], packed=False, dry_run=False)
    assert str(e.value) == ("no conda/mamba on this host; scmb_torch has a packed "
                            "archive - pass packed=True")


def test_install_no_conda_refused_without_archive(linux_no_conda):
    # Matilda -> matilda: no archive in the (faked) manifest, so nothing can
    # provision it here - even with packed=True, and before any download
    env = envs.group_for("Matilda")
    assert env not in envs.packed_manifest()
    with pytest.raises(RuntimeError) as e:
        mtb.env.install(["Matilda", "SCALEX"], packed=True, dry_run=False)
    assert str(e.value) == (f"no conda/mamba on this host; {env} has no packed "
                            f"archive - install conda first")
    with pytest.raises(RuntimeError, match="has no packed archive - install conda first"):
        mtb.env.install(["Matilda"], packed=False, dry_run=False)


def test_install_no_conda_packed_path_works(envs_dir, monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "packed_manifest",
                        lambda: {"scmb_torch": "https://x/scmb_torch.tar.gz"})
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build started"))

    def fake_unpack(env, **kw):
        make_prefix(envs_dir, env)               # what a real unpack leaves behind
        return True
    monkeypatch.setattr(envs, "install_packed", fake_unpack)
    rows = mtb.env.install(["SCALEX"], packed=True, dry_run=False)
    assert [r["state"] for r in rows] == ["PACKED"]
    assert envs.installed_envs() == ["scmb_torch"]
    # a second call sees it as installed: no download, no build
    monkeypatch.setattr(envs, "install_packed",
                        lambda env, **kw: pytest.fail("downloaded an installed env"))
    assert [r["state"] for r in mtb.env.install(["SCALEX"], packed=True, dry_run=False)] == ["have"]


def test_install_dry_run_needs_no_conda(linux_no_conda):
    rows = mtb.env.install(["SCALEX"], packed=True)
    assert rows[0]["state"] == "packed archive published"
    rows = mtb.env.install(["SCALEX"], packed=False)
    assert rows[0]["state"] == "build(dry-run)"


# ---------------------------------------------------------------- scan env gate
def test_scan_env_gate_flips_on_the_fake_prefix(envs_dir):
    env = envs.group_for("Matilda")
    W._installed_envs.cache_clear()
    try:
        df = mtb.scan("D11", "vertical", methods=["Matilda"], verbose=False)
        row = df[df["modalities"] == "rna+adt"].iloc[0]
        assert not row["env_ok"] and "not installed" in row["env_reason"]
        assert row["command"].startswith("conda run -n " + env)
        make_prefix(envs_dir, env)
        W._installed_envs.cache_clear()
        df = mtb.scan("D11", "vertical", methods=["Matilda"], verbose=False)
        row = df[df["modalities"] == "rna+adt"].iloc[0]
        assert row["env_ok"] and row["files_ok"] and row["runnable"]
        assert row["env_reason"] == ""
        assert row["command"].startswith("bash -c ")
        assert f"CONDA_PREFIX={envs_dir / env}" in row["command"]
        assert "run_matilda.py" in row["command"]
    finally:
        W._installed_envs.cache_clear()
