"""An R method must not read the user's personal R library.

On the benchmark host MOFA2 died with ``CXXABI_1.3.15 not found``: R loaded
plyr.so from R_LIBS_USER (~/R/x86_64-conda-linux-gnu-library/4.3), built with
another compiler, before the env's own copy. With R_LIBS_USER pointing at
nothing it ran. The method subprocess and the env build both get an
environment in which R sees only the env's library.
"""
import os

import pytest

from multibench.engine import envs, runner

USER = {"R_LIBS_USER": "/home/u/R/x86_64-conda-linux-gnu-library/4.3",
        "R_LIBS": "/home/u/Rlibs", "R_LIBS_SITE": "/home/u/Rsite",
        "R_ENVIRON_USER": "/home/u/.Renviron", "R_PROFILE_USER": "/home/u/.Rprofile"}


def test_without_user_r_library():
    env = envs.without_user_r_library({**USER, "PATH": "/bin", "HOME": "/home/u"})
    assert "R_LIBS" not in env and "R_LIBS_SITE" not in env
    # not empty: R's etc/Renviron turns an empty R_LIBS_USER into the default
    assert env["R_LIBS_USER"] == env["R_ENVIRON_USER"] == env["R_PROFILE_USER"] == os.devnull
    assert not os.path.isdir(os.devnull)
    assert env["PATH"] == "/bin" and env["HOME"] == "/home/u"


def _launch_env(monkeypatch, tmp_path, method, category):
    for k, v in USER.items():
        monkeypatch.setenv(k, v)
    captured = {}

    def fake_popen(cmd, **kw):
        captured["env"] = kw.get("env")
        raise RuntimeError("stop after env capture")

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner, "linux_only_sentence", lambda: None)
    monkeypatch.setattr(runner.envs, "installed_envs",
                        lambda conda=None: [envs.group_for(method)])
    monkeypatch.setattr(runner.envs, "other_env_prefix", lambda name: None)
    import multibench as mtb
    inp = mtb.engine.resolve.inputs_for("D11", category, method, modalities=["rna", "adt"])
    with pytest.raises(RuntimeError, match="stop after env capture"):
        runner.run(method=method, category=category, inputs=inp,
                   out_dir=str(tmp_path / "out"))
    return captured["env"]


def test_an_r_method_runs_without_the_user_r_library(monkeypatch, tmp_path):
    env = _launch_env(monkeypatch, tmp_path, "MOFA2", "vertical")
    assert "R_LIBS" not in env and "R_LIBS_SITE" not in env
    assert env["R_LIBS_USER"] == env["R_ENVIRON_USER"] == env["R_PROFILE_USER"] == os.devnull
    assert env["PYTHONNOUSERSITE"] == "1"
    # the process's own environment is left as it was
    assert os.environ["R_LIBS_USER"] == USER["R_LIBS_USER"]


def test_the_env_build_does_not_read_the_user_r_library(monkeypatch):
    """A post-install script would find rliger in the user's library, skip
    it, and leave an env whose run (which does not read that library) fails."""
    for k, v in USER.items():
        monkeypatch.setenv(k, v)
    seen = []

    def fake(cmd, **kw):
        seen.append(kw.get("env"))

        class R:
            returncode = 0
            stdout = stderr = ""
        return R()

    monkeypatch.setattr(envs.subprocess, "run", fake)
    envs._run_all([["true"], ["true"]])
    assert len(seen) == 2
    for env in seen:
        assert "R_LIBS" not in env and env["R_LIBS_USER"] == os.devnull
