"""A notebook kernel's MPLBACKEND must not leak into method subprocesses.

Jupyter exports MPLBACKEND=module://matplotlib_inline.backend_inline; passed
through `conda run` into a method env without matplotlib_inline, any method that
imports matplotlib dies at import. Surfaced the first time run_all was executed
FROM a notebook (Portal, diagonal tutorial section 8).
"""
import os

from multibench.engine import runner


def test_method_subprocess_gets_headless_backend(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    captured = {}

    def fake_popen(cmd, **kw):
        captured["env"] = kw.get("env")
        raise RuntimeError("stop after env capture")

    # the runner spawns via Popen (start_new_session, killpg on abort)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    # run() preflights the method's conda env before spawning; this machine
    # need not have `matilda`, so pretend it does to reach the Popen call
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["matilda"])
    try:
        import multibench as mtb
        inp = mtb.inputs_for("D11", "vertical", "Matilda", modalities=["rna", "adt"])
        runner.run(method="Matilda", category="vertical", inputs=inp,
                   out_dir="/tmp/mpl_test")
    except Exception:
        pass
    env = captured.get("env")
    assert env is not None, "subprocess env was never constructed"
    assert env.get("MPLBACKEND") == "Agg", (
        f"kernel backend leaked into the method env: {env.get('MPLBACKEND')}")
    assert env.get("PYTHONNOUSERSITE") == "1"
