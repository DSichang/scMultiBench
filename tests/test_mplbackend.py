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

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_popen(cmd, **kw):
        captured["env"] = kw.get("env")
        raise RuntimeError("stop after env capture")

    monkeypatch.setattr(runner.subprocess, "run", fake_popen)
    try:
        import multibench as mtb
        inp = mtb.inputs_for("D11", "Matilda", "vertical", modalities=["rna", "adt"])
        runner.run(method="Matilda", category="vertical", inputs=inp,
                   out_dir="/tmp/mpl_test")
    except Exception:
        pass
    env = captured.get("env")
    assert env is not None, "subprocess env was never constructed"
    assert env.get("MPLBACKEND") == "Agg", (
        f"kernel backend leaked into the method env: {env.get('MPLBACKEND')}")
    assert env.get("PYTHONNOUSERSITE") == "1"
