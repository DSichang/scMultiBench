"""Regression tests for the dispatch path of ``run_all``.

Every pre-existing workflow test used ``dry_run=True``, so the branch that
actually dispatches a method - and with it ``timeout``, ``skip_existing`` and
override validation - was never executed by the suite. A broken if/elif/else
chain therefore shipped: ``mp`` was bound only in the ``else`` branch, and an
unconditional tail call dispatched every method a second time.

These tests fake ``_run`` so they assert dispatch behaviour without launching a
real method.
"""

import time

import numpy as np
import pytest

import multibench as mtb
from multibench import workflow as W
from multibench.engine import registry


class _FakeRes:
    def __init__(self, out):
        self.output = out


def _fake_run(calls, n_cells=2864, dims=20):
    def _inner(method, category, inputs, out_dir, params=None):
        calls.append({"method": method, "params": params, "out_dir": out_dir})
        return _FakeRes(np.zeros((n_cells, dims), dtype=float))
    return _inner


def _plan_modalities(method, dataset, category):
    plan = mtb.run_all(dataset, category, methods=[method],
                       out_dir="/tmp/unused", dry_run=True, verbose=False)
    ok = plan[plan["runnable"]]               # dry_run keeps blocked rows too
    return str((ok if len(ok) else plan).iloc[0]["modalities"]).split("+")


def test_timeout_path_dispatches_exactly_once(monkeypatch, tmp_path):
    """`timeout=` used to raise UnboundLocalError, then run the method twice."""
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))
    res = mtb.run_all("D11", "vertical", methods=["Matilda"],
                      out_dir=str(tmp_path), timeout=600,
                      evaluate=False, verbose=False)
    assert len(calls) == 1, f"dispatched {len(calls)}x with timeout set, expected 1"
    assert not len(res.failures), res.failures.to_string()


def test_no_timeout_path_dispatches_exactly_once(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))
    res = mtb.run_all("D11", "vertical", methods=["Matilda"],
                      out_dir=str(tmp_path), evaluate=False, verbose=False)
    assert len(calls) == 1, f"dispatched {len(calls)}x, expected 1"
    assert not len(res.failures), res.failures.to_string()


def test_unknown_override_rejected_on_the_timeout_path(monkeypatch, tmp_path):
    """Validation used to be skipped whenever `timeout` was set."""
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))
    res = mtb.run_all("D11", "vertical", methods=["Matilda"],
                      params={"Matilda": {"not_a_real_parameter": 1}},
                      out_dir=str(tmp_path), timeout=600,
                      evaluate=False, verbose=False)
    assert len(res.failures) == 1
    assert "does not accept" in str(res.failures.iloc[0]["error"])
    assert calls == [], "method was dispatched despite an invalid override"


def test_valid_override_reaches_the_method_with_timeout(monkeypatch, tmp_path):
    """A real override must survive the timeout branch, not be dropped."""
    method = "scMDC"
    mods = _plan_modalities(method, "D11", "vertical")
    tunable = (mtb.params_for(method, "vertical", mods) or {}).get("tunable") or {}
    if not tunable:
        pytest.skip(f"{method} exposes no tunable parameters in this build")
    key = sorted(tunable)[0]
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))
    mtb.run_all("D11", "vertical", methods=[method],
                params={method: {key: tunable[key]}},
                out_dir=str(tmp_path), timeout=600,
                evaluate=False, verbose=False)
    assert len(calls) == 1
    assert calls[0]["params"] == {key: tunable[key]}, \
        "override was dropped on the timeout path"


def test_timeout_bounds_evaluation_not_just_dispatch(monkeypatch, tmp_path):
    """Regression: the SIGALRM guard wrapped only the dispatch call and was
    cancelled right after it, so the metric layer ran unbounded. A method that
    finished in 35 s then spent 105 min in evaluation with timeout= set."""
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))

    def _slow_eval(*a, **k):
        time.sleep(60)
        raise AssertionError("evaluation should have been cut short by the deadline")

    monkeypatch.setattr(W, "_evaluate_best_order", _slow_eval)

    t0 = time.time()
    res = mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir=str(tmp_path),
                      timeout=2, evaluate=True, verbose=False)
    elapsed = time.time() - t0
    assert res.summary.iloc[0]["status"] == "TIMEOUT", res.summary.to_string()
    assert elapsed < 30, f"deadline never fired during evaluation ({elapsed:.1f}s)"


def test_deadline_is_disarmed_after_each_method(monkeypatch, tmp_path):
    """A leaked SIGALRM would fire during an unrelated later call."""
    import signal
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))
    mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir=str(tmp_path),
                timeout=600, evaluate=False, verbose=False)
    assert signal.alarm(0) == 0, "an armed alarm outlived run_all"


def test_skip_existing_does_not_redispatch(monkeypatch, tmp_path):
    """The unconditional tail call defeated skip_existing entirely."""
    mods = _plan_modalities("Matilda", "D11", "vertical")
    v0 = registry.get("Matilda").select("vertical", set(mods))
    mdir = tmp_path / "Matilda_D11"
    mdir.mkdir(parents=True)
    (mdir / v0.output.file).write_bytes(b"")   # output already present
    calls = []
    monkeypatch.setattr(W, "_run", _fake_run(calls))
    mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir=str(tmp_path),
                skip_existing=True, evaluate=False, verbose=False)
    assert calls == [], "skip_existing re-ran a method whose output already existed"
