"""env install: what it acts on, its dry run and its rows."""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

import multibench as mtb
from multibench.engine import builder, envs, registry  # noqa: F401


def _messages(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kw)
    return out, [str(w.message) for w in rec]


def _d11():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mtb.load_results("vertical", dataset="D11")


def _mine(base, name="Mine", scored_with="leidenalg/search/0.4.0"):
    rows = base[base.method == "Matilda"].assign(method=name, source="user")
    if scored_with is not None:
        rows = rows.assign(scored_with=scored_with)
    return rows


MATILDA = {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}
SCJOINT = {"rna": "r.h5", "atac_gas": "a.h5", "rna_cty": "c.csv"}


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)


def _dry(method, category, inputs, params, tmp_path):
    return mtb.run(method, category, inputs=inputs, out_dir=str(tmp_path / "o"),
                   params=params, dry_run=True, cmd_template="conda run -n x {cmd}")


# --- environments: X25, X26, X27, X29, X30, X32, X33, X34 --------------------

def test_x27_x26_install_acts_by_default_and_returns_a_frame(monkeypatch):
    import inspect
    from multibench.engine import envs
    sig = inspect.signature(mtb.env.install).parameters
    assert sig["dry_run"].default is False and "packed" not in sig and "force" not in sig
    plan = mtb.env.install(["Matilda"], dry_run=True)
    assert isinstance(plan, pd.DataFrame) and list(plan.columns[:3]) == [
        "env", "methods", "state"]
    assert isinstance(plan.to_dict("records"), list)
    doc = mtb.env.doctor("vertical")
    assert isinstance(doc, pd.DataFrame) and list(doc.columns[:3]) == [
        "env", "methods", "exists"]
    # a real install (the default) refuses off Linux before anything is fetched
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "this host is darwin/arm64")
    with pytest.raises(OSError) as e:
        mtb.env.install(["Matilda"])
    assert isinstance(e.value, RuntimeError)                         # X32: both
    assert "force" not in str(e.value)                               # X33: hidden


def test_x25_packed_is_build():
    with pytest.warns(FutureWarning, match=r"^env\.install\(packed=\) is deprecated and "
                                           r"will be removed in 0\.5; packed=True has no effect, and "
                                           r"packed=False is build=True\.$"):
        old = mtb.env.install(["SCALEX"], packed=False, dry_run=True)
    new = mtb.env.install(["SCALEX"], build=True, dry_run=True)
    pd.testing.assert_frame_equal(old, new)
    assert set(new.state) <= {"build(dry-run)", "NO-LOCK", "have"}


def test_x29_x30_dry_run_rows_say_what_runs():
    plan = mtb.env.install(["Multigrate", "scMM"], dry_run=True).set_index("env")
    published = plan[plan.state == "packed archive published"]
    assert len(published)
    for env, row in published.iterrows():
        assert row["cmds"] == [] and row["fallback_cmds"], env
    from multibench.engine import envs
    for env, row in plan.iterrows():
        if row["flavor"] and envs._single_build(env):
            assert row["flavor"] == "any", env


def test_x34_env_dir_is_its_all():
    # env.doctor is deprecated in 0.4.0: hidden from __all__ and dir()
    assert mtb.env.__all__ == ["install"]
    assert [n for n in dir(mtb.env) if not n.startswith("__")] == ["install"]
    assert callable(mtb.env.recipe)


# Y20: off Linux, env install says it once; its dry run ends with what to do
def test_env_install_off_linux_says_it_once(monkeypatch, capsys):
    from multibench import cli
    monkeypatch.setattr(envs, "host_platform_problem", lambda: "macOS")
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    assert cli.main(["env", "install", "--methods", "Matilda"]) == 1
    err = capsys.readouterr().err
    assert "warning:" not in err and err.count("run only on Linux") == 1
    cli.main(["env", "install", "--methods", "Matilda", "--dry-run"])
    assert "# dry run - run it without --dry-run on a Linux machine" in capsys.readouterr().err
