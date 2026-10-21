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
    return mtb.engine.runner.run(method, category, inputs=inputs, out_dir=str(tmp_path / "o"),
                   params=params, dry_run=True, cmd_template="conda run -n x {cmd}")


# --- environments: X25, X26, X27, X29, X30, X32, X33, X34 --------------------

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


