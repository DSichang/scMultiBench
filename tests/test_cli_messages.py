"""The command line: help texts, evaluate, convert, plot, config, cmd templates."""
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


from tests.test_evaluate_and_io_messages import _adata  # noqa: E402 - an AnnData of 60 cells

# --- command line: X15, X17-X24, X31, X33, X53, X77, X78, X95, X97, X98 ------

# Y03: the cmd-template placeholder is replaced token by token
def test_cmd_template_bare_quoted_and_mid_string():
    from multibench.engine.runner import wrap_command
    cmd = ["python", "x.py", "--a", "1"]
    assert wrap_command(cmd, "srun {cmd}") == ["srun", *cmd]
    assert wrap_command(cmd, "sbatch --wrap '{cmd}'") == ["sbatch", "--wrap",
                                                         "python x.py --a 1"]
    assert wrap_command(cmd, "bash -c '{cmd} > log'") == ["bash", "-c",
                                                         "python x.py --a 1 > log"]


def test_cmd_template_needs_one_placeholder(tmp_path):
    for bad in ("srun", "{env_cmd} {env_cmd}"):
        with pytest.raises(ValueError, match=r"must hold \{env_cmd\} \(or \{cmd\}\) once"):
            mtb.run("Matilda", "vertical", inputs=MATILDA, out_dir=str(tmp_path),
                    dry_run=True, cmd_template=bad)


def test_cmd_template_env_cmd_in_quotes(tmp_path):
    argv = mtb.run("Matilda", "vertical", inputs=MATILDA, out_dir=str(tmp_path),
                   dry_run=True, cmd_template="sbatch --wrap '{env_cmd}'")
    assert argv[:2] == ["sbatch", "--wrap"] and len(argv) == 3
    assert argv[2].startswith("conda run -n ") or " run_matilda.py " in argv[2]
