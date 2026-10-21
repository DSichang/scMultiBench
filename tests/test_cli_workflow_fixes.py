"""Work package J (workshop re-study, personas Chen and Ben): run error
scoping, compact CLI tables, dry-run command preview, --param, one env-name
resolver for every env command, did-you-mean before I/O, stdout/stderr
discipline.

Every test here is host-independent: where a verdict depends on which conda
envs are installed, ``workflow._installed_envs`` is pinned to "none".
"""
import json
import os
import re

import pandas as pd
import multibench
import pytest

import multibench as mtb
from multibench import workflow
from multibench.engine import envs, registry
from tests.conftest import full_scan



@pytest.fixture
def no_envs(monkeypatch):
    """Pretend no method env is installed (the workshop laptop situation)."""
    monkeypatch.setattr(workflow, "_installed_envs", lambda: frozenset())


# ------------------------------------------------------------------ J1: scoped "nothing is runnable"
def test_nothing_runnable_lists_only_requested_methods(no_envs):
    with pytest.raises(ValueError) as e:
        mtb.run("D11", "vertical", methods=["Matilda"], out_dir="/tmp/unused",
                    verbose=False, modalities="all")
    msg = str(e.value)
    assert msg.startswith("None of the requested methods (Matilda) can run on D11 "
                          "(vertical).")
    assert "one line per requested row" in msg
    # every requested variant on its own line, with ITS reason
    assert "\n  Matilda (rna+adt): Environment matilda is not installed" in msg
    assert "\n  Matilda (rna+atac): " in msg
    # nobody else's reason - Concerto/MIRA/MOFA2 were not requested
    for other in ("Concerto", "MIRA", "MOFA2", "scmb_concerto"):
        assert other not in msg
    assert "mtb.scan('D11', 'vertical', methods=['Matilda'], modalities=\"all\") says why." in msg


def test_nothing_runnable_without_methods_says_first_3_of_n(no_envs):
    with pytest.raises(ValueError) as e:
        mtb.run("D11", "vertical", out_dir="/tmp/unused", verbose=False,
                    modalities="all")
    msg = str(e.value)
    m = re.search(r"The first 3 of (\d+) blocked rows:\n", msg)
    assert m and int(m.group(1)) >= 10
    body = msg.split("blocked rows:\n", 1)[1]
    lines = [l for l in body.splitlines() if l.startswith("  ")]
    assert len(lines) == 3
    assert all(re.match(r"  \w+ \([\w+()]+\): \S", l) for l in lines), lines


def test_requested_method_with_no_variant_in_category_is_a_request_error(no_envs):
    """methods=['Matilda'] under cross used to yield the generic 'nothing is
    runnable' text with other methods' reasons; plan() already said it right."""
    for call in (lambda: mtb.run("D52", "cross", methods=["Matilda"], out_dir="/tmp/unused",
                                 verbose=False),
                 lambda: mtb.scan("D52", "cross", methods=["Matilda"], verbose=False)):
        with pytest.raises(ValueError) as e:
            call()
        assert str(e.value) == "Matilda does not run on cross data. Its categories: vertical."
        assert "nothing is runnable" not in str(e.value)


# ------------------------------------------------------------------ J6: did-you-mean before any I/O
@pytest.mark.parametrize("call", [
    lambda: full_scan("NOPE", "vertical", data_path="/nonexistent/dir", methods=["StabMapp"]),
    lambda: mtb.run("/nonexistent/dir/NOPE", "vertical", out_dir="/tmp/unused",
                        methods=["StabMapp"], verbose=False),
    lambda: mtb.run("/nonexistent/dir/NOPE", "vertical", out_dir="/tmp/unused",
                        params={"StabMapp": {"k": 1}}, verbose=False),
    lambda: full_scan("NOPE", "vertical", data_path="/nonexistent/dir", methods=["StabMapp"],
                     out_dir="runs"),
])
def test_unknown_method_raises_did_you_mean_before_filesystem(call):
    """A typo'd id must not surface as FileNotFoundError for the dataset folder
    (the folder here does not exist either): KeyError comes first."""
    with pytest.raises(KeyError) as e:
        call()
    assert "Did you mean StabMap" in str(e.value)


def test_scan_command_column_renders_for_out_dir():
    df = full_scan("D11", "vertical", out_dir="runs", methods=["Matilda"], verbose=False)
    plan = full_scan("D11", "vertical", methods=["Matilda"], verbose=False)
    # the '<out_dir>' placeholder by default; a real out_dir renders paste-ready lines
    assert list(df.columns) == list(plan.columns) == workflow.PLAN_COLUMNS
    pd.testing.assert_frame_equal(df.drop(columns="command"), plan.drop(columns="command"))
    assert (plan[plan["files_ok"]]["command"].str.contains("<out_dir>/Matilda_D11")).all()
    assert (df[df["files_ok"]]["command"].str.contains("runs/Matilda_D11")).all()
    ok = df[df["files_ok"]]
    assert (ok["command"].str.contains("conda run -n matilda")).all()
    assert (df[~df["files_ok"]]["command"] == "").all()


def test_run_dry_run_mirrors_run_builder():
    inp = mtb.engine.resolve.inputs_for("D11", "vertical", "Matilda", modalities=["rna", "adt"])
    argv = mtb.engine.runner.preview("Matilda", "vertical", inputs=inp, out_dir="o", params={"epochs": 5})[0]
    assert argv[:4] == ["conda", "run", "-n", "matilda"] or argv[1:4] == ["run", "-n", "matilda"]
    assert "--epochs" in argv and argv[argv.index("--epochs") + 1] == "5"
    assert argv[argv.index("--save_path") + 1] == os.path.join(os.path.abspath("o"), "")
    assert not os.path.exists("o")                       # nothing created
    with pytest.raises(KeyError, match="Did you mean"):
        mtb.engine.runner.preview("Matlda", "vertical", inputs=inp, out_dir="o")[0]


def test_run_all_params_unknown_method_before_io_and_dry_run_key_check():
    with pytest.raises(KeyError, match="Did you mean Matilda"):
        mtb.run("/nonexistent/NOPE", "vertical", out_dir="/tmp/unused",
                    params={"Matlda": {"epochs": 5}}, verbose=False)
    with pytest.raises(KeyError) as e:
        mtb.scan("D11", "vertical", methods=["Matilda"], params={"Matilda": {"bogus": 1}}, verbose=False)
    assert "does not accept ['bogus']" in str(e.value)
    # a valid key passes and the plan comes back
    plan = mtb.scan("D11", "vertical", methods=["Matilda"], params={"Matilda": {"epochs": 5}}, verbose=False)
    assert set(plan["method"]) == {"Matilda"}


# ------------------------------------------------------------------ J5: one env-name resolver
_SAMPLE = ["Matilda", "UINMF", "SCALEX", "Concerto", "totalVI", "MOFA2", "scMoMaT"]


def test_group_for_is_the_env_every_entry_point_uses():
    scan_env = dict(zip(*[full_scan("D11", "vertical")[c] for c in ("method", "env")]))
    for m in _SAMPLE:
        assert envs.group_for(m) == mtb.method_info(m)["env"]
        if m in scan_env:
            assert envs.group_for(m) == scan_env[m]
    assert envs.group_for("Matilda") == "matilda"
    assert envs.own_env_name("Matilda") == "scmb_matilda"       # the old answer, renamed


# ------------------------------------------------------------------ J7: stdout / stderr discipline
_READ_ONLY = [
    ["list", "--category", "vertical"],
    ["find", "--category", "vertical", "--tunable", "false"],
    ["scan", "D11", "--category", "vertical"],
    ["scan", "D11", "--category", "vertical", "--format", "tsv", "--columns", "all"],
    ["layout", "vertical"],
    ["env", "status"],
    ["env", "groups"],
    ["env", "plan", "--category", "vertical"],
    ["env", "doctor", "--category", "vertical"],
    ["env", "install", "--category", "vertical", "--dry-run"],
    ["env", "recipe", "Matilda"],
    ["env", "yml", "Matilda"],
    ["env", "create", "Matilda"],
    ["run-all", "D11", "--category", "vertical", "--out-dir", "/tmp/unused", "--dry-run",
     "--methods", "Matilda"],
    ["run", "--method", "Matilda", "--category", "vertical", "--input", "rna=data/D11/rna.h5",
     "--input", "adt=data/D11/adt.h5", "--input", "cty=data/D11/cty.csv", "--out", "o",
     "--dry-run"],
]


