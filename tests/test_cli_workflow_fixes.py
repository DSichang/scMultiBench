"""Work package J (workshop re-study, personas Chen and Ben): run_all error
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
from multibench import cli, workflow
from multibench.engine import envs, registry



@pytest.fixture
def no_envs(monkeypatch):
    """Pretend no method env is installed (the workshop laptop situation)."""
    monkeypatch.setattr(workflow, "_installed_envs", lambda: frozenset())


# ------------------------------------------------------------------ J1: scoped "nothing is runnable"
def test_nothing_runnable_lists_only_requested_methods(no_envs):
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", methods=["Matilda"], out_dir="/tmp/unused",
                    verbose=False)
    msg = str(e.value)
    assert "nothing is runnable" in msg and "methods=['Matilda']" in msg
    assert "one line per requested variant" in msg
    # every requested variant on its own line, with ITS reason
    assert "\n  Matilda (rna+adt): conda env 'matilda' is not installed" in msg
    assert "\n  Matilda (rna+atac): " in msg
    # nobody else's reason - Concerto/MIRA/MOFA2 were not requested
    for other in ("Concerto", "MIRA", "MOFA2", "scmb_concerto"):
        assert other not in msg
    assert "mtb.scan('D11', 'vertical', methods=['Matilda'])" in msg


def test_nothing_runnable_without_methods_says_first_3_of_n(no_envs):
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir="/tmp/unused", verbose=False)
    msg = str(e.value)
    m = re.search(r"First 3 of (\d+) blocked variants:\n", msg)
    assert m and int(m.group(1)) >= 10
    body = msg.split("blocked variants:\n", 1)[1]
    lines = [l for l in body.splitlines() if l.startswith("  ")]
    assert len(lines) == 3
    assert all(re.match(r"  \w+ \([\w+()]+\): \S", l) for l in lines), lines


def test_requested_method_with_no_variant_in_category_is_a_request_error(no_envs):
    """methods=['Matilda'] under cross used to yield the generic 'nothing is
    runnable' text with other methods' reasons; plan() already said it right."""
    for dry in (False, True):
        with pytest.raises(ValueError) as e:
            mtb.run_all("D52", "cross", methods=["Matilda"], out_dir="/tmp/unused",
                        dry_run=dry, verbose=False)
        assert "no 'cross' variant matches dataset='D52' methods=['Matilda']" in str(e.value)
        assert "nothing is runnable" not in str(e.value)


# ------------------------------------------------------------------ J6: did-you-mean before any I/O
@pytest.mark.parametrize("call", [
    lambda: mtb.scan("NOPE", "vertical", data_path="/nonexistent/dir", methods=["Stabmap"]),
    lambda: mtb.run_all("NOPE", "vertical", out_dir="/tmp/unused", data_path="/nonexistent/dir",
                        methods=["Stabmap"], verbose=False),
    lambda: mtb.run_all("NOPE", "vertical", out_dir="/tmp/unused", data_path="/nonexistent/dir",
                        params={"Stabmap": {"k": 1}}, verbose=False),
    lambda: mtb.scan("NOPE", "vertical", data_path="/nonexistent/dir", methods=["Stabmap"],
                     out_dir="runs"),
])
def test_unknown_method_raises_did_you_mean_before_filesystem(call):
    """A typo'd id must not surface as FileNotFoundError for the dataset folder
    (the folder here does not exist either): KeyError comes first."""
    with pytest.raises(KeyError) as e:
        call()
    assert "did you mean 'StabMap'" in str(e.value)


def test_cli_scan_unknown_method_did_you_mean_exit_1(capsys):
    rc = cli.main(["scan", "NOPE", "--category", "vertical", "--data-path", "/nonexistent",
                   "--methods", "Matlda"])
    cap = capsys.readouterr()
    assert rc == 1 and "did you mean 'Matilda'" in cap.err and cap.out == ""


# ------------------------------------------------------------------ J2: compact tables
def _widest(text):
    return max(len(l) for l in text.splitlines())


def test_cli_scan_default_table_is_compact(capsys):
    rc = cli.main(["scan", "D11", "--category", "vertical"])
    out = capsys.readouterr().out
    assert rc == 0
    header = out.splitlines()[0].split()
    assert header == cli._COMPACT_PLAN_COLUMNS
    assert _widest(out) <= 160, _widest(out)
    assert "..." in out                      # long reasons are clipped
    assert "env_reason" not in out


def test_cli_scan_columns_all_and_machine_formats_are_full(capsys):
    full_cols = list(mtb.scan("D11", "vertical", verbose=False).columns)
    assert "command" in full_cols                     # --columns all includes it
    rc = cli.main(["scan", "D11", "--category", "vertical", "--columns", "all"])
    out = capsys.readouterr().out
    assert rc == 0 and out.splitlines()[0].split() == full_cols
    assert _widest(out) > 400                  # the wide table is opt-in
    rc = cli.main(["scan", "D11", "--category", "vertical", "--format", "tsv"])
    out = capsys.readouterr().out
    assert rc == 0 and out.splitlines()[0].split("\t") == full_cols
    assert "..." not in out.split("\n")[1]     # never clipped in machine formats
    rc = cli.main(["scan", "D11", "--category", "vertical", "--format", "json",
                   "--methods", "Matilda"])
    out = capsys.readouterr().out
    rows = json.loads(out)
    assert rc == 0 and isinstance(rows, list) and rows and set(full_cols) <= set(rows[0])
    assert all(r["method"] == "Matilda" for r in rows)
    # an explicit --columns list is printed unclipped
    rc = cli.main(["scan", "D11", "--category", "vertical", "--columns", "method,reason",
                   "--methods", "Matilda"])
    out = capsys.readouterr().out
    # the CLI must not clip in machine formats - but scan() itself tail-truncates
    # a very long reason with " ... " (longer paths on the benchmark host), so
    # compare against the frame instead of asserting the marker is absent
    # (on a host with the envs installed the rna+adt row has no reason at all,
    # so check every printed row's reason rather than one fixed phrase)
    _frame = multibench.scan("D11", "vertical", methods=["Matilda"])
    assert rc == 0 and all(str(r) in out for r in _frame["reason"] if r)


def test_cli_scan_modalities_filter(capsys):
    rc = cli.main(["scan", "D11", "--category", "vertical", "--modalities", "rna,protein",
                   "--format", "csv"])
    out = capsys.readouterr().out
    df = pd.read_csv(__import__("io").StringIO(out))
    assert rc == 0 and set(df["modalities"]) == {"rna+adt"}


def test_cli_run_all_dry_run_is_compact_too(capsys, tmp_path):
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path),
                   "--dry-run"])
    cap = capsys.readouterr()
    assert rc == 0
    table = cap.out.split("\n# commands")[0]
    assert table.splitlines()[0].split() == cli._COMPACT_PLAN_COLUMNS
    assert _widest(table) <= 160
    assert "# dry run" in cap.err and "# dry run" not in cap.out


# ------------------------------------------------------------------ J3: dry-run prints the commands
def test_cli_run_all_dry_run_prints_commands(capsys, tmp_path, monkeypatch):
    # conda mode: the runner would print the bash prefix wrapper instead on a
    # host where the matilda prefix is on disk (tests/test_prefix_mode.py
    # pins that line); this test pins the conda line on every host
    monkeypatch.setattr(envs, "env_prefix", lambda env, conda=None: None)
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path),
                   "--dry-run", "--methods", "Matilda,scMoMaT"])
    cap = capsys.readouterr()
    assert rc == 0
    assert "# commands (" in cap.out
    cmd_lines = [l for l in cap.out.splitlines() if l.startswith(("Matilda (", "scMoMaT ("))]
    assert len(cmd_lines) == 2                 # rna+adt rows only: the others lack files
    mat = next(l for l in cmd_lines if l.startswith("Matilda (rna+adt)"))
    assert "conda run -n matilda" in mat and "run_matilda.py" in mat
    assert f"--save_path {tmp_path}/Matilda_D11/" in mat
    assert "rna.h5" in mat and "adt.h5" in mat and "cty.csv" in mat
    # machine formats carry the same thing as a column
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path),
                   "--dry-run", "--methods", "Matilda", "--format", "json"])
    rows = json.loads(capsys.readouterr().out)
    assert rc == 0 and "command" in rows[0]
    by_mods = {r["modalities"]: r["command"] for r in rows}
    assert "conda run -n matilda" in by_mods["rna+adt"]
    assert by_mods["rna+atac"] == ""           # files missing -> no command


def test_run_all_dry_run_help_matches_behaviour():
    """The --help promise and what the command prints must agree."""
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.dest == "command").choices["run-all"]
    help_txt = next(a for a in sub._actions if a.dest == "dry_run").help
    assert "command line" in help_txt and "execute" in help_txt


def test_scan_command_column_renders_for_out_dir():
    df = mtb.scan("D11", "vertical", out_dir="runs", methods=["Matilda"], verbose=False)
    plan = mtb.scan("D11", "vertical", methods=["Matilda"], verbose=False)
    # the '<out_dir>' placeholder by default; a real out_dir renders paste-ready lines
    assert list(df.columns) == list(plan.columns) == workflow.SCAN_COLUMNS
    pd.testing.assert_frame_equal(df.drop(columns="command"), plan.drop(columns="command"))
    assert (plan[plan["files_ok"]]["command"].str.contains("<out_dir>/Matilda_D11")).all()
    assert (df[df["files_ok"]]["command"].str.contains("runs/Matilda_D11")).all()
    ok = df[df["files_ok"]]
    assert (ok["command"].str.contains("conda run -n matilda")).all()
    assert (df[~df["files_ok"]]["command"] == "").all()


def test_run_dry_run_mirrors_run_builder():
    inp = mtb.inputs_for("D11", "vertical", "Matilda", modalities=["rna", "adt"])
    argv = mtb.run("Matilda", "vertical", inputs=inp, out_dir="o",
                   params={"epochs": 5}, dry_run=True)
    assert argv[:4] == ["conda", "run", "-n", "matilda"] or argv[1:4] == ["run", "-n", "matilda"]
    assert "--epochs" in argv and argv[argv.index("--epochs") + 1] == "5"
    assert argv[argv.index("--save_path") + 1] == os.path.join(os.path.abspath("o"), "")
    assert not os.path.exists("o")                       # nothing created
    # a custom template is honoured verbatim, like the real run
    argv2 = mtb.run("Matilda", "vertical", inputs=inp, out_dir="o",
                    cmd_template="srun {cmd}", dry_run=True)
    assert argv2[0] == "srun" and "conda" not in argv2
    with pytest.raises(KeyError, match="did you mean"):
        mtb.run("Matlda", "vertical", inputs=inp, out_dir="o", dry_run=True)


def test_cli_run_dry_run_prints_command(capsys):
    rc = cli.main(["run", "--method", "Matilda", "--category", "vertical",
                   "--input", "rna=data/D11/rna.h5", "--input", "adt=data/D11/adt.h5",
                   "--input", "cty=data/D11/cty.csv", "--out", "runs/M", "--dry-run",
                   "-p", "epochs=5"])
    cap = capsys.readouterr()
    assert rc == 0
    # conda may be printed as a bare word or as the resolved absolute path
    _first = cap.out.split()[0]
    assert _first.endswith("conda") and " run -n matilda " in cap.out and "--epochs 5" in cap.out
    assert "dry run" in cap.err


# ------------------------------------------------------------------ J4: --param
@pytest.mark.parametrize("text,expected", [
    ("5", 5), ("0.1", 0.1), ("1e-3", 0.001), ("true", True), ("False", False),
    ("none", None), ("cuda", "cuda"), ("[1,2]", [1, 2]), ("-3", -3), ("3.0", 3.0),
])
def test_parse_scalar(text, expected):
    got = cli._parse_scalar(text)
    assert got == expected and type(got) is type(expected)


def test_parse_params_shape_and_errors():
    assert cli._parse_params(["Matilda:epochs=5", "Matilda:lr=0.001", "totalVI:x=a"]) == \
        {"Matilda": {"epochs": 5, "lr": 0.001}, "totalVI": {"x": "a"}}
    assert cli._parse_params(["epochs=5"], default_method="Matilda") == \
        {"Matilda": {"epochs": 5}}
    with pytest.raises(KeyError, match="did you mean 'Matilda'"):
        cli._parse_params(["Matlda:epochs=5"])
    with pytest.raises(SystemExit):
        cli._parse_params(["Matilda:epochs"])
    with pytest.raises(SystemExit):
        cli._parse_params(["epochs=5"])            # run-all: METHOD: is required


def test_cli_run_all_param_reaches_dry_run_command(capsys, tmp_path):
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path),
                   "--dry-run", "--methods", "Matilda", "-p", "Matilda:epochs=5",
                   "--param", "Matilda:lr=0.001"])
    cap = capsys.readouterr()
    assert rc == 0
    line = next(l for l in cap.out.splitlines() if l.startswith("Matilda (rna+adt)"))
    assert "--epochs 5" in line and "--lr 0.001" in line
    # unknown method: did-you-mean, exit 1, nothing on stdout
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path),
                   "--dry-run", "-p", "Matlda:epochs=5"])
    cap = capsys.readouterr()
    assert rc == 1 and "did you mean 'Matilda'" in cap.err and cap.out == ""
    # unknown key: rejected in the dry run, naming the accepted keys
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path),
                   "--dry-run", "--methods", "Matilda", "-p", "Matilda:bogus=1"])
    cap = capsys.readouterr()
    assert rc == 1 and "does not accept ['bogus']" in cap.err and "'epochs'" in cap.err
    # malformed: usage error, exit 2
    with pytest.raises(SystemExit) as ei:
        cli.main(["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path),
                  "--dry-run", "-p", "Matilda:epochs"])
    assert ei.value.code == 2
    assert "METHOD:KEY=VALUE" in capsys.readouterr().err


def test_cli_run_param_forwarded_and_prefix_checked(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run(**kw):
        captured.update(kw)

        class R:
            out_dir = tmp_path
        return R()
    monkeypatch.setattr(mtb, "run", fake_run)
    rc = cli.main(["run", "--method", "SCALEX", "--category", "diagonal",
                   "--input", "rna=a.h5", "--input", "atac_gas=b.h5", "--out", str(tmp_path),
                   "-p", "epochs=5", "--param", "SCALEX:lr=1e-3", "-p", "gpu=true"])
    assert rc == 0
    assert captured["params"] == {"epochs": 5, "lr": 0.001, "gpu": True}
    with pytest.raises(SystemExit) as ei:
        cli.main(["run", "--method", "SCALEX", "--category", "diagonal",
                  "--input", "rna=a.h5", "--out", str(tmp_path), "-p", "Matilda:epochs=5"])
    assert ei.value.code == 2
    assert "--method is 'SCALEX'" in capsys.readouterr().err


def test_param_help_has_an_example():
    p = cli.build_parser()
    subs = next(a for a in p._actions if a.dest == "command").choices
    for name in ("run", "run-all"):
        act = next(a for a in subs[name]._actions if a.dest == "param")
        assert "-p" in act.option_strings and "--param" in act.option_strings
        assert "epochs=5" in act.help


def test_run_all_params_unknown_method_before_io_and_dry_run_key_check():
    with pytest.raises(KeyError, match="did you mean 'Matilda'"):
        mtb.run_all("NOPE", "vertical", out_dir="/tmp/unused", data_path="/nonexistent",
                    params={"Matlda": {"epochs": 5}}, verbose=False)
    with pytest.raises(KeyError) as e:
        mtb.run_all("D11", "vertical", out_dir=None, dry_run=True, methods=["Matilda"],
                    params={"Matilda": {"bogus": 1}}, verbose=False)
    assert "does not accept ['bogus']" in str(e.value)
    # a valid key passes and the plan comes back
    plan = mtb.run_all("D11", "vertical", out_dir=None, dry_run=True, methods=["Matilda"],
                       params={"Matilda": {"epochs": 5}}, verbose=False)
    assert set(plan["method"]) == {"Matilda"}


# ------------------------------------------------------------------ J5: one env-name resolver
_SAMPLE = ["Matilda", "UINMF", "SCALEX", "Concerto", "totalVI", "MOFA2", "scMoMaT"]


def test_default_env_name_is_the_env_every_entry_point_uses():
    scan_env = dict(zip(*[mtb.scan("D11", "vertical")[c] for c in ("method", "env")]))
    for m in _SAMPLE:
        assert envs.default_env_name(m) == envs.group_for(m) == mtb.method_info(m)["env"]
        if m in scan_env:
            assert envs.default_env_name(m) == scan_env[m]
    assert envs.default_env_name("Matilda") == "matilda"
    assert envs.own_env_name("Matilda") == "scmb_matilda"       # the old answer, renamed
    with pytest.raises(KeyError, match="did you mean"):
        envs.default_env_name("Matlda")


@pytest.mark.parametrize("method", _SAMPLE)
def test_cli_env_recipe_yml_create_name_the_scan_env(capsys, method):
    expected = envs.group_for(method)
    rc = cli.main(["env", "recipe", method])
    out = capsys.readouterr().out
    assert rc == 0
    cmds = [l for l in out.splitlines() if not l.startswith("#")]
    assert cmds and all(f" -n {expected} " in l for l in cmds), cmds
    assert out.splitlines()[0].startswith("#") and expected in out.splitlines()[0]
    rc = cli.main(["env", "yml", method])
    out = capsys.readouterr().out
    assert rc == 0 and f"\nname: {expected}\n" in out
    rc = cli.main(["env", "create", method])
    cap = capsys.readouterr()
    assert rc == 0 and all(f" -n {expected} " in l for l in cap.out.splitlines())
    rc = cli.main(["env", "plan", "--methods", method])
    out = capsys.readouterr().out
    assert rc == 0 and out.split()[0] == expected


def test_cli_env_custom_name_is_flagged(capsys):
    rc = cli.main(["env", "recipe", "Matilda", "--name", "mine"])
    out = capsys.readouterr().out
    assert rc == 0 and "custom --name" in out.splitlines()[0] and "'matilda'" in out.splitlines()[0]
    assert " -n mine " in out.splitlines()[1]


def test_env_name_help_shows_resolved_example():
    p = cli.build_parser()
    env = next(a for a in p._actions if a.dest == "command").choices["env"]
    subs = next(a for a in env._actions if a.dest == "env_cmd").choices
    for name in ("recipe", "yml", "create"):
        act = next(a for a in subs[name]._actions if a.dest == "name")
        assert "default_env_name" in act.help and "matilda" in act.help


def test_cli_env_status_accepts_category_and_methods(capsys):
    rc = cli.main(["env", "status", "--methods", "Matilda,UINMF"])
    out = capsys.readouterr().out
    assert rc == 0
    assert len(out.strip().splitlines()) == 2 and " matilda " in out and " scmb_r " in out
    rc = cli.main(["env", "status", "--category", "vertical"])
    out = capsys.readouterr().out
    assert rc == 0 and "Matilda" in out and "SCALEX" not in out


def test_env_status_reports_the_real_env():
    rows = {r["method"]: r for r in envs.status()}
    assert rows["Matilda"]["env"] == rows["Matilda"]["group"] == "matilda"
    assert rows["Matilda"]["own_env"] == "scmb_matilda"


# ------------------------------------------------------------------ J7: stdout / stderr discipline
_READ_ONLY = [
    ["list", "--category", "vertical"],
    ["find", "--category", "vertical", "--tunable", "false"],
    ["scan", "D11", "--category", "vertical"],
    ["scan", "D11", "--category", "vertical", "--format", "tsv", "--columns", "all"],
    ["layout", "vertical"],
    ["cite", "Matilda"],
    ["env", "status"],
    ["env", "groups"],
    ["env", "plan", "--category", "vertical"],
    ["env", "doctor", "--category", "vertical"],
    ["env", "install", "--category", "vertical"],
    ["env", "recipe", "Matilda"],
    ["env", "yml", "Matilda"],
    ["env", "create", "Matilda"],
    ["run-all", "D11", "--category", "vertical", "--out-dir", "/tmp/unused", "--dry-run",
     "--methods", "Matilda"],
    ["run", "--method", "Matilda", "--category", "vertical", "--input", "rna=data/D11/rna.h5",
     "--input", "adt=data/D11/adt.h5", "--input", "cty=data/D11/cty.csv", "--out", "o",
     "--dry-run"],
]


@pytest.mark.parametrize("argv", _READ_ONLY, ids=lambda a: " ".join(a[:2]))
def test_read_only_commands_put_data_on_stdout_and_notes_on_stderr(argv, capsys, monkeypatch):
    monkeypatch.setattr(workflow, "_installed_envs", lambda: frozenset())
    rc = cli.main(argv)
    cap = capsys.readouterr()
    assert rc == 0
    assert cap.out.strip(), "a read-only command must print its data on stdout"
    bad = [l for l in cap.out.splitlines()
           if l.startswith(("error:", "warning:", "[run_all]", "[env]", "# dry"))]
    assert bad == [], f"diagnostics leaked to stdout: {bad}"
    notes = [l for l in cap.err.splitlines() if l.strip()]
    assert all(l.startswith(("#", "warning:")) for l in notes), notes


@pytest.mark.parametrize("argv,code,needle", [
    (["scan", "NOPE", "--category", "vertical", "--data-path", "/nonexistent"], 1,
     "does not exist"),
    (["cite", "BOGUS"], 1, "BOGUS"),
    (["env", "recipe", "Matlda"], 1, "did you mean 'Matilda'"),
    (["run-all", "D52", "--category", "cross", "--out-dir", "/tmp/unused", "--methods",
      "Matilda"], 1, "no 'cross' variant matches"),
])
def test_runtime_errors_go_to_stderr_with_exit_1(argv, code, needle, capsys):
    rc = cli.main(argv)
    cap = capsys.readouterr()
    assert rc == code
    assert cap.out == "" and cap.err.startswith("error: ") and needle in cap.err


@pytest.mark.parametrize("argv", [
    ["env", "freeze"],
    ["scan"],
    ["run-all", "D11", "--category", "vertical", "--out-dir", "x", "-p", "k"],
])
def test_usage_errors_exit_2_on_stderr(argv, capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(argv)
    assert ei.value.code == 2
    cap = capsys.readouterr()
    assert cap.out == "" and "usage:" in cap.err


def test_env_install_runtime_error_is_on_stderr(monkeypatch, capsys):
    def boom(**kw):
        raise RuntimeError("conda/mamba not found on this machine")
    # a Linux host with conda: install()'s own platform / no-conda prechecks
    # (which run before create_all) stay quiet, so the build error is the one
    # that surfaces
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "_find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(envs, "create_all", boom)
    rc = cli.main(["env", "install", "--run", "--methods", "Matilda"])
    cap = capsys.readouterr()
    assert rc == 1 and cap.out == "" and "error: conda/mamba not found" in cap.err


def test_warnings_are_formatted_on_stderr(monkeypatch, capsys):
    import warnings

    def warn_scan(*a, **k):
        warnings.warn("something odd in the data")
        return pd.DataFrame({"method": ["M"], "runnable": [True]})
    monkeypatch.setattr(mtb, "scan", warn_scan)
    monkeypatch.delenv("MULTIBENCH_DEBUG", raising=False)
    rc = cli.main(["scan", "D11", "--category", "vertical"])
    cap = capsys.readouterr()
    assert rc == 0 and "M" in cap.out
    assert "warning: something odd in the data" in cap.err
    assert "UserWarning" not in cap.err and ".py:" not in cap.err


def test_run_all_progress_goes_to_stderr(monkeypatch, tmp_path, capsys):
    class FakeRes:
        summary = pd.DataFrame({"method": ["M"], "status": ["CHAIN_OK"]})

    def fake_run_all(*a, **k):
        print("[run_all] M (vertical/D11) ...")          # library progress on stdout
        return FakeRes()
    monkeypatch.setattr(mtb, "run_all", fake_run_all)
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path)])
    cap = capsys.readouterr()
    assert rc == 0
    assert "[run_all]" in cap.err and "[run_all]" not in cap.out
    assert cap.out.splitlines()[0].split() == ["method", "status"]
