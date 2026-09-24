"""CLI fixes from the round-1 virtual-student study (ledger L07, L22-L24,
L26-L29, L32, L41, L52, L61 for cli.py / envs.py / config.py).

Each test fails on the code before its fix: the subcommand, flag or message
it checks did not exist or read differently.
"""
import importlib
import json
import re
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multibench
import multibench as mtb
from multibench import cli, config, workflow
from multibench.engine import envs, ingest, runner


def _sub(name):
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    return sub.choices[name]


def _all_parsers():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    out = [parser]
    for p in sub.choices.values():
        out.append(p)
        for a in p._actions:
            if getattr(a, "choices", None) and isinstance(a.choices, dict):
                out.extend(a.choices.values())
    return out


@pytest.fixture
def no_envs(monkeypatch):
    monkeypatch.setattr(workflow, "_installed_envs", lambda: frozenset())


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])


# ================================================================ L22 info / config / fetch
def test_info_prints_what_to_check_before_a_run(capsys):
    rc = cli.main(["info", "StabMap"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("StabMap (R), env scmb_r")
    assert re.search(r"GPU:\s+not used", out) and "needs_labels: no" in out
    assert re.search(r"atac:\s+peak", out)
    assert re.search(r"mosaic\s+rna1\+rna2\+rna3\+adt1\+atac2", out)
    assert re.search(r"cross\s+rna1\+rna2\+rna3\+adt1\+adt2\+adt3", out)
    assert "209 s on D46 (21,416 cells)" in out and "RTX 4090" in out
    rc = cli.main(["info", "Matilda", "--format", "json"])
    info = json.loads(capsys.readouterr().out)
    assert rc == 0 and info["id"] == "Matilda" and info["env"] == "matilda"


def test_info_unknown_method_names_the_cli_listing(capsys):
    rc = cli.main(["info", "Matlida"])
    err = capsys.readouterr().err
    assert rc == 1 and "did you mean 'Matilda'" in err
    assert "see `multibench list`" in err and "mtb.list_methods()" not in err


def test_config_prints_each_path_with_its_source(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(config.DATA_PATH_VAR, str(tmp_path / "data"))
    monkeypatch.setenv(config.ENVS_DIR_VAR, str(tmp_path / "envs"))
    monkeypatch.setattr(config, "DEFAULT", config.Config())
    rc = cli.main(["config"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.splitlines()
    names = [l.split()[0] for l in lines if not l.startswith(" ")]
    assert names == ["data_path", "envs_dir", "repo_path", "scripts_commit", "result_path",
                     "leiden_flavor"]
    i = names.index("data_path") * 2
    assert lines[i].split()[1] == str(tmp_path / "data")
    assert "environment variable MULTIBENCH_DATA_PATH" in lines[i + 1]
    assert "environment variable MULTIBENCH_ENVS_DIR" in out
    # the scripts_commit row says whether the scripts are there, and at which commit
    assert "scripts_commit" in out
    rc = cli.main(["config", "--get", "data_path"])
    assert rc == 0 and capsys.readouterr().out == f"{tmp_path / 'data'}\n"
    rc = cli.main(["config", "--format", "json"])
    rows = json.loads(capsys.readouterr().out)
    assert rc == 0 and {r["name"] for r in rows} >= {"data_path", "repo_path", "envs_dir"}


def test_fetch_wraps_data_fetch_and_fetch_outputs(monkeypatch, tmp_path, capsys):
    mod = importlib.import_module("multibench.data.fetch")
    calls = []

    def fake_fetch(*ids, data_path=None, quiet=False):
        calls.append(("fetch", ids, data_path))
        print("downloading ...")              # library progress: must reach stderr
        return tmp_path

    def fake_outputs(ds, methods=None, *, data_path=None, quiet=False):
        calls.append(("outputs", ds, data_path))
        return tmp_path / "outputs" / ds
    monkeypatch.setattr(mod, "fetch", fake_fetch)
    monkeypatch.setattr(mod, "fetch_outputs", fake_outputs)
    rc = cli.main(["fetch", "D11", "D46,D52", "--data-path", str(tmp_path)])
    cap = capsys.readouterr()
    assert rc == 0
    assert [c[1] for c in calls] == [("D11",), ("D46",), ("D52",)]
    assert cap.out.splitlines() == [f"D11: {tmp_path / 'D11'}", f"D46: {tmp_path / 'D46'}",
                                    f"D52: {tmp_path / 'D52'}"]
    assert "downloading" in cap.err and "downloading" not in cap.out
    calls.clear()
    rc = cli.main(["fetch", "D11", "--outputs"])
    assert rc == 0 and calls == [("outputs", "D11", None)]
    assert capsys.readouterr().out == f"D11: {tmp_path / 'outputs' / 'D11'}\n"


def test_fetch_without_ids_is_a_usage_error_listing_them(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["fetch"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "D11" in err and "--scripts" in err


def test_fetch_scripts_reports_present_or_clones(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "_ROOT", tmp_path / "pkg")          # no tools_scripts there
    have = tmp_path / "have"
    (have / "tools_scripts").mkdir(parents=True)
    monkeypatch.setattr(config.DEFAULT, "repo_path", have)
    rc = cli.main(["fetch", "--scripts"])
    assert rc == 0
    assert capsys.readouterr().out == (f"method scripts present: {have / 'tools_scripts'} "
                                       f"(not a git checkout, commit unknown)\n")
    # absent: the same clone the first run performs (git faked)
    ran = []

    def fake_run(argv, check):
        ran.append(argv)
        (Path(argv[-1]) / "tools_scripts").mkdir(parents=True)
    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)
    fresh = tmp_path / "fresh"
    monkeypatch.setattr(config.DEFAULT, "repo_path", fresh)
    rc = cli.main(["fetch", "--scripts"])
    cap = capsys.readouterr()
    assert rc == 0 and ran and ran[0][:2] == ["git", "clone"]
    assert cap.out == (f"method scripts fetched: {fresh / 'tools_scripts'} "
                       f"(not a git checkout, commit unknown)\n")
    assert "fetching PYangLab/scMultiBench" in cap.err


def test_new_commands_name_their_python_function():
    helps = {name: _sub(name).format_help() for name in ("fetch", "config", "info")}
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    short = {a.dest: a.help for a in sub._choices_actions}
    assert "mtb.data.fetch" in short["fetch"] and "mtb.data.fetch_outputs" in short["fetch"]
    assert "mtb.config.DEFAULT" in short["config"]
    assert "mtb.method_info" in short["info"]
    assert "MULTIBENCH_DATA_PATH" in helps["config"]


# ================================================================ L23 path variables
def test_data_and_repo_path_variables(monkeypatch, tmp_path):
    for var, field in ((config.DATA_PATH_VAR, "data_path"), (config.REPO_PATH_VAR, "repo_path")):
        monkeypatch.delenv(var, raising=False)
        default = getattr(config.Config(), field)
        assert default.parent == config._BASE
        monkeypatch.setenv(var, str(tmp_path / field))
        cfg = config.Config()
        assert getattr(cfg, field) == tmp_path / field           # the variable wins
        setattr(cfg, field, str(tmp_path / "mine"))              # an assignment wins over it
        assert getattr(cfg, field) == tmp_path / "mine"
        assert isinstance(getattr(cfg, field), Path)
        setattr(cfg, field, None)                                # back to the variable
        assert getattr(cfg, field) == tmp_path / field
        monkeypatch.delenv(var)
        assert getattr(cfg, field) == default                    # unset: the default again
    assert config.Config(data_path="x").data_path == Path("x")


def test_config_docstring_lists_the_variables():
    doc = config.Config.__doc__
    for var in ("MULTIBENCH_DATA_PATH", "MULTIBENCH_REPO_PATH", "MULTIBENCH_ENVS_DIR"):
        assert var in doc


# ================================================================ L24 auto flavour note
def test_auto_flavour_on_a_cpu_host_says_how_to_get_gpu_builds(linux, monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    note = "(no NVIDIA GPU on this host); if the jobs run on GPU nodes, pass --flavor gpu"
    # scMoMaT's env has a CPU archive: the plan sums it, says why once and how
    # to get the GPU builds once
    assert cli.main(["env", "plan", "--methods", "scMoMaT"]) == 0
    err = capsys.readouterr().err
    assert "(CPU build, as this host has no NVIDIA GPU)" in err
    assert "# for jobs on GPU nodes, pass --flavor gpu" in err
    assert err.count("NVIDIA GPU") == 1 and err.count("--flavor gpu") == 1
    assert cli.main(["env", "install", "--packed", "--methods", "scMoMaT"]) == 0
    err = capsys.readouterr().err
    assert err.count(f"# installing CPU builds {note}") == 1
    assert err.count("NVIDIA GPU") == 1
    # StabMap's env has no CPU build: the GPU build is taken either way, so
    # neither command gives the --flavor advice
    for argv in (["env", "plan", "--methods", "StabMap"],
                 ["env", "install", "--packed", "--methods", "StabMap"]):
        assert cli.main(argv) == 0
        err = capsys.readouterr().err
        assert "(GPU build; no CPU build is published for it)" in err, argv
        assert "--flavor gpu" not in err and "CPU builds" not in err, argv
    for argv in (["env", "plan", "--methods", "StabMap", "--flavor", "cpu"],
                 ["env", "install", "--packed", "--methods", "StabMap", "--flavor", "gpu"]):
        assert cli.main(argv) == 0
        assert "pass --flavor gpu" not in capsys.readouterr().err, argv
    monkeypatch.setattr(envs, "host_has_gpu", lambda: True)
    assert cli.main(["env", "plan", "--methods", "StabMap"]) == 0
    assert "pass --flavor gpu" not in capsys.readouterr().err


def test_auto_flavour_note_on_the_real_install_path(linux, monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    torch = envs.group_for("scMoMaT")          # an env with a published CPU archive
    missing = [{"env": torch, "exists": False, "has_lock": True, "methods": ["scMoMaT"]}]
    monkeypatch.setattr(envs, "doctor", lambda **kw: [dict(r) for r in missing])
    monkeypatch.setattr(envs, "_find_conda", lambda: "/usr/bin/conda")
    monkeypatch.setattr(envs, "install_packed", lambda env, **kw: True)
    monkeypatch.setattr(envs, "installed_flavor", lambda env, conda=None: "cpu")
    monkeypatch.setattr(envs, "create_all", lambda **kw: [
        {"env": torch, "methods": ["scMoMaT"], "exists": True, "has_lock": True}])
    envs.install(["scMoMaT"], dry_run=False)          # Python: names the keyword
    err = capsys.readouterr().err
    assert err.count("# installing CPU builds (no NVIDIA GPU on this host)") == 1
    assert "pass flavor='gpu'" in err
    assert cli.main(["env", "install", "--packed", "--run", "--methods", "scMoMaT"]) == 0
    err = capsys.readouterr().err
    assert err.count("pass --flavor gpu") == 1
    # an env without a CPU archive gets the GPU build: no CPU-builds note
    missing[0].update(env="scmb_r", methods=["StabMap"])
    envs.install(["StabMap"], dry_run=False)
    assert "installing CPU builds" not in capsys.readouterr().err


# ================================================================ L26 {env_cmd}
def _inputs():
    return {"rna": "/d/rna.h5", "adt": "/d/adt.h5", "cty": "/d/cty.csv"}


def test_env_cmd_keeps_the_env_activation_and_cmd_stays_bare(monkeypatch):
    kept = mtb.run("Matilda", "vertical", inputs=_inputs(), out_dir="/o", dry_run=True,
                   cmd_template="srun --gres=gpu:1 {env_cmd}")
    bare = mtb.run("Matilda", "vertical", inputs=_inputs(), out_dir="/o", dry_run=True,
                   cmd_template="srun --gres=gpu:1 {cmd}")
    assert kept[:2] == ["srun", "--gres=gpu:1"] and bare[:2] == ["srun", "--gres=gpu:1"]
    assert kept[3:6] == ["run", "-n", "matilda"] and kept[2].endswith("conda")
    assert bare[2] == "python" and bare[3:] == kept[6 + 1:]
    # prefix mode: {env_cmd} carries the bash activation
    monkeypatch.delenv("MULTIBENCH_RUN_MODE", raising=False)
    monkeypatch.setattr(runner.envs, "env_prefix", lambda env, conda=None: Path("/envs") / env)
    kept = mtb.run("Matilda", "vertical", inputs=_inputs(), out_dir="/o", dry_run=True,
                   cmd_template="srun {env_cmd}")
    assert kept[:3] == ["srun", "bash", "-c"] and "CONDA_PREFIX=/envs/matilda" in kept[3]
    with pytest.raises(ValueError, match="both"):
        mtb.run("Matilda", "vertical", inputs=_inputs(), out_dir="/o", dry_run=True,
                cmd_template="srun {cmd} {env_cmd}")


def test_env_cmd_keeps_the_env_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.envs, "installed_envs", lambda conda=None: ["base"])
    monkeypatch.setattr(runner.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("must not launch without the env"))
    with pytest.raises(EnvironmentError, match="is not installed"):
        mtb.run("SCALEX", "diagonal", inputs={"rna": str(tmp_path / "a.h5"),
                                              "atac_gas": str(tmp_path / "b.h5")},
                out_dir=str(tmp_path / "o"), convert=False, cmd_template="srun {env_cmd}")


def test_cli_runner_help_and_dry_run(tmp_path, capsys):
    act = next(a for a in _sub("run")._actions if a.dest == "runner")
    assert "{env_cmd}" in act.help and "no env activation" in act.help
    assert "srun --gres=gpu:1 {env_cmd}" in act.help
    rc = cli.main(["run", "--method", "Matilda", "--category", "vertical",
                   "--input", "rna=/d/rna.h5", "--input", "adt=/d/adt.h5",
                   "--input", "cty=/d/cty.csv",
                   "--out-dir", str(tmp_path), "--dry-run", "--runner", "srun {env_cmd}"])
    out = capsys.readouterr().out
    assert rc == 0 and re.match(r"srun \S*conda run -n matilda python ", out)
    assert "{env_cmd}" in runner.run.__doc__ and "srun --gres=gpu:1 {env_cmd}" in runner.run.__doc__


# ================================================================ L27 total line
def test_total_line_does_not_sum_an_incomplete_column_as_a_total():
    rows = [{"env": e} for e in "abcdef"]
    sizes = {e: {"archive_bytes": 2_000_000_000, "unpacked_bytes": None} for e in "abcdef"}
    sizes["a"]["unpacked_bytes"] = sizes["b"]["unpacked_bytes"] = 2_050_000_000
    line = cli._size_total_line(rows, sizes)
    assert line.splitlines() == [
        "# total for 6 envs: 12.0 GB download",
        "# size on disk not recorded for 4 of 6 envs; unpacked envs are larger than the "
        "download, so check with du after the first install"]
    assert "packed_sizes.json" not in line and "on disk," not in line
    assert "size on disk not recorded for all 6 envs" in cli._size_total_line(
        rows, {e: {"archive_bytes": 1} for e in "abcdef"})


# ================================================================ L28 scan --strict
def test_scan_strict_exits_1_when_nothing_is_runnable(no_envs, capsys):
    rc = cli.main(["scan", "D11", "--category", "vertical"])
    capsys.readouterr()
    assert rc == 0                                           # without --strict: unchanged
    rc = cli.main(["scan", "D11", "--category", "vertical", "--strict"])
    cap = capsys.readouterr()
    assert rc == 1 and "Matilda" in cap.out                  # the table is still printed
    assert cap.err.startswith("error: --strict: 0 of ")
    assert "env not ready in" in cap.err


def test_scan_strict_names_a_requested_method_without_a_runnable_row(monkeypatch, capsys):
    every = frozenset(envs.group_for(m) for m in mtb.list_methods())
    monkeypatch.setattr(workflow, "_installed_envs", lambda: every)
    monkeypatch.setattr(workflow.envs, "host_has_gpu", lambda: True)
    rc = cli.main(["scan", "D11", "--category", "vertical", "--strict",
                   "--modalities", "rna,adt"])
    assert rc == 0, capsys.readouterr().err                  # rna+adt rows run on D11
    rc = cli.main(["scan", "D11", "--category", "vertical", "--strict",
                   "--methods", "Matilda,MIRA"])
    err = capsys.readouterr().err
    assert rc == 1 and "no runnable row for MIRA" in err and "  MIRA: " in err
    assert "Matilda:" not in err
    act = next(a for a in _sub("scan")._actions if a.dest == "strict")
    assert "multibench scan DS --category C --strict && sbatch" in act.help


# ================================================================ L29 CLI spellings in messages
def test_nothing_runnable_from_the_cli_names_cli_commands(no_envs, tmp_path, capsys):
    rc = cli.main(["run-all", "D11", "--category", "vertical", "--out-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1 and "nothing is runnable" in err
    assert "`multibench scan D11 --category vertical`" in err
    assert "`multibench env doctor`" in err
    assert "mtb.scan(" not in err and "mtb.env.doctor()" not in err
    assert config._CLI is False                              # reset after the command
    with pytest.raises(ValueError) as e:                     # Python keeps Python spellings
        mtb.run_all("D11", "vertical", out_dir=str(tmp_path), verbose=False)
    assert "mtb.scan('D11', 'vertical')" in str(e.value) and "mtb.env.doctor()" in str(e.value)


def test_nothing_runnable_lists_rows_with_files_in_place_first(no_envs, tmp_path):
    with pytest.raises(ValueError) as e:
        mtb.run_all("D11", "vertical", out_dir=str(tmp_path), verbose=False)
    body = str(e.value).split("blocked variants:\n", 1)[1]
    lines = [l for l in body.splitlines() if l.startswith("  ")]
    # D11 is CITE-seq: the rna+adt rows have their files; MIRA (rna+atac) does not
    assert len(lines) == 3 and all("(rna+adt)" in l for l in lines), lines
    assert [l.split()[0] for l in lines] == sorted(l.split()[0] for l in lines)


def test_scan_reason_names_the_cli_install_check(no_envs, capsys):
    rc = cli.main(["scan", "D11", "--category", "vertical", "--methods", "Matilda",
                   "--format", "csv", "--columns", "reason"])
    out = capsys.readouterr().out
    assert rc == 0 and "see `multibench env doctor`" in out and "mtb.env.doctor()" not in out


def test_evaluate_hints_use_flag_names(tmp_path, capsys):
    np.save(tmp_path / "e.npy", np.zeros((6, 2)))
    pd.DataFrame({"x": list("AABBCC")}).to_csv(tmp_path / "l.csv", index=False)
    rc = cli.main(["evaluate", "--output", str(tmp_path / "e.npy"),
                   "--labels", str(tmp_path / "l.csv"), "--metrics", "batch"])
    err = capsys.readouterr().err
    assert rc == 1 and "pass --batch CSV (or two or more --labels files)" in err
    assert "batch=<vector>" not in err


# ================================================================ L32 compact scan view
def test_compact_view_adds_atac_and_caveat_when_they_apply(capsys):
    rc = cli.main(["scan", "D28", "--category", "diagonal"])
    out = capsys.readouterr().out
    header = out.splitlines()[0].split()
    assert rc == 0
    assert header == ["method", "modalities", "atac", "runnable", "files_ok", "env_ok",
                      "runtime_tier", "reason", "caveat"]
    glue = next(l for l in out.splitlines() if l.strip().startswith("GLUE"))
    assert "setup: GLUE needs" in glue and "peak names" not in glue
    rc = cli.main(["scan", "D11", "--category", "vertical", "--modalities", "rna,adt"])
    header = capsys.readouterr().out.splitlines()[0].split()
    assert rc == 0 and header == cli._COMPACT_PLAN_COLUMNS    # no ATAC row, no caveat


def test_compact_columns_and_word_boundary_clip():
    df = pd.DataFrame({"method": ["A", "B"], "modalities": ["rna+adt", "rna+atac"],
                       "atac": [None, "peak"], "caveat": ["", None], "reason": ["", ""]})
    assert cli._compact_plan_columns(df) == ["method", "modalities", "atac", "runnable",
                                             "files_ok", "env_ok", "runtime_tier", "reason"]
    df["caveat"] = ["", "needs peaks"]
    assert cli._compact_plan_columns(df)[-1] == "caveat"
    clipped = cli._truncate("input files not found on disk: atac_gas.h5 not found here", 40)
    assert clipped == "input files not found on disk..." and len(clipped) <= 40


# ================================================================ L41 evaluate parity
@pytest.fixture
def fake_eval(monkeypatch):
    seen = {}

    def _fake(**kw):
        seen.clear()
        seen.update(kw, leiden=config.DEFAULT.leiden_flavor)
        lab = kw["labels"]
        if isinstance(lab, list) and all(Path(p).is_file() for p in lab):
            seen["label_rows"] = [pd.read_csv(p)["x"].tolist() for p in lab]
        return pd.DataFrame({"Value": [0.5]}, index=pd.Index(["ARI"]))
    monkeypatch.setattr(multibench, "evaluate", _fake)
    return seen


def test_evaluate_repeated_labels_and_python_default_metrics(fake_eval, tmp_path):
    rc = cli.main(["evaluate", "--output", "e.h5", "--labels", "a.csv", "--labels", "b.csv"])
    assert rc == 0
    assert fake_eval["labels"] == ["a.csv", "b.csv"]          # stacked in order, each a batch
    assert fake_eval["metrics"] is None                       # Python's default set
    rc = cli.main(["evaluate", "--output", "e.h5", "--labels", "a.csv"])
    assert rc == 0 and fake_eval["labels"] == "a.csv"


def test_evaluate_column_applies_to_every_labels_file(fake_eval, tmp_path):
    for name, vals in (("a.csv", ["A", "B"]), ("b.csv", ["C"])):
        pd.DataFrame({"bc": [f"c{i}" for i in range(len(vals))], "x": ["?"] * len(vals),
                      "ct": vals}).to_csv(tmp_path / name, index=False)
    rc = cli.main(["evaluate", "--output", "e.h5", "--labels", str(tmp_path / "a.csv"),
                   "--labels", str(tmp_path / "b.csv"), "--column", "ct"])
    assert rc == 0 and fake_eval["label_rows"] == [["A", "B"], ["C"]]
    act = next(a for a in _sub("evaluate")._actions if a.dest == "column")
    assert "each --labels CSV" in act.help and "mtb.evaluate" not in act.help


def test_evaluate_reads_labels_for_without_labels(fake_eval, capsys):
    rc = cli.main(["evaluate", "--output", "e.h5", "--dataset", "D28", "--method", "GLUE",
                   "--category", "diagonal", "--data-path", str(config.DEFAULT.data_path)])
    assert rc == 0
    want = mtb.labels_for("D28", "diagonal", "GLUE")
    assert dict(fake_eval["labels"]) == dict(want)
    assert list(fake_eval["labels"]) == list(want)
    assert "# labels: " in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        cli.main(["evaluate", "--output", "e.h5"])
    assert e.value.code == 2


def test_evaluate_leiden_flavor_and_task_deprecation(fake_eval, capsys):
    before = config.DEFAULT.leiden_flavor
    rc = cli.main(["evaluate", "--output", "e.h5", "--labels", "a.csv",
                   "--leiden-flavor", "leidenalg"])
    assert rc == 0 and fake_eval["leiden"] == "leidenalg"
    assert config.DEFAULT.leiden_flavor == before                # restored
    rc = cli.main(["evaluate", "--output", "e.h5", "--labels", "a.csv", "--task", "batch"])
    err = capsys.readouterr().err
    assert rc == 0 and fake_eval["metrics"] == "batch"
    assert "warning: --task is deprecated; use --metrics batch" in err
    with pytest.raises(SystemExit) as e:
        cli.main(["evaluate", "--output", "e.h5", "--labels", "a.csv",
                  "--task", "dimension_reduction"])
    assert e.value.code == 2
    act = next(a for a in _sub("run-all")._actions if a.dest == "leiden_flavor")
    assert act.choices == ["igraph", "leidenalg"]


def test_evaluate_two_label_files_score_the_batch_family(tmp_path, capsys):
    rng = np.random.default_rng(0)
    labels = np.tile(np.repeat(["A", "B", "C"], 20), 2)
    emb = rng.normal(size=(120, 4)) + np.array([{"A": 0, "B": 6, "C": 12}[x]
                                                for x in labels])[:, None]
    np.save(tmp_path / "e.npy", emb)
    pd.DataFrame({"x": labels[:60]}).to_csv(tmp_path / "l1.csv", index=False)
    pd.DataFrame({"x": labels[60:]}).to_csv(tmp_path / "l2.csv", index=False)
    pd.DataFrame({"x": labels}).to_csv(tmp_path / "clu.csv", index=False)
    rc = cli.main(["evaluate", "--output", str(tmp_path / "e.npy"),
                   "--labels", str(tmp_path / "l1.csv"), "--labels", str(tmp_path / "l2.csv"),
                   "--clustering", str(tmp_path / "clu.csv"), "--out", str(tmp_path / "m.csv")])
    assert rc == 0
    got = pd.read_csv(tmp_path / "m.csv", index_col=0)
    assert {"ARI", "ASW_batch", "iLISI"} <= set(got.index)


def test_find_task_help_has_no_default():
    act = next(a for a in _sub("find")._actions if a.dest == "task")
    assert "(default)" not in act.help


# ================================================================ L52 plot --input filtering
def _long(method, dataset="MYCITE", n=3):
    return pd.DataFrame({"metric": ["ARI", "NMI", "ASW"][:n], "value": [0.5, 0.6, 0.7][:n],
                         "method": method, "dataset": dataset, "category": "vertical"})


def test_plot_input_rows_all_removed_by_dataset_is_an_error(tmp_path, monkeypatch, capsys):
    from multibench import plot as plot_ns
    monkeypatch.setattr(multibench, "load_results", lambda **kw: _long("A", "D11"))
    monkeypatch.setattr(plot_ns, "bubble", lambda df, **kw: pytest.fail("must not draw"))
    mine = tmp_path / "mine.csv"
    pd.concat([_long("Mine"), _long("Mine")]).to_csv(mine, index=False)
    rc = cli.main(["plot", "bubble", "--input", str(mine), "--category", "vertical",
                   "--dataset", "D11", "--source", "rerun", "--out", str(tmp_path / "f.pdf")])
    err = capsys.readouterr().err
    assert rc == 1
    assert ("error: your 6 rows are for dataset MYCITE; --dataset D11 removed all of them "
            "(plot them without --category, or score your method on D11)") in err


def test_plot_input_rows_partly_removed_warns(tmp_path, monkeypatch, capsys):
    from multibench import plot as plot_ns
    seen = {}
    monkeypatch.setattr(multibench, "load_results", lambda **kw: _long("A", "D11"))
    monkeypatch.setattr(plot_ns, "bubble", lambda df, **kw: seen.update(df=df))
    mine = tmp_path / "mine.csv"
    pd.concat([_long("Mine", "D11"), _long("Other", "MYCITE")]).to_csv(mine, index=False)
    rc = cli.main(["plot", "bubble", "--input", str(mine), "--category", "vertical",
                   "--dataset", "D11", "--out", str(tmp_path / "f.pdf")])
    err = capsys.readouterr().err
    assert rc == 0 and sorted(seen["df"]["method"].unique()) == ["A", "Mine"]
    assert "warning: --dataset dropped 3 of your 6 rows (dataset MYCITE; method Other)" in err
    # --methods naming none of your methods removes all of them: an error too
    rc = cli.main(["plot", "bubble", "--input", str(mine), "--category", "vertical",
                   "--dataset", "D11", "--methods", "A", "--out", str(tmp_path / "f.pdf")])
    err = capsys.readouterr().err
    assert rc == 1 and "--methods A removed all of them (add Mine to --methods)" in err


# ================================================================ L07 convert --batch-index
def test_convert_batch_index_is_passed_through(tmp_path, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(ingest, "_to_anndata", lambda src: "ADATA")

    def fake_export(data, out, **kw):
        seen.update(kw)
        Path(out).mkdir(parents=True, exist_ok=True)
        return Path(out)
    monkeypatch.setattr(ingest, "export_dataset", fake_export)
    rc = cli.main(["convert", "B.h5ad", str(tmp_path / "LAB"), "--rna", "X",
                   "--category", "mosaic", "--batch-index", "2"])
    assert rc == 0 and seen["batch_index"] == 2 and seen["batch"] is None
    seen.clear()
    rc = cli.main(["convert", "B.h5ad", str(tmp_path / "LAB"), "--rna", "X"])
    assert rc == 0 and "batch_index" not in seen
    for argv, msg in ((["--batch", "obs:b", "--category", "mosaic", "--batch-index", "1"],
                       "mutually exclusive"),
                      (["--category", "vertical", "--batch-index", "1"], "mosaic or"),
                      (["--category", "cross", "--batch-index", "0"], "counts from 1")):
        with pytest.raises(SystemExit) as e:
            cli.main(["convert", "B.h5ad", str(tmp_path / "LAB"), "--rna", "X"] + argv)
        assert e.value.code == 2 and msg in capsys.readouterr().err


def test_convert_help_shows_the_per_batch_recipe(capsys):
    with pytest.raises(SystemExit):
        cli.main(["convert", "--help"])
    out = capsys.readouterr().out
    lines = [l.strip() for l in out.splitlines()]
    for i in (1, 2, 3):
        assert any(l.startswith("multibench convert") and l.endswith(f"--batch-index {i}")
                   for l in lines), i
    assert "multibench scan LAB --category mosaic --data-path data" in lines


# ================================================================ L61 CLI text
def test_help_text_has_no_internal_names_or_capital_emphasis():
    internal = ("packed_sizes.json", "packed_urls.json", ".multibench_flavor", "engine/",
                "methods.yaml", " gate", "stand-in")
    # acronyms, placeholder names, environment variable names (MULTIBENCH_*),
    # the NO-LOCK state value and mtb.config.DEFAULT are not emphasis
    allowed = {"NVIDIA", "CUDA", "ATAC", "JSON", "PATH", "LOCK", "MYCITE", "UINMF", "MOFA",
               "MULTIBENCH", "DEBUG", "DATA", "REPO", "ENVS", "TEMPLATE", "DATASET",
               "METHOD", "VALUE", "ROLE", "NAME", "GROUP", "LONG", "KIND", "COLUMNS",
               "FORMAT", "TASK", "LABELS", "BATCH", "METRICS", "OUTPUT", "CATEGORY",
               "MODALITY", "LAYER", "OBSM", "DTYPE", "MODALITIES", "INPUT", "TITLE",
               "RUNNER", "SOURCE", "OVERALL", "AGGREGATE", "TIMEOUT", "PARAM", "METHODS",
               "COMMAND", "DEFAULT", "CITE", "SCRIPTS"}
    for parser in _all_parsers():
        text = parser.format_help()
        for word in internal:
            assert word not in text, (parser.prog, word)
        for tok in re.findall(r"[A-Z]{4,}", text):
            assert tok in allowed, (parser.prog, tok)


def test_freeze_all_reports_skipped_in_lower_case(monkeypatch, capsys):
    monkeypatch.setattr(envs, "required_envs", lambda category=None: ["x"])
    monkeypatch.setattr(envs, "freeze", lambda env: (_ for _ in ()).throw(RuntimeError("no")))
    assert cli.main(["env", "freeze", "--all"]) == 0
    assert capsys.readouterr().out == "skipped x: no\n"


def test_blocked_script_tag_has_no_provenance():
    text = envs.DIFFICULTY["blocked-script"]
    assert "benchmark host" not in text and "shim" not in text and "setup_hint" in text


def test_cpu_fallback_warning_names_no_internal_file():
    msg = envs._cpu_fallback_warning("scmb_torch", {"scmb_torch-cpu": "u"}, {})
    assert msg == ("no CPU archive for scmb_torch; installing the GPU build (?) - the CPU "
                   "archive scmb_torch-cpu is not published yet")
