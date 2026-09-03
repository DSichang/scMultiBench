"""The env subcommands agree with each other and with the registry.

Re-test findings: ``env plan`` summarised ``0 of unknown size`` under 16 rows of
``? disk``; ``env recipe`` printed ``python=3.7`` twice; ``env doctor`` marked a
missing env ``[L]`` while ``env status`` marked it ``[ ]``; MIRA was
``blocked-script`` in ``env status`` while method_info called it verified/public
with no word about the block.
"""
import re
from types import SimpleNamespace

import pytest

import multibench as mtb
from multibench import cli, workflow
from multibench.engine import envs, registry


@pytest.fixture(autouse=True)
def _linux_no_envs(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])


# ----------------------------------------------------------------- plan summary
def test_size_total_counts_unknowns_per_column():
    rows = [{"env": "a"}, {"env": "b"}, {"env": "c"}]
    sizes = {"a": {"archive_bytes": 1_000_000_000, "unpacked_bytes": 2_000_000_000},
             "b": {"archive_bytes": 1_000_000_000, "unpacked_bytes": None},
             "c": {"archive_bytes": 1_000_000_000}}
    line = cli._size_total_line(rows, sizes)
    assert "(3 archives; download size unknown for 0, disk size unknown for 2)" in line
    assert line.startswith("# total at least:")             # a floor once anything is unknown
    full = cli._size_total_line(rows[:1], sizes)
    assert "(1 archive; download size unknown for 0, disk size unknown for 0)" in full
    assert full.startswith("# total: ")


def test_env_plan_summary_matches_the_question_marks_in_the_rows(capsys):
    rc = cli.main(["env", "plan", "--category", "cross"])
    cap = capsys.readouterr()
    assert rc == 0
    m = re.search(r"\((\d+) archives?; download size unknown for (\d+), disk size unknown for (\d+)\)", cap.err)
    assert m, cap.err
    rows = [l for l in cap.out.splitlines() if l.strip()]
    assert int(m.group(1)) == len(rows)
    assert int(m.group(2)) == sum("? dl" in l for l in rows)
    assert int(m.group(3)) == sum("? disk" in l for l in rows)


# ----------------------------------------------------------------- recipe
def test_recipe_pins_python_once():
    spec = registry.get("Matilda").env_spec
    assert spec["python_version"] and any(envs._is_python_pin(c) for c in spec["conda_packages"]), \
        "Matilda's recipe is the duplicate-pin case this test guards"
    create = envs.create_commands("Matilda")[0]
    pins = [c for c in create if envs._is_python_pin(c)]
    assert pins == ["python=3.7"]
    yml = envs.environment_yml("Matilda")
    assert yml.count("- python=") == 1


def test_python_pin_detection():
    for p in ("python", "python=3.7", "python==3.7.10", "python>=3.8", "python 3.9"):
        assert envs._is_python_pin(p)
    for p in ("pytorch=1.9.1", "python-igraph", "ipython", "r-base=4"):
        assert not envs._is_python_pin(p)


def test_conda_packages_untouched_without_python_version():
    spec = {"conda_packages": ["python=3.8", "numpy"]}
    assert envs._conda_packages(spec) == ["python=3.8", "numpy"]
    assert envs._conda_packages({**spec, "python_version": "3.8"}) == ["numpy"]


# ----------------------------------------------------------------- one symbol set
def test_status_and_doctor_use_the_same_marks_and_legend(capsys):
    cli.main(["env", "status", "--methods", "Matilda,UINMF"])
    st = capsys.readouterr()
    cli.main(["env", "doctor", "--methods", "Matilda,UINMF"])
    doc = capsys.readouterr()
    st_marks = {l.split()[1]: l[:3] for l in st.out.splitlines()}          # method -> [x]
    doc_marks = {}
    for l in doc.out.splitlines():
        if l.startswith("["):
            for m in l.split("<-")[1].split(","):
                doc_marks[m.strip()] = l[:3]
    assert st_marks == doc_marks == {"Matilda": "[L]", "UINMF": "[L]"}
    assert "[ ]" not in st.out
    assert f"# legend: {envs.MARK_LEGEND}" in st.err
    assert f"# legend: {envs.MARK_LEGEND}" in doc.out
    assert "[x]=installed" in envs.MARK_LEGEND and "[L]=missing" in envs.MARK_LEGEND \
        and "[!]=missing" in envs.MARK_LEGEND


def test_status_mark_reflects_lockfile_and_install(monkeypatch, capsys):
    monkeypatch.setattr(envs, "lockfile", lambda env: None)
    cli.main(["env", "status", "--methods", "Matilda"])
    assert capsys.readouterr().out.startswith("[!] Matilda")
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: ["matilda"])
    cli.main(["env", "status", "--methods", "Matilda"])
    assert capsys.readouterr().out.startswith("[x] Matilda")


def test_env_mark_and_status_rows_carry_has_lock():
    assert envs.env_mark(True, False) == "x" and envs.env_mark(False, True) == "L" \
        and envs.env_mark(False, False) == "!"
    row = next(r for r in envs.status() if r["method"] == "Matilda")
    assert row["has_lock"] == (envs.lockfile(row["env"]) is not None)
    frame = envs.status(as_frame=True)
    assert "has_lock" in frame.columns


# ----------------------------------------------------------------- MIRA: the truth
def test_mira_block_is_declared_where_every_surface_reads_it():
    spec = registry.get("MIRA")
    assert spec.env_spec["difficulty"] == "blocked-script"
    assert spec.variants[0].helpers == ["logger.py"]
    info = mtb.method_info("MIRA", verbose=True)
    assert "logger.py" in info["setup_hint"] and "shim" in info["setup_hint"]
    # the verification row IS real (D12, RUN_OK_NO_EMBEDDING) - the hint says how
    ver = info["verification"]
    assert ver and any(r["status"] == "RUN_OK_NO_EMBEDDING" for r in ver)
    assert "RUN_OK_NO_EMBEDDING" in info["setup_hint"]
    assert "setup_hint" in envs.DIFFICULTY["blocked-script"]


def test_every_blocked_script_method_names_the_block_in_its_setup_hint():
    for s in registry.load():
        if (s.env_spec or {}).get("difficulty") == "blocked-script":
            assert s.setup_hint, f"{s.id}: blocked-script needs a setup_hint saying why"
            assert any(v.helpers for v in s.variants) or "script" in s.setup_hint.lower()


def test_missing_helper_module_blocks_the_script_gate(tmp_path, monkeypatch):
    script = tmp_path / "tools_scripts" / "MIRA" / "main_MIRA.py"
    script.parent.mkdir(parents=True)
    script.write_text("from logger import *")
    monkeypatch.setattr(workflow.config.DEFAULT, "repo_path", tmp_path)
    v = SimpleNamespace(entrypoint="tools_scripts/MIRA/main_MIRA.py", helpers=["logger.py"])
    why = workflow._missing_script(v, method="MIRA")
    assert "main_MIRA.py imports the local module(s) ['logger.py']" in why
    assert "does not ship" in why and "mtb.method_info('MIRA')['setup_hint']" in why
    (script.parent / "logger.py").write_text("")
    assert workflow._missing_script(v, method="MIRA") == ""
    # a variant without helpers (every other method) is untouched
    assert workflow._missing_script(SimpleNamespace(entrypoint="tools_scripts/MIRA/main_MIRA.py")) == ""


def test_scan_reports_the_mira_helper_when_the_checkout_lacks_it(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "_installed_envs", lambda: frozenset())
    script = tmp_path / "tools_scripts" / "MIRA" / "main_MIRA.py"
    script.parent.mkdir(parents=True)
    script.write_text("from logger import *")
    monkeypatch.setattr(workflow.config.DEFAULT, "repo_path", tmp_path)
    row = mtb.scan("D11", "vertical", methods=["MIRA"]).iloc[0]
    assert not row["files_ok"] and "logger.py" in row["files_reason"]
    assert "logger.py" in row["reason"]
