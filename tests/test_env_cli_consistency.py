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
from multibench import workflow
from multibench.engine import runner, envs, registry
from tests.conftest import full_scan


@pytest.fixture(autouse=True)
def _linux_no_envs(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])


# ----------------------------------------------------------------- plan summary
def _count_named(named: str) -> int:
    """How many envs ``a``, ``a and b``, ``a, b and c`` or ``5 envs`` names."""
    m = re.fullmatch(r"(\d+) envs", named)
    return int(m.group(1)) if m else len(re.split(r", | and ", named))


def total_unknowns(text: str) -> tuple[int, bool, bool]:
    """``(n envs, a download unknown, a disk size unknown)`` from the total line:
    a column with an unknown size reads "at least" or "not recorded" (Z27, r3)."""
    m = re.search(r"# (\d+) envs?: ([^\n]*)", text)
    rest = m.group(2)
    dl = "not recorded" in rest or rest.startswith("at least")
    disk = "on disk" not in rest or "at least" in rest.split(",", 1)[-1]
    return int(m.group(1)), dl, disk


def test_size_total_counts_unknowns_per_column():
    rows = [{"env": "a"}, {"env": "b"}, {"env": "c"}]
    sizes = {"a": {"archive_bytes": 1_000_000_000, "unpacked_bytes": 2_000_000_000},
             "b": {"archive_bytes": 1_000_000_000, "unpacked_bytes": None},
             "c": {"archive_bytes": 1_000_000_000}}
    totals, line = envs.plan_totals(rows, sizes, layers={})
    assert total_unknowns("# " + line) == (3, False, True)
    # a partly known column is never printed as a plain total; the known part
    # stays as a lower bound (the figure a storage-quota request needs)
    assert line == "3 envs: 3.0 GB to download, at least 2.0 GB on disk"
    assert totals["total_download_bytes"] == 3_000_000_000
    assert totals["total_disk_bytes"] is None
    totals, full = envs.plan_totals(rows[:1], sizes, layers={})
    assert full == "1 env: 1.0 GB to download, 2.0 GB on disk"
    assert totals["total_disk_bytes"] == 2_000_000_000


def test_python_pin_detection():
    for p in ("python", "python=3.7", "python==3.7.10", "python>=3.8", "python 3.9"):
        assert envs._is_python_pin(p)
    for p in ("pytorch=1.9.1", "python-igraph", "ipython", "r-base=4"):
        assert not envs._is_python_pin(p)


def test_conda_packages_untouched_without_python_version():
    spec = {"conda_packages": ["python=3.8", "numpy"]}
    assert envs._conda_packages(spec) == ["python=3.8", "numpy"]
    assert envs._conda_packages({**spec, "python_version": "3.8"}) == ["numpy"]


def test_every_blocked_script_method_names_the_block_in_its_setup_hint():
    for s in registry.load():
        if (s.env_spec or {}).get("difficulty") == "blocked-script":
            assert s.setup_hint, f"{s.id}: blocked-script needs a setup_hint saying why"
            assert any(v.helpers for v in s.variants) or "script" in s.setup_hint.lower()


def test_missing_helper_module_blocks_the_script_gate(tmp_path, monkeypatch):
    script = tmp_path / "tools_scripts" / "MIRA" / "main_MIRA.py"
    script.parent.mkdir(parents=True)
    script.write_text("from logger import *")
    monkeypatch.setattr(workflow.config, "repo_path", tmp_path)
    v = SimpleNamespace(entrypoint="tools_scripts/MIRA/main_MIRA.py", helpers=["logger.py"])
    # the package ships MIRA's logger.py, so the real check passes without the file
    assert workflow._missing_script(v, method="MIRA") == ""
    # a helper the package does not ship (MIRA's logger.py is shipped since 0.3.4)
    monkeypatch.setattr("multibench.engine.runner.shipped_helper", lambda m, h: None)
    why = workflow._missing_script(v, method="MIRA")
    # user wording (L17): what is missing and where it goes, no host jargon
    assert "MIRA's script imports logger.py" in why
    assert "Put a logger.py next to main_MIRA.py." in why
    assert "does not include it" in why
    assert why.endswith("mtb.method_info('MIRA')['setup_hint'] shows how.")
    assert "shim" not in why and "benchmark host" not in why
    (script.parent / "logger.py").write_text("")
    assert workflow._missing_script(v, method="MIRA") == ""
    # a variant without helpers (every other method) is untouched
    assert workflow._missing_script(SimpleNamespace(entrypoint="tools_scripts/MIRA/main_MIRA.py",
                                                    helpers=[])) == ""


def test_scan_reports_the_mira_helper_when_the_checkout_lacks_it(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "_installed_envs", lambda: frozenset())
    script = tmp_path / "tools_scripts" / "MIRA" / "main_MIRA.py"
    script.parent.mkdir(parents=True)
    script.write_text("from logger import *")
    monkeypatch.setattr(workflow.config, "repo_path", tmp_path)
    # a helper the package does not ship (MIRA's logger.py is shipped since 0.3.4)
    monkeypatch.setattr("multibench.engine.runner.shipped_helper", lambda m, h: None)
    row = full_scan("D11", "vertical", methods=["MIRA"]).iloc[0]
    assert not row["files_ok"] and "logger.py" in row["files_reason"]
    assert "logger.py" in row["reason"]


