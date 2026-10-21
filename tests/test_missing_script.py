"""A method whose script is unreachable must not be reported runnable.

Without this check, scan() calls such a method runnable and the failure only
arrives minutes later, from a shell, as a file-not-found.
"""
import urllib.parse
from types import SimpleNamespace

import multibench as mtb
from multibench import workflow


def _variant(entrypoint):
    return SimpleNamespace(entrypoint=entrypoint, helpers=[])


def test_relative_entrypoint_missing_from_a_present_checkout(tmp_path, monkeypatch):
    (tmp_path / "tools_scripts").mkdir()
    monkeypatch.setattr(workflow.config, "repo_path", tmp_path)
    why = workflow._missing_script(_variant("tools_scripts/Gone/main_Gone.py"))
    assert "missing from the reference checkout" in why


def test_relative_entrypoint_present_in_the_checkout(tmp_path, monkeypatch):
    script = tmp_path / "tools_scripts" / "Here" / "main_Here.py"
    script.parent.mkdir(parents=True)
    script.write_text("print(1)")
    monkeypatch.setattr(workflow.config, "repo_path", tmp_path)
    assert workflow._missing_script(_variant("tools_scripts/Here/main_Here.py")) == ""


def test_no_checkout_yet_reports_nothing(tmp_path, monkeypatch):
    """run()/run() fetch the scripts; flagging them first would be wrong."""
    monkeypatch.setattr(workflow.config, "repo_path", tmp_path / "absent")
    monkeypatch.setattr(workflow._settings, "_ROOT", tmp_path / "pkg")
    assert workflow._missing_script(_variant("tools_scripts/Any/main_Any.py")) == ""


