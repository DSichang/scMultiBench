"""A method whose script is unreachable must not be reported runnable.

Without this check, scan() calls such a method runnable and the failure only
arrives minutes later, from a shell, as a file-not-found.
"""
import urllib.parse
from types import SimpleNamespace

import multibench as mtb
from multibench import workflow


def _variant(entrypoint):
    return SimpleNamespace(entrypoint=entrypoint)


def test_relative_entrypoint_missing_from_a_present_checkout(tmp_path, monkeypatch):
    (tmp_path / "tools_scripts").mkdir()
    monkeypatch.setattr(workflow.config.DEFAULT, "repo_path", tmp_path)
    why = workflow._missing_script(_variant("tools_scripts/Gone/main_Gone.py"))
    assert "missing from the reference checkout" in why


def test_relative_entrypoint_present_in_the_checkout(tmp_path, monkeypatch):
    script = tmp_path / "tools_scripts" / "Here" / "main_Here.py"
    script.parent.mkdir(parents=True)
    script.write_text("print(1)")
    monkeypatch.setattr(workflow.config.DEFAULT, "repo_path", tmp_path)
    assert workflow._missing_script(_variant("tools_scripts/Here/main_Here.py")) == ""


def test_no_checkout_yet_reports_nothing(tmp_path, monkeypatch):
    """run()/run_all() fetch the scripts; flagging them first would be wrong."""
    monkeypatch.setattr(workflow.config.DEFAULT, "repo_path", tmp_path / "absent")
    monkeypatch.setattr(workflow.config, "__file__",
                        str(tmp_path / "pkg" / "multibench" / "config.py"))
    assert workflow._missing_script(_variant("tools_scripts/Any/main_Any.py")) == ""


def test_every_entrypoint_is_a_tools_scripts_path():
    """run() fetches tools_scripts/ from the scMultiBench repository and
    method_info builds scripts_url from the first entrypoint; both rely on
    every entrypoint being a tools_scripts/<folder>/ path."""
    from multibench.engine import registry
    for m in mtb.list_methods():
        eps = [v.entrypoint for v in registry.get(m).variants]
        assert eps and all(e.startswith("tools_scripts/") for e in eps), (m, eps)
        folder = "/".join(eps[0].split("/")[:2])
        assert mtb.method_info(m)["scripts_url"] == (
            "https://github.com/PYangLab/scMultiBench/tree/main/"
            + urllib.parse.quote(folder, safe="/"))
