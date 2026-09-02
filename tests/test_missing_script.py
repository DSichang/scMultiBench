"""A method whose script is unreachable must not be reported runnable.

Four spatial variants (SPIRAL, GPSA, PASTE, PASTE2) carry entrypoints that are
ABSOLUTE paths on the machine the benchmark was produced on. Without this
check, scan() calls them runnable everywhere and the failure only arrives
minutes later, from a shell, as a file-not-found on a path the user has never
seen.
"""
from pathlib import Path
from types import SimpleNamespace

import multibench as mtb
from multibench import workflow


def _variant(entrypoint):
    return SimpleNamespace(entrypoint=entrypoint)


def test_absolute_entrypoint_that_does_not_exist_is_reported(tmp_path):
    why = workflow._missing_script(_variant("/no/such/machine/main_X.py"))
    assert "benchmark-host-only" in why or "absolute path on another machine" in why
    assert "/no/such/machine/main_X.py" in why


def test_absolute_entrypoint_that_exists_is_fine(tmp_path):
    p = tmp_path / "main_X.py"
    p.write_text("print(1)")
    assert workflow._missing_script(_variant(str(p))) == ""


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


def test_only_the_known_two_still_need_a_machine_specific_script():
    """The outstanding list, so it cannot grow silently - or shrink unnoticed.

    PASTE and PASTE2 were repointed at the published scripts once those were
    shown to produce the same output. SPIRAL and GPSA cannot be: upstream
    main_SPIRAL.py is the broken variant, and upstream main_GPSA.py writes an
    elapsed time instead of the aligned slices. When their working scripts are
    published, this set empties and the assertion below is what tells you.
    """
    from multibench.engine import registry
    absolute = {m for m in mtb.list_methods()
                for v in registry.get(m).variants
                if Path(v.entrypoint).is_absolute()}
    # 0.3.0: GPSA runs through a package driver (engine/drivers/run_gpsa.py).
    # SPIRAL is the one method still tied to the benchmark host: its clean-run
    # port failed after mclust (see methods.yaml). If this set grows, a method
    # has become unrunnable off-host - fix that, not this assertion.
    assert absolute == {"SPIRAL"}, absolute
