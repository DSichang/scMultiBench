"""``mtb.env.status`` refuses a method selection passed as ``conda``.

``status``'s only positional parameter is the conda/mamba executable, so the
natural ``status("Matilda")`` was read as a binary (every method's row back,
no error) and ``status(["StabMap", "Matilda"])`` failed with ``unhashable
type: 'list'`` inside the conda probe. Both now raise at the call and name
the call that takes methods. A real executable, by path or by name on PATH,
is still asked, and ``status()`` without arguments is unchanged (pinned by
the other env tests).
"""
from __future__ import annotations

import json
import stat

import pytest

import multibench as mtb
from multibench import config
from multibench.engine import envs


@pytest.fixture
def fake_conda(tmp_path, monkeypatch):
    """An executable ``conda`` stub whose ``env list --json`` reports Matilda's
    env; ``envs_dir`` is empty, so only the stub can make that env exist."""
    env = envs.group_for("Matilda")
    exe = tmp_path / "bin" / "conda"
    exe.parent.mkdir()
    listing = json.dumps({"envs": [str(tmp_path / "prefixes" / env)]})
    exe.write_text(f"#!/bin/sh\necho '{listing}'\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    empty = tmp_path / "envs"
    empty.mkdir()
    monkeypatch.setattr(config.DEFAULT, "envs_dir", empty)
    envs._conda_prefixes.cache_clear()
    yield exe
    envs._conda_prefixes.cache_clear()


def test_a_method_id_is_refused_and_the_message_names_doctor():
    with pytest.raises(ValueError, match=r"mtb\.env\.doctor\(methods=\['Matilda'\]\)"):
        mtb.env.status("Matilda")


def test_a_method_id_in_another_case_names_the_registry_id():
    with pytest.raises(ValueError, match=r"mtb\.env\.doctor\(methods=\['StabMap'\]\)"):
        mtb.env.status("stabmap")


def test_a_list_of_method_ids_is_refused_and_the_message_names_doctor():
    with pytest.raises(TypeError,
                       match=r"mtb\.env\.doctor\(methods=\['StabMap', 'Matilda'\]\)"):
        mtb.env.status(["StabMap", "Matilda"])


def test_as_frame_passed_positionally_is_refused():
    with pytest.raises(TypeError, match="conda"):
        mtb.env.status(True)


def test_a_name_not_on_path_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match="not an executable"):
        mtb.env.status("condaa")


def test_the_executable_given_by_path_is_asked(fake_conda):
    for conda in (fake_conda, str(fake_conda)):
        rows = {r["method"]: r for r in mtb.env.status(conda)}
        assert rows["Matilda"]["exists"] is True
        assert len(rows) == len(mtb.list_methods())


def test_the_executable_given_by_name_on_path_is_asked(fake_conda, monkeypatch):
    monkeypatch.setenv("PATH", str(fake_conda.parent))
    rows = {r["method"]: r for r in mtb.env.status("conda")}
    assert rows["Matilda"]["exists"] is True
