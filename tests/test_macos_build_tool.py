"""The macOS build tool (tools/macos/) against the package: its env list, its hand files and the rules
it applies to a Linux lock. Runs anywhere: nothing is built and conda is not called."""
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from multibench.engine import envs, registry

ROOT = Path(__file__).resolve().parent.parent
MAC = ROOT / "tools" / "macos"
LOCKS = ROOT / "multibench" / "engine" / "env_locks"
HAND_FILES = {"unpin.txt", "drop.txt", "extra.txt", "soft.txt", "pip_overrides.txt", "force_source.txt",
              "build_constraints.txt", "post_macos.sh", "Makevars", "uv.txt"}


def _tool():
    spec = importlib.util.spec_from_file_location("macos_build", MAC / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ENVS = yaml.safe_load((MAC / "envs.yaml").read_text())


def test_each_listed_method_is_served_by_that_env_on_linux_too():
    for env, methods in ENVS.items():
        assert (LOCKS / f"{env}.yml").exists(), f"{env}: no Linux lock"
        for m in methods:
            registry.get(m)
            assert envs.group_for(m) == env, f"{m} runs in {envs.group_for(m)}, not {env}"


def test_the_package_lists_the_archives_of_these_envs():
    """engine/packed_macos.json (written by `build.py listing`) serves what envs.yaml says."""
    import json
    listing = json.loads((ROOT / "multibench" / "engine" / "packed_macos.json").read_text())
    assert listing["base"] == _tool().BASE
    assert {k: v["methods"] for k, v in listing["archives"].items()} == ENVS


def test_a_method_is_listed_once():
    listed = [m for ms in ENVS.values() for m in ms]
    assert len(listed) == len(set(listed))


def test_hand_files_belong_to_listed_envs_and_are_known():
    for d in (MAC / "envs").iterdir():
        assert d.name in ENVS, f"tools/macos/envs/{d.name} is not in envs.yaml"
        unknown = {f.name for f in d.iterdir()} - HAND_FILES
        assert not unknown, f"{d.name}: files build.py does not read: {unknown}"


def test_hand_files_name_what_the_lock_has():
    tool = _tool()
    for env in ENVS:
        hand = MAC / "envs" / env
        conda_yaml, pips = envs.split_lock((LOCKS / f"{env}.yml").read_text())
        names = set(re.findall(r"^\s+- ([A-Za-z0-9_.\-]+)=", conda_yaml, re.M))
        for f in ("unpin.txt", "drop.txt"):
            missing = set(tool.lines(hand / f)) - names
            assert not missing, f"{env}/{f}: not in the lock's conda part: {missing}"
        for f in ("pip_overrides.txt", "force_source.txt", "uv.txt"):
            pins = [ln.split()[0] for ln in tool.lines(hand / f)]
            missing = set(pins) - set(pips)
            assert not missing, f"{env}/{f}: not a pin of the lock: {missing}"


def test_wheels_are_for_listed_envs_and_pins_of_their_locks():
    for pin, spec in yaml.safe_load((MAC / "wheels.yaml").read_text()).items():
        for env in spec["for"]:
            assert env in ENVS
            _conda, pips = envs.split_lock((LOCKS / f"{env}.yml").read_text())
            assert pin in pips, f"{pin} is not a pin of {env}"


def test_pip_part_drops_linux_only_pins_and_local_tags():
    tool = _tool()
    keep, left, changed = tool.pip_part(
        ["numpy==1.24.4", "nvidia-cudnn-cu12==9.1.0.70", "triton==3.0.0", "cuda-bindings==13.3.1",
         "torch==2.4.1+cu121", "louvain==0.7.1"],
        {"louvain==0.7.1": "louvain==0.7.2"})
    assert keep == ["numpy==1.24.4", "torch==2.4.1", "louvain==0.7.2"]
    assert left == ["nvidia-cudnn-cu12==9.1.0.70", "triton==3.0.0", "cuda-bindings==13.3.1"]
    assert len(changed) == 2


def test_pip_override_can_leave_a_pin_out():
    keep, _left, _changed = _tool().pip_part(["a==1", "b==2"], {"b==2": "-"})
    assert keep == ["a==1"]


@pytest.mark.parametrize("dep,version,ok", [
    ("r-base >=4.5,<4.6.0a0", "4.5.3", True),
    ("r-base >=4.5,<4.6.0a0", "4.4.3", False),
    ("r-base 4.2.*", "4.2.3", True),
    ("r-base", "4.2.3", True),
    ("r-base >=4.4,<4.5.0a0|>=4.5,<4.6.0a0", "4.5.1", True),
])
def test_admits(dep, version, ok):
    assert _tool().admits(dep, version) is ok


def test_hand_lines_skip_comments(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("# why\nr-devtools\n\n  # more\nr-x\n")
    assert _tool().lines(f) == ["r-devtools", "r-x"]
    assert _tool().lines(tmp_path / "absent.txt") == []
