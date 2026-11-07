"""P05 (study 2): method envs are linux-64 - refuse BEFORE any download or
build on another host, ship archive sizes, explain the difficulty tags; and
Rin's ``as_frame=`` on the three env tables.

Every test pins the platform through ``envs.host_system`` so it
holds on Linux and macOS alike.
"""
import json
import os
import sys
from pathlib import Path

import pytest

from multibench.engine import envs
from _serve import serve_archives

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def off_linux(monkeypatch):
    monkeypatch.setattr(envs, "host_system", lambda: "macOS")


@pytest.fixture
def on_linux(monkeypatch):
    monkeypatch.setattr(envs, "host_system", lambda: None)


# ----------------------------------------------------------------- the guard
@pytest.mark.parametrize("platform,name", [("linux", None), ("darwin", "macOS"),
                                           ("win32", "Windows"), ("cygwin", "Windows"),
                                           ("freebsd14", "freebsd14")])
def test_host_system_reflects_sys_platform(monkeypatch, platform, name):
    monkeypatch.setattr(sys, "platform", platform)
    assert envs.host_system() == name


def test_install_refuses_off_linux_before_any_download_or_build(off_linux, monkeypatch):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build was started"))
    serve_archives(monkeypatch,
                   lambda *a, **k: pytest.fail("download started on a non-Linux host"))
    for build in (False, True):
        with pytest.raises(OSError, match="^Methods run only on Linux, and this computer "
                                          "runs macOS. Run the install on a Linux machine"):
            envs.install(["Matilda"], build=build)
    # the dry run still plans everywhere
    rows = envs.create_all(methods=["Matilda"], dry_run=True)
    assert rows and rows[0]["env"] == "matilda" and rows[0]["exists"] is False


def test_create_all_run_passes_on_linux(on_linux, monkeypatch):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    ran = []
    monkeypatch.setattr(envs, "_run_all", lambda cmds: ran.append(cmds))
    envs.create_all(methods=["Matilda"], dry_run=False)
    assert ran


def test_builders_dry_run_everywhere(off_linux):
    assert envs.create_env("matilda")                  # commands, no build


# ----------------------------------------------------------------- sizes
def test_every_env_has_a_lockfile_and_a_packed_archive():
    # install plans only "packed archive published" or a lockfile "BUILD": no
    # state, and no refusal, for an env without either
    envs_needed = [env for env, _ in envs._envs_needed(None, None)]
    assert [e for e in envs_needed if envs.lockfile(e) is None] == []
    assert [e for e in envs_needed if e not in envs.packed_manifest()] == []


def test_packed_sizes_cover_every_packed_url():
    urls = json.loads((ROOT / "multibench/engine/packed_urls.json").read_text())
    sizes = envs.packed_sizes()
    assert set(urls) <= set(sizes), set(urls) - set(sizes)
    for env, sz in sizes.items():
        assert set(sz) == {"archive_bytes", "unpacked_bytes"}, env
        for k, v in sz.items():
            assert v is None or (isinstance(v, int) and v > 0), (env, k, v)
    # the evidenced HEAD sizes are shipped
    assert sizes["scmb_r"]["archive_bytes"] == 916953088
    assert sizes["env_sciPENN"]["archive_bytes"] == 2077412003
    assert "_meta" not in sizes


def test_packed_sizes_json_is_package_data():
    """The snapshot is read at runtime, so the wheel must ship it."""
    import fnmatch
    import re
    text = (ROOT / "pyproject.toml").read_text()
    globs = re.findall(r'"([^"]+)"', re.search(r"multibench\s*=\s*\[(.*?)\]", text, re.S).group(1))
    assert any(fnmatch.fnmatch("engine/packed_sizes.json", g) for g in globs), globs


def test_gb_formatting():
    assert envs._gb(3215645570) == "3.2 GB"
    assert envs._gb(None) == "?" and envs._gb("x") == "?"


def test_packed_sizes_tool_counts_an_env_on_a_layer_without_the_layer(tmp_path):
    """An env built on a layer counts its own files, not the layer's files
    hard-linked into it; the CPU build's key is measured from a flavor='cpu'
    install, which uses the env's own name."""
    sys.path.insert(0, str(ROOT / "tools"))
    import packed_sizes as tool
    root = tmp_path / "envs"
    (root / ".layers" / "torch").mkdir(parents=True)
    (root / ".layers" / "torch" / "lib.so").write_bytes(b"x" * 100)
    (root / "a" / "lib").mkdir(parents=True)
    os.link(root / ".layers" / "torch" / "lib.so", root / "a" / "lib" / "lib.so")
    (root / "a" / "own.py").write_bytes(b"y" * 7)
    os.link(root / "a" / "own.py", root / "a" / "again.py")

    class _Down:
        def open(self, req):
            raise OSError("offline")
    urls = {"a": "https://x/a.tar.gz", "a-cpu": "https://x/a-cpu.tar.gz",
            "b": "https://x/b.tar.gz"}
    gpu = tool.build(urls, envs_dir=root, opener=_Down())
    assert gpu["a"]["unpacked_bytes"] == 7 and gpu["a-cpu"]["unpacked_bytes"] is None
    cpu = tool.build(urls, previous=gpu, envs_dir=root, opener=_Down(), flavor="cpu")
    assert cpu["a"]["unpacked_bytes"] == 7 and cpu["a-cpu"]["unpacked_bytes"] == 7
    assert cpu["b"]["unpacked_bytes"] is None


def test_packed_sizes_tool_builds_from_head_requests(tmp_path):
    """tools/packed_sizes.py with an injected opener: Content-Length in,
    archive_bytes out; a failing URL keeps the previous value, never zero."""
    sys.path.insert(0, str(ROOT / "tools"))
    import packed_sizes as tool

    class _Resp:
        def __init__(self, n):
            self.headers = {"Content-Length": str(n)}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req):
            if "bad" in req.full_url:
                raise OSError("boom")
            return _Resp(1234)
    urls = {"good": "https://x/good.tar.gz", "bad": "https://x/bad.tar.gz"}
    prev = {"bad": {"archive_bytes": 99, "unpacked_bytes": None}}
    table = tool.build(urls, previous=prev, opener=_Opener(), today="2026-09-03")
    assert table["good"] == {"archive_bytes": 1234, "unpacked_bytes": None}
    assert table["bad"] == {"archive_bytes": 99, "unpacked_bytes": None}
    assert table["_meta"]["measured"] == "2026-09-03"
    # --check against the shipped file is byte-stable (no network: opener fails
    # on every URL, so every value falls back to the shipped one)
    shipped = json.loads((ROOT / "multibench/engine/packed_sizes.json").read_text())
    real_urls = json.loads((ROOT / "multibench/engine/packed_urls.json").read_text())

    class _Down:
        def open(self, req):
            raise OSError("offline")
    again = tool.build(real_urls, previous=shipped, opener=_Down())
    assert {k: v for k, v in again.items() if k != "_meta"} == \
        {k: v for k, v in shipped.items() if k != "_meta"}


