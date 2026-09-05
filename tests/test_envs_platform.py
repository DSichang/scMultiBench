"""P05 (study 2): method envs are linux-64 - refuse BEFORE any download or
build on another host, ship archive sizes, explain the difficulty tags; and
Rin's ``as_frame=`` on the three env tables.

Every test pins the platform through ``envs.host_platform_problem`` so it
holds on Linux and macOS alike.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from multibench.engine import envs, registry

ROOT = Path(__file__).resolve().parents[1]
DARWIN = "method environments are linux-64 conda envs (packed archives + lockfiles); this host is darwin/arm64"


@pytest.fixture
def off_linux(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: DARWIN)


@pytest.fixture
def on_linux(monkeypatch):
    monkeypatch.setattr(envs, "host_platform_problem", lambda: None)


# ----------------------------------------------------------------- the guard
def test_host_platform_problem_reflects_sys_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert envs.host_platform_problem() is None
    monkeypatch.setattr(sys, "platform", "darwin")
    msg = envs.host_platform_problem()
    assert msg and "linux-64" in msg and "darwin" in msg


def test_create_all_run_refuses_off_linux(off_linux, monkeypatch):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build was started"))
    with pytest.raises(RuntimeError, match="linux-64") as e:
        envs.create_all(methods=["Matilda"], dry_run=False)
    assert "force=True / --force" in str(e.value)
    # the dry run still plans everywhere
    rows = envs.create_all(methods=["Matilda"], dry_run=True)
    assert rows and rows[0]["env"] == "matilda" and rows[0]["exists"] is False


def test_create_all_force_skips_the_guard(off_linux, monkeypatch):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    ran = []
    monkeypatch.setattr(envs, "_run_all", lambda cmds: ran.append(cmds))
    envs.create_all(methods=["Matilda"], dry_run=False, force=True)
    assert ran, "force=True must reach the build"


def test_create_all_run_passes_on_linux(on_linux, monkeypatch):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    ran = []
    monkeypatch.setattr(envs, "_run_all", lambda cmds: ran.append(cmds))
    envs.create_all(methods=["Matilda"], dry_run=False)
    assert ran


def test_install_packed_refuses_off_linux_before_download(off_linux, monkeypatch, tmp_path):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda *a, **k: pytest.fail("download started on a non-Linux host"))
    with pytest.raises(RuntimeError, match="linux-64"):
        envs.install_packed("scmb_r")
    # force=True reaches the download (stubbed here to fail loudly)
    monkeypatch.setattr(envs, "_conda_bin", lambda prefer="mamba": "conda")
    monkeypatch.setattr(envs.shutil, "which", lambda c: "/usr/bin/conda")
    monkeypatch.setattr(envs, "_envs_dir", lambda b: Path("/nonexistent/envs"))
    monkeypatch.setenv("MULTIBENCH_ENVS_DIR", str(tmp_path / "no-envs"))   # no prefix on disk ...
    monkeypatch.setattr(envs, "env_prefix", lambda env, conda=None: None)  # ... and none via conda
    reached = []

    def _retrieve(url, *a, **k):
        reached.append(url)
        raise urllib.error.HTTPError(url, 404, "nope", {}, None)
    import urllib.error
    monkeypatch.setattr(urllib.request, "urlretrieve", _retrieve)
    assert envs.install_packed("scmb_r", force=True, envs_dir=tmp_path / "no-envs") is False
    assert reached and reached[0].endswith("scmb_r.tar.gz")


@pytest.mark.parametrize("fn", [
    lambda: envs.create("Matilda", dry_run=False),
    lambda: envs.create_group("scmb_r", dry_run=False),
    lambda: envs.create_env("matilda", dry_run=False),
])
def test_every_builder_refuses_off_linux(off_linux, monkeypatch, fn):
    monkeypatch.setattr(envs, "_run_all", lambda cmds: pytest.fail("a build was started"))
    with pytest.raises(RuntimeError, match="linux-64"):
        fn()


def test_builders_dry_run_everywhere(off_linux):
    assert envs.create("Matilda")                      # commands, no build
    assert envs.create_env("matilda")


# ----------------------------------------------------------------- vocabulary
def test_difficulty_vocabulary_covers_every_tag_in_env_specs():
    tags = {r["difficulty"] for r in envs.status()}
    assert tags <= set(envs.DIFFICULTY), tags - set(envs.DIFFICULTY)
    for tag in ("easy", "old-scvi", "old-tensorflow", "R", "verified", "blocked-script"):
        assert tag in envs.DIFFICULTY and len(envs.DIFFICULTY[tag]) > 20
    assert "verified_working" in envs.VERIFIED_STAR


# ----------------------------------------------------------------- sizes
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


# ----------------------------------------------------------------- as_frame
def test_env_tables_as_frame(monkeypatch):
    monkeypatch.setattr(envs, "installed_envs", lambda conda=None: [])
    for fn, kw in ((envs.plan, {"category": "vertical"}),
                   (envs.doctor, {"category": "vertical"}),
                   (envs.status, {})):
        rows = fn(**kw)
        assert isinstance(rows, list) and isinstance(rows[0], dict)     # default kept
        df = fn(**kw, as_frame=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == list(rows[0]) and len(df) == len(rows)
    assert "availability" in envs.plan(category="cross")[0]


def test_env_docstrings_have_parameters_and_returns():
    import inspect
    for fn in (envs.plan, envs.doctor, envs.status, envs.create_all, envs.create_env,
               envs.create, envs.create_group, envs.install_packed, envs.packed_sizes,
               envs.host_platform_problem):
        doc = inspect.getdoc(fn)
        assert doc and "Returns" in doc, fn.__name__
        assert doc.splitlines()[0].strip().endswith((".", "?")), fn.__name__
    assert "multibench env install --run" in inspect.getdoc(envs.doctor)
    assert "availability" in inspect.getdoc(envs.plan)
