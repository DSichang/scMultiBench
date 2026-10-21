"""Committed lockfiles must be installable on a machine that is not this one.

`conda env export` records entries that resolve only on the exporting host, so a
lockfile can be complete, substantive, and still fail every single install:

  * `pkg @ file:///home/conda/feedstock_root/build_artifacts/...` - conda
    packages pip merely observed. Install aborts with
    `OSError: [Errno 2] No such file or directory`.
  * `torch==2.6.0+cu118` - published on download.pytorch.org, never on PyPI.
    Install aborts with `No matching distribution found`.

Several committed lockfiles carried one of these and could not be built
from scratch. The pre-existing test only checked a lockfile was long enough,
which is why it passed on all of them. This checks the property that actually
matters, statically - a real build is far too slow for the suite.
"""

import pytest

from multibench.engine import envs


@pytest.mark.parametrize("raw,expect_gone,expect_index", [
    ("name: e\ndependencies:\n  - pip:\n      - argcomplete @ file:///home/conda/x\n",
     "file://", None),
    ("name: e\ndependencies:\n  - pip:\n      - torch==2.6.0+cu118\n",
     None, "download.pytorch.org/whl/cu118"),
    # freeze()'s fallback path indents four spaces, conda's export six - a
    # hard-coded width silently skipped half the files
    ("name: e\ndependencies:\n  - pip:\n    - torch==1.13.1+cu117\n",
     None, "download.pytorch.org/whl/cu117"),
])
def test_sanitizer_handles_both_indent_styles(raw, expect_gone, expect_index):
    got = envs.sanitize_lock(raw)
    if expect_gone:
        assert expect_gone not in got, got
    if expect_index:
        assert expect_index in got, got


def test_create_env_runs_any_post_install_script():
    """A conda lockfile cannot capture install.packages()/install_github()
    packages. scmb_r's rliger is one, so a lockfile-only rebuild produced an env
    that built cleanly and then failed UINMF with 'no package called rliger'."""
    post = envs.post_install("scmb_r")
    assert post is not None, "scmb_r needs a post-install script to restore rliger"
    assert post.is_file()
    body = post.read_text(encoding="utf-8")
    assert "rliger" in body

    cmds = envs.create_env("scmb_r", dry_run=True)
    flat = " ".join(" ".join(c) for c in cmds)
    assert post.name in flat, f"create_env would not run {post.name}: {flat}"


def test_env_without_post_install_has_none():
    assert envs.post_install("scmb_scvi") is None
