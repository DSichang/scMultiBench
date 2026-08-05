"""Committed lockfiles must be installable on a machine that is not this one.

`conda env export` records entries that resolve only on the exporting host, so a
lockfile can be complete, substantive, and still fail every single install:

  * `pkg @ file:///home/conda/feedstock_root/build_artifacts/...` - conda
    packages pip merely observed. Install aborts with
    `OSError: [Errno 2] No such file or directory`.
  * `torch==2.6.0+cu118` - published on download.pytorch.org, never on PyPI.
    Install aborts with `No matching distribution found`.

Six of the 29 committed lockfiles carried one of these and could not be built
from scratch. The pre-existing test only checked a lockfile was long enough,
which is why it passed on all six. This checks the property that actually
matters, statically - a real build is far too slow for the suite.
"""

import pytest

from multibench.engine import envs


def _locks():
    out = []
    for env in sorted(set(envs.required_envs())):
        lock = envs.lockfile(env)
        if lock is not None:
            out.append((env, lock))
    return out


def test_no_machine_local_pip_paths():
    bad = []
    for env, lock in _locks():
        for i, ln in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
            if envs._LOCAL_PIP_RE.search(ln):
                bad.append(f"{env}:{i}: {ln.strip()[:80]}")
    assert not bad, (
        "lockfiles reference paths that exist only on the exporting machine; "
        "pip install will fail with OSError:\n  " + "\n  ".join(bad[:10]))


def test_cuda_pins_declare_their_index():
    """A +cuNNN pin needs the pytorch index, or pip cannot resolve it at all."""
    bad = []
    for env, lock in _locks():
        text = lock.read_text(encoding="utf-8")
        tags = {m.group(1) for m in envs._CUDA_PIN_RE.finditer(text)}
        for tag in sorted(tags):
            if f"download.pytorch.org/whl/{tag}" not in text:
                bad.append(f"{env}: pins +{tag} but declares no --extra-index-url for it")
    assert not bad, "\n  ".join([""] + bad)


def test_no_conda_only_packages_in_pip_sections():
    """`pip freeze` inside a conda env reports conda's own machinery, which pip
    cannot supply. scmb_r pinned `conda==23.3.1` and the whole install aborted."""
    bad = []
    for env, lock in _locks():
        in_pip, pip_indent = False, None
        for i, ln in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
            stripped, indent = ln.strip(), len(ln) - len(ln.lstrip())
            if stripped == "- pip:":
                in_pip, pip_indent = True, indent
                continue
            if in_pip and stripped:
                if indent <= pip_indent:
                    in_pip = False
                    continue
                import re as _re
                name = _re.split(r"[=<>!~\s\[]", stripped[2:].strip())[0].lower()
                if name in envs._CONDA_ONLY_PIP:
                    bad.append(f"{env}:{i}: {stripped[:60]}")
    assert not bad, (
        "conda-only distributions pinned as pip requirements; pip resolves "
        "nothing and the install aborts:\n  " + "\n  ".join(bad))


def test_sanitize_lock_is_idempotent():
    """Committed lockfiles must already be sanitised - re-running changes nothing."""
    dirty = [env for env, lock in _locks()
             if envs.sanitize_lock(lock.read_text(encoding="utf-8"))
             != lock.read_text(encoding="utf-8")]
    assert not dirty, f"lockfiles still need sanitising: {dirty}"


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
