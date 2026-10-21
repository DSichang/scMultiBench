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


import re

#: what only the exporting machine can resolve (the sanitizer that removed these
#: left with the command line in 0.4.0; the shipped lockfiles stay checked)
_LOCAL_PIP_RE = re.compile(r"@\s*file://|feedstock_root")
_CUDA_PIN_RE = re.compile(r"==[0-9][^\s]*\+(cu\d+)")
_CONDA_ONLY_PIP = frozenset({
    "conda", "mamba", "libmambapy", "boa",
    "conda-build", "conda-libmamba-solver", "conda-content-trust",
})


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


# ---- restored from before the 0.4 cut: kept behaviour
def _locks():
    out = []
    from multibench.engine import registry
    for env in sorted({envs.group_for(m) for m in registry.list_methods()}):
        lock = envs.lockfile(env)
        if lock is not None:
            out.append((env, lock))
    return out


def test_no_machine_local_pip_paths():
    bad = []
    for env, lock in _locks():
        for i, ln in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
            if _LOCAL_PIP_RE.search(ln):
                bad.append(f"{env}:{i}: {ln.strip()[:80]}")
    assert not bad, (
        "lockfiles reference paths that exist only on the exporting machine; "
        "pip install will fail with OSError:\n  " + "\n  ".join(bad[:10]))


def test_cuda_pins_declare_their_index():
    """A +cuNNN pin needs the pytorch index, or pip cannot resolve it at all."""
    bad = []
    for env, lock in _locks():
        text = lock.read_text(encoding="utf-8")
        tags = {m.group(1) for m in _CUDA_PIN_RE.finditer(text)}
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
                if name in _CONDA_ONLY_PIP:
                    bad.append(f"{env}:{i}: {stripped[:60]}")
    assert not bad, (
        "conda-only distributions pinned as pip requirements; pip resolves "
        "nothing and the install aborts:\n  " + "\n  ".join(bad))
