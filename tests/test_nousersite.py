"""Provisioning must not inherit packages from ~/.local/lib/pythonX/site-packages.

pip treats a package importable from user site as already satisfied and skips
installing it into the target env. The build then reports success while
producing an env that only works if user-site leakage is allowed - and run()
correctly sets PYTHONNOUSERSITE=1, so the method fails at dispatch.

Six of the 29 envs were built short of 32 packages this way. VIPCCA died with
ModuleNotFoundError on astunparse, a package its own lockfile pins.
"""
import os

from multibench.engine import envs


def test_run_all_blocks_user_site(monkeypatch):
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    seen = {}

    def fake(cmd, **kw):
        seen["v"] = os.environ.get("PYTHONNOUSERSITE")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(envs.subprocess, "run", fake)
    envs._run_all([["true"]])
    assert seen.get("v") == "1", (
        "provisioning ran without PYTHONNOUSERSITE=1, so pip can skip packages "
        "it finds in ~/.local and silently under-populate the env")


def test_matilda_has_a_post_install_for_its_own_package():
    """matilda is an editable install in the working env, so freeze() skips it
    and a lockfile-only rebuild fails with 'No module named matilda'."""
    post = envs.post_install("matilda")
    assert post is not None and post.is_file()
    body = post.read_text(encoding="utf-8")
    # check the EXECUTABLE lines, not the whole file: the script documents in a
    # comment why the private remote is avoided, and that comment must not fail
    installs = [ln for ln in body.splitlines() if ln.strip().startswith("pip install")]
    assert installs, "post-install script does not install anything"
    assert any("matilda-sc==" in ln for ln in installs), installs
    assert not any("github.com/DSichang" in ln for ln in installs), (
        f"installs from the private checkout's remote, which a new user cannot "
        f"clone: {installs}")
