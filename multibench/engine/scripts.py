"""The method scripts: where ``run`` finds them (``mtb.config.repo_path``),
how it fetches them, and the commit a run record names."""
from __future__ import annotations

import os
import re
import socket
from pathlib import Path

from .. import _settings
from .._settings import REPO_PATH_VAR, config

#: The commit or tag of the method scripts a fetch checks out; unset = the
#: repository's default branch.
SCRIPTS_REF_VAR = "MULTIBENCH_SCRIPTS_REF"
#: Where the method scripts are fetched from.
SCRIPTS_URL = "https://github.com/PYangLab/scMultiBench.git"
#: How to fetch the method scripts by hand, on a machine with network.
FETCH_SCRIPTS = f"git clone --depth 1 {SCRIPTS_URL} <folder>"


def _scripts_checkout() -> Path | None:
    """The folder holding ``tools_scripts/`` that :func:`ensure_repo` would use, or ``None``."""
    for p in (Path(config.repo_path), _settings._ROOT):
        if (p / "tools_scripts").is_dir():
            return p
    return None


def _git_dir(repo: Path) -> Path | None:
    """``<repo>/.git`` (a worktree's ``.git`` file followed), or ``None``."""
    git = repo / ".git"
    if git.is_file():
        text = git.read_text().strip()
        if not text.startswith("gitdir:"):
            return None
        git = (repo / text.split(":", 1)[1].strip()).resolve()
    return git if git.is_dir() else None


def scripts_commit(repo=None) -> str | None:
    """``git rev-parse HEAD`` of the scripts folder ``repo`` (default: the one
    :func:`ensure_repo` would use), read from ``.git`` without git; ``None`` if not a checkout."""
    repo = _scripts_checkout() if repo is None else Path(repo)
    if repo is None:
        return None
    try:
        git = _git_dir(repo)
        if git is None:
            return None
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head.lower() if re.fullmatch(r"[0-9a-fA-F]{40}", head) else None
        ref = head[4:].strip()
        common = git
        if (git / "commondir").is_file():                  # a linked worktree
            common = (git / (git / "commondir").read_text().strip()).resolve()
        for base in dict.fromkeys((git, common)):
            if (base / ref).is_file():
                return (base / ref).read_text().strip().lower()
        for base in dict.fromkeys((git, common)):
            if (base / "packed-refs").is_file():
                for line in (base / "packed-refs").read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0].lower()
    except OSError:
        return None
    return None


#: A file in the checkout's ``.git`` that records the ref a fetch was asked for.
_REF_RECORD = "multibench-scripts-ref"


def _ref_mismatch(repo: Path, ref: str, *, folder: bool = False) -> str | None:
    """The sentence saying the scripts in ``repo`` are not at ``ref``, or ``None``.

    ``ref`` matches a commit prefix (7+ hex), the ref fetched there, or its ``git
    rev-parse``; a folder that is not a checkout passes."""
    import subprocess
    head = scripts_commit(repo)
    if head is None or (re.fullmatch(r"[0-9a-fA-F]{7,40}", ref)
                        and head.startswith(ref.lower())):
        return None
    git = _git_dir(repo)
    try:
        if (git / _REF_RECORD).read_text().strip() == ref:
            return None
    except OSError:
        pass
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet",
                              f"{ref}^{{commit}}"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip().lower() == head:
            return None
    except (OSError, subprocess.SubprocessError):
        pass
    fix = "Unset the variable, or fetch that ref into a new repo_path."
    what = f"The method scripts in {repo}" if folder else "The method scripts"
    return f"{what} are at {head[:7]}, not {ref} ({SCRIPTS_REF_VAR}). {fix}"


def _check_ref(repo: Path, ref: str) -> None:
    """``RuntimeError`` with :func:`_ref_mismatch`'s sentence."""
    problem = _ref_mismatch(repo, ref, folder=True)
    if problem:
        raise RuntimeError(problem)


def scripts_ref_problem(repo=None) -> str | None:
    """Why the method scripts do not match ``$MULTIBENCH_SCRIPTS_REF``, or ``None``
    (also when it is unset or no scripts are here yet: the first fetch checks it out)."""
    ref = os.environ.get(SCRIPTS_REF_VAR) or None
    repo = _scripts_checkout() if repo is None else Path(repo)
    if ref is None or repo is None or not (repo / "tools_scripts").is_dir():
        return None
    return _ref_mismatch(repo, ref)


def run_provenance(env: str | None, repo=None) -> dict:
    """``scripts_commit``, ``env_flavor`` (from the env's install record, else
    ``'unknown'``) and ``hostname`` for a run record."""
    flavor = None
    if env:
        try:
            from . import envs
            flavor = envs.installed_flavor(env)
        except Exception:  # noqa: BLE001 - a record field must never fail a run
            flavor = None
    return {"scripts_commit": scripts_commit(repo), "env_flavor": flavor or "unknown",
            "hostname": socket.gethostname()}


#: Where a fetch into an existing empty ``repo_path`` clones, inside that folder.
_PARTIAL = ".multibench-fetch.partial"


def _fillable(p: Path) -> bool:
    """Whether a fetch may put the method scripts at ``p``: absent, or empty but for
    the leftover of an interrupted fetch."""
    if not p.exists():
        return True
    try:
        return p.is_dir() and all(e.name == _PARTIAL for e in p.iterdir())
    except OSError:
        return False


def scripts_folder_problem(repo=None) -> str | None:
    """Why a fetch refuses the scripts folder ``repo`` (default ``repo_path``): it holds
    other files. No final period: ``mtb.scan`` joins it into its reason column."""
    p = Path(config.repo_path if repo is None else repo)
    if any((q / "tools_scripts").is_dir() for q in (p, _settings._ROOT)) or _fillable(p):
        return None
    where = "MULTIBENCH_REPO_PATH or mtb.config.repo_path"
    return (f"{p} holds other files and no method scripts. Set {where} to an empty "
            f"or new folder; the first run fetches the scripts into it. If the folder is "
            f"left over from an earlier fetch, you can remove it instead")


def ensure_repo(path=None):
    """Return a directory that contains ``tools_scripts/``, provisioning it if needed.

    Resolution order: the given (or configured) ``repo_path``; the package root
    itself (the merged-repository layout, where ``tools_scripts/`` sits next to
    ``multibench/``); otherwise a one-time shallow fetch of the public
    scMultiBench repository into the configured location, so methods run on a
    fresh machine or Colab, where the package does not carry the upstream
    method scripts.

    The location may be an existing empty folder: it is filled in place and
    keeps its owner and permissions. A folder that holds other files is
    refused (:func:`scripts_folder_problem`); nothing in it is deleted.

    ``$MULTIBENCH_SCRIPTS_REF`` names a commit or tag: a fetch
    checks it out instead of the default branch, and scripts already present
    must be at it (``RuntimeError`` otherwise). A failed fetch raises
    ``RuntimeError`` naming the offline route, and leaves nothing behind.
    """
    import subprocess
    import shutil as _sh

    ref = os.environ.get(SCRIPTS_REF_VAR) or None
    p = Path(path) if path else config.repo_path
    for have in (p, _settings._ROOT):
        if (have / "tools_scripts").is_dir():
            if ref:
                _check_ref(have, ref)
            return have
    problem = scripts_folder_problem(p)
    if problem:
        raise RuntimeError(problem + ".")
    at = f" at {ref}" if ref else ""
    print(f"Fetching the method scripts from PYangLab/scMultiBench{at} into {p} ...",
          flush=True)
    # an existing empty folder is cloned into from the inside, then its
    # contents are moved up: the folder stays, and its parent need not be
    # writable (a folder made for the user on shared storage)
    inside = p.is_dir()
    part = p / _PARTIAL if inside else p.with_name(p.name + ".partial")
    _sh.rmtree(part, ignore_errors=True)
    try:
        if ref is None:
            # -q: in a notebook, git's per-file progress fills the cell
            subprocess.run(["git", "clone", "-q", "--depth", "1", SCRIPTS_URL, str(part)],
                           check=True)
        else:
            # clone --branch takes branches and tags only; fetch also takes a commit
            subprocess.run(["git", "init", "-q", str(part)], check=True)
            subprocess.run(["git", "-C", str(part), "fetch", "-q", "--depth", "1",
                            SCRIPTS_URL, ref], check=True)
            subprocess.run(["git", "-C", str(part), "checkout", "-q", "--detach",
                            "FETCH_HEAD"], check=True)
            if (part / ".git").is_dir():
                (part / ".git" / _REF_RECORD).write_text(ref + "\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        _sh.rmtree(part, ignore_errors=True)
        # run's FAIL line shows the last 200 characters of an error: each
        # text keeps its subject within them ("RuntimeError: " in front, and
        # with a ref of up to 40 characters, a full commit SHA)
        if ref and not isinstance(e, FileNotFoundError):
            raise RuntimeError(
                f"Could not fetch the method scripts at {ref!r}. There is no network, "
                f"or no such commit or tag. Offline, copy a fetched scripts folder and "
                f"set {REPO_PATH_VAR}.") from e
        first = ("Git is not installed. It is needed to fetch the method scripts."
                 if isinstance(e, FileNotFoundError) else
                 "Could not reach github.com to fetch the method scripts.")
        raise RuntimeError(
            f"{first} On a host without network, run {FETCH_SCRIPTS} "
            f"on a connected machine, copy the folder and set {REPO_PATH_VAR}.") from e
    if inside:
        for entry in part.iterdir():
            entry.rename(p / entry.name)
        part.rmdir()
    else:
        part.rename(p)
    return p
