"""The method scripts: where ``run`` finds them (``mtb.config.repo_path``),
how it fetches them, and what a run record names them by."""
from __future__ import annotations

import functools
import json
import os
import re
import socket
from pathlib import Path

from .. import _settings
from .._settings import REPO_PATH_VAR, config

#: A commit or tag of PYangLab/scMultiBench: the scripts are then fetched from
#: there with git instead of the package's archive.
SCRIPTS_REF_VAR = "MULTIBENCH_SCRIPTS_REF"
#: Where the scripts of a ``MULTIBENCH_SCRIPTS_REF`` commit are fetched from.
SCRIPTS_URL = "https://github.com/PYangLab/scMultiBench.git"
#: The archive of the package's method scripts: ``url``, ``sha256`` and
#: ``bytes``, written by tools/pack_scripts.py.
_ARCHIVE = Path(__file__).resolve().parent / "scripts_archive.json"
#: A file a download leaves in the scripts folder: the sha256 of the archive.
_ARCHIVE_RECORD = ".multibench-scripts-archive"
#: Where a missing archive of the package's scripts is reported.
ISSUES_URL = "https://github.com/DSichang/scMultiBench/issues"


@functools.lru_cache(maxsize=1)
def scripts_archive() -> dict:
    """``{"url", "sha256", "bytes"}`` of the archive of the package's method scripts."""
    return json.loads(_ARCHIVE.read_text(encoding="utf-8"))


def archive_id() -> str:
    """The first twelve digits of the archive's sha256, which name the package's scripts."""
    return scripts_archive()["sha256"][:12]


def folders(path=None) -> tuple:
    """Where :func:`ensure_repo` looks for ``tools_scripts/``: ``path`` (default
    ``repo_path``), then the package root, which holds them in a repository
    checkout. With ``$MULTIBENCH_SCRIPTS_REF`` set, ``path`` alone."""
    p = Path(config.repo_path if path is None else path)
    return (p,) if os.environ.get(SCRIPTS_REF_VAR) else (p, _settings._ROOT)


def _scripts_checkout() -> Path | None:
    """The folder holding ``tools_scripts/`` that :func:`ensure_repo` would use, or ``None``."""
    return next((p for p in folders() if (p / "tools_scripts").is_dir()), None)


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


def downloaded(repo) -> str | None:
    """``tools_scripts-<id>``, the archive the scripts folder ``repo`` was
    downloaded from, or ``None``."""
    try:
        sha = (Path(repo) / _ARCHIVE_RECORD).read_text().strip()
    except OSError:
        return None
    return f"tools_scripts-{sha[:12]}" if sha else None


#: A file in the checkout's ``.git`` that records the ref a fetch was asked for.
_REF_RECORD = "multibench-scripts-ref"


def _ref_mismatch(repo: Path, ref: str, *, folder: bool = False) -> str | None:
    """The sentence saying the scripts in ``repo`` are not at ``ref``, or ``None``.

    ``ref`` matches a commit prefix (7+ hex), the ref fetched there, or its ``git
    rev-parse``; the package's downloaded scripts never match, and a folder that is
    neither passes."""
    import subprocess
    head = scripts_commit(repo)
    fix = "Unset the variable, or fetch that ref into a new repo_path."
    what = f"The method scripts in {repo}" if folder else "The method scripts"
    if head is None:
        archive = downloaded(repo)
        return archive and f"{what} are {archive}, not {ref} ({SCRIPTS_REF_VAR}). {fix}"
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", ref) and head.startswith(ref.lower()):
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


def run_provenance(env: str | None) -> dict:
    """``scripts_commit`` (the git commit of the scripts, else the archive they
    were downloaded from), ``env_flavor`` (from the env's install record, else
    ``'unknown'``) and ``hostname`` for a run record."""
    flavor = None
    if env:
        try:
            from . import envs
            flavor = envs.installed_flavor(env)
        except Exception:  # noqa: BLE001 - a record field must never fail a run
            flavor = None
    repo = _scripts_checkout()
    return {"scripts_commit": scripts_commit(repo) or (repo and downloaded(repo)),
            "env_flavor": flavor or "unknown", "hostname": socket.gethostname()}


#: Where a fetch into an existing empty ``repo_path`` writes, inside that folder.
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
    if any((q / "tools_scripts").is_dir() for q in folders(p)) or _fillable(p):
        return None
    return (f"{p} holds other files and no method scripts. Set mtb.config.repo_path to "
            f"an empty or new folder")


def ensure_repo(path=None):
    """A folder with ``tools_scripts/``: ``path`` (default ``repo_path``), the
    package root, else the package's scripts downloaded into ``path`` and
    checked against their sha256. An empty folder is filled in place; one with
    other files is refused. ``$MULTIBENCH_SCRIPTS_REF`` fetches that commit of
    PYangLab/scMultiBench with git instead. A failed fetch raises
    ``RuntimeError`` and leaves nothing."""
    ref = os.environ.get(SCRIPTS_REF_VAR) or None
    p = Path(path) if path else config.repo_path
    for have in folders(p):
        if (have / "tools_scripts").is_dir():
            if ref:
                _check_ref(have, ref)
            return have
    problem = scripts_folder_problem(p)
    if problem:
        raise RuntimeError(problem + ".")
    # an existing empty folder is filled from the inside, then its contents
    # are moved up: the folder stays, and its parent need not be writable (a
    # folder made for the user on shared storage)
    inside = p.is_dir()
    part = p / _PARTIAL if inside else p.with_name(p.name + ".partial")
    if ref:
        _clone(part, ref)
    else:
        _download(part)
    if inside:
        for entry in part.iterdir():
            entry.rename(p / entry.name)
        part.rmdir()
    else:
        part.rename(p)
    return p


#: What a host without network copies instead of fetching.
_FOLDER = f"the scripts folder from a machine with network and set {REPO_PATH_VAR}."


def _archive_bytes(url: str) -> bytes:
    """The bytes at ``url``; ``OSError`` when it cannot be reached."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _download(part: Path) -> None:
    """Download the package's scripts, check their sha256 and unpack them into
    ``part``; a failure removes ``part`` and raises ``RuntimeError``."""
    import hashlib
    import io
    import shutil
    import tarfile
    from .._data.fetch import safe_extract

    shutil.rmtree(part, ignore_errors=True)
    import urllib.error

    rec = scripts_archive()
    try:
        data = _archive_bytes(rec["url"])
    except urllib.error.HTTPError as e:
        from .. import __version__
        raise RuntimeError(f"The method scripts of multibench {__version__} are not at "
                           f"{rec['url']} (HTTP {e.code}). Report it at {ISSUES_URL}.") from e
    except OSError as e:
        raise RuntimeError(f"Could not reach github.com to fetch the method scripts. "
                           f"Copy {_FOLDER}") from e
    if hashlib.sha256(data).hexdigest() != rec["sha256"]:
        raise RuntimeError("The downloaded method scripts are not the ones this multibench "
                           "version names. Run the call again.")
    try:
        part.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            safe_extract(t, part)
        (part / _ARCHIVE_RECORD).write_text(rec["sha256"] + "\n")
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise


def _clone(part: Path, ref: str) -> None:
    """Fetch the scripts of PYangLab/scMultiBench at ``ref`` into ``part`` with
    git; a failure removes ``part`` and raises ``RuntimeError``."""
    import shutil
    import subprocess

    shutil.rmtree(part, ignore_errors=True)
    try:
        # clone --branch takes branches and tags only; fetch also takes a commit
        subprocess.run(["git", "init", "-q", str(part)], check=True)
        subprocess.run(["git", "-C", str(part), "fetch", "-q", "--depth", "1",
                        SCRIPTS_URL, ref], check=True)
        subprocess.run(["git", "-C", str(part), "checkout", "-q", "--detach",
                        "FETCH_HEAD"], check=True)
        if (part / ".git").is_dir():
            (part / ".git" / _REF_RECORD).write_text(ref + "\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        shutil.rmtree(part, ignore_errors=True)
        if isinstance(e, FileNotFoundError):
            raise RuntimeError(f"Git is not installed. Install git, or copy {_FOLDER}") from e
        raise RuntimeError(f"Could not fetch the method scripts at {ref!r}: no "
                           f"network, or no such commit or tag.") from e
