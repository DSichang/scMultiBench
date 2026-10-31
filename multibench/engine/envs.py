"""Method environments: check and install the conda envs the methods run in.

Methods whose dependencies conflict run in separate envs; methods that fit
together share one. ``mtb.method_info(m)['env']`` names a method's env.

``mtb.env.install(methods)`` installs the envs a selection of methods needs:
a prebuilt archive when one exists, else a build from the pinned package
list. ``dry_run=True`` shows each env's state and size and installs nothing.

The envs are Linux-only.
"""
from __future__ import annotations

import functools
import json as _json
import shutil
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from .. import _text

from .._settings import config
from . import registry


# --- host platform --------------------------------------------------------
def host_system() -> str | None:
    """``None`` on Linux (WSL included), else this computer's system for users:
    ``'macOS'``, ``'Windows'`` or the platform name. The envs are linux-64."""
    if sys.platform == "linux":
        return None
    return {"darwin": "macOS", "win32": "Windows", "cygwin": "Windows"}.get(
        sys.platform, sys.platform)


def linux_only_text() -> str | None:
    """``"Methods run only on Linux, and this computer runs macOS."``, or
    ``None`` on Linux."""
    system = host_system()
    return system and f"Methods run only on Linux, and this computer runs {system}."


#: The first sentence of the install error without conda (the docs quote it).
NO_CONDA = "Conda is not installed on this computer."


_SIZES_JSON = Path(__file__).resolve().parent / "packed_sizes.json"


@functools.lru_cache(maxsize=1)
def packed_sizes() -> dict:
    """Byte sizes of the published packed archives, per env.

    Read from the shipped ``engine/packed_sizes.json`` (written by
    ``tools/packed_sizes.py``); no request is made at runtime.

    Returns
    -------
    dict
        ``{env: {"archive_bytes": int | None, "unpacked_bytes": int | None}}``;
        ``None`` means "not measured yet". An unpacked size below the archive
        size cannot be right and reads ``None``.
    """
    data = _json.loads(_SIZES_JSON.read_text())
    out = {}
    for k, v in data.items():
        if k.startswith("_"):                       # metadata
            continue
        archive, unpacked = v.get("archive_bytes"), v.get("unpacked_bytes")
        if (isinstance(archive, (int, float)) and isinstance(unpacked, (int, float))
                and unpacked < archive):
            v = {**v, "unpacked_bytes": None}
        out[k] = v
    return out


def _gb(n) -> str:
    """``3215645570`` -> ``"3.2 GB"``; ``None`` / non-numeric -> ``"?"``."""
    try:
        if n is None:
            return "?"
        return f"{float(n) / 1e9:.1f} GB"
    except (TypeError, ValueError):
        return "?"


# --- archive flavours (CPU-only vs CUDA builds) ----------------------------
#: The accepted ``flavor=`` values of :func:`install_packed` / :func:`install`.
FLAVORS = ("auto", "cpu", "gpu")

#: The file in an unpacked prefix that names the archive it came from.
FLAVOR_FILE = ".multibench_flavor"

#: Present when an NVIDIA driver is loaded, even without ``nvidia-smi`` on PATH.
_NVIDIA_PROC = Path("/proc/driver/nvidia/version")


def check_flavor(flavor) -> str:
    """``flavor`` when it is one of :data:`FLAVORS`, else ``ValueError``."""
    if flavor not in FLAVORS:
        raise ValueError(f"flavor={flavor!r}: choose one of "
                         + ", ".join(repr(f) for f in FLAVORS))
    return flavor


@functools.lru_cache(maxsize=1)
def host_has_gpu() -> bool:
    """Whether an NVIDIA GPU is visible: ``nvidia-smi -L`` lists a device, or
    the driver is loaded (``/proc/driver/nvidia/version``). Cached per process."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            proc = subprocess.run([smi, "-L"], capture_output=True, text=True,
                                  timeout=30)
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is not None and proc.returncode == 0 and any(
                line.strip() for line in proc.stdout.splitlines()):
            return True
    return _NVIDIA_PROC.exists()


def resolve_flavor(flavor: str = "auto") -> str:
    """``'cpu'`` or ``'gpu'`` for a ``flavor=`` value (``'auto'``: by :func:`host_has_gpu`)."""
    if flavor == "auto":
        return "gpu" if host_has_gpu() else "cpu"
    return flavor


def archive_key(env: str, flavor: str) -> str:
    """The archive key of an env: ``'<env>-cpu'`` for ``'cpu'``, else ``'<env>'``."""
    return f"{env}-cpu" if flavor == "cpu" else env


def archive_for(env: str, flavor: str = "auto", *, manifest: dict | None = None,
                sizes: dict | None = None) -> tuple[str, str]:
    """Which published archive serves ``env`` for a flavour.

    ``manifest`` and ``sizes`` stand in for :func:`packed_manifest` and
    :func:`packed_sizes`.

    Returns
    -------
    tuple of (str, str)
        ``('<env>-cpu', 'cpu')`` when a CPU build is wanted and published (in
        the manifest with a measured size; a ``null`` size marks an archive
        not uploaded yet), else ``('<env>', 'gpu')``, which every env has.
    """
    wanted = resolve_flavor(flavor)
    if wanted == "cpu":
        key = archive_key(env, "cpu")
        manifest = packed_manifest() if manifest is None else manifest
        sizes = packed_sizes() if sizes is None else sizes
        if key in manifest and (sizes.get(key) or {}).get("archive_bytes") is not None:
            return key, "cpu"
    return env, "gpu"


def _single_build(env: str, *, manifest: dict | None = None) -> bool:
    """``True`` when ``env`` has one archive for CPU and GPU hosts."""
    manifest = packed_manifest() if manifest is None else manifest
    return archive_key(env, "cpu") not in manifest


def auto_flavor_note(flavor: str, *, dry_run: bool = False) -> str | None:
    """The stderr note for ``flavor='auto'`` on a host without a GPU, else
    ``None``: pass ``'gpu'`` when the jobs run on GPU nodes. ``dry_run`` says
    what would be installed."""
    if flavor != "auto" or resolve_flavor(flavor) != "cpu" or host_system():
        # off Linux nothing is installed here: no GPU node to plan for
        return None
    gpu = "flavor='gpu'"
    verb = "would install" if dry_run else "installing"
    return (f"# {verb} CPU builds (no NVIDIA GPU on this host); if the jobs run on GPU "
            f"nodes, pass {gpu}")


def installed_flavor(env: str, conda: str | None = None) -> str | None:
    """Which archive flavour an installed env came from.

    Returns
    -------
    str or None
        ``'cpu'``, ``'gpu'`` or ``'single'`` from :data:`FLAVOR_FILE` (an env
        with a single build reads ``'single'``); ``None`` when the env is not
        installed or has no record, or the record is unreadable or holds
        anything else.
    """
    prefix = env_prefix(env, conda)
    if prefix is None:
        return None
    try:
        words = (prefix / FLAVOR_FILE).read_text().split()
    except OSError:
        return None
    word = words[0] if words and words[0] in ("cpu", "gpu", "single") else None
    if word in ("gpu", "single"):
        return "single" if _single_build(env) else "gpu"
    return word


_GROUPS_YAML = Path(__file__).resolve().parent / "env_groups.yaml"
# per-env lockfiles (`conda env export --no-builds`)
_LOCKS_DIR = Path(__file__).resolve().parent / "env_locks"


def own_env_name(method: str) -> str:
    """The method's own env name, ``scmb_<method>``: :func:`group_for`'s
    fallback for a method in no shared group."""
    return f"scmb_{method.lower()}"


# --- shared group environments --------------------------------------------
@functools.lru_cache(maxsize=1)
def _env_groups() -> dict:
    """``env_groups.yaml``: ``groups`` (shared envs and their members) and
    ``method_env`` (``{method: env}`` overrides, which win over membership)."""
    return yaml.safe_load(_GROUPS_YAML.read_text()) or {}


def group_for(method: str) -> str:
    """The env that serves this method: its override, its shared group, or
    its own ``scmb_<method>`` env."""
    override = _env_groups().get("method_env", {}) or {}
    if method in override:
        return override[method]
    for name, g in _env_groups().get("groups", {}).items():
        if method in g.get("members", []):
            return name
    return own_env_name(method)


# --- env existence / status -----------------------------------------------
PACKED_URL = "https://github.com/DSichang/scMultiBench/releases/download/envs-v1"


def _envs_dir(conda_bin: str) -> Path | None:
    """Where this conda/mamba keeps named environments, or None.

    ``info --json`` names the folders differently in conda and mamba >= 2,
    so every spelling is read, then ``<root>/envs`` of the launcher's
    ``<root>/bin/<tool>``. The first writable candidate wins: an env in a
    folder conda does not search is invisible by name.
    """
    def _paths(value):
        items = value if isinstance(value, (list, tuple)) else [value]
        return [Path(str(v)) for v in items if str(v or "").startswith("/")]

    cands: list[Path] = []
    try:
        out = subprocess.run([conda_bin, "info", "--json"],
                             capture_output=True, text=True).stdout
        info = _json.loads(out) if out.strip().startswith("{") else {}
    except Exception:  # noqa: BLE001 - any failure just means "ask elsewhere"
        info = {}
    for key in ("envs_dirs", "envs directories"):
        cands += _paths(info.get(key))
    for key in ("root_prefix", "base environment"):
        cands += [p / "envs" for p in _paths(info.get(key))]
    exe = shutil.which(conda_bin)
    if exe:
        root = Path(exe).resolve().parent.parent
        if (root / "conda-meta").is_dir() or (root / "envs").is_dir():
            cands.append(root / "envs")
    for var in ("CONDA_ROOT", "MAMBA_ROOT_PREFIX"):
        cands += [p / "envs" for p in _paths(os.environ.get(var))]

    for d in cands:
        probe = d if d.is_dir() else d.parent
        if probe.is_dir() and os.access(probe, os.W_OK):
            return d
    return None


def _find_conda() -> str | None:
    """Path of the conda (else mamba) binary on PATH, or ``None``."""
    return shutil.which("conda") or shutil.which("mamba")


@functools.lru_cache(maxsize=4)
def _conda_prefixes(conda: str) -> tuple:
    """The prefixes ``<conda> env list --json`` reports, cached per binary;
    ``()`` when the tool is absent or answers garbage. Cleared after an
    env is created."""
    try:
        out = subprocess.run([conda, "env", "list", "--json"],
                             capture_output=True, text=True, check=True).stdout
        prefixes = _json.loads(out).get("envs", [])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError,
            TypeError, AttributeError, OSError):
        return ()
    return tuple(str(p) for p in prefixes)


def _prefixes_on_disk(envs_dir: Path) -> dict[str, Path]:
    """``{env: prefix}`` for every ``<envs_dir>/<env>`` that carries ``bin/``."""
    try:
        return {p.name: p for p in sorted(envs_dir.iterdir())
                if (p / "bin").is_dir()}
    except OSError:
        return {}


def _creation_command(prefix: Path) -> str:
    """The first ``# cmd:`` line of ``<prefix>/conda-meta/history`` (the command
    that created the env), or ``""``."""
    try:
        with open(prefix / "conda-meta" / "history", encoding="utf-8",
                  errors="replace") as fh:
            for line in fh:
                if line.startswith("# cmd:"):
                    return line[len("# cmd:"):].strip()
    except OSError:
        pass
    return ""


def _lockfile_build_record(prefix: Path, env: str) -> bool:
    """Was ``prefix`` created by a lockfile build of an earlier multibench?

    Those builds ran ``<conda> env create -n <env> -f <file>`` (older conda:
    ``conda-env create ...``) with ``<file>``
    one of ``.../multibench_envbuild/<env>/conda.yml`` (installed package),
    ``.../env_locks/.build/<env>/conda.yml`` (source checkout), or, before
    0.2, ``.../env_locks/<env>.yml``; conda keeps that command as the first
    ``# cmd:`` line of ``conda-meta/history``.
    """
    cmd = _creation_command(prefix)
    if not cmd:
        return False
    e = re.escape(env)
    # `conda env create`, `mamba env create`, or conda's older `conda-env create`
    if not re.search(r"(?:^|\s|/)(?:conda-env|env)\s+create\s", cmd):
        return False
    if not re.search(rf"\s(?:-n|--name)(?:\s+|=){e}(?:\s|$)", cmd):
        return False
    return re.search(rf"\s(?:-f|--file)(?:\s+|=).*/(?:multibench_envbuild/{e}/conda\.yml|"
                     rf"env_locks/\.build/{e}/conda\.yml|env_locks/{e}\.yml)\s*$",
                     cmd) is not None


def made_by_multibench(prefix: Path | str, env: str | None = None) -> bool:
    """Did multibench install the env at ``prefix``?

    ``True`` when the prefix holds :data:`FLAVOR_FILE` (every install path
    writes it), or when conda's history shows a lockfile build of an earlier
    multibench (:func:`_lockfile_build_record`). An env someone made by hand
    under the same name is not a method env.
    """
    prefix = Path(prefix)
    return (prefix / FLAVOR_FILE).is_file() or _lockfile_build_record(prefix, env or prefix.name)


def _same_named(env: str, conda: str | None = None):
    """Every prefix named ``env``, lazily: ``<envs_dir>/<env>`` when it has
    ``bin/`` (no conda asked), then those ``conda env list`` reports."""
    candidate = Path(config.envs_dir) / env
    if (candidate / "bin").is_dir():
        yield candidate
    conda_exe = conda or _find_conda()
    if conda_exe is not None:
        for p in _conda_prefixes(conda_exe):
            if os.path.basename(p.rstrip("/")) == env and Path(p) != candidate:
                yield Path(p)


def other_env_prefix(env: str, conda: str | None = None) -> Path | None:
    """A prefix named ``env`` that multibench did not install, or ``None``.

    Such an env is never used for a method; :func:`other_env_text` says what to do.
    """
    for p in _same_named(env, conda):
        if p.is_dir() and not made_by_multibench(p, env):
            return p
    return None


def other_env_text(env: str, prefix, then: str) -> str:
    """``"Environment matilda was not installed by this version of multibench, so
    it is not used. Rename or remove /x/envs/matilda, then <then>"``: true also
    for an env of an older version that :func:`made_by_multibench` cannot
    recognise. The action comes before the path, so a clipped line keeps it."""
    return (f"Environment {env} was not installed by this version of multibench, so "
            f"it is not used. Rename or remove {prefix}, then {then}")


def _mark_built(env: str, conda: str | None, before: set) -> None:
    """Write ``build`` to :data:`FLAVOR_FILE` in the env a lockfile build just
    created under the name ``env``: a prefix of that name that was not there
    ``before`` the build."""
    _conda_prefixes.cache_clear()
    for p in _same_named(env, conda):
        if p not in before and p.is_dir():
            (p / FLAVOR_FILE).write_text("build\n")


def env_prefix(env: str, conda: str | None = None) -> Path | None:
    """The prefix of the method env ``env`` multibench installed, or ``None``:
    ``<mtb.config.envs_dir>/<env>``, else the one ``conda env list`` reports
    (``conda`` = the conda/mamba to ask). An env of that name made by hand
    is never used."""
    for p in _same_named(env, conda):
        if made_by_multibench(p, env):
            return p
    return None


def archive_urls(key: str, manifest: dict | None = None) -> list[str]:
    """The download URLs of the archive ``key``, in order: one, or the parts
    of an archive over 2 GiB; ``<PACKED_URL>/<key>.tar.gz`` without an entry."""
    manifest = packed_manifest() if manifest is None else manifest
    entry = manifest.get(key) or f"{PACKED_URL}/{key}.tar.gz"
    return [entry] if isinstance(entry, str) else list(entry)


def download_archive(urls: list[str], target: Path, *, label: str,
                     total: int | None = None, attempts: int = 4,
                     timeout: float = 60) -> int:
    """Stream ``urls`` in order into ``target``; return the bytes written.

    Prints a line at the start, then one progress line that is rewritten per
    tenth of ``total`` (per 500 MB when it is unknown). A dropped or stalled connection or an HTTP 5xx resumes where
    it stopped, up to ``attempts`` times per URL; another HTTP error raises.
    """
    import time
    import urllib.error
    import urllib.request
    step = max(total // 10, 1) if total else 500_000_000
    # MB below a gigabyte, so a 53 MB file does not read "0.1 GB"
    fmt = _gb if (total or 0) >= 1e9 else (lambda n: f"{n / 1e6:.0f} MB")
    size = f" {fmt(total)}" if total else ""
    print(f"[env] downloading {label}{size} ...", flush=True)
    done, mark, t0 = 0, step, time.monotonic()

    def copy(resp, out):
        nonlocal done, mark
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done >= mark:
                rate = done / max(time.monotonic() - t0, 1e-3) / 1e6
                print(f"\r[env]   {fmt(done)}" + (f" of {fmt(total)}" if total else "")
                      + f" ({rate:.0f} MB/s)", end="", flush=True)
                mark = (done // step + 1) * step

    with open(target, "wb") as out:
        for url in urls:
            start = done
            for attempt in range(1, attempts + 1):
                have = done - start
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={have}-"} if have else {})
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        if have and getattr(resp, "status", 200) != 206:
                            # the whole file again, not the rest
                            out.seek(start)
                            out.truncate()
                            done = start
                        copy(resp, out)
                    break
                except (OSError, TimeoutError) as e:
                    final = isinstance(e, urllib.error.HTTPError) and e.code < 500
                    if final or attempt == attempts:
                        raise
                    print(f"\n[env]   connection lost ({type(e).__name__}); resuming",
                          flush=True)
                    time.sleep(3 * attempt)
    print(flush=True)                   # ends the progress line
    return done


_LAYERS_JSON = Path(__file__).resolve().parent / "packed_layers.json"

#: Folder under the envs dir that holds the unpacked layers.
LAYERS_DIR = ".layers"
#: Written by ``tools/pack_layers.py`` into an env archive that needs layers.
LAYER_MARKER = ".multibench_layers.json"
#: Carried by a layer archive: its name, file count and digest.
LAYER_INFO = ".multibench_layer.json"


@functools.lru_cache(maxsize=1)
def packed_layers() -> dict:
    """The layers of ``engine/packed_layers.json``: ``{layer: {"urls",
    "files", "digest", "archive_bytes", "unpacked_bytes", "envs"}}``.

    A layer holds files several env archives share (torch and CUDA
    libraries); it is downloaded once and hard-linked into each env.
    """
    return _json.loads(_LAYERS_JSON.read_text())


def layers_for(keys, *, layers: dict | None = None, envs_dir=None,
               missing_only: bool = True) -> dict:
    """``{layer: entry}`` the archive ``keys`` need, each layer once;
    ``missing_only`` leaves out those ``envs_dir`` already holds."""
    layers = packed_layers() if layers is None else layers
    root = Path(envs_dir or config.envs_dir) / LAYERS_DIR
    keys = set(keys)
    return {name: entry for name, entry in layers.items()
            if keys & set(entry.get("envs", ()))
            and not (missing_only and (root / name / LAYER_INFO).is_file())}


def plan_totals(rows, sizes: dict | None = None, *, layers: dict | None = None
                ) -> tuple[dict, str]:
    """``(totals, line)`` for the envs in ``rows``, each shared library layer counted
    once. ``totals``: ``total_download_bytes``, ``total_disk_bytes`` (``None`` when
    an env is not measured) and ``shared_layers`` (``{name: {archive_bytes,
    unpacked_bytes, envs}}``); ``line``: ``"2 envs: 5.4 GB to download, 5.6 GB on
    disk"``. The install plan uses it."""
    sizes = packed_sizes() if sizes is None else sizes
    keys = [archive_key(r["env"], r.get("flavor")) for r in rows]
    layers = layers_for(keys) if layers is None else layers
    shared = {name: {"archive_bytes": e.get("archive_bytes"),
                     "unpacked_bytes": e.get("unpacked_bytes"),
                     "envs": [r["env"] for r, k in zip(rows, keys)
                              if k in set(e.get("envs", ()))]}
              for name, e in layers.items()}

    def total(part):
        known = [v for v in ((sizes.get(k) or {}).get(part) for k in keys) if v is not None]
        return (sum(known) + sum(e[part] or 0 for e in shared.values()),
                len(known) == len(keys), bool(known))

    dl, dl_all, dl_any = total("archive_bytes")
    disk, disk_all, disk_any = total("unpacked_bytes")
    n = len(rows)
    line = f"{n} env{'s' if n != 1 else ''}: " + (
        f"{_gb(dl)} to download" if dl_all else
        f"at least {_gb(dl)} to download" if dl_any else "download size not recorded")
    if disk_any:
        line += f", {'' if disk_all else 'at least '}{_gb(disk)} on disk"
    if shared:
        # the rows show each env's own archive; the layer is in the total only
        line += (" (with the shared layer " + ", ".join(
            f"{name}, {_gb(e['archive_bytes'])}" for name, e in shared.items())
            + ", downloaded once)")
    return ({"total_download_bytes": dl if dl_all else None,
             "total_disk_bytes": disk if disk_all else None,
             "shared_layers": shared}, line)


def _ensure_layer(name: str, info: dict, envs_root: Path) -> Path:
    """``<envs_root>/.layers/<name>``, downloaded and unpacked on first use;
    ``RuntimeError`` when its digest is not the one ``info`` names."""
    import tarfile
    from .._data.fetch import safe_extract
    dest = envs_root / LAYERS_DIR / name
    if not (dest / LAYER_INFO).is_file():
        entry = packed_layers().get(name)
        if not entry or not entry.get("urls"):
            raise RuntimeError(f"this multibench version does not list the layer {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tgz = dest.parent / f".{name}.tar.gz.download"
        part = dest.with_name(name + ".partial")
        try:
            download_archive(list(entry["urls"]), tgz, label=f"{name} (shared layer)",
                             total=entry.get("archive_bytes"))
            print(f"[env] unpacking {name} -> {dest} ...", flush=True)
            shutil.rmtree(part, ignore_errors=True)
            shutil.rmtree(dest, ignore_errors=True)
            part.mkdir(parents=True)
            with tarfile.open(tgz) as t:
                safe_extract(t, part)
            part.rename(dest)
        finally:
            tgz.unlink(missing_ok=True)
            shutil.rmtree(part, ignore_errors=True)
    have = _json.loads((dest / LAYER_INFO).read_text())
    if have.get("digest") != info.get("digest"):
        raise RuntimeError(f"the layer {name} on disk is not the one this archive was "
                           f"packed with; delete {dest} and install again")
    return dest


def _link_layer(layer: Path, prefix: Path, info: dict) -> None:
    """Hard-link (else copy) every file of ``layer`` into ``prefix``;
    ``RuntimeError`` when a file is already there or the count is wrong."""
    n = 0
    for dirpath, _dirs, files in os.walk(layer):
        rel = Path(dirpath).relative_to(layer)
        (prefix / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if rel == Path(".") and f == LAYER_INFO:
                continue
            src, dst = Path(dirpath) / f, prefix / rel / f
            if dst.exists():
                raise RuntimeError(f"{rel / f} is in the env archive and in the layer")
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            n += 1
    if n != info.get("files"):
        raise RuntimeError(f"the layer {layer.name} has {n} files, expected {info.get('files')}")


def install_packed(env: str, *, envs_dir: Path | str | None = None,
                   conda: str | None = None, flavor: str = "auto") -> bool:
    """Provision ``env`` from a prebuilt conda-pack archive, if one is published.

    Downloads the archive, extracts it into ``<envs_dir>/<env>`` and runs its
    own ``bin/conda-unpack``; no conda is needed. ``envs_dir`` defaults to
    that of ``conda`` when only it is given, else
    ``mtb.config.envs_dir``. ``flavor`` picks the archive as in
    :func:`install`; the prefix is the same whatever the flavour, and the
    flavour installed is written to :data:`FLAVOR_FILE`.

    Returns
    -------
    bool
        ``True`` on success or when the prefix exists; ``False`` when no
        archive is published or the unpack failed (the caller then builds
        from the lockfile).
    """
    import tarfile
    import urllib.error

    if envs_dir is not None:
        envs_root = Path(envs_dir)
    elif conda is not None and shutil.which(conda):
        envs_root = _envs_dir(conda) or Path(config.envs_dir)
    else:
        envs_root = Path(config.envs_dir)
    dest = envs_root / env
    if dest.exists():
        if made_by_multibench(dest, env):
            return True
        raise RuntimeError(other_env_text(env, dest, "install again."))
    manifest, sizes = packed_manifest(), packed_sizes()
    key, installed = archive_for(env, flavor, manifest=manifest, sizes=sizes)
    if (resolve_flavor(flavor) == "cpu" and installed == "gpu"
            and not _single_build(env, manifest=manifest)):
        size = _gb((sizes.get(env) or {}).get("archive_bytes"))
        _text.warn(f"No CPU build of {env} yet; installing the GPU build ({size}).",
                   UserWarning)
    urls = archive_urls(key, manifest)
    build = "single" if _single_build(env, manifest=manifest) else installed
    envs_root.mkdir(parents=True, exist_ok=True)
    tgz = envs_root / f".{key}.tar.gz.download"
    try:
        download_archive(urls, tgz, label=f"{env} ({build} build)",
                         total=(sizes.get(key) or {}).get("archive_bytes"))
    except urllib.error.HTTPError:
        tgz.unlink(missing_ok=True)
        return False
    except BaseException:
        tgz.unlink(missing_ok=True)
        raise
    print(f"[env] unpacking {env} -> {dest} ...", flush=True)
    from .._data.fetch import safe_extract
    part = dest.with_name(dest.name + ".partial")
    try:
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True)
        with tarfile.open(tgz) as t:
            safe_extract(t, part)
        if not (part / "bin").is_dir():
            raise RuntimeError("archive did not contain bin/")
        # the files this env shares with others: one download, hard links
        marker = part / LAYER_MARKER
        if marker.is_file():
            for name, info in _json.loads(marker.read_text())["layers"].items():
                _link_layer(_ensure_layer(name, info, envs_root), part, info)
        # conda-unpack writes the final path into the env: move first
        part.rename(dest)
        _conda_unpack(dest)
        # last, so a prefix that failed to unpack claims no flavour
        (dest / FLAVOR_FILE).write_text(
            ("single" if build == "single" else installed) + "\n")
        _conda_prefixes.cache_clear()
        return True
    except BaseException as e:  # noqa: BLE001 - degrade to the lockfile build
        shutil.rmtree(part, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)
        if not isinstance(e, Exception):    # an interrupt leaves no half env
            raise
        print(f"[env] prebuilt {env} failed ({type(e).__name__}: "
              f"{str(e)[:120]}); falling back to the lockfile build",
              flush=True)
        return False
    finally:
        tgz.unlink(missing_ok=True)


def _conda_unpack(prefix: Path) -> None:
    """Run ``<prefix>/bin/conda-unpack``, if there, with the env's own python
    (first on ``PATH``, or explicitly for a script that lost its executable bit)."""
    unpack = prefix / "bin" / "conda-unpack"
    if not unpack.exists():
        return
    py = prefix / "bin" / "python"
    argv = [str(unpack)] if os.access(unpack, os.X_OK) else [str(py), str(unpack)]
    env = {**os.environ,
           "PATH": f"{prefix / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"}
    subprocess.run(argv, check=True, capture_output=True, env=env)


def _all_prefixes(conda: str | None = None) -> list[tuple[str, Path]]:
    """``(name, prefix)`` of every env under ``envs_dir``, then of those
    ``conda`` lists."""
    out = list(_prefixes_on_disk(Path(config.envs_dir)).items())
    conda_exe = conda or _find_conda()
    if conda_exe is not None:
        for p in _conda_prefixes(conda_exe):
            name = os.path.basename(str(p).rstrip("/"))
            if name:
                out.append((name, Path(p)))
    return out


def installed_envs(conda: str | None = None) -> list[str]:
    """Names of the envs :func:`env_prefix` would find: those multibench
    installed (:func:`made_by_multibench`), under ``envs_dir`` first, then
    those ``conda`` lists, without duplicates."""
    return list(dict.fromkeys(name for name, p in _all_prefixes(conda)
                              if made_by_multibench(p, name)))


# --- lockfile-based provisioning (the reproducible install path) -----------
def lockfile(env_name: str) -> Path | None:
    """Path of the shipped lockfile ``env_locks/<env_name>.yml``, or ``None``."""
    p = _LOCKS_DIR / f"{env_name}.yml"
    return p if p.exists() else None


def split_lock(text: str):
    """Split a lockfile into (conda-only YAML, pip requirement lines).

    The pip section is a full ``pip freeze`` closure: re-resolving it, as
    ``conda env create`` does, fails on combinations pip calls inconsistent.
    It is installed separately with ``--no-deps``, which a requirements file
    cannot express.
    """
    conda_lines, pip_lines = [], []
    pip_indent = None
    for ln in text.splitlines():
        stripped = ln.strip()
        indent = len(ln) - len(ln.lstrip())
        if stripped == "- pip:":
            pip_indent = indent
            continue
        if pip_indent is not None and stripped:
            if indent <= pip_indent:
                pip_indent = None
            else:
                pip_lines.append(stripped[2:].strip() if stripped.startswith("- ")
                                 else stripped)
                continue
        conda_lines.append(ln)
    return "\n".join(conda_lines) + "\n", pip_lines


def _materialise_split(env_name: str, lock):
    """Write the conda-only YAML and pip requirements a two-phase install needs."""
    # an installed package's folder stays untouched: build files go to the temp dir
    if (_LOCKS_DIR.parents[2] / "pyproject.toml").is_file():
        build_dir = _LOCKS_DIR / ".build" / env_name
    else:
        import tempfile
        build_dir = Path(tempfile.gettempdir()) / "multibench_envbuild" / env_name
    build_dir.mkdir(parents=True, exist_ok=True)
    conda_yaml, pip_lines = split_lock(lock.read_text(encoding="utf-8"))
    y = build_dir / "conda.yml"
    y.write_text(conda_yaml, encoding="utf-8")
    r = build_dir / "requirements.txt"
    r.write_text("\n".join(pip_lines) + "\n" if pip_lines else "", encoding="utf-8")
    return y, r, pip_lines


def post_install(env_name: str):
    """Path to the shipped post-install script of an env, or None.

    It restores what a lockfile cannot record: R packages from
    ``install.packages()``, pip packages on no index and editable installs (matilda, scMVP).
    """
    p = _LOCKS_DIR / f"{env_name}.post.sh"
    return p if p.is_file() else None


def create_env(env_name: str, conda: str | None = None) -> list[list[str]]:
    """The commands that build one env from its shipped lockfile: conda phase,
    pip ``--no-deps`` phase, post-install script when shipped. A missing
    lockfile raises ``FileNotFoundError``."""
    lock = lockfile(env_name)
    if lock is None:
        raise FileNotFoundError(
            f"no lockfile for env {env_name!r} (expected {_LOCKS_DIR / (env_name + '.yml')}).")
    conda = conda or _find_conda() or "conda"
    conda_yaml, req, pip_lines = _materialise_split(env_name, lock)
    cmds = [[conda, "env", "create", "-n", env_name, "-f", str(conda_yaml)]]
    if pip_lines:
        cmds.append([conda, "run", "-n", env_name,
                     "pip", "install", "--no-deps", "-r", str(req)])
    post = post_install(env_name)
    if post is not None:
        cmds.append([conda, "run", "-n", env_name, "bash", str(post)])
    return cmds


def _build(env_name: str, cmds: list[list[str]], conda: str | None) -> None:
    """Run the build ``cmds`` of ``env_name`` and mark the new env as made by
    multibench."""
    before = set(_same_named(env_name, conda))
    _run_all(cmds)
    _mark_built(env_name, conda, before)


def create_all(category: str | None = None, methods: list[str] | None = None,
               conda: str | None = None, dry_run: bool = True) -> list[dict]:
    """Provision every env needed to run the methods, from lockfiles.

    With ``dry_run=False`` the missing envs are built (:func:`install` checks
    the host and the methods first).

    Parameters
    ----------
    category : str, optional
        Methods with a variant in this category; default: all.
    methods : list of str, optional
        Explicit method ids, instead of ``category``.
    conda : str, optional
        conda executable; default ``conda``.
    dry_run : bool
        ``True`` (default) plans only.

    Returns
    -------
    list of dict
        One entry per env, largest first: ``{env, methods, exists, cmds}``
        (``cmds`` = the commands run, or that would run).
    """
    have = set(installed_envs(conda))
    out = []
    for env, ms in _envs_needed(category, methods):
        exists = env in have
        cmds = [] if exists else create_env(env, conda=conda)
        if cmds and not dry_run:
            _build(env, cmds, conda)
        out.append({"env": env, "methods": ms, "exists": exists, "cmds": cmds})
    return out


_PACKED_MANIFEST = Path(__file__).resolve().parent / "packed_urls.json"


def packed_manifest() -> dict:
    """``{archive key: URL or list of part URLs}`` of the published archives
    (``engine/packed_urls.json``)."""
    return _json.loads(_PACKED_MANIFEST.read_text())


#: The columns of :func:`install`'s table; the rows hold more, for the package.
INSTALL_COLUMNS = ("env", "methods", "state", "flavor", "archive_bytes", "unpacked_bytes")


def install(methods: list[str] | None = None, *, category: str | None = None,
            dry_run: bool = False, flavor: str = "auto", build: bool = False):
    """Install the environments a set of methods needs; Linux only.

    Each env is unpacked from its published archive, or built from its
    lockfile where none is published. Installed envs are skipped.

    Parameters
    ----------
    methods : list[str] | None
        Method ids to cover; ``None`` = every method of ``category``, or every
        method.
    category : str | None
        ``vertical``, ``diagonal``, ``mosaic`` or ``cross``: the methods to
        cover when ``methods`` is ``None``.
    dry_run : bool
        ``True`` = install nothing and return each env's state and size.
    flavor : str
        ``'cpu'``, ``'gpu'`` or ``'auto'`` (``'gpu'`` when an NVIDIA GPU is
        visible): the archive build of each env.
    build : bool
        ``True`` = build every env from its lockfile instead of unpacking the
        archive.

    Returns
    -------
    pandas.DataFrame
        One row per env, the env serving the most methods first: ``env``,
        ``methods``, ``state`` (``'have'``, ``'packed archive published'``,
        ``'name taken'`` ... on a dry run; ``'PACKED'``, ``'BUILD'`` after an
        install), ``flavor`` (``'cpu'``, ``'gpu'``, or ``'any'`` for one
        archive for both), ``archive_bytes`` (the download) and
        ``unpacked_bytes`` (on disk).

    Raises
    ------
    KeyError
        Unknown method id.
    OSError
        A real install on a host that is not Linux.
    RuntimeError
        ``build=True`` without conda, or a failed build.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.env.install(["Matilda", "totalVI"], dry_run=True)
    >>> mtb.env.install(category="vertical")
    """
    rows = _install_rows(methods, category=category, packed=not build, dry_run=dry_run,
                         flavor=flavor)
    import pandas as pd
    plan = pd.DataFrame(rows, columns=list(INSTALL_COLUMNS))
    if dry_run and not build:
        todo = to_download(rows)
        totals, line = plan_totals(todo)
        plan.attrs.update(totals)
        if todo:
            print(f"[env] {line}", flush=True)
    return plan


def to_download(rows) -> list[dict]:
    """The rows of an install plan whose archive would be downloaded."""
    return [r for r in rows if not r["exists"] and r.get("packed_url")
            and r["state"] != "name taken"]


def _install_rows(methods: list[str] | None = None, *, category: str | None = None,
                  packed: bool = True, dry_run: bool = False, conda: str | None = None,
                  flavor: str = "auto") -> list[dict]:
    """The rows of :func:`install`, as dicts."""
    check_flavor(flavor)
    registry.check_id_list(methods, "methods")
    methods = None if methods is None else [registry.check_method(m) for m in methods]
    manifest = packed_manifest() if packed else {}
    sizes = packed_sizes() if packed else {}
    unpacked = set() if dry_run else _install_missing(
        category, methods, packed=packed, manifest=manifest, sizes=sizes, conda=conda,
        flavor=flavor)
    rows = create_all(category=category, methods=methods, conda=conda, dry_run=dry_run)
    out = []
    for r in rows:
        env = r["env"]
        state = _row_state(r, unpacked=unpacked, dry_run=dry_run, packed=packed, conda=conda)
        # URL and sizes of the archive this host would fetch
        if packed:
            key, would_install = archive_for(env, flavor, manifest=manifest, sizes=sizes)
        else:
            key, would_install = env, None
        size = sizes.get(key) or {}
        flav = (installed_flavor(env, conda) if (r["exists"] or env in unpacked)
                else would_install)
        if flav == "gpu" and packed and _single_build(env, manifest=manifest):
            flav = "any"            # one archive for CPU and GPU hosts
        out.append({**r, "state": state,
                    "packed_url": (archive_urls(key, manifest)[0]
                                   if key in manifest else None),
                    "archive_bytes": size.get("archive_bytes"),
                    "unpacked_bytes": size.get("unpacked_bytes"), "flavor": flav})
    if dry_run and packed and any(not r["exists"] and r["flavor"] == "cpu"
                                  and r["state"] != "name taken" for r in out):
        note = auto_flavor_note(flavor, dry_run=True)
        if note:
            print(note, file=sys.stderr, flush=True)
    return out


def _install_missing(category, methods, *, packed, manifest, sizes, conda,
                     flavor) -> set:
    """Refuse before any download or build (off Linux, an env name taken,
    ``build=True`` without conda), then unpack the published archives
    (``packed``) of the envs not installed; the names of the envs unpacked."""
    linux_only = linux_only_text()
    if linux_only:
        raise OSError(f"{linux_only} Run the install on a Linux machine, such as Colab.")
    have = set(installed_envs(conda))
    missing = [env for env, _ in _envs_needed(category, methods) if env not in have]
    for env in missing:
        other = other_env_prefix(env, conda)
        if other is not None:
            raise RuntimeError(other_env_text(env, other, "install again."))
    if not packed and (conda or _find_conda()) is None:
        raise RuntimeError(f"{NO_CONDA} build=True needs it. Leave out build=True to "
                           f"unpack the published archives.")
    if not packed:
        return set()
    # said only when a CPU build is taken
    if any(archive_for(env, flavor, manifest=manifest, sizes=sizes)[1] == "cpu"
           for env in missing):
        note = auto_flavor_note(flavor)
        if note:
            print(note, file=sys.stderr, flush=True)
    return {env for env in missing if install_packed(env, conda=conda, flavor=flavor)}


def _row_state(r: dict, *, unpacked, dry_run, packed, conda) -> str:
    """The ``state`` of one row of :func:`install`'s table."""
    env = r["env"]
    if env in unpacked:
        return "PACKED"
    if r["exists"]:
        return "have"
    if dry_run and other_env_prefix(env, conda) is not None:
        return "name taken"
    if not dry_run:
        return "BUILD"
    return "packed archive published" if packed else "build(dry-run)"


def _envs_needed(category: str | None, methods: list[str] | None) -> list:
    """``[(env, sorted method ids)]`` of the envs the methods (else those of
    ``category``) need, the env serving the most methods first."""
    if methods is None:
        methods = registry.list_methods(category=category)
    by_env: dict[str, list[str]] = {}
    for m in methods:
        by_env.setdefault(group_for(m), []).append(m)
    return [(env, sorted(ms))
            for env, ms in sorted(by_env.items(), key=lambda kv: (-len(kv[1]), kv[0]))]


_R_USER_LIBRARY_VARS = ("R_LIBS", "R_LIBS_SITE")
_R_NO_USER_FILES = ("R_LIBS_USER", "R_ENVIRON_USER", "R_PROFILE_USER")


def without_user_r_library(environ) -> dict:
    """``environ`` as a dict in which R reads only the env's own library:
    ``R_LIBS``/``R_LIBS_SITE`` removed, ``R_LIBS_USER``, ``R_ENVIRON_USER`` and
    ``R_PROFILE_USER`` set to a path that is not a directory (R's Renviron
    turns an empty ``R_LIBS_USER`` back into ``~/R/...``)."""
    out = {k: v for k, v in environ.items() if k not in _R_USER_LIBRARY_VARS}
    out.update(dict.fromkeys(_R_NO_USER_FILES, os.devnull))
    return out


def _run_all(cmds: list[list[str]]) -> None:
    # a package in ~/.local or the user's R library would be skipped by pip or
    # a post-install script, and the run, which reads neither, fails
    env = {**without_user_r_library(os.environ), "PYTHONNOUSERSITE": "1"}
    if cmds and shutil.which(cmds[0][0]) is None:
        raise RuntimeError(f"{NO_CONDA} Install conda first.")
    for c in cmds:
        proc = subprocess.run(c, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"env command failed: {' '.join(c)}\nstderr tail:\n{proc.stderr[-2000:]}"
            )
    _conda_prefixes.cache_clear()


