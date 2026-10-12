"""Maintainer tool: split the files several packed envs share into one layer.

Envs that pin the same torch build each carry the same torch and CUDA
libraries, most of their archive. This tool reads the full conda-pack
archives of such envs and writes::

    <out>/<name>.tar.gz          the layer: every file under ``--roots`` that has
                                 the same path, mode and content in all archives
    <out>/<env>.tar.gz           each env without those files ("slim")
    <out>/<name>.json            the entry for ``engine/packed_layers.json``

    python tools/pack_layers.py --name torch-2.8.0-cu12-py39 --out out \\
        packed/scmb_torch.tar.gz packed/scmb_cobolt3.tar.gz ...

The layer is computed from the archives, never listed by hand: a file that
differs in one env stays in that env's slim archive. ``mtb.env.install``
downloads the layer once and hard-links it into each env
(:func:`multibench.engine.envs.install_packed`).

A slim archive names its layer in ``.multibench_layers.json``, and its
``bin/conda-unpack`` no longer lists the layer's files: conda-unpack opens
every listed file for writing, and the layer's files are shared between envs.
The tool stops when a shared file holds a prefix conda-unpack would rewrite
in it.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from contextlib import contextmanager
from pathlib import Path

ROOTS = ("lib/python3.9/site-packages/torch/",
         "lib/python3.9/site-packages/nvidia/",
         "lib/python3.9/site-packages/triton/")
UNPACK = "bin/conda-unpack"
#: Read by ``install_packed`` from an unpacked slim archive.
MARKER = ".multibench_layers.json"
#: Carried by the layer archive: its name, file count and digest.
LAYER_INFO = ".multibench_layer.json"
_RECORD = re.compile(r"^\('(?P<path>[^']+)', '(?P<prefix>[^']+)', '(?:text|binary)'\),$")


def _name(member: tarfile.TarInfo) -> str:
    return member.name.removeprefix("./")


@contextmanager
def _read(path: Path):
    """A streaming ``TarFile`` over ``path`` (``pigz`` when it is installed)."""
    if shutil.which("pigz"):
        proc = subprocess.Popen(["pigz", "-dc", str(path)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                yield tar
        finally:
            proc.stdout.close()
            if proc.wait() != 0:
                raise RuntimeError(f"pigz failed on {path}")
    else:
        with tarfile.open(path, mode="r|gz") as tar:
            yield tar


@contextmanager
def _write(path: Path):
    """A streaming gzip ``TarFile`` at ``path`` (``pigz`` when it is installed)."""
    if shutil.which("pigz"):
        with open(path, "wb") as raw:
            proc = subprocess.Popen(["pigz", "-6", "-c"], stdin=subprocess.PIPE, stdout=raw)
            with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
                yield tar
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"pigz failed on {path}")
    else:
        with tarfile.open(path, mode="w|gz", compresslevel=6) as tar:
            yield tar


def _sha(fh) -> str:
    h = hashlib.sha256()
    while chunk := fh.read(1 << 20):
        h.update(chunk)
    return h.hexdigest()


def scan(archive: Path, roots=ROOTS) -> tuple[dict, set, dict]:
    """``({path: (sha256, mode)}, link targets, {path: prefix})`` of one archive.

    The first maps every regular file under ``roots``; the second holds the
    targets of the archive's hard links; the third the prefix its
    ``bin/conda-unpack`` rewrites in each file it lists.
    """
    files, targets, prefixes = {}, set(), {}
    with _read(archive) as tar:
        for m in tar:
            name = _name(m)
            if m.islnk():
                targets.add(m.linkname.removeprefix("./"))
            elif m.isreg() and name.startswith(tuple(roots)):
                files[name] = (_sha(tar.extractfile(m)), m.mode)
            elif m.isreg() and name == UNPACK:
                text = tar.extractfile(m).read().decode()
                prefixes = {r["path"]: r["prefix"] for line in text.splitlines()
                            if (r := _RECORD.match(line))}
    return files, targets, prefixes


def shared(scans: dict) -> dict:
    """``{path: sha256}`` of the files every scanned archive holds alike."""
    first, *rest = (files for files, _, _ in scans.values())
    common = {p: v[0] for p, v in first.items() if all(o.get(p) == v for o in rest)}
    for archive, (_, targets, _) in scans.items():
        if linked := sorted(targets & common.keys())[:3]:
            raise SystemExit(f"{archive}: hard links point at shared files {linked}")
    return common


def digest(layer: dict) -> str:
    return hashlib.sha256("".join(f"{p} {h}\n" for p, h in sorted(layer.items()))
                          .encode()).hexdigest()


def _add(tar: tarfile.TarFile, name: str, data: bytes, like: tarfile.TarInfo | None = None):
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), like.mode if like else 0o644
    info.mtime = like.mtime if like else 0
    tar.addfile(info, io.BytesIO(data))


def write_layer(archive: Path, layer: dict, info: dict, out: Path, prefixes: dict) -> None:
    """The layer archive: the shared files of ``archive`` and ``LAYER_INFO``.

    ``prefixes`` maps a path to the prefixes conda-unpack would rewrite in
    it, one per scanned archive; a shared file that holds one stops the tool.
    """
    left = set(layer)
    with _read(archive) as src, _write(out) as dst:
        _add(dst, LAYER_INFO, json.dumps(info).encode())
        for m in src:
            name = _name(m)
            if name in left:
                left.discard(name)
                data = src.extractfile(m).read()
                if held := [p for p in prefixes.get(name, ()) if p.encode() in data]:
                    raise SystemExit(f"{name} is shared and holds the prefix {held[0]}")
                dst.addfile(m, io.BytesIO(data))
    if left:
        raise SystemExit(f"{archive}: {len(left)} shared files not found on the second read")


def strip_records(text: str, layer: dict) -> str:
    """``bin/conda-unpack`` without the records of the layer's files."""
    kept = [line for line in text.splitlines(keepends=True)
            if not ((r := _RECORD.match(line.rstrip("\n"))) and r["path"] in layer)]
    return "".join(kept)


def write_slim(archive: Path, layer: dict, info: dict, out: Path) -> None:
    """``archive`` without the layer's files, naming the layer in ``MARKER``."""
    with _read(archive) as src, _write(out) as dst:
        for m in src:
            name = _name(m)
            if name in layer:
                continue
            if name == UNPACK:
                _add(dst, m.name, strip_records(src.extractfile(m).read().decode(),
                                                layer).encode(), like=m)
            else:
                dst.addfile(m, src.extractfile(m) if m.isreg() else None)
        _add(dst, MARKER, json.dumps({"layers": {info["name"]: info}}).encode())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archives", nargs="+", type=Path, help="full conda-pack archives")
    ap.add_argument("--name", required=True, help="layer name, e.g. torch-2.8.0-cu12-py39")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--roots", nargs="+", default=list(ROOTS))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    scans = {}
    for a in args.archives:
        scans[a] = scan(a, args.roots)
        print(f"scanned {a.name}: {len(scans[a][0])} files under the roots", flush=True)
    layer = shared(scans)
    if not layer:
        raise SystemExit("the archives share no file under the roots")
    info = {"name": args.name, "files": len(layer), "digest": digest(layer)}
    target = args.out / f"{args.name}.tar.gz"
    prefixes = {p: {s[2][p] for s in scans.values() if p in s[2]} for p in layer}
    write_layer(args.archives[0], layer, info, target, prefixes)
    print(f"layer {target.name}: {len(layer)} files, {target.stat().st_size / 1e9:.2f} GB",
          flush=True)
    envs = []
    for a in args.archives:
        slim = args.out / a.name
        write_slim(a, layer, info, slim)
        envs.append(a.name.removesuffix(".tar.gz"))
        print(f"slim {slim.name}: {slim.stat().st_size / 1e9:.2f} GB "
              f"(was {a.stat().st_size / 1e9:.2f} GB)", flush=True)
    entry = {args.name: {**info, "archive_bytes": target.stat().st_size, "envs": envs}}
    (args.out / f"{args.name}.json").write_text(json.dumps(entry, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
