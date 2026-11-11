"""Pack method environments built on a Windows computer into the archives ``mtb.env.install``
unpacks on Windows, and write their entries of ``multibench/engine/packed_windows.json``.

usage: python tools/pack_windows.py OUT --envs FOLDER [--cpu-envs FOLDER] [--only KEY[,KEY...]] [--jobs N]

``--envs`` holds the GPU builds and the environments with one build (``<FOLDER>/<env>``); ``--cpu-envs``
the CPU builds, packed as ``<env>-cpu``. An environment is a conda prefix or an R installation that runs
from any folder; it names the methods it was built for in ``.multibench_methods``, one per line.

An archive is ``OUT/<key>.tar.xz``, cut into ``.001``, ``.002`` ... when it is over 1,900 MiB (a release
file may have 2 GiB). Left out of it: ``__pycache__`` folders, link libraries and debug symbols
(``*.lib``, ``*.pdb``: not loaded when a program runs), an installer's uninstaller, and every file whose
path inside the environment has more than 160 characters (listed per archive). ``OUT/packed_windows.json``
gets, per archive: its files, the SHA-256 of the whole archive, its size, the size and number of the files
in it, its longest path and its methods. Copy that file to ``multibench/engine/`` after the archives are
uploaded to the release its ``base`` names.

Standard library only; run it with any Python 3.9 or later.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import lzma
import os
import tarfile
import time

BASE = "https://github.com/DSichang/scMultiBench/releases/download/envs-win-v1"
PART = 1900 * 2**20
#: A file with a longer path inside the environment is left out: Windows allows 259 characters for the
#: whole path, and the environment's own folder needs room. What exceeds this in the environments packed
#: so far are test fixtures, C++ headers and assets of a JupyterLab extension, which no method reads.
LONGEST = 160
SKIP_SUFFIXES = (".lib", ".pdb")
SKIP_NAMES = ("unins000.exe", "unins000.dat")


class Parts:
    """A file cut into parts of at most ``PART`` bytes; the SHA-256 of all of it."""

    def __init__(self, out: str, name: str):
        self.out, self.name, self.files, self.fh, self.room = out, name, [], None, 0
        self.sha, self.bytes = hashlib.sha256(), 0

    def write(self, data) -> int:
        view = memoryview(data)
        while len(view):
            if self.fh is None or self.room == 0:
                if self.fh:
                    self.fh.close()
                self.files.append(f"{self.name}.{len(self.files) + 1:03d}")
                self.fh = open(os.path.join(self.out, self.files[-1]), "wb")
                self.room = PART
            n = min(len(view), self.room)
            self.fh.write(view[:n])
            self.sha.update(view[:n])
            self.room -= n
            self.bytes += n
            view = view[n:]
        return len(data)

    def close(self) -> list[str]:
        if self.fh:
            self.fh.close()
        if len(self.files) == 1:                        # one part: no suffix
            os.replace(os.path.join(self.out, self.files[0]), os.path.join(self.out, self.name))
            self.files = [self.name]
        return self.files


def pack(key: str, prefix: str, out: str) -> dict:
    t0 = time.time()
    for old in os.listdir(out):
        if old == f"{key}.tar.xz" or old.startswith(f"{key}.tar.xz."):
            os.remove(os.path.join(out, old))
    methods = open(os.path.join(prefix, ".multibench_methods"), encoding="utf-8").read().split()
    parts = Parts(out, f"{key}.tar.xz")
    files = size = longest = 0
    too_long: dict[str, int] = {}                # top folders of the files left out -> how many
    with lzma.open(parts, "wb", preset=6) as xz, tarfile.open(fileobj=xz, mode="w|",
                                                              format=tarfile.PAX_FORMAT) as tar:
        for folder, dirs, names in os.walk(prefix):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            rel = os.path.relpath(folder, prefix).replace("\\", "/")
            for d in dirs:
                info = tar.gettarinfo(os.path.join(folder, d), d if rel == "." else f"{rel}/{d}")
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info)
            for f in sorted(names):
                if f.lower().endswith(SKIP_SUFFIXES) or (rel == "." and f in SKIP_NAMES):
                    continue
                arc = f if rel == "." else f"{rel}/{f}"
                if len(arc) > LONGEST:
                    top = "/".join(arc.split("/")[:4])
                    too_long[top] = too_long.get(top, 0) + 1
                    continue
                path = os.path.join(folder, f)
                info = tar.gettarinfo(path, arc)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(path, "rb") as fh:
                    tar.addfile(info, fh)
                files += 1
                size += info.size
                longest = max(longest, len(arc))
    names = parts.close()
    return {"key": key, "files": names, "sha256": parts.sha.hexdigest(), "archive_bytes": parts.bytes,
            "unpacked_bytes": size, "file_count": files, "longest_path": longest, "methods": methods,
            "left_out_for_path_length": too_long, "seconds": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--envs", required=True)
    ap.add_argument("--cpu-envs")
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo = {}
    for folder, suffix in ((a.envs, ""), (a.cpu_envs, "-cpu")):
        if not folder:
            continue
        for env in sorted(os.listdir(folder)):
            prefix = os.path.join(folder, env)
            if os.path.isfile(os.path.join(prefix, ".multibench_methods")):
                todo[env + suffix] = prefix
    if a.only:
        todo = {k: v for k, v in todo.items() if k in a.only.split(",")}
    manifest_file = os.path.join(a.out, "packed_windows.json")
    manifest = (json.load(open(manifest_file, encoding="utf-8")) if os.path.isfile(manifest_file)
                else {"base": BASE, "archives": {}})

    def save():
        manifest["_meta"] = {"written": time.strftime("%Y-%m-%d %H:%M"),
                             "note": "Written by tools/pack_windows.py. archive_bytes: the archive, all its "
                                     "files together; unpacked_bytes: the files in it."}
        manifest["archives"] = dict(sorted(manifest["archives"].items()))
        with open(manifest_file, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, min(a.jobs, len(todo)))) as pool:
        # the small ones first, so that a result is there early
        jobs = [pool.submit(pack, key, prefix, a.out) for key, prefix in todo.items()]
        for job in concurrent.futures.as_completed(jobs):
            r = job.result()
            seconds = r.pop("seconds")
            manifest["archives"][r.pop("key")] = r
            save()
            print(f"{r['files'][0].split('.tar.xz')[0]}: {r['file_count']} files, "
                  f"{r['unpacked_bytes'] / 2**20:.0f} MB -> {r['archive_bytes'] / 2**20:.0f} MB in "
                  f"{len(r['files'])} file(s), {seconds} s", flush=True)
    total = sum(v["archive_bytes"] for v in manifest["archives"].values())
    print(f"{len(manifest['archives'])} archives in {manifest_file}: {total / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
