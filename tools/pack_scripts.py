"""Maintainer tool: pack the package's method scripts into the archive a wheel downloads.

    python tools/pack_scripts.py            # write dist/tools_scripts-<id>.tar.gz and
                                            # multibench/engine/scripts_archive.json
    python tools/pack_scripts.py --check    # exit 1 when the recorded archive is not
                                            # the one tools_scripts/ packs to

A wheel does not carry ``tools_scripts/``: ``multibench.engine.scripts.ensure_repo``
downloads the archive named in ``engine/scripts_archive.json`` and checks its
sha256. The archive is built the same way every time (sorted names, no times
or owners), so the recorded sha256 is a function of the files alone. After a
script changes: run this tool, upload the archive to the ``scripts-v1``
release, and commit the JSON file with the scripts.
"""
import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "tools_scripts"
RECORD = ROOT / "multibench" / "engine" / "scripts_archive.json"
RELEASE = "https://github.com/DSichang/scMultiBench/releases/download/scripts-v1"


def pack() -> bytes:
    """The archive of ``tools_scripts/`` as bytes; the same bytes for the same files."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        # the files git tracks: a method run in a checkout leaves others beside them
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "tools_scripts"],
                                 capture_output=True, check=True).stdout.decode().split("\0")
        for f in sorted(ROOT / name for name in tracked if name):
            info = tarfile.TarInfo(f"tools_scripts/{f.relative_to(SCRIPTS).as_posix()}")
            data = f.read_bytes()
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=9) as z:
        z.write(raw.getvalue())
    return out.getvalue()


def record(data: bytes) -> dict:
    sha = hashlib.sha256(data).hexdigest()
    return {"url": f"{RELEASE}/tools_scripts-{sha[:12]}.tar.gz", "sha256": sha,
            "bytes": len(data)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    data = pack()
    rec = record(data)
    if args.check:
        have = json.loads(RECORD.read_text()) if RECORD.is_file() else {}
        if have != rec:
            print("engine/scripts_archive.json does not match tools_scripts/ - run "
                  "python tools/pack_scripts.py and upload the archive", file=sys.stderr)
            return 1
        return 0
    out = ROOT / "dist" / rec["url"].rsplit("/", 1)[1]
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(data)
    RECORD.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {out} ({len(data) / 1e6:.1f} MB) and {RECORD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
