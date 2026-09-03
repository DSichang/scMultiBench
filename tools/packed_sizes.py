"""Maintainer tool: refresh ``multibench/engine/packed_sizes.json``.

``multibench env plan`` / ``env install --packed`` print how large each packed
archive is (download) and how much disk it needs unpacked. Those numbers are
a SHIPPED SNAPSHOT, never fetched at runtime (offline nodes, Zenodo rate
limits) - so they rot whenever an archive is republished. Rerun this after
publishing::

    python tools/packed_sizes.py                    # HEAD every URL in packed_urls.json
    python tools/packed_sizes.py --envs-dir /opt/conda/envs   # also measure unpacked sizes
    python tools/packed_sizes.py --check            # exit 1 when the shipped file is stale

Only ``archive_bytes`` needs the network; ``unpacked_bytes`` needs the envs on
disk (``--envs-dir``). An env that cannot be measured keeps ``null`` rather
than a guess.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URLS = ROOT / "multibench" / "engine" / "packed_urls.json"
SIZES = ROOT / "multibench" / "engine" / "packed_sizes.json"


def head_content_length(url: str, opener=None) -> int | None:
    """``Content-Length`` of ``url`` after redirects, or ``None`` when unknown.

    ``opener`` (an ``urllib.request.OpenerDirector``-like object with
    ``open(request)``) is injectable so the tool is testable without network.
    """
    req = urllib.request.Request(url, method="HEAD")
    opener = opener or urllib.request.build_opener()
    try:
        with opener.open(req) as resp:
            n = resp.headers.get("Content-Length")
    except Exception as e:  # noqa: BLE001 - one bad URL must not stop the table
        print(f"  ! {url}: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def du_bytes(path: Path) -> int | None:
    """Total size of the files under ``path`` (symlinks not followed), or None."""
    if not path.is_dir():
        return None
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


def build(urls: dict, *, previous: dict | None = None, envs_dir: Path | None = None,
          opener=None, today: str | None = None) -> dict:
    """The sizes table for ``urls`` (``{env: url}``).

    Measures ``archive_bytes`` by HEAD request and ``unpacked_bytes`` under
    ``envs_dir`` when given; a value that cannot be measured falls back to
    ``previous`` (the shipped file) and finally ``None``.
    """
    previous = previous or {}
    out = {"_meta": {
        "written_by": "tools/packed_sizes.py",
        "note": "archive_bytes = Content-Length of the packed archive in packed_urls.json "
                "(a HEAD request); unpacked_bytes = size on disk after conda-unpack. "
                "null = not measured yet; rerun the tool after republishing an archive.",
        "measured": today or _dt.date.today().isoformat(),
    }}
    for env in sorted(urls):
        prev = previous.get(env) or {}
        arch = head_content_length(urls[env], opener=opener)
        unp = du_bytes(envs_dir / env) if envs_dir else None
        out[env] = {"archive_bytes": arch if arch is not None else prev.get("archive_bytes"),
                    "unpacked_bytes": unp if unp is not None else prev.get("unpacked_bytes")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--envs-dir", type=Path, help="conda envs directory on a host that has "
                    "the envs unpacked, to measure unpacked_bytes")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the shipped table differs")
    ap.add_argument("--out", type=Path, default=SIZES, help="where to write (default: the "
                    "shipped engine/packed_sizes.json)")
    args = ap.parse_args(argv)
    urls = json.loads(URLS.read_text())
    previous = json.loads(args.out.read_text()) if args.out.is_file() else {}
    table = build(urls, previous=previous, envs_dir=args.envs_dir)
    text = json.dumps(table, indent=1) + "\n"
    if args.check:
        # the measurement date is not a difference worth failing on
        old = {k: v for k, v in previous.items() if k != "_meta"}
        new = {k: v for k, v in table.items() if k != "_meta"}
        if old != new:
            print("packed_sizes.json is stale - rerun tools/packed_sizes.py", file=sys.stderr)
            return 1
        return 0
    args.out.write_text(text)
    known = sum(1 for k, v in table.items() if k != "_meta" and v["archive_bytes"])
    print(f"wrote {args.out} ({known}/{len(urls)} archive sizes known)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
