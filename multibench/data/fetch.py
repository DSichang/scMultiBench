"""Download reference datasets from the repository's release assets."""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path


def safe_extract(tar: tarfile.TarFile, dest) -> None:
    """extractall with a path-traversal guard (absolute paths, .., links out).

    A crafted archive could otherwise write outside ``dest``; every tarball we
    open (datasets, packed envs) goes through here. On a Python whose
    ``tarfile`` has the extraction filters (3.8.17+/3.9.17+/3.12+) the
    ``'data'`` filter is passed as well - the archives are plain data, and
    the bare call raises ``DeprecationWarning`` on 3.12/3.13 and changes
    behaviour on 3.14; older interpreters keep the bare call (the guard
    above is what they have).
    """
    import os
    dest = Path(dest).resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise RuntimeError(f"archive entry escapes target dir: {m.name!r}")
        if m.islnk() or m.issym():
            link = (target.parent / m.linkname).resolve() if not m.linkname.startswith("/") else Path(m.linkname)
            if not str(link).startswith(str(dest) + os.sep):
                raise RuntimeError(f"archive link escapes target dir: {m.name!r}")
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
    else:                                 # pragma: no cover - pre-filter Pythons
        tar.extractall(dest)

from .. import config

RELEASE_URL = "https://github.com/DSichang/scMultiBench/releases/download/data-v1"

# datasets published as release assets, with approximate download sizes
AVAILABLE = {"D11": "11 MB", "D28": "137 MB", "D45": "290 MB",
             "D46": "97 MB", "D52": "179 MB"}


def fetch(*datasets: str, data_path=None, quiet: bool = False) -> Path:
    """Ensure the named reference datasets exist locally, downloading if needed.

    Idempotent: datasets already present under the data root are left alone.
    Returns the data root. The full 65-dataset collection is linked from the
    scMultiBench README; this helper covers the tutorial reference sets.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    root = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    for ds in datasets:
        d = root / ds
        if d.is_dir() and any(d.iterdir()):
            continue
        if d.is_dir():                       # empty leftover from a failed run
            d.rmdir()
        if ds not in AVAILABLE:
            raise ValueError(
                f"{ds!r} is not in the release assets ({', '.join(sorted(AVAILABLE))}); "
                "see 'Get the data' in the installation guide for the full collection")
        root.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"downloading {ds} ({AVAILABLE[ds]}) ...", flush=True)
        tgz, _ = urllib.request.urlretrieve(f"{RELEASE_URL}/{ds}.tar.gz")
        # extract to a scratch dir first and move into place atomically, so an
        # interrupted download/extract can never masquerade as a complete
        # dataset on the next run
        tmp = Path(_tempfile.mkdtemp(dir=root, prefix=f".{ds}-"))
        try:
            with tarfile.open(tgz) as t:
                safe_extract(t, tmp)
            if not (tmp / ds).is_dir():
                raise RuntimeError(f"downloaded archive did not contain {ds}/")
            (tmp / ds).rename(root / ds)
        finally:
            _shutil.rmtree(tmp, ignore_errors=True)
    return root
