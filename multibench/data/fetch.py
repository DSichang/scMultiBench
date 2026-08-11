"""Download reference datasets from the repository's release assets."""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

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
    root = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    for ds in datasets:
        if (root / ds).is_dir():
            continue
        if ds not in AVAILABLE:
            raise ValueError(
                f"{ds!r} is not in the release assets ({', '.join(sorted(AVAILABLE))}); "
                "see 'Get the data' in the installation guide for the full collection")
        root.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"downloading {ds} ({AVAILABLE[ds]}) ...", flush=True)
        tgz, _ = urllib.request.urlretrieve(f"{RELEASE_URL}/{ds}.tar.gz")
        with tarfile.open(tgz) as t:
            t.extractall(root)
        if not (root / ds).is_dir():
            raise RuntimeError(f"downloaded archive did not contain {ds}/")
    return root
