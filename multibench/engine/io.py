"""Load method outputs by kind."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_output(out_dir: Path | str, spec):
    """The output ``spec`` names in ``out_dir``, loaded by its kind."""
    path = Path(out_dir) / spec.file
    if spec.kind in ("embedding", "markers", "graph"):
        with h5py.File(path, "r") as f:
            return np.asarray(f[spec.dataset or "data"])
    if spec.kind == "labels":
        return [line for line in path.read_text().splitlines() if line != ""]
    raise ValueError(f"unknown output kind {spec.kind!r}")
