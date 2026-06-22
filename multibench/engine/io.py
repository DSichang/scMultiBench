"""Load method outputs by kind."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_output(out_dir: Path | str, spec):
    """Dispatch by OutputSpec.kind."""
    path = Path(out_dir) / spec.file
    if spec.kind in ("embedding", "imputed", "markers", "graph"):
        with h5py.File(path, "r") as f:
            return np.asarray(f[spec.dataset or "data"])
    if spec.kind == "labels":
        return [ln for ln in path.read_text().splitlines() if ln != ""]
    if spec.kind == "coords":
        return path  # caller handles multi-file / glob
    raise ValueError(f"unknown output kind {spec.kind!r}")
