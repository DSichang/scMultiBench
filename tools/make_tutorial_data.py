"""Write the tutorial-size demo datasets (D28mini, D46mini, D52mini).

usage: python tools/make_tutorial_data.py <data_path> <out_dir>

Each is a seeded random subset of a benchmark dataset, small enough that a
tutorial runs its two methods twice and scores them in 10-15 minutes on a
Colab runtime (2 vCPUs). Files with the same cell count are the same cells
(a batch's RNA, ADT and labels; D28's two ATAC files), so every modality
file stays aligned with its label file. Peak matrices keep the peaks open
in the most sampled cells, at most MAX_PEAKS. Only the canonical inputs are
written (``*.h5`` and ``*cty*.csv``); other files in a benchmark folder are
outputs of earlier runs. Each output folder is also written as
``<name>.tar.gz`` for the data-v1 release.
"""
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SEED = 0
MAX_PEAKS = 20_000
# dataset -> (mini name, cells kept per group of files with the same cell count)
PLAN = {"D28": ("D28mini", 1500), "D46": ("D46mini", 1000), "D52": ("D52mini", 1000)}


def _n_cells(path: Path) -> int:
    if path.suffix == ".csv":
        return len(pd.read_csv(path))
    with h5py.File(path) as f:
        return f["matrix/data"].shape[1]          # features x cells


def _is_peaks(path: Path) -> bool:
    return path.suffix == ".h5" and ("peak" in path.stem or path.stem.startswith("atac")
                                     and "gas" not in path.stem)


def _copy_h5(src: Path, dst: Path, cells: np.ndarray) -> tuple[int, int]:
    with h5py.File(src) as f:
        data = f["matrix/data"][:, cells]
        feats = np.arange(data.shape[0])
        if _is_peaks(src) and data.shape[0] > MAX_PEAKS:
            open_in = (data > 0).sum(axis=1)
            feats = np.sort(np.argsort(-open_in, kind="stable")[:MAX_PEAKS])
            data = data[feats]
        with h5py.File(dst, "w") as g:
            m = g.create_group("matrix")
            m.create_dataset("data", data=data, compression="gzip")
            for name, idx in (("barcodes", cells), ("features", feats)):
                if f"matrix/{name}" in f:
                    m.create_dataset(name, data=f[f"matrix/{name}"][()][idx], compression="gzip")
            if "matrix/.data_dimnames" in f:
                d = m.create_group(".data_dimnames")
                d.create_dataset("1", data=f["matrix/.data_dimnames/1"][()][cells], compression="gzip")
                d.create_dataset("2", data=f["matrix/.data_dimnames/2"][()][feats], compression="gzip")
    return data.shape


def make(src: Path, dst: Path, per_group: int) -> None:
    rng = np.random.default_rng(SEED)
    files = sorted(p for p in src.iterdir() if p.suffix == ".h5" or
                   (p.suffix == ".csv" and "cty" in p.stem))
    counts = {p: _n_cells(p) for p in files}
    keep = {n: np.sort(rng.choice(n, size=min(per_group, n), replace=False))
            for n in sorted(set(counts.values()))}
    dst.mkdir(parents=True, exist_ok=False)
    for p in files:
        cells = keep[counts[p]]
        if p.suffix == ".csv":
            pd.read_csv(p).iloc[cells].to_csv(dst / p.name, index=False)
            print(f"  {p.name}: {len(cells)} of {counts[p]} cells")
        else:
            shape = _copy_h5(p, dst / p.name, cells)
            print(f"  {p.name}: {shape[1]} of {counts[p]} cells, {shape[0]} features")


def main(argv):
    data_path, out = Path(argv[1]), Path(argv[2])
    for ds, (name, per_group) in PLAN.items():
        print(name, "from", ds)
        make(data_path / ds, out / name, per_group)
        with tarfile.open(out / f"{name}.tar.gz", "w:gz") as t:
            t.add(out / name, arcname=name)
        print(f"  -> {out / name}.tar.gz {(out / f'{name}.tar.gz').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv)
