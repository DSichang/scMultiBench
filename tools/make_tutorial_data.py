"""Write the small demo datasets the tutorials and method checks run on.

usage: python tools/make_tutorial_data.py <benchmark data dir> <out_dir>

Each is a smaller copy of a benchmark dataset: a seeded random subset of the
cells, the most variable genes of the RNA and the most open ATAC peaks, so
that a tutorial runs in 10-15 minutes on a Colab runtime (2 vCPUs) and every
method that accepts the dataset's layout runs on it. The sizes are the
smallest at which no method failed:

- online_iNMF needs about 5,000 cells in total (D28mini: 3,000 + 3,000);
- Multigrate's mosaic variant needs about 6,000 (D45mini: 2,000 per batch);
- MIRA needs about 2,000 genes, VIPCCA selects 2,000 variable genes itself
  and Conos' variance fit fails on 1,000, so the RNA + ATAC datasets keep
  2,500 genes; the mosaic and cross datasets run on 1,000;
- scMVP drops every peak open in more than 10% of the cells, so half of the
  peaks kept are the most open ones and half the most open below that limit.

Files with the same cell count are the same cells (a batch's RNA, ADT and
labels; D28's two ATAC files), so every modality file stays aligned with its
label file. One gene set serves every RNA and gene-activity file of a
dataset, and one peak set every peak file, so batches keep the same features.
ADT is never reduced. Only the canonical inputs are written (``*.h5`` and
``*cty*.csv``). Each folder is also written as ``<name>.tar.gz`` for the
data-v1 release.

D27 holds RNA, ATAC peaks and gene activity of the same cells; three datasets
come from it: D27mini (vertical, RNA + peaks), D27mini_gas (vertical, RNA +
gene activity) and D27mini_paired (diagonal layout with paired cells, which
Seurat_v5 needs as its bridge).
"""
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SEED = 0
# name -> source dataset, cells per group of files with the same cell count,
# genes kept, peaks kept, and for D27 the {output file: source file} map
PLAN = {
    "D28mini": dict(src="D28", cells=3000, genes=2500, peaks=5000),
    "D45mini": dict(src="D45", cells=2000, genes=1000, peaks=5000),
    "D46mini": dict(src="D46", cells=600, genes=1000, peaks=5000),
    "D52mini": dict(src="D52", cells=1000, genes=1000, peaks=5000),
    "D27mini": dict(src="D27", cells=5000, genes=2500, peaks=5000,
                    files={"rna.h5": "rna.h5", "atac.h5": "peak.h5", "cty.csv": "rna_cty.csv"}),
    "D27mini_gas": dict(src="D27", cells=2000, genes=1000, peaks=5000,
                        files={"rna.h5": "rna.h5", "atac.h5": "atac_gas.h5",
                               "cty.csv": "rna_cty.csv"}),
    "D27mini_paired": dict(src="D27", cells=3000, genes=2500, peaks=5000,
                           files={"rna.h5": "rna.h5", "atac_peak.h5": "peak.h5",
                                  "atac_gas.h5": "atac_gas.h5", "rna_cty.csv": "rna_cty.csv",
                                  "atac_cty.csv": "peak_cty.csv"}),
}


def _n_cells(path: Path) -> int:
    if path.suffix == ".csv":
        return len(pd.read_csv(path))
    with h5py.File(path) as f:
        return f["matrix/data"].shape[1]          # features x cells


def _names(path: Path) -> np.ndarray:
    with h5py.File(path) as f:
        return np.array([x.decode() for x in f["matrix/features"][()]])


def _kind(src: Path) -> str:
    """'rna', 'gas', 'peak' or 'adt' from the source file name."""
    stem = src.stem
    if stem.startswith("adt"):
        return "adt"
    if "gas" in stem:
        return "gas"
    if stem.startswith(("peak", "atac")):
        return "peak"
    return "rna"


def _top_genes(rna: Path, cells: np.ndarray, n: int, among=None) -> np.ndarray:
    """Names of the ``n`` most variable genes of an RNA file, in the file's
    order (scanpy, Seurat flavour); ``among``: the names to choose from."""
    import anndata as ad
    import scanpy as sc
    with h5py.File(rna) as f:
        a = ad.AnnData(np.asarray(f["matrix/data"][:, cells]).T)
    a.var_names = _names(rna)
    if among is not None:
        a = a[:, np.isin(a.var_names, among)].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=n, flavor="seurat")
    return np.array(a.var_names[a.var["highly_variable"].values])


def _top_peaks(files: dict, n: int) -> np.ndarray:
    """Names of ``n`` peaks, in the files' order: the ``n // 2`` open in the
    most cells, then the most open ones among the peaks open in at most 10%
    of the cells. Counted over the peak files (``{path: cells}``), which must
    list the same peaks."""
    paths = list(files)
    names = _names(paths[0])
    open_in, total = np.zeros(len(names)), 0
    for p, cells in files.items():
        assert np.array_equal(_names(p), names), f"{p.name}: another peak list"
        with h5py.File(p) as f:
            open_in += (f["matrix/data"][:, cells] > 0).sum(axis=1)
        total += len(cells)
    order = np.argsort(-open_in, kind="stable")
    top = order[: n // 2]
    rare = [i for i in order if open_in[i] <= 0.1 * total and i not in set(top)][: n - len(top)]
    return names[np.sort(np.concatenate([top, np.array(rare, dtype=int)]))]


def _copy_h5(src: Path, dst: Path, cells: np.ndarray, keep=None) -> tuple:
    """Write ``cells`` of ``src``; ``keep``: the feature names to keep, written
    in that order (None: all, in the file's order)."""
    with h5py.File(src) as f:
        feats = np.arange(f["matrix/data"].shape[0])
        if keep is not None:
            at = {name: i for i, name in enumerate(_names(src))}
            feats = np.array([at[name] for name in keep if name in at])
        data = f["matrix/data"][:, cells][feats]
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


def make(src: Path, dst: Path, cells, genes=None, peaks=None, files=None) -> None:
    rng = np.random.default_rng(SEED)
    if files is None:
        files = {p.name: p.name for p in sorted(src.iterdir())
                 if p.suffix == ".h5" or (p.suffix == ".csv" and "cty" in p.stem)}
    pairs = {out: src / name for out, name in files.items()}
    counts = {out: _n_cells(p) for out, p in pairs.items()}
    keep = {n: (np.arange(n) if cells is None or cells >= n else
                np.sort(rng.choice(n, size=cells, replace=False)))
            for n in sorted(set(counts.values()))}
    h5 = {out: p for out, p in pairs.items() if p.suffix == ".h5"}
    rna = [out for out, p in h5.items() if _kind(p) == "rna"]
    gene_set = None
    if genes and rna:
        # one gene list, in the RNA's order, for the RNA and gene-activity files
        among = None
        for p in h5.values():
            if _kind(p) == "gas":
                among = _names(p) if among is None else np.intersect1d(among, _names(p))
        gene_set = _top_genes(h5[rna[0]], keep[counts[rna[0]]], genes, among)
    peak_files = {p: keep[counts[out]] for out, p in h5.items() if _kind(p) == "peak"}
    peak_set = None
    if peaks and peak_files and len(_names(next(iter(peak_files)))) > peaks:
        peak_set = _top_peaks(peak_files, peaks)
    dst.mkdir(parents=True, exist_ok=False)
    for out, p in pairs.items():
        c = keep[counts[out]]
        if p.suffix == ".csv":
            pd.read_csv(p).iloc[c].to_csv(dst / out, index=False)
            print(f"  {out}: {len(c)} of {counts[out]} cells")
        else:
            which = {"rna": gene_set, "gas": gene_set, "peak": peak_set, "adt": None}[_kind(p)]
            shape = _copy_h5(p, dst / out, c, which)
            print(f"  {out}: {shape[1]} of {counts[out]} cells, {shape[0]} features")


def main(argv):
    data_path, out = Path(argv[1]), Path(argv[2])
    for name, spec in PLAN.items():
        if not (data_path / spec["src"]).is_dir():
            print(name, "skipped:", spec["src"], "is not in", data_path)
            continue
        print(name, "from", spec["src"])
        make(data_path / spec["src"], out / name, spec["cells"], spec["genes"],
             spec["peaks"], spec.get("files"))
        with tarfile.open(out / f"{name}.tar.gz", "w:gz") as t:
            t.add(out / name, arcname=name)
        print(f"  -> {name}.tar.gz {(out / f'{name}.tar.gz').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv)
