"""Write the small demo datasets the tutorials and method checks run on.

usage: python tools/make_tutorial_data.py <benchmark data dir> <out_dir> [name ...]

One dataset per integration task, and every method of the task runs on it:

    vertical   RNA + ADT          D11 (the benchmark dataset itself, 2,864 cells)
    vertical   RNA + ATAC         D27mini_vertical
    vertical   RNA + ADT + ATAC   D22mini
    diagonal   RNA + ATAC         D27mini
    mosaic     RNA + ATAC         D45mini
    mosaic     RNA + ADT + ATAC   D46mini
    diagonal   multiple RNA, multiple ATAC      D37mini
    mosaic     mixed, no shared modality        D49mini
    cross      multiple RNA + ATAC              D56mini
    cross      multiple ADT + ATAC              D58mini
    cross      multiple RNA + ADT + ATAC        D59mini
    mosaic     RNA + ADT          D38mini
    cross      RNA + ADT          D52mini

Each mini is a smaller copy of a benchmark dataset: a seeded random subset of
the cells, the most variable genes of the RNA and a set of ATAC peaks, so that
a tutorial runs in 10-15 minutes on a Colab runtime (2 vCPUs). The sizes are
the smallest at which no method of the task failed:

- online_iNMF needs about 5,000 cells in total (D27mini: 3,000 + 3,000);
- Multigrate's mosaic runs need about 6,000 (D45mini: 2,000 per batch);
- MIRA needs about 2,000 genes and 5,000 cells, VIPCCA selects 2,000 variable
  genes itself and Conos' variance fit fails on 1,000, so the RNA + ATAC
  datasets keep 2,500 genes; the others run on 1,000;
- scMVP drops every peak open in more than 10% of the cells, so half of the
  peaks kept are the most open ones and half the most open below that limit.

D27 holds RNA, ATAC peaks and gene activity of the same cells. D27mini_vertical
carries both ATAC files, so the methods that read peaks and those that read
gene activity run on the one dataset; D27mini is its diagonal layout, whose
paired cells Seurat_v5 needs as its bridge.

Files with the same cell count are the same cells (a batch's RNA, ADT and
labels), so every modality file stays aligned with its label file. One gene
list (in the RNA's order) serves every RNA and gene-activity file of a
dataset, and one peak list every peak file. ADT is never reduced. Only the
canonical inputs are written. Each folder is also written as
``<name>.tar.gz`` for the data-v1 release.
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
    # vertical
    "D27mini_vertical": dict(src="D27", cells=5000, genes=2500, peaks=5000,
                             files={"rna.h5": "rna.h5", "atac_peak.h5": "peak.h5",
                                    "atac_gas.h5": "atac_gas.h5", "cty.csv": "rna_cty.csv"}),
    "D22mini": dict(src="D22", cells=3000, genes=1000, peaks=5000,
                    files={"rna.h5": "rna.h5", "adt.h5": "adt.h5", "atac.h5": "atac.h5",
                           "cty.csv": "cty.csv"}),
    # diagonal
    "D27mini": dict(src="D27", cells=3000, genes=2500, peaks=5000,
                    files={"rna.h5": "rna.h5", "atac_peak.h5": "peak.h5",
                           "atac_gas.h5": "atac_gas.h5", "rna_cty.csv": "rna_cty.csv",
                           "atac_cty.csv": "peak_cty.csv"}),
    "D37mini": dict(src="D37", cells=1000, genes=2500, peaks=5000,
                    files={f"{m}{i}.{e}": f"{m}{i}.{e}" for i in (1, 2, 3)
                           for m, e in (("rna", "h5"), ("atac_peak", "h5"), ("atac_gas", "h5"),
                                        ("rna_cty", "csv"), ("atac_cty", "csv"))},
                    # D37's gene-activity files name the cells otherwise than its
                    # peak files, in the same order (checked in make)
                    barcodes_from={f"atac_gas{i}.h5": f"atac_peak{i}.h5" for i in (1, 2, 3)}),
    # mosaic
    "D45mini": dict(src="D45", cells=2000, genes=1000, peaks=5000),
    "D46mini": dict(src="D46", cells=600, genes=1000, peaks=5000),
    "D38mini": dict(src="D38", cells=2000, genes=1000, peaks=5000,
                    files={"rna1.h5": "rna1.h5", "rna2.h5": "rna2.h5", "adt2.h5": "adt2.h5",
                           "adt3.h5": "adt3.h5", "cty1.csv": "cty1.csv", "cty2.csv": "cty2.csv",
                           "cty3.csv": "cty3.csv"}),
    "D49mini": dict(src="D49", cells=1300, genes=1000, peaks=5000),
    # cross
    "D52mini": dict(src="D52", cells=1000, genes=1000, peaks=5000),
    "D56mini": dict(src="D56", cells=1500, genes=1000, peaks=5000,
                    files={f"{m}{i}.{e}": f"{m}{i}.{e}" for i in (1, 2, 3)
                           for m, e in (("rna", "h5"), ("atac", "h5"), ("cty", "csv"))}),
    "D58mini": dict(src="D58", cells=1500, genes=1000, peaks=5000,
                    files={"adt1.h5": "adt1.h5", "adt2.h5": "adt2.h5", "atac1.h5": "peak1.h5",
                           "atac2.h5": "peak2.h5", "cty1.csv": "cty1.csv", "cty2.csv": "cty2.csv"}),
    "D59mini": dict(src="D59", cells=1500, genes=1000, peaks=5000,
                    files={"rna1.h5": "rna1.h5", "rna2.h5": "rna2.h5", "adt1.h5": "adt1.h5",
                           "adt2.h5": "adt2.h5", "atac1.h5": "peak1.h5", "atac2.h5": "peak2.h5",
                           "cty1.csv": "cty1.csv", "cty2.csv": "cty2.csv"}),
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
    of the cells. Counted over the peak files (``{path: cells}``), among the
    peaks that every file lists."""
    paths = list(files)
    names = _names(paths[0])
    for p in paths[1:]:                 # D37's batches differ in a few peaks
        names = names[np.isin(names, _names(p))]
    open_in, total = np.zeros(len(names)), 0
    for p, cells in files.items():
        own = _names(p)
        at = {name: i for i, name in enumerate(own)}
        rows = np.array([at[name] for name in names])
        with h5py.File(p) as f:
            open_in += (f["matrix/data"][:, cells][rows] > 0).sum(axis=1)
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


def _same_cell_order(a: Path, b: Path) -> bool:
    """Whether two files of one assay list the same cells in the same order,
    judged by the counts per cell: their correlation is near 1 in the same
    order (0.99 on D37) and near 0 in any other."""
    with h5py.File(a) as f, h5py.File(b) as g:
        x, y = f["matrix/data"][()].sum(axis=0), g["matrix/data"][()].sum(axis=0)
    return len(x) == len(y) and np.corrcoef(x, y)[0, 1] > 0.9


def _take_barcodes(dst: Path, src: Path) -> None:
    """Write the cell names of ``src`` into ``dst``."""
    with h5py.File(src) as f, h5py.File(dst, "r+") as g:
        for name in ("matrix/barcodes", "matrix/.data_dimnames/1"):
            if name in f and name in g:
                del g[name]
                g.create_dataset(name, data=f[name][()], compression="gzip")


def make(src: Path, dst: Path, cells, genes=None, peaks=None, files=None,
         barcodes_from=None) -> None:
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
    for out, other in (barcodes_from or {}).items():
        assert _same_cell_order(pairs[out], pairs[other]), f"{out}: not the cells of {other}"
        _take_barcodes(dst / out, dst / other)
        print(f"  {out}: cell names of {other}")


def main(argv):
    data_path, out = Path(argv[1]), Path(argv[2])
    only = set(argv[3:])                # dataset names; none: all
    for name, spec in PLAN.items():
        if only and name not in only:
            continue
        if not (data_path / spec["src"]).is_dir():
            print(name, "skipped:", spec["src"], "is not in", data_path)
            continue
        print(name, "from", spec["src"])
        make(data_path / spec["src"], out / name, spec["cells"], spec["genes"],
             spec["peaks"], spec.get("files"), spec.get("barcodes_from"))
        with tarfile.open(out / f"{name}.tar.gz", "w:gz") as t:
            t.add(out / name, arcname=name)
        print(f"  -> {name}.tar.gz {(out / f'{name}.tar.gz').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv)
