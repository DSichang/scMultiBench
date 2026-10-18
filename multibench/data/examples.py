"""Example files for the own-data section of the tutorials.

``write_example`` takes a part of a tutorial dataset and writes it as the
``.h5ad`` files a reader's own data usually comes in, so that the section runs
before the reader has put a file of their own in place.

Only the standard library is imported here: ``import multibench`` loads this
module, and anndata and scanpy are imported when ``write_example`` runs.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["write_example"]

#: the example files of each dataset, in writing order
EXAMPLES: dict[str, tuple[str, ...]] = {
    "D11": ("my_file.h5ad",),
    "D22mini": ("my_file.h5ad", "my_atac.h5ad"),
    "D27mini": ("my_rna.h5ad", "my_atac.h5ad"),
    "D27mini_vertical": ("my_rna.h5ad", "my_atac.h5ad"),
    "D37mini": ("my_rna1.h5ad", "my_rna2.h5ad", "my_rna3.h5ad",
                "my_atac1.h5ad", "my_atac2.h5ad", "my_atac3.h5ad"),
    "D38mini": ("my_rna1.h5ad", "my_rna2.h5ad", "my_adt2.h5ad", "my_adt3.h5ad"),
    "D45mini": ("my_rna1.h5ad", "my_rna2.h5ad", "my_atac2.h5ad", "my_atac3.h5ad"),
    "D46mini": ("my_rna1.h5ad", "my_adt1.h5ad", "my_rna2.h5ad", "my_atac2.h5ad",
                "my_rna3.h5ad"),
    "D49mini": ("my_rna1.h5ad", "my_adt1.h5ad", "my_rna2.h5ad", "my_atac2.h5ad",
                "my_adt3.h5ad"),
    "D52mini": ("my_file.h5ad",),
    "D56mini": ("my_rna.h5ad", "my_atac.h5ad"),
    "D58mini": ("my_adt.h5ad", "my_atac.h5ad"),
    "D59mini": ("my_file.h5ad", "my_atac.h5ad"),
}

#: HDF5 root attribute of an example file; its value is ``<dataset>@<version>``
_MARK = "multibench_example"

#: what the printed line says X holds
_RNA = "RNA counts in X"
_RNA_ADT = 'RNA counts in X, protein counts in obsm["protein"]'
_ADT = "protein counts in X"
_PEAK = "ATAC peak counts in X"
_GAS = "ATAC gene-activity scores in X"


# Each builder returns [(file name, AnnData, what X holds)]. The subsample
# fractions, seeds, cuts and concat arguments fix the folder a tutorial
# exports from the example files: change none of them.

def _labels(path):
    import pandas as pd
    return pd.read_csv(path)["x"].values


def _with_obs_of(a, main):
    """``a`` (the cells of ``main``, in its order) with ``main``'s cell-type and
    batch columns."""
    for col in ("celltype", "batch"):
        if col in main.obs:
            a.obs[col] = main.obs[col].values
    return a


def _cite(d, read):
    """RNA and protein of the same cells in one object (D11)."""
    import scanpy as sc

    adata = read(d / "rna.h5")
    adata.obsm["protein"] = read(d / "adt.h5").to_df()
    adata.obs["celltype"] = _labels(d / "cty.csv")
    adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
    return [("my_file.h5ad", adata, _RNA_ADT)]


def _cite_atac(d, read):
    """RNA and protein in one object, peaks of the same cells in another (D22mini)."""
    import scanpy as sc

    adata = read(d / "rna.h5")
    adata.obsm["protein"] = read(d / "adt.h5").to_df()
    adata.obs["celltype"] = _labels(d / "cty.csv")
    atac = read(d / "atac.h5")
    adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
    atac = atac[adata.obs_names].copy()
    return [("my_file.h5ad", adata, _RNA_ADT),
            ("my_atac.h5ad", _with_obs_of(atac, adata), _PEAK)]


def _rna_peaks(d, read):
    """RNA and peaks of the same cells, as two objects (D27mini_vertical)."""
    import scanpy as sc

    rna = read(d / "rna.h5")
    rna.obs["celltype"] = _labels(d / "cty.csv")
    atac = read(d / "atac_peak.h5")
    rna = sc.pp.subsample(rna, fraction=0.6, random_state=0, copy=True)
    atac = atac[rna.obs_names].copy()
    return [("my_rna.h5ad", rna, _RNA), ("my_atac.h5ad", _with_obs_of(atac, rna), _PEAK)]


def _unpaired(d, read):
    """RNA of some cells and gene activity of other cells (D27mini)."""
    import scanpy as sc

    rna = read(d / "rna.h5")
    rna.obs["celltype"] = _labels(d / "rna_cty.csv")
    atac = read(d / "atac_gas.h5")
    atac.obs["celltype"] = _labels(d / "atac_cty.csv")
    rna = sc.pp.subsample(rna, fraction=0.9, random_state=0, copy=True)
    atac = sc.pp.subsample(atac, fraction=0.9, random_state=1, copy=True)
    return [("my_rna.h5ad", rna, _RNA), ("my_atac.h5ad", atac, _GAS)]


def _unpaired_batches(d, read):
    """Three RNA batches and three gene-activity batches of other cells (D37mini)."""
    n = 900
    rna, atac = [], []
    for b in (1, 2, 3):
        r = read(d / f"rna{b}.h5")
        r.obs["celltype"] = _labels(d / f"rna_cty{b}.csv")
        a = read(d / f"atac_gas{b}.h5")
        a.obs["celltype"] = _labels(d / f"atac_cty{b}.csv")
        rna.append(r[:n].copy())
        atac.append(a[:n].copy())
    return [(f"my_rna{b}.h5ad", rna[b - 1], _RNA) for b in (1, 2, 3)] \
        + [(f"my_atac{b}.h5ad", atac[b - 1], _GAS) for b in (1, 2, 3)]


def _mosaic(d, read, *, n, files):
    """One object per batch and modality: the first ``n`` cells of each file
    of ``files``, ``(stem, batch number, what X holds)``."""
    labels = [_labels(d / f"cty{b}.csv")[:n] for b in (1, 2, 3)]
    out = []
    for stem, b, what in files:
        a = read(d / f"{stem}{b}.h5")[:n].copy()
        a.obs["celltype"] = labels[b - 1]
        out.append((f"my_{stem}{b}.h5ad", a, what))
    return out


def _cross_cite(d, read, *, keys, atac=False):
    """RNA and protein of every batch in one object with a batch column
    (D52mini), and with ``atac`` the peaks of the same cells in another
    (D59mini)."""
    import anndata as ad
    import scanpy as sc

    batches, peaks = [], []
    for b in keys:
        a = read(d / f"rna{b}.h5")
        a.obsm["protein"] = read(d / f"adt{b}.h5").to_df()
        a.obs["celltype"] = _labels(d / f"cty{b}.csv")
        batches.append(a)
        if atac:
            peaks.append(read(d / f"atac{b}.h5"))
    adata = ad.concat(batches, label="batch", keys=list(keys), index_unique="-")
    if atac:
        peaks = ad.concat(peaks, keys=list(keys), index_unique="-")
    adata = sc.pp.subsample(adata, fraction=0.6, random_state=0, copy=True)
    out = [("my_file.h5ad", adata, _RNA_ADT)]
    if atac:
        peaks = peaks[adata.obs_names].copy()
        out.append(("my_atac.h5ad", _with_obs_of(peaks, adata), _PEAK))
    return out


def _cross_pair(d, read, *, keys, stem, what):
    """One modality of every batch in an object with a batch column, and the
    peaks of the same cells in another (D56mini: RNA; D58mini: protein)."""
    import anndata as ad
    import scanpy as sc

    main, atac = [], []
    for b in keys:
        a = read(d / f"{stem}{b}.h5")
        a.obs["celltype"] = _labels(d / f"cty{b}.csv")
        main.append(a)
        atac.append(read(d / f"atac{b}.h5"))
    main = ad.concat(main, label="batch", keys=list(keys), index_unique="-")
    atac = ad.concat(atac, keys=list(keys), index_unique="-")
    main = sc.pp.subsample(main, fraction=0.6, random_state=0, copy=True)
    atac = atac[main.obs_names].copy()
    return [(f"my_{stem}.h5ad", main, what), ("my_atac.h5ad", _with_obs_of(atac, main), _PEAK)]


_BUILD = {
    "D11": (_cite, {}),
    "D22mini": (_cite_atac, {}),
    "D27mini": (_unpaired, {}),
    "D27mini_vertical": (_rna_peaks, {}),
    "D37mini": (_unpaired_batches, {}),
    "D38mini": (_mosaic, dict(n=1500, files=[("rna", 1, _RNA), ("rna", 2, _RNA),
                                              ("adt", 2, _ADT), ("adt", 3, _ADT)])),
    "D45mini": (_mosaic, dict(n=1500, files=[("rna", 1, _RNA), ("rna", 2, _RNA),
                                              ("atac", 2, _PEAK), ("atac", 3, _PEAK)])),
    "D46mini": (_mosaic, dict(n=1000, files=[("rna", 1, _RNA), ("adt", 1, _ADT),
                                              ("rna", 2, _RNA), ("atac", 2, _PEAK),
                                              ("rna", 3, _RNA)])),
    "D49mini": (_mosaic, dict(n=1000, files=[("rna", 1, _RNA), ("adt", 1, _ADT),
                                              ("rna", 2, _RNA), ("atac", 2, _PEAK),
                                              ("adt", 3, _ADT)])),
    "D52mini": (_cross_cite, dict(keys=("1", "2", "3"))),
    "D56mini": (_cross_pair, dict(keys=("1", "2", "3"), stem="rna", what=_RNA)),
    "D58mini": (_cross_pair, dict(keys=("1", "2"), stem="adt", what=_ADT)),
    "D59mini": (_cross_cite, dict(keys=("1", "2"), atac=True)),
}


def _mark_of(path: Path) -> str | None:
    """The example marker of ``path`` (``'D11@0.3.5'``), or ``None`` for a
    file without one or one that is not HDF5."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            mark = f.attrs.get(_MARK)
    except OSError:
        return None
    if mark is None:
        return None
    return mark.decode() if isinstance(mark, bytes) else str(mark)


def _write_marked(adata, target: Path, mark: str) -> None:
    """Write ``adata`` to ``target`` with the example marker, through a
    temporary file next to it: a failed write leaves no file under the
    target's name."""
    import tempfile

    import h5py

    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".tmp")
    os.close(fd)
    try:
        adata.write_h5ad(tmp, compression="gzip")
        with h5py.File(tmp, "a") as f:
            f.attrs[_MARK] = mark
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_example(dataset: str, folder: Path | str = ".", *,
                  data_path=None, quiet: bool = False) -> Path:
    """Write the example files of a tutorial dataset as .h5ad files.

    The files hold a part of the dataset in the shape that your own data
    usually has: one ``.h5ad`` file per object, with the cell types in
    ``obs["celltype"]``. The own-data section of each tutorial reads them
    until you put your own files in their place.

    Parameters
    ----------
    dataset : str
        Id of a tutorial dataset, such as ``"D11"``. It has to be in the data
        root: ``mtb.data.fetch(dataset)`` downloads it.
    folder : Path | str
        Folder the files are written to; it is created when it does not
        exist. The default is the current folder.
    data_path : Path | str | None
        Data root that holds the dataset folder; ``None`` =
        ``mtb.config.DEFAULT.data_path``.
    quiet : bool
        Suppress the one line printed per file.

    Returns
    -------
    pathlib.Path
        ``folder``, which holds the files, named ``my_*.h5ad``; the printed
        lines say what each holds. A file of your own with such a name is kept.

    Raises
    ------
    ValueError
        No example is defined for ``dataset`` (the message lists the ids).
    FileNotFoundError
        The dataset is not in the data root; the message gives the ``fetch`` call.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.data.fetch("D11")
    >>> mtb.data.write_example("D11")       # my_file.h5ad: RNA in X, ADT in obsm

    See Also
    --------
    mtb.data.fetch : download the dataset that the example is made from.
    mtb.io.export_dataset : write the objects read from the files as a dataset folder.
    """
    from .. import __version__, config
    from ..engine.ingest import read_canonical

    if dataset not in EXAMPLES:
        raise ValueError(f"No example is defined for {dataset!r}. Examples exist for "
                         f"{', '.join(EXAMPLES)}.")
    root = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    if not (root / dataset).is_dir():
        raise FileNotFoundError(f"{root / dataset} does not exist. Run "
                                f"mtb.data.fetch({dataset!r}) first.")
    folder = Path(folder)
    mark = f"{dataset}@{__version__}"
    # a file without the marker is the reader's; this dataset's example is kept
    state = {}
    for name in EXAMPLES[dataset]:
        target = folder / name
        if not (target.exists() or target.is_symlink()):
            state[name] = "write"
        else:
            found = _mark_of(target)
            state[name] = "kept" if found is None else "same" if found == mark else "write"
    built = {}
    if "write" in state.values():
        build, kw = _BUILD[dataset]
        made = build(root / dataset, read_canonical, **kw)
        if tuple(name for name, _, _ in made) != EXAMPLES[dataset]:
            raise RuntimeError(f"The example of {dataset} is built as "
                               f"{[name for name, _, _ in made]}, and EXAMPLES lists "
                               f"{list(EXAMPLES[dataset])}.")
        built = {name: (a, what) for name, a, what in made}
        folder.mkdir(parents=True, exist_ok=True)
    for name in EXAMPLES[dataset]:
        target = folder / name
        if state[name] == "kept":
            line = (f"{target} is already here and is not an example file of multibench. "
                    f"It was left as it is.")
        elif state[name] == "same":
            line = (f"{target} is already here (the example of {dataset}). Delete the file "
                    f"to write it again.")
        else:
            a, what = built[name]
            if "protein" in a.obsm:
                # an array and its names, as a file from another tool holds them
                import numpy as np

                df = a.obsm["protein"]
                a.uns["protein_names"] = np.asarray(df.columns, dtype=object)
                a.obsm["protein"] = df.to_numpy()
            _write_marked(a, target, mark)
            parts = [what, 'cell types in obs["celltype"]'] \
                + (['batches in obs["batch"]'] if "batch" in a.obs else [])
            line = f"wrote {target}: {a.n_obs:,} cells. {', '.join(parts)}."
        if not quiet:
            print(line)
    return folder
