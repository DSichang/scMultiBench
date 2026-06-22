"""Adapt arbitrary input formats to the canonical scMultiBench .h5.

Canonical layout: matrix/data (features x cells), matrix/features, matrix/barcodes.
Method scripts are never modified.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _is_canonical_h5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return "matrix/data" in f
    except (OSError, KeyError):
        return False


def _to_anndata(src):
    import anndata as ad
    import pandas as pd

    if hasattr(src, "X") and hasattr(src, "obs"):     # already AnnData
        return src
    p = Path(src)
    suf = p.suffix.lower()
    if suf == ".h5ad":
        return ad.read_h5ad(p)
    if suf in (".csv", ".tsv"):
        sep = "," if suf == ".csv" else "\t"
        df = pd.read_csv(p, sep=sep, index_col=None)
        return ad.AnnData(df.to_numpy(dtype=float))
    if suf == ".loom":
        try:
            import loompy  # noqa: F401  (anndata.read_loom needs it)
        except ModuleNotFoundError as exc:
            raise ImportError(
                "reading .loom requires the optional 'loompy' package "
                "(pip install 'multibench[loom]' or pip install loompy); "
                "alternatively convert the input to .h5ad/.csv first."
            ) from exc
        return ad.read_loom(p)
    raise ValueError(f"unsupported input format: {p.name}")


def to_canonical(src, out: Path | str | None = None, modality: str | None = None,
                 convert: bool = True) -> Path:
    """Convert `src` to a canonical .h5; return the path. Passthrough if already canonical."""
    # A path that is already a canonical .h5 is ALWAYS returned as-is: there is
    # nothing to convert, even when run() passes convert=True and an out path.
    if isinstance(src, (str, Path)) and _is_canonical_h5(Path(src)):
        return Path(src)
    if out is None:
        raise ValueError("out path required to write a canonical .h5")
    out = Path(out)

    adata = _to_anndata(src)
    X = np.asarray(adata.X)
    if hasattr(X, "toarray"):
        X = X.toarray()
    data = X.T  # cells x genes -> genes x cells
    feats = [str(v) for v in getattr(adata, "var_names", range(X.shape[1]))]
    bars = [str(v) for v in getattr(adata, "obs_names", range(X.shape[0]))]

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("matrix/data", data=np.asarray(data, dtype=float))
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))
    return out


def read_canonical(path: Path | str):
    """Inverse: canonical .h5 -> AnnData (cells x genes)."""
    import anndata as ad
    with h5py.File(path, "r") as f:
        data = np.array(f["matrix/data"]).T  # genes x cells -> cells x genes
        a = ad.AnnData(np.asarray(data, dtype=float))
        if "matrix/features" in f:
            a.var_names = [x.decode() if isinstance(x, bytes) else str(x)
                           for x in np.array(f["matrix/features"])]
        if "matrix/barcodes" in f:
            a.obs_names = [x.decode() if isinstance(x, bytes) else str(x)
                           for x in np.array(f["matrix/barcodes"])]
    return a
