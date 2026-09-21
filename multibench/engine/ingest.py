"""Adapt arbitrary input formats to the canonical scMultiBench .h5.

Canonical layout: matrix/data (features x cells), matrix/features, matrix/barcodes.
Method scripts are never modified.

Entry points (exposed as ``mtb.io``):

* :func:`to_canonical` - one matrix (AnnData / .h5ad / .h5mu / .csv / .tsv /
  .loom) -> one canonical ``.h5``; sparse-safe, gzip-compressed, with
  ``modality=`` / ``layer=`` / ``obsm=`` / ``mod=`` selectors.
* :func:`export_dataset` - a whole AnnData/MuData -> a canonical dataset folder
  (``rna.h5``, ``adt.h5``, ``atac_peak.h5``/``atac_gas.h5``, ``cty.csv``),
  optionally split per batch (``rna1.h5`` ...). A MuData's modalities can be
  named directly (``rna='rna'``).
* :func:`read_canonical` - the inverse (canonical ``.h5`` -> AnnData).
* :func:`normalize_peak_names` - rewrite peak ids to ``chr:start-end``.

The single-column ``x`` label CSV the benchmark reads is written by the
private :func:`_write_labels` (``export_dataset`` calls it per batch).
"""
from __future__ import annotations

import os
import re
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np

__all__ = ["export_dataset", "to_canonical", "read_canonical", "normalize_peak_names"]


def __dir__() -> list[str]:
    """Tab completion (``dir(mtb.io)``) shows the public API and the
    underscore names, not the imports this module merely uses (PEP 562)."""
    return sorted(n for n in globals() if n in __all__ or n.startswith("_"))

# ATAC peak ids come in chr_start_end / chr-start-end / chr:start-end flavours.
# Signac CreateChromatinAssay(sep=c(":","-")) (Seurat_v3 etc.) needs chr:start-end.
_PEAK_RE = re.compile(r"^(.+?)[-_:](\d+)[-_](\d+)$")

# Modality roles a canonical file can be written for, the user-facing aliases,
# and the on-disk filename each role gets when ``out`` is a directory. The
# names follow engine/resolve._ROLE_FILE_CANDIDATES and the shipped datasets
# (D11: rna.h5/adt.h5; D28: rna.h5/atac_peak.h5/atac_gas.h5).
_MODALITIES = ("rna", "adt", "atac", "atac_peak", "atac_gas")
_ALIASES = {"protein": "adt", "peak": "atac_peak", "gas": "atac_gas",
            "gene_activity": "atac_gas"}
_MOD_FILE = {"rna": "rna.h5", "adt": "adt.h5", "atac": "atac.h5",
             "atac_peak": "atac_peak.h5", "atac_gas": "atac_gas.h5"}
_ATAC_KINDS = ("peak", "gene_activity")
#: :func:`to_canonical` warns when ``matrix/data`` (stored dense, features x
#: cells) would exceed this many bytes uncompressed on disk.
DENSE_WARN_BYTES = 10 ** 9
#: ``category=`` values accepted by to_canonical / export_dataset. Only
#: ``vertical`` changes anything (its ATAC role reads plain ``atac.h5``).
_CATEGORIES = ("vertical", "diagonal", "mosaic", "cross")


def _check_category(category):
    if category is None:
        return None
    if category not in _CATEGORIES:
        raise ValueError(f"unknown category {category!r}; valid: {list(_CATEGORIES)}")
    return category


def _atac_filename(modality: str, category: str | None) -> str:
    """On-disk name of an ATAC-family canonical file.

    ``atac.h5`` is the one name every vertical (paired multiome) variant
    resolves, whatever representation the method wants
    (``method_info(m)['atac']``): it is the ``atac`` role's file and the
    ``atac_gas`` role's fallback, and no vertical variant reads
    ``atac_peak.h5``. So ``category='vertical'`` maps every ATAC modality to
    ``atac.h5``; any other category (or None) keeps the representation-named
    file.
    """
    if category == "vertical" and modality in ("atac", "atac_peak", "atac_gas"):
        return "atac.h5"
    return _MOD_FILE[modality]


def _is_canonical_h5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return "matrix/data" in f
    except (OSError, KeyError):
        return False


def _is_mudata(obj) -> bool:
    # Duck-typed so ``mudata`` stays an optional import.
    return hasattr(obj, "mod") and isinstance(getattr(obj, "mod", None), dict) \
        and type(obj).__name__ == "MuData"


def _to_anndata(src):
    """Load ``src`` into an AnnData (or MuData for ``.h5mu``); in-memory objects
    are returned as-is."""
    import anndata as ad
    import pandas as pd

    if hasattr(src, "X") and hasattr(src, "obs"):     # already AnnData / MuData
        return src
    p = Path(src)
    if not p.exists():
        # h5py's own error ("Unable to synchronously open file ... errno = 2")
        # does not read like a missing file, nor name the cwd a relative path
        # was tried against.
        raise FileNotFoundError(
            f"input file does not exist: {p} (cwd {os.getcwd()})")
    suf = p.suffix.lower()
    if suf == ".h5":
        # exists but is not canonical (to_canonical passes canonical files
        # through before reaching here): say what is inside
        try:
            with h5py.File(p, "r") as f:
                keys = sorted(f.keys())
        except OSError as exc:
            raise ValueError(f"{p} is not a readable HDF5 file: {exc}") from exc
        raise ValueError(
            f"{p} has no dataset 'matrix/data'; found keys {keys} - a canonical "
            f"input .h5 holds matrix/data (features x cells), matrix/features and "
            f"matrix/barcodes"
            + (" (a top-level 'data' dataset is a method OUTPUT such as "
               "out/<method>/embedding.h5, which mtb.evaluate reads)" if "data" in keys else "")
            + "; pass an AnnData / .h5ad / .csv to convert instead")
    if suf == ".h5ad":
        return ad.read_h5ad(p)
    if suf == ".h5mu":
        try:
            import mudata
        except ModuleNotFoundError as exc:
            raise ImportError(
                "reading .h5mu requires the optional 'mudata' package "
                "(pip install mudata); alternatively pass the modality's "
                ".h5ad / AnnData directly."
            ) from exc
        return mudata.read_h5mu(p)
    if suf in (".csv", ".tsv"):
        sep = "," if suf == ".csv" else "\t"
        df = pd.read_csv(p, sep=sep)
        # A leading non-numeric column is row labels (cell barcodes): use it as
        # the index (obs_names) instead of feeding it to the numeric matrix.
        if df.shape[1] > 1 and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            df = df.set_index(df.columns[0])
        a = ad.AnnData(df.to_numpy(dtype=float))
        a.obs_names = [str(x) for x in df.index]
        a.var_names = [str(c) for c in df.columns]
        return a
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
    if not p.exists():
        raise FileNotFoundError(f"input file does not exist: {p}")
    raise ValueError(f"unsupported input format: {p.name}")


def _norm_modality(modality):
    """Resolve aliases and validate; ``None`` passes through."""
    if modality is None:
        return None
    m = _ALIASES.get(str(modality), str(modality))
    if m not in _MODALITIES:
        raise ValueError(
            f"unknown modality {modality!r}; valid: {list(_MODALITIES)} "
            f"(aliases: {_ALIASES})")
    return m


def _pick_matrix(adata, *, layer=None, obsm=None, feature_names=None, what=None):
    """Return ``(X, feature_names)`` for the requested slot of ``adata``.

    ``obsm`` matrices carry no var axis in AnnData, so feature names come from
    ``feature_names`` when given, else the columns of a DataFrame-valued obsm,
    else ``adata.uns[f"{obsm}_names"]`` (when its length matches), else
    ``feature_0..`` - with a ``UserWarning`` naming the fallback, because a
    protein panel written as ``feature_0..feature_29`` loses its marker names
    in every downstream readout. ``what`` labels the warning (``'adt'``).
    """
    if layer is not None and obsm is not None:
        raise ValueError("pass at most one of layer= / obsm=")
    if obsm is not None:
        if obsm not in adata.obsm:
            raise KeyError(f"obsm key {obsm!r} not found; available: {list(adata.obsm)}")
        X = adata.obsm[obsm]
        names = None
        if hasattr(X, "columns"):                       # DataFrame
            names = [str(c) for c in X.columns]
            X = X.to_numpy()
        n_feat = X.shape[1] if getattr(X, "ndim", 0) == 2 else None
        if feature_names is not None:
            names = [str(v) for v in feature_names]
            if n_feat is not None and len(names) != n_feat:
                raise ValueError(f"{len(names)} feature names for {n_feat} features "
                                 f"in obsm[{obsm!r}]")
            return X, names
        uns_names = adata.uns.get(f"{obsm}_names") if hasattr(adata, "uns") else None
        if names is None and uns_names is not None and n_feat is not None \
                and len(uns_names) == n_feat:
            names = [str(v) for v in uns_names]
        if names is None and n_feat is not None:
            names = [f"feature_{i}" for i in range(n_feat)]
            warnings.warn(
                f"{what or 'obsm:' + obsm}: no feature names found (obsm[{obsm!r}] is a "
                f"bare array and adata.uns[{obsm + '_names'!r}] is absent); writing "
                f"feature_0..feature_{n_feat - 1} - pass feature_names=[...] "
                f"(export_dataset: adt_names=) or store a DataFrame in obsm",
                UserWarning, stacklevel=3)
        return X, names
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"layer {layer!r} not found; available: {list(adata.layers)}")
        X = adata.layers[layer]
    else:
        X = adata.X
    if feature_names is not None:
        names = [str(v) for v in feature_names]
        if len(names) != X.shape[1]:
            raise ValueError(f"{len(names)} feature names for {X.shape[1]} features")
        return X, names
    return X, [str(v) for v in adata.var_names]


def _warn_dense_size(out: Path, n_feat: int, n_cell: int, dtype) -> None:
    """Warn when the dense ``matrix/data`` will exceed :data:`DENSE_WARN_BYTES`.

    The canonical layout stores the matrix dense (features x cells x
    itemsize); gzip shrinks a sparse matrix on disk but every reader
    densifies it, and a 100k-cell x 200k-peak multiome is 160 GB.
    """
    itemsize = int(np.dtype(dtype).itemsize)
    size = int(n_feat) * int(n_cell) * itemsize
    if size > DENSE_WARN_BYTES:
        warnings.warn(
            f"{out.name}: matrix/data is stored DENSE (features x cells): "
            f"{n_feat} x {n_cell} x {itemsize} B = {size / 1e9:.1f} GB uncompressed "
            f"on disk (limit for this warning: {DENSE_WARN_BYTES / 1e9:.0f} GB); "
            f"filter to highly-variable genes / informative peaks before export, "
            f"or pass dtype='float32' to halve it",
            UserWarning, stacklevel=3)


def _peak_fraction(feats) -> float:
    if not len(feats):
        return 0.0
    return sum(1 for x in feats if _PEAK_RE.match(x)) / len(feats)


def _check_peak_names(modality, feats):
    """Warn when the feature names contradict the declared ATAC role."""
    if modality not in ("atac_gas", "atac_peak"):
        return
    frac = _peak_fraction(feats)
    if modality == "atac_gas" and frac > 0.5:
        warnings.warn(
            f"modality='atac_gas' but {frac:.0%} of the features look like peaks "
            "(chr:start-end), not gene activity; did you mean modality='atac_peak' / "
            "atac_kind='peak'?", UserWarning, stacklevel=3)
    elif modality == "atac_peak" and frac < 0.5:
        warnings.warn(
            f"modality='atac_peak' but only {frac:.0%} of the features look like "
            "peaks (chr:start-end); did you mean modality='atac_gas' / "
            "atac_kind='gene_activity'?", UserWarning, stacklevel=3)


def _write_canonical(out: Path, X, feats, bars, *, dtype, compression, block):
    """Stream ``X`` (cells x features, dense or sparse) to ``out`` as
    features x cells. Sparse input is converted to CSC so that each block of
    features is a contiguous column slice that densifies cheaply."""
    import scipy.sparse as sp

    if sp.issparse(X):
        X = X.tocsc()
    else:
        X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"matrix must be 2-D (cells x features), got shape {X.shape}")
    n_cell, n_feat = X.shape
    if len(feats) != n_feat:
        raise ValueError(f"{len(feats)} feature names for {n_feat} features")
    if len(bars) != n_cell:
        raise ValueError(f"{len(bars)} barcodes for {n_cell} cells")
    block = max(1, int(block))
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        d = f.create_dataset("matrix/data", shape=(n_feat, n_cell), dtype=dtype,
                             chunks=True if compression else None,
                             compression=compression)
        for i in range(0, n_feat, block):
            blk = X[:, i:i + block]
            if sp.issparse(blk):
                blk = blk.toarray()
            d[i:i + block, :] = np.asarray(blk).T.astype(dtype, copy=False)
        f.create_dataset("matrix/features", data=np.array(feats, dtype="S"))
        f.create_dataset("matrix/barcodes", data=np.array(bars, dtype="S"))
    return out


def to_canonical(src, out: Path | str | None = None, modality: str | None = None,
                 convert: bool = True, *, layer: str | None = None,
                 obsm: str | None = None, mod: str | None = None,
                 dtype: str = "float64", compression: str | None = "gzip",
                 block: int = 1024, category: str | None = None,
                 feature_names: list | None = None) -> Path:
    """Convert one matrix to a canonical scMultiBench ``.h5`` and return its path.

    The canonical layout is ``matrix/data`` (features x cells),
    ``matrix/features`` and ``matrix/barcodes``. A path that is already a
    canonical ``.h5`` is returned untouched, whatever the other arguments.

    Parameters
    ----------
    src : AnnData, MuData, Path or str
        Matrix to convert: an AnnData, a MuData (with ``mod``), or a path to
        ``.h5ad`` / ``.h5mu`` / ``.csv`` / ``.tsv`` (cells x features) /
        ``.loom`` / canonical ``.h5``.
    out : Path | str | None
        Output file, or a directory (existing or ending in ``/``) that gets
        the ``modality`` filename; ``None`` = that filename in the current
        directory (needs ``modality``).
    modality : str | None
        ``'rna'``, ``'adt'``, ``'atac'``, ``'atac_peak'`` or ``'atac_gas'``;
        names the file when ``out`` is a directory or ``None``, and enables
        the ADT / ATAC checks (Notes).
    convert : bool
        ``False`` = never write: pass a canonical ``.h5`` through, raise for
        anything else.
    layer : str | None, keyword-only
        Take the matrix from ``adata.layers[layer]`` instead of ``adata.X``.
    obsm : str | None, keyword-only
        Take the matrix from ``adata.obsm[obsm]`` (e.g. ``'protein'`` for CITE-seq).
    mod : str | None, keyword-only
        MuData modality to convert (``mdata.mod[mod]``); required for a MuData.
    dtype : str, keyword-only
        Stored dtype of ``matrix/data``; ``'float32'`` halves the file (see Notes).
    compression : str | None, keyword-only
        h5py compression filter; ``None`` = uncompressed.
    block : int, keyword-only
        Number of features written per streaming step.
    category : str | None, keyword-only
        Integration category the file is for (``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``); only changes ATAC filenames (Notes).
    feature_names : list | None, keyword-only
        Feature names that override ``var_names`` / ``uns`` / DataFrame
        columns; the way to name a bare ``obsm`` array.

    Returns
    -------
    pathlib.Path
        The canonical ``.h5`` written, or ``src`` itself on passthrough.

    Raises
    ------
    FileNotFoundError
        ``src`` is a path that does not exist.
    ValueError
        Invalid or conflicting arguments, or a non-canonical ``.h5`` (full list in Notes).
    KeyError
        ``layer``, ``obsm`` or ``mod`` names an entry the object does not have.
    ImportError
        ``.h5mu`` without ``mudata``, or ``.loom`` without ``loompy``.

    Warns
    -----
    UserWarning
        Feature names contradict an ATAC ``modality`` or are missing; a dense matrix over 1 GB.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.io.to_canonical(adata, "data/MYCITE/", modality="rna")     # data/MYCITE/rna.h5
    >>> mtb.io.to_canonical(adata, "data/MYCITE/", modality="adt", obsm="protein")
    >>> mtb.io.to_canonical("counts.csv", "rna.h5")                     # explicit file name
    >>> mtb.io.to_canonical(atac, "data/MYMULTI/", modality="peak", category="vertical")

    Notes
    -----
    **Output location.**

    - ``out`` a file path: written there; ``category`` does not rename it;
    - ``out`` a directory and ``modality`` given: the modality's canonical
      filename is appended (``rna.h5``, ``adt.h5``, ...);
    - ``out=None`` and ``modality`` given: that filename in the current
      directory; ``out=None`` without ``modality`` raises ``ValueError``.

    **Modality aliases and checks.** ``'protein'`` -> ``'adt'``,
    ``'peak'`` -> ``'atac_peak'``, ``'gas'`` / ``'gene_activity'`` ->
    ``'atac_gas'``. The ADT check is the ``obsm=`` rule under *ADT
    matrices*. ``'atac_peak'`` warns when fewer than half of the feature
    names look like peaks (``chr1:100-200``, ``chr1_100_200``,
    ``chr1-100-200``), ``'atac_gas'`` when more than half do; plain
    ``'atac'`` is not checked.

    **ADT matrices.** With ``modality='adt'`` on an AnnData that has
    ``obsm`` keys, ``obsm=`` or ``layer=`` must say where the protein matrix
    is; otherwise ``adata.X`` (usually the RNA) would be written as
    ``adt.h5``, so it raises. ``layer`` and ``obsm`` are mutually exclusive.
    The feature names of an ``obsm`` matrix come from the first of:

    1. ``feature_names``;
    2. the columns, when the ``obsm`` entry is a DataFrame;
    3. ``adata.uns[f'{obsm}_names']``, when its length matches;
    4. ``feature_0..``, with a ``UserWarning`` (the protein names would be
       lost in every downstream readout).

    **ATAC filenames.** Without ``category``, and for ``'diagonal'`` /
    ``'mosaic'`` / ``'cross'``, the file is named after the representation:
    ``atac_peak.h5``, ``atac_gas.h5``, or ``atac.h5`` for the plain
    ``'atac'`` role. ``category='vertical'`` writes ``atac.h5`` whatever the
    representation, because that is the one name every vertical (paired
    multiome) variant resolves and none reads ``atac_peak.h5``.

    **Check the ATAC kind.** Without ``category='vertical'``,
    ``to_canonical(atac, d, modality='peak')`` writes ``atac_peak.h5`` and
    ``mtb.scan(d, 'vertical')`` finds no ATAC method runnable. The
    representation is recorded NOWHERE on disk: whether a vertical method
    expects peaks or gene activity in ``atac.h5`` is
    ``method_info(m)['atac']``.

    **Streaming.** Sparse matrices (CSR/CSC, in memory or inside an
    ``.h5ad``/``.h5mu``) are converted to CSC and written ``block`` features
    at a time, without densifying the whole matrix. The defaults (gzip,
    ``'float64'``) match the shipped benchmark files; any compression
    enables chunking. A ``.csv`` / ``.tsv`` is read as cells x features, and
    a non-numeric first column is used as the cell barcodes.

    **Size on disk.** ``matrix/data`` is stored dense (features x cells x
    itemsize): gzip shrinks the file, but every reader densifies it. A
    ``UserWarning`` states the size when it exceeds ``DENSE_WARN_BYTES``
    (1 GB) and suggests filtering features or ``dtype='float32'``, which
    h5py / rhdf5 / hdf5r read (as double in R).

    **Errors.** ``ValueError`` is raised for:

    - ``out=None`` without ``modality``;
    - an unknown ``modality`` or ``category``;
    - ``convert=False`` on a non-canonical input;
    - an ``.h5`` without ``matrix/data`` (the message lists the keys it does
      hold - a top-level ``data`` dataset is a method output, not an input);
    - an unsupported suffix;
    - ``mod`` missing for a MuData, or given for anything else;
    - ``layer`` and ``obsm`` together;
    - ``modality='adt'`` on an AnnData with ``obsm`` keys but no ``obsm=`` /
      ``layer=``;
    - a feature-name or barcode count that does not match the matrix.

    The ``FileNotFoundError`` message names the path and the current
    directory; the ``KeyError`` message lists the entries the object has.

    See Also
    --------
    mtb.io.export_dataset : a whole dataset folder (all modalities, labels, batches) in one call.
    mtb.io.read_canonical : the inverse, canonical ``.h5`` -> AnnData.
    mtb.describe_layout : the folder layout and the role -> filename mapping.
    """
    # A canonical .h5 is returned as is, even when run() passes convert=True
    # and an out path: there is nothing to convert.
    if isinstance(src, (str, Path)) and _is_canonical_h5(Path(src)):
        return Path(src)
    if convert is False:
        raise ValueError(
            f"src is not a canonical .h5 and convert=False: {src if isinstance(src, (str, Path)) else type(src).__name__}")
    modality = _norm_modality(modality)
    category = _check_category(category)
    if out is None:
        if modality is None:
            raise ValueError("out path required to write a canonical .h5 "
                             "(or pass modality= to use the canonical filename)")
        out = Path(".") / _atac_filename(modality, category)
    else:
        is_dir_like = Path(out).is_dir() or str(out).endswith(("/", os.sep))
        out = Path(out)
        if modality is not None and is_dir_like:
            out = out / _atac_filename(modality, category)

    adata = _to_anndata(src)
    if _is_mudata(adata):
        if mod is None:
            raise ValueError(
                f"MuData given; pass mod=<name> to choose the modality (found: {list(adata.mod)})")
        if mod not in adata.mod:
            raise KeyError(f"mod {mod!r} not in MuData; found: {list(adata.mod)}")
        adata = adata.mod[mod]
    elif mod is not None:
        raise ValueError("mod= only applies to MuData input")

    if modality == "adt" and obsm is None and layer is None and len(adata.obsm):
        raise ValueError(
            f"adt requested but obsm= not given; found obsm keys {list(adata.obsm)} - "
            "pass obsm='<key>' (or layer=) to select the protein matrix, otherwise "
            "adata.X (usually the RNA) would be written as adt.h5")

    X, feats = _pick_matrix(adata, layer=layer, obsm=obsm,
                            feature_names=feature_names, what=modality)
    bars = [str(v) for v in adata.obs_names]
    _check_peak_names(modality, feats)
    if getattr(X, "ndim", 0) == 2:
        _warn_dense_size(out, X.shape[1], X.shape[0], dtype)
    return _write_canonical(out, X, feats, bars, dtype=dtype,
                            compression=compression, block=block)


def read_canonical(path: Path | str, sparse: bool | None = None):
    """Read a canonical ``.h5`` back into an AnnData (cells x features).

    The inverse of ``mtb.io.to_canonical``: ``matrix/data`` is transposed
    back to cells x features, ``matrix/features`` become ``var_names`` and
    ``matrix/barcodes`` become ``obs_names``.

    Parameters
    ----------
    path : Path | str
        Canonical ``.h5`` file.
    sparse : bool | None
        ``True`` = CSR ``.X``; ``False`` = dense ndarray; ``None`` = CSR when
        fewer than half of the entries are non-zero.

    Returns
    -------
    anndata.AnnData
        ``.X`` as float, cells x features; names taken from the file when it
        has them.

    Examples
    --------
    >>> import multibench as mtb
    >>> a = mtb.io.read_canonical("data/D11/rna.h5")
    >>> a.shape, a.var_names[:3]
    >>> dense = mtb.io.read_canonical("data/D11/adt.h5", sparse=False)

    Notes
    -----
    **Memory.** The whole matrix is read densely and transposed before the
    sparsity decision, so memory peaks at the dense size (features x cells
    x 8 bytes) even when the result is CSR. It is meant for the shipped
    benchmark inputs and for what ``to_canonical`` writes, not for a
    100k-cell peak matrix.

    **No validation.** A file without ``matrix/data`` raises h5py's
    ``KeyError``. ``matrix/features`` and ``matrix/barcodes`` are optional;
    without them AnnData's default names (``'0'``, ``'1'``, ...) are kept.

    See Also
    --------
    mtb.io.to_canonical : write a canonical file from an AnnData / path.
    """
    import anndata as ad
    import scipy.sparse as sp
    with h5py.File(path, "r") as f:
        data = np.array(f["matrix/data"]).T  # features x cells -> cells x features
        X = np.asarray(data, dtype=float)
        if sparse is None:
            sparse = X.size > 0 and np.count_nonzero(X) / X.size < 0.5
        a = ad.AnnData(sp.csr_matrix(X) if sparse else X)
        if "matrix/features" in f:
            a.var_names = [x.decode() if isinstance(x, bytes) else str(x)
                           for x in np.array(f["matrix/features"])]
        if "matrix/barcodes" in f:
            a.obs_names = [x.decode() if isinstance(x, bytes) else str(x)
                           for x in np.array(f["matrix/barcodes"])]
    return a


def normalize_peak_names(src, dst):
    """Copy a canonical ``.h5``, rewriting ATAC peak names to ``chr:start-end``.

    Signac's ``CreateChromatinAssay(sep=c(":","-"))`` (Seurat v3 and the
    other Signac-based methods) expects that spelling. The source file is
    never modified.

    Parameters
    ----------
    src : Path | str
        Canonical ``.h5`` whose ``matrix/features`` holds the peak ids.
    dst : Path | str
        Path of the copy; parent folders are created, an existing file is overwritten.

    Returns
    -------
    pathlib.Path
        ``dst``.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.io.normalize_peak_names("data/MYMULTI/atac_peak.h5", "tmp/atac_peak.h5")

    Notes
    -----
    **Recognised peaks.** Peak ids come as ``chr_start_end``,
    ``chr-start-end`` or ``chr:start-end``. A feature counts as a peak when
    it reads ``<chr><sep><start><sep><end>`` with two integers, the first
    separator one of ``_``, ``-``, ``:`` and the second ``_`` or ``-``; it
    is rewritten as ``<chr>:<start>-<end>``.

    **Everything else is kept.** Other features (gene symbols, ids without
    two numbers) pass through unchanged, so the file stays usable whatever
    the matrix holds. ``matrix/data`` and ``matrix/barcodes`` are copied as
    they are; only ``matrix/features`` is replaced.

    **Inside ``mtb.run``.** ``mtb.run`` applies this itself for the variants
    whose registry entry declares ``normalize_peaks`` roles, writing a
    per-run ``<role>_normpeaks.h5`` copy next to the converted inputs. Call
    it by hand only to prepare a file for a script you run outside the
    wrapper.

    See Also
    --------
    mtb.io.to_canonical : write the canonical file in the first place.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        feats = f["matrix/features"][:]
        def _norm(b):
            x = b.decode() if isinstance(b, bytes) else str(b)
            m = _PEAK_RE.match(x)
            return f"{m.group(1)}:{m.group(2)}-{m.group(3)}" if m else x
        new = np.array([_norm(v) for v in feats], dtype="S")
        del f["matrix/features"]
        f.create_dataset("matrix/features", data=new)
    return dst


# --------------------------------------------------------------------------
# dataset-level export
# --------------------------------------------------------------------------

def _write_labels(labels, path: Path | str) -> Path:
    """Write cell-type labels as the single-column CSV the benchmark reads
    (header ``x``, one label per line) and return the path.

    This is the shipped ``cty.csv`` format: ``eval.io.read_labels``, the
    reader behind ``evaluate`` and ``workflow._read_cty``, returns the column
    named ``x``.

    Parameters
    ----------
    labels : sequence
        Any 1-D sequence (list, ndarray, pandas Series / Categorical); values
        are written as strings.
    path : path-like
        Destination ``.csv`` path (parent directories are created).

    Returns
    -------
    pathlib.Path
        ``path``.
    """
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = np.asarray(labels)
    if vals.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {vals.shape}")
    pd.DataFrame({"x": [str(v) for v in vals]}).to_csv(path, index=False)
    return path


def _select(data, spec, *, what):
    """Parse a matrix selector into ``(adata, to_canonical kwargs)``.

    Selectors: ``'X'`` | ``'obsm:<key>'`` | ``'layer:<key>'`` |
    ``'mod:<name>'`` (MuData; ``adata.X`` of that modality) |
    ``'mod:<name>.obsm:<key>'`` / ``'mod:<name>.layer:<key>'``.
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"{what}= must be a selector string like 'X', 'obsm:<key>', "
                         f"'layer:<key>' or 'mod:<name>', got {spec!r}")
    adata = data
    rest = spec
    if rest.startswith("mod:"):
        if not _is_mudata(data):
            raise ValueError(f"{what}={spec!r}: 'mod:<name>' selectors need a MuData "
                             f"input, got {type(data).__name__}")
        name, _, rest = rest[4:].partition(".")
        if name not in data.mod:
            raise KeyError(f"{what}={spec!r}: mod {name!r} not in MuData; found: {list(data.mod)}")
        adata = data.mod[name]
        rest = rest or "X"
    elif _is_mudata(data):
        raise ValueError(f"{what}={spec!r}: MuData input needs a 'mod:<name>' selector "
                         f"(found mods: {list(data.mod)})")
    if rest == "X":
        return adata, {}
    if rest.startswith("obsm:"):
        return adata, {"obsm": rest[5:]}
    if rest.startswith("layer:"):
        return adata, {"layer": rest[6:]}
    raise ValueError(f"{what}={spec!r}: unknown selector; use 'X', 'obsm:<key>', "
                     f"'layer:<key>' or 'mod:<name>[.obsm:<key>|.layer:<key>]'")


def _select_obs(data, spec, *, what):
    """Parse an obs-column selector ``'obs:<col>'`` / ``'mod:<name>.obs:<col>'``
    (also ``'<name>:<col>'`` for MuData) into a pandas Series."""
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"{what}= must be 'obs:<col>' or 'mod:<name>.obs:<col>', got {spec!r}")
    obj = data
    rest = spec
    if rest.startswith("mod:"):
        if not _is_mudata(data):
            raise ValueError(f"{what}={spec!r}: 'mod:<name>' needs a MuData input")
        name, _, rest = rest[4:].partition(".")
        if name not in data.mod:
            raise KeyError(f"{what}={spec!r}: mod {name!r} not in MuData; found: {list(data.mod)}")
        obj = data.mod[name]
    elif _is_mudata(data) and not rest.startswith("obs:"):
        # MuData convenience: 'rna:celltype' = mod:rna.obs:celltype
        name, _, col = rest.partition(":")
        if name in data.mod and col:
            obj, rest = data.mod[name], f"obs:{col}"
    if not rest.startswith("obs:"):
        raise ValueError(f"{what}={spec!r}: use 'obs:<col>' (or 'mod:<name>.obs:<col>')")
    col = rest[4:]
    if col not in obj.obs.columns:
        raise KeyError(f"{what}={spec!r}: column {col!r} not in obs; available: "
                       f"{list(obj.obs.columns)}")
    return obj.obs[col]


def _link_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _is_anndata(obj) -> bool:
    return hasattr(obj, "X") and hasattr(obj, "obs") and hasattr(obj, "obs_names")


def _as_modality(data, spec, *, what, master, feature_names=None):
    """Turn a modality argument of :func:`export_dataset` into ``(adata, kw)``.

    ``spec`` is a selector string (resolved against ``data``), an AnnData, a
    DataFrame (index = cell barcodes, columns = features) or a 2-D array /
    sparse matrix in ``master`` order (no barcodes to check - pass a DataFrame
    or AnnData when the order is not known to be right).
    """
    import anndata as ad
    import scipy.sparse as sp

    if isinstance(spec, str):
        if data is None:
            raise ValueError(
                f"{what}={spec!r} is a selector but data is None; pass the AnnData/"
                f"MuData as the first argument, or give {what}= an AnnData / DataFrame")
        return _select(data, spec, what=what)
    if _is_anndata(spec):
        return spec, {}
    if hasattr(spec, "columns") and hasattr(spec, "index"):        # DataFrame
        a = ad.AnnData(spec.to_numpy(dtype=float))
        a.obs_names = [str(x) for x in spec.index]
        a.var_names = [str(c) for c in spec.columns]
        return a, {}
    if sp.issparse(spec) or isinstance(spec, np.ndarray):
        if getattr(spec, "ndim", 0) != 2:
            raise ValueError(f"{what}= array must be 2-D (cells x features), got shape "
                             f"{getattr(spec, 'shape', None)}")
        if master is None:
            raise ValueError(
                f"{what}= is a bare array, so the cell barcodes are unknown; pass an "
                f"AnnData/DataFrame for {what}= or give data= / another modality as "
                f"an AnnData first")
        if spec.shape[0] != len(master):
            raise ValueError(f"{what}= array has {spec.shape[0]} rows for {len(master)} "
                             f"cells (the master cell order)")
        a = ad.AnnData(spec if sp.issparse(spec) else np.asarray(spec, dtype=float))
        a.obs_names = list(master)
        if feature_names is None:
            n_feat = spec.shape[1]
            warnings.warn(
                f"{what}: no feature names found (bare array); writing "
                f"feature_0..feature_{n_feat - 1} - pass {'adt_names' if what == 'adt' else 'a DataFrame'}"
                f"=[...] or a DataFrame / AnnData with named features",
                UserWarning, stacklevel=3)
            feature_names = [f"feature_{i}" for i in range(n_feat)]
        a.var_names = [str(x) for x in feature_names]
        return a, {}
    raise ValueError(
        f"{what}= must be a selector string like 'X', 'obsm:<key>', 'layer:<key>' or "
        f"'mod:<name>', an AnnData, a DataFrame (cells x features) or a 2-D array, "
        f"got {type(spec).__name__}")


def _align_cells(a, *, master, master_name, role):
    """Return ``a`` re-indexed to the master cell order.

    Same barcodes in another order -> reordered, so two modality files are
    never written mis-paired. A barcode set that differs raises ``ValueError``
    naming the strays. Non-unique barcodes on either side cannot be matched by
    name: the cells are paired positionally and a ``UserWarning`` says so (a
    different cell count still raises).
    """
    names = [str(x) for x in a.obs_names]
    master = [str(x) for x in master]
    if names == master:
        return a
    unique = len(set(names)) == len(names) and len(set(master)) == len(master)
    if unique and set(names) == set(master):
        return a[master]
    if unique:
        stray = [x for x in names if x not in set(master)]
        lack = [x for x in master if x not in set(names)]
        raise ValueError(
            f"{role} has {len(names)} cells but {len(stray)} barcodes are not in "
            f"{master_name} ({stray[:5]}{'...' if len(stray) > 5 else ''}"
            f"{'; ' + str(len(lack)) + ' of ' + master_name + ' missing from ' + role if lack else ''}); "
            f"all modalities of one dataset must cover the same cells, in one order - "
            f"subset every modality to the shared barcodes first")
    if len(names) != len(master):
        raise ValueError(
            f"{role}: {len(names)} cells but {master_name} has {len(master)}; "
            f"all modalities of one dataset must cover the same cells")
    warnings.warn(
        f"{role}: obs_names are not unique, so the cells cannot be matched to "
        f"{master_name} by barcode; pairing them positionally", UserWarning, stacklevel=4)
    return a


def _as_obs_vector(data, spec, *, what, master):
    """Labels / batch as a 1-D array in master order.

    ``spec``: an ``'obs:<col>'`` selector against ``data``, a pandas Series
    (aligned by index to the master barcodes; a plain RangeIndex of the right
    length is taken positionally) or any 1-D sequence of the right length.
    """
    import pandas as pd

    if isinstance(spec, str):
        if data is None:
            raise ValueError(f"{what}={spec!r} is a selector but data is None; pass a "
                             f"Series/array instead")
        ser = _select_obs(data, spec, what=what)
        src_names = [str(x) for x in ser.index]
        if master is not None and src_names != [str(x) for x in master] \
                and set(src_names) == set(master) and len(set(src_names)) == len(src_names):
            ser = ser.reindex([str(x) for x in master]) if ser.index.dtype == object \
                else ser.iloc[[src_names.index(str(x)) for x in master]]
        return np.asarray(ser)
    if isinstance(spec, pd.Series):
        if master is None:
            return np.asarray(spec)
        idx = [str(x) for x in spec.index]
        m = [str(x) for x in master]
        if idx == m:
            return np.asarray(spec)
        if len(set(idx)) == len(idx) and set(m) <= set(idx):
            pos = {k: i for i, k in enumerate(idx)}
            return np.asarray(spec)[[pos[x] for x in m]]
        if isinstance(spec.index, pd.RangeIndex) and len(spec) == len(m):
            return np.asarray(spec)             # unlabeled: positional
        lack = [x for x in m if x not in set(idx)]
        raise ValueError(
            f"{what}= Series index does not match the cell barcodes: {len(lack)} of "
            f"{len(m)} cells have no entry ({lack[:5]}{'...' if len(lack) > 5 else ''}); "
            f"index it by obs_names (or pass a plain list in cell order)")
    vals = np.asarray(spec)
    if vals.ndim != 1:
        raise ValueError(f"{what}= must be 1-D, got shape {vals.shape}")
    if master is not None and len(vals) != len(master):
        raise ValueError(f"{what} has {len(vals)} entries for {len(master)} cells")
    return vals


def export_dataset(data, dataset_dir: Path | str, *, rna="X",
                   adt=None, atac=None,
                   atac_kind: str | None = None, labels=None,
                   batch=None, dtype: str = "float64",
                   compression: str | None = "gzip",
                   category: str | None = None,
                   adt_names: list | None = None) -> Path:
    """Write an AnnData / MuData (or loose objects) as a canonical dataset folder.

    Produces the layout ``mtb.describe_layout`` documents (``rna.h5``,
    ``adt.h5``, ATAC files, ``cty.csv``), so ``mtb.scan`` and
    ``mtb.run_all`` work on your own data.

    Parameters
    ----------
    data : AnnData, MuData or None
        Object the selector strings refer to; ``None`` when every modality is
        passed as an object. With a MuData, name each modality (``rna='rna'``).
    dataset_dir : Path | str
        Folder to create; its name is the dataset id for ``mtb.scan`` /
        ``mtb.run_all``, and its parent their ``data_path``.
    rna : str, AnnData, DataFrame, array or None, keyword-only
        RNA matrix: a selector against ``data`` (``'X'``, ``'layer:counts'``)
        or an object (forms in Notes); ``None`` = no RNA.
    adt : str, AnnData, DataFrame, array or None, keyword-only
        Protein (ADT) matrix, same forms as ``rna`` (e.g. ``'obsm:protein'``);
        ``None`` = no ADT.
    atac : str, AnnData, DataFrame, array or None, keyword-only
        ATAC matrix, same forms as ``rna``; needs ``atac_kind``. ``None`` = no ATAC.
    atac_kind : str or None, keyword-only
        What ``atac`` holds: ``'peak'`` or ``'gene_activity'``; decides the
        ATAC filename.
    labels : str, Series, sequence or None, keyword-only
        Cell-type labels for ``cty.csv``: ``'obs:<col>'``, a Series indexed by
        barcode, or a sequence in cell order.
    batch : str, Series, sequence or None, keyword-only
        Batch per cell, same forms as ``labels``; splits every file per batch
        (``rna1.h5``, ``rna2.h5``, ...).
    dtype : str, keyword-only
        Stored dtype of ``matrix/data``, forwarded to ``mtb.io.to_canonical``.
    compression : str or None, keyword-only
        h5py compression filter, forwarded to ``mtb.io.to_canonical``.
    category : str or None, keyword-only
        Integration category the folder is for (``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``); changes the ATAC filenames (Notes).
    adt_names : list or None, keyword-only
        Protein names for the ADT matrix; they override any it carries and
        are needed when it has none.

    Returns
    -------
    pathlib.Path
        ``dataset_dir``.

    Raises
    ------
    ValueError
        Missing or conflicting arguments, or cells that do not pair across modalities (Notes).
    KeyError
        A selector names a ``mod``, ``obsm``, ``layer`` or ``obs`` column that is absent.

    Warns
    -----
    UserWarning
        Feature names missing or contradicting ``atac_kind``; non-unique barcodes; a dense matrix over 1 GB.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.io.export_dataset(adata, "data/MYCITE", adt="obsm:protein", labels="obs:celltype")
    >>> mtb.io.export_dataset(mdata, "data/MYMU", rna="rna", atac="atac", atac_kind="peak",
    ...                       labels="rna:celltype", batch="rna:sample")    # MuData, per batch
    >>> mtb.scan("MYCITE", "vertical", data_path="data")

    Notes
    -----
    **Modality arguments.** ``rna``, ``adt`` and ``atac`` each accept:

    - a selector string against ``data``: ``'X'``, ``'obsm:<key>'``,
      ``'layer:<key>'``, ``'mod:<name>'``, ``'mod:<name>.obsm:<key>'`` or
      ``'mod:<name>.layer:<key>'``;
    - an AnnData (its ``.X`` is written);
    - a DataFrame (index = cell barcodes, columns = features);
    - a 2-D array or sparse matrix already in the master cell order.

    An ``obsm`` selector takes its feature names from ``adt_names`` (ADT
    only), else the columns of a DataFrame-valued entry, else
    ``adata.uns['<key>_names']`` (order in ``mtb.io.to_canonical``).

    All matrices are cells x features and are written transposed. With
    ``data=None`` the default ``rna='X'`` means "no RNA": pass
    ``rna=<AnnData>``. Separate objects are paired by barcode:

    ```python
    mtb.io.export_dataset(rna_adata, "data/MYMULTI", atac=atac_adata,
                          atac_kind="peak", labels=rna_adata.obs["celltype"])
    ```

    **MuData.** A bare modality name selects ``mdata.mod[name].X``
    (``rna='rna'``); a full ``'mod:<name>'`` selector works too. Every
    modality of a MuData needs one of these, so the default ``rna='X'``
    raises: pass ``rna=<mod name>`` or ``rna=None``. ``labels`` / ``batch``
    accept ``'<mod>:<obs column>'`` (``'rna:celltype'``) or
    ``'mod:<mod>.obs:<col>'``.

    **Master cell order.** Every modality is written in one order -
    ``data.obs_names`` when ``data`` is given, else the first modality
    object's - and re-indexed to it by barcode:

    - the same barcodes in another order are reordered;
    - a barcode set that differs raises ``ValueError`` naming the strays;
    - non-unique barcodes cannot be matched by name, so the cells are paired
      positionally with a ``UserWarning`` (a different cell count still
      raises).

    A bare array carries no barcodes, so nothing can be checked - pass a
    DataFrame or AnnData when the order is not known to be right.

    **Labels.** ``cty.csv`` is the single-column CSV the benchmark reads
    (header ``x``, one label per line); ``mtb.labels_for`` finds these files
    again. A ``labels`` / ``batch`` Series is aligned by index to the master
    barcodes; a plain ``RangeIndex`` of the right length is taken
    positionally.

    **Batches.** With ``batch``, cells are split per batch value (sorted)
    and numbered files are written instead: ``rna1.h5``, ``rna2.h5`` ...,
    ``adt1.h5`` ..., ``cty1.csv`` ... (the layout of the shipped D52).

    **ATAC filenames.**

    - no ``category``: ``atac_kind='peak'`` (``chr:start-end`` features) is
      written as ``atac_peak.h5`` plus a hard-linked ``atac.h5`` (copied when
      the filesystem refuses); ``'gene_activity'`` as ``atac_gas.h5`` only,
      so the gene-activity role never silently falls back to a peak matrix;
    - ``category='vertical'``: plain ``atac.h5`` only, for both kinds - the
      one name every vertical (paired multiome) variant resolves;
    - ``'diagonal'`` / ``'mosaic'`` / ``'cross'``: ``atac_peak.h5`` or
      ``atac_gas.h5`` exactly as named, and no ``atac.h5`` link.

    The ``atac.h5`` link exists because most vertical multiome methods that
    read the plain ``atac`` role want peaks (Matilda wants gene activity;
    see ``method_info(m)['atac']``). Editing a hard-linked file edits both.

    **Check the ATAC kind.** The representation is recorded nowhere on
    disk, so check that ``method_info(m)['atac']`` is the kind exported: a
    gene-activity ``atac.h5`` fed to a peak method runs and returns a wrong
    embedding (``mtb.scan`` flags the mismatch as a caveat).

    **Errors.** ``ValueError`` is raised for:

    - nothing to export (``rna``, ``adt``, ``atac`` and ``labels`` all ``None``);
    - ``atac`` without a valid ``atac_kind``, or ``atac_kind`` without ``atac``;
    - an unknown ``category``;
    - a selector string with ``data=None``, or a malformed selector;
    - a bare array whose cell order cannot be checked (no barcodes anywhere)
      or with the wrong row count;
    - a modality whose barcodes differ from the master order (the message
      names the strays);
    - a ``labels`` / ``batch`` Series missing cells, or a sequence of the
      wrong length.

    The ``KeyError`` message lists the names the object does have.

    **Warnings.** ``UserWarning`` is emitted for feature names that
    contradict ``atac_kind``; a bare array or ``obsm`` matrix without
    feature names (``feature_0..`` written; for ADT pass ``adt_names``,
    otherwise a DataFrame or AnnData); non-unique barcodes (paired
    positionally); a dense ``matrix/data`` over 1 GB.

    See Also
    --------
    mtb.io.to_canonical : one matrix -> one canonical file (what this calls per modality).
    mtb.describe_layout : the layout being produced and the role -> filename mapping.
    mtb.scan : confirm the folder is found and which methods can run on it.
    """
    out = Path(dataset_dir)
    out.mkdir(parents=True, exist_ok=True)
    category = _check_category(category)
    if data is None and isinstance(rna, str) and rna == "X":
        rna = None                      # no data to select from: no RNA
    if _is_mudata(data):
        # a bare modality name selects mdata.mod[name].X; full
        # 'mod:<name>[...]' selectors pass through
        rna, adt, atac = (f"mod:{x}" if isinstance(x, str) and x in data.mod else x
                          for x in (rna, adt, atac))
    if atac is not None and atac_kind not in _ATAC_KINDS:
        raise ValueError(
            f"atac={atac!r} needs atac_kind= one of {list(_ATAC_KINDS)} "
            f"(got {atac_kind!r}): 'peak' -> atac_peak.h5 (+atac.h5), "
            "'gene_activity' -> atac_gas.h5")
    if atac is None and atac_kind is not None:
        raise ValueError("atac_kind= given without atac=")
    if rna is None and adt is None and atac is None and labels is None:
        raise ValueError("nothing to export: give at least one of rna=, adt=, atac=, labels=")

    # master cell order: data's obs_names, else the first modality object's
    master, master_name = None, None
    if data is not None and hasattr(data, "obs_names"):
        master, master_name = [str(x) for x in data.obs_names], "data"
    else:
        for what, spec in (("rna", rna), ("adt", adt), ("atac", atac)):
            if spec is not None and (_is_anndata(spec) or hasattr(spec, "index")):
                master = [str(x) for x in (spec.obs_names if _is_anndata(spec) else spec.index)]
                master_name = what
                break

    mats = []   # (role_base, adata, kwargs, user-facing name)
    for what, spec in (("rna", rna), ("adt", adt)):
        if spec is not None:
            a, kw = _as_modality(data, spec, what=what, master=master,
                                 feature_names=adt_names if what == "adt" else None)
            mats.append((what, a, kw, what))
    if atac is not None:
        a, kw = _as_modality(data, atac, what="atac", master=master)
        mats.append(("atac_peak" if atac_kind == "peak" else "atac_gas", a, kw, "atac"))
    if master is None and mats:
        master, master_name = [str(x) for x in mats[0][1].obs_names], mats[0][3]
    mats = [(role, _align_cells(a, master=master, master_name=master_name, role=what), kw)
            for role, a, kw, what in mats]
    n_ref = len(master) if master is not None else None
    lab = _as_obs_vector(data, labels, what="labels", master=master) if labels is not None else None
    if lab is not None and n_ref is not None and len(lab) != n_ref:
        raise ValueError(f"labels has {len(lab)} entries for {n_ref} cells")

    if batch is None:
        groups = [(None, None)]
    else:
        bvals = _as_obs_vector(data, batch, what="batch", master=master)
        if n_ref is not None and len(bvals) != n_ref:
            raise ValueError(f"batch has {len(bvals)} entries for {n_ref} cells")
        keys = sorted(set(bvals.tolist()), key=lambda v: str(v))
        groups = [(i + 1, np.flatnonzero(bvals == k)) for i, k in enumerate(keys)]

    for idx, mask in groups:
        suf = "" if idx is None else str(idx)
        for role, a, kw in mats:
            sub = a if mask is None else a[mask]
            # vertical: the plain atac.h5 the `atac` role reads, nothing else;
            # an explicit other category: the representation-named file only;
            # no category: representation-named file (+ atac.h5 link for peaks)
            fname = (f"atac{suf}.h5" if category == "vertical" and role.startswith("atac")
                     else f"{role}{suf}.h5")
            names = adt_names if role == "adt" else None
            p = to_canonical(sub, out / fname, modality=role,
                             dtype=dtype, compression=compression,
                             feature_names=names, **kw)
            if role == "atac_peak" and category is None:
                _link_or_copy(p, out / f"atac{suf}.h5")
        if lab is not None:
            _write_labels(lab if mask is None else np.asarray(lab)[mask],
                          out / f"cty{suf}.csv")
    return out

