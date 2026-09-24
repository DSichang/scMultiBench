"""Adapt arbitrary input formats to the canonical scMultiBench .h5.

Canonical layout: matrix/data (features x cells), matrix/features, matrix/barcodes.
Method scripts are never modified.

Entry points (exposed as ``mtb.io``):

* :func:`to_canonical` - one matrix (AnnData / .h5ad / .h5mu / .csv / .tsv /
  .loom) -> one canonical ``.h5``; sparse-safe, gzip-compressed, with
  ``modality=`` / ``layer=`` / ``obsm=`` / ``mod=`` selectors.
* :func:`export_dataset` - a whole AnnData/MuData -> a canonical dataset folder
  (``rna.h5``, ``adt.h5``, the ATAC file, ``cty.csv``), optionally per batch
  (``rna1.h5`` ...) or as unpaired diagonal data. A MuData's modalities can
  be named directly (``rna='rna'``).
* :func:`read_canonical` - the inverse (canonical ``.h5`` -> AnnData).
* :func:`normalize_peak_names` - rewrite peak ids to ``chr:start-end``.

Label files are single-column CSVs: a header ``x``, then one label per cell.
"""
from __future__ import annotations

import os
import re
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np

from .. import config

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
#: ``category=`` values accepted by to_canonical / export_dataset.
_CATEGORIES = ("vertical", "diagonal", "mosaic", "cross")
#: Roles whose matrix must hold raw counts. Gene-activity scores (atac_gas)
#: and the representation-free ``atac`` role are not checked.
_COUNT_ROLES = ("rna", "adt", "atac_peak")
#: Stored non-zero values sampled by the raw-count check.
_COUNT_SAMPLE = 5000
# '<selector>[<var column>=<value>]': keep the features whose var column equals value
_VAR_FILTER_RE = re.compile(r"^(?P<base>.*?)\[(?P<col>[^\[\]=]+)=(?P<val>[^\[\]]*)\]$")


def _has_fraction(vals) -> bool:
    """True when a sample of up to ``_COUNT_SAMPLE`` non-zero values of
    ``vals`` holds a value whose fractional part exceeds 1e-6."""
    v = np.asarray(vals).ravel()
    if v.size == 0 or v.dtype.kind != "f":
        return False                    # integer / bool storage: whole numbers
    if v.size > 20 * _COUNT_SAMPLE:     # thin out before the non-zero filter
        v = v[np.linspace(0, v.size - 1, 20 * _COUNT_SAMPLE).astype(int)]
    v = v[(v != 0) & np.isfinite(v)]
    if v.size > _COUNT_SAMPLE:
        v = v[np.linspace(0, v.size - 1, _COUNT_SAMPLE).astype(int)]
    return bool(np.any(np.abs(v - np.round(v)) > 1e-6))


def _matrix_has_fraction(X) -> bool | None:
    """Raw-count check of an in-memory matrix (cells x features): ``True``
    when sampled stored values are not whole numbers, ``None`` when ``X``
    cannot be read."""
    import scipy.sparse as sp
    try:
        if sp.issparse(X):
            return _has_fraction(X.data if getattr(X, "format", None) in ("csr", "csc", "coo")
                                 else X.tocsr().data)
        arr = np.asarray(X)
        if arr.ndim != 2 or arr.size == 0:
            return None
        rows = np.unique(np.linspace(0, arr.shape[0] - 1, min(arr.shape[0], 500)).astype(int))
        return _has_fraction(arr[rows])
    except Exception:                   # backed or exotic storage: no verdict
        return None


def _warn_not_counts(X, name: str, hint: str, *, stacklevel: int) -> None:
    """``UserWarning`` when ``X`` holds non-integer values (log-normalised?)."""
    if _matrix_has_fraction(X):
        warnings.warn(
            f"{name} values are not whole numbers (log-normalised?). The methods "
            f"normalise raw counts themselves: export raw counts, e.g. {hint}.",
            UserWarning, stacklevel=stacklevel + 1)


#: a trailing ``-<n>`` (10x GEM-well) suffix of a cell barcode
_BARCODE_SUFFIX_RE = re.compile(r"-\d+$")


def _cell_order(names, ref):
    """How barcodes ``names`` relate to ``ref``.

    Returns ``("same", None)`` (same cells, same order), ``("order", idx)``
    (same cells; ``names[idx]`` is in ``ref`` order), ``("other", None)``
    (different cells) or ``("unknown", None)`` (non-unique barcodes). Names
    that differ only in a trailing ``-<n>`` suffix count as the same cell when
    the suffix-free names stay unique on both sides.
    """
    names, ref = [str(x) for x in names], [str(x) for x in ref]
    judged = False
    for key in (None, _BARCODE_SUFFIX_RE):
        a = names if key is None else [key.sub("", x) for x in names]
        b = ref if key is None else [key.sub("", x) for x in ref]
        if len(set(a)) != len(a) or len(set(b)) != len(b):
            continue
        judged = True
        if a == b:
            return "same", None
        if set(a) == set(b):
            pos = {k: i for i, k in enumerate(a)}
            return "order", [pos[k] for k in b]
    return ("other" if judged else "unknown"), None


def _follow_peak_order(out: Path, X, bars):
    """``(X, bars)`` of a gene-activity matrix put in the cell order of the
    ``atac_peak.h5`` next to ``out``, which ``atac_cty.csv`` follows.

    Same cells in another order: rows reordered, one note on stderr.
    Different cells: ``UserWarning``. No peak file, or the same order:
    unchanged.
    """
    import sys

    import scipy.sparse as sp

    folder = Path(out).parent
    peak = folder / "atac_peak.h5"
    if Path(out).name == peak.name or not _is_canonical_h5(peak):
        return X, bars
    with h5py.File(peak, "r") as f:
        ref = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f["matrix/barcodes"][:]] if "matrix/barcodes" in f else None
    if ref is None:
        return X, bars
    kind, idx = _cell_order(bars, ref)
    if kind == "order":
        X = X.tocsr()[idx] if sp.issparse(X) else np.asarray(X)[idx]
        bars = [bars[i] for i in idx]
        print(f"to_canonical: wrote the rows of {Path(out).name} in the cell order of "
              f"{peak}, which atac_cty.csv follows", file=sys.stderr, flush=True)
    elif kind == "other":
        shared = len(set(bars) & set(ref))
        warnings.warn(
            f"{Path(out).name} holds different cells than {peak} ({len(bars):,} and "
            f"{len(ref):,}, {shared:,} shared). Diagonal methods need gene activity for "
            f"the ATAC cells of atac_peak.h5.", UserWarning, stacklevel=3)
    return X, bars


def _split_var_filter(spec: str):
    """``'X[feature_types=Peaks]'`` -> ``('X', ('feature_types', 'Peaks'))``;
    a selector without a filter -> ``(spec, None)``."""
    m = _VAR_FILTER_RE.match(spec)
    if not m:
        return spec, None
    return m.group("base"), (m.group("col").strip(), m.group("val").strip())


def _filter_var(adata, flt, *, what, spec):
    """The features of ``adata`` whose ``var[col]`` equals ``value`` (a view)."""
    col, val = flt
    if col not in adata.var.columns:
        raise KeyError(f"{what}={spec!r}: var column {col!r} not found; available: "
                       f"{list(adata.var.columns)}")
    vals = adata.var[col].astype(str).to_numpy()
    mask = vals == val
    if not mask.any():
        levels = sorted(set(vals.tolist()))
        shown = levels[:20] + (["..."] if len(levels) > 20 else [])
        raise ValueError(f"{what}={spec!r}: no feature has var[{col!r}] == {val!r}; "
                         f"the values of var[{col!r}] are {shown}")
    return adata[:, mask]


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
            + (" (a top-level 'data' dataset is a method output such as "
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
                "(pip install 'multibench-sc[loom]' or pip install loompy); "
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
            f"{out.name}: matrix/data is stored dense (features x cells): "
            f"{n_feat} x {n_cell} x {itemsize} B = {size / 1e9:.1f} GB uncompressed "
            f"on disk (limit for this warning: {DENSE_WARN_BYTES / 1e9:.0f} GB); "
            f"filter to highly-variable genes / informative peaks before export, "
            f"or pass dtype='float32' to halve it",
            UserWarning, stacklevel=3)


def _peak_fraction(feats) -> float:
    if not len(feats):
        return 0.0
    return sum(1 for x in feats if _PEAK_RE.match(x)) / len(feats)


def _check_peak_names(modality, feats, *, rna_feats=None, kind_arg=False):
    """Warn when the feature names contradict the declared ATAC role.

    Names declared as peaks that do not look like peaks get the rename advice.
    Gene activity is suggested only when the names support it: when
    ``rna_feats`` (the RNA feature names of the same export) is given, only
    if most ATAC names are among them; without it, as the second option.
    ``kind_arg`` spells the fix as ``export_dataset``'s ``atac_kind=``
    instead of ``to_canonical``'s ``modality=``.
    """
    if modality not in ("atac_gas", "atac_peak"):
        return
    frac = _peak_fraction(feats)
    if kind_arg:
        as_peak = config.hint("atac_kind='peak'", "--atac-kind peak")
        as_gas = config.hint("atac_kind='gene_activity'", "--atac-kind gene_activity")
    else:
        as_peak = config.hint("modality='atac_peak'", "--modality atac_peak")
        as_gas = config.hint("modality='atac_gas'", "--modality atac_gas")
    if modality == "atac_gas" and frac > 0.5:
        warnings.warn(
            f"{frac:.0%} of the ATAC feature names look like peaks such as "
            f"chr1:100-200, not genes. If the matrix holds peaks, pass {as_peak}.",
            UserWarning, stacklevel=3)
    elif modality == "atac_peak" and frac < 0.5:
        head = (f"Only {frac:.0%} of the ATAC feature names look like peaks such as "
                "chr1:100-200")
        rename = (". If they are peaks, rename them to chr:start-end. Some methods read "
                  "the peak positions from the names, and "
                  f"{config.hint('mtb.scan', 'multibench scan')} names these methods.")
        gas = f" If the matrix holds gene activity, pass {as_gas}."
        genes = set(rna_feats or ())
        share = sum(1 for x in feats if x in genes) / len(feats) if len(feats) else 0.0
        if rna_feats is None:                   # nothing to compare with: both options
            text = head + rename + gas
        elif share > 0.5:
            text = head + f", and {share:.0%} are RNA gene names." + gas
        else:
            text = head + rename
        warnings.warn(text, UserWarning, stacklevel=3)


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
    layer : str | None
        Take the matrix from ``adata.layers[layer]`` instead of ``adata.X``.
    obsm : str | None
        Take the matrix from ``adata.obsm[obsm]`` (e.g. ``'protein'`` for CITE-seq).
    mod : str | None
        MuData modality to convert (``mdata.mod[mod]``); required for a MuData.
    dtype : str
        Stored dtype of ``matrix/data``; ``'float32'`` halves the file (see Notes).
    compression : str | None
        h5py compression filter; ``None`` = uncompressed.
    block : int
        Number of features written per streaming step.
    category : str | None
        Integration category the file is for (``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``); only changes ATAC filenames (Notes).
    feature_names : list | None
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
        Values not whole numbers; feature names missing or contradicting ``modality``; size over 1 GB.
    UserWarning
        ``modality='gas'`` for other cells than the ``atac_peak.h5`` next to ``out``.

    Examples
    --------
    >>> import multibench as mtb
    >>> # writes data/MYCITE/rna.h5
    >>> mtb.io.to_canonical(adata, "data/MYCITE/", modality="rna")
    >>> mtb.io.to_canonical(adata, "data/MYCITE/", modality="adt", obsm="protein")
    >>> mtb.io.to_canonical("counts.csv", "rna.h5")  # explicit file name
    >>> mtb.io.to_canonical(atac, "data/MYMULTI/", modality="peak",
    ...                     category="vertical")

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
    4. ``feature_0..``, with a ``UserWarning``: pass ``feature_names`` to
       keep the protein names.

    **ATAC filenames.** Vertical methods read ``atac.h5``;
    ``method_info(m)['atac']`` says whether it must hold peaks or gene
    activity. Diagonal methods read ``atac_peak.h5`` (peaks) and
    ``atac_gas.h5`` (gene activity). Mosaic methods read ``atac<i>.h5``
    (peaks). ``peak.h5``, and ``atac.h5`` for gene activity, are accepted as
    older names.

    With ``out`` a directory, ``category='vertical'`` writes ``atac.h5`` for
    either representation. Any other ``category``, or none, names the file
    after the representation: ``atac_peak.h5``, ``atac_gas.h5``, or
    ``atac.h5`` for the plain ``'atac'`` role. For a mosaic batch, pass the
    numbered file as ``out`` (``'data/MYMOSAIC/atac2.h5'``).

    **Check the ATAC kind.** Without ``category='vertical'``,
    ``to_canonical(atac, d, modality='peak')`` writes ``atac_peak.h5``, and
    ``mtb.scan(d, 'vertical')`` finds no ATAC method runnable. The
    representation is not recorded on disk: ``method_info(m)['atac']`` says
    which kind a method expects.

    **Gene activity next to peaks.** With ``modality='gas'`` (no
    ``category``, or ``'diagonal'``) and an ``atac_peak.h5`` in the output
    folder, the rows are written in that file's cell order, because
    ``atac_cty.csv`` follows it. One line on stderr says so. Gene activity
    for other cells gives a ``UserWarning``.
    A canonical ``.h5`` is returned untouched: to reorder an existing
    ``atac_gas.h5``, pass ``mtb.io.read_canonical(path)`` as ``src``.

    **Raw counts.** Methods normalise the data themselves; give raw counts.
    For ``modality`` ``'rna'``, ``'adt'`` or ``'atac_peak'``, a
    ``UserWarning`` says when sampled values are not whole numbers
    (log-normalised data); ``layer='counts'`` usually holds the counts, and
    for an ADT matrix taken with ``obsm=``, another ``obsm`` key.

    **Streaming.** Sparse matrices (CSR/CSC, in memory or inside an
    ``.h5ad``/``.h5mu``) are converted to CSC and written ``block`` features
    at a time, without densifying the whole matrix. The defaults (gzip,
    ``'float64'``) match the shipped benchmark files; any compression
    enables chunking. A ``.csv`` / ``.tsv`` is read as cells x features, and
    a non-numeric first column is used as the cell barcodes.

    **Size on disk.** ``matrix/data`` is stored dense (features x cells x
    itemsize): gzip shrinks the file, but every reader densifies it. A
    ``UserWarning`` states the size when it exceeds 1 GB and suggests
    filtering features or ``dtype='float32'``, which h5py / rhdf5 / hdf5r
    read (as double in R).

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
    if modality in _COUNT_ROLES:
        hint = (config.hint("obsm='protein_counts'", "--obsm protein_counts")
                if obsm is not None and modality == "adt"
                else config.hint("layer='counts'", "--layer counts"))
        _warn_not_counts(X, modality, hint, stacklevel=2)
    if modality == "atac_gas" and category in (None, "diagonal"):
        X, bars = _follow_peak_order(out, X, bars)
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

    Raises
    ------
    FileNotFoundError
        ``path`` does not exist; the message names ``mtb.config.DEFAULT.data_path``.

    Examples
    --------
    >>> import multibench as mtb
    >>> d = mtb.config.DEFAULT.data_path / "D11"
    >>> rna = mtb.io.read_canonical(d / "rna.h5")
    >>> rna.shape, rna.var_names[:3]
    >>> adt = mtb.io.read_canonical(d / "adt.h5", sparse=False)

    Notes
    -----
    **Where the demo data is.** ``mtb.data.fetch("D11")`` downloads a demo
    dataset into ``mtb.config.DEFAULT.data_path``, not into the current
    directory.

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
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{os.fspath(path)} not found. Demo datasets are under "
            f"mtb.config.DEFAULT.data_path ({config.DEFAULT.data_path}).")
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
    two numbers) pass through unchanged. ``matrix/data`` and
    ``matrix/barcodes`` are copied as they are; only ``matrix/features`` is
    replaced.

    **Inside ``mtb.run``.** ``mtb.run`` applies this itself for GLUE and
    Seurat_v3, writing a per-run ``<role>_normpeaks.h5`` copy next to the
    converted inputs. Call it by hand only to prepare a file for a script
    you run outside the wrapper.

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
    ``'mod:<name>.obsm:<key>'`` / ``'mod:<name>.layer:<key>'``, each
    optionally followed by a feature filter ``[<var column>=<value>]``.
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"{what}= must be a selector string like 'X', 'obsm:<key>', "
                         f"'layer:<key>' or 'mod:<name>', got {spec!r}")
    rest, flt = _split_var_filter(spec)
    adata = data
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
        kw = {}
    elif rest.startswith("obsm:"):
        kw = {"obsm": rest[5:]}
    elif rest.startswith("layer:"):
        kw = {"layer": rest[6:]}
    else:
        raise ValueError(f"{what}={spec!r}: unknown selector; use 'X', 'obsm:<key>', "
                         f"'layer:<key>' or 'mod:<name>[.obsm:<key>|.layer:<key>]', "
                         f"optionally followed by [<var column>=<value>]")
    if flt is not None:
        if "obsm" in kw:
            raise ValueError(f"{what}={spec!r}: a [<column>=<value>] filter selects "
                             f"features by adata.var; an obsm matrix has no var")
        adata = _filter_var(adata, flt, what=what, spec=spec)
    return adata, kw


def _mudata_selector(mdata, spec):
    """A bare modality name (``'rna'``, ``'rna[feature_types=...]'``) of a
    MuData -> the full ``'mod:<name>...'`` selector; anything else unchanged."""
    if not isinstance(spec, str):
        return spec
    base, flt = _split_var_filter(spec)
    if base in mdata.mod:
        return f"mod:{base}" + (spec[len(base):] if flt else "")
    return spec


def _obs_column(obs, col, *, what, spec, where):
    if col not in obs.columns:
        raise KeyError(f"{what}={spec!r}: column {col!r} not in obs: searched {where}; "
                       f"available: {list(obs.columns)}")
    return obs[col]


def _select_obs(data, spec, *, what):
    """Parse an obs-column selector into a pandas Series.

    ``'obs:<col>'`` reads ``data.obs`` (a MuData's global ``.obs``);
    ``'mod:<name>.obs:<col>'`` reads ``mdata[name].obs``; for a MuData,
    ``'<name>:<col>'`` reads ``mdata[name].obs``, then muon's prefixed copy
    ``mdata.obs['<name>:<col>']``.
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"{what}= must be 'obs:<col>' or 'mod:<name>.obs:<col>', got {spec!r}")
    mu = _is_mudata(data)
    if spec.startswith("mod:"):
        if not mu:
            raise ValueError(f"{what}={spec!r}: 'mod:<name>' needs a MuData input")
        name, _, rest = spec[4:].partition(".")
        if name not in data.mod:
            raise KeyError(f"{what}={spec!r}: mod {name!r} not in MuData; found: {list(data.mod)}")
        if not rest.startswith("obs:"):
            raise ValueError(f"{what}={spec!r}: use 'obs:<col>' (or 'mod:<name>.obs:<col>')")
        return _obs_column(data.mod[name].obs, rest[4:], what=what, spec=spec,
                           where=f"mdata[{name!r}].obs")
    if spec.startswith("obs:"):
        col = spec[4:]
        if mu and col not in data.obs.columns:
            # muon keeps a modality's labels in mdata[mod].obs: name that spelling
            has = [n for n in data.mod if col in data.mod[n].obs.columns]
            if has:
                raise KeyError(
                    f"{what}={spec!r}: column {col!r} not in obs: searched mdata.obs; "
                    f"available: {list(data.obs.columns)}. mdata[{has[0]!r}].obs has "
                    f"{col!r}: use " + config.hint(f"{what}='{has[0]}:{col}'",
                                                   f"--{what} {has[0]}:{col}"))
        return _obs_column(data.obs, col, what=what, spec=spec,
                           where="mdata.obs" if mu else "data.obs")
    if mu:
        name, _, col = spec.partition(":")
        if name in data.mod and col:
            obs = data.mod[name].obs
            if col in obs.columns:
                return obs[col]
            if f"{name}:{col}" in data.obs.columns:
                return data.obs[f"{name}:{col}"]
            msg = (f"{what}={spec!r}: column {col!r} not in obs: searched "
                   f"mdata[{name!r}].obs and mdata.obs[{name + ':' + col!r}]; "
                   f"mdata[{name!r}].obs has {list(obs.columns)}")
            if col in data.obs.columns:
                msg += (f". The global mdata.obs has {col!r}: pass "
                        + config.hint(f"{what}='obs:{col}'", f"--{what} obs:{col}"))
            raise KeyError(msg)
    raise ValueError(f"{what}={spec!r}: use 'obs:<col>' (or 'mod:<name>.obs:<col>')")


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
    pair = {role, master_name}
    if master_name == "data":
        # the parameter name alone reads as "the data"
        master_name = "the object passed as data"
    unique = len(set(names)) == len(names) and len(set(master)) == len(master)
    if unique and set(names) == set(master):
        return a[master]
    if unique:
        in_master, in_names = set(master), set(names)
        stray = [x for x in names if x not in in_master]
        lack = [x for x in master if x not in in_names]
        same = "All modalities of one dataset must hold the same cells."
        if len(stray) == len(names):            # no cell in common
            if "atac" in pair and pair & {"rna", "data"}:
                raise ValueError(
                    "RNA and ATAC have no cells in common. For RNA and ATAC from "
                    "different cells (diagonal integration), pass "
                    + config.hint('category="diagonal"', "--category diagonal") + ".")
            raise ValueError(f"{role} and {master_name} have no cells in common. {same}")
        if stray:
            head = (f"{role} has {len(names):,} cells, and {len(stray):,} of them are not "
                    f"in {master_name}, for example {stray[:3]}.")
        else:
            head = (f"{role} has {len(names):,} cells and {master_name} has "
                    f"{len(master):,}. {len(lack):,} cells of {master_name} are not in "
                    f"{role}.")
        raise ValueError(f"{head} {same} Subset each modality to the shared barcodes first.")
    if len(names) != len(master):
        raise ValueError(
            f"{role} has {len(names):,} cells and {master_name} has {len(master):,}. "
            f"All modalities of one dataset must hold the same cells.")
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
                   adt_names: list | None = None,
                   batch_index: int | None = None,
                   overwrite: bool = False) -> Path:
    """Write an AnnData / MuData (or loose objects) as a canonical dataset folder.

    Produces the layout ``mtb.describe_layout`` documents (``rna.h5``,
    ``adt.h5``, ATAC files, ``cty.csv``), so ``mtb.scan`` and
    ``mtb.run_all`` work on your own data. Give raw counts.

    Parameters
    ----------
    data : AnnData, MuData or None
        Object the selector strings refer to; ``None`` when every modality is
        passed as an object. With a MuData, name each modality (``rna='rna'``).
    dataset_dir : Path | str
        Folder to create; its name is the dataset id for ``mtb.scan`` /
        ``mtb.run_all``, and its parent their ``data_path``.
    rna : str, AnnData, DataFrame, array or None
        Raw-count matrix for RNA: a selector against ``data`` (``'layer:counts'``,
        ``'X[feature_types=Gene Expression]'``) or an object (Notes); ``None`` = no RNA.
    adt : str, AnnData, DataFrame, array or None
        Raw-count matrix for protein (ADT), same forms as ``rna`` (e.g.
        ``'obsm:protein'``); ``None`` = no ADT.
    atac : str, AnnData, DataFrame, array or None
        ATAC matrix, same forms as ``rna`` (e.g. ``'X[feature_types=Peaks]'``);
        needs ``atac_kind``. ``None`` = no ATAC.
    atac_kind : str or None
        What ``atac`` holds: ``'peak'`` or ``'gene_activity'``; decides the
        ATAC filename.
    labels : str, Series, sequence or None
        Cell-type labels: ``'obs:<col>'`` (a MuData's global ``.obs``),
        ``'<mod>:<col>'`` (``mdata[mod].obs``), a Series indexed by barcode, or
        a sequence.
    batch : str, Series, sequence or None
        Batch per cell, same forms as ``labels``; splits the files per batch
        (``rna1.h5``, ``rna2.h5``, ...) for mosaic or cross.
    dtype : str
        Stored dtype of ``matrix/data``, forwarded to ``mtb.io.to_canonical``.
    compression : str or None
        h5py compression filter, forwarded to ``mtb.io.to_canonical``.
    category : str or None
        Integration category the folder is for (``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``); sets the ATAC and label filenames (Notes).
    adt_names : list or None
        Protein names for the ADT matrix; they override any it carries and
        are needed when it has none.
    batch_index : int or None
        Write the whole object as batch ``N`` (``rna<N>.h5``, ``cty<N>.csv``);
        mosaic or cross, one call per batch file.
    overwrite : bool
        ``True`` = replace files already in ``dataset_dir``; ``False`` = raise
        before writing anything.

    Returns
    -------
    pathlib.Path
        ``dataset_dir``.

    Raises
    ------
    ValueError
        Missing or conflicting arguments, or cells that do not pair across modalities (Notes).
    KeyError
        A selector names a ``mod``, ``obsm``, ``layer``, ``obs`` or ``var`` column that is absent.
    FileExistsError
        A file this call would write exists and ``overwrite`` is ``False``.

    Warns
    -----
    UserWarning
        Values that are not whole numbers, feature-name problems, and the other cases in Notes.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.io.export_dataset(adata, "data/MYCITE", rna="layer:counts",
    ...                       adt="obsm:protein", labels="obs:celltype")
    >>> mtb.io.export_dataset(mdata, "data/MYMULTIOME", rna="rna", atac="atac",
    ...                       atac_kind="peak", labels="obs:celltype",
    ...                       category="vertical")
    >>> mtb.io.export_dataset(rna, "data/LUNG", atac=atac, atac_kind="peak",
    ...                       labels="obs:cell_type", category="diagonal")
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

    **Feature filter.** A selector may end with ``[<var column>=<value>]``.
    Only the features whose ``adata.var`` column equals the value are
    written. A 10x Multiome AnnData read with ``gex_only=False`` holds genes
    and peaks in one ``X``:

    ```python
    mtb.io.export_dataset(adata, "data/MYARC",
                          rna="X[feature_types=Gene Expression]",
                          atac="X[feature_types=Peaks]", atac_kind="peak",
                          labels="obs:celltype", category="vertical")
    ```

    **Raw counts.** Methods normalise the data themselves; give raw counts. A
    ``UserWarning`` says when sampled RNA, ADT or peak values are not whole
    numbers (log-normalised data). ``rna='layer:counts'`` usually selects the
    counts. Gene-activity scores are not checked.

    **MuData.** A bare modality name selects ``mdata.mod[name].X``
    (``rna='rna'``); a full ``'mod:<name>'`` selector works too. The default
    ``rna='X'`` raises for a MuData: pass ``rna=<mod name>`` or ``rna=None``.

    ``labels`` / ``batch`` read ``'obs:<col>'`` from the global ``mdata.obs``.
    ``'<mod>:<col>'`` reads ``mdata[mod].obs``, then muon's copy
    ``mdata.obs['<mod>:<col>']``; ``'mod:<mod>.obs:<col>'`` works too.

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

    **Diagonal.** ``category='diagonal'`` pairs no barcodes: RNA and ATAC may
    come from different cells. ``rna`` is written as ``rna.h5`` and ``atac``
    as ``atac_peak.h5`` or ``atac_gas.h5``. ``labels`` is read from each
    object's own ``.obs`` into ``rna_cty.csv`` and ``atac_cty.csv``. A call
    that writes one modality writes only that modality's label file.

    For a MuData, ``labels='<col>'`` or ``'<mod>:<col>'`` reads ``<col>``
    from each modality's own ``.obs`` (``mdata['rna'].obs`` for the RNA
    cells, ``mdata['atac'].obs`` for the ATAC cells). When the two columns
    have different names, export RNA and ATAC in two calls.

    **Batches.** With ``batch``, cells are split per batch value (sorted)
    and numbered files are written: ``rna1.h5``, ``rna2.h5`` ...,
    ``adt1.h5`` ..., ``cty1.csv`` ... (the layout of the shipped D52). Only
    cross and mosaic methods read numbered files. For a vertical folder,
    export without ``batch`` and pass the batch column to
    ``mtb.run_all(batch=...)`` or ``mtb.evaluate(batch=...)``.

    ``batch_index=N`` writes the whole object as batch ``N``. A mosaic
    delivery of one file per batch (the D46 pattern: CITE-seq, Multiome,
    RNA only) is one call per file:

    ```python
    kw = dict(labels="obs:cell_type", category="mosaic")
    mtb.io.export_dataset(cite, "data/LAB", adt="obsm:protein",
                          batch_index=1, **kw)
    mtb.io.export_dataset(multiome, "data/LAB", atac="obsm:atac",
                          atac_kind="peak", batch_index=2, **kw)
    mtb.io.export_dataset(rna_only, "data/LAB", batch_index=3, **kw)
    ```

    ``mtb.describe_layout('mosaic')`` lists the batch patterns the methods
    read.

    **ATAC filenames.** Vertical methods read ``atac.h5``;
    ``method_info(m)['atac']`` says whether it must hold peaks or gene
    activity. Diagonal methods read ``atac_peak.h5`` (peaks) and
    ``atac_gas.h5`` (gene activity). Mosaic methods read ``atac<i>.h5``
    (peaks). ``peak.h5``, and ``atac.h5`` for gene activity, are accepted as
    older names. By ``category``:

    - ``'vertical'``: ``atac.h5`` for both kinds;
    - ``'diagonal'``: ``atac_peak.h5`` or ``atac_gas.h5``;
    - ``'mosaic'``: ``atac<i>.h5``;
    - no ``category``: ``atac_peak.h5`` plus a hard-linked ``atac.h5`` (a
      copy when the file system refuses links), or ``atac_gas.h5`` only.
      Editing a hard-linked file edits both.

    The representation is not recorded on disk. Check that
    ``method_info(m)['atac']`` is the kind exported. ``mtb.scan`` and
    ``run_all`` skip a method whose file holds the other representation,
    unless ``allow_atac_mismatch=True``. ``mtb.run`` only warns, and the
    method gives a wrong embedding.

    **Existing files.** Every check runs before ``dataset_dir`` is created;
    a failed call writes nothing. When a file the call would write already
    exists, ``FileExistsError`` lists it; pass ``overwrite=True`` to replace
    it. Other files in the folder are kept.

    **Errors.** ``ValueError`` is raised for:

    - nothing to export (``rna``, ``adt``, ``atac`` and ``labels`` all ``None``);
    - ``atac`` without a valid ``atac_kind``, or ``atac_kind`` without ``atac``;
    - an unknown ``category``;
    - a selector string with ``data=None``, a malformed selector, or a
      feature filter that matches nothing;
    - a bare array whose cell order cannot be checked (no barcodes anywhere)
      or with the wrong row count;
    - a modality whose barcodes differ from the master order (the message
      names the strays);
    - a ``labels`` / ``batch`` Series missing cells, or a sequence of the
      wrong length;
    - a label that is missing (NaN, None or ``''``): ``mtb.evaluate`` would
      score it as one more cell type;
    - ``batch`` with ``category='vertical'`` or ``'diagonal'``, or together
      with ``batch_index``;
    - ``batch_index`` that is not a positive integer, or without
      ``category='mosaic'`` / ``'cross'``;
    - ``category='diagonal'`` with labels but no matrix, or with a plain
      label sequence for both RNA and ATAC.

    The ``KeyError`` message lists the names the object does have.

    **Warnings.** ``UserWarning`` is emitted for:

    - RNA, ADT or peak values that are not whole numbers;
    - feature names that contradict ``atac_kind``, or none at all
      (``feature_0..`` is written; for ADT pass ``adt_names``);
    - non-unique barcodes (paired positionally);
    - a dense ``matrix/data`` over 1 GB;
    - ``batch`` with ``atac`` and no ``category``: only mosaic methods read
      numbered ATAC files;
    - ``category='mosaic'`` with ``atac_kind='gene_activity'``: every mosaic
      method reads peaks;
    - ``category='mosaic'`` or ``'cross'`` without ``batch`` or
      ``batch_index``.

    See Also
    --------
    mtb.io.to_canonical : one matrix -> one canonical file.
    mtb.describe_layout : the layout being produced and the role -> filename mapping.
    mtb.scan : confirm the folder is found and which methods can run on it.
    """
    out = Path(dataset_dir)
    category = _check_category(category)
    if data is None and isinstance(rna, str) and rna == "X":
        rna = None                      # no data to select from: no RNA
    if _is_mudata(data):
        # a bare modality name selects mdata.mod[name].X; full
        # 'mod:<name>[...]' selectors pass through
        rna, adt, atac = (_mudata_selector(data, x) for x in (rna, adt, atac))
    if atac is not None and atac_kind not in _ATAC_KINDS:
        raise ValueError(
            f"atac={atac!r} needs atac_kind= one of {list(_ATAC_KINDS)} "
            f"(got {atac_kind!r}): 'peak' -> atac_peak.h5 (+atac.h5), "
            "'gene_activity' -> atac_gas.h5")
    if atac is None and atac_kind is not None:
        raise ValueError("atac_kind= given without atac=")
    if rna is None and adt is None and atac is None and labels is None:
        raise ValueError("nothing to export: give at least one of rna=, adt=, atac=, labels=")
    _check_batch_args(batch, batch_index, category, adt=adt, atac=atac)
    diagonal = category == "diagonal"
    if diagonal and labels is not None and rna is None and adt is None and atac is None:
        raise ValueError(
            "category='diagonal' names the label file after the modality it labels "
            "(rna_cty.csv, atac_cty.csv): pass rna= or atac= together with labels=")
    if category == "mosaic" and atac is not None and atac_kind == "gene_activity":
        warnings.warn("atac_kind='gene_activity' with category='mosaic': every mosaic "
                      "method reads peaks (atac<i>.h5)", UserWarning, stacklevel=2)
    if category in ("mosaic", "cross") and batch is None and batch_index is None:
        warnings.warn(f"category={category!r} methods read numbered files (rna1.h5, "
                      f"cty1.csv, ...): pass batch_index= (this object is one batch) "
                      f"or batch= (a column that splits it)", UserWarning, stacklevel=2)

    # --- the modality matrices, each re-indexed to its side's cell order.
    # Paired categories have one side; diagonal keeps the ATAC cells apart.
    sides = {"rna": [], "atac": []}     # side -> [(role, adata, kw, what, source)]
    first = _first_master(data, [("rna", rna), ("adt", adt)]
                          + ([] if diagonal else [("atac", atac)]), diagonal=diagonal)
    master, master_name = first
    for what, spec in (("rna", rna), ("adt", adt)):
        if spec is not None:
            a, kw = _as_modality(data, spec, what=what, master=master,
                                 feature_names=adt_names if what == "adt" else None)
            sides["rna"].append((what, a, kw, what, _source_name(data, spec, what)))
    if atac is not None:
        atac_master = master
        if diagonal:
            atac_master = _names_of(atac) or (
                [str(x) for x in data.obs_names] if _is_anndata(data) and not _is_mudata(data)
                else None)
        a, kw = _as_modality(data, atac, what="atac", master=atac_master)
        role = "atac_peak" if atac_kind == "peak" else "atac_gas"
        sides["atac" if diagonal else "rna"].append(
            (role, a, kw, "atac", _source_name(data, atac, "atac")))
    masters = {}
    for side, mats in sides.items():
        if not mats:
            continue
        if side == "rna" and master is not None:
            m, mname = master, master_name
        else:
            m, mname = [str(x) for x in mats[0][1].obs_names], mats[0][3]
        masters[side] = m
        sides[side] = [(role, _align_cells(a, master=m, master_name=mname, role=what), kw, what, src)
                       for role, a, kw, what, src in mats]
    if not masters and master is not None:
        masters["rna"] = master         # labels only

    # --- labels: one vector per side (diagonal) or one for the paired cells
    lab = {}
    if labels is not None:
        if diagonal:
            lab = _diagonal_labels(data, labels, sides)
        else:
            m = masters.get("rna")
            vec = _as_obs_vector(data, labels, what="labels", master=m)
            if m is not None and len(vec) != len(m):
                raise ValueError(f"labels has {len(vec)} entries for {len(m)} cells")
            _check_labels_complete(
                vec, spec=labels, where=f"the cells ({_labels_where(data, labels)})",
                advice="give every cell a label, or subset the data to the labelled cells "
                       "first")
            lab["rna"] = vec

    # --- batches
    if batch is not None:
        m = masters.get("rna")
        bvals = _as_obs_vector(data, batch, what="batch", master=m)
        if m is not None and len(bvals) != len(m):
            raise ValueError(f"batch has {len(bvals)} entries for {len(m)} cells")
        keys = sorted(set(bvals.tolist()), key=lambda v: str(v))
        groups = [(str(i + 1), np.flatnonzero(bvals == k)) for i, k in enumerate(keys)]
    elif batch_index is not None:
        groups = [(str(int(batch_index)), None)]
    else:
        groups = [("", None)]

    # --- select every matrix and run the content checks, before any write
    prepared = []                       # (side, role, X, feats, bars, what)
    for side, mats in sides.items():
        for role, a, kw, what, src in mats:
            if role == "adt" and not kw and len(getattr(a, "obsm", {})):
                raise ValueError(
                    f"adt requested but obsm= not given; found obsm keys {list(a.obsm)} - "
                    "pass obsm='<key>' (or layer=) to select the protein matrix, otherwise "
                    "adata.X (usually the RNA) would be written as adt.h5")
            X, feats = _pick_matrix(a, layer=kw.get("layer"), obsm=kw.get("obsm"),
                                    feature_names=adt_names if role == "adt" else None,
                                    what=role)
            if role in ("atac_peak", "atac_gas"):
                # the RNA is prepared first (side 'rna', then the ATAC)
                rna_feats = next((f for _, r, _, f, _, _ in prepared if r == "rna"), None)
                _check_peak_names(role, feats, rna_feats=rna_feats, kind_arg=True)
            if role in _COUNT_ROLES:
                # the hint follows the selector: raw ADT counts of a CITE-seq
                # object usually sit in another obsm key, not in a layer
                if role == "adt" and kw.get("obsm") is not None:
                    hint = config.hint("adt='obsm:protein_counts'",
                                       "--adt obsm:protein_counts")
                else:
                    hint = config.hint(f"{what}='layer:counts' (MuData: "
                                       f"{what}='mod:{what}.layer:counts')",
                                       f"--{what} layer:counts (MuData: "
                                       f"--{what} mod:{what}.layer:counts)")
                _warn_not_counts(X, what, hint, stacklevel=2)
            prepared.append((side, role, X, feats, [str(x) for x in a.obs_names], what))

    # --- the files this call writes
    plan = []                           # (path, kind, payload)
    for suf, mask in groups:
        for side, role, X, feats, bars, what in prepared:
            sub = X if mask is None else X[mask]
            sub_bars = bars if mask is None else [bars[i] for i in mask]
            path = out / _export_filename(role, suf, category)
            plan.append((path, "matrix", (sub, feats, sub_bars)))
            if role == "atac_peak" and category is None:
                plan.append((out / f"atac{suf}.h5", "link", path))
        for side, vec in lab.items():
            name = (f"{side}_cty.csv" if diagonal else f"cty{suf}.csv")
            plan.append((out / name, "labels", vec if mask is None else np.asarray(vec)[mask]))
    existing = [pth.name for pth, _, _ in plan if pth.exists() or pth.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{out} already holds {existing}, which this call would write; "
            f"pass {config.hint('overwrite=True', '--overwrite')} to replace them")
    for pth, kind, payload in plan:
        if kind == "matrix":
            sub, feats, sub_bars = payload
            if getattr(sub, "ndim", 0) == 2:
                _warn_dense_size(pth, sub.shape[1], sub.shape[0], dtype)

    out.mkdir(parents=True, exist_ok=True)
    for pth, kind, payload in plan:
        if pth.exists() or pth.is_symlink():
            pth.unlink()                # also breaks an old hard link
        if kind == "matrix":
            sub, feats, sub_bars = payload
            _write_canonical(pth, sub, feats, sub_bars, dtype=dtype,
                             compression=compression, block=1024)
        elif kind == "link":
            _link_or_copy(payload, pth)
        else:
            _write_labels(payload, pth)
    return out


def _check_batch_args(batch, batch_index, category, *, adt, atac) -> None:
    """Refuse the batch / batch_index combinations no method can read."""
    if batch is not None and batch_index is not None:
        raise ValueError("pass batch= or batch_index=, not both: batch= splits the object "
                         "by a column, batch_index= writes the whole object as one batch")
    if batch_index is not None:
        if isinstance(batch_index, bool) or not isinstance(batch_index, (int, np.integer)) \
                or batch_index < 1:
            raise ValueError(f"batch_index= must be a positive integer (1 writes rna1.h5), "
                             f"got {batch_index!r}")
        if category not in ("mosaic", "cross"):
            raise ValueError(
                f"batch_index= writes numbered files (rna{batch_index}.h5, ...), which only "
                f"mosaic and cross methods read: pass category='mosaic' or category='cross' "
                f"(got category={category!r})")
    if batch is not None and category in ("vertical", "diagonal"):
        from .resolve import _one_file_advice
        raise ValueError(
            config.hint("batch=", "--batch") + " writes per-batch files (rna1.h5, "
            "rna2.h5, ...); "
            + _one_file_advice(category, has_adt=adt is not None and atac is None))
    if batch is not None and category is None and atac is not None:
        from .resolve import _batch_column_advice
        warnings.warn(
            "batch= with atac=: only mosaic methods read numbered ATAC files. For mosaic "
            "pass category='mosaic'; otherwise " + _batch_column_advice(),
            UserWarning, stacklevel=3)


def _names_of(spec):
    """Cell barcodes an AnnData / DataFrame argument carries, else ``None``."""
    if _is_anndata(spec):
        return [str(x) for x in spec.obs_names]
    if hasattr(spec, "index") and hasattr(spec, "columns"):
        return [str(x) for x in spec.index]
    return None


def _first_master(data, specs, *, diagonal):
    """``(barcodes, name)`` of the paired cell order: ``data.obs_names``, else
    the first modality object's. A diagonal MuData has no single order (its
    global ``obs_names`` joins both sides); the RNA side then sets it."""
    if data is not None and hasattr(data, "obs_names") and not (diagonal and _is_mudata(data)):
        return [str(x) for x in data.obs_names], "data"
    for what, spec in specs:
        names = _names_of(spec) if spec is not None else None
        if names is not None:
            return names, what
    return None, None


def _source_name(data, spec, what) -> str:
    """How an error names the object a modality came from."""
    if isinstance(spec, str):
        if spec.startswith("mod:"):
            name = _split_var_filter(spec)[0][4:].partition(".")[0]
            return f"mdata[{name!r}]"
        return "data"
    return what


def _diagonal_labels(data, spec, sides) -> dict:
    """``{side: labels}`` for a diagonal export: each side's labels from its
    own cells. ``'obs:<col>'`` reads each object's own ``.obs`` (a MuData's
    global ``.obs`` when a modality lacks the column); for a MuData,
    ``'<mod>:<col>'`` and a bare ``'<col>'`` read ``<col>`` from each
    modality's own ``.obs``."""
    import pandas as pd

    present = [side for side in ("rna", "atac") if sides[side]]
    if not isinstance(spec, (str, pd.Series)) and len(present) > 1:
        raise ValueError(
            "category='diagonal' with both rna= and atac=: labels= must be 'obs:<col>' "
            "or a Series indexed by barcode; a plain sequence cannot be split between "
            "the RNA and the ATAC cells")
    mu = _is_mudata(data)
    col_each = None                     # the column read from each side's own obs
    if isinstance(spec, str):
        if spec.startswith("obs:"):
            col_each = spec[4:]
        elif mu and not spec.startswith("mod:"):
            name, sep, col = spec.partition(":")
            if sep and name in data.mod and col:
                col_each = col
            elif not sep:
                col_each = spec
    out = {}
    for side in present:
        role, a, kw, what, src = sides[side][0]
        bars = [str(x) for x in a.obs_names]
        if col_each is not None and col_each in a.obs.columns:
            out[side] = np.asarray(a.obs[col_each])
            _check_labels_complete(out[side], spec=spec, where=f"the {side.upper()} cells "
                                   f"({src})", advice=_diagonal_label_advice(data, spec, sides))
            continue
        if col_each is not None:
            if spec.startswith("obs:") and mu and col_each in data.obs.columns:
                ser = data.obs[col_each]
            else:
                raise KeyError(
                    f"labels={spec!r}: column {col_each!r} not in obs of the "
                    f"{side.upper()} object ({src}); available: {list(a.obs.columns)}")
        elif isinstance(spec, str):
            ser = _select_obs(data, spec, what="labels")
        else:
            ser = spec
        if isinstance(ser, pd.Series):
            idx = {str(k): i for i, k in enumerate(ser.index)}
            lack = [b for b in bars if b not in idx]
            if lack:
                raise ValueError(
                    f"labels={repr(spec) if isinstance(spec, str) else 'Series'}: "
                    f"{len(lack)} of the {len(bars)} {side.upper()} cells have no label "
                    f"({lack[:5]}{'...' if len(lack) > 5 else ''}); "
                    + _diagonal_label_advice(data, spec, sides))
            out[side] = np.asarray(ser)[[idx[b] for b in bars]]
        else:
            out[side] = _as_obs_vector(None, ser, what="labels", master=bars)
        _check_labels_complete(out[side], spec=spec, where=f"the {side.upper()} cells ({src})",
                               advice=_diagonal_label_advice(data, spec, sides))
    return out


def _diagonal_label_advice(data, spec, sides) -> str:
    """What to pass instead, for a diagonal label selector that failed."""
    if _is_mudata(data):
        col = spec.rsplit(":", 1)[-1] if isinstance(spec, str) else "<col>"
        where = " and ".join(src + ".obs" for side in ("rna", "atac") for _r, _a, _k, _w, src
                             in sides[side][:1])
        return (f"with category='diagonal', labels={col!r} reads that column from each "
                f"modality's own obs ({where}); when the column names differ, export RNA "
                f"and ATAC in two calls")
    return ("with category='diagonal', labels='obs:<col>' reads the column from each "
            "object's own obs")


def _n_missing_labels(vals) -> int:
    """How many entries of a label vector are NaN, None or an empty string."""
    import pandas as pd

    s = pd.Series(np.asarray(vals, dtype=object))
    return int((s.isna() | (s.astype(str).str.strip() == "")).sum())


def _check_labels_complete(vals, *, spec, where: str, advice: str) -> None:
    """Refuse a label vector with missing values: evaluate would score them as
    one more cell type."""
    n = _n_missing_labels(vals)
    if n:
        shown = repr(spec) if isinstance(spec, str) else type(spec).__name__
        raise ValueError(
            f"labels={shown}: {n} of the {len(vals)} labels for {where} are missing "
            f"(NaN, None or ''); {advice}")


def _labels_where(data, spec) -> str:
    """How an error names the object a paired ``labels`` selector read."""
    if not isinstance(spec, str):
        return "the labels you passed"
    mu = _is_mudata(data)
    if spec.startswith("obs:"):
        return "mdata.obs" if mu else "data.obs"
    if spec.startswith("mod:"):
        return f"mdata[{spec[4:].partition('.')[0]!r}].obs"
    if mu:
        return f"mdata[{spec.partition(':')[0]!r}].obs"
    return "data.obs"


def _export_filename(role: str, suf: str, category: str | None) -> str:
    """On-disk name of one modality file of :func:`export_dataset`."""
    if role.startswith("atac") and category in ("vertical", "mosaic"):
        return f"atac{suf}.h5"
    return f"{role}{suf}.h5"


def _select_object(obj, spec: str, *, what: str):
    """An AnnData holding the matrix ``spec`` selects from ``obj`` (with its
    ``.obs``): what the CLI's ``--atac-from`` passes on as ``atac=``."""
    import anndata as ad

    if _is_mudata(obj):
        spec = _mudata_selector(obj, spec)
    a, kw = _select(obj, spec, what=what)
    if not kw:
        return a
    X, feats = _pick_matrix(a, layer=kw.get("layer"), obsm=kw.get("obsm"), what=what)
    b = ad.AnnData(X, obs=a.obs.copy())
    b.var_names = [str(x) for x in feats]
    return b
