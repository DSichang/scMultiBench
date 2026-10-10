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
    """``UserWarning`` when ``X`` holds non-integer values, which look
    log-normalised; ``hint`` is the selector of the raw counts."""
    if _matrix_has_fraction(X):
        warnings.warn(
            f"The {name} values are not whole numbers, so they look log-normalised. "
            f"The methods normalise raw counts themselves. Export raw counts, for "
            f"example with {hint}.",
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
    """``matrix/data`` is there as a 2-D dataset. A 10x Genomics ``.h5`` also
    has a ``matrix/data``: the 1-D values of a sparse matrix, next to
    ``matrix/indptr``."""
    try:
        with h5py.File(path, "r") as f:
            d = f.get("matrix/data")
            return isinstance(d, h5py.Dataset) and d.ndim == 2
    except (OSError, KeyError):
        return False


def _is_tenx_h5(path: Path) -> bool:
    """A 10x Genomics feature-barcode ``.h5``: a sparse matrix, which has
    ``matrix/indptr``."""
    try:
        with h5py.File(path, "r") as f:
            return "matrix/indptr" in f
    except OSError:
        return False


def _tenx_message(path) -> str:
    """What to do with a 10x Genomics ``.h5`` instead of passing it as a
    dataset file."""
    recipe = 'The documentation has the steps as the recipe "A 10x .h5 file".'
    return config.hint(
        f"{path} is a 10x Genomics file, not a multibench dataset file. Read it with "
        f"scanpy, sc.read_10x_h5(path, gex_only=False), split it by var['feature_types'] "
        f"and write a dataset folder with mtb.io.export_dataset. {recipe}",
        f"{path} is a 10x Genomics file. multibench convert reads .h5ad files, not this "
        f"file. Read it in Python with scanpy, save one .h5ad file per modality and "
        f"convert those. {recipe}")


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
                d = f.get("matrix/data")
                # there, but not as the 2-D dataset of a canonical file
                flat = d is not None and not (isinstance(d, h5py.Dataset) and d.ndim == 2)
        except OSError as exc:
            raise ValueError(f"{p} is not a readable HDF5 file: {exc}") from exc
        if _is_tenx_h5(p):
            raise ValueError(_tenx_message(p))
        raise ValueError(
            f"{p} has no {'2-D ' if flat else ''}dataset 'matrix/data'; found keys {keys} "
            f"- a canonical input .h5 holds matrix/data (features x cells), "
            f"matrix/features and matrix/barcodes"
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
        alias = {m: [a for a, t in _ALIASES.items() if t == m] for m in _MODALITIES}
        valid = [m + (f" (or {', '.join(alias[m])})" if alias[m] else "")
                 for m in _MODALITIES]
        raise ValueError(f"Unknown modality {modality}. The modalities are "
                         f"{', '.join(valid[:-1])} and {valid[-1]}.")
    return m


def _no_names_message(what: str, source: str, n_feat: int, fix: str) -> str:
    """The warning for a matrix written with the fallback names ``feature_<i>``."""
    return (f"The {what} matrix {source} has no feature names. The file names them "
            f"feature_0 to feature_{n_feat - 1}. {fix}")


def _shown(spec) -> str:
    """How a message shows the value of an argument: a selector as it was
    written, an object by its type."""
    return repr(spec) if isinstance(spec, str) else type(spec).__name__


#: the name a 10x Genomics file gives the features of each modality in
#: ``var['feature_types']``
_TENX_TYPE = {"rna": "Gene Expression", "adt": "Antibody Capture", "atac": "Peaks"}   # wording only


def _check_one_feature_type(adata, *, arg, spec, role=None) -> None:
    """``ValueError`` when the matrix of ``adata`` holds features of more than
    one type (``var['feature_types']``, as in a 10x Genomics file read with
    ``gex_only=False``): written as it is, genes and peaks would be one file.

    The values that occur are counted, not the categories of the column, so
    a filtered view passes. ``arg`` is the keyword the user wrote (``'rna'``)
    and ``spec`` the selector, ``None`` for an object; ``to_canonical`` has
    neither and gives the modality as ``role``. They only word the message.
    """
    var = getattr(adata, "var", None)
    if var is None or "feature_types" not in var.columns:
        return
    counts = var["feature_types"].astype(str).value_counts()
    if len(counts) < 2:
        return
    from .resolve import _and_list
    types = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    listed = [f"{n:,} {t!r}" for t, n in types[:4]] \
        + ([f"{len(types) - 4} more"] if len(types) > 4 else [])
    head = (f"{int(counts.sum()):,} features of {len(types)} types: {_and_list(listed)} "
            f"(var['feature_types']). One file holds one modality.")
    family = str(arg or role or "").split("_")[0]
    family = family if family in _TENX_TYPE else None
    example = _TENX_TYPE.get(family)
    if example not in counts.index or role == "atac_gas":
        example = None                          # a 10x file has no gene-activity type
    kind = f"{family.upper()} features" if family else "features of one modality"
    subset = "adata[:, adata.var['feature_types'] == " + repr(example or "<type>") + "]"
    obj = "adata" if arg is None or spec is not None else arg
    way_out = config.hint(
        f"If all of them are {kind}, delete the column from the object first, for "
        f"example del {obj}.var['feature_types'].",
        f"If all of them are {kind}, delete the var column feature_types in Python and "
        f"save the file again.")
    if arg is None:                             # to_canonical, and what mtb.run converts
        flags = f'--{family or "rna"} "X[feature_types={example or "<type>"}]"'
        if family == "atac":                    # --atac does not go without its kind
            flags += " --atac-kind " + ("gene_activity" if role == "atac_gas" else "peak")
        how = "for example" if example else "as"
        raise ValueError(f"The matrix holds {head} " + config.hint(
            f"Pass the features of one type, {how} {subset}, or use "
            f"mtb.io.export_dataset, whose selectors take a filter such as "
            f"'X[feature_types={example or '<type>'}]'. {way_out}",
            f"Write the dataset folder instead, {how} multibench convert SRC OUT {flags}."))
    if spec is not None and _split_var_filter(spec)[1] is None:
        filtered = f"{spec}[feature_types={example or '<type>'}]"
        filtered = config.hint(f"{arg}='{filtered}'", f'--{arg} "{filtered}"')
        add = (f"Add a filter, for example {filtered}." if example else
               f"Add a filter that names the type holding the {kind}, as {filtered}.")
        raise ValueError(config.hint(f"{arg}={spec!r}", f"--{arg} {spec}")
                         + f" selects {head} {add} {way_out}")
    shown = repr(spec) if spec is not None else type(adata).__name__
    raise ValueError(
        config.hint(f"{arg}={shown}", f"--{arg} {spec}") + f" holds {head} " + config.hint(
            f"Pass the features of one type, {'for example' if example else 'as'} "
            f"{arg}={subset}. {way_out}",
            f"A selector takes one filter. Keep the features of one type in the file "
            f"first. In Python: {subset}.write_h5ad(\"one_type.h5ad\")."))


def _pick_matrix(adata, *, layer=None, obsm=None, feature_names=None, what=None,
                 names_arg=None, arg=None, spec=None, role=None):
    """Return ``(X, feature_names)`` for the requested slot of ``adata``.

    ``obsm`` matrices carry no var axis in AnnData, so feature names come from
    ``feature_names`` when given, else the columns of a DataFrame-valued obsm,
    else ``adata.uns[f"{obsm}_names"]`` (when its length matches), else
    ``feature_0..`` - with a ``UserWarning`` naming the fallback, because a
    protein panel written as ``feature_0..feature_29`` loses its marker names
    in every downstream readout. ``what`` labels the warning (``'adt'``);
    ``names_arg`` is the caller's keyword that takes the names
    (``'adt_names'``, ``'feature_names'``), offered first in Python.

    ``X`` and a layer follow the var axis, which must hold one feature type
    (:func:`_check_one_feature_type`; ``arg``, ``spec`` and ``role`` word its
    message, and ``role`` is ``what`` unless given).
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
            slot, uns = f"obsm[{obsm!r}]", f"adata.uns[{obsm + '_names'!r}]"
            store = f"store {slot} as a DataFrame, or put the names in {uns}"
            fix = (f"Pass {names_arg}=, {store}." if names_arg and not config._CLI
                   else f"{store[:1].upper()}{store[1:]}.")
            warnings.warn(_no_names_message(what or "obsm", slot, n_feat, fix),
                          UserWarning, stacklevel=3)
        return X, names
    _check_one_feature_type(adata, arg=arg, spec=spec, role=role or what)
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


def _adt_obsm_keys(adata) -> list:
    """The ``obsm`` keys that can hold a protein matrix: the ones that do not
    start with ``X_``, where scanpy keeps embeddings (``X_pca``, ``X_umap``)."""
    return [k for k in getattr(adata, "obsm", {}) if not str(k).startswith("X_")]


def _layout_family(name: str) -> str | None:
    """``'rna'``, ``'adt'`` or ``'atac'`` for a matrix file name of the
    dataset layout (``rna2.h5``, ``atac_peak.h5``), else ``None``."""
    from .resolve import _LAYOUT_FILE_RE
    m = _LAYOUT_FILE_RE.match(name)
    if not m or m.group(1) is None:
        return None
    return "atac" if name.startswith("atac") else name[:3]


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
        Invalid or conflicting arguments, a non-canonical ``.h5``, or features of several types (full list in Notes).
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
    ``adt.h5``, so it raises. Keys that start with ``X_`` (``X_pca``,
    ``X_umap``) hold embeddings and do not count. ``layer`` and ``obsm`` are
    mutually exclusive.
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

    **One feature type.** One file holds one modality. When
    ``var['feature_types']`` of the object names several types, as in a 10x
    Genomics file read with ``gex_only=False``, ``X`` or a layer would be
    written with genes and peaks together, so it raises. Pass the features
    of one type, or use ``mtb.io.export_dataset``, whose selectors take a
    filter.

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
    - a 10x Genomics ``.h5`` file;
    - an unsupported suffix;
    - ``mod`` missing for a MuData, or given for anything else;
    - ``layer`` and ``obsm`` together;
    - ``modality='adt'`` on an AnnData with ``obsm`` keys but no ``obsm=`` /
      ``layer=``;
    - ``X`` or a layer with several values in ``var['feature_types']``;
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
    if isinstance(src, (str, Path)) and Path(src).suffix.lower() == ".h5" \
            and _is_tenx_h5(Path(src)):
        # its matrix/data made 0.3.4 pass the file through. Said before the
        # checks of the other arguments, which would ask for an out path
        raise ValueError(_tenx_message(Path(src)))
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

    if modality == "adt" and obsm is None and layer is None and _adt_obsm_keys(adata):
        raise ValueError(
            f"adt requested but obsm= not given; found obsm keys {list(adata.obsm)} - "
            "pass obsm='<key>' (or layer=) to select the protein matrix, otherwise "
            "adata.X (usually the RNA) would be written as adt.h5")

    # without modality=, the file name says which features the matrix should
    # hold (mtb.run converts an input to inputs/<role>.h5)
    X, feats = _pick_matrix(adata, layer=layer, obsm=obsm,
                            feature_names=feature_names, what=modality,
                            names_arg="feature_names",
                            role=modality or _layout_family(out.name))
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
    >>> mtb.io.normalize_peak_names("data/MYMULTI/atac.h5", "tmp/atac.h5")

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
    from ..eval import io as eio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = np.asarray(labels)
    if vals.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {vals.shape}")
    # The rows DataFrame.to_csv writes, written here: its csv writer follows
    # the Python version, and up to 3.10 it leaves a lone carriage return
    # bare, which the reader takes for the end of a row. An empty label is
    # written "", because the reader skips an empty line.
    rows = [eio._csv_field(str(v)) or '""' for v in vals]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(os.linesep.join(["x", *rows]) + os.linesep)
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
    ``mdata.obs['<name>:<col>']``. A name without a prefix is a column of
    ``data.obs``, as ``'obs:<col>'`` is.
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
    if spec in data.obs.columns:
        return data.obs[spec]
    if _is_file_name(spec):
        raise ValueError(_file_name_message(spec, what))
    raise ValueError(
        config.hint(f"{what}={spec!r}", f"--{what} {spec}") + " is not a column of "
        + config.hint("mdata.obs" if mu else "data.obs", "the obs of SRC")
        + f". Its columns are {list(data.obs.columns)}. Name a column, as "
        + config.hint(f"{what}='obs:<col>' or {what}='<col>', or pass the values as a "
                      f"Series indexed by barcode.", f"--{what} obs:<col>."))


def _is_file_name(spec: str) -> bool:
    """A ``labels`` / ``batch`` string that names a table, not a column."""
    return Path(spec).suffix.lower() in (".csv", ".tsv", ".txt") or Path(spec).is_file()


def _file_name_message(spec: str, what: str) -> str:
    """``labels`` / ``batch`` take a column of the data: what to do with a file."""
    name = "celltype" if what == "labels" else "batch"
    return config.hint(
        f"{what}={spec!r} is a file name, and {what}= takes a column of the data. Read "
        f"the file and add the column first, for example adata.obs[{name!r}] = "
        f"pd.read_csv({spec!r}, index_col=0)['<column>'], then pass {what}='obs:{name}'.",
        f"--{what} {spec} is a file name, and --{what} takes a column of SRC, as "
        f"obs:<col>. Add the column to the .h5ad file first.")


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
            named = f"give {what}= as a DataFrame or AnnData with named features"
            fix = (f"Pass adt_names=, or {named}." if what == "adt"
                   else f"{named[:1].upper()}{named[1:]}.")
            warnings.warn(_no_names_message(what, "array", n_feat, fix),
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
                   overwrite: bool = False,
                   n_batches: int | None = None) -> Path:
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
        Cell-type labels: a column, named ``'obs:<col>'`` or ``'<col>'`` (a
        MuData's global ``.obs``), ``'<mod>:<col>'`` (``mdata[mod].obs``), a
        Series indexed by barcode, or a sequence.
    batch : str, Series, sequence or None
        Batch per cell, same forms as ``labels``; splits the files per batch
        (``rna1.h5``, ``rna2.h5``, ...) for mosaic or cross. Every cell
        needs a batch.
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
        mosaic, cross or diagonal in several batches, one call per batch.
    overwrite : bool
        ``True`` = replace files already in ``dataset_dir``; ``False`` = raise
        before writing anything.
    n_batches : int or None
        With ``batch_index``: how many batches the calls write in total.
        Files of a higher batch number in ``dataset_dir`` are refused.

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
        A file this call would write exists and ``overwrite`` is ``False``, or the folder holds files of other data.

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

    Without a filter on such data, ``export_dataset`` raises ``ValueError``.

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

    **Column names.** ``"obs:<column>"`` and ``"<column>"`` name a column of
    the object you pass. ``export_dataset`` and ``evaluate`` take both.
    ``run_all`` and ``rescore`` are given a folder, not the object, so they
    take the column itself, for example ``batch=adata.obs["donor"]``.

    **Master cell order.** Every modality is re-indexed by barcode to one
    order: ``data.obs_names``, or the first modality's ``obs_names`` when
    ``data`` is not given. A call that writes labels only and is given no
    object takes its cells from the index of the ``labels`` / ``batch``
    Series. An index of numbers that names no cell of the folder is taken
    as row numbers: the values are written row by row, as a list is.

    - the same barcodes in another order are reordered;
    - a barcode set that differs raises ``ValueError`` naming the strays;
    - non-unique barcodes cannot be matched by name, so the cells are paired
      positionally with a ``UserWarning`` (a different cell count still
      raises).

    A bare array has no barcodes, so the order cannot be checked. Pass a
    DataFrame or AnnData when you are not sure of the order.

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

    Each side keeps the row order of its own object. When ``rna`` and
    ``atac`` hold the same cells in two orders, a later paired export of
    these cells into the same folder (``category='vertical'``, or no
    ``category``) raises ``ValueError``: write the paired files first, or
    use another folder.

    Diagonal data in several batches is one call per batch with
    ``batch_index=N``: ``rna<N>.h5``, ``atac_gas<N>.h5`` or
    ``atac_peak<N>.h5``, ``rna_cty<N>.csv`` and ``atac_cty<N>.csv``.

    For a MuData, ``labels='<col>'`` or ``'<mod>:<col>'`` reads ``<col>``
    from each modality's own ``.obs`` (``mdata['rna'].obs`` for the RNA
    cells, ``mdata['atac'].obs`` for the ATAC cells). When the two columns
    have different names, export RNA and ATAC in two calls.

    **Batches.** With ``batch``, cells are split per batch value (sorted)
    and numbered files are written: ``rna1.h5``, ``rna2.h5`` ...,
    ``adt1.h5`` ..., ``cty1.csv`` ... (the layout of the shipped D52). Only
    cross and mosaic methods read numbered files. For a vertical folder,
    export without ``batch`` and pass the batch column itself, such as
    ``adata.obs["donor"]``, to ``mtb.run_all(batch=...)`` or
    ``mtb.evaluate(batch=...)``.

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
    read. ``n_batches=3`` in these calls says that the data has three
    batches: a file of batch 4 in the folder is then refused (*Existing
    files*).

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
    it. A call also raises ``FileExistsError`` when the folder holds files of
    other data that it does not write: files with other cells, files of
    other batch numbers after ``batch=``, and with ``batch_index=`` files of
    a batch number above ``n_batches``. ``overwrite=True`` does not remove
    them. Files that are not part of the dataset layout are never touched. A
    file written for cells that the folder already holds follows the cell
    order of the files in the folder.

    **Errors.** ``ValueError`` is raised for:

    - nothing to export (``rna``, ``adt``, ``atac`` and ``labels`` all ``None``);
    - ``atac`` without a valid ``atac_kind``, or ``atac_kind`` without ``atac``;
    - an unknown ``category``;
    - a selector string with ``data=None``, a malformed selector, or a
      feature filter that matches nothing;
    - ``X`` or a layer with several values in ``var['feature_types']`` and
      no filter that keeps one;
    - ``adt`` taking ``X`` of an object that also holds an ``obsm`` matrix
      under a key that does not start with ``X_``;
    - data with 0 cells;
    - a bare array whose cell order cannot be checked (no barcodes anywhere)
      or with the wrong row count;
    - a modality whose barcodes differ from the master order (the message
      names the strays);
    - a ``labels`` / ``batch`` Series missing cells, or a sequence of the
      wrong length;
    - ``labels`` / ``batch`` naming a file or a column the object does not
      have;
    - a label that is missing (NaN, None, ``''`` or text that a label file
      reads as missing, such as ``'NA'``): ``mtb.evaluate`` does not score a
      missing label;
    - a batch that is missing, by the same definition;
    - ``batch`` with ``category='vertical'`` or ``'diagonal'``, or together
      with ``batch_index``;
    - ``batch_index`` that is not a positive integer, or without
      ``category='mosaic'`` / ``'cross'`` / ``'diagonal'``;
    - ``n_batches`` without ``batch_index``, not a positive integer, or
      smaller than ``batch_index``;
    - cells that the folder already holds in two different row orders;
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
    given = {"rna": rna, "adt": adt, "atac": atac}      # as written, for messages
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
    _check_batch_args(batch, batch_index, category, adt=adt, atac=atac, n_batches=n_batches)
    diagonal = category == "diagonal"
    if diagonal and labels is not None and rna is None and adt is None and atac is None:
        raise ValueError(
            "category='diagonal' names the label file after the modality it labels "
            "(rna_cty.csv, atac_cty.csv): pass rna= or atac= together with labels=")
    if category == "mosaic" and atac is not None and atac_kind == "gene_activity":
        warnings.warn("Every mosaic method reads peak ATAC. " + config.hint(
            "Export the ATAC as peaks (atac_kind='peak'), or pass "
            "allow_atac_mismatch=True to scan and run_all.",
            "Convert the ATAC as peaks (--atac-kind peak), or pass "
            "--allow-atac-mismatch to scan and run-all."), UserWarning, stacklevel=2)
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
    if not masters:                     # labels only
        if master is None:
            # no object names the cells: a Series indexed by barcode does
            labels, batch = (_rows_numbered(out, x) for x in (labels, batch))
            master = _series_cells(labels, batch)
        if master is not None:
            masters["rna"] = master
    if any(len(m) == 0 for m in masters.values()):
        raise ValueError(_NO_CELLS)

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
            if len(vec) == 0:
                raise ValueError(_NO_CELLS)     # labels alone, and none of them
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
        _check_batch_complete(bvals, spec=batch)
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
            if role == "adt" and not kw and _adt_obsm_keys(a):
                raise ValueError(_adt_in_x_message(given["adt"], _adt_obsm_keys(a)))
            X, feats = _pick_matrix(a, layer=kw.get("layer"), obsm=kw.get("obsm"),
                                    feature_names=adt_names if role == "adt" else None,
                                    what=role,
                                    names_arg="adt_names" if role == "adt" else None,
                                    arg=what, spec=given[what]
                                    if isinstance(given[what], str) else None)
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
                    # the MuData form only for a selector that reads a modality
                    sel = {"rna": rna, "adt": adt, "atac": atac}.get(what)
                    sel = _split_var_filter(sel)[0] if isinstance(sel, str) else ""
                    mod = sel[4:].partition(".")[0] if sel.startswith("mod:") else None
                    counts = f"mod:{mod}.layer:counts" if mod else "layer:counts"
                    hint = config.hint(f"{what}='{counts}'", f"--{what} {counts}")
                _warn_not_counts(X, what, hint, stacklevel=2)
            prepared.append((side, role, X, feats, [str(x) for x in a.obs_names], what))

    # --- the files this call writes, each with its unit: the side and the
    # batch number ('' without one) whose cells it holds
    plan = []                           # (path, kind, payload, unit)
    units = {}                          # unit -> (barcodes or None, number of cells)
    for suf, mask in groups:
        for side, role, X, feats, bars, what in prepared:
            sub = X if mask is None else X[mask]
            if np.ndim(sub) != 2:
                # a sparse X left on disk (backed): _write_canonical refuses
                # it, but only after the folder is made
                raise ValueError(f"matrix must be 2-D (cells x features), got shape "
                                 f"{np.asarray(sub).shape}")
            sub_bars = bars if mask is None else [bars[i] for i in mask]
            path = out / _export_filename(role, suf, category)
            plan.append((path, "matrix", (sub, feats, sub_bars), (side, suf)))
            if role == "atac_peak" and category is None:
                plan.append((out / f"atac{suf}.h5", "link", path, (side, suf)))
        for side, vec in lab.items():
            name = (f"{side}_cty{suf}.csv" if diagonal else f"cty{suf}.csv")
            plan.append((out / name, "labels", vec if mask is None else np.asarray(vec)[mask],
                         (side, suf)))
        for side in dict.fromkeys(list(masters) + list(lab)):
            m = masters.get(side)
            if m is None:               # labels alone, without barcodes
                units[(side, suf)] = (None, len(lab[side]) if mask is None else len(mask))
            else:
                units[(side, suf)] = (m if mask is None else [m[i] for i in mask],
                                      len(m) if mask is None else len(mask))
    existing = [pth.name for pth, _, _, _ in plan if pth.exists() or pth.is_symlink()]
    found = _leftovers(out, {pth.name for pth, _, _, _ in plan}, units, diagonal=diagonal,
                       by_column=batch is not None, batch_index=batch_index,
                       n_batches=n_batches)
    if any(found):
        raise FileExistsError(_leftover_message(
            out, found, written=[int(suf) for suf, _ in groups if suf],
            n_batches=n_batches, replace=bool(existing) and not overwrite))
    if existing and not overwrite:
        from .resolve import _and_list
        raise FileExistsError(
            f"{out} already holds {_and_list(existing)}. Pass "
            f"{config.hint('overwrite=True', '--overwrite')} to replace "
            f"{'it' if len(existing) == 1 else 'them'}.")
    # --- cells that the folder already holds keep the row order of its files
    follow = _row_order(out, units, diagonal=diagonal)
    plan = [(pth, kind, _in_order(kind, payload, follow[unit][0]) if unit in follow
             else payload, unit) for pth, kind, payload, unit in plan]
    for pth, kind, payload, _ in plan:
        if kind == "matrix":
            sub, feats, sub_bars = payload
            if getattr(sub, "ndim", 0) == 2:
                _warn_dense_size(pth, sub.shape[1], sub.shape[0], dtype)

    out.mkdir(parents=True, exist_ok=True)
    for pth, kind, payload, _ in plan:
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
    if follow:
        import sys

        from .resolve import _and_list
        for unit, (_, ref) in follow.items():
            files = [pth.name for pth, _, _, u in plan if u == unit]
            print(f"export_dataset: wrote the rows of {_and_list(files)} in the cell order "
                  f"of {ref}, which the files of {out} follow.", file=sys.stderr, flush=True)
    return out


#: Z1: nothing to write
_NO_CELLS = ("The data holds 0 cells, so there is nothing to write. If you kept the cells "
             "that have a label, no cell had one: check that the barcodes of the label "
             "table are spelled as the cell names of the data (obs_names) are, for example "
             "with the same -1 ending.")


def _check_batch_complete(vals, *, spec) -> None:
    """Refuse a batch vector with missing values (``eval.io.missing_mask``, the
    definition the labels follow): such cells would be left out, or the text
    ``'nan'`` would become a batch of its own."""
    from ..eval import io as eio

    miss = eio.missing_mask(vals)
    if miss.any():
        raise ValueError(
            config.hint(f"batch={_shown(spec)}", f"--batch {spec}")
            + f": {int(miss.sum()):,} of the {len(vals):,} cells have no batch. "
            f"{eio._missing_seen(vals)} Give every cell a batch, or leave these cells out "
            f"first.")


def _adt_in_x_message(spec, keys) -> str:
    """A1: ``adt`` takes ``X`` of an object that also holds ``obsm`` matrices
    (:func:`_adt_obsm_keys`), one of which may be the protein matrix."""
    from .resolve import _and_list
    one = len(keys) == 1
    return (config.hint(f"adt={_shown(spec)}", f"--adt {spec}")
            + f" takes X of this object, and the object also holds the obsm "
            f"{'key' if one else 'keys'} {_and_list([repr(k) for k in keys])}. If one of "
            f"them is the protein matrix, pass "
            + config.hint("adt='obsm:<key>'", "--adt obsm:<key>")
            + ". If X is the protein matrix, delete "
            + config.hint(f"{'it' if one else 'them'} first, for example del "
                          f"adata.obsm[{keys[0]!r}].",
                          "the obsm entries in Python and save the file again."))


#: file stems of the dataset layout, in the order messages list them
_LAYOUT_STEMS = ("rna", "adt", "atac", "atac_peak", "atac_gas", "cty", "rna_cty", "atac_cty")


def _layout_unit(name: str, diagonal: bool):
    """``(side, batch number, is a matrix)`` of a file of the dataset layout,
    else ``None``.

    The files of one unit - a side and a batch number - hold the same cells.
    The batch number is the digits of the name, ``''`` without any. Paired
    data has one side, ``'rna'``. A diagonal call has two: ``rna*``, ``adt*``
    and ``rna_cty*`` are the RNA side, ``atac*`` and ``atac_cty*`` the ATAC
    side, and a plain ``cty*.csv`` has none (``None``).
    """
    from .resolve import _LAYOUT_FILE_RE
    m = _LAYOUT_FILE_RE.match(name)
    if not m:
        return None
    matrix = m.group(1) is not None
    suf = m.group(1) if matrix else m.group(2)
    if not diagonal:
        return "rna", suf, matrix
    if matrix or "_cty" in name:
        return ("atac" if name.startswith("atac") else "rna"), suf, matrix
    return None, suf, matrix


def _layout_order(name: str):
    """Sort key of a layout file name: by batch number, then by
    :data:`_LAYOUT_STEMS`."""
    stem = name.rsplit(".", 1)[0]
    digits = stem[len(stem.rstrip("0123456789")):]
    return int(digits or 0), _LAYOUT_STEMS.index(stem[:len(stem) - len(digits)])


def _holds_other_cells(names, mine) -> bool:
    """True when barcodes ``names`` hold a cell that is not among ``mine``.

    Compared as :func:`_cell_order` compares: a trailing ``-<n>`` suffix does
    not count when the suffix-free names stay unique on both sides.
    Non-unique barcodes cannot be judged and give ``False``.
    """
    names, mine = [str(x) for x in names], [str(x) for x in mine]
    judged = False
    for key in (None, _BARCODE_SUFFIX_RE):
        a = names if key is None else [key.sub("", x) for x in names]
        b = mine if key is None else [key.sub("", x) for x in mine]
        if len(set(a)) != len(a) or len(set(b)) != len(b):
            continue
        judged = True
        if set(a) <= set(b):
            return False
    return judged


def _folder_readers():
    """The resolver's readers of a matrix file's barcodes and of a label
    file's row count, with nothing remembered.

    The resolver keeps what it read by path and modification time, and a
    file written twice within one tick of the file system's clock keeps its
    time. The rules of :func:`export_dataset` judge the folder as it is now.
    """
    from . import resolve
    resolve._sniff_barcode_list.cache_clear()
    resolve._count_label_rows.cache_clear()
    return (resolve._barcodes_of,
            lambda p: resolve._count_label_rows(str(p), p.stat().st_mtime_ns))


def _leftovers(out: Path, planned, units, *, diagonal, by_column, batch_index, n_batches):
    """The layout files of ``out`` that hold other data than this call writes,
    as three lists of names: ``(other batches, other cells, other label counts)``.

    ``planned`` are the names the call writes and ``units`` the cells of each
    unit it writes (:func:`_layout_unit`). A method reads every file of the
    layout as part of the dataset, so a file the call does not write is a
    leftover when one of these holds:

    (a) it has a batch number, its unit is not written, and the call splits
        by a batch column (``by_column``: the column gives the whole set of
        batches), or it is a ``batch_index`` call with ``n_batches`` and the
        number is above ``n_batches``;
    (b) it is a matrix of a unit the call writes, with other cells. A unit
        of the call without barcodes, or with barcodes that are not unique,
        has no names to compare: another number of cells is other cells;
    (c) it is a label file of a unit the call writes, with another number of
        rows than the unit has cells;
    (d) it is a matrix of another unit, the call is not a ``batch_index``
        call, and it holds a cell the call does not write on that side;
    (e) it is a label file without a batch number, the call splits by a batch
        column, and its rows are not as many as the cells of the call.

    Not judged: a matrix without barcodes or with barcodes that are not
    unique; in a diagonal call the files of the side it does not write and a
    plain ``cty*.csv``; in a ``batch_index`` call the files without a batch
    number and those of other batch numbers (up to ``n_batches``); a
    numbered label file when the call writes no numbered file; a label file
    that cannot be read. ``n_batches`` counts the batches of the data, so a
    batch number above it is a leftover under (a) on either side of a
    diagonal call.
    """
    other_batch, other_cells, other_count = [], [], []
    if not out.is_dir():
        return other_batch, other_cells, other_count
    barcodes_of, n_rows = _folder_readers()
    side_cells, side_n = {}, {}         # what the call writes on a side, over its units
    for (side, _), (bars, n) in units.items():
        side_n[side] = side_n.get(side, 0) + n
        if bars is not None:
            side_cells.setdefault(side, []).extend(bars)
    for p in sorted(out.iterdir(), key=lambda p: p.name):
        unit = _layout_unit(p.name, diagonal)
        if unit is None or unit[0] is None or p.name in planned or not p.is_file():
            continue
        side, suf, matrix = unit
        if (side, suf) in units:                        # (b), (c)
            bars, n = units[(side, suf)]
            if matrix:
                old = barcodes_of(p)
                if old is None:
                    continue
                named = bars is not None and len(set(bars)) == len(bars)
                if (_cell_order(old, bars)[0] == "other") if named else (len(old) != n):
                    other_cells.append(p.name)
            elif n_rows(p) not in (None, n):
                other_count.append(p.name)
        elif suf and by_column:                         # (a)
            other_batch.append(p.name)
        elif batch_index is not None:                   # (a), or a file of the build
            if suf and n_batches is not None and int(suf) > n_batches:
                other_batch.append(p.name)
        elif matrix:                                    # (d)
            old = barcodes_of(p)
            if old is not None and side in side_cells \
                    and _holds_other_cells(old, side_cells[side]):
                other_cells.append(p.name)
        elif not suf and by_column and side in side_n:  # (e)
            if n_rows(p) not in (None, side_n[side]):
                other_count.append(p.name)
    return other_batch, other_cells, other_count


def _leftover_message(out: Path, found, *, written, n_batches, replace: bool) -> str:
    """L1: the leftovers of :func:`_leftovers`, why each kind is one, and what
    to do. ``written`` are the batch numbers of the call; ``replace`` adds
    that the files of the call are there too and ``overwrite`` is not set."""
    from .resolve import _and_list, _span

    def names(files):
        files = sorted(files, key=_layout_order)
        return _and_list(files[:6] + ([f"{len(files) - 6} more"] if len(files) > 6 else []))

    def batches(nums):
        return f"{'batch' if len(nums) == 1 else 'batches'} {_span(nums)}"

    other_batch, other_cells, other_count = found
    call = config.hint("this call", "this command")
    text = [f"{out} holds files of other data that {call} does not write."]
    if other_batch:
        theirs = sorted({_layout_order(f)[0] for f in other_batch})
        if n_batches is not None:
            said = (config.hint(f"n_batches={n_batches}", f"--n-batches {n_batches}")
                    + f" says that the data has {n_batches} "
                    f"{'batch' if n_batches == 1 else 'batches'}")
        else:
            said = f"{call} writes {batches(written)}"
        text.append(f"{names(other_batch)} {'belongs' if len(other_batch) == 1 else 'belong'} "
                    f"to {batches(theirs)}, and {said}.")
    if other_cells:
        text.append(f"{names(other_cells)} {'holds' if len(other_cells) == 1 else 'hold'} "
                    f"other cells than {call} writes.")
    if other_count:
        text.append(f"{names(other_count)} {'has' if len(other_count) == 1 else 'have'} "
                    f"another number of labels than {call} has cells.")
    text.append("A method would read them as part of this dataset. " + config.hint(
        "Export to a new folder, and use that folder name in scan and run_all too. To keep "
        "this folder, delete these files and run the export again.",
        "Convert into a new folder, and use that folder name in scan and run-all too. To "
        "keep this folder, delete these files and run convert again."))
    if replace:
        text.append(f"The files {call} writes are there too, so also pass "
                    f"{config.hint('overwrite=True', '--overwrite')}.")
    return " ".join(text)


def _row_order(out: Path, units, *, diagonal) -> dict:
    """``{unit: (index, reference file name)}`` for the units of this call
    whose cells the folder holds in another row order.

    The reference of a unit is a matrix file of that unit in ``out`` with
    unique barcodes that are the cells of the unit; a file the call replaces
    counts too. ``index`` puts the rows of the call in the reference's order
    (:func:`_cell_order`). A label file has no barcodes, so rows written in
    the order of the object would label other cells than the same rows of
    the matrix next to it, and nothing could tell afterwards.

    ``ValueError`` when two references of a unit disagree about the order.
    """
    follow = {}
    if not out.is_dir():
        return follow
    barcodes_of, _ = _folder_readers()
    held = {}                           # unit -> [(file name, barcodes)]
    for p in sorted(out.iterdir(), key=lambda p: p.name):
        unit = _layout_unit(p.name, diagonal)
        if unit is None or not unit[2] or unit[:2] not in units or not p.is_file():
            continue
        bars, old = units[unit[:2]][0], barcodes_of(p)
        if bars is not None and old is not None \
                and _cell_order(bars, old)[0] in ("same", "order"):
            held.setdefault(unit[:2], []).append((p.name, old))
    for unit, refs in held.items():
        refs.sort(key=lambda r: _layout_order(r[0]))
        ref, first = refs[0]
        for name, bars in refs[1:]:
            if _cell_order(bars, first)[0] != "same":
                raise ValueError(
                    f"{ref} and {name} of {out} hold the same cells in different orders, "
                    f"so a row of one is not the cell of the same row of the other. "
                    + config.hint("Export this data again in one call, into a new folder.",
                                  "Convert this data again in one command, into a new "
                                  "folder."))
        kind, idx = _cell_order(units[unit][0], first)
        if kind == "order":
            follow[unit] = (idx, ref)
    return follow


def _in_order(kind: str, payload, idx):
    """The payload of one planned file with its rows taken in ``idx`` order."""
    import scipy.sparse as sp

    if kind == "matrix":
        sub, feats, sub_bars = payload
        sub = sub.tocsr()[idx] if sp.issparse(sub) else np.asarray(sub)[idx]
        return sub, feats, [sub_bars[i] for i in idx]
    if kind == "labels":
        return np.asarray(payload)[idx]
    return payload                      # a link follows the file it links to


def _check_batch_args(batch, batch_index, category, *, adt, atac, n_batches=None) -> None:
    """Refuse the batch / batch_index combinations no method can read, and an
    ``n_batches`` that does not go with ``batch_index``."""
    if batch is not None and batch_index is not None:
        raise ValueError("pass batch= or batch_index=, not both: batch= splits the object "
                         "by a column, batch_index= writes the whole object as one batch")
    if batch_index is not None:
        if isinstance(batch_index, bool) or not isinstance(batch_index, (int, np.integer)) \
                or batch_index < 1:
            raise ValueError(f"batch_index= must be a positive integer (1 writes rna1.h5), "
                             f"got {batch_index!r}")
        if category not in ("mosaic", "cross", "diagonal"):
            raise ValueError(
                f"batch_index= writes numbered files (rna{batch_index}.h5, ...), which only "
                f"mosaic and cross methods read, and the diagonal methods that take "
                f"several batches: pass category='mosaic', category='cross' or "
                f"category='diagonal' (got category={category!r})")
    if n_batches is not None:
        if batch_index is None:
            raise ValueError(config.hint(
                "n_batches= goes with batch_index=. It says how many batches the calls "
                "with batch_index= write in total. With batch=, the column gives the "
                "number.",
                "--n-batches goes with --batch-index. It says how many batches the "
                "commands with --batch-index write in total. With --batch, the column "
                "gives the number."))
        if isinstance(n_batches, bool) or not isinstance(n_batches, (int, np.integer)) \
                or n_batches < 1:
            raise ValueError(config.hint("n_batches=", "--n-batches")
                             + f" must be a positive integer, got {n_batches!r}")
        if batch_index > n_batches:
            raise ValueError(
                config.hint(f"batch_index={batch_index} is larger than "
                            f"n_batches={n_batches}.",
                            f"--batch-index {batch_index} is larger than "
                            f"--n-batches {n_batches}."))
    if batch is not None and category in ("vertical", "diagonal"):
        from .resolve import _one_file_advice
        raise ValueError(
            config.hint("batch=", "--batch") + " writes per-batch files (rna1.h5, "
            "rna2.h5, ...). "
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


def _series_cells(*specs):
    """Cell barcodes of a call that writes labels only and is given no object:
    the index of the first ``labels`` / ``batch`` Series that has one, else
    ``None``.

    A plain ``RangeIndex`` numbers the rows and names no cell, and an index
    that is not unique cannot be matched by name: with either, the call is
    taken row by row, as a list is, and judged by its number of cells.
    """
    import pandas as pd

    named = [[str(x) for x in s.index] for s in specs
             if isinstance(s, pd.Series) and not isinstance(s.index, pd.RangeIndex)]
    if not named or any(len(set(names)) != len(names) for names in named):
        return None
    return named[0]


#: what pandas calls an index of numbers (``Index.inferred_type``)
_NUMBER_INDEX = ("integer", "floating", "mixed-integer-float")


def _rows_numbered(out: Path, spec):
    """``labels`` / ``batch`` as a call that is given no object takes it: a
    Series whose index is numbers that name no cell of ``out`` gets a plain
    ``RangeIndex``; anything else is returned as it is.

    A column of a filtered table keeps the row numbers of the table as its
    index. They are no barcodes, so the values are taken row by row, as
    those of a list are, and the matrix of ``out`` is not held to be a file
    of other cells. Numbers that a matrix of ``out`` has a cell for (an
    object without barcodes names its cells ``'0'``, ``'1'``, ...) are the
    names of cells and stay.
    """
    import pandas as pd

    if isinstance(spec, pd.Series) and not isinstance(spec.index, pd.RangeIndex) \
            and spec.index.inferred_type in _NUMBER_INDEX \
            and not _holds_a_cell(out, spec.index):
        return spec.reset_index(drop=True)
    return spec


def _holds_a_cell(out: Path, names) -> bool:
    """True when a matrix file of the layout in ``out`` holds a cell that one
    of ``names`` names, with or without the trailing ``-<n>`` of its
    barcode."""
    if not out.is_dir():
        return False
    barcodes_of, _ = _folder_readers()
    names = {str(x) for x in names}
    for p in sorted(out.iterdir(), key=lambda p: p.name):
        unit = _layout_unit(p.name, False)
        if unit is None or not unit[2] or not p.is_file():
            continue
        old = barcodes_of(p) or ()
        if not names.isdisjoint(old) \
                or not names.isdisjoint(_BARCODE_SUFFIX_RE.sub("", b) for b in old):
            return True
    return False


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
    global ``.obs`` when a modality lacks the column), and so does a bare
    ``'<col>'``; for a MuData, ``'<mod>:<col>'`` and a bare ``'<col>'`` read
    ``<col>`` from each modality's own ``.obs``."""
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
        elif not mu and not spec.startswith("mod:"):
            col_each = spec                 # '<col>' names the column as 'obs:<col>' does
            if _is_file_name(spec) and not any(spec in sides[side][0][1].obs.columns
                                               for side in present):
                raise ValueError(_file_name_message(spec, "labels"))
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
                    f"{side.upper()} object ({src}); available: {list(a.obs.columns)}"
                    f" Without cell types for the {side.upper()} cells, leave "
                    + config.hint("labels= out and run with evaluate=False.",
                                  "--labels out and run with --no-evaluate."))
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
    """How many entries of a label vector are missing: NaN, None, an empty
    string, or text that a label file reads as missing (``eval.io.missing_mask``)."""
    from ..eval import io as eio

    return int(eio.missing_mask(vals).sum())


def _check_labels_complete(vals, *, spec, where: str, advice: str) -> None:
    """Refuse a label vector with missing values: mtb.evaluate does not score
    a missing label, and the label file would hold one for every such cell."""
    from ..eval import io as eio

    n = _n_missing_labels(vals)
    if n:
        words = eio.missing_words(vals)
        forms, rename = "NaN, None or ''", ""
        if words:
            forms = ("NaN, None, '' or text that a label file reads as missing: "
                     + eio._and(eio._missing_shown(words)))
            rename = (f". If {repr(words[0]) if len(words) == 1 else 'one of them'} is the "
                      f"name of a cell type, rename it")
        raise ValueError(
            f"labels={_shown(spec)}: {n} of the {len(vals)} labels for {where} are missing "
            f"({forms}); {advice}{rename}")


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
        name, sep, _ = spec.partition(":")
        return f"mdata[{name!r}].obs" if sep and name in data.mod else "mdata.obs"
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

    user_spec = spec
    if _is_mudata(obj):
        spec = _mudata_selector(obj, spec)
    a, kw = _select(obj, spec, what=what)
    if "obsm" not in kw:
        # 'X' never reaches _pick_matrix, and the selector the user wrote is
        # known only here
        _check_one_feature_type(a, arg=what, spec=user_spec)
    if not kw:
        return a
    X, feats = _pick_matrix(a, layer=kw.get("layer"), obsm=kw.get("obsm"), what=what)
    b = ad.AnnData(X, obs=a.obs.copy())
    b.var_names = [str(x) for x in feats]
    return b
