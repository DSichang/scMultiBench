"""Adapt arbitrary input formats to the canonical scMultiBench .h5.

Canonical layout: matrix/data (features x cells), matrix/features, matrix/barcodes.

Entry points (exposed as ``mtb.io``):

* :func:`to_canonical` - one matrix (AnnData / .h5ad / .h5mu / .csv / .tsv /
  .loom) -> one canonical ``.h5``; sparse-safe, gzip-compressed, with
  ``modality=`` / ``layer=`` / ``obsm=`` / ``mod=`` selectors.
* :func:`export_dataset` - a whole AnnData/MuData -> a canonical dataset folder
  (``rna.h5``, ``adt.h5``, the ATAC file, ``cty.csv``), optionally per batch
  (``rna1.h5`` ...) or as unpaired diagonal data. A MuData's modalities can
  be named directly (``rna='rna'``).
* :func:`read_canonical` - the inverse (canonical ``.h5`` -> AnnData).

Label files are single-column CSVs: a header ``x``, then one label per cell.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

from .. import _compat, config
from .registry import ATAC_VALUES as _ATAC_KINDS

__all__ = ["export_dataset", "to_canonical", "read_canonical"]


def __dir__() -> list[str]:
    """``dir(mtb.io)`` lists the public API and the dunders only (PEP 562); the
    other names stay reachable."""
    return sorted(set(__all__) | {n for n in globals() if n.startswith("__")})

# peak ids come as chr_start_end, chr-start-end or chr:start-end; Signac needs the last
_PEAK_RE = re.compile(r"^(.+?)[-_:](\d+)[-_](\d+)$")

# the file names are the ones resolve._ROLE_FILE_CANDIDATES looks up
_MODALITIES = ("rna", "adt", "atac", "atac_peak", "atac_gas")
_ALIASES = {"protein": "adt", "peak": "atac_peak", "gas": "atac_gas",
            "gene_activity": "atac_gas"}
_MOD_FILE = {"rna": "rna.h5", "adt": "adt.h5", "atac": "atac.h5",
             "atac_peak": "atac_peak.h5", "atac_gas": "atac_gas.h5"}
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
# '<selector>[<var column>=<value>]'
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
            return _has_fraction(X.data if X.format in ("csr", "csc", "coo")
                                 else X.tocsr().data)
        arr = np.asarray(X)
        if arr.ndim != 2 or arr.size == 0:
            return None
        rows = np.unique(np.linspace(0, arr.shape[0] - 1, min(arr.shape[0], 500)).astype(int))
        return _has_fraction(arr[rows])
    except Exception:                   # backed or exotic storage: no verdict
        return None


def _warn_not_counts(X, name: str, hint: str, *,
                     how: str | None = None) -> None:
    """``UserWarning`` when ``X`` holds non-integer values, which look
    log-normalised; ``hint`` is the selector of the raw counts. ``how``
    replaces the sentence that names it, for a matrix no selector reads."""
    if _matrix_has_fraction(X):
        _compat.warn(
            f"The {name} values are not whole numbers, so they look log-normalised. "
            + (how or f"The methods normalise raw counts themselves. Export raw counts, "
                      f"for example with {hint}."),
            UserWarning)


def _counts_of_object_advice(what: str, spec) -> str | None:
    """The last sentence of the raw-count warning for a modality passed as its
    own object; ``None`` for a selector, which reads ``data`` instead."""
    if isinstance(spec, str) or config._CLI:
        return None
    if _is_anndata(spec) and "counts" in getattr(spec, "layers", {}):
        return f"Set {what}.X = {what}.layers[\"counts\"] first."
    if _is_anndata(spec):
        return f"Put the raw counts in {what}.X first."
    return f"Pass the raw counts as {what}=."


def _raw_count_hint(arg: str, spec, *, to_canonical: bool = False):
    """``(hint, how)`` of :func:`_warn_not_counts`: the selector of the raw
    counts of the matrix ``spec`` reads, else a sentence. With
    ``to_canonical``, ``spec`` is the ``obsm=`` key and ``arg`` the modality.
    A matrix in ``obsm`` has no layer, so its counts are named as an
    ``obsm`` entry (``protein_counts`` for the ADT)."""
    if to_canonical:
        key, pre, var_filter = spec, "", None
        shown = lambda k: config.hint(f"obsm='{k}'", f"--obsm {k}")     # noqa: E731
        layer = config.hint("layer='counts'", "--layer counts")
    else:
        mod, base, var_filter = _parse_selector(spec if isinstance(spec, str) else "")
        pre = f"mod:{mod}." if mod else ""
        key = base[5:] if base.startswith("obsm:") else None
        shown = lambda k: config.hint(f"{arg}='{pre}obsm:{k}'",          # noqa: E731
                                      f"--{arg} {pre}obsm:{k}")
        counts = f"{pre}layer:counts" + (f"[{var_filter[0]}={var_filter[1]}]"
                                         if var_filter else "")
        layer = config.hint(f"{arg}='{counts}'",
                            f"--{arg} \"{counts}\"" if var_filter else f"--{arg} {counts}")
    if key is None:
        return layer, None
    if arg == "adt" and key != "protein_counts":
        return shown("protein_counts"), None
    return None, f"Name the obsm entry that holds the raw counts, as {shown('<key>')}."


def _matrix_key(data, spec):
    """A key equal for two modality arguments that read one matrix; ``data``
    passed again as an object is its ``X``."""
    if isinstance(spec, str):
        mod, base, var_filter = _parse_selector(spec)
        return "selector", mod, base or "X", var_filter
    if spec is data:
        return "selector", None, "X", None
    return "object", id(spec)


def _check_one_matrix_each(data, specs, given) -> None:
    """``ValueError`` when two modalities read the same matrix
    (``rna='X', adt='X'``). ``specs`` are the arguments as resolved, ``given``
    as written."""
    seen = {}
    for what, spec in specs.items():
        if spec is None:
            continue
        key = _matrix_key(data, spec)
        first = seen.setdefault(key, what)
        if first == what:
            continue
        if what == "adt" and key[:3] == ("selector", None, "X") and _adt_obsm_keys(data):
            # the protein matrix is probably one of these: the message that names them
            raise ValueError(_adt_in_x_message(given["adt"], _adt_obsm_keys(data)))
        # a selector as it was written, an object by its keyword alone
        py, cli = ([f"{w}={given[w]!r}" if isinstance(given[w], str) else f"{w}="
                    for w in (first, what)],
                   [f"--{w} {given[w]}" if isinstance(given[w], str) else f"--{w}"
                    for w in (first, what)])
        pair = config.hint(" and ".join(py), " and ".join(cli))
        if what == "adt" and (isinstance(given[what], str) or specs[what] is data
                              or config._CLI):
            # a selector reads data: it is the way out only where one can be read
            place = config.hint("adt='obsm:<key>'", "--adt obsm:<key>")
            how = f"Name where the ADT is, as {place}."
        elif what == "adt":
            how = "Pass the ADT as its own object, as adt=<AnnData>."
        else:
            # the filter is shown only for a file whose features have that type
            types = getattr(data, "var", {}).get("feature_types", ())
            how = config.hint(
                "Pass the ATAC as its own object, as atac=<AnnData>.",
                f"Name where the ATAC is, as --atac \"X[feature_types={_TENX_TYPE[what]}]\"."
                if _TENX_TYPE[what] in set(map(str, types)) else
                "Name where the ATAC is in SRC, with --atac.")
        raise ValueError(f"{pair} name the same matrix. {how}")


#: a trailing ``-<n>`` (10x GEM-well) suffix of a cell barcode
_BARCODE_SUFFIX_RE = re.compile(r"-\d+$")


def _cell_order(names, ref):
    """``("same", None)``, ``("order", idx)`` (``names[idx]`` is in ``ref``
    order), ``("other", None)`` or ``("unknown", None)`` (non-unique
    barcodes). A trailing ``-<n>`` does not count when the names stay unique
    without it."""
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
    """``(X, bars)`` of a gene-activity matrix in the cell order of the
    ``atac_peak.h5`` next to ``out``, which ``atac_cty.csv`` follows; other
    cells warn."""
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
        _compat.warn(
            f"{Path(out).name} holds different cells than {peak} ({len(bars):,} and "
            f"{len(ref):,}, {shared:,} shared). Diagonal methods need gene activity for "
            f"the ATAC cells of atac_peak.h5.", UserWarning)
    return X, bars


def _split_var_filter(spec: str):
    """``'X[feature_types=Peaks]'`` -> ``('X', ('feature_types', 'Peaks'))``;
    a selector without a filter -> ``(spec, None)``."""
    m = _VAR_FILTER_RE.match(spec)
    if not m:
        return spec, None
    return m.group("base"), (m.group("col").strip(), m.group("val").strip())


def _parse_selector(spec: str):
    """``'mod:rna.layer:c[col=v]'`` -> ``('rna', 'layer:c', ('col', 'v'))``;
    the modality is ``None`` without ``mod:``."""
    base, var_filter = _split_var_filter(spec)
    mod = None
    if base.startswith("mod:"):
        mod, _, base = base[4:].partition(".")
    return mod, base, var_filter


def _filter_var(adata, var_filter, *, what, spec):
    """The features of ``adata`` whose ``var[col]`` equals ``value`` (a view)."""
    col, val = var_filter
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
    """On-disk name of a modality file: every vertical variant reads its ATAC,
    of either representation, from ``atac.h5``."""
    if category == "vertical" and modality in ("atac", "atac_peak", "atac_gas"):
        return "atac.h5"
    return _MOD_FILE[modality]


def _is_canonical_h5(path: Path) -> bool:
    """``matrix/data`` is a 2-D dataset; a 10x Genomics ``.h5`` has a 1-D one."""
    try:
        with h5py.File(path, "r") as f:
            d = f.get("matrix/data")
            return isinstance(d, h5py.Dataset) and d.ndim == 2
    except (OSError, KeyError):
        return False


def _is_tenx_h5(path: Path) -> bool:
    """A 10x Genomics ``.h5``: a sparse matrix with ``matrix/indptr``."""
    try:
        with h5py.File(path, "r") as f:
            return "matrix/indptr" in f
    except OSError:
        return False


def _tenx_message(path) -> str:
    return f'{path} is a 10x Genomics file. See the recipe "A 10x .h5 file" in the docs.'


def _is_mudata(obj) -> bool:
    # duck-typed: mudata is an optional import
    return isinstance(getattr(obj, "mod", None), dict) and type(obj).__name__ == "MuData"


def _to_anndata(src):
    """Load ``src`` into an AnnData (a MuData for ``.h5mu``); an in-memory
    object is returned as is."""
    import anndata as ad
    import pandas as pd

    if hasattr(src, "X") and hasattr(src, "obs"):
        return src
    p = Path(src)
    if not p.exists():
        # h5py's own error neither reads like a missing file nor names the cwd
        raise FileNotFoundError(
            f"input file does not exist: {p} (cwd {os.getcwd()})")
    suffix = p.suffix.lower()
    if suffix == ".h5":
        # to_canonical returns a canonical .h5 before this: say what the file holds
        try:
            with h5py.File(p, "r") as f:
                keys = sorted(f.keys())
                d = f.get("matrix/data")
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
    if suffix == ".h5ad":
        return ad.read_h5ad(p)
    if suffix == ".h5mu":
        try:
            import mudata
        except ModuleNotFoundError as exc:
            raise ImportError(
                "reading .h5mu requires the optional 'mudata' package "
                "(pip install mudata); alternatively pass the modality's "
                ".h5ad / AnnData directly."
            ) from exc
        return mudata.read_h5mu(p)
    if suffix in (".csv", ".tsv"):
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(p, sep=sep)
        # a leading non-numeric column holds the cell barcodes
        if df.shape[1] > 1 and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            df = df.set_index(df.columns[0])
        a = ad.AnnData(df.to_numpy(dtype=float))
        a.obs_names = [str(x) for x in df.index]
        a.var_names = [str(c) for c in df.columns]
        return a
    if suffix == ".loom":
        try:
            import loompy  # noqa: F401  (anndata.read_loom needs it)
        except ModuleNotFoundError as exc:
            raise ImportError(
                "reading .loom requires the optional 'loompy' package "
                "(pip install 'multibench-sc[loom]' or pip install loompy); "
                "alternatively convert the input to .h5ad/.csv first."
            ) from exc
        return ad.read_loom(p)
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
                         f"{config._and_list(valid)}.")
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
    """``ValueError`` when ``var['feature_types']`` of ``adata`` holds more
    than one value: one file holds one modality. ``arg``, ``spec`` and
    ``role`` only word the message. Without a 10x type name (``protein_coding``
    and ``lncRNA``) the advice is to delete the column instead."""
    var = adata.var                     # None from runner for an unreadable .h5ad
    if var is None or "feature_types" not in var.columns:
        return
    counts = var["feature_types"].astype(str).value_counts()
    if len(counts) < 2:
        return
    types = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    head = (f"features of 2 types, {types[0]!r} and {types[1]!r}." if len(types) == 2
            else f"features of {len(types)} types, for example {types[0]!r}.")
    family = str(arg or role or "").split("_")[0]
    family = family if family in _TENX_TYPE else None
    example = _TENX_TYPE.get(family)
    if example not in counts.index or role == "atac_gas":
        example = None                          # a 10x file has no gene-activity type
    one = example or "<type>"
    subset = "adata[:, adata.var['feature_types'] == " + repr(one) + "]"
    if not set(counts.index) & set(_TENX_TYPE.values()):
        kind = f"{family.upper()} features" if family else "features of one modality"
        where = (config.hint(f"{arg}={_shown(spec) if spec is not None else type(adata).__name__}",
                             f"--{arg} {spec}") + (" selects" if spec is not None and
                                                   _split_var_filter(spec)[1] is None
                                                   else " holds")
                 if arg is not None else "The matrix holds")
        own = arg is not None and spec is None and not config._CLI
        raise ValueError(f"{where} {head} If all are {kind}, " + config.hint(
            f"run del {arg}.var['feature_types'] first." if own else
            "delete the column var['feature_types'] first.",
            "delete the var column feature_types in Python and save the file again."))
    if arg is None:                             # to_canonical, and what mtb.run converts
        flags = f'--{family or "rna"} "X[feature_types={one}]"'
        if family == "atac":                    # --atac does not go without its kind
            flags += " --atac-kind " + ("gene_activity" if role == "atac_gas" else "peak")
        raise ValueError(f"The matrix holds {head} " + config.hint(
            f"Pass one type, as {subset}.",
            f"Write a dataset folder, as multibench convert SRC OUT {flags}."))
    if spec is not None and _split_var_filter(spec)[1] is None:
        filtered = f"{spec}[feature_types={one}]"
        filtered = config.hint(f"{arg}='{filtered}'", f'--{arg} "{filtered}"')
        raise ValueError(config.hint(f"{arg}={spec!r}", f"--{arg} {spec}")
                         + f" selects {head} Add a filter, as {filtered}.")
    shown = repr(spec) if spec is not None else type(adata).__name__
    raise ValueError(
        config.hint(f"{arg}={shown}", f"--{arg} {spec}") + f" holds {head} " + config.hint(
            f"Pass one type, as {arg}={subset}.",
            f"Save one type first, as {subset}.write_h5ad(\"one_type.h5ad\")."))


def _pick_matrix(adata, *, layer=None, obsm=None, feature_names=None, what=None,
                 names_arg=None, arg=None, spec=None, role=None):
    """``(X, feature_names)`` of ``X``, a layer or an ``obsm`` entry of ``adata``.
    An ``obsm`` matrix has no var: its names come from ``feature_names``, a
    DataFrame's columns or ``uns['<key>_names']``, else ``feature_0..`` with a
    warning. ``names_arg`` is the caller's keyword for the names."""
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
        uns_names = adata.uns.get(f"{obsm}_names")
        if names is None and uns_names is not None and n_feat is not None \
                and len(uns_names) == n_feat:
            names = [str(v) for v in uns_names]
        if names is None and n_feat is not None:
            names = [f"feature_{i}" for i in range(n_feat)]
            slot, uns = f"obsm[{obsm!r}]", f"adata.uns[{obsm + '_names'!r}]"
            store = f"store {slot} as a DataFrame, or put the names in {uns}"
            fix = (f"Pass {names_arg}=, {store}." if names_arg and not config._CLI
                   else f"{store[:1].upper()}{store[1:]}.")
            if n_feat:                  # 0 features have no names, and are refused
                _compat.warn(_no_names_message(what or "obsm", slot, n_feat, fix),
                             UserWarning)
        return X, names
    _check_one_feature_type(adata, arg=arg, spec=spec, role=role or what)
    if layer is not None:
        if layer not in adata.layers:
            # anndata 0.13 lists X among the layers, under the key None
            layers = [k for k in adata.layers if k is not None]
            raise KeyError(f"layer {layer!r} not found" + (
                f"; available: {layers}" if layers else ", and the object has no layers."))
        X = adata.layers[layer]
    else:
        X = adata.X
    if feature_names is not None:
        names = [str(v) for v in feature_names]
        if len(names) != X.shape[1]:
            raise ValueError(f"{len(names)} feature names for {X.shape[1]} features")
        return X, names
    return X, [str(v) for v in adata.var_names]


def _duplicate_names_warning(feats, *, arg, spec, role, obsm=None,
                             mudata=False) -> str | None:
    """The warning for feature names repeated in one matrix, else ``None``.
    ``var_names_make_unique()`` is advised only for an object that holds
    these features alone: on a mixed object it would also rename features of
    another type."""
    import pandas as pd

    names = pd.Index([str(f) for f in feats])
    dup = names[names.duplicated()].unique()
    if not len(dup):
        return None
    selector = spec if isinstance(spec, str) else getattr(spec, "_selector", None)
    where = config.hint(
        f"{arg}={spec!r}" if isinstance(spec, str) else f"{arg}={type(spec).__name__}",
        f"--{arg} {selector or 'X'}")
    kind = "peak" if role == "atac_peak" else "feature"
    head = (f"{where} has {len(dup):,} repeated {kind} {'name' if len(dup) == 1 else 'names'}"
            f", for example {dup[0]}.")
    when = config.hint("before the export", "in Python and save the file again")
    if role == "atac_peak":
        # a peak name is a region: no advice that renames one
        return f"{head} Keep one column for each peak."
    mod, base, var_filter = (_parse_selector(selector) if isinstance(selector, str)
                             else (None, None, None))
    obj = "the object"
    if mod is not None:
        obj, base = f"mod[{mod!r}] of the object", base or "X"
    elif mudata and base is not None:
        mod, _, rest = base.partition(".")
        obj, base = f"mod[{mod!r}] of the object", rest or "X"
    if obsm is not None or (base or "").startswith("obsm:") \
            or not (isinstance(spec, str) or _is_anndata(spec)):
        return f"{head} Give each column its own name."
    if not isinstance(spec, str) and not config._CLI:
        return f"{head} Run {arg}.var_names_make_unique() {when}."
    if var_filter is None:
        return f"{head} Run var_names_make_unique() on {obj} " + config.hint(
            "you export.", "in Python and save the file again.")
    # on the whole object the call also renames a feature of another type
    return f'{head} Split the object first, as in the recipe "A 10x .h5 file" in the docs.'


def _adt_obsm_keys(adata) -> list:
    """The ``obsm`` keys that can hold a protein matrix: the ones that do not
    start with ``X_``, where scanpy keeps embeddings (``X_pca``, ``X_umap``)."""
    # export_dataset may pass data=None
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
    """Warn when the dense ``matrix/data`` exceeds :data:`DENSE_WARN_BYTES`:
    every reader densifies it, however well gzip shrinks the file."""
    itemsize = int(np.dtype(dtype).itemsize)
    size = int(n_feat) * int(n_cell) * itemsize
    if size > DENSE_WARN_BYTES:
        _compat.warn(
            f"{out.name}: matrix/data is stored dense (features x cells): "
            f"{n_feat} x {n_cell} x {itemsize} B = {size / 1e9:.1f} GB uncompressed "
            f"on disk (limit for this warning: {DENSE_WARN_BYTES / 1e9:.0f} GB); "
            f"filter to highly-variable genes / informative peaks before export, "
            f"or pass dtype='float32' to halve it",
            UserWarning)


def _peak_fraction(feats) -> float:
    if not len(feats):
        return 0.0
    return sum(1 for x in feats if _PEAK_RE.match(x)) / len(feats)


def _check_peak_names(modality, feats, *, rna_feats=None, kind_arg=False):
    """Warn when the feature names contradict the ATAC role. Gene activity is
    suggested only when most names are among ``rna_feats``, or when there
    is none to compare with. ``kind_arg`` words the fix as ``atac_kind=``."""
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
        _compat.warn(
            f"{frac:.0%} of the ATAC feature names look like peaks such as "
            f"chr1:100-200, not genes. If the matrix holds peaks, pass {as_peak}.",
            UserWarning)
    elif modality == "atac_peak" and frac < 0.5:
        head = (f"Only {frac:.0%} of the ATAC feature names look like peaks such as "
                "chr1:100-200")
        rename = (". If they are peaks, rename them to chr:start-end. Some methods read "
                  "the peak positions from the names, and "
                  f"{config.hint('mtb.scan', 'multibench scan')} names these methods.")
        gas = f" If the matrix holds gene activity, pass {as_gas}."
        genes = set(rna_feats or ())
        share = sum(1 for x in feats if x in genes) / len(feats) if len(feats) else 0.0
        if rna_feats is None:
            text = head + rename + gas
        elif share > 0.5:
            text = head + f", and {share:.0%} are RNA gene names." + gas
        else:
            text = head + rename
        _compat.warn(text, UserWarning)


#: the ATAC role of each ``atac_kind=``
_ATAC_ROLES = {"peak": "atac_peak", "gene_activity": "atac_gas"}


def _atac_role(feats, rna_feats) -> str:
    """The ATAC role the feature names show, for ``atac_kind=None``: peaks
    (``chr1:100-200``) or the RNA's gene names; neither raises."""
    if _peak_fraction(feats) > 0.5:
        return "atac_peak"
    genes = set(rna_feats or ())
    if genes and sum(1 for x in feats if x in genes) > 0.5 * len(feats):
        return "atac_gas"
    what = ("neither peaks such as chr1:100-200 nor the RNA's gene names" if genes
            else "not peaks such as chr1:100-200")
    raise ValueError(
        f"The ATAC feature names are {what}. Pass "
        + config.hint("atac_kind='peak' or atac_kind='gene_activity'.",
                      "--atac-kind peak or --atac-kind gene_activity."))


def _write_canonical(out: Path, X, feats, bars, *, dtype, compression, block):
    """Stream ``X`` (cells x features) to ``out`` as features x cells, one
    block of features at a time; CSC makes each block a cheap column slice."""
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


@_compat.keywords("io.to_canonical", private="_to_canonical", positional=("convert",), kept={
    "convert": _compat.note("leave it out: a canonical .h5 is passed through as it is"),
    "compression": _compat.note("leave it out for gzip"),
    "block": _compat.note("leave it out")})
def to_canonical(src, out: Path | str | None = None, modality: str | None = None, *,
                 layer: str | None = None, obsm: str | None = None,
                 mod: str | None = None, dtype: str = "float64",
                 category: str | None = None,
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
        the ADT / ATAC checks.
    layer : str | None
        Take the matrix from ``adata.layers[layer]`` instead of ``adata.X``.
    obsm : str | None
        Take the matrix from ``adata.obsm[obsm]`` (e.g. ``'protein'`` for CITE-seq).
    mod : str | None
        MuData modality to convert (``mdata.mod[mod]``); required for a MuData.
    dtype : str
        Stored dtype of ``matrix/data``; ``'float32'`` halves the file.
    category : str | None
        Integration category the file is for (``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``); only changes ATAC filenames.
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
        Invalid or conflicting arguments, a non-canonical ``.h5``, or features
        of several types.
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

    See Also
    --------
    mtb.io.export_dataset : a whole dataset folder (all modalities, labels, batches) in one call.
    mtb.io.read_canonical : the inverse, canonical ``.h5`` -> AnnData.
    mtb.describe_layout : the folder layout and the role -> filename mapping.
    """
    return _to_canonical(src, out, modality, layer=layer, obsm=obsm, mod=mod,
                         dtype=dtype, category=category, feature_names=feature_names)


def _to_canonical(src, out: Path | str | None = None, modality: str | None = None,
                  convert: bool = True, *, layer: str | None = None,
                  obsm: str | None = None, mod: str | None = None,
                  dtype: str = "float64", compression: str | None = "gzip",
                  block: int = 1024, category: str | None = None,
                  feature_names: list | None = None) -> Path:
    """:func:`to_canonical`, with the deprecated ``convert`` (``False``: never
    write), ``compression`` (the h5py filter) and ``block`` (features per
    streaming step)."""
    if isinstance(src, (str, Path)) and _is_canonical_h5(Path(src)):
        return Path(src)
    if isinstance(src, (str, Path)) and Path(src).suffix.lower() == ".h5" \
            and _is_tenx_h5(Path(src)):
        # before the checks of the other arguments, which would ask for an out path
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

    # without modality=, the file name (mtb.run writes inputs/<role>.h5) names it
    X, feats = _pick_matrix(adata, layer=layer, obsm=obsm,
                            feature_names=feature_names, what=modality,
                            names_arg="feature_names",
                            role=modality or _layout_family(out.name))
    if feats is not None and len(feats) == 0:
        raise ValueError(_no_features_message(
            len(adata.obs_names), typed=obsm is None and _has_feature_types(adata, None)))
    bars = [str(v) for v in adata.obs_names]
    _check_peak_names(modality, feats)
    if modality in _COUNT_ROLES:
        hint, how = _raw_count_hint(modality, obsm, to_canonical=True)
        _warn_not_counts(X, modality, hint, how=how)
    if modality == "atac_gas" and category in (None, "diagonal"):
        X, bars = _follow_peak_order(out, X, bars)
    if getattr(X, "ndim", 0) == 2:      # X=None: _write_canonical raises
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
    ValueError
        ``path`` is a 10x Genomics ``.h5`` file; the message says how to read it.

    Examples
    --------
    >>> import multibench as mtb
    >>> d = mtb.config.DEFAULT.data_path / "D11"
    >>> rna = mtb.io.read_canonical(d / "rna.h5")
    >>> rna.shape, rna.var_names[:3]
    >>> adt = mtb.io.read_canonical(d / "adt.h5", sparse=False)

    Notes
    -----
    ``mtb.data.fetch("D11")`` downloads a demo dataset into
    ``mtb.config.DEFAULT.data_path``, not into the current directory.

    The whole matrix is read densely, so memory peaks at the dense size
    (features x cells x 8 bytes) even when the result is CSR: it is meant
    for the shipped benchmark inputs and for what ``to_canonical`` writes,
    not for a 100k-cell peak matrix.

    A method's ``embedding.h5`` (one dataset ``data``) is read into
    ``.obsm["X_emb"]`` of an AnnData without features; another file without
    ``matrix/data`` raises ``ValueError``.
    ``matrix/features`` and ``matrix/barcodes`` are optional; without them
    AnnData's default names (``'0'``, ``'1'``, ...) are kept.

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
    if not _is_canonical_h5(path) and _is_tenx_h5(path):
        # its 1-D matrix/data would fail below with a line about unpacking
        raise ValueError(_tenx_message(path))
    with h5py.File(path, "r") as f:
        if "matrix/data" not in f and isinstance(f.get("data"), h5py.Dataset):
            # a method's embedding.h5: the embedding goes to .obsm["X_emb"]
            from ..eval.io import read_embedding
            emb = read_embedding(path)
            a = ad.AnnData(np.zeros((emb.shape[0], 0)))
            a.obsm["X_emb"] = emb
            return a
        if "matrix/data" not in f:
            raise ValueError(f"{Path(path).name} has no matrix/data: it is not a "
                             f"canonical .h5. Its keys: {sorted(f.keys())}.")
        data = np.array(f["matrix/data"]).T
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


def _normalize_peak_names(src, dst):
    """Copy the canonical ``.h5`` ``src`` to ``dst`` with ATAC peak names rewritten to
    ``chr:start-end``, the spelling Signac's ``CreateChromatinAssay(sep=c(":","-"))``
    expects: a name ``<chr><sep><start><sep><end>`` with ``_``/``-``/``:`` then
    ``_``/``-`` as separators; gene symbols pass unchanged. ``run`` applies it for
    GLUE and Seurat_v3. Returns ``dst``."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        feats = f["matrix/features"][:]
        def _colon_form(b):
            x = b.decode() if isinstance(b, bytes) else str(b)
            m = _PEAK_RE.match(x)
            return f"{m.group(1)}:{m.group(2)}-{m.group(3)}" if m else x
        new = np.array([_colon_form(v) for v in feats], dtype="S")
        del f["matrix/features"]
        f.create_dataset("matrix/features", data=new)
    return dst


# --------------------------------------------------------------------------
# dataset-level export
# --------------------------------------------------------------------------

def _write_labels(labels, path: Path | str) -> Path:
    """Write 1-D ``labels`` as the benchmark's label CSV (header ``x``, one
    label per line, as strings) and return ``path``."""
    from ..eval import io as eio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = np.asarray(labels)
    if vals.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {vals.shape}")
    # not DataFrame.to_csv: up to Python 3.10 its writer leaves a lone carriage
    # return bare, which the reader takes for a row end. "" because the reader
    # skips an empty line.
    rows = [eio._csv_field(str(v)) or '""' for v in vals]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(os.linesep.join(["x", *rows]) + os.linesep)
    return path


def _select(data, spec, *, what):
    """``(adata, to_canonical kwargs)`` of a selector: ``'X'``,
    ``'obsm:<key>'``, ``'layer:<key>'``, each after an optional
    ``'mod:<name>.'``, then an optional ``[<var column>=<value>]``."""
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"{what}= must be a selector string like 'X', 'obsm:<key>', "
                         f"'layer:<key>' or 'mod:<name>', got {spec!r}")
    rest, var_filter = _split_var_filter(spec)
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
    if var_filter is not None:
        if "obsm" in kw:
            raise ValueError(f"{what}={spec!r}: a [<column>=<value>] filter selects "
                             f"features by adata.var; an obsm matrix has no var")
        adata = _filter_var(adata, var_filter, what=what, spec=spec)
    return adata, kw


def _mudata_selector(mdata, spec):
    """A bare modality name (``'rna'``, ``'rna[feature_types=...]'``) of a
    MuData -> the full ``'mod:<name>...'`` selector; anything else unchanged."""
    if not isinstance(spec, str):
        return spec
    base, var_filter = _split_var_filter(spec)
    if base in mdata.mod:
        return f"mod:{base}" + (spec[len(base):] if var_filter else "")
    return spec


def _obs_column(obs, col, *, what, spec, where):
    if col not in obs.columns:
        raise KeyError(f"{what}={spec!r}: column {col!r} not in obs: searched {where}; "
                       f"available: {list(obs.columns)}")
    return obs[col]


def _select_obs(data, spec, *, what):
    """The Series an obs selector names: ``'obs:<col>'`` or ``'<col>'``
    (``data.obs``), ``'mod:<name>.obs:<col>'``, or for a MuData
    ``'<name>:<col>'`` (``mdata[name].obs``, then ``mdata.obs['<name>:<col>']``)."""
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
            # muon keeps a modality's labels in mdata[mod].obs
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
    cols = [str(c) for c in data.obs.columns]
    raise ValueError(
        config.hint(f"{what}={spec!r}", f"--{what} {spec}") + " is not a column of "
        + config.hint("mdata.obs" if mu else "data.obs", "the obs of SRC")
        + (f". Columns: {', '.join(cols[:5])}"
           f"{f' and {len(cols) - 5:,} more' if len(cols) > 5 else ''}."
           if cols else ", which has no columns."))


def _is_file_name(spec: str) -> bool:
    """A ``labels`` / ``batch`` string that names a table, not a column."""
    return Path(spec).suffix.lower() in (".csv", ".tsv", ".txt") or Path(spec).is_file()


def _file_name_message(spec: str, what: str) -> str:
    return config.hint(
        f'{what}={spec!r} is a file name. Add the column first, as in the recipe '
        f'"Labels in a CSV file" in the docs.' if what == "labels" else
        f"{what}={spec!r} is a file name. Add the column to the object first, and pass "
        f"{what}='obs:<col>'.",
        f"--{what} {spec} is a file name. Pass a column of SRC, as --{what} obs:<col>.")


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
    """``(adata, kw)`` of a modality argument: a selector against ``data``, an
    AnnData, a DataFrame (cells x features) or a 2-D array in ``master`` order."""
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
            if n_feat:                  # 0 features have no names, and are refused
                _compat.warn(_no_names_message(what, "array", n_feat, fix),
                             UserWarning)
            feature_names = [f"feature_{i}" for i in range(n_feat)]
        a.var_names = [str(x) for x in feature_names]
        return a, {}
    raise ValueError(
        f"{what}= must be a selector string like 'X', 'obsm:<key>', 'layer:<key>' or "
        f"'mod:<name>', an AnnData, a DataFrame (cells x features) or a 2-D array, "
        f"got {type(spec).__name__}")


def _check_unique_barcodes(names) -> None:
    """``ValueError`` when cell barcodes repeat: they could not be told apart,
    nor matched between modalities."""
    import pandas as pd
    idx = pd.Index([str(x) for x in names])
    dup = idx[idx.duplicated()].unique()
    if len(dup):
        raise ValueError(f"{len(dup):,} barcodes repeat, e.g. {dup[0]}. Make them unique "
                         f"(prefix the sample).")


def _align_cells(a, *, master, master_name, role):
    """``a`` in the master cell order. Other barcodes raise, and so do repeated
    ones (:func:`_check_unique_barcodes`)."""
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
                theirs, ours = (("ATAC", "RNA") if role == "atac" else ("RNA", "ATAC"))
                raise ValueError(
                    f"RNA and ATAC share no barcode ({ours} e.g. {master[0]}, {theirs} "
                    f"e.g. {names[0]}). Make the names match; for different cells pass "
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
    _check_unique_barcodes(master)
    _check_unique_barcodes(names)
    return a


def _as_obs_vector(data, spec, *, what, master):
    """Labels or batch as a 1-D array in master order, from an obs selector, a
    Series aligned by index (a RangeIndex by position) or a sequence."""
    import pandas as pd

    if isinstance(spec, str):
        if data is None:
            raise ValueError(f"{what}={spec!r} is a selector but data is None; pass a "
                             f"Series/array instead")
        ser = _select_obs(data, spec, what=what)
        src_names = [str(x) for x in ser.index]
        if master is not None and src_names != [str(x) for x in master]:
            if len(set(src_names)) != len(src_names):
                # by position every row would go to another cell
                raise ValueError(
                    f"The cell names of {_labels_where(data, spec)} are not unique, so its "
                    f"{what} cannot be matched to the cells. Give each cell a unique name, "
                    f"the same in every modality, first.")
            if set(src_names) == set(master):
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
            if what == "batch":
                # a filtered or sorted column would go to other cells
                from ..eval import io as eio
                _compat.warn(eio._by_position(what),
                             UserWarning)
            return np.asarray(spec)
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


@_compat.keywords("io.export_dataset", private="_export_dataset",
                  kept={"compression": _compat.note("leave it out for gzip")})
def export_dataset(data, dataset_dir: Path | str, *, rna="X",
                   adt=None, atac=None,
                   atac_kind: str | None = None, labels=None,
                   batch=None, dtype: str = "float64",
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
        ``'X[feature_types=Gene Expression]'``) or an AnnData, DataFrame or
        array; ``None`` = no RNA.
    adt : str, AnnData, DataFrame, array or None
        Raw-count matrix for protein (ADT), same forms as ``rna`` (e.g.
        ``'obsm:protein'``); ``None`` = no ADT.
    atac : str, AnnData, DataFrame, array or None
        ATAC matrix, same forms as ``rna`` (e.g. ``'X[feature_types=Peaks]'``);
        ``None`` = no ATAC.
    atac_kind : str or None
        What ``atac`` holds: ``'peak'`` or ``'gene_activity'``; decides the
        ATAC filename. ``None`` = read from the feature names: peaks
        (``chr1:100-200``) or the RNA's gene names.
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
    category : str or None
        Integration category the folder is for (``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``); sets the ATAC and label filenames.
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
        Missing or conflicting arguments, a missing label or batch, or cells
        that do not pair across modalities.
    KeyError
        A selector names a ``mod``, ``obsm``, ``layer``, ``obs`` or ``var`` column that is absent.
    FileExistsError
        A file this call would write exists and ``overwrite`` is ``False``, or the folder holds files of other data.

    Warns
    -----
    UserWarning
        Values that are not whole numbers, or feature names that are
        missing, repeated or contradict ``atac_kind``.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.io.export_dataset(adata, "data/MYCITE", rna="layer:counts",
    ...                       adt="obsm:protein", labels="obs:celltype")
    >>> mtb.io.export_dataset(mdata, "data/MYMULTIOME", rna="rna", atac="atac",
    ...                       atac_kind="peak", labels="obs:celltype",
    ...                       category="vertical")

    See Also
    --------
    mtb.io.to_canonical : one matrix -> one canonical file.
    mtb.describe_layout : the layout being produced and the role -> filename mapping.
    mtb.scan : confirm the folder is found and which methods can run on it.
    """
    return _export_dataset(data, dataset_dir, rna=rna, adt=adt, atac=atac,
                           atac_kind=atac_kind, labels=labels, batch=batch, dtype=dtype,
                           category=category, adt_names=adt_names, batch_index=batch_index,
                           overwrite=overwrite, n_batches=n_batches)


def _export_dataset(data, dataset_dir: Path | str, *, rna="X",
                    adt=None, atac=None,
                    atac_kind: str | None = None, labels=None,
                    batch=None, dtype: str = "float64",
                    category: str | None = None,
                    adt_names: list | None = None,
                    batch_index: int | None = None,
                    overwrite: bool = False,
                    n_batches: int | None = None,
                    compression="gzip") -> Path:
    """:func:`export_dataset`, with the deprecated ``compression``."""
    out = Path(dataset_dir)
    category = _check_category(category)
    if data is None and isinstance(rna, str) and rna == "X":
        rna = None                      # no data to select from: no RNA
    given = {"rna": rna, "adt": adt, "atac": atac}      # as written, for messages
    if _is_mudata(data):
        # a bare modality name selects mdata.mod[name].X; full
        # 'mod:<name>[...]' selectors pass through
        rna, adt, atac = (_mudata_selector(data, x) for x in (rna, adt, atac))
    if atac_kind is not None and atac_kind not in _ATAC_KINDS:
        raise ValueError(
            f"atac_kind= must be one of {list(_ATAC_KINDS)} (got {atac_kind!r}): "
            "'peak' -> atac_peak.h5 (+atac.h5), 'gene_activity' -> atac_gas.h5")
    if atac is None and atac_kind is not None:
        raise ValueError("atac_kind= given without atac=")
    if rna is None and adt is None and atac is None and labels is None:
        raise ValueError("nothing to export: give at least one of rna=, adt=, atac=, labels=")
    _check_batch_args(batch, batch_index, category, adt=adt, atac=atac, n_batches=n_batches)
    if isinstance(adt_names, str):
        # a string is a sequence of characters: 13 letters would name 13 proteins
        # a key of uns is named only when the string is one
        key = repr(adt_names) if adt_names in (getattr(data, "uns", None) or {}) else "'<key>'"
        raise ValueError(
            f"adt_names={adt_names!r} is one string. Pass a list of names, as "
            f"adt_names=list(adata.uns[{key}]).")
    _check_one_matrix_each(data, {"rna": rna, "adt": adt, "atac": atac}, given)
    diagonal = category == "diagonal"
    if diagonal and labels is not None and rna is None and adt is None and atac is None:
        raise ValueError(
            "category='diagonal' names the label file after the modality it labels "
            "(rna_cty.csv, atac_cty.csv): pass rna= or atac= together with labels=")
    if category in ("mosaic", "cross") and batch is None and batch_index is None:
        _compat.warn(f"category={category!r} methods read numbered files (rna1.h5, "
                     f"cty1.csv, ...): pass batch_index= (this object is one batch) "
                     f"or batch= (a column that splits it)", UserWarning)

    # --- the modality matrices, each re-indexed to its side's cell order.
    # Paired categories have one side; diagonal keeps the ATAC cells apart.
    sides = {"rna": [], "atac": []}     # side -> [(role, adata, kw, what, source)]
    master, master_name = _first_master(data, [("rna", rna), ("adt", adt)]
                                        + ([] if diagonal else [("atac", atac)]),
                                        diagonal=diagonal)
    for what, spec in (("rna", rna), ("adt", adt)):
        if spec is not None:
            a, kw = _as_modality(data, spec, what=what, master=master,
                                 feature_names=adt_names if what == "adt" else None)
            sides["rna"].append((what, a, kw, what, _source_name(spec, what)))
    if atac is not None:
        atac_master = master
        if diagonal:
            atac_master = _names_of(atac) or (
                [str(x) for x in data.obs_names] if _is_anndata(data) and not _is_mudata(data)
                else None)
        a, kw = _as_modality(data, atac, what="atac", master=atac_master)
        role = _ATAC_ROLES.get(atac_kind, "atac")     # "atac": from the feature names
        sides["atac" if diagonal else "rna"].append(
            (role, a, kw, "atac", _source_name(atac, "atac")))
    masters = {}
    for side, mats in sides.items():
        if not mats:
            continue
        if side == "rna" and master is not None:
            order, order_name = master, master_name
        else:
            order, order_name = [str(x) for x in mats[0][1].obs_names], mats[0][3]
        _check_unique_barcodes(order)
        masters[side] = order
        sides[side] = [(role, _align_cells(a, master=order, master_name=order_name, role=what),
                        kw, what, src) for role, a, kw, what, src in mats]
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
    labels_by_side = {}
    if labels is not None:
        if diagonal:
            labels_by_side = _diagonal_labels(data, labels, sides)
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
            labels_by_side["rna"] = vec

    # --- batches
    if batch is not None:
        m = masters.get("rna")
        batch_values = _as_obs_vector(data, batch, what="batch", master=m)
        if m is not None and len(batch_values) != len(m):
            raise ValueError(f"batch has {len(batch_values)} entries for {len(m)} cells")
        _check_batch_complete(batch_values, spec=batch)
        keys = sorted(set(batch_values.tolist()), key=lambda v: str(v))
        groups = [(str(i + 1), np.flatnonzero(batch_values == k)) for i, k in enumerate(keys)]
    elif batch_index is not None:
        groups = [(str(int(batch_index)), None)]
    else:
        groups = [("", None)]

    # --- select every matrix and run the content checks, before any write
    prepared = []                       # (side, role, X, feats, bars, what)
    for side, mats in sides.items():
        for role, a, kw, what, src in mats:
            if role == "adt" and not kw and _adt_obsm_keys(a):
                raise ValueError(_adt_in_x_message(
                    given["adt"], _adt_obsm_keys(a),
                    own=not isinstance(given["adt"], str) and given["adt"] is not data))
            X, feats = _pick_matrix(a, layer=kw.get("layer"), obsm=kw.get("obsm"),
                                    feature_names=adt_names if role == "adt" else None,
                                    what=role,
                                    names_arg="adt_names" if role == "adt" else None,
                                    arg=what, spec=given[what]
                                    if isinstance(given[what], str) else None)
            if feats is not None and len(feats) == 0:
                raise ValueError(_no_features_message(
                    len(a.obs_names), what,
                    typed=not kw.get("obsm") and _has_feature_types(a, given[what])))
            if feats is not None:
                repeated = _duplicate_names_warning(
                    feats, arg=what, spec=given[what], role=role, obsm=kw.get("obsm"),
                    mudata=_is_mudata(data))
                if repeated:
                    _compat.warn(repeated, UserWarning)
            if what == "atac":
                # the RNA is prepared first (side 'rna', then the ATAC)
                rna_feats = next((f for _, r, _, f, _, _ in prepared if r == "rna"), None)
                if role == "atac":
                    role = _atac_role(feats, rna_feats)
                _check_peak_names(role, feats, rna_feats=rna_feats, kind_arg=True)
                if category == "mosaic" and role == "atac_gas":
                    _compat.warn("Every mosaic method reads peak ATAC. " + config.hint(
                        "Export the ATAC as peaks (atac_kind='peak'), or pass "
                        "allow_atac_mismatch=True to scan and run_all.",
                        "Convert the ATAC as peaks (--atac-kind peak), or pass "
                        "--allow-atac-mismatch to scan and run-all."),
                        UserWarning)
            if role in _COUNT_ROLES:
                # the command line's --atac-from hands over an object and its selector
                selector = {"rna": rna, "adt": adt, "atac": atac}[what]
                if not isinstance(selector, str) and config._CLI:
                    selector = getattr(selector, "_selector", None)
                hint, how = _raw_count_hint(what, selector)
                _warn_not_counts(X, what, hint,
                                 how=_counts_of_object_advice(what, given[what]) or how)
            prepared.append((side, role, X, feats, [str(x) for x in a.obs_names], what))

    # --- the files this call writes, each with its unit: (side, batch number or '')
    plan = []                           # (path, kind, payload, unit)
    units = {}                          # unit -> (barcodes or None, number of cells)
    for number, mask in groups:
        for side, role, X, feats, bars, what in prepared:
            sub = X if mask is None else X[mask]
            if np.ndim(sub) != 2:
                # a backed sparse X: _write_canonical refuses it only after mkdir
                raise ValueError(f"matrix must be 2-D (cells x features), got shape "
                                 f"{np.asarray(sub).shape}")
            sub_bars = bars if mask is None else [bars[i] for i in mask]
            path = out / _export_filename(role, number, category)
            plan.append((path, "matrix", (sub, feats, sub_bars), (side, number)))
            if role == "atac_peak" and category is None:
                plan.append((out / f"atac{number}.h5", "link", path, (side, number)))
        for side, vec in labels_by_side.items():
            name = (f"{side}_cty{number}.csv" if diagonal else f"cty{number}.csv")
            plan.append((out / name, "labels", vec if mask is None else np.asarray(vec)[mask],
                         (side, number)))
        for side in dict.fromkeys(list(masters) + list(labels_by_side)):
            m = masters.get(side)
            if m is None:               # labels alone, without barcodes
                units[(side, number)] = (None, len(labels_by_side[side]) if mask is None
                                         else len(mask))
            else:
                units[(side, number)] = (m if mask is None else [m[i] for i in mask],
                                         len(m) if mask is None else len(mask))
    existing = [pth.name for pth, _, _, _ in plan if pth.exists() or pth.is_symlink()]
    found = _leftovers(out, {pth.name for pth, _, _, _ in plan}, units, diagonal=diagonal,
                       by_column=batch is not None, batch_index=batch_index,
                       n_batches=n_batches)
    if any(found):
        held = []
        if not prepared and category is None and batch_index is None:
            # labels alone, for a folder that may be written with a category
            held = _matrices_of(out, masters.get("rna"), len(labels_by_side["rna"]),
                                skip={name for names in found for name in names})
        raise FileExistsError(_leftover_message(out, found, n_batches=n_batches, held=held))
    if existing and not overwrite:
        raise FileExistsError(
            f"{out} already holds {config._and_list(existing)}. Pass "
            f"{config.hint('overwrite=True', '--overwrite')} to replace "
            f"{'it' if len(existing) == 1 else 'them'}.")
    # --- cells that the folder already holds keep the row order of its files
    follow = _row_order(out, units, diagonal=diagonal)
    plan = [(pth, kind, _in_order(kind, payload, follow[unit][0]) if unit in follow
             else payload, unit) for pth, kind, payload, unit in plan]
    for pth, kind, payload, _ in plan:
        if kind == "matrix":
            _warn_dense_size(pth, payload[0].shape[1], payload[0].shape[0], dtype)

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

        for unit, (_, refs) in follow.items():
            files = [pth.name for pth, _, _, u in plan if u == unit]
            # a file this call replaced is no file that the new one follows
            ref = next((name for name in refs if name not in files), None)
            names = files[0] + (f" and {len(files) - 1} more" if len(files) > 1 else "")
            print(f"export_dataset: wrote {names} in the cell order "
                  + (f"{refs[0]} had before." if ref is None else f"of {ref}."),
                  file=sys.stderr, flush=True)
    return out


def _has_feature_types(adata, spec) -> bool:
    """The features were selected by type: ``adata`` has a var column
    ``feature_types``, or the selector ``spec`` filters on it."""
    var_filter = _split_var_filter(spec)[1] if isinstance(spec, str) else None
    return "feature_types" in adata.var.columns \
        or bool(var_filter and var_filter[0] == "feature_types")


_NO_CELLS = ("The data holds 0 cells, so there is nothing to write. Check the filter that "
             "selects the cells.")


def _no_features_message(n_cells: int, what: str | None = None, *,
                         typed: bool = False) -> str:
    """The error for a selected matrix with 0 features. ``typed``: the features
    were selected by type (:func:`_has_feature_types`), so the type name is
    the first thing to check."""
    return (f"The {what + ' ' if what else ''}matrix holds 0 features for {n_cells:,} cells. "
            + ("Check the feature type you selected: var['feature_types'] of the whole "
               "object lists the names." if typed
               else "For a 10x file, read it with gex_only=False."))


def _check_batch_complete(vals, *, spec) -> None:
    """Refuse missing batch values (``eval.io.missing_mask``): ``'nan'`` would
    become a batch of its own."""
    from ..eval import io as eio

    miss = eio.missing_mask(vals)
    if miss.any():
        raise ValueError(
            config.hint(f"batch={_shown(spec)}", f"--batch {spec}")
            + f": {int(miss.sum()):,} of the {len(vals):,} cells have no batch"
            f"{eio._missing_example(vals)}. Give every cell a batch.")


def _adt_in_x_message(spec, keys, own: bool = False) -> str:
    """The error for ``adt`` reading ``X`` of an object whose ``obsm`` may hold
    the protein matrix. ``own``: the ADT came as its own object."""
    one = len(keys) == 1
    head = (config.hint(f"adt={_shown(spec)}", f"--adt {spec}") + " takes X, and the object "
            + (f"also holds obsm[{keys[0]!r}]." if one else
               f"also holds {len(keys)} obsm entries, for example {keys[0]!r}."))
    if own and not config._CLI:
        return (f"{head} If X is the protein matrix, run del adt.obsm[{keys[0]!r}] first."
                if one else
                f"{head} If X is the protein matrix, delete them first, as del "
                f"adt.obsm[{keys[0]!r}].")
    key = keys[0] if one else "<key>"
    return (f"{head} If {'that' if one else 'one'} is the protein matrix, pass "
            + config.hint(f"adt='obsm:{key}'", f"--adt obsm:{key}") + ".")


#: file stems of the dataset layout, in the order messages list them
_LAYOUT_STEMS = ("rna", "adt", "atac", "atac_peak", "atac_gas", "cty", "rna_cty", "atac_cty")


def _layout_unit(name: str, diagonal: bool):
    """``(side, batch number or '', is a matrix)`` of a layout file, else
    ``None``; the files of one unit hold the same cells. Paired data has the
    one side ``'rna'``; in a diagonal folder a plain ``cty*.csv`` has none."""
    from .resolve import _LAYOUT_FILE_RE
    m = _LAYOUT_FILE_RE.match(name)
    if not m:
        return None
    matrix = m.group(1) is not None
    number = m.group(1) if matrix else m.group(2)
    if not diagonal:
        return "rna", number, matrix
    if matrix or "_cty" in name:
        return ("atac" if name.startswith("atac") else "rna"), number, matrix
    return None, number, matrix


def _layout_order(name: str):
    """Sort key of a layout file name: by batch number, then by
    :data:`_LAYOUT_STEMS`."""
    stem = name.rsplit(".", 1)[0]
    digits = stem[len(stem.rstrip("0123456789")):]
    return int(digits or 0), _LAYOUT_STEMS.index(stem[:len(stem) - len(digits)])


def _holds_other_cells(names, mine) -> bool:
    """True when ``names`` hold a cell not among ``mine``, compared as
    :func:`_cell_order` compares; ``False`` for non-unique barcodes."""
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
    """The resolver's barcode and label-row readers with their caches cleared:
    a file rewritten within one tick of the file system clock keeps its mtime."""
    from . import resolve
    resolve._sniff_barcode_list.cache_clear()
    resolve._count_label_rows.cache_clear()
    return (resolve._barcodes_of,
            lambda p: resolve._count_label_rows(str(p), p.stat().st_mtime_ns))


def _leftovers(out: Path, planned, units, *, diagonal, by_column, batch_index, n_batches):
    """The layout files of ``out`` the call does not write that hold other
    data: ``(other batches, other cells, other label counts)``. A method
    reads every layout file, so such a file would join the dataset.
    ``units`` maps each unit the call writes to its barcodes and cell count.
    Files without unique barcodes, or unreadable label files, are not judged."""
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
        side, number, matrix = unit
        if (side, number) in units:     # a unit the call writes: other cells or rows
            bars, n = units[(side, number)]
            if matrix:
                old = barcodes_of(p)
                if old is None:
                    continue
                named = bars is not None and len(set(bars)) == len(bars)
                if (_cell_order(old, bars)[0] == "other") if named else (len(old) != n):
                    other_cells.append(p.name)
            elif n_rows(p) not in (None, n):
                other_count.append(p.name)
        elif number and by_column:      # a batch column gives every batch
            other_batch.append(p.name)
        elif batch_index is not None:   # other batches are the build, up to n_batches
            if number and n_batches is not None and int(number) > n_batches:
                other_batch.append(p.name)
        elif matrix:                    # another unit with a cell the call does not write
            old = barcodes_of(p)
            if old is not None and side in side_cells \
                    and _holds_other_cells(old, side_cells[side]):
                other_cells.append(p.name)
        elif not number and by_column and side in side_n:   # unnumbered labels, batch split
            if n_rows(p) not in (None, side_n[side]):
                other_count.append(p.name)
    return other_batch, other_cells, other_count


def _matrices_of(out: Path, cells, n: int, *, skip) -> list:
    """The matrix files of the layout in ``out`` that hold the cells of a call
    which writes labels only: the barcodes ``cells``, or ``n`` cells when the
    call has no unique barcodes to compare. ``skip`` are the leftovers."""
    barcodes_of, _ = _folder_readers()
    named = cells is not None and len(set(cells)) == len(cells)
    held = []
    for p in sorted(out.iterdir(), key=lambda p: p.name):
        unit = _layout_unit(p.name, False)
        if unit is None or not unit[2] or p.name in skip or not p.is_file():
            continue
        old = barcodes_of(p)
        if old is not None and ((_cell_order(old, cells)[0] in ("same", "order")) if named
                                else (len(old) == n)):
            held.append(p.name)
    return held


def _leftover_message(out: Path, found, *, n_batches, held=()) -> str:
    """The error for the leftovers ``found`` by :func:`_leftovers`. With
    ``held`` (:func:`_matrices_of`) the leftovers are likely the rest of the
    same dataset, so the advice is to rerun the export with labels."""
    from .resolve import _span

    def names(files):
        files = sorted(files, key=_layout_order)
        more = len(files) - 1
        return files[0] + (f" and {more} more {'file' if more == 1 else 'files'}" if more else "")

    other_batch, other_cells, other_count = found
    if sum(map(bool, found)) > 1:
        what = f"{names([f for kind in found for f in kind])} of other data"
    elif other_batch:
        theirs = sorted({_layout_order(f)[0] for f in other_batch})
        what = (f"{names(other_batch)} of {'batch' if len(theirs) == 1 else 'batches'} "
                f"{_span(theirs)}")
        if n_batches is not None:
            what += ", above " + config.hint(f"n_batches={n_batches}",
                                             f"--n-batches {n_batches}")
    elif other_cells:
        what = f"{names(other_cells)} with other cells"
    else:
        what = f"{names(other_count)} with another number of labels"
    if held:
        how = config.hint(
            "To add labels, run the export call that wrote the folder again with labels= "
            "and overwrite=True.",
            "To add labels, run the convert command that wrote the folder again with "
            "--labels and --overwrite.")
    else:
        how = config.hint("Export to a new folder.", "Convert into a new folder.")
    return f"{out} holds {what}. {how}"


def _row_order(out: Path, units, *, diagonal) -> dict:
    """``{unit: (index, reference file names)}`` for the units whose cells the
    folder holds in another row order; ``index`` puts the call's rows in that
    order, as a label file has no barcodes to pair it by. ``ValueError``
    when two files of a unit disagree."""
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
                    f"{ref} and {name} of {out} hold the same cells in different orders. "
                    + config.hint("Export this data again in one call, into a new folder.",
                                  "Convert this data again in one command, into a new "
                                  "folder."))
        kind, idx = _cell_order(units[unit][0], first)
        if kind == "order":
            follow[unit] = (idx, [name for name, _ in refs])
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
    return payload                      # a link


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
                "n_batches= goes with batch_index= only. Leave n_batches= out.",
                "--n-batches goes with --batch-index only. Leave --n-batches out."))
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
        _compat.warn(
            "batch= with atac=: only mosaic methods read numbered ATAC files. For mosaic "
            "pass category='mosaic'; otherwise " + _batch_column_advice(),
            UserWarning)


def _names_of(spec):
    """Cell barcodes an AnnData / DataFrame argument carries, else ``None``."""
    if _is_anndata(spec):
        return [str(x) for x in spec.obs_names]
    if hasattr(spec, "index") and hasattr(spec, "columns"):
        return [str(x) for x in spec.index]
    return None


def _series_cells(*specs):
    """The cells of a labels-only call given no object: the index of the
    first ``labels`` / ``batch`` Series, unless it is a ``RangeIndex`` or not
    unique (then ``None``: the call is taken row by row)."""
    import pandas as pd

    named = [[str(x) for x in s.index] for s in specs
             if isinstance(s, pd.Series) and not isinstance(s.index, pd.RangeIndex)]
    if not named or any(len(set(names)) != len(names) for names in named):
        return None
    return named[0]


#: what pandas calls an index of numbers (``Index.inferred_type``)
_NUMBER_INDEX = ("integer", "floating", "mixed-integer-float")


def _rows_numbered(out: Path, spec):
    """``spec`` with a ``RangeIndex`` when it is a Series indexed by numbers
    that name no cell of ``out``: a filtered table's row numbers, not
    barcodes. An object without barcodes names its cells ``'0'``, ``'1'``."""
    import pandas as pd

    if isinstance(spec, pd.Series) and not isinstance(spec.index, pd.RangeIndex) \
            and spec.index.inferred_type in _NUMBER_INDEX \
            and not _holds_a_cell(out, spec.index):
        return spec.reset_index(drop=True)
    return spec


def _holds_a_cell(out: Path, names) -> bool:
    """True when a layout matrix in ``out`` holds a cell of ``names``, with or
    without the trailing ``-<n>`` of its barcode."""
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


def _source_name(spec, what) -> str:
    """How an error names the object a modality came from."""
    if isinstance(spec, str):
        if spec.startswith("mod:"):
            return f"mdata[{_parse_selector(spec)[0]!r}]"
        return "data"
    return what


def _diagonal_labels(data, spec, sides) -> dict:
    """``{side: labels}`` of a diagonal export, each side from its own cells:
    a column name is read from each object's own ``.obs`` (a MuData's global
    ``.obs`` when a modality lacks it)."""
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
            columns = {c for side in present for c in sides[side][0][1].obs.columns}
            if _is_file_name(spec) and spec not in columns:
                raise ValueError(_file_name_message(spec, "labels"))
            name, _, col = spec.partition(":")
            if name and col and spec not in columns:
                each = all(col in sides[side][0][1].obs.columns for side in present)
                raise ValueError(_mod_prefix_message(spec, col if each else None))
    out = {}
    for side in present:
        role, a, kw, what, src = sides[side][0]
        bars = [str(x) for x in a.obs_names]
        if col_each is not None and col_each in a.obs.columns:
            out[side] = np.asarray(a.obs[col_each])
            _check_labels_complete(out[side], spec=spec, where=f"the {side.upper()} cells "
                                   f"({src})", advice=_diagonal_label_advice(data, spec, sides),
                                   when_none=_no_cell_types_advice(side))
            continue
        if col_each is not None:
            if spec.startswith("obs:") and mu and col_each in data.obs.columns:
                ser = data.obs[col_each]
            else:
                raise KeyError(
                    config.hint(f"labels={spec!r}", f"--labels {spec}")
                    + f": column {col_each!r} not in obs of the "
                    f"{side.upper()} object ({src}); available: {list(a.obs.columns)}. "
                    + (_other_column_advice(data, col_each, side, sides)
                       or _no_cell_types_advice(side)))
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
                               advice=_diagonal_label_advice(data, spec, sides),
                               when_none=_no_cell_types_advice(side))
    return out


def _other_column_advice(data, col: str, side: str, sides) -> str:
    """Advice for a side whose object lacks the label column ``col`` but has
    others, where its cell types may be; ``""`` when there is no way out."""
    role, a, _kw, _what, src = sides[side][0]
    if not len(a.obs.columns) or (config._CLI and _is_mudata(data)):
        return ""
    obj = src if src.startswith("mdata[") else side
    return config.hint(
        f"Copy its cell-type column first, as {obj}.obs[{col!r}] = "
        f"{obj}.obs['<its column>'].",
        "Convert each side with its own command and its own --labels.")


def _no_cell_types_advice(side: str) -> str:
    """Advice for a diagonal side without cell types."""
    return (f"Without cell types for the {side.upper()} cells, leave "
            + config.hint("labels= out.", "--labels out."))


def _mod_prefix_message(spec: str, col: str | None) -> str:
    """The error for ``labels='<mod>:<col>'`` on data that is not a MuData;
    ``col`` when it is a column of every object, else ``None``."""
    return config.hint(
        f"labels={spec!r} names a modality of a MuData, and this data is not one. "
        + (f"Write labels='obs:{col}'." if col else "Write labels='obs:<col>'."),
        f"--labels {spec} names a modality of a .h5mu file, and SRC is not one. "
        + (f"Write --labels obs:{col}." if col else "Write --labels obs:<col>."))


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


def _check_labels_complete(vals, *, spec, where: str, advice: str,
                           when_none: str = "") -> None:
    """Refuse missing labels (``eval.io.missing_mask``): ``mtb.evaluate`` does
    not score them. ``when_none`` closes the message when no cell has one."""
    from ..eval import io as eio

    n = int(eio.missing_mask(vals).sum())
    if n:
        words = eio.missing_words(vals)
        forms = f"NaN, None, '' or text such as {words[0]!r}" if words else "NaN, None or ''"
        if when_none and n == len(vals):
            advice = f"{advice}. {when_none}"
        raise ValueError(
            f"labels={_shown(spec)}: {n} of the {len(vals)} labels for {where} are missing "
            f"({forms}); {advice}")


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


def _export_filename(role: str, number: str, category: str | None) -> str:
    """On-disk name of one modality file of :func:`export_dataset`."""
    if role.startswith("atac") and category in ("vertical", "mosaic"):
        return f"atac{number}.h5"
    return f"{role}{number}.h5"


def _select_object(obj, spec: str, *, what: str):
    """An AnnData of the matrix ``spec`` selects from ``obj``, with its
    ``.obs``: the ``atac=`` of the command line's ``--atac-from``."""
    import anndata as ad

    user_spec = spec
    if _is_mudata(obj):
        spec = _mudata_selector(obj, spec)
    a, kw = _select(obj, spec, what=what)
    if "obsm" not in kw:
        # 'X' never reaches _pick_matrix, and only here is the selector as written
        _check_one_feature_type(a, arg=what, spec=user_spec)
    if kw:
        X, feats = _pick_matrix(a, layer=kw.get("layer"), obsm=kw.get("obsm"), what=what)
        b = ad.AnnData(X, obs=a.obs.copy())
        b.var_names = [str(x) for x in feats]
        a = b
    a._selector = spec          # for the raw-count hint of export_dataset
    return a


__getattr__ = _compat.module_getattr(__name__, "io", old={
    "normalize_peak_names": (_normalize_peak_names, _compat.note("run renames the peaks itself")),
})
