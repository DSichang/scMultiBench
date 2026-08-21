"""Resolve a catalog dataset id + method to concrete input file paths."""
from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path

from .. import config
from . import registry
from .schema import base_modality, is_label_role

# A few variant roles name a modality *representation* whose on-disk filename
# differs from the role token (e.g. the diagonal ATAC roles). Candidate bases
# are tried in order; the first existing file wins.
_ROLE_FILE_CANDIDATES = {
    "atac_gas": ("atac_gas", "atac"),    # ATAC gene-activity score
    "atac_peak": ("atac_peak", "peak"),  # raw ATAC peaks
}


def _resolve_role(ds_dir: Path, role: str) -> Path:
    """Pick the real on-disk file for a modality role in a flat dataset dir.

    Tries the role token and known aliases. Label roles (anything matching
    ``cty`` or ``label``) try ``.csv`` first since cell-type files are CSV;
    other roles try ``.h5`` then ``.h5`` with a trailing digit (some datasets
    store batched files). Returns the canonical ``<base>.h5`` (or ``.csv``
    for label roles) as a fallback when nothing matches.
    """
    is_label = ("cty" in role) or ("label" in role)
    bases = _ROLE_FILE_CANDIDATES.get(role, (role,))
    exts = (".csv", ".h5") if is_label else (".h5",)
    for base in bases:
        for ext in exts:
            for suffix in (ext, f"1{ext}"):
                p = ds_dir / f"{base}{suffix}"
                if p.exists():
                    return p
    fallback_ext = ".csv" if is_label else ".h5"
    return ds_dir / f"{bases[0]}{fallback_ext}"


def _resolve_data_dir(ds_dir: Path, method: str | None = None) -> str:
    """Directory of spatial slices for a registration ``data_dir`` role.

    Spatial-registration methods (PASTE/PASTE2/SPIRAL/GPSA) take a DIRECTORY of
    per-slice ``.h5ad`` files, not a per-feature file. Datasets keep those slices
    under ``<ds_dir>/processed/`` (sometimes directly in ``<ds_dir>/``). Return the
    first dir that actually holds ``*.h5ad`` slices, WITH a trailing separator —
    the upstream scripts string-concatenate ``data_dir + "*.h5ad"``.
    """
    import os
    for cand in (ds_dir / "processed", ds_dir):
        if cand.is_dir() and any(cand.glob("*.h5ad")):
            # SPIRAL derives each slice's cross-slice ID prefix from
            # filename.split('_')[0]; when every slice shares that leading
            # token (e.g. D63's `modified_*.h5ad`) the prefixed obs_names
            # collide and `coord.loc[ann.obs_names]` explodes. Stage a
            # sibling dir of symlinks with a UNIQUE leading token per slice.
            if method == "SPIRAL":
                return _stage_unique_leading_token(cand)
            return os.path.join(str(cand), "")
    # No `.h5ad` slices anywhere: this is a NON-spatial `data_dir` role (e.g.
    # scBridge, which takes the dataset DIRECTORY plus bare filenames). Prefer a
    # real `processed/` if one exists (keeps the spatial error message pointing
    # at the slice dir); otherwise fall back to the dataset dir itself, which
    # always exists, instead of a bogus `<ds>/processed/`.
    proc = ds_dir / "processed"
    return os.path.join(str(proc if proc.is_dir() else ds_dir), "")


def _stage_unique_leading_token(slice_dir: Path) -> str:
    """Return a dir of the same `.h5ad` slices but with a unique leading
    filename token (the part before the first '_').

    SPIRAL builds each slice's cross-slice cell-ID prefix from
    ``filename.split('_')[0]``. Datasets whose slice files share that token
    (D63: ``modified_E14-16h_a_S07.h5ad`` ... all split to ``modified``)
    make SPIRAL's per-cell prefixes collide across slices, which inflates
    ``coord.loc[ann.obs_names]`` into a cartesian product and crashes the
    coordinate-assignment step. If the leading tokens are already unique we
    return the original dir untouched; otherwise we materialize a sibling
    ``<dir>__spiral_uniqtok/`` of SYMLINKS (no data copy, original dir is
    never mutated) whose names start with a unique token derived from the
    trailing token of each stem.
    """
    import os
    files = sorted(p for p in slice_dir.glob("*.h5ad") if p.is_file())
    if not files:
        return os.path.join(str(slice_dir), "")
    lead = [f.name.split("_")[0] for f in files]
    if len(set(lead)) == len(lead):
        return os.path.join(str(slice_dir), "")  # already unique
    staged = slice_dir.parent / (slice_dir.name + "__spiral_uniqtok")
    staged.mkdir(parents=True, exist_ok=True)
    used = set()
    for f in files:
        stem = f.stem
        tok = stem.split("_")[-1] or stem  # trailing token, e.g. S07
        base = tok
        i = 1
        while tok in used:
            i += 1
            tok = f"{base}{i}"
        used.add(tok)
        link = staged / f"{tok}_{f.name}"
        if link.is_symlink() or link.exists():
            try:
                link.unlink()
            except OSError:
                pass
        os.symlink(os.path.realpath(str(f)), str(link))
    return os.path.join(str(staged), "")


def inputs_for(dataset: str, method: str, category: str,
               modalities: list[str] | set[str] | None = None,
               data_path: Path | str | None = None,
               check: bool | None = None) -> dict:
    """Return ``{modality_role: path}`` for a method's variant on a dataset.

    The dataset tree is **flat** (``<data_path>/<dataset>/<file>``). Each role is
    resolved to the actual file present in that dir (the role token, or a known
    alias such as ``atac_peak``->``peak.h5`` / ``atac_gas``->``atac.h5``),
    falling back to ``<role>.h5`` when no candidate exists. Use
    :func:`labels_for` to get the matching cell-type label CSVs.

    Parameters
    ----------
    dataset : dataset folder name under ``data_path``.
    method : registry id (``KeyError`` with a did-you-mean hint otherwise).
    category : ``vertical``/``diagonal``/``mosaic``/``cross`` (validated:
        ``ValueError`` listing the valid tokens on a typo).
    modalities : the variant's modality tokens (see
        ``method_info(m)['supports']``); ``protein`` is accepted for ``adt``,
        unknown tokens raise ``ValueError`` naming the vocabulary.
    data_path : root that CONTAINS the dataset folder; default
        ``config.DEFAULT.data_path``.
    check : what to do when a resolved path does not exist on disk.
        ``None`` (default) - return the best-effort paths but emit a
        ``UserWarning`` listing the missing ones; ``True`` - raise
        ``FileNotFoundError`` and also run the content preflight: the
        matrix-orientation check (``ValueError`` for a cells x features file),
        the label-length check (``ValueError`` when a label CSV has a different
        number of rows than the modality file it labels) and, for ``data_dir``
        methods, the directory-content check (``FileNotFoundError`` when a
        spatial-registration method finds fewer than two ``*.h5ad`` slices, or
        a slice lacks ``obsm['spatial']``; when scBridge's bare filenames are
        absent). This is what :func:`multibench.scan` reports per row as
        ``files_ok`` / ``files_reason``; ``False`` - fully silent.

    Variant selection:
      * If ``modalities`` is given, the exact variant matching
        ``(category, set(modalities))`` is selected via ``spec.select``.
      * If ``modalities`` is None and exactly one variant matches ``category``,
        that variant is used.
      * If ``modalities`` is None and MORE than one variant matches ``category``,
        a ``ValueError`` is raised listing the available modality-sets and asking
        the caller to disambiguate with ``modalities=``.
      * If no variant matches ``category``, a ``KeyError`` is raised.
    """
    base = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    ds_dir = base / dataset  # flat layout: data/<dataset>/<file>
    spec = registry.get(method)
    registry.check_category(category)
    modalities = registry.normalize_modalities(modalities)

    if modalities is not None:
        variant = spec.select(category, set(modalities))
    else:
        candidates = [v for v in spec.variants if v.when.get("category") == category]
        if not candidates:
            raise KeyError(f"{method} has no variant for category={category!r}")
        if len(candidates) > 1:
            available = [v.when.get("modalities", []) for v in candidates]
            raise ValueError(
                f"{method} has multiple variants for category={category!r}: "
                f"modality-sets {available}; pass modalities= to disambiguate"
            )
        variant = candidates[0]

    # Skip args with a const (they don't need on-disk resolution) and out_dir.
    # an arg may name ONE role (a.role) or GROUP several under one flag (a.roles)
    roles = [r
             for a in variant.args if getattr(a, "const", None) is None
             for r in (getattr(a, "roles", None) or [a.role])
             # "=VALUE" entries are literals emitted verbatim (e.g. scMoMaT's
             # per-batch `None` placeholders), not input roles to find on disk
             if r and not str(r).startswith("=")
             and r not in ("out_dir", "data_dir")]
    out = {role: str(_resolve_role(ds_dir, role)) for role in roles}
    # A `data_dir` role points at the DIRECTORY of spatial slices (registration).
    if any(a.role == "data_dir" for a in variant.args):
        out["data_dir"] = _resolve_data_dir(ds_dir, method)
    missing = {r: p for r, p in out.items() if not Path(p).exists()}
    if check:
        if missing:
            raise FileNotFoundError(
                f"{method}/{dataset}/{category}: input files not found on disk: "
                f"{missing}. Available files in {ds_dir}: "
                f"{sorted(q.name for q in ds_dir.glob('*')) if ds_dir.is_dir() else '(dir missing)'}"
            )
        _check_orientation(method, dataset, category, out)
        _check_label_lengths(method, dataset, category, out)
        if "data_dir" in out:
            ok, why = _check_data_dir(variant, out["data_dir"])
            if not ok:
                raise FileNotFoundError(f"{method}/{dataset}/{category}: {why}")
    elif check is None and missing:
        # The default used to hand back phantom paths in silence, so a typo in
        # the dataset folder only surfaced minutes later inside the method's
        # conda env. Warn here; check=True raises, check=False stays quiet.
        warnings.warn(
            f"{method}/{dataset}/{category}: {len(missing)} resolved input path(s) "
            f"do not exist: {missing}; pass check=True to raise, check=False to silence",
            UserWarning, stacklevel=2)
    return out



@functools.lru_cache(maxsize=512)
def _sniff_h5(path: str, mtime_ns: int):
    """Read (shape, n_features, n_cells) of a canonical .h5, cached by mtime.

    scan() runs the orientation preflight once per (method, variant) row, so
    the same handful of files used to be opened and sniffed dozens of times
    per call; the mtime key keeps the cache honest across rewrites.
    """
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "matrix/data" not in f:
                return None
            shape = tuple(f["matrix/data"].shape)
            if len(shape) != 2:
                return None
            if "matrix/features" not in f or "matrix/barcodes" not in f:
                return None
            return (shape, int(f["matrix/features"].shape[0]),
                    int(f["matrix/barcodes"].shape[0]))
    except OSError:
        return None


def _check_orientation(method, dataset, category, resolved):
    """Reject a transposed matrix at preflight instead of many minutes later.

    Modality files store ``matrix/data`` as (features x cells). Storing it the
    other way round is the easy mistake - cells x features is the scanpy/AnnData
    convention, and describe_layout only says "the matrix under matrix/data"
    without stating an orientation. The file-existence check cannot see it, so
    the method is dispatched, pays its conda-env startup (and for a slow method
    potentially hours of compute) and only then dies inside third-party code
    with an error that does not mention orientation at all.

    ``matrix/features`` and ``matrix/barcodes`` pin the intended orientation
    without reference to the labels, so this is checkable up front. A square
    matrix is genuinely ambiguous and is left alone.
    """
    for path in resolved.values():
        p = Path(path)
        if p.suffix != ".h5" or not p.is_file():
            continue
        sniff = _sniff_h5(str(p), p.stat().st_mtime_ns)
        if sniff is None:
            continue
        shape, n_feat, n_cell = sniff
        if n_feat == n_cell or shape == (n_feat, n_cell):
            continue                  # ambiguous, or already correct
        if shape == (n_cell, n_feat):
            raise ValueError(
                f"{method}/{dataset}/{category}: {p.name} stores matrix/data as "
                f"{shape}, which is cells x features. This layout expects "
                f"features x cells - here ({n_feat}, {n_cell}), matching "
                f"matrix/features ({n_feat}) and matrix/barcodes ({n_cell}). "
                f"Re-export with mtb.io.to_canonical(src, dst), or transpose "
                f"matrix/data."
            )


@functools.lru_cache(maxsize=512)
def _sniff_features(path: str, mtime_ns: int, n: int = 50) -> tuple:
    """First ``n`` feature names of a canonical .h5 (cached by mtime), or ()."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "matrix/features" not in f:
                return ()
            raw = f["matrix/features"][:n]
    except OSError:
        return ()
    return tuple(x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in raw)


@functools.lru_cache(maxsize=512)
def _count_label_rows(path: str, mtime_ns: int):
    """Number of label rows in a label CSV (header excluded), cached by mtime;
    ``None`` when the file cannot be parsed."""
    import pandas as pd

    try:
        return int(len(pd.read_csv(path, usecols=[0])))
    except Exception:
        return None


def _label_partners(label_role: str, roles) -> list[str]:
    """Which modality roles a label role labels (same-cell pairing rule).

    * ``cty`` (one label set, paired data) -> every modality role;
    * ``cty<N>`` (one label file per batch) -> the roles numbered ``<N>``;
    * ``rna_cty`` / ``atac_cty`` / ``peak_cty`` -> the roles of that base
      modality (``atac_cty`` covers ``atac``, ``atac_gas`` and ``atac_peak``);
    * anything else (``source_cty`` ...) -> nothing (no safe pairing).
    """
    mods = [r for r in roles if not is_label_role(r) and r != "data_dir"]
    if label_role == "cty":
        return mods
    digits = "".join(ch for ch in label_role if ch.isdigit())
    if label_role.startswith("cty") and digits:
        return [r for r in mods if r.endswith(digits)]
    if label_role.endswith("_cty"):
        prefix = label_role[:-4]
        base = {"peak": "atac"}.get(prefix, prefix)
        return [r for r in mods if base_modality(r) == base and not r[-1:].isdigit()]
    return []


def _check_label_lengths(method, dataset, category, resolved):
    """Reject a label file whose row count differs from the cells it labels.

    ``evaluate`` would refuse the pair later ("emb has N cells, celltype has
    M") and the method itself may fail or, worse, silently mis-align. The
    pairing follows the role names (see :func:`_label_partners`); a file that
    cannot be parsed or a modality file without ``matrix/barcodes`` is left
    alone - no verdict is invented.
    """
    for role, path in resolved.items():
        if not is_label_role(role):
            continue
        p = Path(path)
        if p.suffix != ".csv" or not p.is_file():
            continue
        n_lab = _count_label_rows(str(p), p.stat().st_mtime_ns)
        if n_lab is None:
            continue
        for partner in _label_partners(role, resolved):
            q = Path(resolved[partner])
            if q.suffix != ".h5" or not q.is_file():
                continue
            sniff = _sniff_h5(str(q), q.stat().st_mtime_ns)
            if sniff is None:
                continue
            shape, n_feat, n_cell = sniff
            if n_feat == n_cell:          # orientation ambiguous: cannot tell cells
                continue
            if n_lab != n_cell:
                raise ValueError(
                    f"{method}/{dataset}/{category}: {p.name} has {n_lab} labels but "
                    f"{q.name} has {n_cell} cells (matrix/barcodes) - every cell needs "
                    f"exactly one label, in the same order as the cells "
                    f"(see mtb.describe_layout({category!r}))")


def _check_data_dir(variant, data_dir) -> tuple[bool, str]:
    """Does a ``data_dir`` really hold what the method needs? -> (ok, why).

    ``data_dir`` resolves to the dataset directory itself when there is no
    ``processed/`` subdir, so the path ALWAYS exists and existence proves
    nothing. Spatial-registration methods (``output.kind == 'coords'``) need
    >= 2 ``*.h5ad`` slices, each carrying ``obsm['spatial']`` coordinates (the
    upstream scripts glob ``data_dir + '*.h5ad'`` and align ``.obsm['spatial']``);
    other ``data_dir`` methods (scBridge) name their files via ``const`` args.
    """
    d = Path(data_dir)
    if not d.is_dir():
        return False, f"no such directory: {d}"
    if variant.output.kind == "coords":
        slices = sorted(d.glob("*.h5ad"))
        if len(slices) < 2:
            return False, ("spatial registration needs >=2 .h5ad slice files; "
                           f"found {len(slices)} in {d}")
        import h5py
        for sl in slices:
            try:
                with h5py.File(sl, "r") as f:
                    has = "obsm" in f and "spatial" in f["obsm"]
            except OSError:
                return False, f"{sl.name} is not a readable .h5ad file"
            if not has:
                return False, (f"{sl.name} has no obsm['spatial'] coordinates; "
                               f"registration needs .X plus obsm['spatial'] per slice")
        return True, ""
    # non-spatial data_dir methods (e.g. scBridge) name their files via `const`
    needed = [a.const for a in variant.args if a.const and str(a.const).endswith((".h5", ".csv"))]
    missing = [f for f in needed if not (d / f).exists()]
    if missing:
        return False, f"missing files in {d}: {missing}"
    return True, ""


#: Caveat text appended by scan() when an ``atac_gas`` role resolves to a peak matrix.
PEAK_IN_GAS_CAVEAT = "atac_gas resolved to a PEAK matrix (features look like chr:start-end)"


def _preflight_caveats(resolved) -> list[str]:
    """Non-fatal content observations about resolved inputs (never raises).

    Today one check: when the ``atac_gas`` role fell back to ``atac.h5`` (no
    ``atac_gas.h5`` present) and >= 90% of the first 50 feature names look like
    peaks (``chr1:1-200`` / ``chr1_1_200``), report :data:`PEAK_IN_GAS_CAVEAT`.
    Methods that want peaks behind that role name (``atac: peak`` in the
    registry) are fine; :func:`multibench.scan` applies the caveat only to the
    ones that want gene activity.
    """
    from .ingest import _PEAK_RE

    out: list[str] = []
    p = Path(resolved.get("atac_gas", ""))
    if p.name and p.stem != "atac_gas" and p.suffix == ".h5" and p.is_file():
        feats = _sniff_features(str(p), p.stat().st_mtime_ns)
        if feats:
            frac = sum(1 for x in feats if _PEAK_RE.match(x)) / len(feats)
            if frac >= 0.9:
                out.append(PEAK_IN_GAS_CAVEAT)
    return out


def labels_for(dataset: str, method: str | None = None, category: str | None = None,
               *, data_path: Path | str | None = None) -> dict:
    """Return ``{name: path}`` of the cell-type label CSVs for a dataset.

    The benchmark stores cell-type labels as ``*cty*.csv`` in the (flat) dataset
    dir, under dataset-specific names (``cty.csv``, ``rna_cty.csv``, ``cty1.csv``,
    ...). Returns the primary label files (excluding tool-specific ``*_scjoint*``
    reformats), keyed by filename stem. Pass the single path - or, for a
    multi-file dataset, ``list(labels_for(ds).values())`` in batch order - to
    ``mtb.evaluate(labels=...)``: a one-entry dict is accepted directly, a
    multi-entry dict raises with that hint. Raises ``FileNotFoundError`` if the
    dataset dir is absent.

    Parameters
    ----------
    dataset : dataset folder name under ``data_path``.
    method, category : accepted so the call mirrors
        ``inputs_for(dataset, method, category, ...)``; labels are per DATASET,
        so both are ignored (no per-method label selection exists).
    data_path : keyword-only. Root that CONTAINS the dataset folder; default
        ``config.DEFAULT.data_path``. For back-compat, a ``Path`` (or a string
        containing a path separator / naming an existing directory) passed as
        the 2nd positional argument is still treated as ``data_path`` - with a
        ``DeprecationWarning``; a bare, non-existent relative name there is taken
        as a method id and ignored.
    """
    if method is not None and (
            isinstance(method, Path)
            or (isinstance(method, str) and (os.sep in method or Path(method).is_dir()))):
        warnings.warn("labels_for: pass data_path= by keyword "
                      "(the 2nd positional argument is now `method`)",
                      DeprecationWarning, stacklevel=2)
        data_path, method = method, None
    base = Path(data_path) if data_path is not None else config.DEFAULT.data_path
    ds_dir = base / dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"no dataset dir at {ds_dir}")
    out = {}
    for p in sorted(ds_dir.glob("*cty*.csv")):
        if "scjoint" in p.name.lower():
            continue
        out[p.stem] = str(p)
    return out
