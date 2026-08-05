"""Resolve a catalog dataset id + method to concrete input file paths."""
from __future__ import annotations

from pathlib import Path

from .. import config
from . import registry

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
               check: bool = False) -> dict:
    """Return ``{modality_role: path}`` for a method's variant on a dataset.

    The dataset tree is **flat** (``<data_path>/<dataset>/<file>``). Each role is
    resolved to the actual file present in that dir (the role token, or a known
    alias such as ``atac_peak``->``peak.h5`` / ``atac_gas``->``atac.h5``),
    falling back to ``<role>.h5`` when no candidate exists. Pass ``check=True`` to
    raise ``FileNotFoundError`` if any resolved path is missing (instead of
    returning a best-effort path that only fails later inside ``run``). Use
    :func:`labels_for` to get the matching cell-type label CSVs.

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
             if r and r not in ("out_dir", "data_dir")]
    out = {role: str(_resolve_role(ds_dir, role)) for role in roles}
    # A `data_dir` role points at the DIRECTORY of spatial slices (registration).
    if any(a.role == "data_dir" for a in variant.args):
        out["data_dir"] = _resolve_data_dir(ds_dir, method)
    if check:
        missing = {r: p for r, p in out.items() if not Path(p).exists()}
        if missing:
            raise FileNotFoundError(
                f"{method}/{dataset}/{category}: input files not found on disk: "
                f"{missing}. Available files in {ds_dir}: "
                f"{sorted(q.name for q in ds_dir.glob('*')) if ds_dir.is_dir() else '(dir missing)'}"
            )
        _check_orientation(method, dataset, category, out)
    return out



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
    import h5py

    for path in resolved.values():
        p = Path(path)
        if p.suffix != ".h5" or not p.is_file():
            continue
        try:
            with h5py.File(p, "r") as f:
                if "matrix/data" not in f:
                    continue
                shape = tuple(f["matrix/data"].shape)
                if len(shape) != 2:
                    continue
                if "matrix/features" not in f or "matrix/barcodes" not in f:
                    continue          # nothing to compare against
                n_feat = int(f["matrix/features"].shape[0])
                n_cell = int(f["matrix/barcodes"].shape[0])
        except OSError:
            continue                  # an unreadable file is a different error
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


def labels_for(dataset: str, data_path: Path | str | None = None) -> dict:
    """Return ``{name: path}`` of the cell-type label CSVs for a dataset.

    The benchmark stores cell-type labels as ``*cty*.csv`` in the (flat) dataset
    dir, under dataset-specific names (``cty.csv``, ``rna_cty.csv``, ``cty1.csv``,
    ...). Returns the primary label files (excluding tool-specific ``*_scjoint*``
    reformats), keyed by filename stem, for use as ``mtb.evaluate(labels=...)``.
    Raises ``FileNotFoundError`` if the dataset dir is absent.
    """
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
