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


def _resolve_data_dir(ds_dir: Path) -> str:
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
            return os.path.join(str(cand), "")
    return os.path.join(str(ds_dir / "processed"), "")


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
    roles = [a.role for a in variant.args
             if a.role not in ("out_dir", "data_dir") and getattr(a, "const", None) is None]
    out = {role: str(_resolve_role(ds_dir, role)) for role in roles}
    # A `data_dir` role points at the DIRECTORY of spatial slices (registration).
    if any(a.role == "data_dir" for a in variant.args):
        out["data_dir"] = _resolve_data_dir(ds_dir)
    if check:
        missing = {r: p for r, p in out.items() if not Path(p).exists()}
        if missing:
            raise FileNotFoundError(
                f"{method}/{dataset}/{category}: input files not found on disk: "
                f"{missing}. Available files in {ds_dir}: "
                f"{sorted(q.name for q in ds_dir.glob('*')) if ds_dir.is_dir() else '(dir missing)'}"
            )
    return out


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
