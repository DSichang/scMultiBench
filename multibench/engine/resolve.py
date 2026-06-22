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

    Tries the role token and known aliases, each as ``<base>.h5`` then
    ``<base>1.h5`` (some datasets store batched files), returning the first that
    exists; falls back to the canonical ``<base>.h5`` when nothing matches.
    """
    bases = _ROLE_FILE_CANDIDATES.get(role, (role,))
    for base in bases:
        for fname in (f"{base}.h5", f"{base}1.h5"):
            p = ds_dir / fname
            if p.exists():
                return p
    return ds_dir / f"{bases[0]}.h5"


def inputs_for(dataset: str, method: str, category: str,
               modalities: list[str] | set[str] | None = None,
               data_path: Path | str | None = None) -> dict:
    """Return ``{modality_role: path}`` for a method's variant on a dataset.

    The dataset tree is **flat** (``<data_path>/<dataset>/<file>``). Each role is
    resolved to the actual file present in that dir (the role token, or a known
    alias such as ``atac_peak``->``peak.h5`` / ``atac_gas``->``atac.h5``),
    falling back to ``<role>.h5`` when no candidate exists. Use
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

    roles = [a.role for a in variant.args if a.role != "out_dir"]
    return {role: str(_resolve_role(ds_dir, role)) for role in roles}


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
