"""Resolve a catalog dataset id + method to concrete input file paths."""
from __future__ import annotations

from pathlib import Path

from .. import config
from . import registry


def inputs_for(dataset: str, method: str, category: str,
               modalities: list[str] | set[str] | None = None,
               data_path: Path | str | None = None) -> dict:
    """Return {modality_role: path} for a method's variant on a dataset.

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
    ds_dir = base / config.category_folder(category) / dataset
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
    return {role: str(ds_dir / f"{role}.h5") for role in roles}
