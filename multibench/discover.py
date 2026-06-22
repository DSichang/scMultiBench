"""Discovery: filter methods by need; enrich method_info with catalog metadata."""
from __future__ import annotations

from pathlib import Path

from .engine import registry, envs
from .engine.runner import _AUX_ROLES


def _base_modality(role: str) -> str:
    """Map an arg role to its base modality type (rna/adt/atac/...)."""
    # compound roles with an explicit modality prefix
    for prefix in ("rna", "adt", "atac"):
        if role.startswith(prefix + "_"):
            return prefix
    # strip a single trailing digit (rna1 -> rna, atac2 -> atac)
    if role and role[-1].isdigit():
        return role[:-1]
    return role


def _modality_types(spec) -> set[str]:
    """Union of base modality types consumed across all of a method's variants.

    Auxiliary roles (data_dir, source/target data & cty, out_dir) are excluded.
    Methods with no variants (declared stubs) yield an empty set.
    """
    types: set[str] = set()
    for v in spec.variants:
        for a in v.args:
            if a.role in _AUX_ROLES:
                continue
            types.add(_base_modality(a.role))
    return types


def find_methods(category: str | None = None, task: str | None = None,
                 needs_labels: bool | None = None,
                 atac: str | None = None,
                 modalities: list[str] | set[str] | None = None,
                 runnable: bool | None = None) -> list[str]:
    """Return method ids matching all supplied filters.

    ``atac`` is an exact match on the method's declared ATAC representation;
    valid values are ``"peak"`` or ``"gene_activity"`` (not a boolean flag).
    ``modalities`` keeps methods that consume ALL of the requested base modality
    types (e.g. ``["rna", "atac"]``); because modality info is derived from a
    method's variants, this filter implicitly excludes the declared-but-unwired
    stub methods (those without variants). ``runnable=True`` restricts to methods
    with at least one variant (usable by ``inputs_for``/``run``); ``runnable=False``
    returns only the stubs.
    """
    want = set(modalities) if modalities is not None else None
    out = []
    for s in registry.load():
        if category and category not in s.categories:
            continue
        if task and task not in s.tasks:
            continue
        if needs_labels is not None and s.needs_labels != needs_labels:
            continue
        if atac and s.atac != atac:
            continue
        if want is not None and not want <= _modality_types(s):
            continue
        if runnable is not None and bool(s.variants) != runnable:
            continue
        out.append(s.id)
    return out


def method_info(method: str, files_dir: Path | str | None = None) -> dict:
    """Return a flat dict combining registry spec + (optional) catalog row."""
    s = registry.get(method)
    info = {
        "id": s.id, "language": s.language, "categories": s.categories,
        # `env` is the conda env run() actually executes in: the shared group
        # env, or the method's own scmb_<method> env (see engine.envs.group_for).
        "tasks": s.tasks, "env": envs.group_for(s.id), "atac": s.atac,
        "needs_labels": s.needs_labels, "status": s.status,
        "setup_hint": s.setup_hint,
        "variants": [v.entrypoint for v in s.variants],
    }
    if files_dir is not None:
        from .data import catalog
        cat = catalog.methods(files_dir)
        match = cat[cat["canonical_id"] == s.id]
        if len(match):
            row = match.iloc[0]
            info["deep_learning"] = row["deep_learning"]
            info["output"] = row["output"]
    return info
