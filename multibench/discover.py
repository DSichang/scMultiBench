"""Discovery: filter methods by need; enrich method_info with catalog metadata."""
from __future__ import annotations

from pathlib import Path

from .engine import registry, upstream, envs
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
            for r in (a.roles or [a.role]):
                if not r or r in _AUX_ROLES:
                    continue
                types.add(_base_modality(r))
    return types


def find_methods(category: str | None = None, task: str | None = None,
                 needs_labels: bool | None = None,
                 atac: str | None = None,
                 modalities: list[str] | set[str] | None = None,
                 runnable: bool | None = None,
                 tunable: bool | None = None) -> list[str]:
    """Return method ids matching all supplied filters.

    ``atac`` is an exact match on the method's declared ATAC representation;
    valid values are ``"peak"`` or ``"gene_activity"`` (not a boolean flag).
    ``modalities`` keeps methods that consume ALL of the requested base modality
    types (e.g. ``["rna", "atac"]``); because modality info is derived from a
    method's variants, this filter implicitly excludes the declared-but-unwired
    stub methods (those without variants). ``runnable=True`` restricts to methods
    with at least one variant (usable by ``inputs_for``/``run``); ``runnable=False``
    returns only the stubs. ``tunable=True`` keeps only methods that expose at least
    one hyperparameter on their command line (i.e. where ``run(params=...)`` can
    change anything) - the rest hardcode their settings upstream.
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
        if tunable is not None:
            has = any(v.tunable for v in s.variants)
            if has != tunable:
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
        # Where this method's UNMODIFIED upstream scripts live - the folder
        # carries the method's own imports/reference; the practical citation
        # pointer the README promises.
        "scripts_url": (
            "https://github.com/PYangLab/scMultiBench/tree/main/"
            + "/".join(s.variants[0].entrypoint.split("/")[:2])
            if s.variants and s.variants[0].entrypoint.startswith("tools_scripts/")
            else None),
        # What this method can actually be dispatched for. Methods like Multigrate
        # support several integration categories, each with its OWN modality
        # combination - this is the list to pass to run()/inputs_for().
        "supports": [{"category": v.when.get("category"),
                      "modalities": list(v.when.get("modalities", [])),
                      "output_kind": v.output.kind,
                      "n_tunable": len(v.tunable)}
                     for v in s.variants],
        # what the caller may pass to run(params=...): see params_for()
        "params": {_variant_key(v): {"defaults": dict(v.params),
                                     "tunable": dict(v.tunable)}
                   for v in s.variants},
    }
    # An empty `tunable` says the SCRIPT exposes nothing, not that the method
    # has no hyperparameters - so ship what it pins and what its library
    # documents, or the honest answer reads as a false one.
    info.update(upstream.knobs_for(s.id))
    if files_dir is not None:
        from .data import catalog
        cat = catalog.methods(files_dir)
        match = cat[cat["canonical_id"] == s.id]
        if len(match):
            row = match.iloc[0]
            info["deep_learning"] = row["deep_learning"]
            info["output"] = row["output"]
    return info


def _variant_key(v) -> str:
    """Stable human-readable key for a variant: 'category:mod1+mod2'."""
    mods = "+".join(v.when.get("modalities", [])) or "-"
    return f"{v.when.get('category')}:{mods}"


def params_for(method: str, category: str | None = None,
               modalities: list[str] | set[str] | None = None) -> dict:
    """Return the hyperparameters of one method variant.

    Returns ``{"method", "variant", "defaults", "tunable"}`` where:

    * ``defaults`` — parameters the package emits on every run. Override them
      with ``run(..., params={...})``; the override is merged over these.
    * ``tunable`` — documentation of the parameters the UPSTREAM script accepts
      on its command line, as ``{name: {"default": ..., "type": ...}}``.

    An **empty** ``tunable`` means the upstream script exposes no hyperparameters
    on its command line. Because this project never modifies method scripts,
    such a method cannot be tuned through the wrapper - but it is not
    parameterless, so two further keys say what it actually does:

    * ``fixed_in_script`` — the values the script pins, each with the
      ``file:line`` that pins it.
    * ``upstream_knobs`` — what the wrapped library documents (with its own
      defaults), unreachable without editing the script.

    Both are empty for methods outside the upstream audit.

    ``category``/``modalities`` select the variant, exactly like ``run``. They may
    be omitted when the method has only one variant.
    """
    s = registry.get(method)
    if not s.variants:
        raise KeyError(f"{method}: no variants (declared stub); nothing to tune")
    if category is None and modalities is None:
        if len(s.variants) > 1:
            raise KeyError(
                f"{method} has {len(s.variants)} variants - pass category (and modalities "
                f"if that is still ambiguous); available: {[_variant_key(v) for v in s.variants]}")
        v = s.variants[0]
    elif modalities is None:
        # category alone is enough whenever it selects exactly one variant. This is
        # the ONLY way to reach a data_dir variant (scBridge, the spatial methods),
        # which has no modalities to pass.
        cands = [x for x in s.variants if x.when.get("category") == category]
        if not cands:
            raise KeyError(
                f"{method}: no {category!r} variant; available: "
                f"{[_variant_key(x) for x in s.variants]}")
        if len(cands) > 1:
            raise KeyError(
                f"{method} has {len(cands)} {category!r} variants - also pass modalities; "
                f"available: {[_variant_key(x) for x in cands]}")
        v = cands[0]
    elif category is None:
        cands = [x for x in s.variants
                 if set(x.when.get("modalities", [])) == set(modalities)]
        if len(cands) != 1:
            raise KeyError(
                f"{method}: modalities {sorted(modalities)} match {len(cands)} variants - "
                f"also pass category; available: {[_variant_key(x) for x in s.variants]}")
        v = cands[0]
    else:
        v = s.select(category, set(modalities))
    up = upstream.knobs_for(s.id)
    return {"method": s.id, "variant": _variant_key(v),
            "defaults": dict(v.params), "tunable": dict(v.tunable),
            "fixed_in_script": up["fixed_in_script"],
            "upstream_knobs": up["upstream_knobs"],
            "upstream_url": up["upstream_url"]}
