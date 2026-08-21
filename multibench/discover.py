"""Discovery: filter methods by need; enrich method_info with catalog metadata; cite."""
from __future__ import annotations

from pathlib import Path

from .engine import registry, upstream, envs
from .engine.runner import _AUX_ROLES  # noqa: F401  (re-exported for back-compat)
from .engine.schema import base_modality as _base_modality  # noqa: F401  (tests import it from here)


def _modality_types(spec) -> set[str]:
    """Union of base modality types consumed across all of a method's variants.

    Auxiliary roles (data_dir, source/target data & cty, out_dir) and label roles
    are excluded. Methods with no variants (declared stubs) yield an empty set.
    (Thin alias of ``MethodSpec.modality_types``, kept for callers/tests.)
    """
    return spec.modality_types


def find_methods(category: str | None = None, task: str | None = None,
                 needs_labels: bool | None = None,
                 atac: str | None = None,
                 modalities: list[str] | set[str] | None = None,
                 runnable: bool | None = None,
                 tunable: bool | None = None) -> list[str]:
    """Return method ids matching all supplied filters.

    Every token is validated: a typo raises ``ValueError`` naming the valid
    vocabulary instead of silently matching nothing.

    ``category`` is one of ``vertical``/``diagonal``/``mosaic``/``cross``;
    ``task`` one of :func:`multibench.list_tasks`.
    ``needs_labels`` is derived from the variants' roles: ``True`` means at least
    one variant takes a cell-type-label (``cty``) role as a REQUIRED input - see
    ``method_info(m)['supports'][i]['needs_labels']`` for the per-variant answer.
    ``atac`` is an exact match on the method's declared ATAC representation:
    ``"peak"`` or ``"gene_activity"`` (``"peaks"``/``"gas"`` accepted as aliases);
    it is the representation the UPSTREAM script expects, which is not always
    what its role name suggests (moETM/scMM/iPOLNG take role ``atac_gas`` but
    consume peaks).
    ``modalities`` keeps methods that consume ALL of the requested base modality
    types (e.g. ``["rna", "atac"]``); ``"protein"`` is accepted for ``adt`` and
    role tokens (``atac_gas``, ``atac_peak``, ``rna1`` ...) are reduced to their
    base type, so ``["rna", "atac_gas"]`` means ``["rna", "atac"]``. Because
    modality info is derived from a method's variants, this filter implicitly
    excludes the declared-but-unwired stub methods (those without variants).
    ``runnable=True`` restricts to methods with at least one variant (usable by
    ``inputs_for``/``run``); ``runnable=False`` returns only the stubs.
    ``tunable=True`` keeps only methods that expose at least one hyperparameter
    on their command line (i.e. where ``run(params=...)`` can change anything) -
    the rest hardcode their settings upstream.
    """
    registry.check_category(category)
    registry.check_task(task)
    atac = registry.check_atac(atac)
    want = (set(registry.normalize_modalities(modalities, base=True))
            if modalities is not None else None)
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
        if want is not None and not want <= s.modality_types:
            continue
        if runnable is not None and bool(s.variants) != runnable:
            continue
        if tunable is not None:
            has = any(v.tunable for v in s.variants)
            if has != tunable:
                continue
        out.append(s.id)
    return out


def _effective(v) -> dict:
    """Upstream argparse defaults overlaid with what the wrapper actually emits.

    ``tunable[k]['default']`` stays the UPSTREAM default (documented contract);
    ``defaults`` is what the package passes on the command line; ``effective`` is
    the merge - the value the script will really run with when the caller
    passes no ``params``.
    """
    eff = {k: (t or {}).get("default") for k, t in v.tunable.items()}
    eff.update(dict(v.params))
    return eff


def method_info(method: str, files_dir: Path | str | None = None, *,
                verbose: bool = False) -> dict:
    """Return a flat dict combining registry spec + provenance + (optional) catalog row.

    Parameters
    ----------
    method : registry id (``KeyError`` with a did-you-mean hint otherwise).
    files_dir : optional dir holding the benchmark's ``method.csv`` catalog; when
        given, the paper-only columns ``deep_learning`` and ``output`` are added.
    verbose : when True also return ``notes_long`` - the raw upstream-knob audit
        prose for this method (None for methods outside the audit). The default
        ``notes`` is the short third-person summary from engine/references.yaml.

    Keys
    ----
    id, language, categories, tasks, env, atac, needs_labels (derived: any
    variant takes a cty role), status, setup_hint,
    variants (distinct upstream entrypoints, in order), driver (package-side
    wrapper actually executed, or None when the upstream script runs directly),
    scripts_url (the benchmark's tools_scripts folder), repo_url, version,
    reference ({doi, title, authors, journal, year} or None), notes,
    supports (per variant: category, modalities, output_kind, n_tunable,
    needs_labels), params (per variant key 'category:mods': defaults, tunable,
    effective), fixed_in_script, upstream_knobs, upstream_url.
    """
    s = registry.get(method)
    ref = s.reference or {}
    info = {
        "id": s.id, "language": s.language, "categories": s.categories,
        # `env` is the conda env run() actually executes in: the shared group
        # env, or the method's own scmb_<method> env (see engine.envs.group_for).
        "tasks": s.tasks, "env": envs.group_for(s.id), "atac": s.atac,
        "needs_labels": s.needs_labels, "status": s.status,
        "setup_hint": s.setup_hint,
        # distinct entrypoints, in declaration order (several variants may share
        # one script)
        "variants": list(dict.fromkeys(v.entrypoint for v in s.variants)),
        # package-relative wrapper the runner executes INSTEAD of the entrypoint
        # (it source()s/imports the unmodified upstream script); None = the
        # upstream script itself is run
        "driver": next((v.driver for v in s.variants if v.driver), None),
        # Where this method's UNMODIFIED upstream scripts live - the folder
        # carries the method's own imports/reference; the practical citation
        # pointer the README promises.
        "scripts_url": (
            "https://github.com/PYangLab/scMultiBench/tree/main/"
            + "/".join(s.variants[0].entrypoint.split("/")[:2])
            if s.variants and s.variants[0].entrypoint.startswith("tools_scripts/")
            else None),
        # provenance (engine/references.yaml): the upstream repository / docs, the
        # version the benchmark ran, and the paper to cite (see mtb.cite)
        "repo_url": ref.get("repo_url") or upstream.knobs_for(s.id)["upstream_url"],
        "version": ref.get("version"),
        "reference": dict(ref["reference"]) if ref.get("reference") else None,
        # What this method can actually be dispatched for. Methods like Multigrate
        # support several integration categories, each with its OWN modality
        # combination - this is the list to pass to run()/inputs_for().
        "supports": [{"category": v.when.get("category"),
                      "modalities": list(v.when.get("modalities", [])),
                      "output_kind": v.output.kind,
                      "n_tunable": len(v.tunable),
                      "needs_labels": v.needs_labels}
                     for v in s.variants],
        # what the caller may pass to run(params=...): see params_for()
        "params": {_variant_key(v): {"defaults": dict(v.params),
                                     "tunable": dict(v.tunable),
                                     "effective": _effective(v)}
                   for v in s.variants},
    }
    # An empty `tunable` says the SCRIPT exposes nothing, not that the method
    # has no hyperparameters - so ship what it pins and what its library
    # documents, or the honest answer reads as a false one.
    up = upstream.knobs_for(s.id)
    info["fixed_in_script"] = up["fixed_in_script"]
    info["upstream_knobs"] = up["upstream_knobs"]
    info["upstream_url"] = up["upstream_url"]
    # `notes` is the curated one-line summary; the audit's long first-person
    # prose is available on request as notes_long.
    info["notes"] = ref.get("summary") or None
    if verbose:
        info["notes_long"] = up["notes"]
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

    Returns ``{"method", "variant", "defaults", "tunable", "effective", ...}`` where:

    * ``defaults`` — parameters the package emits on every run. Override them
      with ``run(..., params={...})``; the override is merged over these.
    * ``tunable`` — documentation of the parameters the UPSTREAM script accepts
      on its command line, as ``{name: {"default": ..., "type": ...}}``. The
      ``default`` here is the upstream argparse default, NOT necessarily what a
      wrapper run uses.
    * ``effective`` — ``tunable`` defaults overlaid with ``defaults``: the value
      each knob really takes when you pass no ``params``.

    An **empty** ``tunable`` means the upstream script exposes no hyperparameters
    on its command line. Because this project never modifies method scripts,
    such a method cannot be tuned through the wrapper - but it is not
    parameterless, so two further keys say what it actually does:

    * ``fixed_in_script`` — the values the script pins, each with the
      ``file:line`` that pins it.
    * ``upstream_knobs`` — what the wrapped library documents (with its own
      defaults), unreachable without editing the script.

    Both are empty for methods outside the upstream audit.

    ``category``/``modalities`` select the variant, exactly like ``run``
    (``category`` is validated; ``modalities`` accepts ``protein`` for ``adt``).
    They may be omitted when the method has only one variant.
    """
    s = registry.get(method)
    registry.check_category(category)
    modalities = registry.normalize_modalities(modalities)
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
            "effective": _effective(v),
            "fixed_in_script": up["fixed_in_script"],
            "upstream_knobs": up["upstream_knobs"],
            "upstream_url": up["upstream_url"]}


# ------------------------------------------------------------------ citations
_CITE_FORMATS = ("bibtex", "text")


def _bibtex_key(tag: str, year) -> str:
    return f"{tag.replace(' ', '_')}_{year}" if year else tag.replace(" ", "_")


def _format_entry(tag: str, ref: dict, fmt: str) -> str:
    doi = ref.get("doi")
    authors = ref.get("authors") or ""
    title = ref.get("title") or ""
    journal = ref.get("journal") or ""
    year = ref.get("year") or ""
    if fmt == "bibtex":
        fields = [("author", authors.replace(", ", " and ") if authors else ""),
                  ("title", title), ("journal", journal), ("year", year)]
        if ref.get("volume"):
            fields.append(("volume", ref["volume"]))
        if ref.get("pages"):
            fields.append(("pages", ref["pages"]))
        fields.append(("doi", doi))
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v not in (None, ""))
        return f"@article{{{_bibtex_key(tag, year)},\n{body}\n}}"
    line = f"{authors}. {title}. {journal} ({year})."
    if doi:
        line += f" https://doi.org/{doi}"
    return line


def cite(methods: list[str] | str | None = None, fmt: str = "bibtex") -> str:
    """Citation text for the benchmark and (optionally) the methods you ran.

    Parameters
    ----------
    methods : None (default) -> the benchmark entry only; ``"all"`` -> every
        registry method; or a list of method ids (``KeyError`` with a
        did-you-mean hint for an unknown id). Methods whose DOI is not yet
        curated in engine/references.yaml are emitted as a ``% <id>: no verified
        reference; see <repo_url>`` comment (bibtex) / ``<id>: ... <repo_url>``
        line (text) rather than silently dropped.
    fmt : ``"bibtex"`` (one ``@article`` per entry) or ``"text"`` (one
        "Authors. Title. Journal (year). https://doi.org/..." line per entry).

    Returns one string: the benchmark entry first, then one entry per method in
    the order given. Every DOI in the table was resolved against Crossref.
    """
    if fmt not in _CITE_FORMATS:
        raise ValueError(f"unknown fmt {fmt!r}; valid: {list(_CITE_FORMATS)}")
    if methods is None:
        ids: list[str] = []
    elif isinstance(methods, str):
        ids = registry.list_methods() if methods == "all" else [registry.check_method(methods)]
    else:
        ids = [registry.check_method(m) for m in methods]
    parts = [_format_entry("scMultiBench", registry.benchmark_reference(), fmt)]
    for m in ids:
        s = registry.get(m)
        ref = (s.reference or {}).get("reference")
        if ref:
            parts.append(_format_entry(m, ref, fmt))
        else:
            url = (s.reference or {}).get("repo_url") or "(no repo_url)"
            note = f"{m}: no verified reference in engine/references.yaml; see {url}"
            parts.append(("% " + note) if fmt == "bibtex" else note)
    return "\n\n".join(parts) if fmt == "bibtex" else "\n".join(parts)
