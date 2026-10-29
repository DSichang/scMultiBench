"""Discovery: filter methods by need; enrich method_info with catalog metadata."""
from __future__ import annotations

import functools
from pathlib import Path

from ._text import _and_list
from .engine import registry, upstream, envs
from .engine.schema import AmbiguousVariantError, is_label_role
from .engine.schema import base_modality as _base_modality

#: representation role tokens -> the ``atac=`` value they stand for
_REPRESENTATION_TOKENS = {"atac_peak": "peak", "atac_gas": "gene_activity"}


def _representations(toks) -> set[str]:
    """The ATAC representations (``peak``, ``gene_activity``) the normalised tokens name."""
    return {_REPRESENTATION_TOKENS[t] for t in toks if t in _REPRESENTATION_TOKENS}


#: naming both ATAC roles selects the variants that read both files (MultiMAP, Seurat_v3)
_BOTH_FILES = frozenset(_REPRESENTATION_TOKENS)


def _modality_filter(modalities) -> tuple[set[str] | None, str | None]:
    """``(base types every matching variant must consume, ATAC representation)``:
    the modality rule of ``find_methods`` and ``rank`` (see the
    ``find_methods`` Notes)."""
    registry.check_id_list(modalities, "modalities")
    if modalities is None:
        return None, None
    toks = registry.normalize_modalities(modalities)
    reps = sorted(_representations(toks))
    return {_base_modality(t) for t in toks}, (reps[0] if len(reps) == 1 else None)


def _reads_both_files(modalities) -> bool:
    """Whether ``modalities`` names both ATAC representations, in any spelling."""
    if modalities is None:
        return False
    return _BOTH_FILES <= set(registry.normalize_modalities(modalities))


def _variant_matches(v, category, want, needs_labels) -> bool:
    """Whether variant ``v`` satisfies every per-variant filter."""
    if category and v.when.get("category") != category:
        return False
    if want is not None and not want <= v.modality_types:
        return False
    if needs_labels is not None and v.needs_labels != needs_labels:
        return False
    return True


def find_methods(category: str | None = None, *,
                 modalities: list[str] | set[str] | None = None,
                 needs_labels: bool | None = None) -> list[str]:
    """Return the method ids that match every filter you pass.

    A method matches when one of its variants meets ``category``,
    ``modalities`` and ``needs_labels`` together.

    Parameters
    ----------
    category : str | None
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``; ``None`` = any.
    modalities : list[str] | set[str] | None
        Modalities a variant must consume, e.g. ``["rna", "adt"]``;
        ``"atac_peak"`` / ``"atac_gas"`` also select by ATAC representation.
        ``None`` = no filter.
    needs_labels : bool | None
        ``True`` = a matching variant needs cell-type labels; ``False`` = one
        runs without them; ``None`` = no filter.

    Returns
    -------
    list[str]
        Method ids in the package's order.

    Raises
    ------
    ValueError
        Unknown ``category`` or modality token.
    TypeError
        ``modalities`` is a bare string, not a list.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.find_methods("vertical", modalities=["rna", "adt"])
    >>> mtb.find_methods("vertical", modalities=["rna", "adt"], needs_labels=False)

    See Also
    --------
    mtb.method_info : the per-variant ``supports`` table these filters are read from.
    mtb.scan : which of these methods can actually run on a given dataset.
    """
    registry.check_category(category)
    want, atac = _modality_filter(modalities)
    both = _reads_both_files(modalities)
    out = []
    for s in registry.load():
        if category and category not in s.categories:
            continue
        # atac is declared per method, but only a variant with an ATAC input honours it
        if atac and s.atac != atac:
            continue
        if any(_variant_matches(v, category, want, needs_labels)
               and (not atac or v.consumes_atac)
               and (not both or _BOTH_FILES <= {
                   r.rstrip("0123456789") for r in v.when.get("modalities") or []})
               for v in s.variants):
            out.append(s.id)
    return out


def _effective(v) -> dict:
    """The value of every tunable key in a run without ``params``: the script's
    default, or what the package passes."""
    return {k: v.params.get(k, (t or {}).get("default")) for k, t in v.tunable.items()}


def _tunable(v) -> dict:
    """``{name: {"type", "default"}}``: ``default`` is what a run uses."""
    eff = _effective(v)
    return {k: {**(t or {}), "default": eff[k]} for k, t in v.tunable.items()}


def _reference_batch(variant):
    """Batch ``N`` of a variant's ``reference`` const ``data<N>``, else ``None``."""
    ref = next((a.const for a in variant.args if a.role == "reference" and a.const), None)
    ref = str(ref) if ref is not None else ""
    return int(ref[4:]) if ref.startswith("data") and ref[4:].isdigit() else None


def method_info(method: str) -> dict:
    """Return everything known about one method as a dict.

    Parameters
    ----------
    method : str
        Method id, e.g. ``"Matilda"``; see ``mtb.find_methods()``.

    Returns
    -------
    dict
        One flat record:

        ```text
        id, language, notes     the method id, 'python' or 'R', a summary
        categories              the integration categories it runs in
        env                     the conda env run executes it in
        atac                    'peak' / 'gene_activity' / None
        needs_labels            any variant reads cell-type labels
        supports                one entry per variant: category, modalities,
                                output_kind, needs_labels, labels,
                                reference_batch
        runtime                 measured_sec, tier, worst_sec, observed, host, note
        gpu                     'required' / 'used when present' / 'not used' / 'unknown'
        cpu_params              the flags that turn CUDA off on a CPU host
        setup_hint              a step to take before the first run, or ''
        repo_url                the upstream code the benchmark ran
        reference               {doi, title, authors, journal, year}, or None
        ```

    Raises
    ------
    KeyError
        Unknown method id.

    Examples
    --------
    >>> import multibench as mtb
    >>> info = mtb.method_info("Matilda")
    >>> # one entry per variant: category, modalities, labels ...
    >>> info["supports"]

    See Also
    --------
    mtb.params_for : the hyperparameters of one variant, with upstream defaults.
    mtb.find_methods : filter methods by category, modalities, labels, ATAC representation.
    """
    s = registry.get(method)
    ref = s.reference
    return dict(
        id=s.id, language=s.language, notes=ref.get("summary") or None,
        categories=s.categories, env=envs.group_for(s.id), atac=s.atac,
        needs_labels=s.needs_labels,
        supports=[{"category": v.when.get("category"),
                   "modalities": v.shown_modalities,
                   "output_kind": v.output.kind,
                   "needs_labels": v.needs_labels,
                   "labels": [r for r in v.roles() if is_label_role(r)],
                   "reference_batch": _reference_batch(v)}
                  for v in s.variants],
        runtime=_runtime_hint(s.id), gpu=s.gpu, cpu_params=dict(s.cpu_params),
        setup_hint=s.setup_hint,
        repo_url=ref.get("repo_url") or upstream.knobs_for(s.id)["upstream_url"],
        reference=dict(ref["reference"]) if ref.get("reference") else None)


# ------------------------------------------------------------ observed runtimes
_RUNTIMES_YAML = Path(__file__).resolve().parent / "engine" / "runtimes.yaml"


@functools.lru_cache(maxsize=1)
def _runtimes() -> dict:
    """Observed per-method runtimes (engine/runtimes.yaml)."""
    import yaml
    with open(_RUNTIMES_YAML) as fh:
        return yaml.safe_load(fh) or {}


def _runtime_hint(method: str) -> dict:
    """``method_info(method)["runtime"]``."""
    rt = dict(_runtimes().get(method, {"tier": "unknown", "worst_sec": None,
                                       "observed": []}))
    # every observation was taken on the GPU benchmark host
    rt["host"] = "gpu"
    rt["note"] = "times observed on the benchmark host (NVIDIA RTX 4090)"
    return rt


def _variant_key(v) -> str:
    """``'category:mod1+mod2'``, with the files a folder-reading variant reads."""
    return registry._variant_key({**v.when, "modalities": v.shown_modalities})


def params_for(method: str, category: str | None = None,
               modalities: list[str] | set[str] | None = None) -> dict:
    """Return the hyperparameters of one method variant.

    Call it before ``run(params=...)`` to see what a method accepts and
    what it runs with when you pass nothing.

    Parameters
    ----------
    method : str
        Method id, e.g. ``"Matilda"``; see ``mtb.find_methods()``.
    category : str | None
        Integration category of the variant: ``vertical``, ``diagonal``,
        ``mosaic`` or ``cross``; ``None`` = infer it from the other arguments.
    modalities : list[str] | set[str] | None
        Modality tokens of the variant, e.g. ``["rna", "adt"]``; ``None`` =
        infer it from the other arguments.

    Returns
    -------
    dict
        ``method``, ``variant`` (``'category:mods'``, or several that take
        the same parameters), ``tunable`` (what
        ``params=`` can set, ``{name: {"type", "default"}}``; ``default`` is
        what a run uses when you pass nothing), ``fixed_in_script`` (what the
        script pins, with ``file:line``), ``upstream_knobs`` (what its library
        documents but the script does not expose) and ``upstream_url``.

    Raises
    ------
    KeyError
        Unknown method, or no variant for ``category`` /
        ``modalities``.
    ValueError
        Unknown ``category`` or modality token, or several variants with other
        parameters fit (the message spells out the call that selects one).

    Examples
    --------
    >>> import multibench as mtb
    >>> p = mtb.params_for("Matilda", "vertical", ["rna", "adt"])
    >>> p["tunable"]["epochs"]["default"]
    30

    See Also
    --------
    mtb.run : takes these keys as ``params={method: {key: value}}``.
    """
    s = registry.get(method)
    method = s.id
    registry.check_category(category)
    modalities = registry.normalize_modalities(modalities)

    def example(v):
        mods = list(v.when.get("modalities", []))
        return f"params_for({method!r}, {v.when.get('category')!r}, {mods})"

    def mods_of(vs):
        return _and_list("+".join(x.when.get("modalities", [])) for x in vs)

    if category is None and modalities is None:
        v = _one(s.variants,
                 f"{method} has {len(s.variants)} variants, "
                 f"{_and_list(_variant_key(x) for x in s.variants)}. Pass the "
                 f"category and modalities of one, for example {example(s.variants[0])}.")
    elif modalities is None:
        # the only way to reach a data_dir variant (scBridge), which has no modalities
        cands = [x for x in s.variants if x.when.get("category") == category]
        if not cands:
            from .engine.schema import no_category_message
            raise KeyError(no_category_message(method, s.wired_categories, category))
        v = _one(cands, f"{method} has {len(cands)} {category} variants, {mods_of(cands)}. "
                        f"Pass the modalities too, for example {example(cands[0])}.")
    elif category is None:
        from .engine.schema import modality_family
        want = {modality_family(m) for m in modalities}
        cands = ([x for x in s.variants if set(x.when.get("modalities", [])) == set(modalities)]
                 or [x for x in s.variants
                     if {modality_family(m) for m in x.when.get("modalities", [])} == want]
                 or _same_types(s.variants, modalities))
        if not cands:
            keys = _and_list(_variant_key(x) for x in s.variants)
            raise KeyError(
                f"No variant of {method} takes the modalities "
                f"{_and_list(sorted(modalities))}. Its variants: {keys}.")
        v = _one(cands, f"The modalities {_and_list(sorted(modalities))} match "
                        f"{len(cands)} variants of {method}, "
                        f"{_and_list(_variant_key(x) for x in cands)}. Pass the category "
                        f"too, for example {example(cands[0])}.")
    else:
        try:
            v = s.select(category, set(modalities), loose=True)
        except AmbiguousVariantError:
            raise
        except KeyError:
            # base tokens, as find_methods takes them: rna + atac for rna1..3 + atac1..3
            cands = _same_types([x for x in s.variants
                                 if x.when.get("category") == category], modalities)
            if not cands:
                raise
            v = _one(cands, f"{method} has {len(cands)} {category} variants of "
                            f"{_and_list(sorted(modalities))}, {mods_of(cands)}. Pass the "
                            f"modalities of one, for example {example(cands[0])}.")
    up = upstream.knobs_for(s.id)
    vs = v if isinstance(v, list) else [v]
    return {"method": s.id, "variant": _and_list(_variant_key(x) for x in vs),
            "tunable": _tunable(vs[0]),
            "fixed_in_script": up["fixed_in_script"],
            "upstream_knobs": up["upstream_knobs"],
            "upstream_url": up["upstream_url"]}


def _one(cands, ambiguous: str):
    """The one variant of ``cands``, or all of them when they take the same
    parameters; else ``AmbiguousVariantError(ambiguous)``."""
    if len(cands) == 1:
        return cands[0]
    v = _agreed(cands)
    if v is None:
        raise AmbiguousVariantError(ambiguous) from None
    return v


def _agreed(variants) -> list | None:
    """``variants`` when they take the same parameters, else ``None``."""
    first = _tunable(variants[0])
    return list(variants) if all(_tunable(x) == first for x in variants[1:]) else None


def _same_types(variants, modalities) -> list:
    """The variants whose inputs are exactly the base types ``modalities`` name
    (``rna``, ``adt``, ``atac``; numbers and ATAC forms dropped), else those
    that read at least them, as ``find_methods`` matches."""
    want = {_base_modality(m) for m in modalities}
    exact = [v for v in variants if v.modality_types == want]
    return exact or [v for v in variants if want <= v.modality_types]
