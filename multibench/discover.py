"""Discovery: filter methods by need; enrich method_info with catalog metadata."""
from __future__ import annotations

import functools
from pathlib import Path

from ._text import _and_list
from .engine import registry, upstream, envs
from .engine.schema import (AmbiguousVariantError, is_label_role, modality_family,
                            no_category_message, no_variant_message)
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
    """``{name: {"type", "default"}}``: ``default`` is what a run uses. Every
    true/false parameter shows as ``bool``, whichever way the script reads it
    (``boolean`` and ``flag`` in params.yaml decide only the command line)."""
    eff = _effective(v)
    out = {}
    for k, t in v.tunable.items():
        t = dict(t or {})
        if t.get("type") in ("boolean", "flag"):
            t["type"] = "bool"
        out[k] = {**t, "default": eff[k]}
    return out


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
    mtb.params_for : the parameters of one variant, with upstream defaults.
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
                   "reference_batch": v.reference_batch}
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
    """Return the parameters of one method variant.

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
        what a run uses when you pass nothing, and ``None`` leaves it to the
        method's library), ``fixed_in_script`` (what the
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
    cands = [v for v in s.variants if category in (None, v.when.get("category"))]
    if not cands:
        raise KeyError(no_category_message(method, s.wired_categories, category))
    if modalities is not None:
        want = {modality_family(m) for m in modalities}
        cands = ([v for v in cands if set(v.when.get("modalities", [])) == set(modalities)]
                 or [v for v in cands
                     if {modality_family(m) for m in v.when.get("modalities", [])} == want]
                 or _same_types(cands, modalities))
        if not cands:
            raise KeyError(no_variant_message(method, modalities, category))
    first = _tunable(cands[0])
    if any(_tunable(v) != first for v in cands[1:]):
        one = cands[0]
        raise AmbiguousVariantError(
            f"{method} has variants that take other parameters. Pass the category and "
            f"modalities of one, for example params_for({method!r}, "
            f"{one.when.get('category')!r}, {list(one.when.get('modalities', []))}).")
    up = upstream.knobs_for(s.id)
    return {"method": s.id, "variant": _and_list(_variant_key(v) for v in cands),
            "tunable": first,
            "fixed_in_script": up["fixed_in_script"],
            "upstream_knobs": up["upstream_knobs"],
            "upstream_url": up["upstream_url"]}


def _same_types(variants, modalities) -> list:
    """The variants whose inputs are exactly the base types ``modalities`` name
    (``rna``, ``adt``, ``atac``; numbers and ATAC forms dropped), else those
    that read at least them, as ``find_methods`` matches."""
    want = {_base_modality(m) for m in modalities}
    exact = [v for v in variants if v.modality_types == want]
    return exact or [v for v in variants if want <= v.modality_types]
