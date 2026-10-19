"""Discovery: filter methods by need; enrich method_info with catalog metadata; cite."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from urllib.parse import quote as _quote

from . import _compat
from .config import _and_list
from .engine import registry, upstream, envs
from .engine import resolve as _resolve
from .engine.schema import AmbiguousVariantError, is_label_role
from .engine.schema import base_modality as _base_modality

#: representation role tokens -> the ``atac=`` value they stand for
_REPRESENTATION_TOKENS = {"atac_peak": "peak", "atac_gas": "gene_activity"}


def _representations(toks) -> set[str]:
    """The ATAC representations (``peak``, ``gene_activity``) the normalised tokens name."""
    return {_REPRESENTATION_TOKENS[t] for t in toks if t in _REPRESENTATION_TOKENS}


#: naming both ATAC roles selects the variants that read both files (MultiMAP, Seurat_v3)
_BOTH_FILES = frozenset(_REPRESENTATION_TOKENS)


def _modality_filter(modalities, atac) -> tuple[set[str] | None, str | None]:
    """``(base types every matching variant must consume, atac)``: the modality rule
    of ``find_methods`` and ``recommend`` (see the ``find_methods`` Notes)."""
    if isinstance(modalities, str):
        raise TypeError(
            f"modalities must be a list of modality tokens, got the string "
            f"{modalities!r} - did you mean modalities=[{modalities!r}]?")
    atac = registry.check_atac(atac)
    if modalities is None:
        return None, atac
    toks = registry.normalize_modalities(modalities)
    reps = sorted(_representations(toks))
    if len(reps) == 1:
        if atac is not None and atac != reps[0]:
            raise ValueError(
                f"modalities={list(modalities)} selects methods that read "
                f"{reps[0].replace('_', ' ')} but atac={atac!r}; pass one of the two")
        atac = reps[0]
    return {_base_modality(t) for t in toks}, atac


def _reads_both_files(modalities) -> bool:
    """Whether ``modalities`` names both ATAC representations, in any spelling."""
    if modalities is None or isinstance(modalities, str):
        return False
    return _BOTH_FILES <= set(registry.normalize_modalities(modalities))


def _variant_matches(v, category, want, needs_labels, atac) -> bool:
    """Whether variant ``v`` satisfies every per-variant filter; ``v=None`` (a stub)
    satisfies only filters that ask nothing of a variant."""
    if v is None:
        return want is None and atac is None and not needs_labels
    if category and v.when.get("category") != category:
        return False
    if want is not None and not want <= v.modality_types:
        return False
    if needs_labels is not None and v.needs_labels != needs_labels:
        return False
    return True


@_compat.keywords("find_methods", private="_find_methods", kept={
    "task": _compat.note("it has no replacement"), "runnable": _compat.use("mtb.scan"),
    "atac": _compat.use('modalities=["atac_peak"] or ["atac_gas"]'),
    "tunable": _compat.use("mtb.params_for")})
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
    return _find_methods(category, needs_labels=needs_labels, modalities=modalities)


def _find_methods(category: str | None = None, *, task: str | None = None,
                  needs_labels: bool | None = None, atac: str | None = None,
                  modalities: list[str] | set[str] | None = None,
                  runnable: bool | None = None, tunable: bool | None = None) -> list[str]:
    """:func:`find_methods`, with the deprecated ``task``, ``runnable`` and
    ``tunable``: ``task`` keeps the methods that declare it, ``runnable=True``
    those with a variant the package runs and ``False`` the declared stubs,
    ``tunable=True`` those with a hyperparameter ``params=`` can set."""
    registry.check_category(category)
    registry.check_task(task)
    want, atac = _modality_filter(modalities, atac)
    both = _reads_both_files(modalities)
    out = []
    for s in registry.load():
        if category and category not in s.categories:
            continue
        if task and task not in s.tasks:
            continue
        if runnable is not None and bool(s.variants) != runnable:
            continue
        if tunable is not None and any(v.tunable for v in s.variants) != tunable:
            continue
        # atac is declared per method, but only a variant with an ATAC input honours it
        if atac and s.atac != atac:
            continue
        cands = s.variants or [None]
        hits = [v for v in cands
                if _variant_matches(v, category, want, needs_labels, atac)
                and (not atac or v is None or v.consumes_atac)
                and (not both or v is not None and _BOTH_FILES <= {
                    r.rstrip("0123456789") for r in v.when.get("modalities") or []})]
        if not hits:
            continue
        out.append(s.id)
    return out


#: deprecated alias of :func:`find_methods`
list_methods = _compat.renamed("list_methods", "find_methods", find_methods)


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


#: the keys ``method_info`` returns by default: what a user reads before a run
_INFO_KEYS = ("id", "language", "notes", "categories", "env", "atac", "needs_labels",
              "supports", "runtime", "gpu", "cpu_params",
              "setup_hint", "repo_url", "reference")


class _Info(dict):
    """``method_info``'s dict: a hyperparameter key names ``params_for``."""

    def __missing__(self, key):
        if key in ("params", "fixed_in_script", "upstream_knobs", "upstream_url"):
            raise KeyError(f"{key!r} is in mtb.params_for({self.get('id')!r}, "
                           f"category, modalities)")
        raise KeyError(key)


#: the keys ``verbose=True`` adds (the maintainer's)
_VERBOSE_KEYS = ("status", "variants", "driver", "scripts_url", "version", "params",
                 "fixed_in_script", "upstream_knobs", "upstream_url", "gpu_evidence",
                 "notes_long")


@_compat.keywords("method_info", private="_method_info",
                  kept={"verbose": _compat.use("mtb.params_for")})
def method_info(method: str) -> dict:
    """Return everything known about one method as a flat dict.

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
        runtime                 tier, worst_sec, observed, host, note
        gpu                     'required' / 'used when present' / 'not used'
        cpu_params              the flags that turn CUDA off on a CPU host
        setup_hint              a step to take before the first run, or ''
        repo_url                the upstream code the benchmark ran
        reference               {doi, title, authors, journal, year}
        ```

    Raises
    ------
    KeyError
        Unknown method id. The message suggests a close match.

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
    mtb.cite : the paper to cite for a method.
    """
    return _method_info(method)


def _method_info(method: str, verbose: bool = False) -> dict:
    """:func:`method_info`, with the deprecated ``verbose=True``, which adds the
    maintainer's keys (``params``, ``status``, ``variants``, ``driver`` ...)."""
    s = registry.get(method)
    method = s.id
    if not s.variants:
        raise KeyError(f"{method}: no variants (declared stub)")
    ref = s.reference
    info = {
        "id": s.id, "language": s.language, "categories": s.categories,
        "env": envs.group_for(s.id), "atac": s.atac,
        "needs_labels": s.needs_labels, "status": s.status,
        "setup_hint": s.setup_hint,
        "variants": list(dict.fromkeys(v.entrypoint for v in s.variants)),
        "driver": next((v.driver for v in s.variants if v.driver), None),
        # percent-encoded: 'online iNMF' has a space
        "scripts_url": ("https://github.com/PYangLab/scMultiBench/tree/main/"
                        + _quote("/".join(s.variants[0].entrypoint.split("/")[:2]),
                                 safe="/")),
        "repo_url": ref.get("repo_url") or upstream.knobs_for(s.id)["upstream_url"],
        "version": ref.get("version"),
        "reference": dict(ref["reference"]) if ref.get("reference") else None,
        "supports": [{"category": v.when.get("category"),
                      "modalities": list(v.when.get("modalities", [])),
                      "output_kind": v.output.kind,
                      "needs_labels": v.needs_labels,
                      "labels": [r for r in v.roles() if is_label_role(r)],
                      "reference_batch": _reference_batch(v)}
                     for v in s.variants],
        "params": {_variant_key(v): {"tunable": _tunable(v)} for v in s.variants},
    }
    up = upstream.knobs_for(s.id)
    info["fixed_in_script"] = up["fixed_in_script"]
    info["upstream_knobs"] = up["upstream_knobs"]
    info["upstream_url"] = up["upstream_url"]
    info["notes"] = ref.get("summary") or None
    info["runtime"] = _runtime_hint(s.id)
    info["gpu"] = s.gpu
    info["cpu_params"] = dict(s.cpu_params)
    info["gpu_evidence"] = s.gpu_evidence or None
    info["notes_long"] = up["notes"]
    keys = _INFO_KEYS + (_VERBOSE_KEYS if verbose else ())
    return _Info((k, info[k]) for k in keys)


# ------------------------------------------------------------ observed runtimes
_RUNTIMES_YAML = Path(__file__).resolve().parent / "engine" / "runtimes.yaml"


@functools.lru_cache(maxsize=1)
def _runtimes() -> dict:
    """Observed per-method runtimes (engine/runtimes.yaml)."""
    import yaml
    with open(_RUNTIMES_YAML) as fh:
        return yaml.safe_load(fh) or {}


def _runtime_hint(method: str) -> dict:
    """``method_info(method)["runtime"]`` (see its Notes); a typo raises ``KeyError``."""
    method = registry.check_method(method)
    rt = dict(_runtimes().get(method, {"tier": "unknown", "worst_sec": None,
                                       "observed": []}))
    # every observation was taken on the GPU benchmark host
    rt["host"] = "gpu"
    rt["note"] = "times observed on the benchmark host (NVIDIA RTX 4090)"
    return rt


def _variant_key(v) -> str:
    """``'category:mod1+mod2'`` (``-`` for no modalities)."""
    return registry._variant_key(v.when)


_DATASET_VARIANT = _compat.use("params_for(method, category, modalities)")


@_compat.keywords("params_for", private="_params_for",
                  kept={"dataset": _DATASET_VARIANT, "data_path": _DATASET_VARIANT})
def params_for(method: str, category: str | None = None,
               modalities: list[str] | set[str] | None = None) -> dict:
    """Return the hyperparameters of one method variant.

    Call it before ``run_all(params=...)`` to see what a method accepts and
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
        Unknown method, a declared stub, or no variant for ``category`` /
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
    mtb.run_all : takes these keys as ``params={method: {key: value}}``.
    """
    return _params_for(method, category, modalities)


def _params_for(method: str, category: str | None = None,
                modalities: list[str] | set[str] | None = None, *,
                dataset: str | None = None,
                data_path: Path | str | None = None) -> dict:
    """:func:`params_for`, with the deprecated ``dataset=`` and ``data_path=``:
    the variant whose input files the dataset folder holds settles an
    ambiguous selection."""
    if dataset is not None:
        dataset, data_path = _resolve.split_dataset(dataset, data_path)
    s = registry.get(method)
    method = s.id
    registry.check_category(category)
    modalities = registry.normalize_modalities(modalities)
    if not s.variants:
        raise KeyError(f"{method}: no variants (declared stub); nothing to tune")

    def _example(v):
        return (f"params_for({method!r}, {v.when.get('category')!r}, "
                f"{list(v.when.get('modalities', []))})")

    ds_dir = None
    if dataset is not None:
        from . import config
        root = data_path if data_path is not None else config.DEFAULT.data_path
        ds_dir = Path(os.path.abspath(os.fspath(root))) / dataset

    def _by_folder(cands):
        """The one candidate the dataset folder satisfies (of nested ones the largest,
        as in ``inputs_for``), else None."""
        if ds_dir is None or not ds_dir.is_dir():
            return None
        ok = _resolve._drop_nested(
            [x for x in cands if _resolve._variant_satisfiable(x, ds_dir, s.id)])
        return ok[0] if len(ok) == 1 else None

    if category is None and modalities is None:
        if len(s.variants) > 1:
            v = _by_folder(s.variants) or _agreed(s.variants)
            if v is None:
                keys = _and_list(_variant_key(x) for x in s.variants)
                raise AmbiguousVariantError(
                    f"{method} has {len(s.variants)} variants, {keys}. Pass the "
                    f"category and modalities of one, for example "
                    f"{_example(s.variants[0])}.")
        else:
            v = s.variants[0]
    elif modalities is None:
        # the only way to reach a data_dir variant (scBridge), which has no modalities
        cands = [x for x in s.variants if x.when.get("category") == category]
        if not cands:
            from .engine.schema import no_category_message
            raise KeyError(no_category_message(method, s.wired_categories, category))
        if len(cands) > 1:
            v = _by_folder(cands) or _agreed(cands)
            if v is None:
                mods = _and_list("+".join(x.when.get("modalities", []))
                                          for x in cands)
                raise AmbiguousVariantError(
                    f"{method} has {len(cands)} {category} variants, {mods}. Pass the "
                    f"modalities too, for example {_example(cands[0])}.")
        else:
            v = cands[0]
    elif category is None:
        from .engine.schema import modality_family
        want = {modality_family(m) for m in modalities}
        cands = [x for x in s.variants
                 if set(x.when.get("modalities", [])) == set(modalities)]
        if not cands:
            cands = [x for x in s.variants
                     if {modality_family(m) for m in x.when.get("modalities", [])} == want]
        if not cands:
            cands = _same_types(s.variants, modalities)
        if not cands:
            keys = _and_list(_variant_key(x) for x in s.variants)
            raise KeyError(
                f"No variant of {method} takes the modalities "
                f"{_and_list(sorted(modalities))}. Its variants: {keys}.")
        if len(cands) > 1:
            v = _by_folder(cands) or _agreed(cands)
            if v is None:
                keys = _and_list(_variant_key(x) for x in cands)
                raise AmbiguousVariantError(
                    f"The modalities {_and_list(sorted(modalities))} match "
                    f"{len(cands)} variants of {method}, {keys}. Pass the category "
                    f"too, for example {_example(cands[0])}.")
        else:
            v = cands[0]
    else:
        try:
            v = s.select(category, set(modalities), loose=True)
        except AmbiguousVariantError:
            raise
        except KeyError:
            # base tokens, as find_methods takes them: rna + atac for rna1..3 + atac1..3
            cands = _same_types([x for x in s.variants
                                 if x.when.get("category") == category], modalities)
            if len(cands) > 1:
                v = _by_folder(cands) or _agreed(cands)
                if v is None:
                    keys = _and_list("+".join(x.when.get("modalities", [])) for x in cands)
                    raise AmbiguousVariantError(
                        f"{method} has {len(cands)} {category} variants of "
                        f"{_and_list(sorted(modalities))}, {keys}. Pass the modalities "
                        f"of one, for example {_example(cands[0])}.") from None
            elif cands:
                v = cands[0]
            else:
                raise
    up = upstream.knobs_for(s.id)
    vs = v if isinstance(v, list) else [v]
    return {"method": s.id, "variant": _and_list(_variant_key(x) for x in vs),
            "tunable": _tunable(vs[0]),
            "fixed_in_script": up["fixed_in_script"],
            "upstream_knobs": up["upstream_knobs"],
            "upstream_url": up["upstream_url"]}


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


@_compat.keywords("cite", renamed={"fmt": "format"})
def cite(*methods, format: str = "text") -> str:
    """Return citation text for the benchmark and, optionally, the methods you ran.

    Parameters
    ----------
    *methods : str | list[str] | BatchResult
        Method ids, one per argument or as one list; a ``BatchResult`` = the
        methods that ran; ``"all"`` = every method; none = the benchmark only.
    format : str
        ``"text"`` (one line per entry) or ``"bibtex"`` (one ``@article`` per
        entry).

    Returns
    -------
    str
        The benchmark entry first, then one entry per method in the order
        given.

    Raises
    ------
    ValueError
        ``format`` is neither ``"text"`` nor ``"bibtex"``; the message lists both.
    KeyError
        Unknown method id; the message suggests a close match, if any.
    TypeError
        Several ids are given and one is not a string.

    Examples
    --------
    >>> import multibench as mtb
    >>> print(mtb.cite())                                  # the benchmark only
    >>> print(mtb.cite("Matilda", "MOFA2"))
    >>> print(mtb.cite(mtb.load_batch("out/"), format="bibtex"))   # what ran

    Notes
    -----
    **Two spellings.** Both forms return the same text, and ``cite(None)`` is
    ``cite()``:

    - ``cite('Matilda', 'MOFA2')`` - one id per argument, like the CLI
      ``multibench cite Matilda MOFA2``.
    - ``cite(['Matilda', 'MOFA2'])`` - one list or tuple.

    **Formats.**

    - ``"text"`` - one ``Authors. Title. Journal (year). https://doi.org/...``
      line per entry.
    - ``"bibtex"`` - one ``@article{<id>_<year>, ...}`` per entry, separated
      by a blank line; the benchmark's key is ``scMultiBench_<year>``.

    **Methods without a reference.** A method without a verified reference
    is emitted as a ``% <id>: no verified reference; see <repo_url>``
    comment (bibtex) or the same line without ``%`` (text).

    See Also
    --------
    mtb.method_info : carries the same ``reference``, ``repo_url`` and ``version`` per method.
    """
    args = list(methods)
    if format not in _CITE_FORMATS:
        raise ValueError(f"unknown format {format!r}; valid: {list(_CITE_FORMATS)}")
    if len(args) == 1 and hasattr(args[0], "records") and hasattr(args[0], "summary"):
        # a BatchResult: the methods that ran, in its order
        args = [[r["method"] for r in args[0].records
                 if r.get("status") != "SKIPPED"
                 and not str(r.get("status", "")).startswith(("FAIL", "TIMEOUT"))]]
    if not args:
        ids: list[str] = []
    elif len(args) == 1 and not isinstance(args[0], str):
        ids = [] if args[0] is None else [registry.check_method(m) for m in args[0]]
    elif len(args) == 1 and args[0] == "all":
        ids = registry.list_methods()
    else:
        bad = [a for a in args if not isinstance(a, str)]
        if bad:
            raise TypeError(
                f"cite(): method ids must be strings, got {type(bad[0]).__name__}; "
                f"pass one list (cite(['Matilda', 'MOFA2'])) or one id per argument")
        ids = [registry.check_method(m) for m in args]
    parts = [_format_entry("scMultiBench", registry.benchmark_reference(), format)]
    for m in ids:
        s = registry.get(m)
        ref = s.reference.get("reference")
        if ref:
            parts.append(_format_entry(m, ref, format))
        else:
            url = s.reference.get("repo_url") or "(no repo_url)"
            note = f"{m}: no verified reference; see {url}"
            parts.append(("% " + note) if format == "bibtex" else note)
    return _compat.Text("\n\n".join(parts) if format == "bibtex" else "\n".join(parts))
