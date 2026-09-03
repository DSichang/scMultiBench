"""Discovery: filter methods by need; enrich method_info with catalog metadata; cite."""
from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path

from .engine import registry, upstream, envs
from .engine import resolve as _resolve
from .engine.runner import _AUX_ROLES  # noqa: F401  (re-exported for back-compat)
from .engine.schema import AmbiguousVariantError, is_label_role
from .engine.schema import base_modality as _base_modality  # noqa: F401  (tests import it from here)


def _modality_types(spec) -> set[str]:
    """Union of base modality types consumed across all of a method's variants.

    Auxiliary roles (data_dir, source/target data & cty, out_dir) and label roles
    are excluded. Methods with no variants (declared stubs) yield an empty set.
    (Thin alias of ``MethodSpec.modality_types``, kept for callers/tests.)
    """
    return spec.modality_types


def _variant_matches(v, category, want, needs_labels, atac) -> bool:
    """Does ONE variant satisfy every per-variant filter at once?

    ``v`` may be ``None`` for a declared-but-unwired stub (no variants): it
    satisfies only the filters that ask for nothing a variant could supply
    (``needs_labels`` None/False, no ``modalities``, no ``atac``).
    """
    if v is None:
        return want is None and atac is None and not needs_labels
    if category and v.when.get("category") != category:
        return False
    # A directory-fed variant that names no matrix (the spatial registration
    # methods) cannot be judged by modality: keep it, find_methods warns.
    if want is not None and not v.modalities_unknown and not want <= v.modality_types:
        return False
    if needs_labels is not None and v.needs_labels != needs_labels:
        return False
    return True


def find_methods(category: str | None = None, task: str | None = None,
                 needs_labels: bool | None = None,
                 atac: str | None = None,
                 modalities: list[str] | set[str] | None = None,
                 runnable: bool | None = None,
                 tunable: bool | None = None,
                 *, available: bool | None = None) -> list[str]:
    """Return method ids matching all supplied filters.

    Every token is validated: a typo raises ``ValueError`` naming the valid
    vocabulary instead of silently matching nothing.

    **Filters are evaluated per VARIANT**: a method matches when at least ONE
    of its variants satisfies ``category`` AND ``modalities`` AND
    ``needs_labels`` AND ``atac`` *together*. So
    ``find_methods(category='vertical', modalities=['rna', 'adt'],
    needs_labels=False)`` keeps scMoMaT (its vertical rna+adt variant takes no
    labels; only its mosaic variant does) and drops Multigrate from
    ``find_methods(category='vertical', modalities=['rna', 'atac'])`` (rna+atac
    exists only as a mosaic variant; ``inputs_for(..., 'vertical',
    modalities=['rna','atac'])`` would raise). ``task``, ``runnable``,
    ``tunable`` and ``available`` are method-level.

    ``category`` is one of ``vertical``/``diagonal``/``mosaic``/``cross``;
    ``task`` one of :func:`multibench.list_tasks`.
    ``needs_labels`` is derived from the variants' roles: ``True`` keeps methods
    with a matching variant that takes a cell-type-label (``cty``) role as a
    REQUIRED input, ``False`` one that takes none. Note the difference from
    ``method_info(m)['needs_labels']``, which is the METHOD-level "any variant
    needs labels" flag; the per-variant answer is
    ``method_info(m)['supports'][i]['needs_labels']``.
    ``atac`` is an exact match on the method's declared ATAC representation:
    ``"peak"`` or ``"gene_activity"`` (``"peaks"``/``"gas"`` accepted as aliases);
    it is the representation the UPSTREAM script expects, which is not always
    what its role name suggests (moETM/scMM/iPOLNG take role ``atac_gas`` but
    consume peaks). Only variants that actually consume an ATAC input can
    satisfy it (Multigrate declares ``atac: peak`` for its mosaic rna+atac
    variant; its vertical rna+adt variant does not match ``atac='peak'``).
    ``modalities`` keeps methods with a variant that consumes ALL of the
    requested base modality types (e.g. ``["rna", "atac"]``); ``"protein"`` is
    accepted for ``adt`` and role tokens (``atac_gas``, ``atac_peak``, ``rna1``
    ...) are reduced to their base type, so ``["rna", "atac_gas"]`` means
    ``["rna", "atac"]``. Because modality info is derived from a method's
    variants, this filter implicitly excludes the declared-but-unwired stub
    methods (those without variants). A method fed a DIRECTORY is judged by
    the bare filenames its variant names (scBridge's ``rna.h5`` /
    ``atac_gas.h5`` make it an rna+atac method); the spatial-registration
    variants name nothing (their ``data_dir`` holds ``.h5ad`` slices), so
    they cannot be filtered by modality: they are KEPT and a ``UserWarning``
    names them - ``task='registration'`` (or ``category``) selects or
    excludes them deliberately.
    ``runnable=True`` restricts to methods with at least one variant (usable by
    ``inputs_for``/``run``); ``runnable=False`` returns only the stubs.
    ``tunable=True`` keeps only methods that expose at least one hyperparameter
    on their command line (i.e. where ``run(params=...)`` can change anything) -
    the rest hardcode their settings upstream.
    ``available`` (keyword-only; default ``None`` = no filter): ``True`` keeps
    methods whose scripts a public install can run
    (``method_info(m)['availability'] == 'public'``); ``False`` keeps the
    ``'benchmark-host-only'`` ones, whose entrypoint is an absolute path on the
    benchmark host and is not published (SPIRAL, GPSA) - they are wired and
    verified there, but ``scan`` reports them not runnable elsewhere.
    """
    registry.check_category(category)
    registry.check_task(task)
    atac = registry.check_atac(atac)
    want = (set(registry.normalize_modalities(modalities, base=True))
            if modalities is not None else None)
    out = []
    unfiltered: list[str] = []
    for s in registry.load():
        if category and category not in s.categories:
            continue
        if task and task not in s.tasks:
            continue
        if runnable is not None and bool(s.variants) != runnable:
            continue
        if tunable is not None:
            has = any(v.tunable for v in s.variants)
            if has != tunable:
                continue
        if available is not None and (s.availability == "public") != available:
            continue
        # `atac` is declared once per method but only variants that take an
        # ATAC input can honour it; fold that into the per-variant test.
        if atac and s.atac != atac:
            continue
        cands = s.variants or [None]
        hits = [v for v in cands
                if _variant_matches(v, category, want, needs_labels, atac)
                and (not atac or v is None or v.consumes_atac)]
        if not hits:
            continue
        out.append(s.id)
        if want is not None and all(v is not None and v.modalities_unknown for v in hits):
            unfiltered.append(s.id)
    if unfiltered:
        warnings.warn(
            f"find_methods: {len(unfiltered)} method(s) take a directory (data_dir role) "
            f"and could not be filtered by modalities={sorted(want)}; kept: "
            f"{', '.join(unfiltered)} - see method_info(m)['supports'] "
            f"(task='registration' selects the spatial ones)",
            UserWarning, stacklevel=2)
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
        prose for this method (None for methods outside the audit; for a
        ``benchmark-host-only`` method one sentence saying why is appended) -
        and ``verification``, the per-method verification record (see below).
        The default ``notes`` is the short third-person summary from
        engine/references.yaml.

    Keys
    ----
    id, language, categories, tasks, env, atac, needs_labels, status,
    availability, setup_hint,
    variants (distinct upstream entrypoints, in order), driver (package-side
    wrapper actually executed, or None when the upstream script runs directly),
    scripts_url (the benchmark's tools_scripts folder), repo_url, version,
    reference ({doi, title, authors, journal, year} or None), notes,
    supports (per variant: category, modalities, output_kind, n_tunable,
    needs_labels, labels - the label roles the variant reads, e.g.
    ``['cty']`` / ``['rna_cty']`` / ``[]``), params (per variant key
    'category:mods': defaults, tunable,
    effective), fixed_in_script, upstream_knobs, upstream_url;
    with ``verbose=True`` also notes_long and verification.

    ``needs_labels`` is the METHOD-level flag: True when ANY variant takes a
    cell-type-label (``cty``) role as a required input ("needs labels in at
    least one variant"). It is NOT per category: scMoMaT is True because its
    mosaic variant takes ``cty1..3``, while its vertical/cross variants take no
    labels. For the per-variant answer read ``supports[i]['needs_labels']``
    (``find_methods(needs_labels=...)`` filters per variant).

    ``status`` is the registry's wiring status: ``'verified'`` means the
    command template was cross-checked against the upstream entrypoint AND
    the method was executed end to end on a reference dataset (``'declared'``
    = wired but not run). It says nothing about where the script lives (see
    ``availability``) nor about the numbers that run produced (see
    ``verification``).

    ``availability`` is ``'public'`` (every entrypoint lives in the public
    scMultiBench repository; a public install can run it) or
    ``'benchmark-host-only'`` (an entrypoint is an absolute path on the
    benchmark host and is not published - SPIRAL, GPSA; ``scan`` reports them
    not runnable, ``find_methods(available=True)`` drops them). Derived from
    the entrypoints, no hand flag.

    ``verification`` (``verbose=True`` only) is the evidence behind
    ``status='verified'``: a list of dicts (one per recorded run of this
    method) with ``dataset, category, status, wall_s, ARI, baseline, verdict,
    note`` read from ``files/final_verification.tsv``; ``None`` when the
    method has no recorded run. ``status`` there is the run outcome
    (``CHAIN_OK`` = ran and its embedding was scored; ``CHAIN_OK_GRAPH_METHOD``
    = ran, graph output scored via its UMAP; ``RUN_OK_NO_EMBEDDING`` = ran to
    completion but produces no embedding to score - Seurat_WNN, MIRA and the
    registration methods, so ``ARI`` is None), ``ARI`` the re-run's score,
    ``baseline`` the benchmark's own figure for the same dataset and
    ``verdict`` ``OK`` (reproduced) or ``DRIFT`` (ran, but the score moved
    away from the baseline - the method is still ``verified`` in the wiring
    sense; compare the two numbers before trusting either).
    """
    s = registry.get(method)
    ref = s.reference or {}
    info = {
        "id": s.id, "language": s.language, "categories": s.categories,
        # `env` is the conda env run() actually executes in: the shared group
        # env, or the method's own scmb_<method> env (see engine.envs.group_for).
        "tasks": s.tasks, "env": envs.group_for(s.id), "atac": s.atac,
        "needs_labels": s.needs_labels, "status": s.status,
        # public | benchmark-host-only, derived from the entrypoints (schema)
        "availability": s.availability,
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
                      "needs_labels": v.needs_labels,
                      "labels": [r for r in v.roles() if is_label_role(r)]}
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
        notes_long = up["notes"]
        if s.availability != "public":
            why = _availability_sentence(s)
            notes_long = f"{notes_long.rstrip()} {why}" if notes_long else why
        info["notes_long"] = notes_long
        info["verification"] = verification_for(s.id)
    if files_dir is not None:
        from .data import catalog
        cat = catalog.methods(files_dir)
        match = cat[cat["canonical_id"] == s.id]
        if len(match):
            row = match.iloc[0]
            info["deep_learning"] = row["deep_learning"]
            info["output"] = row["output"]
    return info


def _availability_sentence(spec) -> str:
    """One sentence saying why a method is ``benchmark-host-only``."""
    eps = sorted({v.entrypoint for v in spec.variants if not v.is_public})
    return (f"Availability: benchmark-host-only - the entrypoint "
            f"{', '.join(eps)} is an absolute path on the machine the benchmark "
            f"was produced on and is not published in the scMultiBench repository, "
            f"so a public install cannot fetch or run it (scan reports it not "
            f"runnable; find_methods(available=True) omits it).")


#: per-method verification log shipped with the package (files/*): one row per
#: verified run - method, dataset, category, status, wall_s, ARI, baseline,
#: verdict, note. A copy of notebooks/results/final_verification.tsv; the test
#: suite pins the two byte-identical.
VERIFICATION_TSV = "final_verification.tsv"
_VERIFICATION_COLUMNS = ("dataset", "category", "status", "wall_s", "ARI",
                         "baseline", "verdict", "note")


@functools.lru_cache(maxsize=1)
def _verification_table(path: str) -> dict[str, list[dict]]:
    """``{method_id: [row, ...]}`` from the verification TSV (empty if absent)."""
    import csv

    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, list[dict]] = {}
    with open(p, newline="") as fh:
        for raw in csv.DictReader(fh, delimiter="\t"):
            row = {}
            for k in _VERIFICATION_COLUMNS:
                val = (raw.get(k) or "").strip()
                if k == "wall_s":
                    row[k] = int(float(val)) if val else None
                elif k in ("ARI", "baseline"):
                    row[k] = float(val) if val else None
                else:
                    row[k] = val or None
            out.setdefault((raw.get("method") or "").strip(), []).append(row)
    return out


def verification_for(method: str, files_dir: Path | str | None = None) -> list[dict] | None:
    """The recorded end-to-end verification run(s) of ``method``.

    Reads ``<files_dir>/final_verification.tsv`` (default: the package's
    ``files/``; it is what ``method_info(m, verbose=True)['verification']``
    returns). Each entry is ``{dataset, category, status, wall_s, ARI,
    baseline, verdict, note}`` - see :func:`method_info` for the meaning of the
    fields. ``None`` when the method has no recorded run (or the file is not
    shipped). The method id is validated (``KeyError`` on a typo).
    """
    registry.check_method(method)
    from . import config
    base = Path(files_dir) if files_dir is not None else config.DEFAULT.files_path
    rows = _verification_table(str(base / VERIFICATION_TSV)).get(method)
    return [dict(r) for r in rows] if rows else None


def _variant_key(v) -> str:
    """Stable human-readable key for a variant: 'category:mod1+mod2'."""
    mods = "+".join(v.when.get("modalities", [])) or "-"
    return f"{v.when.get('category')}:{mods}"


def params_for(method: str, category: str | None = None,
               modalities: list[str] | set[str] | None = None, *,
               dataset: str | None = None,
               data_path: Path | str | None = None) -> dict:
    """Return the hyperparameters of one method variant.

    Parameters
    ----------
    method : registry id (``KeyError`` with a did-you-mean hint otherwise).
    category : ``vertical`` / ``diagonal`` / ``mosaic`` / ``cross``; selects
        the variant exactly like ``run`` (validated). May be omitted when the
        method has only one variant.
    modalities : the variant's modality tokens; ``protein`` is accepted for
        ``adt`` and ``atac`` for either ATAC representation role. May be
        omitted when ``category`` alone selects one variant (the only way to
        reach a ``data_dir`` variant such as scBridge's or PASTE's).
    dataset : keyword-only. A dataset folder name; when the selection is
        still ambiguous, the ONE variant whose input files are all present in
        ``<data_path>/<dataset>`` is used (``params_for('Matilda',
        dataset='D11')`` is the rna+adt variant). Nothing changes when the
        folder settles nothing.
    data_path : keyword-only; root containing the dataset folder (default
        ``config.DEFAULT.data_path``).

    Returns
    -------
    dict
        ``{"method", "variant", "defaults", "tunable", "effective",
        "fixed_in_script", "upstream_knobs", "upstream_url"}`` where:

        * ``defaults`` - parameters the package emits on every run. Override
          them with ``run(..., params={...})``; the override is merged over
          these.
        * ``tunable`` - documentation of the parameters the UPSTREAM script
          accepts on its command line, as ``{name: {"default": ..., "type":
          ...}}``. The ``default`` here is the upstream argparse default, NOT
          necessarily what a wrapper run uses.
        * ``effective`` - ``tunable`` defaults overlaid with ``defaults``: the
          value each knob really takes when you pass no ``params``.
        * ``fixed_in_script`` - the values the script pins, each with the
          ``file:line`` that pins it.
        * ``upstream_knobs`` - what the wrapped library documents (with its
          own defaults), unreachable without editing the script.

        An **empty** ``tunable`` means the upstream script exposes no
        hyperparameters on its command line. Because this project never
        modifies method scripts, such a method cannot be tuned through the
        wrapper - but it is not parameterless, which is what the last two keys
        say (both empty for methods outside the upstream audit).

    Raises
    ------
    KeyError
        Unknown method, a declared stub, or no variant for ``category`` /
        ``modalities``.
    ValueError
        Several variants fit (:class:`AmbiguousVariantError`, which also
        derives from ``KeyError`` for older callers); the message spells out
        the call that selects one, e.g. ``params_for('Matilda', 'vertical',
        ['rna', 'adt'])``.
    """
    s = registry.get(method)
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
        """The single candidate the dataset folder satisfies, else None."""
        if ds_dir is None or not ds_dir.is_dir():
            return None
        ok = [x for x in cands if _resolve._variant_satisfiable(x, ds_dir, s.id)]
        return ok[0] if len(ok) == 1 else None

    if category is None and modalities is None:
        if len(s.variants) > 1:
            v = _by_folder(s.variants)
            if v is None:
                raise AmbiguousVariantError(
                    f"{method} has {len(s.variants)} variants - pass category and "
                    f"modalities, e.g. {_example(s.variants[0])}; available: "
                    f"{[_variant_key(x) for x in s.variants]}")
        else:
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
            v = _by_folder(cands)
            if v is None:
                raise AmbiguousVariantError(
                    f"{method} has {len(cands)} {category!r} variants - also pass "
                    f"modalities, e.g. {_example(cands[0])}; available: "
                    f"{[_variant_key(x) for x in cands]}")
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
            raise KeyError(
                f"{method}: no variant takes modalities {sorted(modalities)}; "
                f"available: {[_variant_key(x) for x in s.variants]}")
        if len(cands) > 1:
            v = _by_folder(cands)
            if v is None:
                raise AmbiguousVariantError(
                    f"{method}: modalities {sorted(modalities)} match {len(cands)} "
                    f"variants - also pass category, e.g. {_example(cands[0])}; "
                    f"available: {[_variant_key(x) for x in cands]}")
        else:
            v = cands[0]
    else:
        v = s.select(category, set(modalities), loose=True)
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


def cite(*method_ids, fmt: str = "bibtex", methods=None) -> str:
    """Citation text for the benchmark and (optionally) the methods you ran.

    Both spellings work: ``cite('Matilda', 'MOFA2')`` (one id per argument,
    like the CLI ``multibench cite Matilda MOFA2``) and
    ``cite(['Matilda', 'MOFA2'])`` / ``cite(methods=[...])`` (one list).

    Parameters
    ----------
    *method_ids : method ids, each a ``str`` (``KeyError`` with a did-you-mean
        hint for an unknown id); or a SINGLE list/tuple of ids; or ``"all"``
        for every registry method; nothing -> the benchmark entry only. The
        earlier positional ``cite(methods, fmt)`` form is still accepted.
    fmt : keyword; ``"bibtex"`` (one ``@article`` per entry) or ``"text"``
        (one "Authors. Title. Journal (year). https://doi.org/..." line per
        entry). ``ValueError`` listing the two on anything else.
    methods : keyword alias of the earlier signature (``cite(methods=[...])``);
        not combinable with positional ids.

    Returns
    -------
    str
        The benchmark entry first, then one entry per method in the order
        given. Methods whose DOI is not curated in engine/references.yaml are
        emitted as a ``% <id>: no verified reference; see <repo_url>`` comment
        (bibtex) / ``<id>: ... <repo_url>`` line (text) rather than silently
        dropped. Every DOI in the table was resolved against Crossref.
    """
    args = list(method_ids)
    # legacy positional form cite(methods, fmt): the 2nd positional is a format
    if len(args) == 2 and args[1] in _CITE_FORMATS and not isinstance(args[0], str):
        args, fmt = args[:1], args[1]
    if methods is not None:
        if args:
            raise TypeError("cite(): pass method ids positionally OR as methods=, not both")
        args = [methods]
    if fmt not in _CITE_FORMATS:
        raise ValueError(f"unknown fmt {fmt!r}; valid: {list(_CITE_FORMATS)}")
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
                f"pass ONE list (cite(['Matilda', 'MOFA2'])) or one id per argument")
        ids = [registry.check_method(m) for m in args]
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
