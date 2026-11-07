"""Load methods.yaml into MethodSpec objects; the validators of user-facing
tokens (method, category, task, atac, modalities): a typo raises with the
valid vocabulary instead of matching nothing.
"""
from __future__ import annotations

import difflib
import functools
from pathlib import Path

import yaml

from .schema import ArgSpec, MethodSpec, OutputSpec, Variant

_YAML = Path(__file__).resolve().parent / "methods.yaml"
# the tunable hyperparams, generated from the upstream argparse
_PARAMS_YAML = Path(__file__).resolve().parent / "params.yaml"
# per-method provenance
_REFERENCES = Path(__file__).resolve().parent / "references.yaml"

# what `atac:` may say in methods.yaml
ATAC_VALUES = ("peak", "gene_activity")

# caller spellings -> registry tokens
MODALITY_ALIASES = {
    "protein": "adt",
    "peak": "atac_peak", "peaks": "atac_peak",
    "gas": "atac_gas", "gene_activity": "atac_gas",
}
_BASE_MODALITIES = ("rna", "adt", "atac")


@functools.lru_cache(maxsize=1)
def _tunable_map() -> dict:
    """{method_id: {"category:mods": {param: {default,type}}}} from params.yaml."""
    return yaml.safe_load(_PARAMS_YAML.read_text()) or {}


def _variant_key(when: dict) -> str:
    mods = "+".join(when.get("modalities", [])) or "-"
    return f"{when.get('category')}:{mods}"


def _parse_variant(d: dict, method_id: str | None = None) -> Variant:
    return Variant(
        when=d["when"],
        entrypoint=d["entrypoint"],
        language=d.get("language", "python"),
        args=[ArgSpec(**a) for a in d.get("args", [])],
        output=OutputSpec(**d["output"]),
        params=d.get("params", {}),
        tunable=(d.get("tunable")
                 or _tunable_map().get(method_id, {}).get(_variant_key(d["when"]), {})
                 or {}),
        run_env=d.get("run_env", {}),
        cwd_at_script=d.get("cwd_at_script", False),
        pty=d.get("pty", False),
        driver=d.get("driver"),
        normalize_peaks=d.get("normalize_peaks", []),
        extra_outputs=[OutputSpec(**o) for o in d.get("extra_outputs", [])],
        helpers=list(d.get("helpers", []) or []),
        downloads=list(d.get("downloads", []) or []),
        size_hint=d.get("size_hint", "") or "",
        min_cells=d.get("min_cells"),
        min_cells_per_batch=d.get("min_cells_per_batch"),
        min_genes=d.get("min_genes"),
    )


def _parse_method(d: dict) -> MethodSpec:
    variants = [_parse_variant(v, d["id"]) for v in d.get("variants", [])]
    # params.yaml has no entry for a variant that runs the same script as
    # another (scMDC, scMM cross): it takes that variant's tunables
    for v in variants:
        if not v.tunable:
            v.tunable = next((dict(o.tunable) for o in variants
                              if o.tunable and o.entrypoint == v.entrypoint), v.tunable)
    return MethodSpec(
        id=d["id"], language=d.get("language", "python"),
        tasks=d.get("tasks", []),
        atac=d.get("atac"),
        setup_hint=d.get("setup_hint", ""),
        variants=variants,
        cpu_params=d.get("cpu_params") or {},
        requires_gpu=d.get("requires_gpu", False),
        gpu_evidence=d.get("gpu_evidence", "") or "",
        gpu_use=d.get("gpu_use", "") or "",
    )


@functools.lru_cache(maxsize=1)
def _references() -> dict:
    """Parsed engine/references.yaml."""
    return yaml.safe_load(_REFERENCES.read_text()) or {}


@functools.lru_cache(maxsize=1)
def load() -> list[MethodSpec]:
    """Every method of ``methods.yaml``, in file order, with its references;
    parsed once."""
    data = yaml.safe_load(_YAML.read_text())
    specs = [_parse_method(m) for m in data["methods"]]
    refs = _references()
    for s in specs:
        s.reference = dict(refs.get(s.id) or {})
    return specs


# --------------------------------------------------------------- validators
def check_method(method_id: str) -> str:
    """The registry id of a method name: an id in any letter case
    (``"totalvi"``) or a display name (``"MOFA+"``, ``"Seurat(WNN)"``, as
    ``catalog._canonical_id`` reads them). Anything else raises
    ``KeyError`` with :func:`unknown_method_message`."""
    ids = [s.id for s in load()]
    if method_id in ids:
        return method_id
    if isinstance(method_id, str):
        from .._data import catalog          # lazy: catalog imports registry lazily
        canon = catalog._canonical_id(method_id)
        if canon in ids:
            return canon
    raise KeyError(unknown_method_message(method_id))


def method_of(name) -> str | None:
    """The package method a row name stands for, else ``None``: an id or display
    name as :func:`check_method` reads it, or one followed by a space and
    anything (``"StabMap (this run)"``)."""
    words = str(name).split(" ") if isinstance(name, str) else []
    for n in range(len(words), 0, -1):
        try:
            return check_method(" ".join(words[:n]))
        except (KeyError, TypeError):
            pass
    return None


def unknown_method_message(method_id) -> str:
    """The error text for a name that is no method."""
    return f"Unknown method {method_id}. mtb.find_methods() shows all methods."


def closest_method(method_id: str) -> str | None:
    """The registry id closest to ``method_id``, or ``None`` when none is close."""
    ids = {s.id.lower(): s.id for s in load()}
    hint = difflib.get_close_matches(str(method_id).lower(), list(ids), n=1, cutoff=0.6)
    return ids[hint[0]] if hint else None


def check_id_list(value, name: str):
    """Return ``value``; raise ``TypeError`` if it is a bare string, which
    would be iterated character by character."""
    if isinstance(value, str):
        raise TypeError(f"{name} must be a list, got the string {value!r}.")
    return value


#: The four integration categories.
CATEGORIES = ("cross", "diagonal", "mosaic", "vertical")


def check_category(category: str | None) -> str | None:
    """Validate a category token; ``None`` passes through. An unknown one raises
    ``ValueError`` listing the four."""
    if category is None or (isinstance(category, str) and category in CATEGORIES):
        return category
    raise ValueError(f"unknown category {category!r}; valid: {list(CATEGORIES)}")


@functools.lru_cache(maxsize=1)
def known_modalities() -> set[str]:
    """Every modality token a variant declares, plus the three base types."""
    toks = {m for s in load() for v in s.variants for m in v.when.get("modalities", [])}
    return toks | set(_BASE_MODALITIES)


def normalize_modalities(modalities) -> list[str]:
    """Map caller modality spellings (:data:`MODALITY_ALIASES`) to registry
    tokens, in order, without duplicates. An unknown token raises
    ``ValueError``; ``None`` -> ``None``.
    """
    if modalities is None:
        return None
    if isinstance(modalities, str):
        modalities = [modalities]
    known = known_modalities()
    out: list[str] = []
    for tok in modalities:
        low = str(tok).lower()               # RNA, ADT: the tokens are lower case
        t = MODALITY_ALIASES.get(low, low)
        if t not in known:
            raise ValueError(f"Unknown modality {tok}. Use rna, adt, atac, atac_peak or "
                             f"atac_gas (rna1, adt2 ... for batches).")
        if t not in out:
            out.append(t)
    return out


# ---------------------------------------------------------------- lookups
def get(method_id: str) -> MethodSpec:
    """The :class:`MethodSpec` of a method name, read as :func:`check_method` reads it."""
    method_id = check_method(method_id)
    return next(s for s in load() if s.id == method_id)


def list_methods(category: str | None = None) -> list[str]:
    """Registry method ids in registry order, filtered by a validated ``category``."""
    check_category(category)
    return [s.id for s in load() if not category or category in s.categories]
