"""Resolve a catalog dataset id + method to concrete input file paths."""
from __future__ import annotations

import functools
import os
import re
import sys
import warnings
from pathlib import Path

from .. import _compat, config
from ..config import _and_list
from . import registry
from .schema import (_NON_MODALITY_ROLES, AmbiguousVariantError, _batch_of,
                     base_modality, is_label_role)

#: The docs section that lists the files of a dataset folder, per category.
LAYOUT_URL = "https://dsichang.github.io/scMultiBench/tutorials/run/#folder-layout"

# file stems tried in order for a role; the first existing file wins
_ROLE_FILE_CANDIDATES = {
    "atac_gas": ("atac_gas", "atac"),
    "atac_peak": ("atac_peak", "peak"),
    "cty": ("cty", "rna_cty"),          # older paired folders: rna_cty.csv
}
# atac<i> also reads atac_peak<i>.h5, an older name of the mosaic ATAC files
_NUMBERED_ROLE_RE = re.compile(r"^(rna|adt|atac)(\d+)$")
_NUMBERED_FILE_CANDIDATES = {"atac": ("atac", "atac_peak")}


def _role_stems(role: str) -> tuple[tuple[str, ...], bool]:
    """``(file stems to try, numbered)`` for a role.

    ``atac2`` -> ``(('atac2', 'atac_peak2'), True)``; ``atac_gas`` ->
    ``(('atac_gas', 'atac'), False)``; ``cty1`` -> ``(('cty1',), True)``.
    A numbered role never tries another batch number.
    """
    m = _NUMBERED_ROLE_RE.match(role)
    if m:
        base, digits = m.groups()
        return tuple(f"{b}{digits}" for b in _NUMBERED_FILE_CANDIDATES.get(base, (base,))), True
    return _ROLE_FILE_CANDIDATES.get(role, (role,)), role[-1:].isdigit()


def _resolve_role(ds_dir: Path, role: str) -> Path:
    """The file of ``role`` in ``ds_dir`` (:func:`_role_stems`; ``.csv`` first
    for a label role). An unnumbered role also takes ``<stem>1`` when there
    is no ``<stem>2``. Nothing found: the canonical name, which does not exist."""
    is_label = ("cty" in role) or ("label" in role)
    stems, numbered = _role_stems(role)
    if role == "cty" and (ds_dir / "atac_cty.csv").exists():
        # rna_cty.csv next to atac_cty.csv labels the RNA cells of diagonal data
        stems = ("cty",)
    exts = (".csv", ".h5") if is_label else (".h5",)
    for stem in stems:
        for ext in exts:
            p = ds_dir / f"{stem}{ext}"
            if p.exists():
                return p
            if numbered:
                continue
            first_batch = ds_dir / f"{stem}1{ext}"
            if first_batch.exists() and not (ds_dir / f"{stem}2{ext}").exists():
                return first_batch
    fallback_ext = ".csv" if is_label else ".h5"
    return ds_dir / f"{stems[0]}{fallback_ext}"


def _batch_column_advice() -> str:
    """Export without per-batch files, then where the batch column goes."""
    return ('export without batch= and pass the column itself, as '
                       'run(batch=adata.obs["donor"])')


def _one_file_advice(category: str, *, stem: str = "rna", has_adt: bool = False) -> str:
    """What to do instead of per-batch files for a vertical / diagonal folder."""
    advice = _batch_column_advice()
    advice = advice[:1].upper() + advice[1:] + "."
    if category == "diagonal":
        several = _several_batch_diagonal_methods()
        counts = {len(_variant_batches(v.when.get("modalities") or []))
                  for spec in registry.load() for v in spec.variants
                  if v.when.get("category") == "diagonal"} - {0}
        n = f"exactly {counts.pop()}" if len(counts) == 1 else "several"
        also = (f" {_and_list(several)} also read {n} numbered RNA and ATAC batches, "
                "written one batch per call with "
                + "batch_index=." if several else "")
        return "Diagonal methods read one rna.h5 and one ATAC file. " + advice + also
    cross = 'category="cross"'
    return (f"{category.capitalize()} methods read one {stem}.h5. " + advice
            + (f" For RNA+ADT batches, use {cross}." if has_adt else ""))


#: modality file stems whose numbered copies (rna1.h5, rna2.h5) mark a per-batch folder
_PER_BATCH_STEMS = ("rna", "adt", "atac", "atac_peak", "atac_gas")


def _per_batch_hint(ds_dir: Path, category: str | None, stems=None) -> str | None:
    """Hint for a vertical / diagonal folder of per-batch files (``<stem>1.h5``
    and ``<stem>2.h5`` but no ``<stem>.h5``, for one of ``stems``), else
    ``None``, also for the diagonal layout some methods read per batch."""
    if category not in ("vertical", "diagonal") or not Path(ds_dir).is_dir():
        return None
    ds_dir = Path(ds_dir)
    numbered = [b for b in _PER_BATCH_STEMS
                if (ds_dir / f"{b}1.h5").is_file() and (ds_dir / f"{b}2.h5").is_file()
                and not (ds_dir / f"{b}.h5").is_file()]
    hit = [b for b in numbered if stems is None or b in stems]
    if not hit:
        return None
    if category == "diagonal" and _per_batch_diagonal_methods(ds_dir):
        return None
    ex = hit[0]
    return (f"This folder holds per-batch files ({ex}1.h5, {ex}2.h5, ...). "
            + _one_file_advice(category, stem=ex,
                               has_adt="adt" in numbered
                               and not any(b.startswith("atac") for b in numbered)))


def _several_batch_diagonal_methods() -> list[str]:
    """The methods with a diagonal variant of numbered roles (several RNA and
    several ATAC batches), in registry order."""
    return [spec.id for spec in registry.load()
            if any(v.when.get("category") == "diagonal"
                   and _variant_batches(v.when.get("modalities") or [])
                   for v in spec.variants)]


def _per_batch_diagonal_methods(ds_dir: Path) -> list[str]:
    """:func:`_several_batch_diagonal_methods` whose input files are all in
    ``ds_dir``."""
    return [spec.id for spec in registry.load()
            if any(v.when.get("category") == "diagonal"
                   and _variant_batches(v.when.get("modalities") or [])
                   and _variant_satisfiable(v, Path(ds_dir), spec.id)
                   for v in spec.variants)]


def _resolve_data_dir(ds_dir: Path) -> str:
    """The ``data_dir`` of a dataset, ending with a separator (scBridge
    concatenates ``data_path + filename``): the first of ``processed/`` and
    ``ds_dir`` with a ``.h5ad``, else ``processed/`` if it exists, else ``ds_dir``."""
    for cand in (ds_dir / "processed", ds_dir):
        if cand.is_dir() and any(cand.glob("*.h5ad")):
            return os.path.join(str(cand), "")
    proc = ds_dir / "processed"
    return os.path.join(str(proc if proc.is_dir() else ds_dir), "")


def _vertical_atac_by_form(variant, ds_dir: Path, method: str) -> Path | None:
    """In a vertical folder with both ``atac_peak.h5`` and ``atac_gas.h5``,
    the one of the form ``method`` reads; else ``None``. A folder with
    ``atac_cty.csv`` is diagonal: its ATAC cells are not the RNA cells."""
    if variant.when.get("category") != "vertical":
        return None
    peak, gas = ds_dir / "atac_peak.h5", ds_dir / "atac_gas.h5"
    if not (peak.is_file() and gas.is_file()) or (ds_dir / "atac_cty.csv").exists():
        return None
    return {"peak": peak, "gene_activity": gas}.get(registry.get(method).atac)


def _resolve_variant_inputs(variant, ds_dir: Path, method: str) -> dict:
    """``{role: path}`` for every input role of ``variant`` in ``ds_dir``; a
    missing file resolves to its canonical name."""
    # "=VALUE" roles are literals (scMoMaT's per-batch None), not files
    roles = [r
             for a in variant.args if a.const is None
             for r in (a.roles or [a.role])
             if r and not str(r).startswith("=")
             and r not in ("out_dir", "data_dir")]
    out = {role: str(_resolve_role(ds_dir, role)) for role in roles}
    both = _vertical_atac_by_form(variant, ds_dir, method)
    if both is not None:
        out.update({role: str(both) for role in roles if role in ("atac", "atac_gas")})
    if variant.takes_data_dir:
        out["data_dir"] = _resolve_data_dir(ds_dir)
    return out


def _variant_satisfiable(variant, ds_dir: Path, method: str) -> bool:
    """The folder holds every input file of ``variant`` (existence only; a
    ``data_dir`` must hold the files the method names)."""
    got = _resolve_variant_inputs(variant, ds_dir, method)
    if any(not Path(p).exists() for p in got.values()):
        return False
    if "data_dir" in got:
        return _check_data_dir(variant, got["data_dir"])[0]
    return True


def select_variant(spec, category: str, modalities, *, ds_dir: Path | None = None):
    """The one variant of ``spec`` for ``category``, shared by
    :func:`inputs_for` and :func:`labels_for`.

    ``modalities`` (normalised tokens) select loosely: exact tokens first,
    then ``atac`` for ``atac_gas`` / ``atac_peak``. Without them, several
    variants are settled by the files in ``ds_dir`` (of nested role sets, the
    largest: :func:`_drop_nested`). ``KeyError`` when no variant fits,
    ``AmbiguousVariantError`` when several remain.
    """
    if modalities is not None:
        # 0.3's atac_gas of the vertical peak methods names their atac role now
        modalities = spec.legacy_modalities(category, modalities) or modalities
        v = spec.select(category, set(modalities), loose=True)
        if set(modalities) != set(v.when.get("modalities") or []):
            _check_representation(spec, modalities)
        return v
    candidates = [v for v in spec.variants if v.when.get("category") == category]
    if not candidates:
        from .schema import no_category_message
        raise KeyError(no_category_message(spec.id, spec.wired_categories, category))
    if len(candidates) == 1:
        return candidates[0]
    folder_note = ""
    if ds_dir is not None and Path(ds_dir).is_dir():
        ok = [v for v in candidates if _variant_satisfiable(v, Path(ds_dir), spec.id)]
        largest = _drop_nested(ok)
        if len(largest) == 1:
            return largest[0]
        if ok:
            which = ("both" if len(ok) == len(candidates) == 2 else
                     _and_list("+".join(v.when.get("modalities", [])) for v in ok))
            folder_note = f" The folder {ds_dir} has every input file of {which}."
        else:
            names = sorted(q.name for q in Path(ds_dir).glob("*"))
            holds = f"It holds {_and_list(names)}." if names else "It is empty."
            folder_note = (f" None of them has every input file in {ds_dir}. {holds}")
    raise AmbiguousVariantError(_ambiguous_message(spec.id, category, candidates,
                                                   folder_note))


def _drop_nested(variants) -> list:
    """``variants`` without those whose modality set is a strict subset of
    another's in the same category (scMoMaT's ``rna+adt`` inside its
    ``rna+adt+atac``)."""
    keys = [(v.when.get("category"), set(v.when.get("modalities") or [])) for v in variants]
    return [v for v, (cat, mods) in zip(variants, keys)
            if not any(cat == c and mods < m for c, m in keys)]


def _ambiguous_message(method: str, category: str, candidates, folder_note="") -> str:
    """The error for several ``category`` variants and no ``modalities=``;
    ``folder_note`` starts with a space."""
    mods = [list(v.when.get("modalities", [])) for v in candidates]
    names = _and_list("+".join(m) for m in mods)
    spell = [f"modalities={m}" for m in mods]
    pick = (f"Pass {spell[0]} or {spell[1]}." if len(spell) == 2 else
            f"Pass the modalities of one, for example {spell[0]}.")
    return f"{method} has {len(mods)} {category} variants, {names}.{folder_note} {pick}"


_READS = {"peak": "peaks", "gene_activity": "gene activity"}


def _check_representation(spec, modalities) -> None:
    """``KeyError`` when loosely matched representation tokens disagree with
    what the method reads (``method_info(m)['atac']``)."""
    reps = {{"atac_peak": "peak", "atac_gas": "gene_activity"}[t] for t in modalities
            if t in ("atac_peak", "atac_gas")}
    if not reps or not spec.atac:
        return
    reads = _READS.get(spec.atac, spec.atac)
    if len(reps) > 1:
        raise KeyError(
            f"{spec.id} reads {reads}; modalities name peaks and gene activity, which "
            f"only a variant that reads both files takes (see "
            f"mtb.method_info({spec.id!r})['supports'])")
    rep = next(iter(reps))
    if rep != spec.atac:
        raise KeyError(f"{spec.id} reads {reads}; modalities name {_READS[rep]}")


def missing_dataset_message(base, dataset: str) -> str:
    """The error for a dataset folder that does not exist, listing the
    folders ``base`` holds. A bare name that is a folder of the working directory,
    not of the default data path, gets the path to pass instead."""
    base = Path(base)
    here = Path(str(dataset))
    default = Path(os.path.abspath(config.data_path))
    near = next((p for p in (here, Path("data") / here) if p.is_dir()), None)
    if len(here.parts) == 1 and near is not None and Path(os.path.abspath(base)) == default:
        path = f"./{near}" if near == here else str(near)
        return (f"The folder {base / dataset} does not exist. {path} exists here: pass "
                + f"{path!r}.")
    dirs = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    holds = (f"{base} holds {_and_list(dirs)}." if dirs else
             f"{base} holds no folders." if base.is_dir() else
             f"{base} does not exist either.")
    return (f"The folder {base / dataset} does not exist. {holds} "
            + "dataset= is the folder's path, or its name under "
            + f"mtb.config.data_path. The folder layout is at {LAYOUT_URL}.")


def _note_local_folder(base: Path, name: str) -> None:
    """A warning, at every call site (the default filter shows each line once),
    when a bare dataset name resolves under the default data path while the
    working directory holds a folder of the same name; none for a folder the
    caller named by its path."""
    here = Path(name)
    if len(here.parts) != 1 or not here.is_dir():
        return
    if os.path.abspath(base) != os.path.abspath(config.data_path):
        return
    if os.path.abspath(here) == os.path.abspath(base / name):
        return
    # once per call: a call resolves the name several times. The call is its
    # outermost package frame, held so that no later call reuses its identity
    outer, frame = None, sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_compat._PACKAGE):
        outer, frame = frame, frame.f_back
    key = (os.path.abspath(base), name)
    if _NOTED[0] is outer and _NOTED[1] == key:
        return
    _NOTED[:] = [outer, key]
    path = f"./{name}"
    _compat.warn(f"Using {base / name}; {path} also exists: pass "
                 + f"{path!r} for it.", UserWarning)


#: the call that last got the note of :func:`_note_local_folder`
_NOTED: list = [None, None]


def canonical_dataset(base, dataset: str) -> str:
    """The spelling of ``dataset`` in the listing of ``base``, with a warning
    when it differs only in case: a case-insensitive file system opens
    ``d52`` for ``D52``, and ``'d52'`` would then name a second dataset in
    results. ``dataset`` as written when nothing else matches."""
    base = Path(base)
    name = str(dataset)
    try:
        listing = os.listdir(base)
    except OSError:
        return name
    if name in listing:
        _note_local_folder(base, name)
        return name
    same = [n for n in listing if n.lower() == name.lower() and (base / n).is_dir()]
    if len(same) != 1:
        return name
    _compat.warn(
        f"dataset {name!r} is not a folder under {base}, but {same[0]!r} is - using "
        f"that on-disk spelling (this file system ignores letter case; Linux does "
        f"not, and results, out_dir names and records carry {same[0]!r})",
        UserWarning)
    return same[0]


def split_dataset(dataset, data_path) -> tuple[str, Path | str | None]:
    """``(name, data_path)`` for ``dataset``: a folder name under ``data_path``, or
    a folder path (a ``Path``, or a str with a ``/``), relative to ``data_path``
    when it is given, else to the working directory."""
    if not isinstance(dataset, os.PathLike) and "/" not in str(dataset) \
            and os.sep not in str(dataset):
        return dataset, data_path
    folder = Path(os.path.abspath(Path(data_path or "") / dataset))
    return folder.name, folder.parent


def inputs_for(dataset: str, category: str, method: str, *,
               modalities: list[str] | set[str] | None = None,
               data_path: Path | str | None = None,
               check: bool | None = True) -> dict:
    """Return the input files a method reads from a dataset folder, by role.

    Paths are absolute, ready for ``runner.run(inputs=...)``. The files are
    checked before they are returned, so a long run does not start on a
    missing or malformed file; ``check=False`` only builds the paths (for
    another host).

    Parameters
    ----------
    dataset : str
        Dataset folder: a name under ``data_path`` such as ``"D11"``, or a path.
    category : str
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``.
    method : str
        Method id, e.g. ``"Matilda"``; see ``mtb.find_methods()``.
    modalities : list[str] | set[str] | None
        Modality tokens that pick the variant, e.g. ``["rna", "adt"]``;
        ``None`` = the files in the dataset folder decide.
    data_path : Path | str | None
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.data_path``.
    check : bool | None
        ``True`` (default) = raise on a missing or malformed input;
        ``False`` = no checks; ``None`` = warn about missing files.

    Returns
    -------
    dict
        ``{role: absolute path}``, one entry per input role of the selected
        variant (``data_dir`` for the directory-input methods).

    Raises
    ------
    TypeError
        ``category`` and ``method`` were passed in swapped order.
    KeyError
        Unknown method (the message names the closest method id), or no
        variant matches ``category`` and ``modalities``.
    ValueError
        Unknown category or modality token, or several variants fit
        (``AmbiguousVariantError``).
    ValueError
        ``check=True``: a transposed matrix, or label rows differ from the cell count.
    ValueError
        ``check=True``: files that must hold the same cells, in one order, do not.
    FileNotFoundError
        ``check=True``: an input file is missing, or ``data_dir`` lacks a file the method names.

    Warns
    -----
    UserWarning
        ``dataset`` matches a folder only up to letter case.
    UserWarning
        ``check=None`` and some resolved files do not exist.

    Examples
    --------
    >>> inputs_for("D11", "vertical", "Matilda")
    {'rna': '/abs/data/D11/rna.h5', 'adt': '/abs/data/D11/adt.h5',
     'cty': '/abs/data/D11/cty.csv'}
    """
    dataset, data_path = split_dataset(dataset, data_path)
    root = data_path if data_path is not None else config.data_path
    base = Path(os.path.abspath(os.fspath(root)))
    spec = registry.get(method)
    method = spec.id
    registry.check_category(category)
    dataset = canonical_dataset(base, dataset)
    ds_dir = base / dataset
    modalities = registry.normalize_modalities(modalities)
    variant = select_variant(spec, category, modalities, ds_dir=ds_dir)
    out = _resolve_variant_inputs(variant, ds_dir, method)
    missing = {r: p for r, p in out.items() if not Path(p).exists()}
    near = _near_miss_hints(ds_dir, missing, category, atac=spec.atac, method=method,
                            roles=out)
    batch_hint = _per_batch_hint(
        ds_dir, category, {st for r in missing for st in _role_stems(r)[0]})
    if batch_hint:
        near.append(batch_hint)
    if check:
        if missing:
            raise FileNotFoundError(_missing_inputs_text(method, category, ds_dir,
                                                         missing, near))
        _check_matrix_files(method, dataset, out)
        _check_orientation(method, dataset, out)
        _check_same_cells(method, dataset, out)
        if category == "diagonal":
            _check_atac_gas_cells(method, dataset, out)
        _check_label_lengths(category, out)
        if "data_dir" in out:
            ok, why = _check_data_dir(variant, out["data_dir"], method)
            if not ok:
                raise FileNotFoundError(why)
            named = _data_dir_files(variant, out["data_dir"])
            _check_matrix_files(method, dataset, named)
            if category == "diagonal":
                _check_atac_gas_cells(method, dataset, named)
                _check_diagonal_label_files(named)
    elif check is None and missing:
        _compat.warn(
            _missing_inputs_text(method, category, ds_dir, missing, near)
            + " Pass check=True to raise this as an error, or check=False to silence it.",
            UserWarning)
    return out


def _missing_inputs_text(method: str, category: str, ds_dir: Path, missing: dict,
                         hints: list) -> str:
    """The ``inputs_for`` text for missing input files: what the method needs,
    what the folder holds, then ``hints``."""
    ds_dir = Path(ds_dir)
    names = [Path(p).name if Path(p).parent == ds_dir else str(p)
             for p in dict.fromkeys(missing.values())]
    if ds_dir.is_dir():
        held = sorted(q.name for q in ds_dir.glob("*"))
        holds = f"The folder holds {_and_list(held)}." if held else "The folder is empty."
    else:
        holds = "The folder does not exist."
    return " ".join([f"{method} ({category}) needs {_and_list(names)} in {ds_dir}.",
                     holds, *hints])


# every stem an ATAC role may be looked up under, to name a near miss
_ATAC_FILE_BASES = ("atac", "atac_peak", "atac_gas", "peak")
#: the representation a sibling file holds, by its name (``spec.atac`` values)
_KIND_BY_BASE = {"atac_peak": "peak", "peak": "peak", "atac_gas": "gene_activity"}


def _near_miss_hints(ds_dir: Path, missing: dict, category: str,
                     atac: str | None = None, method: str | None = None,
                     roles=()) -> list[str]:
    """One hint per missing ATAC role with a sibling file present: the file
    names the category reads, and which ATAC the method needs. A vertical
    sibling of the method's representation (``atac``) gets the rename
    instead. ``roles`` with both ``atac_peak`` and ``atac_gas`` say the
    method reads both files."""
    hints: list[str] = []
    reads_both = {"atac_peak", "atac_gas"} <= set(roles)
    if not ds_dir.is_dir():
        return hints
    for role in missing:
        if base_modality(role) != "atac" or is_label_role(role):
            continue
        stems, _ = _role_stems(role)
        m = _NUMBERED_ROLE_RE.match(role)
        digits = m.group(2) if m else ""
        accepted = [f"{st}.h5" for st in stems]
        found = sorted(f"{b}{digits}.h5" for b in _ATAC_FILE_BASES
                       if f"{b}{digits}.h5" not in accepted
                       and (ds_dir / f"{b}{digits}.h5").is_file())
        if not found:
            continue
        same = [f for f in found if category == "vertical" and not m and atac
                and _KIND_BY_BASE.get(f[:-len(".h5")]) == atac]
        if same:
            # every vertical ATAC role reads atac.h5, also the peak methods' atac_gas
            hints.append(
                f"Vertical methods read atac.h5. Rename {same[0]} to atac.h5, or write "
                f"it with " + 'category="vertical"'
                + ".")
            continue
        who = method or "the method"
        if m:
            why = "Every mosaic method reads peak ATAC."
        elif reads_both:
            why = f"{who[:1].upper()}{who[1:]} reads both atac_peak.h5 and atac_gas.h5."
        else:
            why = ((f'method_info("{method}")["atac"]' if method else 'method_info(m)["atac"]')
                   + f" says which ATAC {who} needs.")
        rule = f"{category.capitalize()} methods read {' or '.join(accepted)}."
        if category == "vertical" and not m:
            rule = "Vertical methods read atac.h5."
        hints.append(f"{rule} {why}")
    return hints


@functools.lru_cache(maxsize=512)
def _sniff_h5(path: str, mtime_ns: int):
    """``(shape, n_features, n_cells)`` of a canonical .h5, or None; cached by
    mtime, as scan reads the same files once per method row."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "matrix/data" not in f:
                return None
            shape = tuple(f["matrix/data"].shape)
            if len(shape) != 2:
                return None
            if "matrix/features" not in f or "matrix/barcodes" not in f:
                return None
            return (shape, int(f["matrix/features"].shape[0]),
                    int(f["matrix/barcodes"].shape[0]))
    except OSError:
        return None


def _check_matrix_files(method, dataset, resolved):
    """Reject a 10x Genomics file, or a matrix with 0 features or 0 cells,
    under the name of a dataset file. Other non-canonical files pass here;
    ``mtb.run`` refuses them."""
    from .ingest import _is_canonical_h5, _is_tenx_h5
    for path in dict.fromkeys(resolved.values()):
        p = Path(path)
        if p.suffix != ".h5" or not p.is_file():
            continue
        if not _is_canonical_h5(p) and _is_tenx_h5(p):
            raise ValueError(
                f"{method} reads {p.name} of {dataset}, which is a 10x Genomics file. "
                f'See the recipe "A 10x .h5 file" in the docs.')
        sniff = _sniff_h5(str(p), p.stat().st_mtime_ns)
        if sniff is not None and 0 in sniff[1:]:
            again = "mtb.io.export_dataset"
            raise ValueError(
                f"{method} reads {p.name} of {dataset}, which holds 0 "
                f"{'features' if sniff[1] == 0 else 'cells'}. Write the folder again "
                f"with {again}.")


def _check_orientation(method, dataset, resolved):
    """Reject a ``matrix/data`` stored cells x features before the run, which
    would otherwise fail late inside the method. A square matrix is left alone."""
    for path in resolved.values():
        p = Path(path)
        if p.suffix != ".h5" or not p.is_file():
            continue
        sniff = _sniff_h5(str(p), p.stat().st_mtime_ns)
        if sniff is None:
            continue
        shape, n_feat, n_cell = sniff
        if n_feat == n_cell or shape == (n_feat, n_cell):
            continue                  # ambiguous, or already correct
        if shape == (n_cell, n_feat):
            raise ValueError(
                f"{method} reads {p.name} of {dataset}, which stores matrix/data as "
                f"cells x features, shape {shape}. The file lists {n_feat} features "
                f"and {n_cell} cells. Its matrix/data needs features x cells, shape "
                f"({n_feat}, {n_cell}). Export it again with "
                + "mtb.io.export_dataset"
                + ", or transpose matrix/data."
            )


def _read_features(path: str, n: int | None = None) -> tuple:
    """The first ``n`` (``None``: all) feature names of a canonical .h5, or ``()``,
    also for the ``matrix/features`` group of a 10x Genomics file."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            feats = f.get("matrix/features")
            if not isinstance(feats, h5py.Dataset):
                return ()
            raw = feats[:n]
    except OSError:
        return ()
    return tuple(x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in raw)


@functools.lru_cache(maxsize=512)
def _sniff_features(path: str, mtime_ns: int, n: int = 50) -> tuple:
    """:func:`_read_features` of the first ``n``, cached by mtime."""
    return _read_features(path, n)


@functools.lru_cache(maxsize=512)
def _count_label_rows(path: str, mtime_ns: int):
    """Number of label rows in a label CSV (header excluded), cached by mtime;
    ``None`` when the file cannot be parsed."""
    import pandas as pd

    try:
        return int(len(pd.read_csv(path, usecols=[0])))
    except Exception:
        return None


def _label_partners(label_role: str, roles) -> list[str]:
    """The modality roles ``label_role`` labels: ``cty`` every one, ``cty<N>``
    those numbered ``<N>``, ``<mod>_cty[<N>]`` those of that base modality
    (and number), anything else none."""
    mods = [r for r in roles if not is_label_role(r) and r != "data_dir"]
    if label_role == "cty":
        return mods
    digits = "".join(ch for ch in label_role if ch.isdigit())
    if label_role.startswith("cty") and digits:
        return [r for r in mods if r.endswith(digits)]
    m = _MODALITY_LABEL_RE.match(label_role)
    if m:
        prefix, number = m.groups()
        base = {"peak": "atac"}.get(prefix, prefix)
        batch = int(number) if number else None
        return [r for r in mods if base_modality(r) == base and _batch_of(r) == batch]
    return []


_BATCH_DIGITS_RE = re.compile(r"(\d+)$")
#: label stems named after the modality they label: ``rna_cty``, ``atac_cty``,
#: ``peak_cty``, and per batch ``rna_cty2``, ``atac_cty2``
_MODALITY_LABEL_RE = re.compile(r"^(.+)_cty(\d*)$")


def _batch_label_file(role: str, path, category: str | None = None) -> tuple[str, Path] | None:
    """``(batch, label file)`` for a numbered modality role, else ``None``:
    ``cty<i>.csv``, or ``<mod>_cty<i>.csv`` for diagonal. No method reads it
    as a role, but every evaluation does."""
    if is_label_role(role) or role == "data_dir":
        return None
    m = _BATCH_DIGITS_RE.search(role)
    if not m:
        return None
    p = Path(path)
    if p.suffix != ".h5":
        return None
    if category == "diagonal":
        return m.group(1), p.parent / f"{base_modality(role)}_cty{m.group(1)}.csv"
    return m.group(1), p.parent / f"cty{m.group(1)}.csv"


def _check_label_count(labels: Path, data: Path, *, batch=None) -> None:
    """``ValueError`` when the label file ``labels`` has another number of rows
    than ``data`` has cells; unreadable files and square matrices pass."""
    if not labels.is_file() or data.suffix != ".h5" or not data.is_file():
        return
    n_lab = _count_label_rows(str(labels), labels.stat().st_mtime_ns)
    sniff = _sniff_h5(str(data), data.stat().st_mtime_ns)
    if n_lab is None or sniff is None:
        return
    _, n_feat, n_cell = sniff
    if n_feat != n_cell and n_lab != n_cell:
        raise ValueError(_label_count_text(labels.name, n_lab, data.name, n_cell,
                                           batch=batch))


def _check_label_lengths(category, resolved):
    """Reject a label file whose row count differs from the cells it labels,
    which ``evaluate`` would refuse only after the run: the label roles
    (:func:`_label_partners`), and the numbered label files of each batch
    (:func:`_batch_label_file`)."""
    for role, path in resolved.items():
        pair = _batch_label_file(role, path, category)
        if pair is not None and pair[1].stem not in resolved:   # else a role, below
            _check_label_count(pair[1], Path(path), batch=pair[0])
    if category == "diagonal":
        _check_diagonal_label_files(resolved)
    for role, path in resolved.items():
        if is_label_role(role) and Path(path).suffix == ".csv":
            for partner in _label_partners(role, resolved):
                _check_label_count(Path(path), Path(resolved[partner]))


def _check_diagonal_label_files(resolved):
    """Diagonal: ``rna_cty.csv`` / ``atac_cty.csv`` need one row per cell of
    ``rna.h5`` / the ATAC file, also when the variant does not read them.
    Numbered roles are left to :func:`_check_label_lengths`."""
    for role, path in resolved.items():
        if is_label_role(role) or role == "data_dir" or role[-1:].isdigit():
            continue
        base = base_modality(role)
        label_role = f"{base}_cty"
        if base in ("rna", "atac") and label_role not in resolved:
            _check_label_count(Path(path).parent / f"{label_role}.csv", Path(path))


def _label_count_text(labels: str, n_lab: int, data: str, n_cell: int, *,
                      batch=None) -> str:
    """The error of a label file whose length is not its data file's cell count."""
    each = f"each cell of batch {batch}" if batch is not None else "each cell"
    return (f"{labels} has {n_lab} labels, but {data} has {n_cell} cells. Give {each} "
            f"one label, in the order of the cells. The folder layout is at {LAYOUT_URL}.")


def _data_dir_files(variant, data_dir) -> dict:
    """``{stem: path}`` of the ``.h5`` files a ``data_dir`` method names as
    ``const`` args (scBridge: ``rna.h5``, ``atac_gas.h5``)."""
    d = Path(data_dir)
    return {Path(str(a.const)).stem: str(d / str(a.const)) for a in variant.args
            if a.const and str(a.const).endswith(".h5")}


def _check_data_dir(variant, data_dir, method: str = "The method") -> tuple[bool, str]:
    """``(ok, why)``: the ``data_dir`` holds every ``.h5`` / ``.csv`` file the
    method names as ``const`` args (the folder itself always exists). ``why``
    starts with ``method``."""
    d = Path(data_dir)
    if not d.is_dir():
        return False, f"{method} needs the folder {d}, which does not exist."
    needed = [a.const for a in variant.args if a.const and str(a.const).endswith((".h5", ".csv"))]
    missing = [str(f) for f in needed if not (d / f).exists()]
    if len(missing) == 1:
        return False, f"{method} needs {missing[0]}, and {d} has no such file."
    if missing:
        return False, f"{method} needs {_and_list(missing)}, and {d} has none of them."
    return True, ""


# Every caveat starts with the method name, so a clipped line still shows the problem.
#: an ``atac_gas`` role fell back to a peak matrix in ``atac.h5``
PEAK_IN_GAS_CAVEAT = ("{method} needs gene-activity ATAC. The features of atac.h5 look "
                      "like chr:start-end, so it holds peaks.")
#: the representation-mismatch caveats, when the method's ATAC is known
PEAK_FED_TO_GAS_CAVEAT = ("{method} needs gene-activity ATAC. The features of {file} "
                          "look like chr:start-end, so it holds peaks.")
GAS_FED_TO_PEAK_CAVEAT = ("{method} needs peak ATAC. The features of {file} do not look "
                          "like chr:start-end, so it seems to hold gene activity.")
#: names that are neither peaks nor the folder's genes (``peak_0``): the kind
#: cannot be told, so these two do not block the row
PEAK_NAMES_UNKNOWN_CAVEAT = ("{method} needs peak ATAC. {file} holds names such as "
                             "{example}, not chr:start-end.")
GAS_NAMES_UNKNOWN_CAVEAT = ("{method} needs gene-activity ATAC. {file} holds names such "
                            "as {example}, not the RNA's genes.")
#: peak names that ``mtb.run``'s rewrite to chr:start-end cannot fix
PEAK_NAMES_CAVEAT = ("{method} reads peak names such as chr1:100-200. {file} holds other "
                     "names, for example {example}. {action}")
#: the action of :data:`PEAK_NAMES_CAVEAT`: for other names, and for gene names
RENAME_PEAKS = "Rename them to chr:start-end."
GENES_AS_PEAKS = "They are genes: export the ATAC as gene activity."
NOT_COUNTS_CAVEAT = "{method} needs raw counts. {file} holds non-integer values."
DIAGONAL_CTY_CAVEAT = ("{method} needs rna_cty.csv and atac_cty.csv for diagonal. The "
                       "folder has only cty.csv.")
#: ``UINMF reads batches 1-2 of 3. Batch 3 is not used.``
UNUSED_BATCHES_CAVEAT = "{method} reads batches {used} of {n}. {unused}."
# file stems (batch digits allowed) whose values must be raw counts
_COUNT_FILE_RE = re.compile(r"^(rna|adt|atac_peak)\d*$")


@functools.lru_cache(maxsize=512)
def _h5_has_fraction(path: str, mtime_ns: int) -> bool | None:
    """Raw-count check of a canonical .h5 from a few chunks (cached by mtime):
    ``True`` for non-whole values, ``None`` when the file cannot be read."""
    import h5py
    import numpy as np

    from .ingest import _COUNT_SAMPLE, _has_fraction

    try:
        with h5py.File(path, "r") as f:
            if "matrix/data" not in f:
                return None
            d = f["matrix/data"]
            if d.ndim != 2 or 0 in d.shape:
                return None
            if d.dtype.kind != "f":
                return False
            n_rows, n_cols = d.shape
            chunk_rows, chunk_cols = d.chunks or (max(1, min(n_rows, 200_000 // n_cols)),
                                                  n_cols)
            vals, got = [], 0
            for r, c in zip(np.linspace(0, n_rows - 1, 4).astype(int),
                            np.linspace(0, n_cols - 1, 4).astype(int)):
                r0, c0 = (r // chunk_rows) * chunk_rows, (c // chunk_cols) * chunk_cols
                x = np.asarray(d[r0:r0 + chunk_rows, c0:c0 + chunk_cols]).ravel()
                x = x[x != 0]
                vals.append(x)
                got += x.size
                if got >= _COUNT_SAMPLE:
                    break
            return _has_fraction(np.concatenate(vals))
    except (OSError, KeyError, ValueError):
        return None


def _peak_fraction_of(path: Path) -> float | None:
    """Share of the first 50 feature names that look like ``chr:start-end``;
    ``None`` when the file is not a readable canonical ``.h5``."""
    from .ingest import _PEAK_RE

    if path.suffix != ".h5" or not path.is_file():
        return None
    feats = _sniff_features(str(path), path.stat().st_mtime_ns)
    if not feats:
        return None
    return sum(1 for x in feats if _PEAK_RE.match(x)) / len(feats)


@functools.lru_cache(maxsize=64)
def _all_features(path: str, mtime_ns: int) -> tuple:
    """:func:`_read_features` of every feature, cached by mtime."""
    return _read_features(path)


#: Ensembl gene ids (``ENSG00000141510``, ``ENSMUSG...``)
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}")


def _names_are_genes(path: Path) -> bool | None:
    """Whether the names of a non-peak ATAC file are gene names: mostly
    Ensembl ids, or at least 1% of a sample found (any case) among the
    features of the folder's other ``rna*`` / ``atac_gas*`` files. ``None``
    when there is no file to compare with."""
    def _sample(names):
        return names[::max(1, len(names) // 2000)][:2000]

    def _ensembl(names):
        return sum(1 for x in names if _ENSEMBL_RE.match(x)) > 0.5 * len(names)

    sample = _sample(_all_features(str(path), path.stat().st_mtime_ns))
    if not sample:
        return None
    if _ensembl(sample):
        return True
    known: set[str] = set()
    for q in sorted(path.parent.glob("*.h5")):
        if q.resolve() == path.resolve() or not re.match(r"^(rna|atac_gas)\d*$", q.stem):
            continue
        names = _all_features(str(q), q.stat().st_mtime_ns)
        if names and not _ensembl(_sample(names)):
            known.update(x.upper() for x in names)
    if not known:
        return None
    hits = sum(1 for x in sample if x.upper() in known)
    return hits >= max(1, 0.01 * len(sample))


def _unrewritable_peak_name(path: Path) -> str | None:
    """The first of the first 50 feature names ``ingest._normalize_peak_names`` cannot
    rewrite, when more than 10% are such names; else ``None``."""
    from .ingest import _PEAK_RE

    if path.suffix != ".h5" or not path.is_file():
        return None
    feats = _sniff_features(str(path), path.stat().st_mtime_ns)
    bad = [x for x in feats or [] if not _PEAK_RE.match(x)]
    return bad[0] if feats and len(bad) > 0.1 * len(feats) else None


def _renamed_peak_roles(method: str | None, category: str | None, resolved) -> list[str]:
    """The roles of ``resolved`` whose peak names ``mtb.run`` rewrites
    (``Variant.normalize_peaks``); ``[]`` when the variant is not known."""
    if not method or not category:
        return []
    try:
        mods = {r for r in resolved if not is_label_role(r) and r not in _NON_MODALITY_ROLES}
        variant = registry.get(method).select(category, mods)
    except KeyError:    # no such method or variant
        return []
    return [r for r in variant.normalize_peaks if r in resolved]


#: methods whose script needs the same cells in two roles: Seurat_v5's bridge
#: (``obj.multi[["ATAC"]] <- ...`` accepts only the cells the object holds)
_SAME_CELL_ROLES = {"Seurat_v5": ("rna", "atac_peak")}
SAME_CELLS_REASON = ("{method} needs RNA and ATAC from the same cells as its bridge. In "
                     "{dataset}, {a} and {b} share {shared:,} of {n_a:,} and {n_b:,} cells.")
GAS_OTHER_CELLS_REASON = ("{method} reads {gas} of {dataset}, which holds other cells than "
                          "{peak}. The files hold {n_gas:,} and {n_peak:,} cells and share "
                          "{shared:,}. Both files need the same ATAC cells.")
GAS_OTHER_ORDER_REASON = ("{method} reads {gas} of {dataset}, which lists the ATAC cells in "
                          "another order than {peak}. atac_cty.csv follows {peak}. Write "
                          "{gas} again with {write}.")


@functools.lru_cache(maxsize=64)
def _sniff_barcode_list(path: str, mtime_ns: int) -> tuple | None:
    """The ``matrix/barcodes`` of a canonical .h5 (cached by mtime), or None."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "matrix/barcodes" not in f:
                return None
            raw = f["matrix/barcodes"][:]
    except OSError:
        return None
    return tuple(x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in raw)


def _barcodes_of(path) -> tuple | None:
    """Barcodes of a readable canonical ``.h5``, else None."""
    p = Path(path)
    if p.suffix != ".h5" or not p.is_file():
        return None
    return _sniff_barcode_list(str(p), p.stat().st_mtime_ns)


def _check_same_cells(method, dataset, resolved) -> None:
    """``ValueError`` when ``method`` needs two roles to hold the same cells
    (:data:`_SAME_CELL_ROLES`) and their barcode sets differ."""
    roles = _SAME_CELL_ROLES.get(method or "")
    if not roles or not all(r in resolved for r in roles):
        return
    bars = [_barcodes_of(resolved[r]) for r in roles]
    if any(b is None for b in bars):
        return
    a, b = (set(x) for x in bars)
    if a == b:
        return
    raise ValueError(SAME_CELLS_REASON.format(
        method=method, dataset=dataset, a=Path(resolved[roles[0]]).name,
        b=Path(resolved[roles[1]]).name, n_a=len(a), n_b=len(b), shared=len(a & b)))


def _check_atac_gas_cells(method, dataset, resolved) -> None:
    """Diagonal: each gene-activity file must list the cells of its peak file
    (the ``atac_peak[<i>]`` role, else the file next to it) in that order,
    as ``atac_cty[<i>].csv`` labels the peak file's cells. Compared as
    ``ingest._cell_order`` compares (D28's ATAC barcodes end in -1 and -2)."""
    from .ingest import _cell_order

    for role, gas in resolved.items():
        digits = role[len("atac_gas"):]
        if not role.startswith("atac_gas") or (digits and not digits.isdigit()) or not gas:
            continue
        gas = Path(gas)
        given = resolved.get(f"atac_peak{digits}")
        if given and Path(given).is_file():
            peak = Path(given)
        else:
            peak = next((gas.parent / n for n in (f"atac_peak{digits}.h5", f"peak{digits}.h5")
                         if (gas.parent / n).is_file()), None)
        if peak is None or peak.resolve() == gas.resolve():
            continue
        a, b = _barcodes_of(gas), _barcodes_of(peak)
        if a is None or b is None:
            continue
        kind, _ = _cell_order(a, b)
        if kind == "other":
            raise ValueError(GAS_OTHER_CELLS_REASON.format(
                method=method, dataset=dataset, gas=gas.name, peak=peak.name, n_gas=len(a),
                n_peak=len(b), shared=len(set(a) & set(b))))
        if kind == "order":
            raise ValueError(GAS_OTHER_ORDER_REASON.format(
                method=method, dataset=dataset, gas=gas.name, peak=peak.name,
                write="mtb.io.export_dataset(..., atac_kind='gene_activity')"
            ).replace("atac_cty.csv", f"atac_cty{digits}.csv"))


#: file names that carry a batch number: ``rna2.h5``, ``atac_peak3.h5``,
#: ``cty1.csv``, and the diagonal ``rna_cty2.csv`` / ``atac_cty2.csv``
_BATCH_FILE_RE = re.compile(r"^(?:rna|adt|atac|atac_peak|atac_gas)(\d+)\.h5$"
                            r"|^(?:rna_|atac_)?cty(\d+)\.csv$")
#: the files of the dataset layout: the names of ``_BATCH_FILE_RE``, and the
#: same names without a batch number (``rna.h5``, ``atac_gas.h5``, ``cty.csv``)
_LAYOUT_FILE_RE = re.compile(r"^(?:rna|adt|atac|atac_peak|atac_gas)(\d*)\.h5$"
                             r"|^(?:rna_|atac_)?cty(\d*)\.csv$")
#: role tokens that carry a batch number
_BATCH_ROLE_RE = re.compile(r"^(?:rna|adt|atac|atac_peak|atac_gas|cty|rna_cty|atac_cty)(\d+)$")


def _variant_batches(roles) -> set[int]:
    """Batch numbers named by numbered roles (``rna1``, ``adt2``, ``cty3``)."""
    return {int(m.group(1)) for r in roles if (m := _BATCH_ROLE_RE.match(str(r)))}


def _folder_batches(ds_dir: Path) -> set[int]:
    """Batch numbers of the numbered modality and label files in ``ds_dir``."""
    out = set()
    for p in Path(ds_dir).iterdir():
        m = _BATCH_FILE_RE.match(p.name)
        if m:
            out.add(int(m.group(1) or m.group(2)))
    return out


def _span(nums) -> str:
    """``[1, 2]`` -> ``'1-2'``; ``[1, 3]`` -> ``'1 and 3'``; ``[3]`` -> ``'3'``."""
    nums = sorted(nums)
    if len(nums) > 1 and nums == list(range(nums[0], nums[-1] + 1)):
        return f"{nums[0]}-{nums[-1]}"
    if len(nums) > 2:
        return ", ".join(map(str, nums[:-1])) + f" and {nums[-1]}"
    return " and ".join(map(str, nums))


def _unused_batches_note(resolved, method: str = "The method") -> str | None:
    """:data:`UNUSED_BATCHES_CAVEAT` when the numbered roles of ``resolved``
    name fewer batches than the folder holds (UINMF reads batches 1-2 of a
    3-batch cross folder), else None."""
    numbered = [(r, p) for r, p in resolved.items() if _BATCH_ROLE_RE.match(r)]
    if not numbered:
        return None
    ds_dir = Path(numbered[0][1]).parent
    if not ds_dir.is_dir():
        return None
    used = _variant_batches(r for r, _ in numbered)
    held = _folder_batches(ds_dir) | used
    unused = sorted(held - used)
    if not unused:
        return None
    verb = "Batch {} is not used" if len(unused) == 1 else "Batches {} are not used"
    return UNUSED_BATCHES_CAVEAT.format(method=method, used=_span(used), n=len(held),
                                        unused=verb.format(_span(unused)))


def _preflight_caveats(resolved, *, atac: str | None = None,
                       category: str | None = None,
                       method: str | None = None) -> list[str]:
    """Non-fatal observations about resolved inputs; each starts with
    ``method`` (``"The method"`` without it).

    In order: numbered roles that leave batches of the folder unused; peak
    names ``mtb.run``'s rewrite cannot fix (with ``method`` and
    ``category``); the ATAC representation, judged per role against
    ``atac`` (``method_info(m)['atac']``; MultiMAP and Seurat_v3 read peaks
    from ``atac_peak`` and gene activity from ``atac_gas``) by the share of
    peak-like names, >= 90% or <= 10% (without ``atac``, only an
    ``atac_gas`` role read from ``atac.h5``); values that are not whole
    numbers; a diagonal folder whose only label file is ``cty.csv``.
    """
    out: list[str] = []
    who = method or "The method"
    note = _unused_batches_note(resolved, who)
    if note:
        out.append(note)
    renamed = _renamed_peak_roles(method, category, resolved)
    for role in renamed:
        bad = _unrewritable_peak_name(Path(resolved[role]))
        if bad is not None:
            # gene activity exported as peaks: renaming is the wrong action
            genes = _names_are_genes(Path(resolved[role]))
            out.append(PEAK_NAMES_CAVEAT.format(
                method=who, file=Path(resolved[role]).name, example=bad,
                action=GENES_AS_PEAKS if genes else RENAME_PEAKS))
    if atac is None:
        p = Path(resolved.get("atac_gas", ""))
        if p.name and p.stem != "atac_gas":
            frac = _peak_fraction_of(p)
            if frac is not None and frac >= 0.9:
                out.append(PEAK_IN_GAS_CAVEAT.format(method=who))
    else:
        stems = {r.rstrip("0123456789") for r in resolved}
        both = {"atac_peak", "atac_gas"} <= stems
        for role, path in resolved.items():
            if is_label_role(role) or base_modality(role) != "atac" or role in renamed:
                continue
            frac = _peak_fraction_of(Path(path))
            if frac is None:
                continue
            want = atac
            if both and role.rstrip("0123456789") in ("atac_peak", "atac_gas"):
                want = "peak" if role.startswith("atac_peak") else "gene_activity"
            if want == "gene_activity" and frac >= 0.9:
                out.append(PEAK_FED_TO_GAS_CAVEAT.format(method=who, file=Path(path).name))
            elif frac <= 0.1:
                # gene activity only when the names are the folder's genes
                genes = _names_are_genes(Path(path))
                if genes is False:
                    first = _sniff_features(str(path), Path(path).stat().st_mtime_ns)
                    unknown = (PEAK_NAMES_UNKNOWN_CAVEAT if want == "peak"
                               else GAS_NAMES_UNKNOWN_CAVEAT)
                    out.append(unknown.format(method=who, file=Path(path).name,
                                              example=first[0]))
                elif want == "peak":
                    out.append(GAS_FED_TO_PEAK_CAVEAT.format(method=who,
                                                             file=Path(path).name))
    seen, fractions = set(), []
    for role, path in resolved.items():
        p = Path(path)
        if is_label_role(role) or role == "data_dir" or p in seen:
            continue
        seen.add(p)
        if p.suffix != ".h5" or not _COUNT_FILE_RE.match(p.stem) or not p.is_file():
            continue
        if _h5_has_fraction(str(p), p.stat().st_mtime_ns):
            fractions.append(p.name)
    if fractions:
        # one sentence for the files of several batches
        out.append(NOT_COUNTS_CAVEAT.format(method=who, file=_and_list(fractions)).replace(
            " holds ", " hold " if len(fractions) > 1 else " holds "))
    if category == "diagonal" and resolved:
        ds_dir = next(Path(v) if k == "data_dir" else Path(v).parent
                      for k, v in resolved.items())
        if (ds_dir / "cty.csv").is_file() \
                and not (ds_dir / "rna_cty.csv").is_file() \
                and not (ds_dir / "atac_cty.csv").is_file():
            out.append(DIAGONAL_CTY_CAVEAT.format(method=who))
    return out


_LABEL_MODALITY_ORDER = {"rna": 0, "adt": 1, "atac": 2}


def _label_sort_key(stem: str):
    """Sort key of the default cell-stacking order of label stems: ``cty``;
    ``cty<N>`` by number; ``<mod>_cty[<N>]`` by modality (rna, adt, atac;
    ``peak`` is atac), then number; anything else alphabetically."""
    if stem == "cty":
        return (0, 0, "")
    digits = "".join(ch for ch in stem if ch.isdigit())
    if stem.startswith("cty") and digits and stem == f"cty{digits}":
        return (1, int(digits), "")
    m = _MODALITY_LABEL_RE.match(stem)
    if m:
        prefix, number = m.groups()
        base = {"peak": "atac"}.get(prefix, base_modality(prefix))
        return (2, _LABEL_MODALITY_ORDER.get(base, 9), int(number or 0), stem)
    return (3, 0, stem)


def _variant_label_rank(stems: list[str], variant) -> dict[str, tuple] | None:
    """Rank label stems by where their cells sit in ``variant``'s output
    (``Variant.stacked_roles``); ``None`` when no stem pairs with a role."""
    mods = variant.stacked_roles()
    rank: dict[str, tuple] = {}
    for stem in stems:
        partners = _label_partners(stem, mods)
        pos = [mods.index(r) for r in partners if r in mods]
        if pos:
            rank[stem] = (0, min(pos), stem)
    if not rank:
        return None
    for stem in stems:
        rank.setdefault(stem, (1,) + _label_sort_key(stem))
    return rank


_NUMBERED_LABEL_RE = re.compile(r"(?:(?:rna|atac)_)?cty(\d+)")


def _agreed_label_order(spec, category: str, stems: list[str]) -> list[str] | None:
    """The label order the ``category`` variants reading exactly the batches
    of ``stems`` agree on (StabMap's two three-batch cross variants), else
    ``None``."""
    batches = {int(m.group(1)) for st in stems if (m := _NUMBERED_LABEL_RE.fullmatch(st))}
    orders = set()
    for v in spec.variants:
        if v.when.get("category") != category:
            continue
        roles = {r for a in v.args for r in (a.roles or [a.role]) if r}
        if _variant_batches(roles) != batches:
            continue
        rank = _variant_label_rank(stems, v)
        orders.add(tuple(stems) if rank is None
                   else tuple(sorted(stems, key=lambda st: rank[st])))
    return list(orders.pop()) if len(orders) == 1 else None


#: the start of the warning of ``labels_for`` without a method (internal
#: callers that want the default order filter it with this pattern)
DEFAULT_ORDER_WARNING = r"labels_for\(.*\) returns the files in the default order"


def _reordering_methods(stems: list[str], category: str | None, dataset: str,
                        base: Path) -> list[tuple[str, str]]:
    """``(method, category)`` of the methods whose files the folder holds (the
    file check of ``mtb.scan``) and whose ``labels_for(dataset, category,
    method)`` puts the label files ``stems`` in another order than the
    default one (the order of ``stems``); of ``category`` when given."""
    batches = {int(m.group(1)) for st in stems if (m := _NUMBERED_LABEL_RE.fullmatch(st))}
    ds_dir = base / dataset
    cands: dict[tuple[str, str], list] = {}
    for spec in registry.load():
        for v in spec.variants:
            cat = v.when.get("category")
            if category is not None and cat != category:
                continue
            roles = {r for a in v.args for r in (a.roles or [a.role]) if r}
            used = _variant_batches(roles)
            if batches and used and used != batches:
                continue
            rank = _variant_label_rank(stems, v)
            # every file pairs with an input of the variant: it reads this folder
            if rank is None or any(rank[st][0] != 0 for st in stems):
                continue
            if sorted(stems, key=lambda st: rank[st]) != list(stems) \
                    and _variant_satisfiable(v, ds_dir, spec.id):
                cands.setdefault((spec.id, cat), []).append(v)
    out = []
    with warnings.catch_warnings():       # the calls below only probe the folder
        warnings.simplefilter("ignore")
        for (method, cat), variants in cands.items():
            try:
                order = list(labels_for(dataset, cat, method, data_path=base, check=False))
            except Exception:  # noqa: BLE001 - no order to compare
                continue
            if sorted(order) != sorted(stems) or order == list(stems):
                continue
            for mods in [None] + [v.when.get("modalities") for v in variants]:
                try:
                    inputs_for(dataset, cat, method, modalities=mods, data_path=base,
                               check=True)
                except AmbiguousVariantError:
                    continue
                except Exception:  # noqa: BLE001 - the files fail scan's file check
                    break
                out.append((method, cat))
                break
    return out


#: (dataset folder, category) pairs labels_for already warned about
_ORDER_NOTED: set = set()


def _folder_category(ds_dir: Path) -> str | None:
    """The category a numbered folder's layout reads as: ``cross`` when every
    batch holds the same modalities, else ``mosaic``; ``None`` otherwise."""
    batches: dict = {}
    for p in Path(ds_dir).iterdir():
        m = re.fullmatch(r"(rna|adt|atac|atac_peak)(\d+)\.h5", p.name)
        if m:
            batches.setdefault(int(m.group(2)), set()).add(
                "atac" if m.group(1).startswith("atac") else m.group(1))
    if len(batches) < 2:
        return None
    return "cross" if len({frozenset(v) for v in batches.values()}) == 1 else "mosaic"


def _default_order_warning(dataset: str, category, stems: list[str],
                           base: Path) -> str | None:
    """The warning of ``labels_for`` without a method when a method whose files
    the folder holds reads these label files in another order, else ``None``."""
    if len(stems) < 2:
        return None
    others = _reordering_methods(stems, category, dataset, base)
    if not others:
        return None
    by_cat: dict[str, list[str]] = {}
    for m, cat in others:
        by_cat.setdefault(cat, []).append(m)
    if category is None and len(by_cat) > 1:
        # the dataset's own category only: a cross folder is not a mosaic one
        own = _folder_category(base / dataset)
        if own in by_cat:
            by_cat = {own: by_cat[own]}
    call = (f"labels_for({dataset!r})" if category is None
            else f"labels_for({dataset!r}, {category!r})")
    head = f"{call} returns the files in the default order {', '.join(stems)}."

    def who(names):
        return f"{_and_list(names)} {'reads' if len(names) == 1 else 'read'}"
    if len(by_cat) == 1:
        (cat, names), = by_cat.items()
        where = "" if category is not None else f"In {cat}, "
        doer = who(names)
        method = repr(names[0]) if len(names) == 1 else "<method>"
        whose = "its" if len(names) == 1 else "their"
        fix = (f"mtb.evaluate(..., dataset={dataset!r}, category={cat!r}, "
                          f"method={method})")
        return f"{head} {where}{doer} them in another order: pass {fix} for {whose} output."
    parts = [f"in {cat}, {who(names)} them in another order" if i == 0
             else f"in {cat}, {_and_list(names)} {'does' if len(names) == 1 else 'do'}"
             for i, (cat, names) in enumerate(by_cat.items())]
    text = "; ".join(parts)
    return (f"{head} {text[0].upper()}{text[1:]}: pass mtb.evaluate(..., dataset="
            f"{dataset!r}, category=<category>, method=<method>) for their output.")


class LabelFiles(dict):
    """The ``{stem: path}`` dict :func:`labels_for` returns: a ``dict`` that
    remembers its key order (``stacking_order``), so ``evaluate`` takes it as
    is even when that is not the default order. Reordered keys, or a
    ``dict(...)`` copy, are held to the default order like any dict.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stacking_order = tuple(self)

    def in_stacking_order(self) -> bool:
        """True while the keys are exactly the ones ``labels_for`` returned, in its order."""
        return tuple(self) == self.stacking_order


def labels_for(dataset: str, category: str | None = None, method: str | None = None,
               *, modalities: list[str] | set[str] | None = None,
               data_path: Path | str | None = None,
               check: bool | None = None) -> dict:
    """Return a dataset's cell-type label files, in the method's cell order.

    Hand the dict to ``mtb.evaluate(labels=...)`` as is. It matches an
    embedding only when the embedding's cell order is the dict's order, so
    give ``category`` and ``method`` for a method-specific order.

    Parameters
    ----------
    dataset : str
        Dataset folder: a name under ``data_path`` such as ``"D28"``, or a path.
    category : str | None
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``; with ``method``, selects the variant whose cell order is
        used. ``None`` = the default order.
    method : str | None
        Method id; with ``category``, orders the files in that
        variant's cell order. ``None`` = the default order.
    modalities : list[str] | set[str] | None
        Modality tokens that pick one of several variants; used only with
        ``category`` and ``method``.
    data_path : Path | str | None
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.data_path``.
    check : bool | None
        Vertical or diagonal ``category`` on a per-batch folder: ``None``
        warns, ``True`` raises, ``False`` = no check.

    Returns
    -------
    dict
        ``{stem: absolute path}`` in the method's cell order, keyed by
        filename stem (``cty``, ``rna_cty``, ``cty1`` ...).

    Raises
    ------
    TypeError
        ``category`` and ``method`` swapped, or a path passed positionally as ``category``.
    FileNotFoundError
        ``<data_path>/<dataset>`` does not exist.
    ValueError
        Unknown ``category``; or ``check=True`` and a per-batch folder for vertical / diagonal.
    KeyError
        Unknown ``method``, or it has no variant for ``category``.

    Warns
    -----
    UserWarning
        ``dataset`` matches a folder only up to letter case.
    UserWarning
        ``check=None`` and a per-batch folder for a vertical or diagonal ``category``.
    UserWarning
        No ``method``, and a method that can run on the folder puts these
        cells in another order (StabMap on ``D52``); the warning names it.

    Examples
    --------
    >>> # diagonal: RNA cells first, then ATAC
    >>> labels_for("D28")
    {'rna_cty': '/abs/data/D28/rna_cty.csv',
     'atac_cty': '/abs/data/D28/atac_cty.csv'}
    >>> # StabMap: its reference batch first
    >>> labels_for("D52", "cross", "StabMap")
    {'cty3': '/abs/data/D52/cty3.csv', 'cty1': '/abs/data/D52/cty1.csv',
     'cty2': '/abs/data/D52/cty2.csv'}

    See Also
    --------
    mtb.evaluate : reads these files with ``dataset=``.
    """
    dataset, data_path = split_dataset(dataset, data_path)
    if category is not None:
        registry.check_category(category)
    if method is not None:
        method = registry.check_method(method)
    root = data_path if data_path is not None else config.data_path
    base = Path(os.path.abspath(os.fspath(root)))
    dataset = canonical_dataset(base, dataset)
    ds_dir = base / dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(missing_dataset_message(os.fspath(root), dataset))
    hint = _per_batch_hint(ds_dir, category) if check is not False else None
    if hint:
        msg = f"labels_for({dataset!r}, {category!r}): {hint}"
        if check:
            raise ValueError(msg)
        _compat.warn(msg, UserWarning)
    files = {p.stem: str(p) for p in ds_dir.glob("*cty*.csv")
             if "scjoint" not in p.name.lower()}
    stems = sorted(files, key=_label_sort_key)
    if method is None:
        note = _default_order_warning(dataset, category, stems, base)
        key = (os.path.abspath(base / dataset), category)
        if note and key not in _ORDER_NOTED:      # once a session per dataset
            _ORDER_NOTED.add(key)
            _compat.warn(note, UserWarning)
    if method is not None and category is not None:
        spec = registry.get(method)
        mods = registry.normalize_modalities(modalities)
        try:
            variant = select_variant(spec, category, mods, ds_dir=ds_dir)
        except AmbiguousVariantError:
            variant = None
            stems = _agreed_label_order(spec, category, stems) or stems
        if variant is not None:
            roles = {r for a in variant.args for r in (a.roles or [a.role]) if r}
            # an older paired folder's rna_cty.csv goes under the key cty, as in inputs_for
            if "cty" in roles and "cty" not in files \
                    and _resolve_role(ds_dir, "cty").name == "rna_cty.csv":
                files = {("cty" if k == "rna_cty" else k): v for k, v in files.items()}
                stems = ["cty" if st == "rna_cty" else st for st in stems]
            # only the label files of the batches the variant reads (UINMF: 1-2)
            used = _variant_batches(roles)
            if used:
                stems = [st for st in stems
                         if not (m := _NUMBERED_LABEL_RE.fullmatch(st))
                         or int(m.group(1)) in used]
            rank = _variant_label_rank(stems, variant)
            if rank is not None:
                stems = sorted(stems, key=lambda st: rank[st])
    return LabelFiles((st, files[st]) for st in stems)
