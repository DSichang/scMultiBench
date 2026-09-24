"""Resolve a catalog dataset id + method to concrete input file paths."""
from __future__ import annotations

import functools
import os
import re
import warnings
from pathlib import Path

from .. import config
from . import registry
from .schema import (_NON_MODALITY_ROLES, AmbiguousVariantError, base_modality,
                     is_label_role)

# A few variant roles name a modality *representation* whose on-disk filename
# differs from the role token (e.g. the diagonal ATAC roles). Candidate bases
# are tried in order; the first existing file wins.
_ROLE_FILE_CANDIDATES = {
    "atac_gas": ("atac_gas", "atac"),    # ATAC gene-activity score
    "atac_peak": ("atac_peak", "peak"),  # raw ATAC peaks
    # paired labels: export_dataset writes cty.csv; older folders named the
    # same labels after the RNA cells (UnitedNet's variant read rna_cty.csv).
    # Not next to atac_cty.csv: see _resolve_role.
    "cty": ("cty", "rna_cty"),
}
# Numbered (per-batch) roles: ``atac<i>`` also reads ``atac_peak<i>.h5``, the
# name 0.3.1's export_dataset wrote for mosaic (every mosaic method reads peaks).
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
    """Pick the real on-disk file for a modality role in a flat dataset dir.

    Tries the role token and known aliases (:func:`_role_stems`). Label
    roles (anything matching ``cty`` or ``label``) try ``.csv`` before
    ``.h5`` since cell-type files are CSV. An unnumbered role also takes
    ``<base>1<ext>`` when the folder holds that one numbered file only: with
    ``<base>2<ext>`` present the folder is per batch, and the role stays
    unresolved. Returns the canonical ``<base>.h5`` (``.csv`` for label
    roles) when nothing matches.
    """
    is_label = ("cty" in role) or ("label" in role)
    stems, numbered = _role_stems(role)
    if role == "cty" and (ds_dir / "atac_cty.csv").exists():
        # rna_cty.csv next to atac_cty.csv labels the RNA cells of unpaired
        # (diagonal) data, not the paired cells a cty role reads
        stems = ("cty",)
    exts = (".csv", ".h5") if is_label else (".h5",)
    for stem in stems:
        for ext in exts:
            p = ds_dir / f"{stem}{ext}"
            if p.exists():
                return p
            if numbered:
                continue
            p1 = ds_dir / f"{stem}1{ext}"
            if p1.exists() and not (ds_dir / f"{stem}2{ext}").exists():
                return p1
    fallback_ext = ".csv" if is_label else ".h5"
    return ds_dir / f"{stems[0]}{fallback_ext}"


def _batch_column_advice() -> str:
    """Export without per-batch files, then where the batch column goes."""
    return config.hint("export without batch= and pass the batch column to "
                       "run_all(batch=...) or evaluate(batch=...)",
                       "convert without --batch and pass the batch column to "
                       "run-all --batch or evaluate --batch")


def _one_file_advice(category: str, *, stem: str = "rna", has_adt: bool = False) -> str:
    """What to do instead of per-batch files for a vertical / diagonal folder."""
    if category == "diagonal":
        return ("diagonal methods read one rna.h5 and one ATAC file: "
                + _batch_column_advice())
    return (f"{category} methods read one {stem}.h5: " + _batch_column_advice()
            + (", or use category='cross' (RNA+ADT)" if has_adt else ""))


#: modality file stems whose numbered copies (rna1.h5, rna2.h5) mark a per-batch folder
_PER_BATCH_STEMS = ("rna", "adt", "atac", "atac_peak", "atac_gas")


def _per_batch_hint(ds_dir: Path, category: str | None, stems=None) -> str | None:
    """Hint for a vertical / diagonal folder that holds per-batch files.

    Fires when, for one of ``stems`` (default: the modality stems),
    ``<stem>1.h5`` and ``<stem>2.h5`` exist but ``<stem>.h5`` does not - what
    ``export_dataset(batch=...)`` writes. ``None`` otherwise.
    """
    if category not in ("vertical", "diagonal") or not Path(ds_dir).is_dir():
        return None
    ds_dir = Path(ds_dir)
    numbered = [b for b in _PER_BATCH_STEMS
                if (ds_dir / f"{b}1.h5").is_file() and (ds_dir / f"{b}2.h5").is_file()
                and not (ds_dir / f"{b}.h5").is_file()]
    hit = [b for b in numbered if stems is None or b in stems]
    if not hit:
        return None
    ex = hit[0]
    return (f"this folder holds per-batch files ({ex}1.h5, {ex}2.h5, ...); "
            + _one_file_advice(category, stem=ex,
                               has_adt="adt" in numbered
                               and not any(b.startswith("atac") for b in numbered)))


def _resolve_data_dir(ds_dir: Path) -> str:
    """Directory for a ``data_dir`` role, ending with a path separator.

    The first of ``<ds_dir>/processed/`` and ``<ds_dir>`` that holds a
    ``*.h5ad`` file; when neither does, ``processed/`` if it exists, else
    ``<ds_dir>`` itself (which always exists). scBridge takes this directory
    plus bare filenames relative to it (``const`` args); the upstream script
    string-concatenates ``data_path + filename``.
    """
    for cand in (ds_dir / "processed", ds_dir):
        if cand.is_dir() and any(cand.glob("*.h5ad")):
            return os.path.join(str(cand), "")
    proc = ds_dir / "processed"
    return os.path.join(str(proc if proc.is_dir() else ds_dir), "")


def _resolve_variant_inputs(variant, ds_dir: Path, method: str) -> dict:
    """``{role: path}`` for every input role of ``variant`` in ``ds_dir``
    (best effort: a missing file resolves to its canonical name)."""
    # Args with a const need no on-disk resolution; out_dir is not an input.
    # An arg names one role (a.role) or groups several under one flag (a.roles).
    roles = [r
             for a in variant.args if getattr(a, "const", None) is None
             for r in (getattr(a, "roles", None) or [a.role])
             # "=VALUE" entries are literals emitted verbatim (e.g. scMoMaT's
             # per-batch `None` placeholders), not input roles to find on disk
             if r and not str(r).startswith("=")
             and r not in ("out_dir", "data_dir")]
    out = {role: str(_resolve_role(ds_dir, role)) for role in roles}
    # A `data_dir` role resolves to a directory, not a file.
    if variant.takes_data_dir:
        out["data_dir"] = _resolve_data_dir(ds_dir)
    return out


def _variant_satisfiable(variant, ds_dir: Path, method: str) -> bool:
    """Does the folder hold every input this variant needs? (No content checks:
    existence only, plus the ``data_dir`` content rule for directory-fed
    variants, whose directory always exists.)"""
    got = _resolve_variant_inputs(variant, ds_dir, method)
    if any(not Path(p).exists() for p in got.values()):
        return False
    if "data_dir" in got:
        return _check_data_dir(variant, got["data_dir"])[0]
    return True


def select_variant(spec, category: str, modalities, *, ds_dir: Path | None = None):
    """Pick the one variant of ``spec`` for ``category`` (+ ``modalities``).

    Shared by :func:`inputs_for` and :func:`labels_for` so the two agree on
    what "ambiguous" means (``params_for`` applies the same folder rule through
    :func:`_variant_satisfiable`).

    Parameters
    ----------
    spec : ``MethodSpec``.
    category : validated category token.
    modalities : normalised modality tokens (``registry.normalize_modalities``)
        or None. When given, ``spec.select(..., loose=True)`` is used: exact
        tokens first, then ``atac`` standing for ``atac_gas`` / ``atac_peak``.
    ds_dir : keyword-only; the dataset folder. When ``modalities`` is None and
        several variants exist for ``category``, the one whose input files are
        all present in this folder is chosen; None (or a folder that settles
        nothing) leaves the choice ambiguous.

    Returns
    -------
    Variant

    Raises
    ------
    KeyError
        No variant for ``category`` (or for the given modalities).
    AmbiguousVariantError
        Several variants remain (a ``ValueError``): the message lists the
        modality-sets, the folder contents when a folder was consulted, and
        says to pass ``modalities=``.
    """
    if modalities is not None:
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
    available = [v.when.get("modalities", []) for v in candidates]
    folder_note = ""
    if ds_dir is not None and Path(ds_dir).is_dir():
        ok = [v for v in candidates if _variant_satisfiable(v, Path(ds_dir), spec.id)]
        if len(ok) == 1:
            return ok[0]
        names = sorted(q.name for q in Path(ds_dir).glob("*"))
        folder_note = (f" - {len(ok)} of them have every input file in {ds_dir} "
                       f"(files: {names})")
    raise AmbiguousVariantError(
        f"{spec.id} has multiple variants for category={category!r}: "
        f"modality-sets {available}{folder_note}; pass modalities= to disambiguate, "
        f"e.g. modalities={available[0]}"
    )


_READS = {"peak": "peaks", "gene_activity": "gene activity"}


def _check_representation(spec, modalities) -> None:
    """Refuse representation tokens that disagree with what the method reads.

    Applied when ``modalities`` matched a variant only loosely (not its exact
    roles): ``atac_peak`` / ``atac_gas`` then mean the ATAC representation
    the method reads, ``method_info(m)['atac']``, as in ``scan`` and
    ``find_methods``.
    """
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


def canonical_dataset(base, dataset: str, *, stacklevel: int = 3) -> str:
    """The on-disk spelling of a dataset folder name under ``base``.

    On a case-insensitive filesystem (macOS, Windows) ``data/d52`` opens the
    ``D52`` folder, so a lower-case id would pass every existence check and
    then travel into frames, ``out_dir`` names and saved records as ``'d52'``
    - a second dataset once concatenated with rows keyed ``'D52'``, and a name
    that fails on Linux. The folder listing is the authority: when no entry
    is spelled exactly ``dataset`` but exactly one differs only in case, that
    entry's spelling is returned (with a ``UserWarning`` saying so).

    Parameters
    ----------
    base : path
        The folder that contains the dataset folders.
    dataset : str
        The id as the caller wrote it.
    stacklevel : int
        ``warnings.warn`` stacklevel, so the warning points at the caller's
        caller (the public entry point) by default.

    Returns
    -------
    str
        ``dataset`` unchanged when it is listed as written (or nothing
        matches, so the caller's own missing-folder error fires); otherwise
        the listed spelling.
    """
    base = Path(base)
    name = str(dataset)
    try:
        listing = os.listdir(base)
    except OSError:
        return name
    if name in listing:
        return name
    same = [n for n in listing if n.lower() == name.lower() and (base / n).is_dir()]
    if len(same) != 1:
        return name
    warnings.warn(
        f"dataset {name!r} is not a folder under {base}, but {same[0]!r} is - using "
        f"that on-disk spelling (this file system ignores letter case; Linux does "
        f"not, and results, out_dir names and records carry {same[0]!r})",
        UserWarning, stacklevel=stacklevel)
    return same[0]


def _is_category_token(token) -> bool:
    """``True`` when ``token`` is one of the four category names."""
    try:
        registry.check_category(str(token))
    except ValueError:
        return False
    return True


def _is_method_id(token) -> bool:
    """``True`` when ``token`` is a registry method id (exact spelling)."""
    try:
        registry.check_method(str(token))
    except KeyError:
        return False
    return True


def _old_order_error(fn: str, dataset, category, method) -> None:
    """Raise ``TypeError`` when ``(dataset, method, category)`` - the 0.2
    order - was passed to an entry point whose order is
    ``(dataset, category, method)`` since 0.3.0.

    The guard fires when the category slot holds a registry method id and
    the method slot holds a category token (or, for ``labels_for``, nothing).
    The old order is never accepted silently.
    """
    if _is_method_id(category) and (method is None or _is_category_token(method)):
        raise TypeError(
            f"{fn} argument order is (dataset, category, method) since 0.3.0; "
            f"you passed (dataset, method, category)")


def inputs_for(dataset: str, category: str, method: str, *,
               modalities: list[str] | set[str] | None = None,
               data_path: Path | str | None = None,
               check: bool | None = False) -> dict:
    """Return the input files a method reads from a dataset folder, by role.

    Paths are absolute, ready for ``mtb.run(inputs=...)``. Pass
    ``check=True`` to verify the files before a long run.

    Parameters
    ----------
    dataset : str
        Dataset folder name under ``data_path``, e.g. ``"D11"``.
    category : str
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``.
    method : str
        Method id, e.g. ``"Matilda"``; see ``mtb.list_methods()``.
    modalities : list[str] | set[str] | None
        Modality tokens that pick the variant, e.g. ``["rna", "adt"]``;
        ``None`` = the files in the dataset folder decide.
    data_path : Path | str | None
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.DEFAULT.data_path``.
    check : bool | None
        ``False`` = no checks; ``True`` = raise on a missing or malformed
        input; ``None`` = warn about missing files.

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
        Unknown method, or no variant matches ``category`` and ``modalities``.
    ValueError
        Unknown category or modality token, or several variants fit
        (``mtb.AmbiguousVariantError``).
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
    >>> import multibench as mtb
    >>> mtb.inputs_for("D11", "vertical", "Matilda")
    {'rna': '/abs/data/D11/rna.h5', 'adt': '/abs/data/D11/adt.h5',
     'cty': '/abs/data/D11/cty.csv'}
    >>> inp = mtb.inputs_for("D11", "vertical", "Matilda",
    ...                      modalities=["rna", "adt"], check=True)
    >>> mtb.run("Matilda", "vertical", inputs=inp, out_dir="out/Matilda_D11")
    >>> mtb.inputs_for("D28", "diagonal", "scBridge")
    {'data_dir': '/abs/data/D28/'}

    Notes
    -----
    **Variant selection.** With ``modalities``, the variant matching
    ``(category, modalities)`` is used: exact role tokens first, then
    ``atac`` standing for ``atac_gas`` / ``atac_peak`` when that leaves
    exactly one variant. In the second case a representation token must
    match what the method reads (``method_info(m)['atac']``): SCALEX with
    ``["rna", "peak"]`` raises ``KeyError``.

    Without ``modalities``, a category with one variant uses it. With several,
    the dataset folder decides: the variant whose input files are all present
    is used (Matilda on a ``rna.h5 + adt.h5`` folder is its rna+adt variant).
    When none or several qualify, ``mtb.AmbiguousVariantError`` (a
    ``ValueError``) lists the modality sets and the folder contents and asks
    for ``modalities=``.

    **Modality tokens.** ``method_info(m)['supports']`` lists each variant's
    tokens.

    - ``protein`` is another spelling of ``adt``;
    - ``atac`` stands for either ATAC representation role;
    - the representation tokens ``atac_peak`` (also ``peak``, ``peaks``) and
      ``atac_gas`` (also ``gas``, ``gene_activity``) name what the method
      reads; ``method_info(m)['atac']`` says which one that is.

    An unknown token raises ``ValueError`` naming the vocabulary.

    **Validation.** A misspelt method id raises ``KeyError`` naming the
    closest method id; an unknown category raises ``ValueError`` listing
    the four.

    **File resolution.** The dataset tree is flat
    (``<data_path>/<dataset>/<file>``). Each role resolves to the file present
    in the folder: the role token or a known alias. Label roles look for
    ``.csv`` first. When nothing matches, the role falls back to ``<role>.h5``
    (``<role>.csv`` for a label role), a path that does not exist (see
    ``check``).

    Label files: the ``cty`` role of paired data reads ``cty.csv``. An older
    paired folder may name it ``rna_cty.csv``; that file is read when the
    folder has neither ``cty.csv`` nor ``atac_cty.csv``.

    ATAC files: vertical methods read ``atac.h5``; ``method_info(m)['atac']``
    says whether it must hold peaks or gene activity. Diagonal methods read
    ``atac_peak.h5`` (peaks) and ``atac_gas.h5`` (gene activity). Mosaic
    methods read ``atac<i>.h5`` (peaks). ``peak.h5``, and ``atac.h5`` for gene
    activity, are accepted as older names, and so is ``atac_peak<i>.h5``.

    Numbered files: an unnumbered role (``rna``) also takes ``rna1.h5`` when
    the folder holds no ``rna2.h5``. A folder with ``rna1.h5`` and
    ``rna2.h5`` is per batch: vertical and diagonal roles stay unresolved,
    and the error says to export without ``batch=``.

    A ``data_dir`` role (scBridge) resolves to the first of
    ``<dataset>/processed/`` and the dataset folder that holds a ``.h5ad``
    file; when neither does, to ``processed/`` if that folder exists, else to
    the dataset folder itself.

    **Absolute paths.** Every returned path is absolute (a relative
    ``data_path`` is resolved against the current directory), and a
    ``data_dir`` value ends with the path separator. ``mtb.run`` executes the
    method with ``cwd=out_dir``, where a relative path would point at the
    wrong place.

    **The check argument.**

    - ``False`` (default): the best-effort paths, with no warning.
    - ``None``: the same paths, plus a ``UserWarning`` listing the missing ones.
    - ``True``: ``FileNotFoundError`` for a missing input, plus the content
      preflight below.

    The missing-file error or warning names an ATAC-family sibling that is
    present, e.g. ``atac_peak.h5`` when a vertical variant reads ``atac.h5``,
    and says when the folder holds per-batch files.

    **Content preflight** (``check=True``). The same checks ``mtb.scan``
    reports per row as ``files_ok`` / ``files_reason``:

    - orientation: ``ValueError`` when ``matrix/data`` is stored cells x
      features;
    - label length: ``ValueError`` when a label CSV has a different number of
      rows than the modality file it labels, including the numbered
      ``cty<i>.csv`` of a cross/mosaic batch and the diagonal ``rna_cty.csv``
      / ``atac_cty.csv`` (read by every evaluation, even when the method
      does not take them);
    - ``data_dir`` content: ``FileNotFoundError`` when a file scBridge names
      inside ``data_dir`` (``rna.h5``, ``atac_gas.h5``, the two label CSVs) is
      absent;
    - same cells: ``ValueError`` when Seurat_v5's ``rna.h5`` and
      ``atac_peak.h5`` hold different cells. Seurat_v5 builds its paired
      bridge from these two files;
    - ATAC cell order (diagonal): ``ValueError`` when the ``atac_gas.h5`` a
      method reads holds other cells than ``atac_peak.h5``, or lists them in
      another order. ``atac_cty.csv`` follows ``atac_peak.h5``. Barcodes that
      differ only in a ``-1`` / ``-2`` suffix count as the same cell.

    **Dataset name case.** A spelling that differs from the folder only in
    case (``'d52'`` for ``D52`` on a case-insensitive filesystem) is replaced
    by the on-disk spelling, with a ``UserWarning``.

    **Argument order.** ``(dataset, category, method)``, the same order
    ``mtb.scan``, ``mtb.run_all`` and ``mtb.labels_for`` use.

    See Also
    --------
    mtb.labels_for : the cell-type label CSVs of the same dataset, in the method's cell order.

    mtb.run : consumes the returned dict as ``inputs=``.

    mtb.scan : the same resolution for every method at once, with reasons.

    mtb.describe_layout : the folder layout these roles resolve against.
    """
    _old_order_error("inputs_for", dataset, category, method)
    root = data_path if data_path is not None else config.DEFAULT.data_path
    base = Path(os.path.abspath(os.fspath(root)))
    spec = registry.get(method)
    registry.check_category(category)
    dataset = canonical_dataset(base, dataset)
    ds_dir = base / dataset  # flat layout: data/<dataset>/<file>
    modalities = registry.normalize_modalities(modalities)
    variant = select_variant(spec, category, modalities, ds_dir=ds_dir)
    out = _resolve_variant_inputs(variant, ds_dir, method)
    missing = {r: p for r, p in out.items() if not Path(p).exists()}
    near = _near_miss_hints(ds_dir, missing, category, atac=spec.atac)
    batch_hint = _per_batch_hint(
        ds_dir, category, {st for r in missing for st in _role_stems(r)[0]})
    if batch_hint:
        near.append(batch_hint)
    if check:
        if missing:
            raise FileNotFoundError(
                f"{method}/{dataset}/{category}: input files not found on disk: "
                f"{missing}. Available files in {ds_dir}: "
                f"{sorted(q.name for q in ds_dir.glob('*')) if ds_dir.is_dir() else '(dir missing)'}"
                + (" - " + "; ".join(near) if near else "")
            )
        _check_orientation(method, dataset, category, out)
        _check_same_cells(method, dataset, category, out)
        if category == "diagonal":
            _check_atac_gas_cells(method, dataset, out)
        _check_label_lengths(method, dataset, category, out)
        if "data_dir" in out:
            ok, why = _check_data_dir(variant, out["data_dir"])
            if not ok:
                raise FileNotFoundError(f"{method}/{dataset}/{category}: {why}")
            if category == "diagonal":
                named = _data_dir_files(variant, out["data_dir"])
                _check_atac_gas_cells(method, dataset, named)
                _check_diagonal_label_files(method, dataset, named)
    elif check is None and missing:
        warnings.warn(
            f"{method}/{dataset}/{category}: {len(missing)} resolved input path(s) "
            f"do not exist: {missing}"
            + (" (" + "; ".join(near) + ")" if near else "")
            + "; pass check=True to raise, check=False to silence",
            UserWarning, stacklevel=2)
    return out


# Every on-disk base name an ATAC-family role may be looked up under, so a
# near miss can be named: peaks exported as atac_peak.h5 when a vertical
# variant asks for atac.h5.
_ATAC_FILE_BASES = ("atac", "atac_peak", "atac_gas", "peak")
#: the representation a sibling file holds, by its name (``spec.atac`` values)
_KIND_BY_BASE = {"atac_peak": "peak", "peak": "peak", "atac_gas": "gene_activity"}


def _near_miss_hints(ds_dir: Path, missing: dict, category: str,
                     atac: str | None = None) -> list[str]:
    """For each missing ATAC-family role, name the sibling file that is there.

    The ``atac`` role reads ``atac.h5``; ``atac_gas`` reads ``atac_gas.h5``
    (falling back to ``atac.h5``), ``atac_peak`` reads ``atac_peak.h5``
    (falling back to ``peak.h5``) and a numbered ``atac<i>`` reads
    ``atac<i>.h5`` or ``atac_peak<i>.h5``. Return one hint per such role,
    e.g. ``"atac.h5 not found; found atac_peak.h5 - vertical reads
    atac.h5 (pass the representation this method wants: see
    method_info(m)['atac'])"``; nothing for roles that are not ATAC or have
    no sibling. ``atac`` is the method's representation (``spec.atac``):
    when a vertical sibling holds it (by name: ``atac_peak`` / ``peak`` are
    peaks, ``atac_gas`` gene activity), the hint names the rename instead,
    as scan's short reason does.
    """
    hints: list[str] = []
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
            # one rule for every vertical ATAC role: the atac_gas role of the
            # peak methods (moETM, scMM, iPOLNG) also reads atac.h5
            hints.append(
                f"atac.h5 not found; found {', '.join(found)} - vertical reads "
                f"atac.h5: rename {same[0]} to atac.h5, or write it with "
                + config.hint('category="vertical"', "--category vertical"))
            continue
        why = ("every mosaic method reads peaks" if m else
               "pass the representation this method wants: see method_info(m)['atac']")
        rule = f"{accepted[0]} not found; found {', '.join(found)} - {category} methods " \
               f"read {' or '.join(accepted)}"
        if category == "vertical" and not m:
            # the rule of the peak rows above, also for an atac_gas role
            rule = f"atac.h5 not found; found {', '.join(found)} - vertical reads atac.h5"
        hints.append(f"{rule} ({why})")
    return hints


@functools.lru_cache(maxsize=512)
def _sniff_h5(path: str, mtime_ns: int):
    """Read (shape, n_features, n_cells) of a canonical .h5, cached by mtime.

    scan() runs the orientation preflight once per (method, variant) row, so
    the same few files would otherwise be opened many times per call; the
    mtime key invalidates an entry when the file is rewritten.
    """
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


def _check_orientation(method, dataset, category, resolved):
    """Reject a transposed matrix at preflight instead of many minutes later.

    Modality files store ``matrix/data`` as (features x cells); cells x
    features, the scanpy/AnnData convention, is the easy mistake. The
    file-existence check cannot see it, so the method would start, pay its
    conda-env startup (and possibly hours of compute) and only then fail inside
    third-party code with an error that does not mention orientation.

    ``matrix/features`` and ``matrix/barcodes`` fix the intended orientation
    without reference to the labels, so this is checkable up front. A square
    matrix is ambiguous and is left alone.
    """
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
                f"{method}/{dataset}/{category}: {p.name} stores matrix/data as "
                f"{shape}, which is cells x features. This layout expects "
                f"features x cells - here ({n_feat}, {n_cell}), matching "
                f"matrix/features ({n_feat}) and matrix/barcodes ({n_cell}). "
                f"Re-export with mtb.io.to_canonical(src, dst), or transpose "
                f"matrix/data."
            )


@functools.lru_cache(maxsize=512)
def _sniff_features(path: str, mtime_ns: int, n: int = 50) -> tuple:
    """First ``n`` feature names of a canonical .h5 (cached by mtime), or ()."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "matrix/features" not in f:
                return ()
            raw = f["matrix/features"][:n]
    except OSError:
        return ()
    return tuple(x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in raw)


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
    """Which modality roles a label role labels (same-cell pairing rule).

    * ``cty`` (one label set, paired data) -> every modality role;
    * ``cty<N>`` (one label file per batch) -> the roles numbered ``<N>``;
    * ``rna_cty`` / ``atac_cty`` / ``peak_cty`` -> the roles of that base
      modality (``atac_cty`` covers ``atac``, ``atac_gas`` and ``atac_peak``);
    * anything else (``source_cty`` ...) -> nothing (no safe pairing).
    """
    mods = [r for r in roles if not is_label_role(r) and r != "data_dir"]
    if label_role == "cty":
        return mods
    digits = "".join(ch for ch in label_role if ch.isdigit())
    if label_role.startswith("cty") and digits:
        return [r for r in mods if r.endswith(digits)]
    if label_role.endswith("_cty"):
        prefix = label_role[:-4]
        base = {"peak": "atac"}.get(prefix, prefix)
        return [r for r in mods if base_modality(r) == base and not r[-1:].isdigit()]
    return []


_BATCH_DIGITS_RE = re.compile(r"(\d+)$")


def _batch_label_file(role: str, path) -> tuple[str, Path] | None:
    """``(batch_index, <dir>/cty<i>.csv)`` for a numbered modality role, else ``None``.

    Cross and mosaic datasets label each batch in ``cty<i>.csv`` next to
    ``rna<i>.h5`` / ``adt<i>.h5`` / ``atac<i>.h5``. No cross method takes
    that file as an input role - only the evaluator reads it - so the
    role-driven pairing never sees it; the sibling is looked up on disk.
    """
    if is_label_role(role) or role == "data_dir":
        return None
    m = _BATCH_DIGITS_RE.search(role)
    if not m:
        return None
    p = Path(path)
    if p.suffix != ".h5":
        return None
    return m.group(1), p.parent / f"cty{m.group(1)}.csv"


def _check_label_lengths(method, dataset, category, resolved):
    """Reject a label file whose row count differs from the cells it labels.

    ``evaluate`` would refuse the pair later ("emb has N cells, celltype has
    M") and the method itself may fail or, worse, silently mis-align. The
    pairing follows the role names (see :func:`_label_partners`); a file that
    cannot be parsed or a modality file without ``matrix/barcodes`` is left
    alone.

    Numbered batch files are checked too: ``cty<i>.csv`` (when present next
    to the modality file) against ``rna<i>.h5`` / ``adt<i>.h5`` /
    ``atac<i>.h5`` of the same batch ``i`` - the layout every cross and mosaic
    method reads, although none of them lists ``cty<i>`` as an input role.
    Without this a truncated ``cty1.csv`` passes :func:`multibench.scan` with
    ``files_ok=True`` and only fails inside ``evaluate`` after the run.
    """
    for role, path in resolved.items():
        pair = _batch_label_file(role, path)
        if pair is None:
            continue
        batch, lab = pair
        if f"cty{batch}" in resolved:      # an input role: the loop below checks it
            continue
        q = Path(path)
        if not lab.is_file() or not q.is_file():
            continue
        n_lab = _count_label_rows(str(lab), lab.stat().st_mtime_ns)
        sniff = _sniff_h5(str(q), q.stat().st_mtime_ns)
        if n_lab is None or sniff is None:
            continue
        shape, n_feat, n_cell = sniff
        if n_feat == n_cell:              # orientation ambiguous: cannot tell cells
            continue
        if n_lab != n_cell:
            raise ValueError(
                f"{method}/{dataset}/{category}: {lab.name} has {n_lab} labels but "
                f"{q.name} has {n_cell} cells (matrix/barcodes) - batch {batch}: "
                f"every cell of a batch needs exactly one label in cty{batch}.csv, "
                f"in the same order as the cells (see "
                + config.hint(f"mtb.describe_layout({category!r})",
                              f"multibench layout {category}") + ")")
    if category == "diagonal":
        _check_diagonal_label_files(method, dataset, resolved)
    for role, path in resolved.items():
        if not is_label_role(role):
            continue
        p = Path(path)
        if p.suffix != ".csv" or not p.is_file():
            continue
        n_lab = _count_label_rows(str(p), p.stat().st_mtime_ns)
        if n_lab is None:
            continue
        for partner in _label_partners(role, resolved):
            q = Path(resolved[partner])
            if q.suffix != ".h5" or not q.is_file():
                continue
            sniff = _sniff_h5(str(q), q.stat().st_mtime_ns)
            if sniff is None:
                continue
            shape, n_feat, n_cell = sniff
            if n_feat == n_cell:          # orientation ambiguous: cannot tell cells
                continue
            if n_lab != n_cell:
                raise ValueError(
                    f"{method}/{dataset}/{category}: {p.name} has {n_lab} labels but "
                    f"{q.name} has {n_cell} cells (matrix/barcodes) - every cell needs "
                    f"exactly one label, in the same order as the cells (see "
                    + config.hint(f"mtb.describe_layout({category!r})",
                                  f"multibench layout {category}") + ")")


def _check_diagonal_label_files(method, dataset, resolved):
    """Diagonal: ``rna_cty.csv`` / ``atac_cty.csv`` next to the modality files
    must have one row per cell of ``rna.h5`` / the ATAC file, also when the
    variant does not read them (every evaluation does)."""
    for role, path in resolved.items():
        if is_label_role(role) or role == "data_dir" or role[-1:].isdigit():
            continue
        base = base_modality(role)
        if base not in ("rna", "atac"):
            continue
        label_role = f"{base}_cty"
        if label_role in resolved:          # an input role: the caller checks it
            continue
        q = Path(path)
        lab = q.parent / f"{label_role}.csv"
        if q.suffix != ".h5" or not q.is_file() or not lab.is_file():
            continue
        n_lab = _count_label_rows(str(lab), lab.stat().st_mtime_ns)
        sniff = _sniff_h5(str(q), q.stat().st_mtime_ns)
        if n_lab is None or sniff is None:
            continue
        shape, n_feat, n_cell = sniff
        if n_feat == n_cell:
            continue
        if n_lab != n_cell:
            raise ValueError(
                f"{method}/{dataset}/diagonal: {lab.name} has {n_lab} labels but "
                f"{q.name} has {n_cell} cells (matrix/barcodes) - every cell needs "
                f"exactly one label, in the same order as the cells (see "
                + config.hint("mtb.describe_layout('diagonal')",
                              "multibench layout diagonal") + ")")


def _data_dir_files(variant, data_dir) -> dict:
    """``{stem: path}`` of the ``.h5`` files a ``data_dir`` method names as
    ``const`` args (scBridge: ``rna.h5``, ``atac_gas.h5``), for the checks
    that take resolved roles."""
    d = Path(data_dir)
    return {Path(str(a.const)).stem: str(d / str(a.const)) for a in variant.args
            if a.const and str(a.const).endswith(".h5")}


def _check_data_dir(variant, data_dir) -> tuple[bool, str]:
    """Does a ``data_dir`` really hold what the method needs? -> (ok, why).

    ``data_dir`` resolves to the dataset directory itself when there is no
    ``processed/`` subdir, so the path always exists and existence proves
    nothing. A ``data_dir`` method (scBridge) names its files via ``const``
    args; every named ``.h5`` / ``.csv`` file must be present.
    """
    d = Path(data_dir)
    if not d.is_dir():
        return False, f"no such directory: {d}"
    needed = [a.const for a in variant.args if a.const and str(a.const).endswith((".h5", ".csv"))]
    missing = [f for f in needed if not (d / f).exists()]
    if missing:
        return False, f"missing files in {d}: {missing}"
    return True, ""


# Every caveat leads with the problem: the compact CLI table clips the caveat
# column to 40 characters, so the first 30 carry the warning.
#: Caveat text appended by scan() when an ``atac_gas`` role falls back to a
#: peak matrix in ``atac.h5`` (the wanted representation not given).
PEAK_IN_GAS_CAVEAT = ("expects gene activity; atac.h5 holds peaks (features look like "
                      "chr:start-end)")
#: ``.format(file=...)`` templates of the two representation-mismatch caveats
#: reported when the method's wanted ATAC representation is known
#: (``_preflight_caveats(resolved, atac=method_info(m)['atac'])``).
PEAK_FED_TO_GAS_CAVEAT = ("expects gene activity; {file} holds peaks (features look like "
                          "chr:start-end)")
GAS_FED_TO_PEAK_CAVEAT = ("expects peaks; {file} holds gene activity (features do not "
                          "look like chr:start-end)")
#: ``.format(file=, example=)`` caveat for a file a peak method reads whose
#: names are neither chr:start-end nor the gene names of the folder's other
#: files (``peak_0``): the kind cannot be told from the names, so it does not
#: block the row.
PEAK_NAMES_UNKNOWN_CAVEAT = ("expects peaks; {file} holds names that are not "
                             "chr:start-end (e.g. {example})")
#: The same for a method that reads gene activity: the names are neither
#: chr:start-end nor the genes of the folder's RNA.
GAS_NAMES_UNKNOWN_CAVEAT = ("expects gene activity; {file} holds names that are not "
                            "the RNA's genes (e.g. {example})")
#: ``.format(file=, example=)`` caveat for a peak file of a variant whose
#: peak names ``mtb.run`` rewrites to chr:start-end (``normalize_peaks``) when
#: more than 10% of the first 50 names are not chr<sep>start<sep>end, so the
#: rewrite cannot help; ``example`` is the first such name. No subject, like
#: the other caveats: logs print it after the method name. It ends with the fix.
PEAK_NAMES_CAVEAT = ("reads peak names such as chr1:100-200. {file} holds other "
                     "names, for example {example}. Rename them to chr:start-end.")
#: ``.format(file=...)`` caveat for a modality file whose sampled values are not
#: whole numbers (log-normalised data).
NOT_COUNTS_CAVEAT = "expects raw counts; {file} holds non-integer values"
#: Caveat for a diagonal folder whose only label file is ``cty.csv``.
DIAGONAL_CTY_CAVEAT = ("needs rna_cty.csv and atac_cty.csv for diagonal; the folder has "
                       "only cty.csv")
#: ``.format(used=, n=, unused=)`` caveat for a variant that reads fewer numbered
#: batches than the folder holds: ``reads batches 1-2 of 3; batch 3 is not used``.
UNUSED_BATCHES_CAVEAT = "reads batches {used} of {n}; {unused}"
_UNUSED_BATCHES_RE = re.compile(r"reads batches [\d, and-]+ of \d+; batch(?:es)? "
                                r"[\d, and-]+ (?:is|are) not used")
# file stems (batch digits allowed) whose values must be raw counts
_COUNT_FILE_RE = re.compile(r"^(rna|adt|atac_peak)\d*$")


@functools.lru_cache(maxsize=512)
def _h5_has_fraction(path: str, mtime_ns: int) -> bool | None:
    """Raw-count check of a canonical .h5 (cached by mtime): ``True`` when a
    sample of up to 5,000 stored non-zero values holds a non-whole number,
    ``None`` when the file cannot be read. Reads a few chunks only."""
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
            n_r, n_c = d.shape
            cr, cc = d.chunks or (max(1, min(n_r, 200_000 // n_c)), n_c)
            vals, got = [], 0
            for r, c in zip(np.linspace(0, n_r - 1, 4).astype(int),
                            np.linspace(0, n_c - 1, 4).astype(int)):
                r0, c0 = (r // cr) * cr, (c // cc) * cc
                x = np.asarray(d[r0:r0 + cr, c0:c0 + cc]).ravel()
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
    """Every feature name of a canonical .h5 (cached by mtime), or ()."""
    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "matrix/features" not in f:
                return ()
            raw = f["matrix/features"][:]
    except OSError:
        return ()
    return tuple(x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in raw)


#: Ensembl gene ids (``ENSG00000141510``, ``ENSMUSG...``): gene names whatever
#: the folder's RNA calls its genes
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}")


def _names_are_genes(path: Path) -> bool | None:
    """Whether the names of a non-peak ATAC file are gene names.

    ``True`` when at least 1% of up to 2,000 evenly spaced names (and at
    least one) occur, ignoring case, among the features of the folder's
    ``rna*.h5`` and ``atac_gas*.h5`` files other than this one, or when most
    of them are Ensembl gene ids. ``False`` otherwise (``peak_0``). ``None``
    when the folder has no such file to compare with, or only files named
    by Ensembl id, which gene symbols would not match.
    """
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
    """The first of the first 50 feature names that ``normalize_peak_names``
    cannot rewrite to ``chr:start-end``, when more than 10% of them are such
    names; ``None`` otherwise, or when the file is not a readable canonical
    ``.h5``."""
    from .ingest import _PEAK_RE

    if path.suffix != ".h5" or not path.is_file():
        return None
    feats = _sniff_features(str(path), path.stat().st_mtime_ns)
    bad = [x for x in feats or [] if not _PEAK_RE.match(x)]
    return bad[0] if feats and len(bad) > 0.1 * len(feats) else None


def _renamed_peak_roles(method: str | None, category: str | None, resolved) -> list[str]:
    """The roles of ``resolved`` whose peak names ``mtb.run`` rewrites for this
    method's variant (``Variant.normalize_peaks``); ``[]`` when the variant is
    not known."""
    if not method or not category:
        return []
    try:
        mods = {r for r in resolved if not is_label_role(r) and r not in _NON_MODALITY_ROLES}
        variant = registry.get(method).select(category, mods)
    except Exception:  # noqa: BLE001 - a caveat check never raises
        return []
    return [r for r in (getattr(variant, "normalize_peaks", None) or []) if r in resolved]


#: methods whose script needs the same cells in two of its input roles:
#: Seurat_v5 builds its bridge from rna + atac_peak (main_Seurat_v5.Rmd:39-44;
#: ``obj.multi[["ATAC"]] <- ...`` accepts only the cells the object holds)
_SAME_CELL_ROLES = {"Seurat_v5": ("rna", "atac_peak")}
#: ``.format(method=, n_a=, n_b=, shared=)`` template of the file-check failure
#: when those two files hold different barcode sets
SAME_CELLS_REASON = ("{method} needs RNA and ATAC from the same cells as its bridge. These "
                     "files share {shared:,} of {n_a:,} and {n_b:,} cells")
#: ``.format(gas=, peak=, n_gas=, n_peak=, shared=)``: a diagonal gene-activity
#: file whose cells are not the cells of the peak file next to it
GAS_OTHER_CELLS_REASON = ("{gas} and {peak} hold different cells ({n_gas:,} and "
                          "{n_peak:,}, {shared:,} shared). Both files need the same "
                          "ATAC cells")
#: ``.format(gas=, peak=, write=)``: the same cells in another order
GAS_OTHER_ORDER_REASON = ("{gas} lists the ATAC cells in another order than {peak}; "
                          "atac_cty.csv follows {peak}. Write it again with {write}")


@functools.lru_cache(maxsize=64)
def _sniff_barcode_list(path: str, mtime_ns: int) -> tuple | None:
    """The ``matrix/barcodes`` of a canonical .h5 in file order (cached by
    mtime), or None."""
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


def _check_same_cells(method, dataset, category, resolved) -> None:
    """``ValueError`` (:data:`SAME_CELLS_REASON`) when ``method`` needs two
    roles to hold the same cells and their files' barcode sets differ. The
    order may differ; a file that is not a readable canonical ``.h5`` is left
    alone."""
    roles = _SAME_CELL_ROLES.get(method or "")
    if not roles or not all(r in resolved for r in roles):
        return
    bars = [_barcodes_of(resolved[r]) for r in roles]
    if any(b is None for b in bars):
        return
    a, b = (set(x) for x in bars)
    if a == b:
        return
    raise ValueError(f"{method}/{dataset}/{category}: " + SAME_CELLS_REASON.format(
        method=method, n_a=len(a), n_b=len(b), shared=len(a & b)))


def _check_atac_gas_cells(method, dataset, resolved) -> None:
    """Diagonal: the gene-activity file a variant reads must list the cells of
    its peak file in that file's order, because ``atac_cty.csv`` labels the
    peak file's cells. The peak file is the ``atac_peak`` role when
    ``resolved`` has one, else the file next to the gene-activity file
    (``atac_peak.h5``, else ``peak.h5``).

    Barcodes that differ only in a trailing ``-<n>`` suffix count as the same
    cell when the suffix-free names stay unique (D28's two ATAC files end in
    ``-1`` and ``-2``). Non-unique barcodes, or a file that is not a readable
    canonical ``.h5``, are left alone.
    """
    from .ingest import _cell_order

    gas = resolved.get("atac_gas")
    if not gas:
        return
    gas = Path(gas)
    given = resolved.get("atac_peak")
    if given and Path(given).is_file():
        peak = Path(given)
    else:
        peak = next((gas.parent / n for n in ("atac_peak.h5", "peak.h5")
                     if (gas.parent / n).is_file()), None)
    if peak is None or peak.resolve() == gas.resolve():
        return
    a, b = _barcodes_of(gas), _barcodes_of(peak)
    if a is None or b is None:
        return
    kind, _ = _cell_order(a, b)
    if kind == "other":
        why = GAS_OTHER_CELLS_REASON.format(gas=gas.name, peak=peak.name, n_gas=len(a),
                                            n_peak=len(b), shared=len(set(a) & set(b)))
    elif kind == "order":
        why = GAS_OTHER_ORDER_REASON.format(
            gas=gas.name, peak=peak.name,
            write=config.hint("mtb.io.to_canonical(..., modality='gas')",
                              "multibench convert SRC DIR --modality gas"))
    else:
        return
    raise ValueError(f"{method}/{dataset}/diagonal: {why}")


#: file names that carry a batch number: ``rna2.h5``, ``atac_peak3.h5``, ``cty1.csv``
_BATCH_FILE_RE = re.compile(r"^(?:rna|adt|atac|atac_peak|atac_gas)(\d+)\.h5$|^cty(\d+)\.csv$")
#: role tokens that carry a batch number
_BATCH_ROLE_RE = re.compile(r"^(?:rna|adt|atac|atac_peak|atac_gas|cty)(\d+)$")


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


def _unused_batches_note(resolved) -> str | None:
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
    verb = "batch {} is not used" if len(unused) == 1 else "batches {} are not used"
    return UNUSED_BATCHES_CAVEAT.format(used=_span(used), n=len(held),
                                        unused=verb.format(_span(unused)))


def unused_batches_in(caveat) -> str | None:
    """The ``reads batches ... not used`` note inside a scan ``caveat`` text, or None."""
    m = _UNUSED_BATCHES_RE.search(str(caveat or ""))
    return m.group(0) if m else None


def _preflight_caveats(resolved, *, atac: str | None = None,
                       category: str | None = None,
                       method: str | None = None) -> list[str]:
    """Non-fatal content observations about resolved inputs (never raises).

    Without ``atac`` (wanted representation unknown) one ATAC check runs: when
    the ``atac_gas`` role fell back to ``atac.h5`` (no ``atac_gas.h5``
    present) and >= 90% of the first 50 feature names look like peaks
    (``chr1:1-200`` / ``chr1_1_200``), report :data:`PEAK_IN_GAS_CAVEAT`.
    Without the wanted representation this would also flag the methods that
    expect peaks behind the ``atac_gas`` role name (``atac: peak`` in the
    registry), so :func:`multibench.scan` always passes ``atac=``.

    With ``atac=`` - the representation the method expects,
    ``method_info(m)['atac']`` (``'peak'`` / ``'gene_activity'``) - every
    ATAC-family role (``atac``, ``atac_peak``, ``atac_gas``, ``atac1``...) is
    judged against it, whatever the role name says:

    * ``atac='gene_activity'`` and the file looks like peaks ->
      :data:`PEAK_FED_TO_GAS_CAVEAT` (e.g. Matilda's ``atac`` role on a
      peaks-only multiome folder);
    * ``atac='peak'`` and <= 10% of the features look like peaks ->
      :data:`GAS_FED_TO_PEAK_CAVEAT` (e.g. moETM/scMM/iPOLNG, whose ``atac_gas``
      role resolved to a real gene-activity ``atac_gas.h5``), when the names
      are gene names (:func:`_names_are_genes`: the folder's ``rna*.h5`` or
      ``atac_gas*.h5`` hold them, or no such file exists to compare with).
      Other names (``peak_0``) -> :data:`PEAK_NAMES_UNKNOWN_CAVEAT`, which
      does not block the row;
    * ``atac='gene_activity'`` and such other names ->
      :data:`GAS_NAMES_UNKNOWN_CAVEAT`, which does not block the row either.

    A method with both an ``atac_peak`` and an ``atac_gas`` role (MultiMAP,
    Seurat_v3) reads peaks from the first and gene activity from the second;
    each of those roles is judged by its own name.

    The 10-90% band (mixed names) yields no caveat. The wrong representation
    runs to completion and returns a wrong embedding.

    Always: a ``rna*.h5`` / ``adt*.h5`` / ``atac_peak*.h5`` whose sampled
    values are not whole numbers -> :data:`NOT_COUNTS_CAVEAT`. With
    ``category='diagonal'``: a folder whose only label file is ``cty.csv``
    -> :data:`DIAGONAL_CTY_CAVEAT`.

    First of all: numbered roles that name fewer batches than the folder
    holds -> :data:`UNUSED_BATCHES_CAVEAT`.

    With ``method`` and ``category``: a role the method's variant renames
    (``normalize_peaks``; ``mtb.run`` passes a chr:start-end copy) is judged
    by its names instead of the representation check above. Underscore or
    dash peak names are rewritten and need no caveat; when more than 10% of
    the first 50 names have no chr<sep>start<sep>end form ->
    :data:`PEAK_NAMES_CAVEAT`. The same-cells rule (``_SAME_CELL_ROLES``) is
    a file check of :func:`inputs_for`, not a caveat.
    """
    out: list[str] = []
    note = _unused_batches_note(resolved)
    if note:
        out.append(note)
    renamed = _renamed_peak_roles(method, category, resolved)
    for role in renamed:
        bad = _unrewritable_peak_name(Path(resolved[role]))
        if bad is not None:
            out.append(PEAK_NAMES_CAVEAT.format(file=Path(resolved[role]).name, example=bad))
    if atac is None:
        p = Path(resolved.get("atac_gas", ""))
        if p.name and p.stem != "atac_gas":
            frac = _peak_fraction_of(p)
            if frac is not None and frac >= 0.9:
                out.append(PEAK_IN_GAS_CAVEAT)
    else:
        # a method that reads both files (MultiMAP, Seurat_v3) wants gene
        # activity in its atac_gas role and peaks in its atac_peak role
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
                out.append(PEAK_FED_TO_GAS_CAVEAT.format(file=Path(path).name))
            elif frac <= 0.1:
                # names that are not chr:start-end are gene activity only
                # when they are the folder's gene names; peak_0 names are not
                genes = _names_are_genes(Path(path))
                if genes is False:
                    first = _sniff_features(str(path), Path(path).stat().st_mtime_ns)
                    unknown = (PEAK_NAMES_UNKNOWN_CAVEAT if want == "peak"
                               else GAS_NAMES_UNKNOWN_CAVEAT)
                    out.append(unknown.format(file=Path(path).name, example=first[0]))
                elif want == "peak":
                    out.append(GAS_FED_TO_PEAK_CAVEAT.format(file=Path(path).name))
    seen = set()
    for role, path in resolved.items():
        p = Path(path)
        if is_label_role(role) or role == "data_dir" or p in seen:
            continue
        seen.add(p)
        if p.suffix != ".h5" or not _COUNT_FILE_RE.match(p.stem) or not p.is_file():
            continue
        if _h5_has_fraction(str(p), p.stat().st_mtime_ns):
            out.append(NOT_COUNTS_CAVEAT.format(file=p.name))
    if category == "diagonal" and resolved:
        ds_dir = next((Path(v) if k == "data_dir" else Path(v).parent
                       for k, v in resolved.items()), None)
        if ds_dir is not None and (ds_dir / "cty.csv").is_file() \
                and not (ds_dir / "rna_cty.csv").is_file() \
                and not (ds_dir / "atac_cty.csv").is_file():
            out.append(DIAGONAL_CTY_CAVEAT)
    return out


#: canonical stacking order of the modality-named label files (``rna_cty.csv``
#: before ``adt_cty.csv`` before ``atac_cty.csv``; ``peak_cty`` counts as atac)
_LABEL_MODALITY_ORDER = {"rna": 0, "adt": 1, "atac": 2}


def _label_sort_key(stem: str):
    """Sort key giving the benchmark's cell-stacking order of label files.

    1. ``cty`` (one file, paired cells) first;
    2. ``cty<N>`` numbered per batch, ascending numerically (cty1, cty2, cty10);
    3. ``<modality>_cty`` in the canonical modality order rna, adt, atac
       (``peak_cty`` is treated as atac) - the order in which most diagonal /
       vertical methods stack their cells in the embedding (RNA cells first,
       then ATAC cells; uniPort and Seurat_v5 declare the reverse, see
       ``Variant.stacked_roles``);
    4. anything else (``source_cty`` ...) alphabetically, last.
    """
    if stem == "cty":
        return (0, 0, "")
    digits = "".join(ch for ch in stem if ch.isdigit())
    if stem.startswith("cty") and digits and stem == f"cty{digits}":
        return (1, int(digits), "")
    if stem.endswith("_cty"):
        base = {"peak": "atac"}.get(stem[:-4], base_modality(stem[:-4]))
        return (2, _LABEL_MODALITY_ORDER.get(base, 9), stem)
    return (3, 0, stem)


def _variant_label_rank(stems: list[str], variant) -> dict[str, tuple] | None:
    """Rank label stems by where the cells they label sit in ``variant``'s
    output (``Variant.stacked_roles``: the argument order unless the registry
    declares ``output.cell_order``). ``None`` when no stem pairs with a role."""
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


class LabelFiles(dict):
    """The ``{stem: path}`` dict :func:`labels_for` returns.

    A plain ``dict`` except that it remembers the key order ``labels_for``
    gave it (``stacking_order``), so ``evaluate`` takes it as is even when
    that is not the default order (StabMap's reference batch first). Once
    the keys are in another order it is held to the default order like any
    dict; a ``dict(...)`` copy is a plain dict.
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
        Dataset folder name under ``data_path``, e.g. ``"D28"``.
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
        ``mtb.config.DEFAULT.data_path``.
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

    Examples
    --------
    >>> import multibench as mtb
    >>> # diagonal: RNA cells first, then ATAC
    >>> mtb.labels_for("D28")
    {'rna_cty': '/abs/data/D28/rna_cty.csv',
     'atac_cty': '/abs/data/D28/atac_cty.csv'}
    >>> # StabMap: its reference batch first
    >>> mtb.labels_for("D52", "cross", "StabMap")
    {'cty3': '/abs/data/D52/cty3.csv', 'cty1': '/abs/data/D52/cty1.csv',
     'cty2': '/abs/data/D52/cty2.csv'}
    >>> inp = mtb.inputs_for("D52", "cross", "StabMap")
    >>> res = mtb.run("StabMap", "cross", inputs=inp, out_dir="out/StabMap_D52")
    >>> mtb.evaluate(res.output, labels=mtb.labels_for("D52", "cross", "StabMap"))

    Notes
    -----
    **Which files.** The benchmark stores cell-type labels as ``*cty*.csv`` in
    the flat dataset folder, under dataset-specific names (``cty.csv``,
    ``rna_cty.csv``, ``cty1.csv`` ...). All of them are returned except the
    tool-specific ``*_scjoint*`` reformats.

    One exception: with ``category`` and ``method``, a variant that reads
    fewer numbered batches than the folder holds gets only the label files
    of its batches. UINMF's cross variant reads batches 1 and 2, so on
    ``D52`` the dict holds ``cty1`` and ``cty2``.

    An older paired folder may hold ``rna_cty.csv`` instead of ``cty.csv``.
    With ``category`` and ``method``, a variant that reads ``cty`` gets that
    file under the key ``cty``, as ``mtb.inputs_for`` returns it.

    **Default order.** Without ``category`` and ``method``, the order is
    not alphabetical:

    1. ``cty`` (one file, cells already paired) first;
    2. numbered ``cty1, cty2, ..., cty10``, in numeric order (batch
       order);
    3. modality-named files in the canonical modality order **rna, adt, atac**
       (``peak_cty`` counts as atac) - the diagonal methods other than uniPort
       and Seurat_v5 emit the RNA cells first, then the ATAC cells;
    4. any other ``*cty*`` file, alphabetically, last.

    **Per-method order.** With ``category`` and ``method``, each file goes
    where the cells it labels sit in that variant's output: the order of its
    inputs, unless the method's cell order differs. uniPort and Seurat_v5
    put their ATAC cells before their RNA cells.

    StabMap uses a fixed reference batch: batch 3 in cross, batch 1 in
    mosaic (``method_info('StabMap')['supports'][i]['reference_batch']``).
    Its cell order starts with that batch (``cty3, cty1, cty2`` on ``D52``).
    Number the donor you want as reference accordingly.

    The variant is chosen as ``mtb.inputs_for`` does: by ``modalities=``, else
    the category's only variant or the one whose files the folder holds. If
    the choice stays ambiguous, or no label file pairs with the variant's
    roles, the default order is used; ``labels_for`` does not raise
    ``mtb.AmbiguousVariantError``.

    **Passing it to evaluate.** The dict is a ``dict`` subclass that remembers
    its order, so ``mtb.evaluate(labels=...)`` takes it as is. A copy
    (``dict(d)``) or a dict whose keys were reordered goes in as is only in
    the default order; otherwise name the order with ``evaluate``'s
    ``label_order=``. ``list(labels_for(ds).values())`` is the same files as a
    list, in the same order, which ``evaluate`` also accepts.

    **Validation.** ``category`` and ``method`` are validated whenever given:
    ``ValueError`` listing the four categories on a typo, ``KeyError`` with a
    did-you-mean hint for a method. Either one alone changes nothing.

    **Per-batch folders.** A folder written with ``export_dataset(batch=...)``
    holds ``rna1.h5``, ``rna2.h5`` ... and ``cty1.csv``, ``cty2.csv`` ....
    Vertical and diagonal methods read one ``rna.h5`` and one label file.
    With such a ``category``, ``check=None`` warns and ``check=True`` raises.
    Export without ``batch=`` and pass the batch column to
    ``mtb.run_all(batch=...)`` or ``mtb.evaluate(batch=...)``.

    **Paths and names.** Paths are absolute. A dataset spelling that differs
    from the folder only in case (``'d52'`` for ``D52``) is replaced by the
    on-disk spelling, with a ``UserWarning``. The positional order is
    ``(dataset, category, method)``, like ``mtb.inputs_for``, ``mtb.scan`` and
    ``mtb.run_all``.

    See Also
    --------
    mtb.inputs_for : the modality files of the same dataset.

    mtb.evaluate : scores an embedding against these labels.
    """
    if category is not None and (
            isinstance(category, Path)
            or (isinstance(category, str) and (os.sep in category or Path(category).is_dir()))):
        raise TypeError(
            "labels_for: pass data_path= by keyword; the 2nd positional argument "
            "is category since 0.3.0 (order: (dataset, category, method))")
    _old_order_error("labels_for", dataset, category, method)
    if category is not None:
        registry.check_category(category)
    if method is not None:
        registry.check_method(method)
    root = data_path if data_path is not None else config.DEFAULT.data_path
    base = Path(os.path.abspath(os.fspath(root)))
    dataset = canonical_dataset(base, dataset)
    ds_dir = base / dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"no dataset dir at {ds_dir}")
    hint = _per_batch_hint(ds_dir, category) if check is not False else None
    if hint:
        msg = f"labels_for({dataset!r}, {category!r}): {hint}"
        if check:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    files = {p.stem: str(p) for p in ds_dir.glob("*cty*.csv")
             if "scjoint" not in p.name.lower()}
    stems = sorted(files, key=_label_sort_key)
    if method is not None and category is not None:
        spec = registry.get(method)
        mods = registry.normalize_modalities(modalities)
        try:
            cand = select_variant(spec, category, mods, ds_dir=ds_dir)
        except AmbiguousVariantError:
            cand = None                 # still ambiguous: canonical order
        if cand is not None:
            roles = {r for a in cand.args
                     for r in (getattr(a, "roles", None) or [a.role]) if r}
            # an older paired folder names cty.csv rna_cty.csv; a variant that
            # reads the cty role gets it under that key, as inputs_for gives it
            if "cty" in roles and "cty" not in files \
                    and _resolve_role(ds_dir, "cty").name == "rna_cty.csv":
                files = {("cty" if k == "rna_cty" else k): v for k, v in files.items()}
                stems = ["cty" if st == "rna_cty" else st for st in stems]
            # a variant that reads fewer batches than the folder holds (UINMF:
            # batches 1-2) gets only the label files of its batches
            used = _variant_batches(roles)
            if used:
                stems = [st for st in stems
                         if not re.fullmatch(r"cty\d+", st) or int(st[3:]) in used]
            rank = _variant_label_rank(stems, cand)
            if rank is not None:
                stems = sorted(stems, key=lambda st: rank[st])
    return LabelFiles((st, files[st]) for st in stems)
