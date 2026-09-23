"""Resolve a catalog dataset id + method to concrete input file paths."""
from __future__ import annotations

import functools
import os
import re
import warnings
from pathlib import Path

from .. import config
from . import registry
from .schema import AmbiguousVariantError, base_modality, is_label_role

# A few variant roles name a modality *representation* whose on-disk filename
# differs from the role token (e.g. the diagonal ATAC roles). Candidate bases
# are tried in order; the first existing file wins.
_ROLE_FILE_CANDIDATES = {
    "atac_gas": ("atac_gas", "atac"),    # ATAC gene-activity score
    "atac_peak": ("atac_peak", "peak"),  # raw ATAC peaks
}


def _resolve_role(ds_dir: Path, role: str) -> Path:
    """Pick the real on-disk file for a modality role in a flat dataset dir.

    Tries the role token and known aliases. Label roles (anything matching
    ``cty`` or ``label``) try ``.csv`` before ``.h5`` since cell-type files
    are CSV; each extension is tried as ``<base><ext>`` then ``<base>1<ext>``
    (some datasets number their files per batch). Returns the canonical
    ``<base>.h5`` (``.csv`` for label roles) when nothing matches.
    """
    is_label = ("cty" in role) or ("label" in role)
    bases = _ROLE_FILE_CANDIDATES.get(role, (role,))
    exts = (".csv", ".h5") if is_label else (".h5",)
    for base in bases:
        for ext in exts:
            for suffix in (ext, f"1{ext}"):
                p = ds_dir / f"{base}{suffix}"
                if p.exists():
                    return p
    fallback_ext = ".csv" if is_label else ".h5"
    return ds_dir / f"{bases[0]}{fallback_ext}"


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
        return spec.select(category, set(modalities), loose=True)
    candidates = [v for v in spec.variants if v.when.get("category") == category]
    if not candidates:
        raise KeyError(f"{spec.id} has no variant for category={category!r}")
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
        f"that on-disk spelling (this filesystem matched the two case-insensitively; "
        f"frames, out_dir names and saved records carry {same[0]!r} so they line up "
        f"with the stored results and with Linux, where {name!r} would not exist)",
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

    Paths are ABSOLUTE, ready for ``mtb.run(inputs=...)``. Pass
    ``check=True`` to verify the files before a long run.

    Parameters
    ----------
    dataset : str
        Dataset folder name under ``data_path``, e.g. ``"D11"``.
    category : str
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or
        ``cross``.
    method : str
        Registry method id, e.g. ``"Matilda"``.
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
    ``(category, modalities)`` is used: exact tokens first, then ``atac``
    standing for ``atac_gas`` / ``atac_peak`` when that leaves exactly one
    variant.

    Without ``modalities``, a category with one variant uses it. With several,
    the dataset folder decides: the variant whose input files are all present
    is used (Matilda on a ``rna.h5 + adt.h5`` folder is its rna+adt variant).
    When none or several qualify, ``mtb.AmbiguousVariantError`` (a
    ``ValueError``) lists the modality sets and the folder contents and asks
    for ``modalities=``.

    **Modality tokens.** ``method_info(m)['supports']`` lists each variant's
    tokens. Accepted aliases:

    - ``protein`` for ``adt``;
    - ``peak`` / ``peaks`` for ``atac_peak``, ``gas`` / ``gene_activity`` for
      ``atac_gas``;
    - ``atac`` for either ATAC representation role; the one the method wants
      is ``method_info(m)['atac']``.

    An unknown token raises ``ValueError`` naming the vocabulary.

    **Validation.** A misspelt method id raises ``KeyError`` naming the
    closest registry id; an unknown category raises ``ValueError`` listing
    the four.

    **File resolution.** The dataset tree is flat
    (``<data_path>/<dataset>/<file>``). Each role resolves to the file present
    in the folder: the role token, or a known alias (``atac_peak`` ->
    ``peak.h5``, ``atac_gas`` -> ``atac.h5``), also numbered (``<name>1.h5``).
    Label roles look for ``.csv`` first. When nothing matches, the role falls
    back to ``<role>.h5`` (``<role>.csv`` for a label role), a path that does
    not exist (see ``check``).

    A ``data_dir`` role (scBridge) resolves to the first of
    ``<dataset>/processed/`` and the dataset folder that holds a ``.h5ad``
    file; when neither does, to ``processed/`` if that folder exists, else to
    the dataset folder itself.

    **Absolute paths.** Every returned path is ABSOLUTE (a relative
    ``data_path`` is resolved against the current directory), and a
    ``data_dir`` value ends with the path separator. ``mtb.run`` executes the
    method with ``cwd=out_dir``, where a relative path would point at the
    wrong place.

    **The check argument.**

    - ``False`` (default): the best-effort paths, returned silently.
    - ``None``: the same paths, plus a ``UserWarning`` listing the missing ones.
    - ``True``: ``FileNotFoundError`` for a missing input, plus the content
      preflight below.

    The missing-file error or warning names an ATAC-family sibling that is
    present, e.g. ``atac_peak.h5`` when a vertical variant reads ``atac.h5``.

    **Content preflight** (``check=True``). The same checks ``mtb.scan``
    reports per row as ``files_ok`` / ``files_reason``:

    - orientation: ``ValueError`` when ``matrix/data`` is stored cells x
      features;
    - label length: ``ValueError`` when a label CSV has a different number of
      rows than the modality file it labels, including the numbered
      ``cty<i>.csv`` of a cross/mosaic batch (no method takes it as an input
      role, but every evaluation reads it);
    - ``data_dir`` content: ``FileNotFoundError`` when a file scBridge names
      inside ``data_dir`` (``rna.h5``, ``atac_gas.h5``, the two label CSVs) is
      absent.

    **Dataset name case.** A spelling that differs from the folder only in
    case (``'d52'`` for ``D52`` on a case-insensitive filesystem) is replaced
    by the on-disk spelling, with a ``UserWarning``.

    **Argument order.** ``(dataset, category, method)``, the same order
    ``mtb.scan``, ``mtb.run_all`` and ``mtb.labels_for`` use.

    See Also
    --------
    mtb.labels_for : the cell-type label CSVs of the same dataset, in stacking order.

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
    near = _near_miss_hints(ds_dir, missing, category)
    if check:
        if missing:
            raise FileNotFoundError(
                f"{method}/{dataset}/{category}: input files not found on disk: "
                f"{missing}. Available files in {ds_dir}: "
                f"{sorted(q.name for q in ds_dir.glob('*')) if ds_dir.is_dir() else '(dir missing)'}"
                + (" - " + "; ".join(near) if near else "")
            )
        _check_orientation(method, dataset, category, out)
        _check_label_lengths(method, dataset, category, out)
        if "data_dir" in out:
            ok, why = _check_data_dir(variant, out["data_dir"])
            if not ok:
                raise FileNotFoundError(f"{method}/{dataset}/{category}: {why}")
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


def _near_miss_hints(ds_dir: Path, missing: dict, category: str) -> list[str]:
    """For each missing ATAC-family role, name the sibling file that is there.

    The ``atac`` role reads ``atac.h5``; ``atac_gas`` reads ``atac_gas.h5``
    (falling back to ``atac.h5``) and ``atac_peak`` reads ``atac_peak.h5``
    (falling back to ``peak.h5``). A folder exported for the other layout
    would otherwise fail with a bare "atac.h5 not found" although
    ``atac_peak.h5`` sits next to it. Return one hint per such role, e.g.
    ``"atac.h5 not found; found atac_peak.h5 - vertical methods read atac.h5
    (pass the representation this method wants: see method_info(m)['atac'])"``;
    nothing for roles that are not ATAC or have no sibling.
    """
    hints: list[str] = []
    if not ds_dir.is_dir():
        return hints
    for role in missing:
        if base_modality(role) != "atac" or is_label_role(role):
            continue
        bases = _ROLE_FILE_CANDIDATES.get(role, (role,))
        digits = role[len(base_modality(role)):] if role[-1:].isdigit() else ""
        accepted = [f"{b}{digits}.h5" for b in bases]
        found = sorted(f"{b}{digits}.h5" for b in _ATAC_FILE_BASES
                       if f"{b}{digits}.h5" not in accepted
                       and (ds_dir / f"{b}{digits}.h5").is_file())
        if not found:
            continue
        hints.append(
            f"{accepted[0]} not found; found {', '.join(found)} - {category} methods "
            f"read {' or '.join(accepted)} (pass the representation this method "
            f"wants: see method_info(m)['atac'])")
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
                f"in the same order as the cells (see mtb.describe_layout({category!r}))")
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
                    f"exactly one label, in the same order as the cells "
                    f"(see mtb.describe_layout({category!r}))")


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


#: Caveat text appended by scan() when an ``atac_gas`` role resolves to a peak matrix.
PEAK_IN_GAS_CAVEAT = "atac_gas resolved to a PEAK matrix (features look like chr:start-end)"
#: ``.format(role=...)`` templates of the two representation-mismatch caveats
#: reported when the method's wanted ATAC representation is known
#: (``_preflight_caveats(resolved, atac=method_info(m)['atac'])``).
PEAK_FED_TO_GAS_CAVEAT = ("{role} resolved to a PEAK matrix (features look like "
                          "chr:start-end); this method expects GENE ACTIVITY")
GAS_FED_TO_PEAK_CAVEAT = ("{role} resolved to a matrix whose features do not look "
                          "like peaks (chr:start-end); this method expects PEAKS")


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


def _preflight_caveats(resolved, *, atac: str | None = None) -> list[str]:
    """Non-fatal content observations about resolved inputs (never raises).

    Without ``atac`` (wanted representation unknown) one check runs: when the
    ``atac_gas`` role fell back to ``atac.h5`` (no ``atac_gas.h5`` present) and
    >= 90% of the first 50 feature names look like peaks (``chr1:1-200`` /
    ``chr1_1_200``), report :data:`PEAK_IN_GAS_CAVEAT`. Without the wanted
    representation this would also flag the methods that expect peaks behind
    the ``atac_gas`` role name (``atac: peak`` in the registry), so
    :func:`multibench.scan` always passes ``atac=``.

    With ``atac=`` - the representation the method expects,
    ``method_info(m)['atac']`` (``'peak'`` / ``'gene_activity'``) - every
    ATAC-family role (``atac``, ``atac_peak``, ``atac_gas``, ``atac1``...) is
    judged against it, whatever the role name says:

    * ``atac='gene_activity'`` and the file looks like peaks ->
      :data:`PEAK_FED_TO_GAS_CAVEAT` (e.g. Matilda's ``atac`` role on a
      peaks-only multiome folder);
    * ``atac='peak'`` and <= 10% of the features look like peaks ->
      :data:`GAS_FED_TO_PEAK_CAVEAT` (e.g. moETM/scMM/iPOLNG, whose ``atac_gas``
      role resolved to a real gene-activity ``atac_gas.h5``).

    The 10-90% band (mixed names) yields no caveat. The wrong representation
    runs to completion and returns a plausible but wrong embedding, which is
    why these are surfaced at scan time.
    """
    out: list[str] = []
    if atac is None:
        p = Path(resolved.get("atac_gas", ""))
        if p.name and p.stem != "atac_gas":
            frac = _peak_fraction_of(p)
            if frac is not None and frac >= 0.9:
                out.append(PEAK_IN_GAS_CAVEAT)
        return out
    for role, path in resolved.items():
        if is_label_role(role) or base_modality(role) != "atac":
            continue
        frac = _peak_fraction_of(Path(path))
        if frac is None:
            continue
        if atac == "gene_activity" and frac >= 0.9:
            out.append(PEAK_FED_TO_GAS_CAVEAT.format(role=role))
        elif atac == "peak" and frac <= 0.1:
            out.append(GAS_FED_TO_PEAK_CAVEAT.format(role=role))
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
       then ATAC cells; uniPort declares the reverse, see
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
               data_path: Path | str | None = None) -> dict:
    """Return a dataset's cell-type label files, in cell-stacking order.

    Hand the dict to ``mtb.evaluate(labels=...)`` as is; it matches an
    embedding only if that embedding stacks its cells in the dict's order, so
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
        Registry method id; with ``category``, orders the files as that
        variant stacks its cells. ``None`` = the default order.
    modalities : list[str] | set[str] | None
        Modality tokens that pick one of several variants; used only with
        ``category`` and ``method``.
    data_path : Path | str | None
        Data root that holds the dataset folders; ``None`` =
        ``mtb.config.DEFAULT.data_path``.

    Returns
    -------
    dict
        ``{stem: absolute path}`` in cell-stacking order, keyed by filename
        stem (``cty``, ``rna_cty``, ``cty1`` ...).

    Raises
    ------
    TypeError
        ``category`` and ``method`` swapped, or a path passed positionally as ``category``.
    FileNotFoundError
        ``<data_path>/<dataset>`` does not exist.
    ValueError
        Unknown ``category``; the message lists the valid ones.
    KeyError
        Unknown ``method``, or it has no variant for ``category``.

    Warns
    -----
    UserWarning
        ``dataset`` matches a folder only up to letter case.

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
    tool-specific ``*_scjoint*`` reformats. The set of files depends on the
    dataset only, never on ``category`` or ``method``.

    **Default order.** Without ``category`` and ``method``, the order is
    NOT alphabetical:

    1. ``cty`` (one file, cells already paired) first;
    2. numbered ``cty1, cty2, ..., cty10``, ascending NUMERICALLY (batch
       order);
    3. modality-named files in the canonical modality order **rna, adt, atac**
       (``peak_cty`` counts as atac) - the diagonal methods other than uniPort
       emit the RNA cells first, then the ATAC cells;
    4. any other ``*cty*`` file, alphabetically, last.

    **Per-method order.** With ``category`` and ``method``, each file goes
    where the cells it labels sit in that variant's output. That is the
    variant's argument order unless the registry declares another
    (``output.cell_order`` in ``methods.yaml``): StabMap stacks its reference
    batch first (``cty3, cty1, cty2`` on ``D52``), uniPort its ATAC cells
    before its RNA cells.

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
            rank = _variant_label_rank(stems, cand)
            if rank is not None:
                stems = sorted(stems, key=lambda st: rank[st])
    return LabelFiles((st, files[st]) for st in stems)
