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
    ``cty`` or ``label``) try ``.csv`` first since cell-type files are CSV;
    other roles try ``.h5`` then ``.h5`` with a trailing digit (some datasets
    store batched files). Returns the canonical ``<base>.h5`` (or ``.csv``
    for label roles) as a fallback when nothing matches.
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


def _resolve_data_dir(ds_dir: Path, method: str | None = None) -> str:
    """Directory of spatial slices for a registration ``data_dir`` role.

    Spatial-registration methods (PASTE/PASTE2/SPIRAL/GPSA) take a DIRECTORY of
    per-slice ``.h5ad`` files, not a per-feature file. Datasets keep those slices
    under ``<ds_dir>/processed/`` (sometimes directly in ``<ds_dir>/``). Return the
    first dir that actually holds ``*.h5ad`` slices, WITH a trailing separator —
    the upstream scripts string-concatenate ``data_dir + "*.h5ad"``.
    """
    import os
    for cand in (ds_dir / "processed", ds_dir):
        if cand.is_dir() and any(cand.glob("*.h5ad")):
            # SPIRAL derives each slice's cross-slice ID prefix from
            # filename.split('_')[0]; when every slice shares that leading
            # token (e.g. D63's `modified_*.h5ad`) the prefixed obs_names
            # collide and `coord.loc[ann.obs_names]` explodes. Stage a
            # sibling dir of symlinks with a UNIQUE leading token per slice.
            if method == "SPIRAL":
                return _stage_unique_leading_token(cand)
            return os.path.join(str(cand), "")
    # No `.h5ad` slices anywhere: this is a NON-spatial `data_dir` role (e.g.
    # scBridge, which takes the dataset DIRECTORY plus bare filenames). Prefer a
    # real `processed/` if one exists (keeps the spatial error message pointing
    # at the slice dir); otherwise fall back to the dataset dir itself, which
    # always exists, instead of a bogus `<ds>/processed/`.
    proc = ds_dir / "processed"
    return os.path.join(str(proc if proc.is_dir() else ds_dir), "")


def _stage_unique_leading_token(slice_dir: Path) -> str:
    """Return a dir of the same `.h5ad` slices but with a unique leading
    filename token (the part before the first '_').

    SPIRAL builds each slice's cross-slice cell-ID prefix from
    ``filename.split('_')[0]``. Datasets whose slice files share that token
    (D63: ``modified_E14-16h_a_S07.h5ad`` ... all split to ``modified``)
    make SPIRAL's per-cell prefixes collide across slices, which inflates
    ``coord.loc[ann.obs_names]`` into a cartesian product and crashes the
    coordinate-assignment step. If the leading tokens are already unique we
    return the original dir untouched; otherwise we materialize a sibling
    ``<dir>__spiral_uniqtok/`` of SYMLINKS (no data copy, original dir is
    never mutated) whose names start with a unique token derived from the
    trailing token of each stem.
    """
    import os
    files = sorted(p for p in slice_dir.glob("*.h5ad") if p.is_file())
    if not files:
        return os.path.join(str(slice_dir), "")
    lead = [f.name.split("_")[0] for f in files]
    if len(set(lead)) == len(lead):
        return os.path.join(str(slice_dir), "")  # already unique
    staged = slice_dir.parent / (slice_dir.name + "__spiral_uniqtok")
    staged.mkdir(parents=True, exist_ok=True)
    used = set()
    for f in files:
        stem = f.stem
        tok = stem.split("_")[-1] or stem  # trailing token, e.g. S07
        base = tok
        i = 1
        while tok in used:
            i += 1
            tok = f"{base}{i}"
        used.add(tok)
        link = staged / f"{tok}_{f.name}"
        if link.is_symlink() or link.exists():
            try:
                link.unlink()
            except OSError:
                pass
        os.symlink(os.path.realpath(str(f)), str(link))
    return os.path.join(str(staged), "")


def _resolve_variant_inputs(variant, ds_dir: Path, method: str) -> dict:
    """``{role: path}`` for every input role of ``variant`` in ``ds_dir``
    (best effort: a missing file resolves to its canonical name)."""
    # Skip args with a const (they don't need on-disk resolution) and out_dir.
    # an arg may name ONE role (a.role) or GROUP several under one flag (a.roles)
    roles = [r
             for a in variant.args if getattr(a, "const", None) is None
             for r in (getattr(a, "roles", None) or [a.role])
             # "=VALUE" entries are literals emitted verbatim (e.g. scMoMaT's
             # per-batch `None` placeholders), not input roles to find on disk
             if r and not str(r).startswith("=")
             and r not in ("out_dir", "data_dir")]
    out = {role: str(_resolve_role(ds_dir, role)) for role in roles}
    # A `data_dir` role points at the DIRECTORY of spatial slices (registration).
    if any(a.role == "data_dir" for a in variant.args):
        out["data_dir"] = _resolve_data_dir(ds_dir, method)
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

    Shared by :func:`inputs_for`, :func:`labels_for` and
    :func:`multibench.params_for` so the three agree on what "ambiguous" means.

    Parameters
    ----------
    spec : ``MethodSpec``.
    category : validated category token.
    modalities : normalised modality tokens (``registry.normalize_modalities``)
        or None. When given, ``spec.select(..., loose=True)`` is used: exact
        tokens first, then ``atac`` standing for ``atac_gas`` / ``atac_peak``.
    ds_dir : keyword-only; the dataset folder. When ``modalities`` is None and
        several variants exist for ``category``, the ONE whose input files are
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
    ``D52`` folder, so a lower-case id used to pass every check and then
    travel into frames, ``out_dir`` names and saved records as ``'d52'`` - a
    second dataset once concatenated with rows keyed ``'D52'``, and a name
    that fails on Linux. The folder listing is the authority: when no entry
    is spelled exactly ``dataset`` but exactly one differs only in case, that
    entry's spelling is returned (with a ``UserWarning`` saying so).

    Parameters
    ----------
    base : path
        The folder that CONTAINS the dataset folders.
    dataset : str
        The id as the caller wrote it.
    stacklevel : int, keyword-only
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


#: appended to the error when inputs_for's 2nd and 3rd arguments look swapped
SWAPPED_ARGS_HINT = ("inputs_for's argument order is (dataset, method, category) - "
                     "unlike scan/plan/run_all(dataset, category, ...)")


def inputs_for(dataset: str, method: str, category: str,
               modalities: list[str] | set[str] | None = None,
               data_path: Path | str | None = None,
               check: bool | None = None) -> dict:
    """Return ``{modality_role: path}`` for a method's variant on a dataset.

    The dataset tree is **flat** (``<data_path>/<dataset>/<file>``). Each role is
    resolved to the actual file present in that dir (the role token, or a known
    alias such as ``atac_peak``->``peak.h5`` / ``atac_gas``->``atac.h5``),
    falling back to ``<role>.h5`` when no candidate exists. Every returned
    path is ABSOLUTE (``data_path='data'`` relative to the current directory
    included): ``run`` executes the method with ``cwd=out_dir``, where a
    relative path would point at the wrong place. A ``data_dir`` value ends
    with the path separator. Use :func:`labels_for` to get the matching
    cell-type label CSVs.

    Parameters
    ----------
    dataset : dataset folder name under ``data_path``. A spelling that
        differs from the folder only in case (``'d52'`` for ``D52`` on a
        case-insensitive filesystem) is replaced by the on-disk spelling with
        a ``UserWarning`` (:func:`canonical_dataset`).
    method : registry id (``KeyError`` with a did-you-mean hint otherwise).
        When the id given here is a CATEGORY token and ``category`` is a
        method id - the 2nd and 3rd arguments swapped, which the
        ``scan``/``plan``/``run_all`` order ``(dataset, category, ...)``
        invites - the ``KeyError`` says so and shows the corrected call.
    category : ``vertical``/``diagonal``/``mosaic``/``cross`` (validated:
        ``ValueError`` listing the valid tokens on a typo; the same
        swapped-arguments hint when the token is a method id).
    modalities : the variant's modality tokens (see
        ``method_info(m)['supports']``); ``protein`` is accepted for ``adt``,
        and ``atac`` for ANY ATAC representation role (``atac_gas`` /
        ``atac_peak`` - the file is the same ``atac.h5`` on disk; the
        representation the method wants is ``method_info(m)['atac']``).
        Unknown tokens raise ``ValueError`` naming the vocabulary.
    data_path : root that CONTAINS the dataset folder; default
        ``config.DEFAULT.data_path``. A relative root is resolved against the
        current directory, so the returned paths are absolute.
    check : what to do when a resolved path does not exist on disk.
        ``None`` (default) - return the best-effort paths but emit a
        ``UserWarning`` listing the missing ones; ``True`` - raise
        ``FileNotFoundError`` and also run the content preflight: the
        matrix-orientation check (``ValueError`` for a cells x features file),
        the label-length check (``ValueError`` when a label CSV has a different
        number of rows than the modality file it labels - including the
        numbered ``cty<i>.csv`` of a cross/mosaic batch, which no method takes
        as an input role but every evaluation reads) and, for ``data_dir``
        methods, the directory-content check (``FileNotFoundError`` when a
        spatial-registration method finds fewer than two ``*.h5ad`` slices, a
        slice lacks ``obsm['spatial']``, or a slice lacks an ``obs`` column the
        variant declares in ``slice_obs`` - GPSA's ``Ground_Truth``; when
        scBridge's bare filenames are absent). This is what
        :func:`multibench.scan` reports per row as ``files_ok`` /
        ``files_reason``; ``False`` - fully silent.

    Variant selection:
      * If ``modalities`` is given, the variant matching
        ``(category, set(modalities))`` is selected via ``spec.select``
        (exact tokens first; then ``atac`` standing for ``atac_gas`` /
        ``atac_peak`` when that leaves exactly one variant).
      * If ``modalities`` is None and exactly one variant matches ``category``,
        that variant is used.
      * If ``modalities`` is None and MORE than one variant matches
        ``category``, the dataset folder decides: when exactly ONE of them has
        every input file present on disk, it is used (Matilda on a
        ``rna.h5 + adt.h5`` folder is its rna+adt variant). When none or
        several do, ``ValueError`` (:class:`AmbiguousVariantError`) is raised
        listing the available modality-sets and the folder contents and asking
        the caller to disambiguate with ``modalities=``.
      * If no variant matches ``category``, a ``KeyError`` is raised.

    Returns
    -------
    dict
        ``{role: absolute path}`` - one entry per input role of the selected
        variant (``data_dir`` for the directory-fed methods).
    """
    root = data_path if data_path is not None else config.DEFAULT.data_path
    base = Path(os.path.abspath(os.fspath(root)))
    try:
        spec = registry.get(method)
    except KeyError as e:
        if _is_category_token(method) and _is_method_id(category):
            raise KeyError(
                f"unknown method {method!r}: that is a category token and "
                f"{category!r} is a method id - {SWAPPED_ARGS_HINT}; did you mean "
                f"inputs_for({dataset!r}, {category!r}, {method!r})?") from None
        raise e
    try:
        registry.check_category(category)
    except ValueError:
        if _is_method_id(category):
            raise ValueError(
                f"unknown category {category!r}: that is a method id - "
                f"{SWAPPED_ARGS_HINT}; did you mean "
                f"inputs_for({dataset!r}, {category!r}, {method!r})?") from None
        raise
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
        # The default used to hand back phantom paths in silence, so a typo in
        # the dataset folder only surfaced minutes later inside the method's
        # conda env. Warn here; check=True raises, check=False stays quiet.
        warnings.warn(
            f"{method}/{dataset}/{category}: {len(missing)} resolved input path(s) "
            f"do not exist: {missing}"
            + (" (" + "; ".join(near) + ")" if near else "")
            + "; pass check=True to raise, check=False to silence",
            UserWarning, stacklevel=2)
    return out


# Every on-disk base name an ATAC-family role may be looked up under, so a
# near miss can be named: the user exported peaks as atac_peak.h5 (what
# describe_layout('diagonal') says) and a VERTICAL variant asks for atac.h5.
_ATAC_FILE_BASES = ("atac", "atac_peak", "atac_gas", "peak")


def _near_miss_hints(ds_dir: Path, missing: dict, category: str) -> list[str]:
    """For each missing ATAC-family role, name the sibling file that IS there.

    A vertical variant reads ``atac.h5``; the diagonal/mosaic roles read
    ``atac_gas.h5`` (falling back to ``atac.h5``) / ``atac_peak.h5`` (falling
    back to ``peak.h5``). A folder exported for the other layout therefore
    fails with a bare "atac.h5 not found" although ``atac_peak.h5`` sits right
    next to it. Return one hint per such role, e.g. ``"atac.h5 not found;
    found atac_peak.h5 - vertical methods read atac.h5 (pass the representation
    this method wants: see method_info(m)['atac'])"``; nothing for roles that
    are not ATAC or have no sibling.
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


def benchmark_host_only_reason(entrypoint) -> str:
    """Why a script whose entrypoint is an ABSOLUTE path cannot run here.

    Such an entrypoint names one machine's filesystem - the benchmark host -
    so no download can supply it (``MethodSpec.availability ==
    'benchmark-host-only'``; SPIRAL, GPSA). This is the ``files_reason`` text
    ``scan`` should report for those rows; it starts with the machine-readable
    prefix :data:`BENCHMARK_HOST_ONLY`. Returns ``""`` when the path is
    relative or actually exists (then the script IS reachable).
    """
    ep = Path(entrypoint)
    if not ep.is_absolute() or ep.exists():
        return ""
    return (f"{BENCHMARK_HOST_ONLY}: method script not found at {ep} - this "
            f"entrypoint is an absolute path on the benchmark host; the script is "
            f"not part of the public scMultiBench repository, so it cannot be "
            f"fetched (method_info(m)['availability'])")


#: prefix of :func:`benchmark_host_only_reason`
BENCHMARK_HOST_ONLY = "benchmark-host-only: script not published"



@functools.lru_cache(maxsize=512)
def _sniff_h5(path: str, mtime_ns: int):
    """Read (shape, n_features, n_cells) of a canonical .h5, cached by mtime.

    scan() runs the orientation preflight once per (method, variant) row, so
    the same handful of files used to be opened and sniffed dozens of times
    per call; the mtime key keeps the cache honest across rewrites.
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

    Modality files store ``matrix/data`` as (features x cells). Storing it the
    other way round is the easy mistake - cells x features is the scanpy/AnnData
    convention, and describe_layout only says "the matrix under matrix/data"
    without stating an orientation. The file-existence check cannot see it, so
    the method is dispatched, pays its conda-env startup (and for a slow method
    potentially hours of compute) and only then dies inside third-party code
    with an error that does not mention orientation at all.

    ``matrix/features`` and ``matrix/barcodes`` pin the intended orientation
    without reference to the labels, so this is checkable up front. A square
    matrix is genuinely ambiguous and is left alone.
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
    that file as an INPUT role - only the evaluator reads it - so the
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
    alone - no verdict is invented.

    Numbered batch files are checked too: ``cty<i>.csv`` (when present next
    to the modality file) against ``rna<i>.h5`` / ``adt<i>.h5`` /
    ``atac<i>.h5`` of the same batch ``i`` - the layout every cross and mosaic
    method reads, although none of them lists ``cty<i>`` as an input role, so
    a truncated ``cty1.csv`` used to pass :func:`multibench.scan` with
    ``files_ok=True`` and only fail inside ``evaluate`` after the run.
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
    ``processed/`` subdir, so the path ALWAYS exists and existence proves
    nothing. Spatial-registration methods (``output.kind == 'coords'``) need
    >= 2 ``*.h5ad`` slices, each carrying ``obsm['spatial']`` coordinates (the
    upstream scripts glob ``data_dir + '*.h5ad'`` and align ``.obsm['spatial']``)
    and every ``obs`` column the variant declares in ``slice_obs`` (GPSA's
    driver reads ``obs['Ground_Truth']`` from each slice at load, so a folder
    without it used to pass scan and die after the env build); other
    ``data_dir`` methods (scBridge) name their files via ``const`` args.
    """
    d = Path(data_dir)
    if not d.is_dir():
        return False, f"no such directory: {d}"
    if variant.output.kind == "coords":
        slices = sorted(d.glob("*.h5ad"))
        if len(slices) < 2:
            return False, ("spatial registration needs >=2 .h5ad slice files; "
                           f"found {len(slices)} in {d}")
        import h5py
        for sl in slices:
            try:
                with h5py.File(sl, "r") as f:
                    has = "obsm" in f and "spatial" in f["obsm"]
                    obs_cols = set(f["obs"].keys()) if "obs" in f else set()
            except OSError:
                return False, f"{sl.name} is not a readable .h5ad file"
            if not has:
                return False, (f"{sl.name} has no obsm['spatial'] coordinates; "
                               f"registration needs .X plus obsm['spatial'] per slice")
            for col in (getattr(variant, "slice_obs", None) or []):
                if col not in obs_cols:
                    return False, (f"{sl.name} has no obs[{col!r}] column; this method "
                                   f"reads obs[{col!r}] (a region/layer label per spot) "
                                   f"from EVERY slice - add the column to each .h5ad "
                                   f"(see mtb.describe_layout('cross'))")
        return True, ""
    # non-spatial data_dir methods (e.g. scBridge) name their files via `const`
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

    Without ``atac`` (the legacy call) one check runs: when the ``atac_gas``
    role fell back to ``atac.h5`` (no ``atac_gas.h5`` present) and >= 90% of
    the first 50 feature names look like peaks (``chr1:1-200`` /
    ``chr1_1_200``), report :data:`PEAK_IN_GAS_CAVEAT`. Methods that want peaks
    behind that role name (``atac: peak`` in the registry) are fine;
    :func:`multibench.scan` applies the caveat only to the ones that want gene
    activity.

    With ``atac=`` - the representation the method expects,
    ``method_info(m)['atac']`` (``'peak'`` / ``'gene_activity'``) - EVERY
    ATAC-family role (``atac``, ``atac_peak``, ``atac_gas``, ``atac1``...) is
    judged against it, whatever the role name says:

    * ``atac='gene_activity'`` and the file looks like peaks ->
      :data:`PEAK_FED_TO_GAS_CAVEAT` (e.g. Matilda's ``atac`` role on a
      peaks-only multiome folder);
    * ``atac='peak'`` and <= 10% of the features look like peaks ->
      :data:`GAS_FED_TO_PEAK_CAVEAT` (e.g. moETM/scMM/iPOLNG, whose ``atac_gas``
      role resolved to a real gene-activity ``atac_gas.h5``).

    The 10-90% band (mixed names) yields no verdict. The wrong representation
    runs to completion and returns a plausible but WRONG embedding, which is
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
    2. ``cty<N>`` numbered per batch, ascending NUMERICALLY (cty1, cty2, cty10);
    3. ``<modality>_cty`` in the canonical modality order rna, adt, atac
       (``peak_cty`` is treated as atac) - the order in which the diagonal /
       vertical methods stack their cells in the embedding (RNA cells first,
       then ATAC cells);
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
    """Rank label stems by the position of the modality they label in
    ``variant``'s argument order (the order the runner passes the files and
    the method stacks the cells). ``None`` when no stem pairs with a role."""
    roles = variant.roles()
    mods = [r for r in roles if not is_label_role(r) and r != "data_dir"]
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


def labels_for(dataset: str, method: str | None = None, category: str | None = None,
               *, data_path: Path | str | None = None,
               modalities: list[str] | set[str] | None = None) -> dict:
    """Return ``{name: path}`` of the cell-type label CSVs for a dataset, in
    the benchmark's cell-stacking order.

    The benchmark stores cell-type labels as ``*cty*.csv`` in the (flat) dataset
    dir, under dataset-specific names (``cty.csv``, ``rna_cty.csv``, ``cty1.csv``,
    ...). Returns the primary label files (excluding tool-specific ``*_scjoint*``
    reformats), keyed by filename stem.

    **Order of the returned dict** (it is NOT alphabetical): the order in which
    the methods stack the labelled cells in their output, so that
    ``list(labels_for(ds).values())`` can be handed to ``mtb.evaluate(labels=...)``
    for a multi-file dataset -

    1. ``cty`` (one file, cells already paired) first;
    2. numbered ``cty1, cty2, ..., cty10`` ascending NUMERICALLY (batch order);
    3. modality-named files in the canonical modality order **rna, adt, atac**
       (``rna_cty`` before ``atac_cty``; ``peak_cty`` counts as atac) - the
       diagonal methods emit the RNA cells first, then the ATAC cells, so
       ``D28`` returns ``{'rna_cty': ..., 'atac_cty': ...}``;
    4. any other ``*cty*`` file alphabetically, last.

    When ``method`` AND ``category`` are given the variant's OWN argument
    order decides instead (``modalities=`` disambiguates a method with several
    variants in that category; if it is still ambiguous the canonical order
    above is used): a label file is placed where the modality it labels sits
    in the variant's inputs. Pass the single path - or, for a multi-file
    dataset, ``list(labels_for(ds).values())`` in that order - to
    ``mtb.evaluate(labels=...)``: a dict from ``labels_for`` goes in as is,
    in its stacking order. Raises ``FileNotFoundError`` if the dataset dir is
    absent. Paths are absolute.

    Parameters
    ----------
    dataset : dataset folder name under ``data_path``.
    method, category : optional. ``method`` is validated whenever given
        (``KeyError`` with a did-you-mean hint on a typo). When BOTH are given
        the files are ordered by that variant's modality order (see above;
        the variant is chosen like ``inputs_for`` does - ``modalities=``,
        else the one the folder's files satisfy); labels are per DATASET, so
        the SET of files never depends on them.
    data_path : keyword-only. Root that CONTAINS the dataset folder; default
        ``config.DEFAULT.data_path``. For back-compat, a ``Path`` (or a string
        containing a path separator / naming an existing directory) passed as
        the 2nd positional argument is still treated as ``data_path`` - with a
        ``DeprecationWarning``; a bare, non-existent relative name there is taken
        as a method id and ignored.
    modalities : keyword-only; the variant's modality tokens, used only with
        ``method`` + ``category`` to pick one of several variants.
    """
    if method is not None and (
            isinstance(method, Path)
            or (isinstance(method, str) and (os.sep in method or Path(method).is_dir()))):
        warnings.warn("labels_for: pass data_path= by keyword "
                      "(the 2nd positional argument is now `method`)",
                      DeprecationWarning, stacklevel=2)
        data_path, method = method, None
    if method is not None:
        # validated whenever given (it used to be echoed back unchecked unless
        # category was also passed): a typo raises KeyError with a did-you-mean
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
        registry.check_category(category)
        mods = registry.normalize_modalities(modalities)
        try:
            cand = select_variant(spec, category, mods, ds_dir=ds_dir)
        except AmbiguousVariantError:
            cand = None                 # still ambiguous: canonical order
        if cand is not None:
            rank = _variant_label_rank(stems, cand)
            if rank is not None:
                stems = sorted(stems, key=lambda st: rank[st])
    return {st: files[st] for st in stems}
