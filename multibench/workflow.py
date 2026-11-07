"""Run the methods on a dataset folder, score them and keep the result.

    mtb.scan("D11")                                   # what can run here, and how
    res = mtb.run("D11", "vertical", out_dir="out/")  # run them, with metrics
    res.plot()                                        # one figure

Two mistakes that give wrong numbers without an error are handled here. A method
that writes a graph is recorded as such, not scored with embedding metrics (a kNN
index matrix scores ARI ~ 0). Labels must follow the embedding's cell order, which
length alone cannot tell: every candidate order is scored and the best is kept.
"""
from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import shlex
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import _text

from ._settings import config
from ._text import _and_list
from .discover import _REPRESENTATION_TOKENS
from .engine import envs, registry, resolve as _resolve, scripts
from .engine import runner as _runner
from .engine.schema import base_modality as _schema_base
from .engine.runner import run as _run
from .eval import io as _eio, scib as _escib
from .eval.pipeline import LONG_COLUMNS as _LONG_COLUMNS, _long_table as _to_long, _scores

__all__ = ["scan", "run", "RunResult", "load_run"]


def load_run(out_dir, *, methods=None) -> "RunResult":
    """Reload a saved ``run`` result.

    Parameters
    ----------
    out_dir : path-like
        A ``run`` ``out_dir``: the folder holding ``batch_result.json``.
    methods : list[str] | None
        Methods whose records to keep; ``None`` = every record.

    Returns
    -------
    RunResult
        The reloaded result; ``save()`` with no argument writes back to ``out_dir``.

    Raises
    ------
    FileNotFoundError
        ``out_dir`` holds no ``batch_result.json``.
    KeyError
        A name in ``methods`` has no record; the message lists the methods that do.

    Warns
    -----
    UserWarning
        The scores were made by multibench 0.3.x; ``res.rescore()`` scores
        them as this version does.

    Examples
    --------
    >>> import multibench as mtb
    >>> res = mtb.load_run("out/")
    >>> res.summary
    >>> mtb.load_run("out/", methods=["Matilda", "scMM"]).plot()

    Notes
    -----
    A copied ``run`` folder can be re-scored: each record's output folder
    is looked for next to ``batch_result.json`` when its recorded path is
    gone, and the dataset folder under the recorded data root, then
    ``mtb.config.data_path``.
    """
    folder = Path(out_dir).expanduser()
    if not (folder / "batch_result.json").is_file():
        raise FileNotFoundError(f"{folder} holds no batch_result.json. Pass the out_dir "
                                f"that mtb.run wrote.")
    with open(folder / "batch_result.json") as fh:
        blob = json.load(fh)
    records = blob["records"]
    if methods is not None:
        registry.check_id_list(methods, "methods")
        have = [r.get("method") for r in records]
        # a method id in any letter case, or a display name, names its record
        methods = [m if m in have else registry.check_method(m) for m in methods]
        unknown = [m for m in methods if m not in have]
        if unknown:
            raise KeyError(f"no record for {unknown} in {folder}; methods in the "
                           f"tree: {have}")
        records = [r for r in records if r.get("method") in set(methods)]
    long_path = folder / "long.csv"
    long_table = pd.read_csv(long_path) if long_path.exists() else None
    for r in records:
        # a tree copied from another host: the output folder sits next to the JSON;
        # an older record without out_dir: the folder run names
        saved = Path(r.get("out_dir") or f"{r.get('method')}_{blob['dataset']}")
        if not saved.is_absolute():
            saved = folder / saved            # relative to the tree's folder
        if not saved.exists() and (folder / saved.name).is_dir():
            saved = folder / saved.name
        if saved.exists() and str(saved) != str(r.get("out_dir")):
            r["out_dir"] = str(saved.resolve())
        # left as recorded when no root holds the dataset; rescore then names the roots
        root, found = _dataset_root(r, blob["dataset"])
        if found and root is not None:
            r["data_path"] = r["data_root"] = root
        rows = [] if long_table is None else long_table[long_table["method"] == r.get("method")]
        r["_long"] = rows if len(rows) else None
    if _scored_before_040(long_table, records):
        _text.warn(OLD_TREE, UserWarning)
    res = RunResult(records, blob["dataset"], blob["category"], out_dir=folder)
    for r in records:
        res._batch_text(r.get("batch_file"))       # loads the saved batch, when present
    return res


def _run_record_for(path, dataset: str) -> dict | None:
    """The record of the run of ``dataset`` whose method folder holds the file
    ``path`` (``<out_dir>/<method>_<dataset>/embedding.h5``), else ``None``."""
    if not isinstance(path, (str, os.PathLike)):
        return None
    folder = Path(path).expanduser().resolve().parent
    blob_path = folder.parent / "batch_result.json"
    try:
        with open(blob_path) as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict) or blob.get("dataset") != dataset:
        return None
    for r in blob.get("records") or ():
        saved = Path(str(r.get("out_dir") or f"{r.get('method')}_{dataset}"))
        if saved.name == folder.name and registry.method_of(str(r.get("method"))):
            return {"method": registry.method_of(r["method"]),
                    "category": r.get("category") or blob.get("category"),
                    "labels_used": list(r.get("labels_used") or ())}
    return None


#: load_run's warning for a folder scored before 0.4.0
OLD_TREE = ("These scores come from multibench 0.3.x. res.rescore() scores them as "
            "this version does.")


def _scored_before_040(long_table, records) -> bool:
    """True when a saved folder's scores predate 0.4.0: its ``long.csv`` has no
    ``scored_with`` or an older one, or, without ``long.csv``, a scored record
    was made by an older version."""
    from .plot.style import _version_tuple, old_scoring
    if long_table is not None and len(long_table):
        if "scored_with" not in long_table.columns:
            return True
        return bool(old_scoring(long_table["scored_with"]).any())
    versions = [_version_tuple(r.get("multibench_version") or "")
                for r in records if r.get("metrics")]
    return any(v and v < (0, 4, 0) for v in versions)



# --------------------------------------------------------------------------- scan
def _installed_envs() -> frozenset:
    """The envs on this machine, read on every call so that an env installed
    mid-session counts at once; empty when the probe fails."""
    try:
        return frozenset(envs.installed_envs())
    except Exception:      # an env probe must not break discovery
        return frozenset()


def _other_env(env: str):
    """:func:`envs.other_env_prefix`, ``None`` when the probe fails."""
    try:
        return envs.other_env_prefix(env)
    except Exception:      # an env probe must not break discovery
        return None


def _missing_script(variant, method: str) -> str:
    """Why this variant's script or a helper module it imports is missing from
    the scripts checkout, or "" (also when there is no checkout yet: run fetches it)."""
    entry = Path(variant.entrypoint)
    for root in scripts.folders():
        if (root / "tools_scripts").is_dir():
            # a script the package ships is copied in before the run
            shipped = (entry.name in variant.helpers
                       and _runner.shipped_helper(method, entry.name))
            if not (root / entry).exists() and not shipped:
                return (f"method script {entry} is missing from the reference checkout at "
                        f"{root}: " + _runner.MISSING_SCRIPT_FIX)
            gone = [h for h in variant.helpers
                    if not (root / entry).parent.joinpath(h).exists()
                    and not _runner.shipped_helper(method, h)]
            if gone:
                files = " and ".join(gone)
                return (f"{method}'s script imports {files}, which the public scMultiBench "
                        f"repository does not include. Put {files} next to {entry.name}. "
                        f"mtb.method_info({method!r})['setup_hint'] shows how.")
            return ""
    return ""


def _have_their(k: int, n: int) -> str:
    """The verb after ``k of n <rows>``: ``has its`` when ``k`` or ``n`` is 1."""
    return "has its" if 1 in (k, n) else "have their"


def _join_sentences(parts) -> str:
    """Join the ``reason`` parts as sentences, each ending with a period."""
    parts = [p.strip() for p in parts if p and p.strip()]
    return " ".join(p if p[-1] in ".!?" else p + "." for p in parts)


def _variant_rows(category=None):
    for spec in registry.load():
        for v in spec.variants:
            cat = v.when.get("category")
            if category and cat != category:
                continue
            yield spec, v, cat, list(v.when.get("modalities", []))


#: The columns of the full plan, one row per method variant (``_scan``).
PLAN_COLUMNS = ["method", "category", "modalities", "env", "output_kind", "caveat",
                "runnable", "reason", "files_ok", "files_reason", "env_ok", "env_reason",
                "command"]

#: The columns of ``scan``, one row per method; ``category`` only without one.
SCAN_COLUMNS = ["method", "category", "modalities", "runnable", "reason", "env", "caveat",
                "command"]

#: The ``out_dir`` of scan's ``command`` column.
OUT_DIR_PLACEHOLDER = "<out_dir>"


def _env_hint(env: str, method: str, other=None) -> str:
    """The ``env_reason`` text: the install command first, so a clipped reason
    still holds it; off Linux only the platform sentence; ``other`` = the
    prefix of a same-named env multibench did not install."""
    if envs.linux_only_text():
        return f"Environment {env} runs only on Linux, not on this computer."
    if other is not None:
        return envs.other_env_text(env, other, f"run mtb.env.install([{method!r}]).")
    return f"Environment {env} is not installed. Run mtb.env.install([{method!r}])."


#: ``.format(method=)`` caveat of a GPU-only row that ``scan(assume_gpu=True)``
#: keeps runnable.
GPU_NODE_CAVEAT = ("{method} needs an NVIDIA GPU. This check assumes the job runs on a "
                   "GPU node.")

#: What the first line of ``scan`` adds off Linux.
LINUX_ONLY_CHECK = ("Methods run only on Linux. On this computer you can "
                    "check files, score embeddings and plot.")
_DOCTOR = "mtb.env.install(dry_run=True) checks the environments."


def _first_sentence(text: str) -> str:
    """The first sentence of ``text``; a dotted file name does not end it."""
    m = re.match(r"(.+?[.!?])(\s|$)", text.strip(), re.S)
    return (m.group(1) if m else text).strip()


def _reads_one_of_numbered(missing, category: str) -> bool:
    """A diagonal method that reads one file of a folder of numbered batches
    (``rna.h5`` next to ``rna1.h5``): scan's folder line covers it."""
    return category == "diagonal" and any(
        p.suffix == ".h5" and (p.parent / f"{p.stem}1.h5").is_file() for p in missing)


def _missing_paths(spec, category: str, mods: list, dataset: str, data_path) -> dict:
    """``{role: path}`` of the input files of a variant that are not on disk; ``{}``
    when its inputs cannot be resolved (the caller keeps the exception text)."""
    try:
        paths = _resolve.inputs_for(dataset, category, spec.id, modalities=mods or None,
                                    data_path=data_path, check=False)
    except Exception:           # noqa: BLE001 - fall back to the exception text
        return {}
    return {r: Path(p) for r, p in paths.items() if not Path(p).exists()}


def _missing_files_reason(spec, category: str, missing: dict) -> str:
    """The ``reason`` for the input files ``missing`` (:func:`_missing_paths`), or
    ``""``: ``SCALEX needs gene-activity ATAC (atac_gas.h5). rna_cty.csv is missing.``"""
    from .engine.schema import is_label_role
    if not missing:
        return ""
    if _reads_one_of_numbered(missing.values(), category):
        names = [p.name for p in missing.values() if p.suffix == ".h5"]
        return f"{spec.id} reads {_and_list(names)}."
    parts = []

    def kind(role):
        """The ATAC representation a role reads: its own name, else the method's."""
        stem = role.rstrip("0123456789")
        if stem == "atac_peak":
            return "peak"
        if stem == "atac_gas" and category != "vertical":
            return "gene-activity"
        return {"peak": "peak", "gene_activity": "gene-activity"}.get(spec.atac or "", "")

    def batch_of_many(p):
        """A numbered file whose other batches are here: only this batch is missing."""
        stem = p.stem.rstrip("0123456789")
        return stem != p.stem and any(q.name != p.name
                                      for q in p.parent.glob(f"{stem}[0-9]*.h5"))
    roles = [r for r, p in missing.items() if kind(r) and _schema_base(r) == "atac"
             and not is_label_role(r) and not batch_of_many(p)]
    for want in dict.fromkeys(map(kind, roles)):
        # a cross role atac<i> as the file export writes, atac_peak<i>.h5
        files = [f"atac_peak{r[4:]}.h5" if category == "cross" and r.rstrip("0123456789")
                 == "atac" else missing[r].name for r in roles if kind(r) == want]
        parts.append(f"{spec.id} needs {want} ATAC ({', '.join(files)})")
    other = [p.name for r, p in missing.items() if r not in roles]
    if other:
        parts.append(f"{', '.join(other)} {'is' if len(other) == 1 else 'are'} missing")
    # rna1.h5 + rna2.h5 where a vertical / diagonal method reads one rna.h5
    batch_hint = _resolve._per_batch_hint(
        next(iter(missing.values())).parent, category,
        {st for r in missing for st in _resolve._role_stems(r)[0]})
    if batch_hint:
        parts.append(batch_hint[:1].upper() + batch_hint[1:])
    return _join_sentences(parts)


_ABS_PATH_RE = re.compile(r"(?<![\w./-])/(?:[^\s'\"\[\]{}(),:;]+/)+[^\s'\"\[\]{}(),:;]*")
_EXC_PREFIX_RE = re.compile(r"^[A-Z]\w*(?:Error|Exception|Warning): ")


def _short_reason(text: str, method: str, dataset: str) -> str:
    """A ``files_reason`` part as the ``reason`` column shows it: without the
    exception class, directories, or the ``<method> reads <file> of <dataset>`` opening."""
    if not text:
        return text
    part = _EXC_PREFIX_RE.sub("", text)
    part = re.sub(rf"^{re.escape(method)} reads (\S+) of {re.escape(str(dataset))}, which ",
                  r"\1 ", part)
    return _ABS_PATH_RE.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], part)


#: The representation-mismatch caveats of ``_resolve._preflight_caveats`` after the
#: method name; groups: wanted, file, held.
_WRONG_ATAC_BODY = (r"needs (gene-activity|peak) ATAC\. The features of (\S+) (?:do not )?"
                    r"look like chr:start-end, so it (?:seems to hold|holds) "
                    r"(peaks|gene activity)\.")
_WRONG_ATAC_RE = re.compile(r"^\S+ " + _WRONG_ATAC_BODY)
#: What a peak-name caveat (``_resolve.PEAK_NAMES_CAVEAT``) says after the method
#: name.
PEAK_NAMES_REASON = "reads peak names such as "
_PEAK_NAMES_RE = re.compile(r"^\S+ " + re.escape(PEAK_NAMES_REASON)
                            + r"(\S+)\. (\S+) holds other names")
def _blocking_caveats(caveats) -> list[str]:
    """The caveats that block a row: the ATAC file holds the other
    representation, or peak names the method's script cannot read."""
    return [c for c in caveats if _WRONG_ATAC_RE.match(c) or _PEAK_NAMES_RE.match(c)]


def _run_caveat(caveat) -> str:
    """A scan ``caveat`` without the trailing note on a prepared input, which
    ``run`` handles itself."""
    text = str(caveat or "")
    cut = _runner._prepared_at(text)
    return text[:cut].rstrip() if cut >= 0 else text


def _would_run(row) -> bool:
    """Whether run would run this scan row: runnable, or blocked only by its env."""
    return bool(row["files_ok"]) and (not row["reason"]
                                      or row["reason"] == row.get("env_reason"))


def _modality_matcher(modalities, named=None):
    """A test ``(spec, variant_modalities) -> bool`` for ``scan(modalities=)``; the
    rules are those of the scan docstring. ``named`` are the ``methods=`` ids."""
    from .discover import _modality_filter
    from .engine.schema import modality_family
    _modality_filter(modalities)      # ValueError / TypeError as in find_methods
    exact = set(registry.normalize_modalities(modalities))
    spelled = {str(t).lower() for t in modalities}
    named = set(named or ())
    tokens = []                             # (family, representation, is a base token)
    for given in modalities:
        token = registry.MODALITY_ALIASES.get(str(given).lower(), str(given))
        family = modality_family(token)
        tokens.append((family, _REPRESENTATION_TOKENS.get(token.rstrip("0123456789")),
                       family in ("rna", "adt", "atac")))
    reps = _token_reps(modalities)

    def covers(token, role):
        family, _rep, is_base = token
        return (_schema_base(role) == family) if is_base else modality_family(role) == family

    def match(spec, mods) -> bool:
        if not mods or not tokens:
            return not mods and not tokens
        if spec.id in named and spelled == set(mods):
            return True
        # the representation before the spelling: moETM's atac_gas role reads peaks
        if len(reps) == 1 and spec.atac not in reps:
            return False
        if exact == set(mods):
            return True
        # a variant spelled as asked is the one selected (scJoint rna+atac_gas, not
        # its numbered batches)
        cats = {v.when.get("category") for v in spec.variants
                if list(v.when.get("modalities", [])) == list(mods)}
        if any(set(v.when.get("modalities", [])) == exact for v in spec.variants
               if v.when.get("category") in cats):
            return False
        if any(rep and spec.atac != rep for _f, rep, _b in tokens):
            return False
        return (all(any(covers(t, r) for r in mods) for t in tokens)
                and all(any(covers(t, r) for t in tokens) for r in mods))
    return match


def _token_reps(modalities) -> set:
    """The ATAC representations (``peak``, ``gene_activity``) the tokens name."""
    out = set()
    for given in modalities or ():
        token = registry.MODALITY_ALIASES.get(str(given).lower(), str(given))
        rep = _REPRESENTATION_TOKENS.get(token.rstrip("0123456789"))
        if rep:
            out.add(rep)
    return out


def _no_variant_error(category, methods, modalities) -> ValueError:
    """The ``ValueError`` for named methods without a row, or a selection that
    matches no variant: what does not run, then why."""
    from .engine.schema import no_category_message
    where = f"{category} data" if category else "any category"
    methods = list(methods or ())
    cats = {m: dict.fromkeys(c for s, _v, c, _m in _variant_rows() if s.id == m and c)
            for m in methods}
    # no variant under this category, whatever the modalities
    off = [m for m in methods
           if modalities is None or (category and category not in cats[m])]
    dropped = [m for m in methods if m not in off]
    parts = [no_category_message(m, cats[m], category) for m in off]
    if modalities is not None and (dropped or not methods):
        mods = "+".join(map(str, modalities or ())) or "a data folder"
        parts.append(f"{_and_list(dropped)} {'does' if len(dropped) == 1 else 'do'} not read "
                     f"{mods} in {where}." if dropped else
                     f"No method reads {mods} in {where}.")
        one = dropped[0] if len(dropped) == 1 else None
        parts.append(f"mtb.method_info({one!r})['supports'] lists what {one} reads."
                     if one else "mtb.method_info(m)['supports'] lists what each method reads.")
    return ValueError(" ".join(parts) or f"No method runs on {where}.")


def _command_line(method: str, category: str, inputs: dict, *, out_dir, dataset: str,
                  params: dict | None, gpu: bool | None = None) -> tuple[str, list[str]]:
    """The shell line ``run`` would execute for one scan row, and the notes on
    files that ``run`` writes first; ``(no preview: ...)`` when it cannot be built."""
    try:
        # runner.preview, not _run: a preview is not a dispatch
        argv, prepared = _runner.preview(method, category, inputs=inputs,
                                         out_dir=Path(out_dir) / f"{method}_{dataset}",
                                         params=params, gpu=gpu)
        return shlex.join(argv), [prepared] if prepared else []
    except Exception as e:  # noqa: BLE001 - a preview must never abort the scan
        return f"(no preview: {type(e).__name__}: {e})", []


#: what :func:`_scan` keeps of each row besides the shown columns (its second frame)
_PRIVATE_COLUMNS = ["_other", "_label_inputs", "_folder_caveat"]

_FAMILY_WORDS = {"rna": "RNA", "adt": "ADT", "atac": "ATAC"}


def _shown_modalities(df: "pd.DataFrame") -> "pd.DataFrame":
    """``df`` with a folder-reading row's ``(data_dir)`` shown as the files it
    reads (scBridge: ``rna+atac_gas``)."""
    def files(method, category):
        v = next((v for v in registry.get(method).variants
                  if v.when.get("category") == category and not v.when.get("modalities")),
                 None)
        return "+".join(v.shown_modalities if v else ()) or "(data_dir)"
    if "modalities" not in df.columns or not (df["modalities"] == "(data_dir)").any():
        return df
    cats = df["category"] if "category" in df.columns else [df.attrs.get("category")] * len(df)
    return df.assign(modalities=[files(m, c) if mods == "(data_dir)" else mods
                                 for m, c, mods in zip(df["method"], cats, df["modalities"])])


def _row_modalities(text) -> list:
    """A scan row's ``modalities`` as a list; ``[]`` for a row that reads a folder."""
    return [] if text == "(data_dir)" else str(text).split("+")


def _family(role) -> str:
    """The modality family of a role: ``rna2`` -> ``rna``, ``atac_peak`` -> ``atac``."""
    return _schema_base(str(role)).rstrip("0123456789")


def _row_families(modalities) -> set:
    """The families a scan row reads; none for a row that reads a folder."""
    roles = _row_modalities(modalities)
    return {_family(r) for r in roles} & set(registry._BASE_MODALITIES)


def _role_files(ds_dir: Path, category: str | None) -> dict:
    """``{file name: family}`` of the matrix files a role of some variant of
    ``category`` resolves to; an absent ATAC role counts its batch's other ATAC name."""
    from .engine.schema import is_label_role
    files = {}
    for spec, v, *_ in _variant_rows(category):
        for role, path in _resolve._resolve_variant_inputs(v, ds_dir, spec.id).items():
            fam, p = _family(role), Path(path)
            if fam not in registry._BASE_MODALITIES or is_label_role(role):
                continue
            if p.name not in files and p.exists():
                files[p.name] = fam
            elif fam == "atac" and not p.exists():
                digits = role[len(role.rstrip("0123456789")):]
                for base in _resolve._ATAC_FILE_BASES:
                    q = ds_dir / f"{base}{digits}.h5"
                    if q.name not in files and q.is_file():
                        files[q.name] = fam
    return files


def _held_families(ds_dir: Path, category: str | None) -> list:
    """The families the folder holds a file of (:func:`_role_files`)."""
    held = set(_role_files(ds_dir, category).values())
    return [f for f in registry._BASE_MODALITIES if f in held]


def _listed(df: "pd.DataFrame", need, held, methods) -> list:
    """One flag per row: whether a scan without ``modalities`` lists it (the rule
    of the scan docstring, Listed rows). ``need`` is each row's families."""
    keep = [n <= set(held) or bool(ok) for n, ok in zip(need, df["files_ok"])]
    if methods:
        left = set(df["method"][keep])
        keep = [k or m not in left for k, m in zip(keep, df["method"])]
    if not held or not any(keep):
        return [True] * len(df)
    return keep


def _holds_sentence(dataset: str, ds_dir: Path, held, category: str | None = None) -> str:
    """``D52 holds RNA and ADT, in batches 1-3.``"""
    if not held:
        if not _cells_per_file(ds_dir):
            return f"{dataset} holds no RNA, ADT or ATAC file."
        return (f"{dataset} holds no RNA, ADT or ATAC file under a name that a "
                f"{category + ' ' if category else ''}method reads.")
    nums = sorted(_resolve._folder_batches(ds_dir))
    where = (f", in {'batch' if len(nums) == 1 else 'batches'} {_resolve._span(nums)}"
             if nums else "")
    return f"{dataset} holds {_and_list(_FAMILY_WORDS[f] for f in held)}{where}."


def _not_listed_note(df: "pd.DataFrame", dataset: str, ds_dir: Path,
                     category: str | None = None) -> str:
    """What the folder holds when variants for numbered batches it lacks are not
    listed, else ``""``. The rows that read a modality the folder lacks go
    unsaid: a named method among them has its own error."""
    if not df.attrs.get("not_listed_batch"):
        return ""
    return (_holds_sentence(dataset, ds_dir, df.attrs["held"], category)
            + ' Variants for numbered batches are not listed. modalities="all" lists them.')


def _batch_listed(df: "pd.DataFrame", ds_dir: Path, keep, methods=None) -> list:
    """One flag per row: ``False`` for a row for numbered batches when the
    folder holds one batch (no numbered file); all ``True`` when that would
    leave no row. A named method with no other row keeps its rows, as in
    :func:`_listed`."""
    have = _resolve._folder_batches(ds_dir)
    if have:
        # a numbered folder may lack a batch file: those rows say which
        return [True] * len(df)
    flags = []
    for mods, ok in zip(df["modalities"], df["files_ok"]):
        nums = {int(t[len(t.rstrip("0123456789")):]) for t in str(mods).split("+")
                if t[-1:].isdigit()}
        flags.append(bool(ok) or not nums or nums <= have)
    if methods:
        left = set(df["method"][[k and f for k, f in zip(keep, flags)]])
        flags = [f or m not in left for f, m in zip(flags, df["method"])]
    if not any(k and f for k, f in zip(keep, flags)):
        return [True] * len(df)
    return flags


def _category_hint(df: "pd.DataFrame", category: str | None) -> str:
    """Without a category: when only one category's rows have their files, the
    call that names it; ``""`` otherwise."""
    if category is not None or "category" not in df.columns or df.empty:
        return ""
    fit = set(df.loc[df["files_ok"].astype(bool), "category"])
    if len(fit) != 1 or df["category"].nunique() < 2:
        return ""
    (cat,) = fit
    return f'The files fit {cat}: pass category="{cat}".'


def scan(dataset: str, category: str | None = None, *,
         methods: list[str] | None = None,
         modalities: list[str] | str | None = None,
         params: dict | None = None,
         assume_gpu: bool = False,
         verbose: bool = True,
         command: bool = False) -> pd.DataFrame:
    """Check which methods can run on a dataset, and how ``run`` would run them.

    Nothing is executed and no file is written. Call it first on a new
    dataset: it prints what the folder holds and why each method that
    cannot run cannot.

    Parameters
    ----------
    dataset : str
        Dataset folder: its path, or its name under
        ``mtb.config.data_path``.
    category : str | None
        Integration category to scan; ``None`` = the benchmark's category of
        a dataset it scored (``D11`` is vertical), else all four.
    methods : list[str] | None
        Method ids to include, as a list; ``None`` = every method.
    modalities : list[str] | str | None
        Modality tokens of one combination, e.g. ``["rna", "adt"]``; ``None`` =
        the combinations whose modalities the folder holds; ``"all"`` = every
        combination.
    params : dict | None
        Hyperparameters per method, as ``run`` takes them; they are checked
        and appear in ``command`` when it is asked for.
    assume_gpu : bool
        ``True`` = skip this host's GPU test, to check a job for a GPU node
        from a login node.
    verbose : bool
        Print what the folder holds, the cells per file and each reason.
    command : bool
        ``True`` = add the ``command`` column: the line ``run`` would execute,
        with ``<out_dir>`` for the output folder.

    Returns
    -------
    pandas.DataFrame
        One row per method (per method and category when ``category`` is
        ``None``), runnable methods first: ``method``, ``modalities`` (of the
        variant ``run`` would run, ``+``-joined), ``runnable``, ``reason``
        (why it cannot run), ``env``, ``caveat`` (what to know before
        running it), and ``command`` with ``command=True``.

    Raises
    ------
    FileNotFoundError
        The dataset folder does not exist; the message lists the folders present.
    ValueError
        Unknown ``category`` or modality token, or a method in ``methods``
        without a variant there.
    KeyError
        Unknown method id, or a ``params`` key the method does not accept.

    Examples
    --------
    >>> import multibench as mtb
    >>> mtb.scan("D11", "vertical")
    >>> mtb.scan("data/MYCITE", "vertical", methods=["Matilda"],
    ...          params={"Matilda": {"epochs": 10}}, command=True).command[0]

    See Also
    --------
    mtb.run : runs the runnable methods.
    """
    if category is None:
        category = known_category(dataset)
    df, private = _scan(dataset, category, methods=methods, modalities=modalities,
                        params=params, assume_gpu=assume_gpu)
    out = _per_method(df, private, verbose=verbose,
                      nested=not isinstance(modalities, (list, tuple)))
    # a long line, for troubleshooting: shown when asked for
    return out if command else out.drop(columns=["command"])


def known_category(dataset) -> str | None:
    """The one category the benchmark scored the dataset ``dataset`` names in
    (``catalog.datasets()``), or ``None`` for a dataset of its own."""
    from ._data import catalog
    name = _resolve.split_dataset(dataset, None)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = catalog.datasets()
    cats = rows.loc[rows["dataset"] == str(name), "category"].dropna()
    return cats.iloc[0] if len(cats) == 1 and ";" not in cats.iloc[0] else None


def own_batches_only(dataset) -> None:
    """``ValueError`` for ``batch=`` on a benchmark dataset: a dataset has one
    batch label, and a benchmark dataset's is its label files."""
    if known_category(dataset) is not None:
        name = _resolve.split_dataset(dataset, None)[0]
        raise ValueError(f"{name} is a benchmark dataset: its batches are its label files. "
                         f"Pass batch= only for a dataset of your own.")


def _scan(dataset: str, category: str | None = None, *, methods=None, modalities=None,
          data_path=None, out_dir=OUT_DIR_PLACEHOLDER, params: dict | None = None,
          assume_gpu: bool = False) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """The full plan (every column of :data:`PLAN_COLUMNS`), and a second frame
    aligned with it: ``_other`` (the
    reason without the off-Linux sentence), ``_label_inputs`` and ``_folder_caveat``."""
    dataset, data_path = _resolve.split_dataset(dataset, data_path)
    methods, every, modalities, params = _checked_args(category, methods, modalities,
                                                       params, "scan")
    wanted = _modality_matcher(modalities, methods) if modalities is not None else None
    base = Path(data_path) if data_path is not None else config.data_path
    dataset = _resolve.canonical_dataset(base, dataset)
    ds_dir = base / dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(_resolve.missing_dataset_message(base, dataset))
    installed = _installed_envs()
    repo = _runner._repo_root_no_fetch()
    label_files = {q.resolve(): q.name for q in _label_files(dataset, data_path)}
    # the real run refuses these scripts, so every row is blocked
    wrong_ref = scripts.scripts_ref_problem(repo) or scripts.scripts_folder_problem(repo) or ""
    rows = []
    off_linux = bool(envs.linux_only_text())
    others: dict = {}          # env -> a same-named env multibench did not install
    for spec, v, cat, mods in _variant_rows(category):
        if methods is not None and spec.id not in methods:
            continue
        if wanted is not None and not wanted(spec, mods):
            continue
        rec = {"method": spec.id, "category": cat, "modalities": "+".join(mods) or "(data_dir)",
               "env": envs.group_for(spec.id), "output_kind": v.output.kind,
               "caveat": "", "runnable": False, "reason": "",
               "files_ok": True, "files_reason": "", "env_ok": True, "env_reason": "",
               "command": ""}
        # check 1, files: both the script and the dataset files, so neither hides
        # the other. files_reason keeps the full text; reason the short form.
        # folder_only: said by scan's folder line, not on the method's own line
        file_problems, short_problems, folder_only = [], [], []
        why_script = _missing_script(v, method=spec.id)
        if why_script:
            file_problems.append(why_script)
            short_problems.append(_short_reason(why_script, spec.id, dataset))
        got, wrong_atac = None, ""
        rec["_label_inputs"], rec["_folder_caveat"] = (), ""
        try:
            got = _resolve.inputs_for(dataset, cat, spec.id, modalities=mods or None,
                                      data_path=data_path, check=True)
            rec["_label_inputs"] = _label_inputs(got, v, label_files)
            extra = _resolve._preflight_caveats(got, atac=spec.atac, category=cat,
                                                method=spec.id)
            if extra:
                rec["caveat"] = _join_sentences([rec["caveat"], *extra])
                rec["_folder_caveat"] = _join_sentences(extra)
            wrong_atac = next(iter(_blocking_caveats(extra)), "")
        except Exception as e:  # missing files, no variant, bad layout
            full = f"{type(e).__name__}: {e}"
            file_problems.append(full)
            missing = _missing_paths(spec, cat, mods, dataset, data_path)
            short = (_missing_files_reason(spec, cat, missing)
                     or _short_reason(full, spec.id, dataset))
            short_problems.append(short)
            if _reads_one_of_numbered(missing.values(), cat):
                folder_only.append(short)
        if file_problems:
            rec["files_ok"], rec["files_reason"] = False, " ".join(file_problems)
        env_hint = gpu_hint = ""
        if rec["env"] and (off_linux or rec["env"] not in installed):
            rec["env_ok"] = False
            if not off_linux and rec["env"] not in others:
                others[rec["env"]] = _other_env(rec["env"])
            env_hint = _env_hint(rec["env"], spec.id, others.get(rec["env"]))
        # a script that calls CUDA unconditionally: run() raises this sentence
        if spec.requires_gpu and not envs.host_has_gpu():
            if assume_gpu:
                rec["caveat"] = _join_sentences([rec["caveat"],
                                               GPU_NODE_CAVEAT.format(method=spec.id)])
            else:
                rec["env_ok"] = False
                # off Linux the platform already blocks the method
                gpu_hint = "" if off_linux else spec.requires_gpu_reason or ""
        rec["env_reason"] = " ".join(r for r in (env_hint, gpu_hint) if r)
        rec["runnable"] = bool(rec["files_ok"] and rec["env_ok"] and not wrong_atac
                               and not wrong_ref)
        # the ATAC reason comes last
        rec["reason"] = _join_sentences([wrong_ref, *short_problems, rec["env_reason"],
                                         wrong_atac])
        rec["_other"] = _join_sentences([wrong_ref, *(p for p in short_problems
                                                      if p not in folder_only),
                                         gpu_hint if off_linux else rec["env_reason"],
                                         wrong_atac])
        # the method's command, also for an env-blocked row; setup steps go to caveat
        if rec["files_ok"] and got is not None:
            rec["command"], prepared = _command_line(spec.id, cat, got, out_dir=out_dir,
                                                     dataset=dataset,
                                                     params=params.get(spec.id),
                                                     gpu=True if assume_gpu else None)
            # the setup hint's first sentence; a same-cells hint (Seurat_v5) is
            # checked on the files above instead
            notes = [n if n != spec.setup_hint else
                     "" if spec.id in _resolve._SAME_CELL_ROLES else _first_sentence(n)
                     for n in _runner.script_notes(spec, v, repo)]
            if v.size_hint in notes:
                # the folder's size against the method's, or nothing when it is large enough
                small = _too_small(v, got, ds_dir)
                notes[notes.index(v.size_hint)] = (
                    "" if small is None else v.size_hint if small is True
                    else f"{spec.id} needs about {small}: leave {spec.id} out.")
            notes = [n for n in notes if n] + prepared
            if notes:
                rec["caveat"] = _join_sentences([rec["caveat"], *notes])
        rows.append(rec)
    df = pd.DataFrame(rows, columns=[*PLAN_COLUMNS, *_PRIVATE_COLUMNS])
    df = df.sort_values(["runnable", "category", "method"],
                        ascending=[False, True, True]).reset_index(drop=True)
    private = pd.concat([df.pop(c) for c in _PRIVATE_COLUMNS], axis=1)
    if methods is not None:
        # a named method with no row is a request problem, also next to others
        missing = [m for m in dict.fromkeys(methods) if m not in set(df["method"])]
        if missing or df.empty:
            raise _no_variant_error(category, missing or methods, modalities)
    if params:
        _check_param_keys(df, params)
    # last, so every check above saw every variant
    held = _held_families(ds_dir, category)
    need = [_row_families(m) for m in _shown_modalities(df)["modalities"]]
    keep = ([True] * len(df) if every or modalities is not None
            else _listed(df, need, held, methods))
    left_out = [(m, [f for f in registry._BASE_MODALITIES if f in n and f not in held])
                for m, n, k in zip(df["method"], need, keep) if not k]
    # a variant for numbered batches the folder does not have (a batched
    # variant next to a one-batch folder) is not listed either
    batch_keep = ([True] * len(df) if every or modalities is not None
                  else _batch_listed(df, ds_dir, keep, methods))
    n_batch_out = sum(k and not b for k, b in zip(keep, batch_keep))
    keep = [k and b for k, b in zip(keep, batch_keep)]
    if not all(keep):
        df = df[keep].reset_index(drop=True)
        private = private[keep].reset_index(drop=True)
    df.attrs.update(held=held, not_listed=len(left_out), not_listed_batch=n_batch_out,
                    not_held=[f for f in registry._BASE_MODALITIES
                              if any(f in lack for _, lack in left_out)])
    private.attrs.update(dataset=dataset, folder=ds_dir, category=category)
    return df, private


def _too_small(variant, inputs: dict, ds_dir: Path):
    """``"2,000 genes, and rna.h5 has 408"`` when the files a row reads are below a
    size its ``size_hint`` names (``min_cells``, ``min_cells_per_batch``,
    ``min_genes``), ``True`` when a size cannot be read, else ``None``."""
    files = [Path(str(p)).name for p in inputs.values() if str(p).endswith(".h5")]
    sizes = _cells_per_file(ds_dir)
    # one count per label file: rna.h5 and atac.h5 of cty.csv are the same cells
    label_of = {m.name: lab.stem for lab in ds_dir.glob("*cty*.csv")
                for m in _label_data_files(lab)}
    cells: dict = {}
    for f in files:
        cells[label_of.get(f, f)] = max(cells.get(label_of.get(f, f), 0), sizes.get(f) or 0)
    total = sum(cells.values())
    if variant.min_cells and total < variant.min_cells:
        return f"{variant.min_cells:,} cells, and this folder has {total:,}" if total else True
    if variant.min_cells_per_batch:
        per: dict = {}
        for f in files:
            m = re.search(r"(\d+)\.h5$", f)
            if m:
                per[m.group(1)] = max(per.get(m.group(1), 0), sizes.get(f) or 0)
        if not per or not min(per.values()):
            return True
        b, n = min(per.items(), key=lambda kv: kv[1])
        if n < variant.min_cells_per_batch:
            return (f"{variant.min_cells_per_batch:,} cells in every batch, and batch {b} "
                    f"of this folder has {n:,}")
    if variant.min_genes:
        for f in files:
            if f.startswith("rna"):
                path = ds_dir / f
                sniff = path.is_file() and _resolve._sniff_h5(str(path), path.stat().st_mtime_ns)
                if not sniff:
                    return True
                if sniff[1] < variant.min_genes:
                    return f"{variant.min_genes:,} genes, and {f} has {sniff[1]:,}"
    return None


def _row_paths(row, dataset, data_path) -> dict:
    """The unchecked ``inputs_for`` paths of a scan row, without its warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _resolve.inputs_for(dataset, row["category"], row["method"],
                                   modalities=_row_modalities(row["modalities"]) or None,
                                   data_path=data_path, check=False)


def _missing_inputs(row, dataset, data_path) -> int:
    """How many input files of a scan row are not in the folder."""
    if row["files_ok"]:
        return 0
    try:
        paths = _row_paths(row, dataset, data_path)
    except Exception:  # noqa: BLE001 - unresolvable: the last row to speak for its method
        return 10 ** 6
    return sum(not Path(p).exists() for p in paths.values())


def _cells_per_file(ds_dir: Path) -> dict:
    """``{file name: cells}`` of the layout's matrix files in ``ds_dir``, in layout
    order; ``None`` for a file that cannot be read."""
    from .engine.ingest import _layout_order
    names = sorted((q.name for q in ds_dir.iterdir() if q.suffix == ".h5" and q.is_file()
                    and _resolve._LAYOUT_FILE_RE.match(q.name)), key=_layout_order)
    cells = {}
    for name in names:
        sniff = _resolve._sniff_h5(str(ds_dir / name), (ds_dir / name).stat().st_mtime_ns)
        cells[name] = sniff[2] if sniff else None
    return cells


#: how many files the ``Cells per file`` line of ``scan`` names
_CELLS_SHOWN = 8


def _cells_line(cells: dict) -> str:
    """``Cells per file: rna.h5 1,718, adt.h5 1,718.``; files after the first 8
    are only counted."""
    shown = [f"{name} {'?' if n is None else format(n, ',')}"
             for name, n in list(cells.items())[:_CELLS_SHOWN]]
    more = len(cells) - len(shown)
    return (f"Cells per file: {', '.join(shown)}"
            + (f" and {more} more" if more else "") + ".")


def _per_method(df: "pd.DataFrame", private: "pd.DataFrame", *, data_path=None,
                verbose: bool = True, nested: bool = True) -> "pd.DataFrame":
    """The frame of ``scan`` (one row per method) from :func:`_scan`'s two frames; with
    ``verbose`` it prints the per-method lines (``nested``: leave out nested rows)."""
    dataset, ds_dir = private.attrs["dataset"], private.attrs["folder"]
    data_path = ds_dir.parent if data_path is None else data_path
    rows, reasons, with_files = [], [], set()
    held = set(df.attrs["held"])
    several = df["category"].nunique() > 1
    batches = len(_resolve._folder_batches(ds_dir))
    for (method, cat), group in df.groupby(["method", "category"], sort=False):
        if group["files_ok"].any():
            with_files.add(method)
        if group["runnable"].any():
            # the variant run runs: the one with the most modalities
            i = max(group.index[group["runnable"].to_numpy().astype(bool)],
                    key=lambda i: len(str(group.at[i, "modalities"]).split("+")))
            rows.append((method, cat, group.at[i, "modalities"], True, "",
                         *group.loc[i, ["env", "caveat", "command"]]))
            continue
        # the row that speaks for the method: files on disk, the folder's number of
        # batches, families held, fewest missing, then the most modalities, as
        # run runs
        i = min(group.index, key=lambda i: (
            not _inputs_on_disk(group.loc[i], dataset, data_path),
            len(set(re.findall(r"\d+", str(group.at[i, "modalities"])))) != batches,
            len(_row_families(group.at[i, "modalities"]) - held),
            _missing_inputs(group.loc[i], dataset, data_path),
            -len(str(group.at[i, "modalities"]).split("+")), i))
        rows.append((method, cat, group.at[i, "modalities"], False, group.at[i, "reason"],
                     *group.loc[i, ["env", "caveat", "command"]]))
        reasons.append((method, private.at[i, "_other"]))
    out = pd.DataFrame(rows, columns=SCAN_COLUMNS)
    order = out.sort_values(["runnable", "method", "category"],
                            ascending=[False, True, True]).index
    out = out.loc[order].reset_index(drop=True)
    out = _shown_modalities(out)
    if not several:
        out = out.drop(columns="category")
    cells = _cells_per_file(ds_dir)
    # per-batch files where one file is read: said once for the folder
    per_batch = "" if with_files else (
        _resolve._per_batch_hint(ds_dir, private.attrs["category"]) or "")
    if per_batch:
        out["reason"] = [" ".join(r.replace(per_batch, "").split()) for r in out["reason"]]
    out.attrs = {**df.attrs, "cells": cells}
    if not verbose:
        return out
    # methods, not rows: without a category a method has a row per category
    n, k = out["method"].nunique(), out.loc[out["runnable"], "method"].nunique()
    n_files = len(with_files)
    holds = _holds_sentence(dataset, ds_dir, df.attrs["held"], private.attrs["category"])
    line = (f"[scan] {n_files} of {n} {'method' if n == 1 else 'methods'} "
            f"{_have_their(n_files, n)} input files in {dataset}. {k} of {n} can run on "
            f"this computer. {holds}")
    fit = _category_hint(df, private.attrs["category"])
    if fit:
        line += f" {fit}"
    if envs.linux_only_text():
        line += f" {LINUX_ONLY_CHECK}"
    print(line, flush=True)
    if cells:
        print(f"[scan] {_cells_line(cells)}", flush=True)
    if per_batch:
        print(f"[scan] {per_batch}", flush=True)
    if not several:                   # one category: why each method cannot run
        for method, reason in sorted(reasons):
            # the per-batch sentence was printed once above
            if per_batch:
                reason = reason.replace(per_batch, "").strip()
            if reason:
                print(f"[scan] {method}: {reason}", flush=True)
    would = [i for i, r in df.iterrows() if _would_run(r)]
    if nested:
        inside = set(_nested_rows(df.loc[would])[0])
        would = [i for i in would if i not in inside]
    for note in dict.fromkeys(private.loc[would, "_folder_caveat"]):
        if note:
            print(f"[scan] {note}", flush=True)
    unlabelled = _missing_labels_problem(dataset, data_path, df.loc[would], private,
                                         evaluate=True)
    if unlabelled:
        print(f"[scan] {unlabelled}", flush=True)
    return out


@contextlib.contextmanager
def _deadline(seconds):
    """A SIGALRM deadline over a whole per-method step (run and scoring)."""
    import math
    import signal
    import threading
    if not seconds:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        # signal.signal raises off the main thread
        _text.warn("timeout= is unavailable off the main thread; "
                   "running without a deadline")
        yield
        return

    def _fire(signum, frame):
        raise TimeoutError(f"exceeded timeout of {seconds}s")

    prev = signal.signal(signal.SIGALRM, _fire)
    # ceil, not int(): truncating timeout=0.5 gives alarm(0), which sets no alarm
    signal.alarm(max(1, math.ceil(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ------------------------------------------------------------------- label order
def _label_files(dataset, data_path=None) -> list:
    """The label files of a dataset folder, sorted by name: every
    ``*cty*.csv`` except the tool-specific ``*scjoint*`` reformats."""
    base = Path(data_path) if data_path is not None else config.data_path
    return [p for p in sorted((base / dataset).glob("*cty*.csv"))
            if "scjoint" not in p.name.lower()]


def _label_inputs(inputs: dict, variant, label_files: dict) -> tuple:
    """The names of the dataset's label files among a scan row's inputs (also the
    constant file names of a folder-fed variant), whatever ``needs_labels`` says."""
    paths = [Path(p) for role, p in inputs.items() if role != "data_dir"]
    if "data_dir" in inputs:
        paths += [Path(inputs["data_dir"]) / str(a.const) for a in variant.args
                  if a.const and str(a.const).endswith(".csv")]
    names = (label_files.get(p.resolve()) for p in paths)
    return tuple(dict.fromkeys(n for n in names if n))


def _label_candidates(dataset, n, data_path=None):
    """``[(file names, labels, batch per cell)]``: every label ordering of ``n``
    cells; length alone cannot pick one. The batch is each cell's label file."""
    by_file = {}
    for p in _label_files(dataset, data_path):
        try:
            by_file[p.name] = _eio.read_labels(p)
        except Exception as e:  # noqa: BLE001 - say which file
            raise ValueError(f"{p} cannot be read: {e}") from e
    cands = [([k], v) for k, v in by_file.items()]
    rna = [k for k in by_file if k.startswith("rna_cty")]
    atac = [k for k in by_file if k.startswith("atac_cty")]
    for a, b in itertools.product(rna, atac):
        cands += [([a, b], np.concatenate([by_file[a], by_file[b]])),
                  ([b, a], np.concatenate([by_file[b], by_file[a]]))]
    # diagonal, several batches: the RNA files in number order then the ATAC
    # files, or the reverse; each file counts as one batch
    per_batch = [sorted((k for k in by_file if re.fullmatch(rf"{side}_cty\d+\.csv", k)),
                        key=lambda k: _resolve._label_sort_key(Path(k).stem))
                 for side in ("rna", "atac")]
    if all(per_batch) and sum(map(len, per_batch)) > 2:
        for order in (per_batch[0] + per_batch[1], per_batch[1] + per_batch[0]):
            cands.append((order, np.concatenate([by_file[k] for k in order])))
    nums = sorted(k for k in by_file if k.startswith("cty") and any(c.isdigit() for c in k))
    if len(nums) > 1:
        # permute only the subsets of n cells, not every subset (factorial in files)
        sizes = {k: len(by_file[k]) for k in nums}
        for r in range(len(nums), 1, -1):
            for combo in itertools.combinations(nums, r):
                if sum(sizes[k] for k in combo) != n:
                    continue
                for perm in itertools.permutations(combo):
                    cands.append((list(perm),
                                  np.concatenate([by_file[k] for k in perm])))
    out, seen = [], set()
    for names, lab in cands:
        if len(lab) != n or tuple(names) in seen:
            continue
        seen.add(tuple(names))
        batch = np.concatenate([np.full(len(by_file[k]), i + 1) for i, k in enumerate(names)])
        out.append((list(names), lab, batch))
    return out


def _metric_codes(metrics):
    """The metric codes ``metrics=`` selects; ``None`` = every metric, also
    for a value ``evaluate`` will reject (it raises there, with its message)."""
    from ._data import catalog
    try:
        return catalog.metric_selection(metrics).codes
    except (TypeError, ValueError):
        return None


def _names_batch_metric(metrics) -> bool:
    """Whether ``metrics=`` is ``None`` or names a metric that needs the
    batches: one of the batch or the classification family."""
    from .plot.bar import BATCH_METRICS, CLASSIFICATION_METRICS
    codes = _metric_codes(metrics)
    return codes is None or any(c in BATCH_METRICS + CLASSIFICATION_METRICS
                                for c in codes)


def _without_classification(metrics, multi_batch: bool):
    """``metrics=`` without the classification family, for the UMAP a graph method
    writes; ``None`` when nothing else is left."""
    from .plot.bar import BATCH_METRICS, CLASSIFICATION_METRICS, CLUSTERING_METRICS
    if metrics is None or metrics == "all":
        if multi_batch:
            return CLUSTERING_METRICS + [c for c in BATCH_METRICS if c != "kBET"]
        return "clustering" if metrics is None else metrics
    codes = _metric_codes(metrics)
    if codes is None:
        return metrics                     # evaluate rejects it with its message
    left = [c for c in codes if c not in CLASSIFICATION_METRICS]
    if len(left) == len(codes):
        return metrics
    return left or None


def _evaluate_best_order(emb, category, cands, *, metrics=None, user_batch=False,
                         batch_given=True, on_rank=None, embedding=True):
    """Score each candidate label order and keep the best: ``(names, scores, spread)``.
    ``user_batch``: the candidates carry the caller's batch; ``batch_given``: passed now."""
    # a file-of-origin batch, or the saved one rescore reuses, goes to evaluate only
    # for a metric that uses it
    use_batch = (user_batch and batch_given) or _names_batch_metric(metrics)

    def _full(lab, batch, clusters=None):
        several = len(set(np.asarray(batch).tolist())) > 1
        multi = use_batch and several
        chosen = metrics if embedding else _without_classification(metrics, multi)
        if not embedding:
            # the batches go only with a selection that still uses them
            multi = several and ((user_batch and batch_given)
                                 or _names_batch_metric(chosen))
        # a batch the caller passed goes to evaluate even with one value here: it says so
        return _scores(emb, category=category, labels=lab, verbose=False,
                       batch=(batch if multi or (user_batch and batch_given) else None),
                       clusters=clusters, metrics=chosen)

    if len(cands) == 1:
        names, lab, batch = cands[0]
        try:
            val = _full(lab, batch)
        except Exception as e:  # noqa: BLE001 - surfaced on the record
            # an empty result would make a bad metrics=/batch= look like 'no label file matched'
            return names, None, [{"order": names,
                                  "error": f"{type(e).__name__}: {str(e)[:300]}"}]
        # metrics= may leave ARI out; + 0.0: a tiny negative ARI rounds to 0.0, not -0.0
        ari = round(float(val["Value"]["ARI"]), 4) + 0.0 if "ARI" in val.index else None
        return names, val, [{"order": names, "ARI": ari}]

    # The clusters behind ARI depend on the embedding and the number of cell types
    # alone: cluster once per number, rank every order on it, and compute the full
    # metric set only for the winner.
    import anndata as ad
    scib = _escib.import_scib()

    if on_rank is not None:
        on_rank(len(cands), emb.shape[0])
    searched = {}                   # number of cell types -> clusters
    scored, last_error = [], ""
    for names, lab, batch in cands:
        try:
            cell_types = np.asarray(lab).astype(str)
            n = len(set(cell_types.tolist()))
            if n not in searched:
                searched[n] = _escib.leiden_n_clusters(emb, n)
            obs = pd.DataFrame({"celltype": pd.Categorical(cell_types),
                                "cluster": pd.Categorical(searched[n])},
                               index=np.arange(len(cell_types)).astype(str))
            ari = float(scib.metrics.ari(ad.AnnData(obs=obs), cluster_key="cluster",
                                         label_key="celltype"))
            scored.append((ari, names, lab, batch, searched[n]))
        except Exception as e:  # noqa: BLE001 - keep screening other orders
            last_error = f"{type(e).__name__}: {e}"
    if not scored:
        raise RuntimeError("no label ordering could be screened"
                           + (f"; last error: {last_error}" if last_error else ""))
    scored.sort(key=lambda r: -r[0])
    ari, names, lab, batch, clusters = scored[0]
    try:
        val = _full(lab, batch, clusters=clusters)      # reuses the winning clustering
        if val.attrs.get("clustering") == "user":
            # the clusters came from the screening search, not from the user
            val.attrs.update(clustering="search", leiden_flavor=_escib._resolve_flavor(None))
    except Exception as e:
        raise RuntimeError(
            f"evaluation failed for the winning label order {names}: "
            f"{type(e).__name__}: {e}") from e
    spread = [{"order": n, "ARI": round(a, 4) + 0.0} for a, n, _, _, _ in scored]
    return names, val, spread


# ------------------------------------------------------------------------ results
#: the record status of each status code a 0.3 record holds
_STATUS = {"CHAIN_OK": "scored", "CHAIN_OK_GRAPH_METHOD": "scored", "RUN_OK": "ran",
           "RUN_OK_EVAL_FAILED": "ran", "RUN_OK_NO_EMBEDDING": "ran",
           "RUN_OK_NO_LABEL_MATCH": "ran", "FAIL": "failed", "SKIPPED": "skipped",
           "TIMEOUT": "timeout"}
#: the five record statuses, in the order ``repr`` counts them
STATUSES = ("scored", "ran", "failed", "skipped", "timeout")

#: why a record that ran has no scores, for a 0.3 status code that said it
_RAN_REASON = {"RUN_OK_NO_EMBEDDING": "It writes a graph, not an embedding.",
               "RUN_OK_NO_LABEL_MATCH": "No label file has as many cells as the output."}


def _status(code) -> str:
    """The record status of ``code``: one of :data:`STATUSES`, or a 0.3 code."""
    s = str(code or "")
    if s in STATUSES:
        return s
    return _STATUS.get(s) or ("timeout" if s.startswith("TIMEOUT") else "failed")


def _new_status(rec: dict) -> dict:
    """``rec`` with a 0.3 status code replaced by its record status; the reason
    the code alone said goes to ``note``."""
    code = rec.get("status")
    if code not in STATUSES:
        rec["status"] = _status(code)
        if code in _RAN_REASON and not (rec.get("error") or rec.get("note")):
            rec["note"] = _RAN_REASON[code]
    return rec


def _reason(rec: dict) -> str:
    """Why ``rec`` has no scores, or ``""`` when it has them."""
    if _status(rec.get("status")) == "scored":
        return ""
    return rec.get("error") or rec.get("note") or ""


#: the columns of RunResult.summary; the metric columns come after status
_SUMMARY_COLUMNS = ("method", "status", "run_sec", "caveat", "reason")


#: provenance of a reused output when ``out_dir`` holds no record of it
_UNKNOWN_PROVENANCE = {"scripts_commit": None, "env_flavor": "unknown", "hostname": ""}


def _saved_records(folder: Path) -> dict:
    """``{method: record}`` of the runs saved in ``folder``; records without a
    method are skipped, an unreadable file gives ``{}``."""
    path = Path(folder) / "batch_result.json"
    try:
        with open(path) as fh:
            records = json.load(fh).get("records") or []
    except (OSError, ValueError, AttributeError):
        return {}
    # a saved file may hold any records RunResult was given
    return {r["method"]: _new_status(r) for r in records
            if isinstance(r, dict) and r.get("method") and _status(r.get("status")) != "skipped"}


def _check_save_target(folder: Path, dataset: str, category: str) -> None:
    """Raise ``ValueError`` when ``folder`` holds a saved result of another dataset
    or category: merging the records would mix two runs."""
    path = Path(folder) / "batch_result.json"
    if not path.exists():
        return
    with open(path) as fh:
        blob = json.load(fh)
    have = (blob.get("dataset"), blob.get("category"))
    if have != (dataset, category):
        raise ValueError(
            f"{folder} already holds a saved result for dataset={have[0]!r} "
            f"category={have[1]!r}; this one is dataset={dataset!r} "
            f"category={category!r}. Save it to another folder.")


def _plan_caveats(dataset, category, data_path, methods) -> dict | None:
    """``{(method, modalities): caveat}`` of ``run``'s plan for these methods;
    ``None`` when the plan cannot be made (no dataset folder), so the saved
    caveats stay."""
    methods = [m for m in dict.fromkeys(methods) if m]
    if not methods:
        return {}
    try:
        plan, _ = _scan(dataset, category, methods=methods, modalities="all",
                        data_path=data_path)
    except Exception:  # noqa: BLE001 - no plan: the record keeps its caveat
        return None
    return {(r["method"], tuple(_row_modalities(r["modalities"]))): _run_caveat(r["caveat"])
            for _, r in plan.iterrows()}


def _scorable(rec: dict) -> bool:
    """Whether ``rescore`` scores ``rec``: it ran (scored or not)."""
    return _status(rec.get("status")) in ("scored", "ran")


def _rank_line(method):
    """``rescore(verbose=True)``'s line before the label orders are ranked."""
    def say(orders, cells):
        print(f"[rescore] {method}: ranking {orders} label orders on {cells:,} cells "
              f"with one Leiden clustering ...", flush=True)
    return say


def _warn_unsaved_batch(records, result) -> None:
    """Warn once when a record was scored with a user batch that is not saved."""
    old = [r for r in records if r.get("batch_source") == "user" and "batch_file" not in r]
    lost = [r for r in records if "batch_file" in r
            and result._batch_text(r.get("batch_file")) is None
            and (r.get("batch_file") or r.get("batch_source") == "user")]
    if old or lost:
        _text.warn("This result was scored with your batch, which is not saved with it. "
                   "Pass batch= again to keep the batch metrics.", UserWarning)


class RunResult:
    """The result of ``mtb.run``: its summary table, long table and figure.

    ``mtb.run`` and ``mtb.load_run`` build it. It keeps one record per
    method, which ``rescore`` and ``plot`` read.

    Parameters
    ----------
    records : list[dict]
        One record per method run, as ``run`` builds them.
    dataset : str
        Dataset folder name the run used.
    category : str
        Integration category of the run.
    out_dir : path | None
        Where the run wrote its outputs; ``save()`` defaults to it.

    Attributes
    ----------
    records : list[dict]
        One dict per method, as saved in ``batch_result.json``: ``method``,
        ``status`` (as in ``summary``), ``out_dir`` (its output folder),
        ``params_used``, ``run_sec``, ``reused`` (``True`` when
        ``skip_existing`` reused a saved output), ``metrics``,
        ``labels_used`` (the label files in the cell order scored),
        ``error`` or ``note`` (the ``reason`` of ``summary``) and the run's
        provenance (``multibench_version``, ``scripts_commit``,
        ``env_flavor``, ``hostname``).
    dataset : str
        Dataset folder name.
    category : str
        Integration category.
    out_dir : path | None
        The run's output root, or ``None`` for an in-memory result.

    Examples
    --------
    >>> import multibench as mtb
    >>> res = mtb.run("D11", "vertical", out_dir="out/", timeout=3600)
    >>> res.summary[["method", "status", "ARI", "NMI"]]
    >>> res.summary[res.summary.status != "scored"]   # what did not finish, and why
    >>> fig = res.plot(metrics=["ARI", "NMI", "ASW"])
    >>> res.rescore().save()                      # nothing re-run

    Notes
    -----
    ``len(res)`` is the number of method records.

    See Also
    --------
    mtb.run : produces one.

    mtb.load_run : reloads one from ``save()``'s folder.

    mtb.plot.bubble : the figure ``plot`` draws from ``long``.
    """

    def __init__(self, records, dataset, category, out_dir=None):
        # one status vocabulary: a record of 0.3 gets its 0.4 status
        self.records = [_new_status(r) for r in records]
        self.dataset = dataset
        self.category = category
        self.out_dir = out_dir
        self._batches: dict = {}       # {batch file name: CSV text}, until saved

    def _batch_text(self, name) -> str | None:
        """The text of the batch file ``name``: in memory, else in ``out_dir``."""
        if name and name not in self._batches and self.out_dir is not None:
            p = Path(self.out_dir) / name
            if p.is_file():
                self._batches[name] = p.read_text()
        return self._batches.get(name)

    @property
    def summary(self) -> pd.DataFrame:
        """One row per method: status, timing, label matching and every metric.

        Returns
        -------
        pandas.DataFrame
            One row per method, sorted by ``method``:

            ```text
            method                  method id
            status                  scored, ran, failed, skipped or timeout
            ARI, NMI, ASW, ...      one column per metric
            run_sec                 seconds of the method run
            caveat                  scan's caveat for the method, or ""
            reason                  why the method has no scores, or ""
            ```

        Examples
        --------
        >>> res = mtb.load_run("out/")
        >>> res.summary[["method", "status", "ARI"]]

        See Also
        --------
        mtb.load_run : reads a saved folder; its ``batch_result.json`` holds every
            record, with the label orders tried.
        """
        rows = [{"method": r.get("method"), "status": _status(r.get("status")),
                 "run_sec": r.get("run_sec"),
                 # NaN, not None, for a record saved before the field
                 "caveat": np.nan if r.get("caveat") is None else r.get("caveat"),
                 "reason": _reason(r)}
                # + 0.0: an older saved record may hold -0.0
                | {m: (v + 0.0 if isinstance(v, float) else v)
                   for m, v in (r.get("metrics") or {}).items()}
                for r in self.records]
        table = (pd.DataFrame(rows, columns=None if rows else list(_SUMMARY_COLUMNS))
                 .sort_values("method").reset_index(drop=True))
        scores = [c for c in table.columns if c not in _SUMMARY_COLUMNS]
        return table[["method", "status", *scores, *_SUMMARY_COLUMNS[2:]]]

    @property
    def long(self) -> pd.DataFrame:
        """The scores as a long table, for plotting.

        This is what ``plot`` and ``mtb.plot.bubble`` consume.

        Returns
        -------
        pandas.DataFrame
            Columns ``metric``, ``value``, ``method``, ``dataset``,
            ``category`` and ``source``. When the scores carry
            ``scored_with``, so does the table. Empty, with the first six
            columns, when no method produced metrics.

        Examples
        --------
        >>> res = mtb.load_run("out/")
        >>> mtb.plot.bubble(res.long, metrics=["ARI", "NMI"])
        >>> res.long.pivot_table(index="method", columns="metric", values="value")

        Notes
        -----
        **Source of the rows.** Each record contributes the unrounded frame
        ``run`` attached (or ``long.csv`` via ``mtb.load_run``) when
        present. Otherwise the record contributes its ``metrics`` dict.

        See Also
        --------
        RunResult.plot : draws the bubble figure from this frame.
        """
        frames = []
        for r in self.records:
            if r.get("_long") is not None:
                frames.append(r["_long"])
            elif r.get("metrics"):
                # through to_long, for the columns of an attached frame
                wide = pd.DataFrame({"Value": list(r["metrics"].values())},
                                    index=list(r["metrics"]))
                frame = _to_long(wide, method=r.get("method"),
                                 dataset=r.get("dataset", self.dataset),
                                 category=r.get("category", self.category))
                if r.get("scored_with"):
                    frame["scored_with"] = r["scored_with"]
                else:
                    frame = frame.drop(columns="scored_with")
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=_LONG_COLUMNS)
        # a long.csv written before 0.4.0 has the constant clustering column
        return pd.concat(frames, ignore_index=True).drop(columns="clustering",
                                                         errors="ignore")

    def plot(self, **kw):
        """Bubble figure of every method that produced metrics.

        Rows are methods, best first. Circle size shows the rank within a
        column (bigger is better). Fill compares the values in a column: the
        lightest is the lowest in this figure, not zero.

        Parameters
        ----------
        **kw
            Keyword arguments of ``mtb.plot.bubble``: ``metrics=``,
            ``methods=``, ``title=`` and ``save=``.

        Returns
        -------
        matplotlib.figure.Figure
            Save it with ``fig.savefig("out.png")``.

        Raises
        ------
        ValueError
            Nothing was scored (``summary`` says why), or an unknown name in ``methods=``.

        Examples
        --------
        >>> res = mtb.load_run("out/")
        >>> fig = res.plot()
        >>> fig = res.plot(metrics=["ARI", "NMI", "ASW"], title="D11 vertical")
        >>> fig.savefig("D11_vertical.png", dpi=200)

        Notes
        -----
        **Keywords.** ``metrics=`` sets the column order and ``methods=`` a
        subset.

        **Reading the figure.** Size and colour are relative to the methods
        in this figure, so with few methods a small gap fills the whole colour
        scale. Check the values in ``summary``.

        See Also
        --------
        mtb.plot.bubble : the underlying function and its full keyword list.

        RunResult.long : the frame handed to it.
        """
        from . import plot as _plot
        lng = self.long
        if lng.empty:
            raise ValueError("No method produced metrics. res.summary says why.")
        return _plot.bubble(lng, **kw)

    def rescore(self, *, batch=None, labels=None, metrics=None,
                verbose: bool = True) -> "RunResult":
        """Score the saved outputs again with new labels, batch or metrics.

        No method is re-run. Each record's embedding is read back from its
        ``out_dir``. The Leiden clusters depend on the cell order and the machine,
        so the same embedding can score a few hundredths differently.

        Parameters
        ----------
        batch : array-like | Series | path | None
            Batch ids in the order of the dataset's label files, or a
            barcode-indexed Series or CSV. Every cell needs a batch. ``None`` =
            the batch ``run`` was given, else each cell's label file.
        labels : array-like | Series | path | None
            One cell-type label per cell; a Series or a barcode-indexed CSV is
            aligned by barcode. ``None`` = the dataset's label files, in the
            order the record was scored with (ranked as ``run`` ranks
            them when it has none).
        metrics : str | list[str] | None
            Metric family (``"clustering"``, ``"batch"``, ``"classification"``,
            ``"all"``) or metric codes, which replace those scores and keep the
            others (with new ``batch`` or ``labels``, only these are kept);
            ``None`` = every metric the batch structure allows, as in
            ``mtb.evaluate``.
        verbose : bool
            Print one line per method, and one before each label-order ranking.

        Returns
        -------
        RunResult
            A new result; this one is untouched. Call ``.save(out_dir)`` on it
            to persist it.

        Raises
        ------
        FileNotFoundError
            The output of no method is found; the message gives the first
            method's error. A method that cannot be scored otherwise gets the
            status ``ran``, and ``reason`` says why.
        ValueError
            ``labels`` or ``batch`` holds ids that are not cells of the dataset,
            or a cell has no batch or no label (also in the label files).

        Warns
        -----
        UserWarning
            A Series or barcode-indexed CSV cannot be aligned and is matched by position.

        Examples
        --------
        >>> mtb.load_run("out/").rescore().save()

        See Also
        --------
        mtb.evaluate : the scoring function applied per record.
        RunResult.save : persist the re-scored result.
        """
        import copy
        with warnings.catch_warnings():
            # load_run chose the data folder: no note on a same-named one here
            warnings.filterwarnings("ignore", message=r"Using .*; \./.* also exists")
            new = RunResult([], self.dataset, self.category, out_dir=self.out_dir)
            found = [_dataset_root(r, self.dataset) for r in self.records]
            roots = [root for root, _ in found]
            tried: dict = {}           # the roots searched, named when the folder is not found
            for root, r in zip(roots, self.records):
                tried.setdefault(root, _root_tries(r))
            # labels and batch per data folder, before any record is scored: ids
            # that are not cells of the dataset raise here
            labels_at: dict = {}
            batch_at: dict = {}
            used = dict.fromkeys(root for root, r in zip(roots, self.records)
                                 if _status(r.get("status")) != "skipped")
            for root in used:
                if labels is not None:
                    labels_at[root] = _cell_vector(labels, self.dataset, root, what="labels",
                                                   order=_ROWS, tried=tried[root])
                    _check_gaps(labels_at[root][0], self.dataset, "labels")
                else:
                    # a missing label would be scored as one more cell type
                    unlabelled = _missing_labels_problem(self.dataset, root)
                    if unlabelled:
                        raise ValueError(unlabelled)
                # a batch vector follows labels given in embedding row order
                rows = labels is not None and not labels_at[root][1]
                if batch is not None:
                    vec, by_cell = _cell_vector(batch, self.dataset, root, what="batch",
                                                order=_ROWS if rows else None,
                                                tried=tried[root])
                    _check_gaps(vec, self.dataset)
                    bad = not rows and not by_cell and _batch_length_problem(
                        [(r.get("method"), r.get("modalities")) for r in self.records
                         if _scorable(r)], vec, self.dataset, self.category, root)
                    if bad:
                        raise ValueError(bad)
                    # saved with the result, unless it follows the embedding rows
                    file = None if rows and not by_cell else _batch_csv(vec, self.dataset, root)
                    batch_at[root] = (vec, by_cell, file)
            if batch is None and _names_batch_metric(metrics):
                _warn_unsaved_batch([r for r in self.records if _scorable(r)], self)
            caveats = {root: _plan_caveats(self.dataset, self.category, root,
                                           [r.get("method") for r in self.records
                                            if _scorable(r)])
                       for root in used}
            no_output = []             # records whose saved output is not found
            for r, (root, ok) in zip(self.records, found):
                # n_tunable: a field of 0.3 records, stale since
                rec = copy.deepcopy({k: v for k, v in r.items()
                                     if k not in ("_long", "n_tunable")})
                rec["_long"] = None
                method = rec.get("method")
                if not _scorable(rec):
                    new.records.append(rec)
                    continue
                rec["data_path"] = root
                if ok and root is not None:
                    rec["data_root"] = root
                # the plan's caveat as this version words it
                plan = caveats.get(root)
                if plan is not None:
                    rec["caveat"] = plan.get((method, tuple(rec.get("modalities") or ())), "")
                try:
                    mods = rec.get("modalities") or []
                    variant = registry.get(method).select(self.category, set(mods))
                    try:
                        # a record without out_dir: the folder run names
                        emb = _load_embedding(Path(rec.get("out_dir") or Path(self.out_dir or ".")
                                                   / f"{method}_{self.dataset}"), variant)
                    except FileNotFoundError:
                        no_output.append(method)
                        raise
                    if emb is None:
                        rec["status"], rec["note"] = "ran", _RAN_REASON["RUN_OK_NO_EMBEDDING"]
                    else:
                        label_ids, labels_by_cell = labels_at.get(root, (None, False))
                        if batch is not None:
                            batch_ids, batch_by_cell, file = batch_at[root]
                            rec["batch_file"] = file and file[0]
                        else:
                            # the batch run was given, when it is saved
                            text = self._batch_text(rec.get("batch_file"))
                            file = text and (rec["batch_file"], text)
                            batch_ids, batch_by_cell = (
                                (_read_batch_csv(text)["batch"].to_numpy(), True) if text
                                else (None, False))
                        if file:
                            new._batches[file[0]] = file[1]
                        _score_record(rec, emb, self.dataset, self.category, root, variant,
                                      batch=batch_ids, labels=label_ids, metrics=metrics,
                                      labels_by_cell=labels_by_cell,
                                      batch_by_cell=batch_by_cell,
                                      batch_given=batch is not None,
                                      on_rank=_rank_line(method) if verbose else None,
                                      tried=tried[root])
                except Exception as e:  # noqa: BLE001 - one bad record must not abort the rest
                    _drop_scores(rec)
                    rec["status"] = "ran"
                    rec["error"] = _clip_error(f"{type(e).__name__}: {e}")
                if verbose:
                    print(f"[rescore] {method} -> {_status(rec['status'])} "
                          f"{_ari_tail(rec)}".rstrip(), flush=True)
                if metrics is not None and batch is None and labels is None \
                        and rec.get("metrics") and r.get("metrics"):
                    # the named metrics replace theirs; the other scores stay (not
                    # after new batches or labels: they were scored with the old ones)
                    rec["metrics"] = {**r["metrics"], **rec["metrics"]}
                    both = pd.concat([RunResult([r], self.dataset, self.category).long,
                                      rec["_long"]], ignore_index=True)
                    rec["_long"] = both.drop_duplicates("metric", keep="last").set_index(
                        "metric").loc[list(dict.fromkeys(both["metric"]))].reset_index()
                new.records.append(rec)
            # a saved batch that fits no label order: the batch metrics are gone
            unused = next((r["note"] for r in new.records
                           if str(r.get("note") or "").startswith("The saved batch is not used")),
                          None)
            if unused:
                _text.warn(unused, UserWarning)
            # a tree whose outputs are all gone (moved, or an older layout): nothing
            # to score, so say so instead of returning a table without scores
            tried_records = [r for r in new.records if _scorable(r)]
            if tried_records and len(no_output) == len(tried_records):
                raise FileNotFoundError(
                    f"No method output in {self.out_dir or '.'}. Load the folder that holds "
                    f"the method folders, or run the methods again.")
            return new

    def embedding(self, method: str):
        """The embedding of one method, as an AnnData.

        Parameters
        ----------
        method : str
            A method of this result.

        Returns
        -------
        anndata.AnnData
            ``obsm["X_emb"]`` in the method's cell order; ``obs_names`` are
            the cells' barcodes, each prefixed by its label file
            (``cty2:AAAC...``) when some barcode is in two files; ``obs``
            holds each cell's ``label``, and its ``batch`` when the run had
            more than one batch.

        Raises
        ------
        KeyError
            No record for ``method``.
        ValueError
            The method did not run, or it writes a graph instead of an embedding.
        FileNotFoundError
            Its output folder is gone.

        Examples
        --------
        >>> res = mtb.run("D11", "vertical", methods=["Matilda"], out_dir="out/")
        >>> adata = res.embedding("Matilda")
        >>> sc.pp.neighbors(adata, use_rep="X_emb")
        """
        import anndata as ad
        have = [r.get("method") for r in self.records]
        name = method if method in have else registry.check_method(method)
        rec = next((r for r in self.records if r.get("method") == name), None)
        if rec is None:
            raise KeyError(f"No record for {method!r}; this result has {_and_list(have)}.")
        if not _scorable(rec) or not rec.get("out_dir"):
            raise ValueError(f"{name} has no output: its status is {rec.get('status')}. "
                             f"res.summary says why.")
        variant = registry.get(name).select(self.category, set(rec.get("modalities") or ()))
        emb = _load_embedding(Path(rec["out_dir"]), variant)
        if emb is None:
            raise ValueError(f"{name} writes a graph, not an embedding.")
        root, _ = _dataset_root(rec, self.dataset)
        obs = _embedding_obs(rec, self.dataset, self.category, root, emb.shape[0],
                             self._batch_text(rec.get("batch_file")))
        return ad.AnnData(obs=obs, obsm={"X_emb": np.asarray(emb)})

    def save(self, out_dir=None) -> "Path":
        """Write this result to disk so it outlives the process.

        Parameters
        ----------
        out_dir : path | None
            Target folder, created if missing; ``None`` = the result's own
            ``out_dir``, else the current directory.

        Returns
        -------
        Path
            The folder written.

        Raises
        ------
        ValueError
            The folder holds a saved result of another dataset or category.

        Examples
        --------
        >>> res = mtb.load_run("out/")
        >>> res.rescore(metrics="clustering").save("out/clustering_only")
        >>> mtb.load_run("out/clustering_only").summary

        Notes
        -----
        **Files written.**

        - ``summary.csv`` - the ``summary`` frame.
        - ``long.csv`` - the ``long`` frame, only when some method produced
          metrics.
        - ``batch_result.json`` - dataset, category and the per-method records.
        - ``batch_<hash>.csv`` - a batch vector the records were scored with.

        **Saving into a folder that has a result.** When the folder already
        holds ``batch_result.json`` for the same dataset and category, the
        records are merged. A method in this result replaces its earlier
        record, unless its status is ``skipped`` and the earlier one's is not; the other
        earlier records are kept, every file in the list above is rewritten
        from the merged set, and a line ``[save] Merged with 1 earlier record in <folder>
        (StabMap).`` names them. This result object is not changed. Jobs
        running at the same time should use one ``out_dir`` each; combine
        their ``long`` tables with ``pandas.concat`` to plot them together.

        See Also
        --------
        mtb.load_run : reads the folder back.
        """
        folder = Path(out_dir or self.out_dir or ".").expanduser()
        _check_save_target(folder, self.dataset, self.category)
        before = (load_run(folder).records if (folder / "batch_result.json").exists()
                  else [])
        # a skipped record never replaces an earlier record of a run
        ran = {r.get("method") for r in before if _status(r.get("status")) != "skipped"}
        new = [r for r in self.records
               if not (_status(r.get("status")) == "skipped" and r.get("method") in ran)]
        mine = {r.get("method"): r for r in new}
        kept = [r for r in before if r.get("method") not in mine]
        # the earlier order, each method in its place; new methods last
        order = [mine.get(r.get("method"), r) for r in before]
        order += [r for r in new if r.get("method") not in {b.get("method") for b in before}]
        merged = RunResult(order, self.dataset, self.category,
                           out_dir=folder) if kept else self
        folder.mkdir(parents=True, exist_ok=True)
        merged.summary.to_csv(folder / "summary.csv", index=False)
        long_table = merged.long
        if not long_table.empty:
            long_table.to_csv(folder / "long.csv", index=False)
        elif (folder / "long.csv").exists():
            # an earlier long.csv would attach stale metrics on load_run
            (folder / "long.csv").unlink()
        # failures.csv, written before 0.4.0, would disagree with summary.csv
        (folder / "failures.csv").unlink(missing_ok=True)
        for name in dict.fromkeys(r["batch_file"] for r in merged.records
                                  if r.get("batch_file")):
            text = None if (folder / name).exists() else self._batch_text(name)
            if text is not None:
                (folder / name).write_text(text)
        slim = [{k: v for k, v in r.items() if k != "_long"} for r in merged.records]
        here = folder.resolve()
        for r in slim:
            # inside the folder: relative, so the folder can be moved or packed
            if r.get("out_dir") and Path(r["out_dir"]).resolve().is_relative_to(here):
                r["out_dir"] = str(Path(r["out_dir"]).resolve().relative_to(here))
        with open(folder / "batch_result.json", "w") as fh:
            json.dump({"dataset": self.dataset, "category": self.category,
                       "records": slim}, fh, indent=1, default=str)
        if kept:
            names = ", ".join(dict.fromkeys(str(r.get("method")) for r in kept))
            records = "record" if len(kept) == 1 else "records"
            print(f"[save] Merged with {len(kept)} earlier {records} in {folder} ({names}).",
                  flush=True)
        return folder

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        counts = pd.Series([_status(r.get("status")) for r in self.records]).value_counts()
        text = ", ".join(f"{counts[s]} {s}" for s in STATUSES
                         if s in counts) or "no method"
        return f"<RunResult {self.category}/{self.dataset}: {text}>"


def _embedding_obs(rec, dataset, category, root, n, batch_text=None) -> pd.DataFrame:
    """The ``obs`` of :meth:`RunResult.embedding`: each row's barcode (the
    index), ``label`` and ``batch``, from the label files in the order the
    record was scored with (else the method's order)."""
    folder = _dataset_folder(dataset, root)
    names = [f for f in rec.get("labels_used") or () if (folder / f).is_file()]
    if not names and folder.is_dir():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            files = _resolve.labels_for(dataset, category, rec["method"],
                                        modalities=rec.get("modalities") or None,
                                        data_path=root)
        names = [Path(p).name for p in files.values()]
    labels, barcodes, batch = [], [], []
    # the barcodes of the files the method read (atac_gas.h5, not atac_peak.h5)
    read = _method_files(dataset, category, rec["method"], rec.get("modalities"), root)
    for f in names:
        lab = _eio.read_labels(folder / f)
        bars = next((b for b in map(_resolve._barcodes_of, _label_data_files(folder / f, read))
                     if b is not None and len(b) == len(lab)), None)
        labels.append(np.asarray(lab, dtype=object))
        barcodes.append(bars)
        batch.append(np.full(len(lab), Path(f).stem, dtype=object))
    if sum(map(len, labels)) != n:
        _text.warn(f"The label files of {dataset} do not hold the {n:,} cells of "
                     f"{rec['method']}, so obs has no barcodes, labels or batches.",
                     UserWarning)
        return pd.DataFrame(index=pd.Index([str(i) for i in range(n)]))
    index = (np.concatenate([np.asarray(b, dtype=object) for b in barcodes])
             if all(b is not None for b in barcodes) else np.arange(n).astype(str))
    stems = np.concatenate(batch)
    obs = pd.DataFrame({"label": np.concatenate(labels)}, index=pd.Index(index, dtype=object))
    given = _saved_batch(batch_text, obs.index, names, dataset, root)
    batches = stems if given is None else given
    if len(set(batches.tolist())) > 1:
        obs["batch"] = batches
    if not obs.index.is_unique:
        # batches that share barcodes: each name gets its label file, as cty2:AAAC...
        obs.index = pd.Index([f"{s}:{i}" for s, i in zip(stems, obs.index)], dtype=object)
    obs.attrs["barcodes"] = all(b is not None for b in barcodes)
    return obs


#: The dataset of a run's ``embedding.h5`` that names its cells, one per row.
CELLS_DATASET = "barcodes"


def _write_cell_names(rec, path: Path, dataset, category, data_path) -> None:
    """Write the cells of ``rec``'s embedding, named as ``RunResult.embedding``
    names them, into its output file as :data:`CELLS_DATASET`; nothing when the
    label files do not name them."""
    import h5py
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs = _embedding_obs(rec, dataset, category, data_path, rec["emb_shape"][0])
    if not obs.attrs.get("barcodes") or not path.is_file():
        return
    try:
        with h5py.File(path, "a") as h:
            if CELLS_DATASET in h:
                del h[CELLS_DATASET]
            h.create_dataset(CELLS_DATASET, data=np.asarray(obs.index, dtype=object),
                             dtype=h5py.string_dtype())
    except OSError as e:          # the run stands; evaluate reads its run record instead
        _text.warn(f"Could not write the cell names into {path}: {e}", UserWarning)


def _saved_batch(text, cells, names, dataset, root):
    """The batch a run was given (its saved CSV ``text``) for ``cells``, which
    come from the label files ``names`` in that order; ``None`` without one or
    when it cannot be placed: by cell id, else by label file."""
    if not text:
        return None
    saved = _read_batch_csv(text)
    if "cell" in saved.columns and cells.is_unique and set(cells) <= set(saved["cell"]):
        return saved.set_index("cell")["batch"].reindex(cells).to_numpy(dtype=object)
    by_file = _batch_segments(dataset, root, saved["batch"].to_numpy(dtype=object))
    if by_file is None or not all(f in by_file for f in names):
        return None
    return np.concatenate([np.asarray(by_file[f], dtype=object) for f in names])


# ---------------------------------------------------------------------- run
#: closing lines that name no cause (R's after any error)
_ERROR_TRAILERS = ("Execution halted",)
#: R prints the call trace and then any warnings after the error message
_R_CALLS = re.compile(r"^Calls: ")
_R_WARNINGS = re.compile(r"^In addition: Warning messages?:")


def _clip_error(text: str) -> str:
    """``text`` cut to 600 characters from the left: it ends with the exception."""
    return text if len(text) <= 600 else "... " + text[-596:]


def _ari_tail(rec) -> str:
    """The end of a ``run`` / ``rescore`` result line: ``ARI 0.629`` (never
    ``-0.000``); ``""`` without an ARI."""
    ari = (rec.get("metrics") or {}).get("ARI")
    try:
        ari = float(ari)
    except (TypeError, ValueError):
        return ""
    if np.isnan(ari):
        return ""
    return f"ARI {round(ari, 3) + 0.0:.3f}"


def _error_tail(error, width: int = 200) -> str:
    """The last line of ``error`` that names a cause, clipped from the left to
    ``width``; R's call trace and warnings are skipped, ``Error in f() :`` joined."""
    lines = [l.strip() for l in str(error or "").splitlines()]
    lines = [l for l in lines if l and l not in _ERROR_TRAILERS]
    cut = next((i for i in range(len(lines) - 1, 0, -1) if _R_WARNINGS.match(lines[i])), None)
    if cut is not None:
        lines = lines[:cut]
    lines = [l for l in lines if not _R_CALLS.match(l)]
    if not lines:
        return ""
    last = lines[-1]
    if len(lines) > 1 and lines[-2].startswith("Error") and lines[-2].endswith(":"):
        last = f"{lines[-2]} {last}"
    if len(last) <= width:
        return last
    # clipped at a word boundary: the tail starts with a whole word
    tail = last[-(width - 3):]
    space = tail.find(" ")
    if last[-(width - 3) - 1] != " " and 0 <= space < len(tail) - 1:
        tail = tail[space + 1:]
    return "..." + tail


def _nothing_runnable_message(dataset: str, category: str, blocked: pd.DataFrame,
                              methods, *, given, modalities=None, others=None,
                              every: bool = False, note: str = "") -> str:
    """The ``ValueError`` text when no requested row can start (run's Notes): one
    line per blocked method with its reason (every named method, else the first 3
    whose files are in place), then the ``mtb.scan`` call of these arguments
    (``given``: the dataset as the caller wrote it). Off Linux the platform blocks
    every row, so a method is listed only for what else blocks it (``others``: each
    row's blocks besides the platform)."""
    linux_only = envs.linux_only_text()
    platform = bool(linux_only) and not blocked["env_ok"].all()
    why = (others.reindex(blocked.index).fillna("") if platform and others is not None
           else blocked["reason"])
    # per method, the row whose files are in place first, as run would run it
    order = blocked.assign(_files=~blocked["files_ok"].astype(bool), _why=why).sort_values(
        ["_files", "method"], kind="stable")
    rows = [r for _, r in order.groupby("method", sort=False).head(1).iterrows()
            if r["_why"] and (methods or not r["_files"])]
    lines = [(f"None of the requested methods ({', '.join(methods)}) can run on "
              if methods else "No method can run on ") + f"{dataset} ({category})."]
    if platform:
        lines.append(f"{linux_only} On this computer you can check files, score embeddings "
                     f"and plot. Run the methods on a Linux machine, such as Colab.")
    shown = rows if methods else rows[:3]
    if shown:
        lines.append("Blocked:" if len(shown) == len(rows)
                     else f"The first {len(shown)} of {len(rows)} blocked methods:")
        lines += [f"  {_row_label(r)}: {r['_why']}" for r in shown]
    if note:
        lines.append(note)
    py = [repr(os.fspath(given)), repr(category)]
    if methods:
        py.append(f"methods={list(methods)}")
    if modalities:
        py.append(f"modalities={list(modalities)}")
    elif every:
        py.append('modalities="all"')
    lines.append(f"mtb.scan({', '.join(py)}) says why." + ("" if platform else f" {_DOCTOR}"))
    return "\n".join(lines)


def _row_label(r) -> str:
    """``Matilda (rna+adt)``, or the method alone for a variant that reads a folder."""
    mods = r["modalities"]
    return r["method"] if mods == "(data_dir)" else f"{r['method']} ({mods})"


def _check_params_shape(params, fn: str) -> None:
    """``TypeError`` for a ``params`` dict that is not keyed by method (a value
    that is not a dict)."""
    if isinstance(params, dict) and not all(isinstance(v, dict) for v in params.values()):
        raise TypeError(f"{fn}(params=) takes one dict per method, such as "
                        f"params={{'Matilda': {{'epochs': 3}}}}.")


def _checked_args(category, methods, modalities, params, fn: str) -> tuple:
    """``(methods, every, modalities, params)`` of a ``scan`` / ``run`` call, checked
    before any I/O: method ids in any case or display names, ``"all"`` as
    ``every``, ``params`` keyed by method id."""
    registry.check_category(category)
    registry.check_id_list(methods, "methods")
    every = isinstance(modalities, str) and modalities == "all"
    if every:
        modalities = None
    # another bare string is left for the id-list check to reject
    registry.check_id_list(modalities, "modalities")
    if methods is not None:
        methods = [registry.check_method(m) for m in methods]
    _check_params_shape(params, fn)
    params = {registry.check_method(m): v for m, v in (params or {}).items()}
    return methods, every, modalities, params


def _check_param_keys(plan_df: pd.DataFrame, params: dict) -> None:
    """Raise ``KeyError`` when ``params`` names a key no scanned variant of that
    method accepts, so scan catches the typo; a method without rows is skipped."""
    for method, overrides in params.items():
        rows = plan_df[plan_df["method"] == method]
        if rows.empty or not overrides:
            continue
        variants = [registry.get(method).select(c, set(_row_modalities(m)))
                    for c, m in zip(rows["category"], rows["modalities"])]
        allowed = set().union(*(v.tunable for v in variants))
        unknown = [k for k in overrides if k not in allowed]
        if unknown:
            raise _runner.unknown_params_error(method, unknown, allowed)
        for variant in variants:
            _runner.check_param_types(method, variant, overrides)


def _load_embedding(mdir: Path, variant):
    """The embedding a finished run left in ``mdir`` as cells x dims, or ``None``: the
    output, or a graph method's extra embedding output (scMoMaT's UMAP)."""
    import h5py

    def read(path, key=None):
        with h5py.File(path) as h:
            return np.array(h[key or ("data" if "data" in h else list(h.keys())[0])])

    if variant.output.kind == "embedding":
        p = mdir / variant.output.file
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing: run the method again.")
        return _cells_first(read(p, variant.output.dataset))
    extra = [mdir / o.file for o in variant.extra_outputs if o.kind == "embedding"]
    found = next((p for p in extra if p.exists()), None)
    return None if found is None else _cells_first(read(found))


def _cells_first(emb):
    """``emb`` as cells x dims: a matrix wider than tall is transposed."""
    return emb.T if emb.ndim == 2 and emb.shape[0] < emb.shape[1] else emb


def _label_file_sizes(dataset, category=None, method=None, *, modalities=None,
                      data_path=None) -> list:
    """``[(file name, cells)]`` of ``labels_for(...)``; ``[]`` when unreadable."""
    try:
        with warnings.catch_warnings():     # the default order is meant without a method
            warnings.filterwarnings("ignore", message=_resolve.DEFAULT_ORDER_WARNING)
            files = _resolve.labels_for(dataset, category, method, modalities=modalities,
                                        data_path=data_path)
        return [(Path(p).name, len(_eio.read_labels(p))) for p in files.values()]
    except Exception:  # noqa: BLE001 - no label files: nothing to compare against
        return []


def _batch_segments(dataset, data_path, batch) -> dict | None:
    """``{label file name: its cells' batch ids}`` when ``batch`` has one id per
    cell in the order of ``labels_for(dataset)``; else ``None``."""
    sizes = _label_file_sizes(dataset, data_path=data_path)
    if not sizes or sum(n for _, n in sizes) != len(batch):
        return None
    out, i = {}, 0
    for name, n in sizes:
        out[name] = batch[i:i + n]
        i += n
    return out


def _method_files(dataset, category, method, modalities=None, data_path=None) -> set:
    """The names of the files ``method`` reads from ``dataset``; ``set()`` when
    they cannot be resolved (another layout, or no such method)."""
    if not method or registry.method_of(method) is None:
        return set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            paths = _resolve.inputs_for(dataset, category, registry.method_of(method),
                                        modalities=modalities or None,
                                        data_path=data_path, check=False)
    except (FileNotFoundError, KeyError, ValueError):
        return set()
    return {Path(p).name for p in paths.values()}


def _label_data_files(label: Path, read=()) -> list[Path]:
    """The data files whose barcodes are the cells of one label file:
    ``cty2.csv`` -> ``rna2.h5, adt2.h5, ...``; ``atac_cty.csv`` -> the ATAC files,
    the ones a method read (``read``: file names) first."""
    stem = label.stem
    digits = stem[len(stem.rstrip("0123456789")):]
    base = stem[:len(stem) - len(digits)]
    if base == "cty":
        stems = ["rna", "adt", "atac", "atac_peak", "atac_gas"]
    elif base in ("rna_cty", "adt_cty"):
        stems = [base[:-4]]
    elif base == "atac_cty":
        stems = ["atac_peak", "atac_gas", "atac"]
    else:
        stems = []
    files = [label.parent / f"{s}{digits}.h5" for s in stems]
    return sorted(files, key=lambda f: f.name not in read)


def _dataset_folder(dataset, data_path) -> Path:
    """``<data_path>/<dataset>``, with ``None`` = config's data root."""
    base = config.data_path if data_path is None else data_path
    return Path(base) / dataset


def _root_tries(rec: dict) -> list:
    """The data roots a saved record's dataset is looked up under: the recorded
    ``data_path``, then ``data_root``, then config's (``None``)."""
    tries = [rec.get("data_path")]
    if rec.get("data_root"):
        tries.append(rec["data_root"])
    return tries + [None]


def _dataset_root(rec: dict, dataset: str) -> tuple:
    """``(root, found)``: the first root of :func:`_root_tries` that holds the
    dataset, absolute (``None`` = config's); else the recorded ``data_path``, ``False``."""
    for root in _root_tries(rec):
        if _dataset_folder(dataset, root).is_dir():
            return (None if root is None else str(Path(root).resolve())), True
    return rec.get("data_path"), False


#: what to do when a saved result's dataset folder is not found
_MOVED_FIX = ("Set mtb.config.data_path to the folder that holds it, or run "
              "rescore from the folder where mtb.run ran.")


def _folder_missing(dataset: str, roots) -> str:
    """The clause that the dataset folder is under none of ``roots``, each absolute."""
    where = dict.fromkeys(os.path.abspath(config.data_path if r is None else r)
                          for r in roots)
    return f"the dataset folder {dataset} is not found in {_and_list(where, 'or')}"


def _unplaced_batch_note(dataset: str, data_path, tried, n_ids: int) -> str:
    """The ``note`` of a record whose saved batch fits no label order."""
    if not _dataset_folder(dataset, data_path).is_dir():
        why = _folder_missing(dataset, tried or [data_path])
        return f"The saved batch is not used, because {why}. {_MOVED_FIX}"
    total = sum(n for _, n in _label_file_sizes(dataset, data_path=data_path))
    return (f"The saved batch is not used, because it has {n_ids:,} ids and the label "
            f"files of {dataset} have {total:,} cells.")


def _batch_csv(vec, dataset, data_path) -> tuple[str, str]:
    """``(batch_<sha1>.csv, CSV text)`` of a batch vector, with the dataset's cell
    ids when it has as many."""
    import hashlib
    frame = pd.DataFrame({"batch": np.asarray(vec)})
    if _dataset_folder(dataset, data_path).is_dir():
        ids = _dataset_cell_ids(dataset, data_path)
        if ids is not None and len(ids) == len(frame):
            frame.insert(0, "cell", ids)
    text = frame.to_csv(index=False)
    return f"batch_{hashlib.sha1(text.encode()).hexdigest()[:8]}.csv", text


def _read_batch_csv(text: str) -> pd.DataFrame:
    """The frame of a file :func:`_batch_csv` wrote, as text."""
    import io
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def _default_label_files(dataset, data_path) -> list:
    """The label files of ``labels_for(dataset)``, in that order; ``[]`` when
    there are none."""
    try:
        with warnings.catch_warnings():       # the default order is meant here
            warnings.simplefilter("ignore", UserWarning)
            return list(map(Path, _resolve.labels_for(dataset, data_path=data_path).values()))
    except Exception:  # noqa: BLE001 - no readable files: no ids
        return []


def _dataset_cell_ids(dataset, data_path) -> list | None:
    """The dataset's cell ids in ``labels_for(dataset)`` order (each label file's
    data-file barcodes); ``None`` when some are missing or ids repeat."""
    from .eval.pipeline import _raw_barcodes
    files = _default_label_files(dataset, data_path)
    try:
        ids = files and (_raw_barcodes(files) or [None])[0]
    except Exception:  # noqa: BLE001 - an unreadable label file: no ids
        return None
    return ids if ids and len(set(ids)) == len(ids) else None


def _embedding_cell_ids(dataset, data_path) -> list:
    """The other names of the same cells, in the same order: with each ATAC
    file's barcodes (D28's ``atac_gas.h5`` and ``atac_peak.h5`` differ in
    suffix), as they are and, when they repeat, prefixed by label file as
    :meth:`RunResult.embedding` names them."""
    from .eval.pipeline import _raw_barcodes
    files = _default_label_files(dataset, data_path)
    try:
        sizes = [len(_eio.read_labels(p)) for p in files]
    except Exception:  # noqa: BLE001 - an unreadable label file: no ids
        return []
    out: list = []
    if not files:
        return out
    for read in [()] + [(f.name,) for f in sorted(_dataset_folder(dataset, data_path)
                                                  .glob("*.h5"))]:
        raw = _raw_barcodes(files, sizes, read)
        if raw is None:
            continue
        ids, stems = raw
        named = ids if len(set(ids)) == len(ids) else \
            [f"{s}:{i}" for s, i in zip(stems, ids)]
        out += [x for x in (ids, named) if x not in out]
    return out


#: the order an argument of run / rescore follows when it is matched by position
_ROWS = "the embedding rows"


def _cell_vector(x, dataset, data_path, *, what: str = "batch", order: str | None = None,
                 tried=None) -> tuple[np.ndarray, bool]:
    """``(x per cell, aligned by id)``: an id-carrying Series, DataFrame or CSV in
    ``labels_for`` order."""
    from .eval.pipeline import _carries_ids
    if what == "batch":
        own_batches_only(dataset)
    order = order or f"the order of the label files of {dataset}"
    folder = _dataset_folder(dataset, data_path)
    if not folder.is_dir():      # only rescore: run's scan checked the folder
        why, fix = _folder_missing(dataset, tried or [data_path]), _MOVED_FIX
        no_ids = f"{why}. {fix[:-1]}"
    else:
        why = (f"the files of {dataset} have no usable cell ids (missing or "
               f"repeated barcodes)")
        fix, no_ids = f"Check that it follows {order}.", why
    if isinstance(x, (str, Path)):
        vals, first = _eio.read_labels_ids(x, what=what)
        if first is None:
            return np.asarray(vals), False
        return _eio.by_id_column(
            vals, first, _dataset_cell_ids(dataset, data_path), what=what,
            name=Path(x).name, target=dataset, order=order, no_ids=no_ids)
    vals = _eio.as_vector(x, what=what)
    if not _carries_ids(x):
        if what == "batch" and isinstance(x, pd.Series) \
                and _eio._numbers_rows(x, _dataset_cell_ids(dataset, data_path)):
            _text.warn(_eio._by_position(what), UserWarning)
        return vals, False
    ids = _dataset_cell_ids(dataset, data_path)
    index = pd.Index([str(i) for i in x.index])
    if ids is None or set(index) != set(ids):
        names = _embedding_cell_ids(dataset, data_path)
        if not index.is_unique and list(index) in names:
            # repeated barcodes in the label files' order: each value has its cell
            return vals, True
        # the cell names res.embedding(m) gave
        named = next((c for c in names if index.is_unique and set(c) == set(index)), None)
        if named is not None:
            return np.asarray(pd.Series(vals, index=index).reindex(named).to_numpy()), True
    if ids is None:
        _text.warn(f"The {what} {type(x).__name__} is matched by position, because "
                     f"{why}. {fix}", UserWarning)
        return vals, False
    return _eio._align_vector(pd.Series(vals, index=index), ids, what=what,
                              target=dataset), True


def _has_gap(values) -> bool:
    """Whether some per-cell value is missing (checked on the distinct values)."""
    return bool(_eio.missing_mask(pd.unique(np.asarray(values, dtype=object))).any())


def _check_gaps(vec, dataset: str, what: str = "batch") -> None:
    """``ValueError`` when the ``batch`` or ``labels`` vector has a cell without one."""
    if not _has_gap(vec):
        return
    miss = _eio.missing_mask(vec)
    one = "batch" if what == "batch" else "label"
    raise ValueError(
        f"{what}: {int(miss.sum()):,} of the "
        f"{len(miss):,} cells of {dataset} have no {one}{_eio._missing_example(vec)}. "
        f"Give every cell a {one}.")


def _batch_vector(batch, dataset, data_path) -> np.ndarray:
    """``batch`` as one id per cell (:func:`_cell_vector`), checked for gaps."""
    vec = _cell_vector(batch, dataset, data_path, what="batch")[0]
    _check_gaps(vec, dataset)
    return vec


def _label_gaps(dataset, data_path) -> dict:
    """``{file name: (missing, total, labels)}`` of the label files
    with a cell without a label; an unreadable file is left to the scoring."""
    gaps = {}
    for p in _label_files(dataset, data_path):
        try:
            labels = _eio.read_labels(p)
        except Exception:  # noqa: BLE001 - not a readable label file
            continue
        if not _has_gap(labels):
            continue
        miss = _eio.missing_mask(labels)
        gaps[p.name] = (int(miss.sum()), len(miss),
                        np.asarray(labels, dtype=object))
    return gaps


def _missing_labels_problem(dataset: str, data_path, rows: "pd.DataFrame | None" = None,
                            private: "pd.DataFrame | None" = None, *,
                            evaluate: bool = True) -> str:
    """Why ``run`` or ``rescore`` does not start on label files with a missing
    label, or "". Without ``evaluate`` only the ``rows`` that read such a file as
    an input stop it."""
    gaps = _label_gaps(dataset, data_path)
    if not gaps:
        return ""
    who = ""
    if not evaluate:
        # the methods that read a label file with a gap as an input, with those files
        readers = {m: [n for n in private.at[i, "_label_inputs"] if n in gaps]
                   for i, m in rows["method"].items()}
        readers = {m: read for m, read in readers.items() if read}
        if not readers:
            return ""
        who = f", and {next(iter(readers))} reads the label files"
        gaps = {k: g for k, g in gaps.items() if k in set().union(*readers.values())}
    name, (n, total, shown) = next(iter(gaps.items()))
    return (f"{name} of {dataset} has {n:,} of {total:,} cells without a label"
            f"{_eio._missing_example(shown)}{who}. Write the folder again with a label for "
            f"every cell.")


def _inputs_on_disk(row, dataset, data_path) -> bool:
    """Whether every input file of a scan row is in the folder (its other file
    checks may still fail)."""
    if row["files_ok"]:
        return True
    try:
        paths = _row_paths(row, dataset, data_path)
        if "data_dir" in paths:
            variant = registry.get(row["method"]).select(
                row["category"], set(_row_modalities(row["modalities"])))
            if not _resolve._check_data_dir(variant, paths["data_dir"])[0]:
                return False
    except Exception:  # noqa: BLE001 - unresolvable: not on disk
        return False
    return all(Path(p).exists() for p in paths.values())


def _skipped_rows(blocked, attempted, methods, dataset, data_path) -> tuple[list, int]:
    """``([(row, files on disk)], others)``: the blocked rows a real run reports
    (files on disk, or a named method not attempted) and how many it only counts."""
    named = set(methods or ())
    shown, others = [], 0
    for _, r in blocked.iterrows():
        on_disk = _inputs_on_disk(r, dataset, data_path)
        if on_disk or (r["method"] in named and r["method"] not in attempted):
            shown.append((r, on_disk))
        else:
            others += 1
    return shown, others


def _skipped_records(shown, attempted, methods, *, category, dataset, data_path,
                     version, data_root=None) -> list[dict]:
    """One ``skipped`` record per method not attempted, from its first reported row
    (one with its files on disk first)."""
    named = set(methods or ())
    first: dict = {}
    for r, on_disk in sorted(shown, key=lambda t: not t[1]):
        if r["method"] not in attempted:
            first.setdefault(r["method"], r)
    return [_record(r, {"status": "skipped", "error": r["reason"], "requested": m in named,
                        "_long": None, "caveat": _run_caveat(r["caveat"])},
                    category=category, dataset=dataset, data_path=data_path,
                    data_root=data_root, version=version)
            for m, r in first.items()]


def _record(row, fields: dict, *, category, dataset, data_path, data_root, version) -> dict:
    """The record ``run`` starts for a plan row; ``fields`` go after its ``env``."""
    return {"method": row["method"], "category": category, "dataset": dataset,
            "modalities": _row_modalities(row["modalities"]),
            "output_kind": row["output_kind"], "env": row["env"], **fields,
            "data_path": str(data_path) if data_path else None,
            "data_root": data_root,
            "multibench_version": version,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _nested_rows(plan: "pd.DataFrame") -> tuple[list, list[str]]:
    """``(indices, notes)``: the rows a run without ``modalities=`` leaves out
    because another row of the method holds their modalities, and why."""
    drop, notes = [], []
    # within one category: a scan without one also lists the other categories' rows
    keys = ["method", "category"] if "category" in plan.columns else ["method"]
    for key, rows in plan.groupby(keys, sort=False):
        method = key[0] if isinstance(key, tuple) else key
        sets = {i: set(str(mods).split("+")) for i, mods in rows["modalities"].items()}
        inside = [i for i, a in sets.items() if any(a < b for b in sets.values())]
        if not inside:
            continue
        drop += inside
        kept = [rows.at[i, "modalities"] for i in sets if i not in inside]
        left = [rows.at[i, "modalities"] for i in inside]
        notes.append(f"{method} runs on {_and_list(kept)}, not on "
                     f"{_and_list(left)}, whose files are part of it. Pass "
                     "modalities= to run a smaller combination.")
    return drop, notes


def _batch_length_problem(runs, batch, dataset, category, data_path) -> str:
    """The error when ``batch`` fits neither the dataset's label files nor the
    cells of some ``(method, modalities)`` of ``runs``; "" when it fits."""
    total = sum(n for _, n in _label_file_sizes(dataset, data_path=data_path))
    if not total or len(batch) == total:
        return ""
    own = (sum(n for _, n in _label_file_sizes(dataset, category, m, modalities=list(mods)
                                               or None, data_path=data_path))
           for m, mods in {(m, tuple(x or ())) for m, x in runs})
    if all(n in (0, len(batch)) for n in own):
        return ""
    return (f"batch has {len(batch):,} entries for the {total:,} cells of {dataset}: "
            "give one batch per cell.")


#: what scoring writes into a record
_SCORE_KEYS = ("metrics", "labels_used", "label_order_candidates", "batch_source",
               "n_batches", "_long", "scored_with", "scored_by_version", "note", "error")


def _drop_scores(rec: dict) -> dict:
    """Remove the fields of an earlier scoring."""
    for k in _SCORE_KEYS:
        rec.pop(k, None)
    return rec


def _score_record(rec, emb, dataset, category, data_path, variant, *,
                  batch=None, labels=None, metrics=None, labels_by_cell=False,
                  batch_by_cell=False, on_rank=None, batch_given=True, tried=None):
    """Fill ``rec`` with the status and scores of ``emb``. ``labels`` / ``batch`` in
    ``labels_for`` order (``*_by_cell``) go into each label order, else follow the rows."""
    stored_used = rec.get("labels_used")
    stored_cands = rec.get("label_order_candidates")
    _drop_scores(rec)
    rec["emb_shape"] = list(emb.shape)
    embedding = variant.output.kind == "embedding"
    if not embedding and _without_classification(metrics, True) is None:
        from .eval.pipeline import _classification_error
        raise ValueError(_classification_error(metrics, "graph", rec.get("method")))
    n = emb.shape[0]
    rows = labels is not None and not labels_by_cell      # labels in embedding row order
    segments = None
    if batch is not None:
        batch = np.asarray(batch)
        # in the dataset's cell order: each label order gets the ids of its files
        if batch_by_cell or not rows:
            segments = _batch_segments(dataset, data_path, batch)
        if batch_by_cell and segments is None:
            # no label files to place the ids by: they would land in the wrong rows
            rec["note"] = _unplaced_batch_note(dataset, data_path, tried, len(batch))
            batch = None
        elif len(batch) != n and segments is None:
            raise ValueError(f"batch has {len(batch)} entries, embedding has {n} cells")
    cands = [] if rows else _label_candidates(dataset, n, data_path)
    if labels is not None:
        labels = np.asarray(labels)
        if not rows:
            # the user's labels, in the dataset's cell order, replace each
            # candidate's file labels; the cells form one batch
            by_file = _batch_segments(dataset, data_path, labels)
            cands = [(names, np.concatenate([by_file[k] for k in names]),
                      np.ones(n, dtype=int))
                     for names, _, _ in cands
                     if by_file is not None and all(k in by_file for k in names)]
        if not cands:
            if len(labels) != n:
                raise ValueError(f"labels has {len(labels)} entries, embedding has {n} cells")
            cands = [(["(user labels)"], labels, np.ones(n, dtype=int))]
    if batch is not None:
        # the user's ids replace each candidate's file-of-origin batch, in that
        # candidate's order. With labels in embedding rows: the stored label order
        # when its files hold the embedding's cells, else the variant's own order;
        # a vector as long as the embedding is kept as given
        def covers(keys):
            return (segments is not None and bool(keys) and all(k in segments for k in keys)
                    and sum(len(segments[k]) for k in keys) == n)
        own = None
        if rows and segments is not None:
            stored_order = list(stored_used or ())
            own = (stored_order if covers(stored_order) else
                   [k for k, _ in _label_file_sizes(dataset, category, rec.get("method"),
                                                    modalities=rec.get("modalities") or None,
                                                    data_path=data_path)])
        placed = []
        for names, lab, _ in cands:
            keys = own if rows else names
            if covers(keys):
                placed.append((names, lab, np.concatenate([segments[k] for k in keys])))
            elif len(batch) == n:
                placed.append((names, lab, batch))
        if cands and not placed:
            raise ValueError(f"batch has {len(batch)} entries, embedding has {n} cells")
        cands = placed
    # orders whose labels and batches agree cell for cell are one ordering
    distinct = {}
    for c in cands:
        distinct.setdefault((np.asarray(c[1]).astype(str).tobytes(),
                             np.asarray(c[2]).astype(str).tobytes()), c)
    cands = list(distinct.values())
    # the label order the record was scored with is not ranked again
    same = [c for c in cands if labels is None and list(c[0]) == list(stored_used or ())]
    cands = same or cands
    if not cands:
        rec["status"], rec["note"] = "ran", _RAN_REASON["RUN_OK_NO_LABEL_MATCH"]
        if not _dataset_folder(dataset, data_path).is_dir():
            why = _folder_missing(dataset, tried or [data_path])
            rec["note"] = f"{why[:1].upper()}{why[1:]}. {_MOVED_FIX}"
        return rec
    names, val, spread = _evaluate_best_order(emb, category, cands, metrics=metrics,
                                              user_batch=batch is not None,
                                              batch_given=batch_given,
                                              on_rank=on_rank, embedding=embedding)
    if val is None:
        rec["status"] = "ran"
        errs = [s["error"] for s in spread if s.get("error")]
        if errs:
            rec["error"] = errs[0]
        return rec
    # + 0.0: a tiny negative score rounds to 0.0, not -0.0
    rec["metrics"] = {k: (None if pd.isna(x) else round(float(x), 4) + 0.0)
                      for k, x in val["Value"].items()}
    rec["labels_used"] = names
    if len(spread) > 1 or (same and stored_cands):
        rec["label_order_candidates"] = spread if len(spread) > 1 else stored_cands
    # the batch vector the batch metrics saw
    winning_batch = next(b for order, _, b in cands if order == names)
    n_batches = len(set(np.asarray(winning_batch).tolist()))
    rec["batch_source"] = ("user" if batch is not None
                           else "file_of_origin" if n_batches > 1 else None)
    rec["n_batches"] = n_batches
    rec["_long"] = _to_long(val, method=rec["method"], dataset=dataset, category=category)
    if len(rec["_long"]):
        rec["scored_with"] = str(rec["_long"]["scored_with"].iloc[0])
    from . import __version__
    # next to multibench_version, the version that ran the method
    rec["scored_by_version"] = __version__
    rec["status"] = "scored"
    return rec


def run(dataset: str, category: str, out_dir, *, methods=None, modalities=None,
        params: dict | None = None, batch=None, evaluate: bool = True,
        timeout: float | None = None, skip_existing: bool = False,
        verbose: bool = True) -> "RunResult":
    """Run the runnable methods of a category on a dataset folder, and score them.

    Only the methods ``mtb.scan`` marks runnable run. A method's failure is
    recorded, never raised, and the results are saved under ``out_dir``.
    ``methods=["Matilda"]`` runs one method. ``mtb.scan`` shows the plan
    first, with each command.

    Parameters
    ----------
    dataset : str
        Dataset folder: its path, or its name under
        ``mtb.config.data_path``.
    category : str
        Integration category: ``vertical``, ``diagonal``, ``mosaic`` or ``cross``.
    out_dir : path
        Output root, one ``<out_dir>/<method>_<dataset>/`` per method. An
        ``embedding.h5`` there also holds its cells' names (``barcodes``).
    methods : list[str] | None
        Method ids to run, as a list; ``None`` = every runnable method.
    modalities : list[str] | str | None
        Modality tokens of one combination, e.g. ``["rna", "adt"]``; ``None`` =
        every combination the folder holds; ``"all"`` = every combination.
    params : dict | None
        Hyperparameters per method, ``{"Matilda": {"epochs": 3}}``;
        ``mtb.params_for`` lists the keys.
    batch : array-like | Series | path | None
        One batch per cell; a Series or a barcode-indexed CSV is aligned by
        barcode. ``None`` = each cell's label file.
    evaluate : bool
        Score each embedding; ``False`` only runs (status ``ran``).
    timeout : float | None
        Wall-clock limit per method, in seconds; ``None`` = none.
    skip_existing : bool
        Reuse an output already in ``out_dir`` instead of running the method
        again, to resume an interrupted run.
    verbose : bool
        Print ``[run] ...`` progress lines.

    Returns
    -------
    RunResult
        Read ``summary`` first, ``embedding(method)`` for an embedding.

    Raises
    ------
    FileNotFoundError
        The dataset folder does not exist; the message lists the folders present.
    ValueError
        Nothing in the selection can run, an ``out_dir`` that holds the result
        of another dataset or category, or a ``batch`` that does not fit the
        cells; checked before anything runs.
    KeyError
        Unknown method id. A ``params`` key the method does not accept fails
        that method (status ``failed``).

    Examples
    --------
    >>> import multibench as mtb
    >>> res = mtb.run("D11", "vertical", out_dir="out/")
    >>> res = mtb.run("D11", "vertical", out_dir="out/", methods=["Matilda"],
    ...               params={"Matilda": {"epochs": 10}})
    """
    given = dataset
    dataset, data_path = _resolve.split_dataset(dataset, None)
    out_dir = Path(out_dir).expanduser()
    # the arguments first, so a bad combination is not reported as an I/O error
    methods, every, modalities, params = _checked_args(category, methods, modalities,
                                                       params, "run")
    if skip_existing and params:
        raise ValueError("skip_existing=True would reuse outputs made with other params. "
                         "Pass a new out_dir, or skip_existing=False.")
    root = Path(data_path) if data_path is not None else config.data_path
    # the on-disk spelling, once, so out_dir names and records agree (one warning)
    dataset = _resolve.canonical_dataset(root, dataset)
    # a real run checks params per method and records a bad key as FAIL
    plan_df, private = _scan(dataset, category, data_path=data_path, methods=methods,
                             modalities="all" if every else modalities, out_dir=out_dir)
    not_listed = _not_listed_note(plan_df, dataset, private.attrs["folder"], category)

    if plan_df.empty:
        # a modalities= selection that matches nothing, without methods=
        raise _no_variant_error(category, methods, modalities)
    batch_vec = None if batch is None else _batch_vector(batch, dataset, data_path)
    bad = batch_vec is not None and evaluate and _batch_length_problem(
        [(r["method"], _row_modalities(r["modalities"])) for _, r in plan_df.iterrows()
         if _would_run(r)], batch_vec, dataset, category, data_path)
    if bad:
        raise ValueError(bad)
    if skip_existing:
        # a saved output is only scored: it needs no environment
        saved = [(out_dir / f"{r['method']}_{dataset}" / registry.get(r["method"]).select(
            r["category"], set(_row_modalities(r["modalities"]))).output.file).exists()
            for _, r in plan_df.iterrows()]
        plan_df = plan_df.assign(runnable=plan_df["runnable"].to_numpy(bool) | saved)
    blocked = plan_df[~plan_df["runnable"]]
    plan_df = plan_df[plan_df["runnable"]]
    if plan_df.empty and skip_existing:
        raise ValueError(f"No saved output in {out_dir}. Pass the out_dir of the earlier run.")
    if plan_df.empty:
        # not one method could start: the request is wrong, and an empty result
        # would read "0 failed", as success
        raise ValueError(_nothing_runnable_message(
            dataset, category, blocked, methods, given=given,
            modalities=modalities, others=private["_other"], every=every,
            note="" if methods else not_listed))
    if modalities is None:
        nested, nested_notes = _nested_rows(plan_df)
        plan_df = plan_df.drop(index=nested)
        if verbose and not methods:     # as the "more rows" line: scan has them
            for note in nested_notes:
                print(f"[run] {note}", flush=True)
    # the scores would count a missing label as a cell type, and a method that
    # reads the file would train on it: stop before out_dir is made
    unlabelled = _missing_labels_problem(dataset, data_path, plan_df, private,
                                         evaluate=evaluate)
    if unlabelled:
        raise ValueError(unlabelled)

    # saved in out_dir so that rescore can reuse it
    batch_file = (_batch_csv(batch_vec, dataset, data_path)
                  if batch_vec is not None and evaluate else None)
    # absolute: a saved result read from another directory still finds the data
    data_root = os.path.abspath(root)
    # the save after the run would refuse this folder: refuse it now
    _check_save_target(out_dir, dataset, category)
    attempted = set(plan_df["method"])
    shown, others = _skipped_rows(blocked, attempted, methods, dataset, data_path)
    if verbose:
        several = Counter(r["method"] for r, _ in shown)
        for r, _ in shown:
            mods = f" ({r['modalities']})" if several[r["method"]] > 1 else ""
            print(f"[run] skipping {r['method']}{mods}: {r['reason']}", flush=True)
        if others and not methods:      # with methods=, only other variants of those
            rows = "row needs" if others == 1 else "rows need"
            print(f"[run] {others} more {rows} files this folder does not have. "
                  f"mtb.scan shows {'it' if others == 1 else 'them'}.", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    earlier = _saved_records(out_dir) if skip_existing else {}
    from . import __version__ as _pkg_version

    for _, row in plan_df.iterrows():
        method = row["method"]
        method_dir = out_dir / f"{method}_{dataset}"
        rec = _record(row, {"status": "?", "_long": None, "caveat": _run_caveat(row["caveat"]),
                            "params_used": dict(params.get(method) or {}),
                            "out_dir": str(method_dir)},
                      category=category, dataset=dataset, data_path=data_path,
                      data_root=data_root, version=_pkg_version)
        mods = rec["modalities"]
        # a plan row's own (category, modalities): select cannot fail
        variant = registry.get(method).select(category, set(mods))
        output = method_dir / variant.output.file
        start = time.time()
        saved = earlier.get(method)
        if saved is not None and _scorable(saved) and output.exists():
            # a saved run is scored again, as rescore does: its record is kept,
            # with its folder as found from here
            rec = (RunResult([dict(saved, out_dir=str(method_dir))], dataset, category,
                             out_dir).rescore(batch=batch, verbose=False).records[0]
                   if evaluate else dict(saved))
            rec["reused"] = True
            if variant.output.kind == "embedding" and rec.get("emb_shape"):
                _write_cell_names(rec, output, dataset, category, data_path)
            if verbose:
                print(f"[run] {method} ({category}/{dataset}): reusing the saved run "
                      f"-> {_status(rec['status'])} {_ari_tail(rec)}".rstrip(), flush=True)
            records.append(rec)
            continue
        if verbose:
            print(f"[run] {method} ({category}/{dataset}) ...", flush=True)
            if rec["caveat"]:              # it starts with the method name
                print(f"[run]   {rec['caveat']}", flush=True)
        try:
            with _deadline(timeout):
                inputs = _resolve.inputs_for(dataset, category, method, modalities=mods or None,
                                             data_path=data_path, check=True)
                reused = skip_existing and output.exists()
                overrides = params.get(method)
                _runner.check_params(method, variant, overrides)   # before the method starts
                if reused:
                    if verbose:
                        print(f"[run]   reusing existing output in {method_dir}", flush=True)
                    res = None
                else:
                    res = _run(method=method, category=category, inputs=inputs,
                               out_dir=str(method_dir), params=overrides)
                rec["reused"] = bool(reused)
                rec["run_sec"] = round(time.time() - start, 1)
                emb = (_cells_first(np.asarray(res))
                       if variant.output.kind == "embedding" and res is not None
                       else _load_embedding(method_dir, variant))
                if emb is None:
                    rec["status"], rec["note"] = "ran", _RAN_REASON["RUN_OK_NO_EMBEDDING"]
                else:
                    rec["emb_shape"] = list(emb.shape)
                    if not evaluate:
                        rec["status"] = "ran"
                    else:
                        if batch_file:
                            rec["batch_file"] = batch_file[0]
                        try:
                            _score_record(rec, emb, dataset, category, data_path, variant,
                                          batch=batch_vec)
                        except TimeoutError:
                            raise
                        except Exception as e:      # the run itself succeeded
                            rec["status"] = "ran"
                            rec["error"] = f"{type(e).__name__}: {e}"
                    if variant.output.kind == "embedding":
                        # evaluate on the file then finds each cell by its name
                        _write_cell_names(rec, output, dataset, category, data_path)
        except TimeoutError as e:
            rec["status"] = "timeout"
            rec["error"] = str(e)
            rec["run_sec"] = round(time.time() - start, 1)
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = _clip_error(f"{type(e).__name__}: {e}")
            rec["traceback"] = traceback.format_exc()[-1200:]
            rec["run_sec"] = round(time.time() - start, 1)
        if verbose:
            tail = _error_tail(rec["error"]) if rec.get("error") else _ari_tail(rec)
            print(f"[run]   -> {_status(rec['status'])} ({rec.get('run_sec')}s) {tail}".rstrip(),
                  flush=True)
        # scripts_commit, env_flavor, hostname
        rec.update(_UNKNOWN_PROVENANCE if rec.get("reused")
                   else scripts.run_provenance(row["env"]))
        records.append(rec)
    records += _skipped_records(shown, attempted, methods, category=category,
                                dataset=dataset, data_path=data_path,
                                version=_pkg_version, data_root=data_root)

    result = RunResult(records, dataset, category, out_dir)
    if batch_file and any(r.get("batch_file") for r in records):
        result._batches[batch_file[0]] = batch_file[1]
    if earlier:
        # resuming: a method that cannot run here leaves the saved folder as it was
        kept = RunResult([r for r in records if r["status"] != "skipped"], dataset,
                         category, out_dir)
        kept._batches = result._batches
        kept.save()
    else:
        result.save()
    return result
