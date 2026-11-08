"""The reference tables: ``mtb.catalog.datasets()`` (which datasets) and
``mtb.catalog.metrics()`` (which metrics)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from .. import _settings
from .._text import _and_list
from ..engine import registry

#: display spellings -> method id, for those the folding of :func:`_canonical_id`
#: does not reach
_ALIASES = {
    "seurat(wnn)": "Seurat_WNN",
    "mofa+": "MOFA2",
}


def _registry_ids() -> list[str]:
    """The registry's method ids."""
    return [s.id for s in registry.load()]


def _canonical_id(name: str, *, strict: bool = False) -> str:
    """The method id for any known spelling (``"MOFA+"`` -> ``"MOFA2"``,
    ``"totalvi"`` -> ``"totalVI"``): the alias table, then a case-folded match
    of the ids after collapsing spaces and dots to ``_``; an unknown name comes
    back folded, or with ``strict=True`` raises ``KeyError``."""
    key = str(name).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    ids = _registry_ids()
    by_lower = {i.lower(): i for i in ids}
    folded = re.sub(r"[ .]+", "_", str(name).strip())
    if folded in ids:
        return folded
    if folded.lower() in by_lower:
        return by_lower[folded.lower()]
    if strict:
        raise KeyError(registry.unknown_method_message(name))
    return folded


#: the metric codes of each ``metrics=`` family, in the paper's order. iASW and iF1
#: are isolated-label scores, which scIB files under bio conservation: a
#: single-batch dataset has them and no batch-correction result.
FAMILY_METRICS = {
    "clustering": ["ARI", "NMI", "ASW", "iASW", "iF1", "cLISI"],
    "batch": ["ASW_batch", "ARI_batch", "NMI_batch", "GC", "iLISI", "kBET"],
    "classification": ["OCA", "ACA", "Spec", "Sens", "F1"],
}

#: metric names in lower case -> code: each code, and the spellings of some
#: published metric.csv (scIB's long names)
_METRIC_CANON = {c.lower(): c for codes in FAMILY_METRICS.values() for c in codes} | {
    "ifi": "iF1",
    "ari_cluster/label": "ARI",
    "nmi_cluster/label": "NMI",
    "asw_label": "ASW",
    "isolated_label_f1": "iF1",
    "isolated_label_silhouette": "iASW",
}


def _known_metrics() -> list[str]:
    """The metric codes, in family order: the rows of :func:`metrics`."""
    return [c for codes in FAMILY_METRICS.values() for c in codes]


def _canonical_metric(code: str) -> str | None:
    """The canonical code for a metric name in any case or scIB spelling
    (``"kbet"`` -> ``"kBET"``, ``"isolated_label_f1"`` -> ``"iF1"``); ``None``
    for a blank; an unknown code comes back stripped."""
    key = "" if code is None else str(code).strip().lower()
    if key in ("", "nan"):
        return None
    return _METRIC_CANON.get(key, str(code).strip())


class MetricSelection(NamedTuple):
    """What a ``metrics=`` argument resolved to: ``family`` (the token, or for
    a list the family holding every code, else ``"all"``), ``codes`` (``None``
    = no restriction, else the canonical codes in request order) and
    ``explicit`` (a list was given)."""
    family: str
    codes: list | None
    explicit: bool


def _unknown_metrics(kw: str, what: str) -> str:
    """The one error text of a ``metrics=`` value that names no family or code."""
    return (f"Unknown {kw}= {what}. Pass 'all', 'clustering', 'batch' or "
            f"'classification', or metric codes such as {kw}=['ARI', 'NMI'].")


def metric_selection(metrics, *, extra=(), kw: str = "metrics") -> MetricSelection:
    """Resolve the ``metrics=`` knob shared by ``evaluate``, ``load_results`` and ``rank``.

    Parameters
    ----------
    metrics
        ``None`` (no restriction), a family token (``"clustering"``,
        ``"batch"``, ``"classification"``, ``"all"``), one metric code, or a list
        of codes and families (case/alias tolerant: ``["ari", "kbet"]`` -> ``ARI,
        kBET``; a family in a list adds its codes, kBET only when named).
    extra : collection of str
        Codes accepted on top of :func:`_known_metrics` - the metrics present
        in a frame, so a user's own metric name in a long file can still be
        selected; they are listed in the error too.
    kw : str
        The keyword the value arrived through, named in the errors.

    Returns
    -------
    MetricSelection

    Raises
    ------
    ValueError
        A string that is neither a family token nor a metric code, an unknown
        code in a list, or an empty list.
    """
    def family_word(m) -> bool:
        return isinstance(m, str) and m.lower() in (*FAMILY_METRICS, "all")

    if family_word(metrics):
        metrics = metrics.lower()          # "Clustering" names the family too
    if metrics is None or (isinstance(metrics, str) and metrics == "all"):
        return MetricSelection("all", None, False)
    if isinstance(metrics, str):
        if metrics in FAMILY_METRICS:
            return MetricSelection(metrics, list(FAMILY_METRICS[metrics]), False)
        code = _canonical_metric(metrics)
        if code in _known_metrics() or code in set(map(str, extra)):
            metrics = [code]               # one code: a list of one
        else:
            raise ValueError(_unknown_metrics(kw, f"token {metrics}"))
    wanted = list(metrics)
    if not wanted:
        raise ValueError(
            f"{kw}=[] selects nothing. Pass None for every metric, a family token, "
            f"or a non-empty list of codes.")
    valid = _known_metrics()
    present = sorted(set(map(str, extra)))
    codes, unknown = [], []
    for m in wanted:
        if family_word(m):
            # a family in the list: its codes (kBET only when named, as for the token)
            family = valid if m.lower() == "all" else FAMILY_METRICS[m.lower()]
            codes += [c for c in family if c != "kBET" and c not in codes]
            continue
        c = _canonical_metric(m)
        if c is None or (c not in valid and c not in present):
            unknown.append(m)
        else:
            codes.append(c)
    if unknown:
        raise ValueError(_unknown_metrics(kw, f"token{'s' if len(unknown) > 1 else ''} "
                                              f"{', '.join(map(str, unknown))}")
                         + (f" This frame also has {', '.join(present)}." if present else ""))
    codes = list(dict.fromkeys(codes))
    holding = [f for f, members in FAMILY_METRICS.items() if all(c in members for c in codes)]
    return MetricSelection(holding[0] if holding else "all", codes, True)


def metrics_in(metrics, names, *, table: str = "The table has") -> list | None:
    """The metric names of a table that ``metrics=`` selects, for
    ``load_results``, ``rank``, ``plot.bubble`` and ``plot.bar``.

    :func:`metric_selection` reads the value (``names`` count as codes too);
    a family selects the table's codes of it, a code the table's spelling of
    it. ``None`` = every metric.

    Raises
    ------
    ValueError
        What :func:`metric_selection` raises, a family the table has no code
        of, or a code it has no row of; ``table`` begins the message.
    """
    names = list(dict.fromkeys(n for n in names if n is not None))
    sel = metric_selection(metrics, extra=names)
    if sel.codes is None or not names:
        return sel.codes
    spelling: dict = {}
    for n in names:
        spelling.setdefault(_canonical_metric(n), n)
    out, absent = [], []
    for r in [metrics] if isinstance(metrics, str) else list(metrics):
        word = str(r).strip().lower()
        if word == "all":
            out += [spelling[c] for c in _known_metrics() if c in spelling]
            out += sorted(set(names) - set(out), key=str)
        elif word in FAMILY_METRICS:
            got = [spelling[c] for c in FAMILY_METRICS[word] if c in spelling]
            out += got
            absent += [] if got else [f"{word} metrics"]
        elif r in names or _canonical_metric(r) in spelling:
            out.append(r if r in names else spelling[_canonical_metric(r)])
        else:
            absent.append(str(r))
    if absent:
        raise ValueError(f"{table} no {_and_list(dict.fromkeys(absent))}. "
                         f"Pick metrics= from {_and_list(sorted(map(str, names)))}.")
    return list(dict.fromkeys(out))


def _dataset_sort_key(ds: str):
    """Natural order for dataset ids: ``D2 < D11 < D11s < SD7``."""
    m = re.match(r"^([A-Za-z]*)(\d+)(.*)$", str(ds))
    if not m:
        return (str(ds), 0, "")
    return (m.group(1).upper(), int(m.group(2)), m.group(3))


def datasets(*, category: str | None = None) -> pd.DataFrame:
    """The datasets ``mtb.data.fetch`` downloads or ``mtb.load_results`` holds.

    Parameters
    ----------
    category : str | None
        ``vertical``, ``diagonal``, ``mosaic`` or ``cross``: only the datasets
        of that category; ``None`` = all.

    Returns
    -------
    pandas.DataFrame
        One row per dataset: ``dataset``, ``category`` and ``has_results``
        (``mtb.load_results`` has stored results for it). ``mtb.data.fetch``
        downloads every one.

    Examples
    --------
    >>> import multibench as mtb
    >>> ds = mtb.catalog.datasets()
    >>> ds.dataset.tolist()
    >>> ds[ds.has_results]
    """
    # from the result tree at each call, so no dataset load_results serves is missing
    from . import results
    from .fetch import AVAILABLE, DEMO_CATEGORIES
    cat_of: dict[str, list[str]] = {}
    for _, df in results._read_rerun_files(results._base_path()):
        for ds, cat in df[["dataset", "category"]].drop_duplicates().astype(str).itertuples(
                index=False):
            cat_of.setdefault(ds, []).append(cat)
    served = set(cat_of)
    for ds, cat in DEMO_CATEGORIES.items():
        cat_of.setdefault(ds, [cat])
    # the full datasets first, then the small ones of the tutorials
    ids = sorted(set(AVAILABLE) | served,
                 key=lambda d: (bool(_dataset_sort_key(d)[2]), _dataset_sort_key(d)))
    out = pd.DataFrame({"dataset": ids})
    out["category"] = out["dataset"].map(lambda d: ";".join(cat_of[d]) if d in cat_of else None)
    fetchable = out["dataset"].isin(set(AVAILABLE))
    out["has_results"] = out["dataset"].isin(served)
    # the rows a user can act on: data to fetch or stored results to compare with
    out = out[fetchable | out["has_results"]]
    if category is not None:
        registry.check_category(category)
        out = out[out["category"].fillna("").str.split(";").map(lambda cs: category in cs)]
    return out.reset_index(drop=True)


def metrics() -> pd.DataFrame:
    """Table describing the metrics: one row per metric code.

    Returns
    -------
    pandas.DataFrame
        One row per metric code: ``metric``, ``family`` (``clustering``,
        ``batch`` or ``classification``, the families ``metrics=`` takes) and
        ``description``.

    Examples
    --------
    >>> import multibench as mtb
    >>> tab = mtb.catalog.metrics()
    >>> tab.set_index("metric").loc["kBET", "description"]

    See Also
    --------
    mtb.evaluate : computes these metrics for an embedding.
    """
    raw = pd.read_csv(Path(_settings._FILES_PATH) / "metric_full.csv")
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})
    family = {c: f for f, codes in FAMILY_METRICS.items() for c in codes}
    raw.insert(1, "family", raw["metric"].map(family))
    return raw
