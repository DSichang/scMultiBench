"""The canonical ``.h5`` file of a dataset folder: what a file is, the names it
holds, and how two lists of cell barcodes compare. The resolver and the
exporter both use these."""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

# peak ids come as chr_start_end, chr-start-end or chr:start-end; Signac needs the last
_PEAK_RE = re.compile(r"^(.+?)[-_:](\d+)[-_](\d+)$")
#: Stored non-zero values sampled by the raw-count check.
_COUNT_SAMPLE = 5000


def _has_fraction(vals) -> bool:
    """True when a sample of up to ``_COUNT_SAMPLE`` non-zero values of
    ``vals`` holds a value whose fractional part exceeds 1e-6."""
    v = np.asarray(vals).ravel()
    if v.size == 0 or v.dtype.kind != "f":
        return False                    # integer / bool storage: whole numbers
    if v.size > 20 * _COUNT_SAMPLE:     # thin out before the non-zero filter
        v = v[np.linspace(0, v.size - 1, 20 * _COUNT_SAMPLE).astype(int)]
    v = v[(v != 0) & np.isfinite(v)]
    if v.size > _COUNT_SAMPLE:
        v = v[np.linspace(0, v.size - 1, _COUNT_SAMPLE).astype(int)]
    return bool(np.any(np.abs(v - np.round(v)) > 1e-6))


def _is_canonical_h5(path: Path) -> bool:
    """``matrix/data`` is a 2-D dataset; a 10x Genomics ``.h5`` has a 1-D one."""
    try:
        with h5py.File(path, "r") as f:
            d = f.get("matrix/data")
            return isinstance(d, h5py.Dataset) and d.ndim == 2
    except (OSError, KeyError):
        return False


def _is_tenx_h5(path: Path) -> bool:
    """A 10x Genomics ``.h5``: a sparse matrix with ``matrix/indptr``."""
    try:
        with h5py.File(path, "r") as f:
            return "matrix/indptr" in f
    except OSError:
        return False


def _strings(raw) -> list[str]:
    """The names an h5 string dataset holds, as ``str``."""
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


def _peak_fraction(feats) -> float:
    if not len(feats):
        return 0.0
    return sum(1 for x in feats if _PEAK_RE.match(x)) / len(feats)


#: a trailing ``-<n>`` (10x GEM-well) suffix of a cell barcode
_BARCODE_SUFFIX_RE = re.compile(r"-\d+$")


def _comparable(names, ref):
    """``(names, ref)`` as written, then without a trailing ``-<n>``; a pair
    with a repeated barcode is left out."""
    names, ref = [str(x) for x in names], [str(x) for x in ref]
    for key in (None, _BARCODE_SUFFIX_RE):
        a = names if key is None else [key.sub("", x) for x in names]
        b = ref if key is None else [key.sub("", x) for x in ref]
        if len(set(a)) == len(a) and len(set(b)) == len(b):
            yield a, b


def _cell_order(names, ref):
    """``("same", None)``, ``("order", idx)`` (``names[idx]`` is in ``ref``
    order), ``("other", None)`` or ``("unknown", None)`` (non-unique
    barcodes). A trailing ``-<n>`` does not count when the names stay unique
    without it."""
    judged = False
    for a, b in _comparable(names, ref):
        judged = True
        if a == b:
            return "same", None
        if set(a) == set(b):
            pos = {k: i for i, k in enumerate(a)}
            return "order", [pos[k] for k in b]
    return ("other" if judged else "unknown"), None
