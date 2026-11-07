"""What a method's script still fixes, and what its library offers beyond it.

`params_for` reports the hyperparameters a method script takes on its command
line. This module serves two further facts from ``upstream_knobs.yaml``: the
values a script still fixes (with the `file:line` that fixes it), and the
settings the wrapped library documents that the script has no parameter for.
`tools/gen_upstream_knobs.py` generates the table from the package's scripts
and drops an entry whose code is gone or whose setting became a parameter.
"""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

_DATA = Path(__file__).parent / "upstream_knobs.yaml"


@functools.lru_cache(maxsize=1)
def _table() -> dict:
    return yaml.safe_load(_DATA.read_text()) or {}


def knobs_for(method: str) -> dict:
    """Return ``{fixed_in_script, upstream_knobs, upstream_url}``.

    Empty lists mean the audit has nothing recorded for this method: it was
    not part of the audited set, or everything it recorded became a parameter.
    """
    entry = _table().get(method) or {}
    return {
        "fixed_in_script": list(entry.get("fixed_in_script") or []),
        "upstream_knobs": list(entry.get("upstream_knobs") or []),
        "upstream_url": entry.get("upstream_url") or None,
    }
