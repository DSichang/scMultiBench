"""Small helpers every module uses: :func:`warn`, a warning attributed to the
user's line, :func:`_and_list` and :func:`public_dir`."""
from __future__ import annotations

import os
import sys
import warnings

_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep


def warn(message: str, category: type = UserWarning) -> None:
    """``warnings.warn`` attributed to the user's line: the first frame outside
    the package (a script, a notebook cell, a test), whatever the call depth."""
    level, frame = 1, sys._getframe()
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)


def _and_list(items, last: str = "and") -> str:
    """``'A'``, ``'A and B'``, ``'A, B and C'`` (``last='or'``: ``'A, B or C'``); repeats stay."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} {last} {items[-1]}"


def public_dir(namespace: dict):
    """A module's ``__dir__`` (PEP 562): its ``__all__`` and the dunders only;
    the submodules stay importable."""
    return lambda: sorted(set(namespace["__all__"]) | {n for n in namespace if n.startswith("__")})
