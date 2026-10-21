"""Small helpers every module uses: :class:`Text` for printable text and
:func:`warn`, a warning attributed to the user's line."""
from __future__ import annotations

import os
import sys
import warnings

_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep


class Text(str):
    """A ``str`` whose notebook / REPL echo is the text itself, line breaks and
    all (``cite``); it is a ``str`` in every other way."""

    def __repr__(self) -> str:
        return str(self)


def caller_stacklevel() -> int:
    """``warnings.warn``'s ``stacklevel`` that names the first frame outside the
    package, counted from the function that calls this one."""
    level, frame = 1, sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame, level = frame.f_back, level + 1
    return level


def warn(message: str, category: type = UserWarning) -> None:
    """``warnings.warn`` attributed to the user's line: the first frame outside
    the package (a script, a notebook cell, a test), whatever the call depth."""
    warnings.warn(message, category, stacklevel=caller_stacklevel())
