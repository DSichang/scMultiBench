"""Deprecation shims shared by every namespace of the package.

The 0.3.0 public-surface cut renamed a handful of entry points. Each old
spelling is kept for ONE release as a thin alias that warns and forwards, so
a 0.2.1 script keeps working while saying what to change.
"""
from __future__ import annotations

import functools
import warnings

#: the release in which the aliases were introduced (and the one they warn about)
DEPRECATED_SINCE = "0.3.0"


def deprecated_alias(old: str, new: str, fn):
    """Wrap ``fn`` as the deprecated spelling ``old`` of the public name ``new``.

    Parameters
    ----------
    old : str
        The spelling being retired, e.g. ``"plan"``.
    new : str
        What callers should write instead, e.g. ``"scan"`` or
        ``"method_info(m)['runtime']"``; quoted verbatim in the warning.
    fn : callable
        The function the alias forwards to. It receives the alias's arguments
        unchanged and its return value is returned as is.

    Returns
    -------
    callable
        A function named ``old`` that emits ``DeprecationWarning`` (pointing at
        the caller) and then calls ``fn(*args, **kwargs)``.
    """
    @functools.wraps(fn)
    def alias(*args, **kwargs):
        warnings.warn(
            f"{old} is deprecated since {DEPRECATED_SINCE} and will be removed in "
            f"the next release; use {new}",
            DeprecationWarning, stacklevel=2)
        return fn(*args, **kwargs)

    alias.__name__ = old
    alias.__qualname__ = old
    alias.__doc__ = f"Deprecated alias of ``{new}`` (kept for one release; emits DeprecationWarning)."
    return alias
