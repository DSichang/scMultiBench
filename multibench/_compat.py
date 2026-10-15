"""Deprecation shims: the 0.2 spellings warn and forward for one release."""
from __future__ import annotations

import functools
import warnings

DEPRECATED_SINCE = "0.3.0"


def warn(old: str, new: str, *, stacklevel: int = 3) -> None:
    """Emit the package's ``DeprecationWarning`` for ``old`` -> ``new``."""
    warnings.warn(f"{old} is deprecated since {DEPRECATED_SINCE} and will be removed in "
                  f"0.4; use {new} instead", DeprecationWarning, stacklevel=stacklevel)


def deprecated_alias(old: str, new: str, fn):
    """``fn`` under the retired name ``old``, warning that callers should write ``new``."""
    @functools.wraps(fn)
    def alias(*args, **kwargs):
        warn(old, new)
        return fn(*args, **kwargs)
    alias.__name__ = alias.__qualname__ = old.rsplit(".", 1)[-1]
    alias.__doc__ = f"Deprecated alias of ``{new}`` (removed in 0.4); emits DeprecationWarning."
    return alias


def legacy_kwargs(translate):
    """Decorator: pass the call's ``**kwargs`` through ``translate``, which maps the
    deprecated keywords (warning with ``stacklevel=4``, the caller)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **translate(kwargs))
        return wrapper
    return deco
