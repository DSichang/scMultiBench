"""Deprecation shims: retired spellings warn and forward until their removal."""
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


def deprecated(old: str, new: str | None = None, *, stacklevel: int = 3) -> None:
    """Emit the 0.4 ``FutureWarning`` for ``old``: use ``new``, or it has no effect."""
    tail = f"Use {new}." if new else "It has no effect."
    warnings.warn(f"{old} is deprecated and will be removed in 0.5. {tail}",
                  FutureWarning, stacklevel=stacklevel)


def renamed(old: str, new: str, fn):
    """``fn`` under the deprecated name ``old``, warning that callers should write ``new``."""
    @functools.wraps(fn)
    def alias(*args, **kwargs):
        deprecated(old, new)
        return fn(*args, **kwargs)
    alias.__name__ = alias.__qualname__ = old.rsplit(".", 1)[-1]
    alias.__doc__ = f"Deprecated: use ``{new}``."
    return alias


def keywords(func: str, **old):
    """Decorator for the deprecated keywords of ``func``: ``old`` maps each one to
    the keyword that replaces it, or to ``None`` when it has no effect (dropped)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for name, new in old.items():
                if name in kwargs:
                    deprecated(f"{func}({name}=)", new and f"{new}=")
                    value = kwargs.pop(name)
                    if new in kwargs:
                        raise TypeError(f"{func}() got {name}= and {new}=; pass {new}= only")
                    if new:
                        kwargs[new] = value
            return fn(*args, **kwargs)
        return wrapper
    return deco


def module_getattr(module: str, prefix: str, names: dict):
    """A PEP 562 ``__getattr__`` for ``module`` serving ``names`` (``{name: (value,
    new)}``) with the deprecation warning, which spells them ``prefix.name``."""
    def __getattr__(name):
        if name not in names:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        value, new = names[name]
        deprecated(f"{prefix}.{name}", new)
        return value
    return __getattr__


def legacy_kwargs(translate):
    """Decorator: pass the call's ``**kwargs`` through ``translate``, which maps the
    deprecated keywords (warning with ``stacklevel=4``, the caller)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **translate(kwargs))
        return wrapper
    return deco
