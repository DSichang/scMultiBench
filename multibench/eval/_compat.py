"""Deprecation shims for the 0.3.0 public-surface cut (all removed in 0.4).

Two helpers: :func:`warn` emits the one ``DeprecationWarning`` text every
deprecated spelling uses, and :func:`deprecated_alias` wraps a function under
its old name. :func:`legacy_kwargs` routes a call's keyword arguments through
a per-function translator so a deprecated keyword is mapped (with the warning)
and a removed one fails with a ``TypeError`` naming its replacement, while the
public signature stays exactly the new one (``functools.wraps`` keeps
``inspect.signature`` / ``help`` on the wrapped function).

This module mirrors ``multibench/_compat.py`` (the top-level helper the
discover/run side creates); fold the two into one when both land on main.
"""
from __future__ import annotations

import functools
import warnings


def warn(old: str, new: str, *, stacklevel: int = 3) -> None:
    """Emit the package's ``DeprecationWarning`` for ``old`` -> ``new``."""
    warnings.warn(f"{old} is deprecated since 0.3.0 and will be removed in 0.4; "
                  f"use {new} instead", DeprecationWarning, stacklevel=stacklevel)


def deprecated_alias(old: str, new: str, fn):
    """``fn`` under its deprecated name ``old``: warns, then forwards every call."""
    @functools.wraps(fn)
    def alias(*args, **kwargs):
        warn(old, new)
        return fn(*args, **kwargs)
    alias.__name__ = alias.__qualname__ = old.rsplit(".", 1)[-1]
    alias.__doc__ = f"Deprecated alias of ``{new}`` (removed in 0.4); emits DeprecationWarning."
    return alias


def legacy_kwargs(translate):
    """Decorator: pass the call's ``**kwargs`` through ``translate`` first.

    ``translate(kwargs) -> kwargs`` maps deprecated keywords onto the new
    spelling (calling :func:`warn` with ``stacklevel=4`` so the warning points
    at the caller) and raises ``TypeError`` for removed ones.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **translate(kwargs))
        return wrapper
    return deco
