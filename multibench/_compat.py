"""Deprecations: an old spelling keeps working until 0.5 with one FutureWarning;
a removed one raises, naming what replaces it.

Messages, as the 0.4 policy writes them::

    <old> is deprecated and will be removed in 0.5. Use <new>.
    <old> is deprecated and will be removed in 0.5; it has no effect.
    <old> is deprecated and will be removed in 0.5; <what to do>.
    <old> was removed in <version>. Use <new>.

The tails come from :func:`use` (a name to switch to), :func:`note` (what to do
when no name replaces it) and :data:`NO_EFFECT`.
"""
from __future__ import annotations

import functools
import os
import sys
import warnings

_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep


def use(new: str) -> str:
    """The tail of a message whose old spelling has a replacement ``new``."""
    return f". Use {new}."


def note(text: str) -> str:
    """The tail of a message with no replacement name: ``"; <text>."``."""
    return f"; {text}."


NO_EFFECT = note("it has no effect")


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


def deprecated(old: str, tail: str = NO_EFFECT) -> None:
    """The 0.4 ``FutureWarning`` for ``old``."""
    warnings.warn(f"{old} is deprecated and will be removed in 0.5{tail}",
                  FutureWarning, stacklevel=caller_stacklevel())


def removed_text(old: str, since: str, tail: str) -> str:
    """The error text of a name removed in ``since``."""
    return f"{old} was removed in {since}{tail}"


def renamed(old: str, new: str, fn):
    """``fn`` under the deprecated name ``old``, warning that callers should write ``new``."""
    @functools.wraps(fn)
    def alias(*args, **kwargs):
        deprecated(old, use(new))
        return fn(*args, **kwargs)
    alias.__name__ = alias.__qualname__ = old.rsplit(".", 1)[-1]
    alias.__doc__ = f"Deprecated: use ``{new}``."
    return alias


def keywords(func: str, *, renamed: dict | None = None, dropped: tuple = (),
             kept: dict | None = None, removed: dict | None = None,
             private: str | None = None):
    """Decorator for the old keywords of the public function ``func``. None of them
    is in its signature.

    renamed : ``{old: new}``
        The value goes to ``new``, with ``Use new=.``
    dropped : names
        The value is discarded, with ``; it has no effect.``
    kept : ``{old: tail}``
        The keyword keeps its effect until 0.5, with the warning ``tail``: the
        call goes to ``private``, the name of the module's implementation that
        still takes it.
    removed : ``{old: (version, tail)}``
        ``TypeError`` naming the replacement.
    """
    renamed, kept, removed = renamed or {}, kept or {}, removed or {}
    if kept and not private:
        raise TypeError("keywords(kept=...) needs private=, the implementation's name")

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for name, (since, tail) in removed.items():
                if name in kwargs:
                    raise TypeError(removed_text(f"{func}({name}=)", since, tail))
            target = fn
            for name in [n for n in kwargs if n in renamed or n in dropped or n in kept]:
                value = kwargs.pop(name)
                if name in renamed:
                    new = renamed[name]
                    if new in kwargs:
                        raise TypeError(f"{func}() got {name}= and {new}=; pass {new}= only")
                    deprecated(f"{func}({name}=)", use(f"{new}="))
                    kwargs[new] = value
                elif name in dropped:
                    deprecated(f"{func}({name}=)")
                else:
                    deprecated(f"{func}({name}=)", kept[name])
                    kwargs[name] = value
                    target = fn.__globals__[private]
            return target(*args, **kwargs)
        return wrapper
    return deco


def module_getattr(module: str, prefix: str, *, old: dict | None = None,
                   removed: dict | None = None):
    """A PEP 562 ``__getattr__`` for ``module``. The warnings spell its names
    ``prefix.name`` and the errors ``mtb.prefix.name`` (``prefix=""`` for ``mtb``).

    old : ``{name: (value, tail)}``
        Deprecated names, served with the warning ``tail``.
    removed : ``{name: (version, tail)}``
        ``AttributeError`` naming the replacement.
    """
    old, removed = old or {}, removed or {}

    def __getattr__(name):
        if name in removed:
            since, tail = removed[name]
            path = ".".join(x for x in ("mtb", prefix, name) if x)
            raise AttributeError(removed_text(path, since, tail))
        if name not in old:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        value, tail = old[name]
        deprecated(f"{prefix}.{name}", tail)
        return value
    return __getattr__
