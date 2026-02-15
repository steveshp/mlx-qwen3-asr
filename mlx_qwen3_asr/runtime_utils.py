"""Runtime helper utilities for optional capability checks."""

from __future__ import annotations

import inspect


def supports_kwarg(callable_obj: object, kwarg_name: str) -> bool:
    """Return whether ``callable_obj`` accepts ``kwarg_name``.

    This is used for compatibility with test doubles or older call sites
    that may not support recently added optional keyword arguments.
    """
    if callable_obj is None:
        return False
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return kwarg_name in sig.parameters

