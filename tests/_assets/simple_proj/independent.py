"""Independent module for testing purposes."""

from __future__ import annotations

INDEP_CONST = "hello"


def indep_func() -> str:
    """Example function that returns a constant string."""
    return INDEP_CONST.upper()


print(indep_func())
