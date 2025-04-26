"""Utility functions and constants for the sample project."""

from __future__ import annotations

CONST = 123


class HelperClass:
    def __init__(self, name: str) -> None:
        """Initializes the HelperClass with a name."""
        self.name = name

    def greet(self) -> str:
        """Greets the user with the name."""
        return f"Hello, {self.name}"


def helper_func(x: int) -> int:
    """Example helper function that doubles the input."""
    return x * 2


# Not intended for export
_internal_var = "secret"


def _internal_func() -> None:
    """Internal function that does nothing."""
