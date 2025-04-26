"""Nested package module."""

from __future__ import annotations

from simple_proj.utils import helper_func  # Relative import


def nested_func(y: int) -> int:
    """Example function that uses a helper function from the parent package."""
    return helper_func(y + 1)
