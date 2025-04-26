"""Filter a Python codebase keeping only specified definitions and their dependencies.

Public API classes/functions for the deprive package.

Only classes and functions defined here are considered public API.
All other code is internal implementation details.
"""

from __future__ import annotations

from deprive.collect import collect_module, collect_package
from deprive.handler import handle_module
from deprive.modifier import modify_package
from deprive.tracker import track_dependencies
from deprive.visitor import Definition, DepGraph, Import

__version__ = "0.1.1"
__all__ = [
    "Definition",
    "DepGraph",
    "Import",
    "collect_module",
    "collect_package",
    "handle_module",
    "modify_package",
    "track_dependencies",
]
