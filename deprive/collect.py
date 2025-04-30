"""Collect dependencies for a given Python module or package."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from deprive.names import path_to_fqn
from deprive.visitor import DepGraph, ScopeVisitor

if TYPE_CHECKING:
    from os import PathLike

StrPath: TypeAlias = "str | PathLike[str]"


def collect_module(file_path: StrPath, root_dir: StrPath | None = None) -> DepGraph:
    """Parses a Python file and returns a dictionary of dependencies.

    Args:
        file_path: The path to the Python file.
        root_dir: The root directory of the project. Fully qualified names will be
            relative to this directory.

    Returns:
        A dictionary where keys are definitions for the element (fully qualified name of the module
        the name of the element). Elements can be top-level functions, classes, and constants.
        Values are depencencies of the element: either other top-level elements or necessary
        imports.
    """
    file_path = Path(file_path)
    if root_dir is not None:
        root_dir = Path(root_dir)

    fqn = path_to_fqn(file_path, root_dir)
    visitor = ScopeVisitor(fqn)
    visitor.run(file_path.read_text())
    return visitor.dep_graph


def collect_package(root_dir: StrPath) -> DepGraph:
    """Parses a Python package and returns a dictionary of dependencies."""
    # TODO(tihoph): add support for subpackages by providing a name
    root_dir = Path(root_dir)
    dep_graphs: list[DepGraph] = []

    for path in root_dir.rglob("*.py"):
        dep_graph = collect_module(path, root_dir)
        dep_graphs.append(dep_graph)
    dep_graph_all: DepGraph = {}
    for dep_graph in dep_graphs:
        if set(dep_graph) & set(dep_graph_all):  # pragma: no cover
            raise ValueError("Duplicate module names found")
        dep_graph_all.update(dep_graph)
    return dep_graph_all
