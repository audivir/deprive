"""Collect dependencies for a given Python module or package."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import TypeAlias

from deprive.names import path_to_fqn
from deprive.visitor import DepGraph, ScopeVisitor

if TYPE_CHECKING:
    from collections.abc import Collection
    from os import PathLike

StrPath: TypeAlias = "str | PathLike[str]"


def _split_fqn(fqn: str) -> tuple[str, str | None]:
    """Splits a fully qualified name into its module and element parts."""
    if "." in fqn:
        return tuple(fqn.rsplit(".", maxsplit=1))  # type: ignore[return-value]
    return fqn, None


def collect_module(
    file_path: StrPath, root_dir: StrPath | None = None, additional: Collection[str] | None = None
) -> DepGraph:
    """Parses a Python file and returns a dictionary of dependencies.

    Args:
        file_path: The path to the Python file.
        root_dir: The root directory of the project. Fully qualified names will be
            relative to this directory.
        additional: Additional top-level dependencies to include.
            Either as the top-level name or including the fully qualified name
            of the module.

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
    additional = set(additional or [])
    for elem in additional:
        visitor.add(elem)
    return visitor.dep_graph


def collect_package(root_dir: StrPath, additional: Collection[str] | None = None) -> DepGraph:
    """Parses a Python package and returns a dictionary of dependencies."""
    # TODO(tihoph): add support for subpackages by providing a name
    root_dir = Path(root_dir)
    dep_graphs: list[DepGraph] = []
    split_additional: set[tuple[str, str | None]] = {_split_fqn(fqn) for fqn in additional or []}

    for path in root_dir.rglob("*.py"):
        fqn = path_to_fqn(path, root_dir)
        curr_additional: list[str] = []
        for parent, child in split_additional:
            add_fqn = f"{parent}.{child}"
            if fqn in (add_fqn, parent):
                curr_additional.append(add_fqn)
        dep_graph = collect_module(path, root_dir, additional=curr_additional)
        dep_graphs.append(dep_graph)
    dep_graph_all: DepGraph = {}
    for dep_graph in dep_graphs:
        if set(dep_graph) & set(dep_graph_all):  # pragma: no cover
            raise ValueError("Duplicate module names found")
        dep_graph_all.update(dep_graph)
    return dep_graph_all
