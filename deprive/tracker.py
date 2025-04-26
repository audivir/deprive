"""Track a graph of depencencies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deprive.visitor import Definition, DepGraph

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)


def _recursively_track(
    root_name: str, input_graph: DepGraph, output_graph: DepGraph, elem: Definition
) -> None:
    if elem in output_graph:
        return
    calls = input_graph.get(elem, None)
    if calls is None:
        raise ValueError(f"Required element {elem} not found in the module")

    output_graph[elem] = calls

    # add parents
    parts = elem.module.split(".")
    for ix in range(len(parts), 0, -1):  # root.nested.mod, root.nested, root
        parent = ".".join(parts[:ix])
        parent_def = Definition(parent, None)
        _recursively_track(root_name, input_graph, output_graph, parent_def)
        if parent == root_name:
            break

    for call in calls:
        if isinstance(call, Definition):
            _recursively_track(root_name, input_graph, output_graph, call)
        else:
            if isinstance(call.name, str):
                name = call.name
                new_def = Definition(name, None)
            else:
                name = call.name[0]
                new_def = Definition(
                    name, call.name[1]
                )  # TODO(tihoph): unsafe if call.name[1] is a module
            if name == root_name or name.startswith(f"{root_name}."):
                _recursively_track(root_name, input_graph, output_graph, new_def)
            else:
                logger.debug("External import: %s", call)


def track_dependencies(
    root_name: str, input_graph: DepGraph, required: Collection[str]
) -> DepGraph:
    """Track dependencies."""
    required = set(required)
    definitions: set[Definition] = set()

    for elem in required:
        if not elem.startswith(f"{root_name}."):
            raise ValueError("Required element must start with the module name")
        parent, child = elem.rsplit(".", maxsplit=1)
        definition = Definition(parent, child)
        definitions.add(definition)

    output_graph: DepGraph = {}

    for definition in definitions:
        _recursively_track(root_name, input_graph, output_graph, definition)

    return output_graph
