"""Track a graph of depencencies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deprive.visitor import Definition, DepGraph, Import

if TYPE_CHECKING:
    from collections.abc import Collection


logger = logging.getLogger(__name__)


def _recursively_track(  # noqa: C901,PLR0912
    root_name: str, input_graph: DepGraph, output_graph: DepGraph, elem: Definition
) -> None:
    if elem in output_graph:
        return
    calls = input_graph.get(elem, None)
    if calls is None:
        module_def = Definition(elem.module, None)
        subdefs = input_graph.get(module_def, None)
        if subdefs is None:
            raise ValueError(f"No matching definition or module found for {elem}")
        calls = {x for x in subdefs if isinstance(x, Import) and x.asname == elem.name}
        if not calls:
            raise ValueError(f"No matching definition found for {elem}")
        if len(calls) > 1:  # pragma: no cover
            raise ValueError(f"Multiple matching definitions found for {elem}: {calls}")

    if elem in output_graph:  # pragma: no cover
        raise ValueError(f"Element {elem} already tracked")

    output_graph[elem] = calls

    # add parents
    parts = elem.module.split(".")
    for ix in range(len(parts), 0, -1):  # root.nested.mod, root.nested, root
        parent = ".".join(parts[:ix])
        parent_def = Definition(parent, None)
        _recursively_track(root_name, input_graph, output_graph, parent_def)

    for call in calls:
        if isinstance(call, Definition):  # pragma: no cover
            _recursively_track(root_name, input_graph, output_graph, call)
        else:
            if isinstance(call.name, str):
                name = call.name
                new_def = Definition(name, None)
            else:
                name = call.name[0]
                # TODO(tihoph): unsafe if call.name[1] is a module
                new_def = Definition(name, call.name[1])
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
