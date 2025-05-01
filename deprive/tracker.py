"""Track a graph of depencencies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deprive.visitor import Definition, DepGraph, Import

if TYPE_CHECKING:
    from collections.abc import Collection


logger = logging.getLogger(__name__)


class NoDefinitionError(Exception):
    """No definition was found in the graph."""


def _recursively_track(  # noqa: C901,PLR0912
    root_name: str, input_graph: DepGraph, output_graph: DepGraph, raw_elem: Definition
) -> None:
    calls: set[Definition | Import] | None = None
    elem: Definition | None = None
    for suffix in (".__init__", ""):  # prioritise subpackages over modules
        if suffix and raw_elem.module.endswith(suffix):
            continue
        elem = Definition(f"{raw_elem.module}{suffix}", raw_elem.name)
        if elem in output_graph:
            return
        calls = input_graph.get(elem, None)
        if calls is not None:
            break
    else:
        if calls is None:  # pragma: no cover
            raise NoDefinitionError(f"No matching definition or module found for {elem}")
    if not elem:  # pragma: no cover
        raise ValueError(f"No matching definition found for {elem}")
    if elem in output_graph:  # pragma: no cover
        raise ValueError(f"Element {elem} already tracked")

    output_graph[elem] = calls

    # add parents
    parts = elem.module.split(".")
    for ix in range(len(parts), 0, -1):  # root.nested.mod, root.nested, root
        parent = ".".join(parts[:ix])
        parent_def = Definition(parent, None)
        _recursively_track(root_name, input_graph, output_graph, parent_def)

    def _run_recursion(name: str, new_def: Definition) -> None:
        if name == root_name or name.startswith(f"{root_name}."):
            _recursively_track(root_name, input_graph, output_graph, new_def)
        else:
            logger.debug("External import: %s", call)

    for call in calls:
        if isinstance(call, Definition):  # pragma: no cover
            _recursively_track(root_name, input_graph, output_graph, call)
        elif isinstance(call.name, str):
            _run_recursion(call.name, Definition(call.name, None))
        else:
            if call.asname is None:
                if isinstance(call.name, str):  # pragma: no cover
                    raise ValueError(f"Alias is required for module imports: {call}")
                try:
                    _run_recursion(call.name[0], Definition(call.name[0], call.name[1]))
                except NoDefinitionError:
                    parent, child = call.name[0].rsplit(".", 1)
                    _run_recursion(parent, Definition(parent, child))
                continue
            # TODO(tihoph): what to prioritise?
            try:
                fqn = ".".join(call.name)
                _run_recursion(fqn, Definition(fqn, None))
            except NoDefinitionError:
                _run_recursion(call.name[0], Definition(call.name[0], call.name[1]))


def track_dependencies(
    root_name: str, input_graph: DepGraph, required: Collection[str]
) -> DepGraph:
    """Track dependencies."""
    required = set(required)
    definitions: set[Definition] = set()

    for elem in required:
        if elem == root_name:
            definitions.add(Definition(f"{root_name}.__init__", None))
            continue
        if not elem.startswith(f"{root_name}."):
            raise ValueError(f"Required element must start with the module name: {elem}")
        parent, child = elem.rsplit(".", maxsplit=1)
        definition = Definition(parent, child)
        definitions.add(definition)

    output_graph: DepGraph = {}

    for definition in definitions:
        _recursively_track(root_name, input_graph, output_graph, definition)

    return output_graph
