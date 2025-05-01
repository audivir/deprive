"""Modifies modules to remove unused imports and definitions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from deprive.collect import StrPath, collect_package
from deprive.handler import handle_module
from deprive.tracker import track_dependencies
from deprive.visitor import Definition, DepGraph, Import

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)


# TODO(tihoph): split this into multiple functions.
def _modify_module(
    root_dir: Path, root_name: str, tracked: DepGraph, output: Path, module_def: Definition
) -> None:
    """Modify a single module."""
    calls: set[Definition | Import] = set()
    for k, v in tracked.items():
        # add the element itself
        if k.name:
            calls.add(k)
        # add the imports necessary for the element
        if k.module == module_def.module:
            calls.update(v)

    # remove the root name from the module name
    path_parts = module_def.module[len(root_name) + 1 :].split(".")

    rel_path = Path(*path_parts)

    full_path = root_dir / rel_path

    # append suffix to the path to get the file name
    rel_path = rel_path.with_suffix(".py")
    full_path = full_path.with_suffix(".py")

    code = full_path.read_text()

    required_imports: set[str] = set()
    keep_definitions: set[str] = set()
    for call in calls:
        if isinstance(call, Import):
            if not call.asname:
                continue
            required_imports.add(call.asname)
        elif call.module == module_def.module and call.name:
            keep_definitions.add(call.name)

    new_code = handle_module(code, required_imports, keep_definitions)
    if not new_code and rel_path.name != "__init__.py":
        logger.debug("Skipping empty module %s", rel_path)
        return
    # create the new directory
    parent = output / rel_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # write the modified code to the temporary directory
    (parent / rel_path.name).write_text(new_code or "")


def modify_package(root_dir: StrPath, required: Collection[str], output: StrPath) -> None:
    """Modifies a package to only include required dependencies."""
    root_dir = Path(root_dir)
    root_name = root_dir.stem
    output = Path(output)
    if not output.is_dir():
        raise ValueError(f"Output {output} must be a directory")

    graph = collect_package(root_dir)
    tracked = track_dependencies(root_name, graph, required)

    # select only the modules
    modules = {k for k in tracked if k.name is None}

    for module_def in modules:
        _modify_module(root_dir, root_name, tracked, output, module_def)
