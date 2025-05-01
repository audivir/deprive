"""Handles FQNs and paths for modules and classes and assignments."""

# ruff: noqa: ERA001
from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def path_to_fqn(path: Path, root_dir: Path | None) -> str:
    """Converts a file path relative to root_dir into an FQN."""
    if not path.is_absolute():
        path = path.resolve()

    if not root_dir:
        return path.stem

    if not root_dir.is_absolute():
        root_dir = root_dir.resolve()

    try:
        relative_path = path.relative_to(root_dir)
    except ValueError:
        logger.exception("Path %s is not inside root directory %s", path, root_dir)
        raise  # Re-raise for caller to handle

    parts = list(relative_path.parts)
    if not parts:  # pragma: no cover # Should not happen if path is relative to root_dir
        raise ValueError("Empty path parts for %s relative to %s", path, root_dir)

    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]  # Replace .py extension with module name
    else:
        logger.warning("Path %s does not end with .py or __init__.py", path)
        # Assuming it's a directory reference? This case needs clarification.
        # For now, treat it as a package name based on directory path
        # Keep the directory structure

    return root_dir.name + "." + ".".join(parts)


def _unpack_target(node: ast.expr) -> list[str]:
    """Recursively unpack assignment targets (Names, Tuples, Lists) into a list of names."""
    names: list[str] = []
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, ast.Starred):
        if isinstance(node.value, ast.Name):
            names.append(node.value.id)
        else:  # pragma: no cover
            raise TypeError(f"Invalid starred target type: {type(node.value)}")
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.extend(_unpack_target(elt))
    elif isinstance(node, (ast.Attribute, ast.Subscript)):
        pass
    else:  # pragma: no cover
        raise TypeError(f"Invalid assignment target element type: {type(node).__name__}")
    return names


def get_node_defined_names(  # noqa: C901,PLR0911
    node: ast.AST, strict: bool = True
) -> tuple[str, ...] | str | None:
    """Safely get the name(s) defined by an AST node.

    Args:
        node: The AST node to extract names from.
        strict: If True, raises a warning for unsupported node types.

    Returns:
        Either a single name (str), a tuple of names (tuple[str]), or None if no name is found.
    """
    names: list[str] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return tuple(node.names)
    if isinstance(node, ast.Assign):
        # Handle assignments like MY_CONST = ..., x, y = ...
        for target in node.targets:
            names.extend(_unpack_target(target))
        if len(names) == 0:
            return None
        if len(names) == 1:
            return names[0]
        return tuple(names)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        # Handle MY_CONST: int = ... (target can only be Name, Attribute, Subscript)
        if isinstance(node.target, (ast.Attribute, ast.Subscript)):
            return None
        if isinstance(node.target, ast.Name):
            return node.target.id
        raise TypeError(  # pragma: no cover
            f"Invalid AnnAssign/AugAssign target type: {type(node.target).__name__} (expected Name)"
        )
    if hasattr(node, "custom_name") and isinstance(node.custom_name, str):
        # Handle custom name attributes (e.g., in AST transformations)
        return node.custom_name
    # TODO(tihoph): Could potentially handle other definition types like TypeAlias here if needed
    # if isinstance(node, ast.TypeAlias):
    #     if isinstance(node.name, ast.Name):
    #          return (node.name.id,)

    if strict:  # pragma: no cover
        logger.warning("Cannot extract defined name from node type: %s", type(node).__name__)
    return None  # Return None if no name could be extracted


def get_attribute_parts(node: ast.Attribute) -> list[str]:
    """Get the parts of an attribute chain as a string."""
    parts: list[str] = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value

    # e. g (test / test).test, test().test, test[0].test, 'test'.test
    if not isinstance(value, ast.Name):  # pragma: no cover
        if not isinstance(value, (ast.BinOp, ast.Call, ast.Subscript, ast.Constant)):
            logger.warning("Not a Name node in attribute chain: %s", type(value).__name__)
        return []
    parts.append(value.id)
    return list(reversed(parts))
