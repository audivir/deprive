"""Handle the filtering of single modules."""

from __future__ import annotations

import ast
from typing import TypeVar

import libcst as cst
from libcst._nodes.base import CSTNode
from libcst._nodes.module import Module
from libcst._nodes.op import ImportStar
from libcst._nodes.statement import (
    Assign,
    BaseCompoundStatement,
    Expr,
    Import,
    ImportAlias,
    ImportFrom,
    SimpleStatementLine,
)

from deprive.names import get_node_defined_names

T = TypeVar("T", bound=CSTNode)


def to_ast(elem: CSTNode) -> ast.AST:
    """Convert a CSTNode to its source code representation."""
    code: str = Module([]).code_for_node(elem)
    return ast.parse(code).body[0]


def get_names(elem: CSTNode) -> tuple[str, ...]:
    """Get all names defined by an element."""
    node = to_ast(elem)
    name = get_node_defined_names(node)
    if not name:
        raise ValueError("No names found in element")
    if isinstance(name, str):
        name = (name,)
    return name


def get_node(elem: SimpleStatementLine | None, typ: type[T]) -> T | None:
    """Get the first child of a CSTNode that is not a comment or whitespace."""
    if not elem:
        return None
    if len(elem.body) != 1:
        raise NotImplementedError("Multiple statements per line are not supported.")
    node = elem.body[0]
    if isinstance(node, typ):
        return node
    return None


def _get_alias(import_alias: ImportAlias) -> str:
    """Get the alias of an import."""
    name = import_alias.asname.name.value if import_alias.asname else import_alias.name.value  # type: ignore[union-attr]
    if not isinstance(name, str):
        raise TypeError(f"Expected str for alias, got {type(name)}")
    return name


def handle_import(
    elem: SimpleStatementLine, import_elem: Import | ImportFrom, required: set[str]
) -> list[SimpleStatementLine]:
    """Handle an import statement. Return the nodes that should be kept."""
    if isinstance(import_elem, Import):
        if len(import_elem.names) != 1:
            raise NotImplementedError(
                "Multiple imports in a single import statement are not supported."
            )
        if _get_alias(import_elem.names[0]) in required:
            return [elem]
        return []

    # keep only imports where the alias is in required
    if isinstance(import_elem.names, ImportStar):
        raise NotImplementedError("Import * is not supported.")

    kept_names = [alias for alias in import_elem.names if _get_alias(alias) in required]
    if not kept_names:
        return []
    # Remove trailing comma
    kept_names[-1] = kept_names[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
    new_import_from = import_elem.with_changes(names=kept_names)
    new_elem: SimpleStatementLine = elem.deep_replace(import_elem, new_import_from)  # type: ignore[arg-type]
    return [new_elem]


def handle_elem(
    node: SimpleStatementLine | BaseCompoundStatement,
    elem: BaseCompoundStatement | Assign,
    keep: set[str],
) -> list[SimpleStatementLine | BaseCompoundStatement]:
    """Handle an a definition. Return the nodes that should be kept."""
    names = get_names(elem)
    if any(name in keep for name in names):
        return [node]
    return []


def handle_module(code: str, required: set[str], keep: set[str]) -> str:
    """Filter a module to keep only specified definitions and imports.

    Top-level expressions are always kept.

    Args:
        code: The code of the module to filter.
        required: The alias names of the imports that are required.
        keep: The names of the definitions that should be kept.

    Returns:
        The filtered code.
    """
    module = cst.parse_module(code)
    output: list[SimpleStatementLine | BaseCompoundStatement] = []

    for elem in module.body:
        if isinstance(elem, BaseCompoundStatement):
            output.extend(handle_elem(elem, elem, keep))
        elif import_elem := get_node(elem, Import) or get_node(elem, ImportFrom):
            output.extend(handle_import(elem, import_elem, required))
        elif assign_elem := get_node(elem, Assign):
            output.extend(handle_elem(elem, assign_elem, keep))
        elif get_node(elem, Expr):  # Add this condition to check for expressions
            output.append(elem)  # Add the expression to the output
        else:
            raise ValueError(f"Unexpected element: {elem}")

    # build the new module with only kept nodes
    new_module = module.with_changes(body=output)
    return new_module.code
