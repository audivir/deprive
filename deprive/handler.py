"""Handle the filtering of single modules."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, TypeVar

import libcst as cst
from libcst._nodes.base import CSTNode
from libcst._nodes.module import Module
from libcst._nodes.op import ImportStar
from libcst._nodes.statement import (
    AnnAssign,
    Assign,
    BaseCompoundStatement,
    BaseSmallStatement,
    BaseStatement,
    Else,
    Expr,
    If,
    Import,
    ImportAlias,
    ImportFrom,
    SimpleStatementLine,
)
from typing_extensions import TypeAlias

from deprive.names import get_node_defined_names

if TYPE_CHECKING:
    from collections.abc import Collection

T = TypeVar("T", bound=CSTNode)

AllTypes: TypeAlias = (
    "BaseStatement | BaseSmallStatement | SimpleStatementLine | BaseCompoundStatement | If | Else"
)


def to_ast(elem: CSTNode) -> ast.AST:
    """Convert a CSTNode to its source code representation."""
    code: str = Module([]).code_for_node(elem)
    return ast.parse(code).body[0]


def get_names(elem: CSTNode) -> tuple[str, ...]:
    """Get all names defined by an element."""
    node = to_ast(elem)
    name = get_node_defined_names(node)
    if not name:  # pragma: no cover
        raise ValueError(f"No names found in element: {elem}")
    if isinstance(name, str):
        name = (name,)
    return name


def get_node(
    elem: BaseStatement | BaseSmallStatement | SimpleStatementLine | None, typ: type[T]
) -> T | None:
    """Get the first child of a CSTNode that is not a comment or whitespace."""
    if not elem:
        return None
    if isinstance(elem, SimpleStatementLine):
        if len(elem.body) != 1:
            raise NotImplementedError("Multiple statements per line are not supported.")
        elem = elem.body[0]
    if isinstance(elem, typ):
        return elem
    return None


def _get_alias(import_alias: ImportAlias) -> str:
    """Get the alias of an import."""
    name = import_alias.asname.name.value if import_alias.asname else import_alias.name.value  # type: ignore[union-attr]
    if not isinstance(name, str):
        raise TypeError(f"Expected str for alias, got {type(name)}")
    return name


def handle_import(
    elem: AllTypes, import_elem: Import | ImportFrom, required: set[str]
) -> list[AllTypes]:
    """Handle an import statement. Return the nodes that should be kept."""
    if isinstance(import_elem, Import):
        if len(import_elem.names) != 1:
            raise NotImplementedError(
                "Multiple imports in a single import statement are not supported."
            )
        if _get_alias(import_elem.names[0]) in required | {"__future__"}:
            return [elem]
        return []

    # keep only imports where the alias is in required
    if isinstance(import_elem.names, ImportStar):
        raise NotImplementedError("Import * is not supported.")

    if import_elem.module is None:
        raise ValueError("ImportFrom without a module is not supported.")

    if import_elem.module.value == "__future__":
        return [elem]

    kept_names = [alias for alias in import_elem.names if _get_alias(alias) in required]
    if not kept_names:
        return []

    # Remove trailing comma
    kept_names[-1] = kept_names[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
    new_import_from = import_elem.with_changes(names=kept_names)
    new_elem: AllTypes = elem.deep_replace(import_elem, new_import_from)
    return [new_elem]


def handle_elem(
    node: AllTypes, elem: BaseCompoundStatement | Assign | AnnAssign, keep: set[str]
) -> list[AllTypes]:
    """Handle an a definition. Return the nodes that should be kept."""
    names = get_names(elem)
    if any(name in keep for name in names):
        return [node]
    return []


def _handle_if_else(elem: If | Else, required: set[str], keep: set[str]) -> list[If | Else]:
    new_body: list[AllTypes] = []
    for stmt in elem.body.body:
        _handle_body_elem(stmt, new_body, required, keep)
    elem_body = elem.body.with_changes(body=new_body)
    elem = elem.with_changes(body=elem_body)
    if isinstance(elem, If):
        if elem.orelse:
            new_orelse_list = _handle_if_else(elem.orelse, required, keep)
            new_orelse = new_orelse_list[0] if new_orelse_list else None
            elem = elem.with_changes(orelse=new_orelse)
        return [] if not new_body and not elem.orelse else [elem]
    return [] if not new_body else [elem]


def _handle_body_elem(
    elem: AllTypes, output: list[AllTypes], required: set[str], keep: set[str]
) -> None:
    if isinstance(elem, (If, Else)):
        output.extend(_handle_if_else(elem, required, keep))
    elif isinstance(elem, BaseCompoundStatement):
        output.extend(handle_elem(elem, elem, keep))
    elif import_elem := get_node(elem, Import) or get_node(elem, ImportFrom):
        output.extend(handle_import(elem, import_elem, required))
    elif assign_elem := get_node(elem, Assign) or get_node(elem, AnnAssign):
        output.extend(handle_elem(elem, assign_elem, keep))
    elif get_node(elem, Expr):  # Add this condition to check for expressions
        output.append(elem)  # Add the expression to the output
    else:
        raise ValueError(f"Unexpected element: {elem}")


def handle_module(code: str, required: Collection[str], keep: Collection[str]) -> str | None:
    """Filter a module to keep only specified definitions and imports.

    Top-level expressions are always kept.

    Args:
        code: The code of the module to filter.
        required: The alias names of the imports that are required.
        keep: The names of the definitions that should be kept.

    Returns:
        The filtered code or None if no code is left after filtering.
    """
    required = set(required)
    keep = set(keep)

    module = cst.parse_module(code)
    output: list[AllTypes] = []

    for elem in module.body:
        _handle_body_elem(elem, output, required, keep)

    # build the new module with only kept nodes
    new_module = module.with_changes(body=output)
    if new_module.code.strip():
        return new_module.code
    return None
