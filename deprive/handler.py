"""Handle the filtering of single modules."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, TypeVar

import libcst as cst
from libcst._nodes.base import CSTNode
from libcst._nodes.expression import Attribute, List, Name, SimpleString, Tuple
from libcst._nodes.module import Module
from libcst._nodes.op import ImportStar
from libcst._nodes.statement import (
    AnnAssign,
    Assign,
    AugAssign,
    BaseCompoundStatement,
    BaseSmallStatement,
    BaseStatement,
    Else,
    Expr,
    Finally,
    If,
    Import,
    ImportAlias,
    ImportFrom,
    SimpleStatementLine,
    Try,
)
from typing_extensions import TypeAlias

from deprive.names import get_node_defined_names

if TYPE_CHECKING:
    from collections.abc import Collection

T = TypeVar("T", bound=CSTNode)

AllTypes: TypeAlias = "BaseStatement | BaseSmallStatement | SimpleStatementLine | BaseCompoundStatement | If | Else | Try | Finally"  # noqa: E501


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


def _extract_attribute_name(attr: Attribute) -> str:
    """Extract the name of an attribute."""
    parts: list[str] = [attr.attr.value]
    while not isinstance(attr.value, Name):
        parts.append(attr.attr.value)
        if not isinstance(attr.value, Attribute):  # pragma: no cover
            raise TypeError(f"Unknown attribute type: {type(attr.value)}")
        attr = attr.value
    parts.append(attr.value.value)
    return ".".join(reversed(parts))


def _get_alias(import_alias: ImportAlias) -> str:
    """Get the alias of an import."""
    if import_alias.asname:
        name = import_alias.asname.name
        if not isinstance(name, Name):  # pragma: no cover
            raise TypeError(f"Unknown import asname type: {type(name)}")
        return name.value
    if isinstance(import_alias.name, Name):
        return import_alias.name.value
    if isinstance(import_alias.name, Attribute):
        return _extract_attribute_name(import_alias.name)
    raise TypeError(f"Unknown import alias type: {type(import_alias.name)}")


def handle_import(
    elem: AllTypes, import_elem: Import | ImportFrom, required: set[str]
) -> list[AllTypes]:
    """Handle an import statement. Return the nodes that should be kept."""
    if isinstance(import_elem.names, ImportStar):
        raise NotImplementedError("Import * is not supported.")

    if (
        isinstance(import_elem, ImportFrom)
        and import_elem.module
        and import_elem.module.value == "__future__"
    ):
        return [elem]

    kept_names = [
        alias
        for alias in import_elem.names
        if _get_alias(alias) in required | {"__future__"} or alias.name.value == "__future__"
    ]
    if not kept_names:
        return []

    if len(kept_names) == len(import_elem.names):  # no changes
        return [elem]

    # Remove trailing comma
    kept_names[-1] = kept_names[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
    new_import_from = import_elem.with_changes(names=kept_names)
    new_elem: AllTypes = elem.deep_replace(import_elem, new_import_from)
    return [new_elem]


def _filter_all(
    node: AllTypes,
    elem: BaseCompoundStatement | Assign | AnnAssign | AugAssign,
    required: set[str],
    keep: set[str],
) -> list[AllTypes] | None:
    if isinstance(elem, BaseCompoundStatement):
        # not an assignment
        return None
    if isinstance(elem, Assign) and len(elem.targets) == 1:
        # multiple targets
        single_target = elem.targets[0].target
    elif isinstance(elem, AnnAssign):
        single_target = elem.target
    else:
        return None
    if not isinstance(single_target, Name) or single_target.value != "__all__":
        # assignment not named __all__ or assignment to subscript, attribute, etc.
        return None

    if elem.value is None:
        raise ValueError("No value for __all__ assignment.")

    if not isinstance(elem.value, (Tuple, List)) or not all(
        isinstance(child.value, SimpleString) for child in elem.value.elements
    ):
        raise TypeError(f"Expected a tuple or list of literal strings, got {elem.value}")

    kept_elements = [
        child
        for child in elem.value.elements
        if ast.literal_eval(child.value.value) in keep | required  # type: ignore[attr-defined]
    ]

    if not kept_elements:  # no elements to keep remove __all__ assignment
        return []

    if len(kept_elements) == len(elem.value.elements):  # no changes
        return [node]

    kept_elements[-1] = kept_elements[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
    new_list = elem.value.with_changes(elements=kept_elements)
    new_node: AllTypes = node.deep_replace(elem.value, new_list)  # type: ignore[arg-type]
    return [new_node]


def handle_elem(
    node: AllTypes,
    elem: BaseCompoundStatement | Assign | AnnAssign | AugAssign,
    required: set[str],
    keep: set[str],
) -> list[AllTypes]:
    """Handle an a definition. Return the nodes that should be kept."""
    all_elem = _filter_all(node, elem, required, keep)
    if all_elem is not None:
        return all_elem

    names = get_names(elem)

    if "__all__" in names:
        raise ValueError("Unsupported __all__ definition.")

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


def _handle_try_finally(
    elem: Try | Finally, required: set[str], keep: set[str]
) -> list[Try | Finally]:
    new_body: list[AllTypes] = []
    for stmt in elem.body.body:
        _handle_body_elem(stmt, new_body, required, keep)
    elem_body = elem.body.with_changes(body=new_body)
    elem = elem.with_changes(body=elem_body)
    if isinstance(elem, Try):
        if elem.orelse:
            new_orelse_list = _handle_if_else(elem.orelse, required, keep)
            new_orelse = new_orelse_list[0] if new_orelse_list else None
            elem = elem.with_changes(orelse=new_orelse)
        if elem.finalbody:
            new_finalbody_list = _handle_try_finally(elem.finalbody, required, keep)
            new_finalbody = new_finalbody_list[0] if new_finalbody_list else None
            elem = elem.with_changes(finalbody=new_finalbody)
        return [] if not new_body and not elem.orelse and not elem.finalbody else [elem]
    return [] if not new_body else [elem]


def _handle_body_elem(
    elem: AllTypes, output: list[AllTypes], required: set[str], keep: set[str]
) -> None:
    if isinstance(elem, (If, Else)):
        output.extend(_handle_if_else(elem, required, keep))
    elif isinstance(elem, (Try, Finally)):
        output.extend(_handle_try_finally(elem, required, keep))
    elif isinstance(elem, BaseCompoundStatement):
        output.extend(handle_elem(elem, elem, required, keep))
    elif import_elem := get_node(elem, Import) or get_node(elem, ImportFrom):
        output.extend(handle_import(elem, import_elem, required))
    elif (
        assign_elem := get_node(elem, Assign)
        or get_node(elem, AnnAssign)
        or get_node(elem, AugAssign)
    ):
        output.extend(handle_elem(elem, assign_elem, required, keep))
    elif get_node(elem, Expr):  # Add this condition to check for expressions
        output.append(elem)  # Add the expression to the output
    else:  # pragma: no cover
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
