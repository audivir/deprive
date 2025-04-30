"""Module for tracking scopes and function definitions in the AST."""

from __future__ import annotations

import ast
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from deprive.names import get_node_defined_names

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

FuncType: TypeAlias = "ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda"


@dataclass
class Scope:
    """Represents a scope in the AST."""

    imports: dict[str, str] = field(default_factory=dict)  # alias -> module_fqn
    imports_from: dict[str, tuple[str, str]] = field(
        default_factory=dict
    )  # name -> source_item_fqn (e.g., {'os_path': 'os.path'})
    functions: dict[str | int, FuncType] = field(default_factory=dict)
    names: dict[str, ast.AST | None] = field(default_factory=dict)
    global_names: set[str] = field(default_factory=set)
    nonlocal_names: set[str] = field(default_factory=set)

    @property
    def fields(
        self,
    ) -> tuple[
        dict[str, str],
        dict[str, tuple[str, str]],
        dict[str | int, FuncType],
        dict[str, ast.AST | None],
    ]:
        """Fields of the scope."""
        return (self.imports, self.imports_from, self.functions, self.names)


class ScopeTracker:
    """Tracks function definitions and their scopes."""

    def __init__(self) -> None:
        """Initialize the function tracker."""
        self.scopes: list[Scope] = [Scope()]  # initialize the global scope
        self.visited_funcs: list[int] = []  # list of ids of visited function nodes
        self.all_nodes: dict[int, ast.AST] = {}  # all nodes by object id
        self.all_scopes: dict[int, Scope] = {}  # all scopes by object id

    def is_in(self, name: str, inner_only: bool = False) -> bool:
        """Check if a name is any scope or only the inner ones."""
        scopes = self.scopes[1:] if inner_only else self.scopes
        for scope in reversed(scopes):
            for elem in scope.fields:
                if name in elem:
                    return True
        return False

    def is_local(self, name: str) -> bool:
        """Check if a name is locally defined and not with a global keyword."""
        return name not in self.current_scope.global_names and self.is_in(name, inner_only=True)

    def is_import(self, name: str, outer_only: bool = False) -> tuple[str, str] | str | None:
        """Check if a name is an import."""
        scopes = self.scopes if not outer_only else self.scopes[:1]
        for scope in reversed(scopes):
            if found_import := scope.imports.get(name):
                return found_import
            if found_import_from := scope.imports_from.get(name):
                return found_import_from
        return None

    def build_fqn(self, node: ast.AST) -> str | None:
        """Build a fully qualified name (FQN) for the given node."""
        parts: list[str] = []
        parent = node
        while not isinstance(parent, ast.Module):
            name = get_node_defined_names(parent, strict=False)
            if isinstance(name, tuple):  # pragma: no cover
                raise TypeError("Multi-name nodes should not occur during FQN building.")
            if not name:  # comprehensions, lambdas, etc.
                name = f"<{id(parent)}>"
            parts.append(name)
            parent = parent.parent  # type: ignore[attr-defined]
        parts.append(parent.custom_name)  # type: ignore[attr-defined] # add module name

        return ".".join(reversed(parts))

    @contextmanager
    def scope(self, node: ast.AST) -> Generator[None]:
        """Context manager for a new scope. Runs callback before closing scope."""
        self.push(node)
        try:
            yield
        finally:
            self.pop()

    def push(self, node: ast.AST) -> None:
        """Push a new scope onto the stack."""
        new_scope = Scope()
        if id(node) in self.all_scopes:
            raise ValueError(
                f"Scope for node {node} already exists. This should not happen."
                f" ({ast.unparse(node)})"
            )
        self.all_nodes[id(node)] = node
        self.all_scopes[id(node)] = new_scope
        self.scopes.append(new_scope)

    def pop(self) -> None:
        """Pop the current scope off the stack."""
        self.scopes.pop()

    def add_func(self, name: str | int, node: FuncType) -> None:
        """Add a function to the current scope. Also adds to names."""
        self.current_scope.functions[name] = node

    def add_name(self, name: tuple[str, ...] | str | None, node: ast.AST) -> None:
        """Add a name to the current scope."""
        if not name:
            return
        if isinstance(name, str):
            name = (name,)
        for n in name:
            if self.is_import(n):
                raise NotImplementedError("Redefining imports is not supported yet.")
            self.current_scope.names[n] = node

    def resolve_func(self, name: str) -> FuncType | None:
        """Resolve a function name to its definition."""
        for scope in reversed(self.scopes):
            if name in scope.functions:
                return scope.functions[name]
        return None

    def is_visited(self, node: FuncType) -> bool:
        """Check if a function node has been visited."""
        return id(node) in self.visited_funcs

    def mark_visited(self, node: FuncType) -> None:
        """Mark a function node as visited."""
        self.visited_funcs.append(id(node))

    def add_import(self, node: ast.alias, module: str | None) -> None:
        """Add an import to the current scope."""
        alias_name = node.asname or node.name
        if self.is_import(alias_name):
            raise NotImplementedError("Redefining imports is not supported yet.")
        if module:
            self.current_scope.imports_from[alias_name] = (module, node.name)
            logger.debug("Found import: from %s import %s as %s", module, node.name, alias_name)
        else:
            self.current_scope.imports[alias_name] = node.name
            logger.debug("Found import: import %s as %s", node.name, alias_name)

    def add_global(self, node: ast.Global) -> None:
        """Add global variables to tracker."""
        if len(self.scopes) == 1:
            raise ValueError("Global keyword in outer scope redundant")
        if set(node.names) & self.current_scope.nonlocal_names:
            raise ValueError("Global and nonlocal are mutually exclusive")
        self.current_scope.global_names.update(node.names)

    def add_nonlocal(self, node: ast.Nonlocal) -> None:
        """Add nonlocal variables to tracker."""
        if len(self.scopes) < 3:  # noqa: PLR2004
            raise ValueError("Nonlocal keyword must be used in nested scope")
        if set(node.names) & self.current_scope.global_names:
            raise ValueError("Global and nonlocal are mutually exclusive")
        self.current_scope.nonlocal_names.update(node.names)

    @property
    def current_scope(self) -> Scope:
        """Current scope."""
        return self.scopes[-1]


def add_parents(root: ast.AST) -> None:
    """Add parent attribute to all nodes in the tree."""
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            if not hasattr(child, "parent"):
                child.parent = node  # type: ignore[attr-defined]
