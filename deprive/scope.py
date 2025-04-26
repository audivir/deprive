"""Module for tracking scopes and function definitions in the AST."""

from __future__ import annotations

import ast
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deprive.names import get_node_defined_names

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


@dataclass
class Scope:
    """Represents a scope in the AST."""

    imports: dict[str, str] = field(default_factory=dict)  # alias -> module_fqn
    imports_from: dict[str, tuple[str, str]] = field(
        default_factory=dict
    )  # name -> source_item_fqn (e.g., {'os_path': 'os.path'})
    functions: dict[str | int, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda] = field(
        default_factory=dict
    )
    names: dict[str, ast.AST] = field(default_factory=dict)

    @property
    def fields(
        self,
    ) -> tuple[
        dict[str, str],
        dict[str, tuple[str, str]],
        dict[str | int, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda],
        dict[str, ast.AST],
    ]:
        """Fields of the scope."""
        return (self.imports, self.imports_from, self.functions, self.names)


class ScopeTracker:
    """Tracks function definitions and their scopes."""

    def __init__(self) -> None:
        """Initialize the function tracker."""
        self.scopes: list[Scope] = [Scope()]
        self.visited_funcs: list[int] = []  # list of ids of visited function nodes
        self.all_nodes: dict[int, ast.AST] = {}  # all nodes by object id
        self.all_scopes: dict[int, Scope] = {}  # all scopes by object id

    def is_in(self, name: str, inner_only: bool = False) -> bool:
        """Check if a name is in the current scope and if it's the outermost scope."""
        scopes = self.scopes[1:] if inner_only else self.scopes
        for scope in reversed(scopes):
            for elem in scope.fields:
                if name in elem:
                    return True
        return False

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
        """Context manager for a new scope."""
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

    def add_func(
        self, name: str | int, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    ) -> None:
        """Add a function to the current scope. Also adds to names."""
        self.scopes[-1].functions[name] = node

    def add_name(self, name: tuple[str, ...] | str | None, node: ast.AST) -> None:
        """Add a name to the current scope."""
        if not name:
            return
        if isinstance(name, str):
            name = (name,)
        for n in name:
            if self.is_import(n):
                raise NotImplementedError("Redefining imports is not supported yet.")
            self.scopes[-1].names[n] = node

    def resolve_func(self, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | None:
        """Resolve a function name to its definition."""
        for scope in reversed(self.scopes):
            if name in scope.functions:
                return scope.functions[name]
        return None

    def is_visited(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> bool:
        """Check if a function node has been visited."""
        return id(node) in self.visited_funcs

    def mark_visited(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        """Mark a function node as visited."""
        self.visited_funcs.append(id(node))

    def add_import(self, node: ast.Import | ast.ImportFrom) -> None:
        """Add an import to the current scope."""
        if isinstance(node, ast.Import):
            self._handle_import(node)
        else:
            self._handle_import_from(node)

    def _handle_import(self, node: ast.Import) -> None:
        if len(node.names) != 1:
            raise NotImplementedError(
                "Multiple imports in a single statement are not supported yet."
            )
        alias = node.names[0]
        module_name = alias.name
        alias_name = alias.asname if alias.asname else module_name
        if self.is_import(alias_name):
            raise NotImplementedError("Redefining imports is not supported yet.")
        self.scopes[-1].imports[alias_name] = module_name
        logger.debug("Found import: import %s as %s", module_name, alias_name)

    def _handle_import_from(self, node: ast.ImportFrom) -> None:
        """Handle `from module import name [as alias]` imports."""
        if node.level != 0:
            raise NotImplementedError(
                "Relative imports are not supported yet. Use absolute imports instead."
            )

        if len(node.names) == 1 and node.names[0].name == "*":
            raise NotImplementedError(
                "Star imports are not supported yet. Use explicit imports instead."
            )

        # Handle imported names
        for alias in node.names:
            original_name = alias.name
            alias_name = alias.asname if alias.asname else original_name
            if self.is_import(alias_name):
                raise NotImplementedError("Redefining imports is not supported yet.")
            if not node.module:  # pragma: no cover
                raise ValueError("Import from is missing module")
            self.scopes[-1].imports_from[alias_name] = (node.module, original_name)
