"""Visitor for analyzing and processing AST nodes."""

# ruff: noqa: N802
from __future__ import annotations

import ast
import builtins
import logging
import sys
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

if sys.version_info >= (3, 12):  # pragma: no cover
    from typing import TypeAlias, override
else:
    from typing_extensions import TypeAlias, override

from deprive.names import get_attribute_parts, get_node_defined_names
from deprive.scope import FuncType, ScopeTracker, add_parents

logger = logging.getLogger(__name__)

BUILTINS = frozenset(dir(builtins))

DepGraph: TypeAlias = "dict[Definition, set[Definition | Import]]"


@dataclass(frozen=True)
class Import:
    """Data class for representing an import statement."""

    name: tuple[str, str] | str = field(hash=True)
    asname: str = field(default="0", hash=True)  # 0 is no valid identifier so it can default value

    def __post_init__(self) -> None:
        if self.asname == "0":
            name = self.name if isinstance(self.name, str) else self.name[1]
            object.__setattr__(self, "asname", name)  # fix around frozen dataclass


@dataclass(frozen=True)
class Definition:
    """Data class for representing a definition."""

    module: str = field(hash=True)
    name: str | None = field(hash=True)


def get_args(node: FuncType) -> list[ast.arg]:
    """Get all arguments of a function node."""
    all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
    if node.args.vararg:
        all_args.append(node.args.vararg)
    if node.args.kwarg:
        all_args.append(node.args.kwarg)
    return all_args


class ScopeVisitor(ast.NodeVisitor):
    """Visitor that tracks function definitions and their scopes."""

    def __init__(self, fqn: str, debug: bool = False) -> None:
        """Initialize the visitor."""
        self.module_fqn = fqn

        self.tracker = ScopeTracker()

        self.deferred: deque[FunctionBodyWrapper] = deque()

        self.parent: ast.AST | FunctionBodyWrapper | None = None
        self.dep_graph: DepGraph = {}  # Dependency graph of function dependencies

        self._visited_nodes: list[ast.AST | FunctionBodyWrapper | str] = []
        self.debug = debug

        self.all: list[str] | None = None

    def run(self, code: str) -> None:
        """Run the visitor on a given code string."""
        tree = ast.parse(code)
        tree.custom_name = self.module_fqn  # type: ignore[attr-defined]
        add_parents(tree)
        self.visit(tree)
        self.visit_deferred()
        # verify result and add all outer scope names to the dependency graph
        outer_scope = self.tracker.scopes[0]
        top_level_names = set(outer_scope.names)
        top_level_names |= {x for x in outer_scope.functions if isinstance(x, str)}  # skip lambdas
        top_level_defs = {Definition(self.module_fqn, name) for name in top_level_names}
        top_level_defs |= {Definition(self.module_fqn, None)}
        if unknown_names := set(self.dep_graph) - top_level_defs:  # pragma: no cover
            raise ValueError(f"Unknown names in dependency graph: {unknown_names}")
        for name in top_level_defs:
            if name not in self.dep_graph:
                self.dep_graph[name] = set()

    @override
    def visit(self, node: ast.AST) -> None:
        """Visit a node. If the node is a function body wrapper, visit its body."""
        if self.debug:  # pragma: no cover
            self._visited_nodes.append(node)
        super().visit(node)

    @override
    def visit_Global(self, node: ast.Global) -> None:
        """Handle global statements."""
        # if a variable was used before its global/nonlocal use, a syntaxerror is raised on runtime
        # but ast can parse
        self.tracker.add_global(node)
        self.generic_visit(node)  # Continue traversal

    @override
    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Handle nonlocal statements."""
        # add it to scope one above but not the global scope
        self.tracker.add_nonlocal(node)
        self.generic_visit(node)  # Continue traversal

    @override
    def visit_Import(self, node: ast.Import) -> None:
        """Stores `import module [as alias]`."""
        for alias in node.names:
            self.tracker.add_import(alias, None)
        self.generic_visit(node)  # Continue traversal

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Stores `from module import name [as alias]` including relative imports."""
        if len(node.names) == 1 and node.names[0].name == "*":
            raise ValueError("Star imports are not supported. Use explicit imports instead.")
        if node.level != 0:
            parts = self.module_fqn.split(".")
            if node.level >= len(parts):
                raise ValueError("Relative import is deeper than module FQN.")
            parts = parts[: -node.level]
            module = ".".join(parts)
            if node.module:
                module += f".{node.module}"
        else:
            if not node.module:  # pragma: no cover
                raise ValueError("No module specified for absolute import.")
            module = node.module
        for alias in node.names:
            self.tracker.add_import(alias, module)
        self.generic_visit(node)  # Continue traversal

    def _get_node_def(self, node: ast.AST) -> Definition:
        own_name_with_anonymous = self.tracker.build_fqn(node)
        if not own_name_with_anonymous or not own_name_with_anonymous.startswith(
            f"{self.module_fqn}."
        ):  # pragma: no cover
            raise ValueError("Failed to build fully qualified name for node.")
        # strip anonymous parts
        own_name = own_name_with_anonymous.split("<")[0].rstrip(".")
        # strip module prefix
        own_name = own_name[len(self.module_fqn) + 1 :].split(".")[0]
        return Definition(self.module_fqn, own_name or None)

    def _visit_load(self, name: str, node: ast.AST, strict: bool = True) -> bool:
        """Visit a name being loaded (used)."""
        # Name is being used (Load context)
        # Check if it's known variable
        if self.tracker.is_in(name):
            if import_elem := self.tracker.is_import(name):
                # Imports are always added to graph.
                target_def: Import | Definition = Import(import_elem, name)
            elif self.tracker.is_local(name):
                # Don't add local variables to graph.
                return True
            else:
                target_def = Definition(self.module_fqn, name)
            own_def = self._get_node_def(node)

            self.dep_graph.setdefault(own_def, set()).add(target_def)
            return True
        # 5. Check if it's a built-in
        if name in BUILTINS:
            return True  # Built-in, ignore

        # 6. Unresolved - could be from star import, global, or undefined
        # We don't automatically add dependencies from star imports due to ambiguity.
        if strict:
            logger.warning(
                "Could not resolve name '%s'. Assuming global/builtin or missing dependency.", name
            )
        return False

    @override
    def visit_Name(self, node: ast.Name) -> None:
        """Resolves identifier usage (loading) against scope and imports."""
        ctx = node.ctx
        name = node.id

        # Check if the name is being defined or deleted (Store, Del context)
        if isinstance(ctx, (ast.Store, ast.Del)):
            # Add name to current scope if defined here (e.g., assignment, for loop var)
            if name in self.tracker.current_scope.global_names:
                self.tracker.scopes[0].names[name] = node
            elif name in self.tracker.current_scope.nonlocal_names:
                self.tracker.scopes[-2].names[name] = node
            else:
                self.tracker.add_name(name, node)
            # No dependency resolution needed for definition target itself
        elif isinstance(ctx, ast.Load):
            self._visit_load(name, node)
        else:  # pragma: no cover
            raise TypeError(f"Unexpected context: {ctx}")
        self.generic_visit(node)

    @override
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle augmented assignment to __all__."""
        self.generic_visit(node)  # store first
        # also assume aug assign needs to load the target
        target = node.target
        if isinstance(target, ast.Attribute):
            while isinstance(target, ast.Attribute):
                target = target.value  # type: ignore[assignment]
        if not isinstance(target, ast.Name):  # pragma: no cover
            raise TypeError("No Name target for AugAssign")
        target = deepcopy(target)
        target.ctx = ast.Load()
        self.visit(target)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visits an attribute access node."""
        # if attribute is not nested or for storing, do normal processing
        if isinstance(node.value, ast.Name) or not isinstance(node.ctx, ast.Load):
            self.generic_visit(node)
            return

        parts = get_attribute_parts(node)
        for ix in range(1, len(parts)):
            fqn = ".".join(parts[:ix])
            if self._visit_load(fqn, node, strict=False):
                break

        self.generic_visit(node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        self._handle_function(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        self._handle_function(node)

    @override
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Visit lambda functions."""
        self._handle_function(node)

    def _handle_function(self, node: FuncType) -> None:
        if isinstance(node, ast.Lambda):
            logger.debug("Registering lambda")
            name: str | int = id(node)
        else:
            logger.debug("Registering function: %s", node.name)
            self._visit_decorators(node)
            if node.returns:
                if self.debug:  # pragma: no cover
                    self._visited_nodes.append("returns")
                self.visit(node.returns)

            args = get_args(node)
            for ix, arg in enumerate(args):
                if arg.annotation:
                    if self.debug:  # pragma: no cover
                        self._visited_nodes.append(f"arg{ix}_ann")
                    self.visit(arg.annotation)
            name = node.name

        self.tracker.add_func(name, node)
        # Do not visit the body yet, just register it
        self.deferred.append(FunctionBodyWrapper(node, self.tracker))

    def _visit_decorators(
        self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Visit decorators."""
        # Decorators are not part of the function body, so we need to visit them
        for ix, decorator in enumerate(node.decorator_list):
            if self.debug:  # pragma: no cover
                self._visited_nodes.append(f"decorator{ix}")
            if not isinstance(decorator, (ast.Name, ast.Call)):  # pragma: no cover
                raise TypeError(f"Decorator {decorator} is not a Name or Call")
            self.visit(decorator)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        logger.debug("Registering class: %s", node.name)
        self._visit_decorators(node)
        for ix, base in enumerate(node.bases):
            if self.debug:  # pragma: no cover
                self._visited_nodes.append(f"base{ix}")
            self.visit(base)
        for ix, keyword in enumerate(node.keywords):  # e.g. metaclass=...
            if self.debug:  # pragma: no cover
                self._visited_nodes.append(f"kwarg{ix}")
            self.visit(keyword.value)

        with self.tracker.scope(node):
            for ix, stmt in enumerate(node.body):
                if self.debug:  # pragma: no cover
                    self._visited_nodes.append(f"stmt{ix}")
                self.visit(stmt)

        self.tracker.add_name(node.name, node)

    @override
    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Visit list comprehensions."""
        self._visit_comprehension(node, node.elt)

    @override
    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Visit set comprehensions."""
        self._visit_comprehension(node, node.elt)

    @override
    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Visit dictionary comprehensions."""
        self._visit_comprehension(node, node.key, node.value)

    @override
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Visit generator expressions."""
        self._visit_comprehension(node, node.elt)

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp, *exprs: ast.expr
    ) -> None:
        """Visit comprehensions and their generators."""
        # Comprehensions have complex scoping (target vars are local)
        # Process outer iterables first
        for ix, comp in enumerate(node.generators):
            if self.debug:  # pragma: no cover
                self._visited_nodes.append(f"generator{ix}")
            self.visit(comp.iter)

        with self.tracker.scope(node):
            for ix, comp in enumerate(node.generators):
                # Add loop variables to the scope
                temp_node = ast.Assign(targets=[comp.target], value=None)  # type: ignore[arg-type]
                # Hacky way to use existing unpacker
                target_names = get_node_defined_names(temp_node)
                self.tracker.add_name(target_names, temp_node)
                # Visit conditions within this scope
                for jx, if_clause in enumerate(comp.ifs):
                    if self.debug:  # pragma: no cover
                        self._visited_nodes.append(f"generator{ix}_if{jx}")
                    self.visit(if_clause)

            # Visit the result expression(s) within the scope
            for ix, expr in enumerate(exprs):
                if self.debug:  # pragma: no cover
                    self._visited_nodes.append(f"generator_expr{ix}")
                self.visit(expr)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        """Visit calls. If the name is a deferred function, visit its body."""
        # TODO(tihoph): if the name is assigned a new name, we can't resolve it
        if isinstance(node.func, ast.Name):
            self.resolve_and_visit(node.func.id)
        elif isinstance(node.func, (ast.Attribute, ast.Call)):
            pass
        else:  # pragma: no cover
            raise TypeError(f"Expected ast.Name for Call.func, got {type(node.func)}")
        self.generic_visit(node)

    def resolve_and_visit(self, name: str) -> None:
        """Resolve a name to its function definition and visit it."""
        resolved = self.tracker.resolve_func(name)
        if resolved:
            if self.tracker.is_visited(resolved):
                logger.debug("Resolved function %s has already been visited, skipping", name)
                return
            for wrapper in self.deferred:
                if wrapper.function == resolved:
                    logger.debug("Visiting resolved function %s", name)
                    self.tracker.mark_visited(wrapper.function)
                    wrapper.accept(self)
                    break
            else:  # pragma: no cover
                raise ValueError("Function not in deferred stack")
        elif name in BUILTINS:
            logger.debug("Name %s is a built-in, skipping visit", name)
        else:
            logger.debug("Function %s not found in current scope", name)

    def visit_deferred(self) -> None:
        """Visit deferred functions that have not been visited yet."""
        while self.deferred:
            wrapper = self.deferred.popleft()
            if self.tracker.is_visited(wrapper.function):
                continue
            self.tracker.mark_visited(wrapper.function)
            if self.debug:  # pragma: no cover
                self._visited_nodes.append(wrapper)
            wrapper.accept(self)


class FunctionBodyWrapper:
    """Wrapper for function bodies to track their scopes."""

    def __init__(self, function_node: FuncType, tracker: ScopeTracker) -> None:
        """Initialize the function body wrapper."""
        self.function = function_node
        self.custom_name = get_node_defined_names(function_node)  # forward name
        self.custom_parent = function_node.parent  # type: ignore[union-attr]
        self.tracker = tracker
        # copy the scopes active at the time of the function definition
        self.scopes = tracker.scopes.copy()

    def accept(self, visitor: ScopeVisitor) -> None:
        """Accept the visitor and visit the function body."""
        # store the original deferred functions and only track current ones
        outer_deferred = visitor.deferred
        visitor.deferred = deque()
        # store the scopes active at the time of function runtime
        outer_scopes = self.tracker.scopes
        self.tracker.scopes = self.scopes
        with self.tracker.scope(self.function):
            # add function parameters to the local scope
            args = get_args(self.function)
            for arg in args:
                self.tracker.add_name(arg.arg, arg)

            if isinstance(self.function.body, ast.expr):
                visitor.visit(self.function.body)
            else:
                for stmt in self.function.body:
                    visitor.visit(stmt)
        # visit the inner deferred functions
        visitor.visit_deferred()
        # restore the outer deferred functions
        visitor.deferred = outer_deferred
        # restore the outer scopes
        self.tracker.scopes = outer_scopes
