"""Visitor for analyzing and processing AST nodes."""

# ruff: noqa: N802
from __future__ import annotations

import ast
import builtins
import logging
from dataclasses import dataclass, field

from typing_extensions import TypeAlias, override

from deprive.names import get_attribute_parts, get_node_defined_names
from deprive.scope import ScopeTracker

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


def get_args(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list[ast.arg]:
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
        self.deferred: list[FunctionBodyWrapper] = []

        self.parent: ast.AST | FunctionBodyWrapper | None = None
        self.dep_graph: DepGraph = {}  # Dependency graph of function dependencies

        self._visited_nodes: list[ast.AST | FunctionBodyWrapper | str] = []
        self.debug = debug

    def run(self, code: str) -> None:
        """Run the visitor on a given code string."""
        tree = ast.parse(code)
        tree.custom_name = self.module_fqn  # type: ignore[attr-defined]
        self.visit(tree)
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
    def visit(self, node: ast.AST | FunctionBodyWrapper) -> None:
        """Visit a node. If the node is a function body wrapper, visit its body."""
        if self.debug:  # pragma: no cover
            self._visited_nodes.append(node)
        original_parent = self.parent
        self.parent = node
        if isinstance(node, FunctionBodyWrapper):
            node.parent = node.custom_parent  # type: ignore[attr-defined]
        else:
            node.parent = original_parent  # type: ignore[attr-defined]
        if isinstance(node, FunctionBodyWrapper):
            node.accept(self)
        else:
            super().visit(node)
        self.parent = original_parent

    @override
    def visit_Global(self, node: ast.Global) -> None:
        """Handle global statements."""
        del node  # unused
        raise NotImplementedError("Global statements are not supported yet.")

    @override
    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Handle nonlocal statements."""
        del node  # unused
        raise NotImplementedError("Nonlocal statements are not supported yet.")

    @override
    def visit_Import(self, node: ast.Import) -> None:
        """Stores `import module [as alias]`."""
        self.tracker.add_import(node)
        self.generic_visit(node)

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Stores `from module import name [as alias]` including relative imports."""
        self.tracker.add_import(node)
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
        # 1. Check if it's a local/enclosing scope variable
        if self.tracker.is_in(name):
            if not self.tracker.is_in(name, inner_only=True):
                own_def = self._get_node_def(node)

                if import_elem := self.tracker.is_import(name, outer_only=True):
                    target_def: Import | Definition = Import(import_elem, name)
                else:
                    target_def = Definition(self.module_fqn, name)

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
            # Add name to local scope if defined here (e.g., assignment, for loop var)
            # Check parent context to be more precise (e.g., don't add func/class names again here)
            self.tracker.add_name(name, node)
            # No dependency resolution needed for definition target itself
            self.generic_visit(node)
            return

        if not isinstance(ctx, ast.Load):  # pragma: no cover
            raise TypeError(f"Unexpected context: {ctx}")

        self._visit_load(name, node)

        self.generic_visit(node)

    def _handle_all(self, node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> None:
        if node.value is None:
            raise ValueError("No value for __all__ assignment.")
        code = ast.unparse(node.value)
        contents = ast.literal_eval(code)
        # verify we a re in a module scope
        if len(self.tracker.scopes) != 1:
            raise ValueError("__all__ must be defined at the module level.")
        # TODO(tihoph): handle all
        logger.debug("Handling __all__: %s", contents)

    @override
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle augmented assignment to __all__."""
        if isinstance(node.target, ast.Name) and node.target.id == "__all__":
            self._handle_all(node)
            raise NotImplementedError("Augmented assignment to __all__ is not supported yet.")
        self.generic_visit(node)

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle __all__ seperately from other assignments."""
        # TODO(tihoph): __all__, other = [...], ... is currently not handled correctly.
        if isinstance(node.target, ast.Name) and node.target.id == "__all__":
            self._handle_all(node)

        self.generic_visit(node)

    @override
    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle __all__ seperately from other assignments."""
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        ):
            self._handle_all(node)

        self.generic_visit(node)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visits an attribute access node."""
        # if attribute is not nested or for storing, do normal processing
        if isinstance(node.value, ast.Name) or not isinstance(node.ctx, ast.Load):
            self.generic_visit(node)
            return

        parts = get_attribute_parts(node)
        if parts:
            for ix in range(1, len(parts)):
                fqn = ".".join(parts[:ix])
                if self._visit_load(fqn, node, strict=False):
                    break

        self.generic_visit(node)

    @override
    def visit_Module(self, node: ast.Module) -> None:
        """Visit the module node."""
        for stmt in node.body:
            self.visit(stmt)
        self.visit_deferred()

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

    def _handle_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        if not isinstance(node, ast.Lambda):
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
            name: str | int = node.name
        else:
            logger.debug("Registering lambda")
            name = id(node)
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
        if isinstance(node.func, (ast.Attribute, ast.Call)):
            self.generic_visit(node)
            return

        if not isinstance(node.func, ast.Name):  # pragma: no cover
            raise TypeError(f"Expected ast.Name for Call.func, got {type(node.func)}")

        self.resolve_and_visit(node.func.id)
        self.generic_visit(node)

    def resolve_and_visit(self, name: str) -> None:
        """Resolve a name to its function definition and visit it."""
        resolved = self.tracker.resolve_func(name)
        if resolved:
            if self.tracker.is_visited(resolved):
                logger.debug("Function %s has already been visited, skipping", name)
                return
            logger.debug("Visiting resolved function: %s", name)
            self.visit(FunctionBodyWrapper(resolved, self.tracker))
            self.tracker.mark_visited(resolved)
        elif name in BUILTINS:
            logger.debug("Name %s is a built-in, skipping visit", name)
        else:
            logger.debug("Function %s not found in current scope", name)

    def visit_deferred(self) -> None:
        """Visit deferred functions that have not been visited yet."""
        current_deferred = self.deferred.copy()  # create a copy as .accept mutates deferred
        for wrapper in current_deferred:
            if not self.tracker.is_visited(wrapper.function):
                logger.debug("[END] Visiting deferred function: %s", wrapper.custom_name)
                self.tracker.mark_visited(wrapper.function)
                self.visit(wrapper)

    def add(self, name: str) -> None:
        """Add an external function which is required. Changes the dependency graph."""
        name = name.removeprefix(f"{self.module_fqn}.")
        name = name.split(".")[0]  # TODO(tihoph): subpackage imports?
        node = ast.Name(id=name, ctx=ast.Load())
        dummy_module = ast.Module([node], [])  # type: ignore[list-item]
        dummy_module.custom_name = self.module_fqn  # type: ignore[attr-defined]
        self.visit(dummy_module)


class FunctionBodyWrapper:
    """Wrapper for function bodies to track their scopes."""

    def __init__(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
        tracker: ScopeTracker,
    ) -> None:
        """Initialize the function body wrapper."""
        self.function = function_node
        self.custom_name = get_node_defined_names(function_node)  # forward name
        self.custom_parent = getattr(function_node, "parent", None)
        self.tracker = tracker

    def accept(self, visitor: ScopeVisitor) -> None:
        """Accept the visitor and visit the function body."""
        logger.debug("Entering body of function: %s", self.custom_name)
        with self.tracker.scope(self.function):
            args = get_args(self.function)
            for arg in args:
                self.tracker.add_name(arg.arg, arg)

            # store the original deferred functions and only track current ones
            current_deferred = visitor.deferred.copy()
            visitor.deferred.clear()
            if isinstance(self.function.body, ast.expr):
                visitor.visit(self.function.body)
            else:
                for stmt in self.function.body:
                    visitor.visit(stmt)
            # visit the current deferred functions
            visitor.visit_deferred()
            # restore the original deferred functions
            current_deferred.extend(visitor.deferred)
            visitor.deferred = current_deferred
        logger.debug("Exiting body of function: %s", self.custom_name)
