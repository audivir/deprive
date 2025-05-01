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
    # 0 is no valid identifier so it can default value
    asname: str | None = field(default="0", hash=True)

    def __post_init__(self) -> None:
        if self.asname == "0":
            name = self.name if isinstance(self.name, str) else self.name[1]
            object.__setattr__(self, "asname", name)  # fix around frozen dataclass


@dataclass(frozen=True)
class Definition:
    """Data class for representing a definition."""

    module: str = field(hash=True)
    name: str | None = field(hash=True)


def get_args(node: FuncType) -> tuple[list[ast.arg], list[ast.expr]]:
    """Get all arguments of a function node."""
    all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
    if node.args.vararg:
        all_args.append(node.args.vararg)
    if node.args.kwarg:
        all_args.append(node.args.kwarg)
    defaults = [item for item in node.args.defaults + node.args.kw_defaults if item is not None]
    return all_args, defaults


class ScopeVisitor(ast.NodeVisitor):
    """Visitor that tracks function definitions and their scopes."""

    def __init__(self, fqn: str, init: bool = False, debug: bool = False) -> None:
        """Initialize the visitor."""
        self.module_fqn = fqn
        self.init = init

        self.tracker = ScopeTracker()

        self.deferred: deque[FunctionBodyWrapper] = deque()

        self.parent: ast.AST | FunctionBodyWrapper | None = None
        self.dep_graph: DepGraph = {}  # Dependency graph of function dependencies

        self._visited_nodes: list[ast.AST | FunctionBodyWrapper | str] = []
        self.debug = debug

        self.all: list[str] | None = None

    def _add_top_level_defs(self) -> None:
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

    def _add_imports(self) -> None:
        # add imports to the dependency graph
        outer_scope = self.tracker.scopes[0]
        top_level_imports = outer_scope.imports

        # add names in __all__ to the dependency graph
        if self.all:
            for exported_name in self.all:
                definition = Definition(self.module_fqn, exported_name)
                if definition not in self.dep_graph:
                    if found_imports := top_level_imports.get(exported_name, None):
                        self.dep_graph[definition] = {
                            Import(x[0] if len(x) == 2 else x[:2], exported_name)  # noqa: PLR2004
                            for x in found_imports
                        }
                    else:
                        raise ValueError("Name in __all__ neither defined nor an import")

        # if own name is __init__ imports with an alias name should be added
        if self.module_fqn.endswith(".__init__"):
            for alias, imports in top_level_imports.items():
                definition = Definition(self.module_fqn, alias)
                imports_to_add: set[Definition | Import] = set()
                for curr_import in imports:
                    exported_import = curr_import[-1]
                    actual_import = curr_import[0] if len(curr_import) == 2 else curr_import[:2]  # noqa: PLR2004
                    # only add explicit aliased imports
                    if not exported_import:
                        continue
                    imports_to_add.add(Import(actual_import, exported_import))
                if imports_to_add:
                    self.dep_graph[definition] = imports_to_add

    def run(self, code: str) -> None:
        """Run the visitor on a given code string."""
        tree = ast.parse(code)
        tree.custom_name = self.module_fqn  # type: ignore[attr-defined]
        add_parents(tree)
        self.visit(tree)
        self.visit_deferred()
        self._add_top_level_defs()
        self._add_imports()

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
                # if module is __init__ this fails
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

    def _visit_load(
        self, name: str, node: ast.AST, strict: bool = True
    ) -> tuple[Definition, list[Import]] | bool:
        """Visit a name being loaded (used)."""
        # Name is being used (Load context)
        # Check if it's known variable
        if self.tracker.is_in(name):
            own_def = self._get_node_def(node)

            if import_elem := self.tracker.is_import(name):
                # Imports are always added to graph.
                import_defs: list[Import] = []
                for curr_import in import_elem:
                    import_name = curr_import[0] if len(curr_import) == 2 else curr_import[:2]  # noqa: PLR2004
                    import_def = Import(import_name, name)
                    self.dep_graph.setdefault(own_def, set()).add(import_def)
                    import_defs.append(import_def)
                return own_def, import_defs

            if not self.tracker.is_local(name):
                # Don't add local variables to graph.
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

    def _handle_all(self, node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> None:
        if node.value is None:
            raise ValueError("No value for __all__ assignment.")
        # verify we a re in a module scope
        if len(self.tracker.scopes) != 1:
            raise ValueError("__all__ must be defined at the module level.")
        code = ast.unparse(node.value)
        contents = ast.literal_eval(code)
        if not isinstance(contents, (tuple, list)) or not all(
            isinstance(item, str) for item in contents
        ):
            raise TypeError("Expected a tuple or list of literal strings, got: " + code)
        self.all = list(contents)
        self.generic_visit(node)

    @override
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle augmented assignment to __all__."""
        if isinstance(node.target, ast.Name) and node.target.id == "__all__":
            raise ValueError("Unsupported __all__ definition.")

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
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle __all__ seperately from other assignments."""
        # TODO(tihoph): __all__, other = [...], ... is currently not handled correctly.
        if isinstance(node.target, ast.Name) and node.target.id == "__all__":
            self._handle_all(node)
            return

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
            return

        names = get_node_defined_names(node, strict=False)
        if not names:
            names = ()
        if isinstance(names, str):
            names = (names,)
        # TODO(tihoph): __all__.x should also be forbidden.
        if "__all__" in names:
            raise ValueError("Unsupported __all__ definition.")

        self.generic_visit(node)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Visits an attribute access node."""
        if not isinstance(node.ctx, ast.Load):
            self.generic_visit(node)
            return

        parts = get_attribute_parts(node)
        found: tuple[Definition, list[Import]] | bool = False
        for ix in range(1, len(parts)):
            fqn = ".".join(parts[:ix])
            found = self._visit_load(fqn, node, strict=False)
        if isinstance(found, tuple):
            own_def, import_defs = found
            for import_def in import_defs:
                import_fqn = (
                    import_def.name
                    if isinstance(import_def.name, str)
                    else ".".join(import_def.name)
                )
                if not self.tracker.is_import(f"{import_fqn}.{parts[-1]}"):
                    self.dep_graph.setdefault(own_def, set()).add(
                        Import((import_fqn, parts[-1]), None)
                    )

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

            args, defaults = get_args(node)
            for ix, arg in enumerate(args):
                if arg.annotation:
                    if self.debug:  # pragma: no cover
                        self._visited_nodes.append(f"arg{ix}_ann")
                    self.visit(arg.annotation)
            for ix, default in enumerate(defaults):
                if self.debug:  # pragma: no cover
                    self._visited_nodes.append(f"default{ix}")
                self.visit(default)
            name = node.name

        if not isinstance(node.parent, ast.ClassDef):  # type: ignore[union-attr]
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
        elif isinstance(node.func, (ast.Attribute, ast.Subscript, ast.Call)):
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
        # # store the original deferred functions and only track current ones
        deferred_size = len(visitor.deferred)
        # store the scopes active at the time of function runtime
        outer_scopes = self.tracker.scopes
        self.tracker.scopes = self.scopes

        with self.tracker.scope(self.function):
            # add function parameters to the local scope
            args, _ = get_args(self.function)
            for arg in args:
                self.tracker.add_name(arg.arg, arg)

            if isinstance(self.function.body, ast.expr):
                visitor.visit(self.function.body)
            else:
                for stmt in self.function.body:
                    visitor.visit(stmt)
        # visit the inner deferred functions
        deferred_list = list(visitor.deferred)
        outer_deferred = deferred_list[:deferred_size]
        # inner deferred functions are visited now
        visitor.deferred = deque(deferred_list[deferred_size:])
        # visit the inner deferred functions
        visitor.visit_deferred()
        # restore the outer deferred functions
        visitor.deferred = deque(outer_deferred)
        # restore the outer scopes
        self.tracker.scopes = outer_scopes
