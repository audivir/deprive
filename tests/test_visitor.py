"""Test deprive.visitor."""

# ruff: noqa: N802,SLF001
from __future__ import annotations

import ast
import logging

import pytest

from deprive.scope import ScopeTracker
from deprive.visitor import Definition, FunctionBodyWrapper, Import, ScopeVisitor, get_args

ModDef = Definition("test_module", None)
FuncDef = Definition("test_module", "func")
XDef = Definition("test_module", "x")
OsImp = Import("os")


def _make_name(name: str = "x") -> ast.Name:
    """Create a name node."""
    return ast.parse(name).body[0].value  # type: ignore[attr-defined,no-any-return]


def _make_func(name: str = "func") -> ast.FunctionDef:
    """Create a function node with the given name."""
    return ast.parse(f"def {name}(): pass").body[0]  # type: ignore[return-value]


def parse_and_visit(code: str, module_fqn: str = "test_module") -> ScopeVisitor:
    """Parses code and runs ScopeVisitor on it."""
    visitor = ScopeVisitor(module_fqn, debug=True)
    visitor.run(code)
    return visitor


@pytest.mark.parametrize(
    ("snippet", "expected_args"),
    [
        ("x", {"x"}),
        ("x: int", {"x"}),
        ("x, y", {"x", "y"}),
        ("x: int, y: str", {"x", "y"}),
        ("x, y=1", {"x", "y"}),
        ("x, *y", {"x", "y"}),
        ("x, **y", {"x", "y"}),
        ("*x", {"x"}),
        ("**x", {"x"}),
        ("x, *, y", {"x", "y"}),
        ("x, /", {"x"}),
        ("x, y, /", {"x", "y"}),
        ("x, y, /, z", {"x", "y", "z"}),
        ("x, y, *, z", {"x", "y", "z"}),
        ("x, y, /, z, *, w", {"x", "y", "z", "w"}),
    ],
)
def test_get_args(snippet: str, expected_args: set[str]) -> None:
    """Test get_args function."""
    code = f"def f({snippet}): pass"
    func_node: ast.FunctionDef = ast.parse(code).body[0]  # type: ignore[assignment]
    args = get_args(func_node)
    arg_names = {arg.arg for arg in args}
    assert arg_names == expected_args


def test_init() -> None:
    """Test ScopeVisitor initialization."""
    fqn = "my.module"
    visitor = ScopeVisitor(fqn)
    assert visitor.module_fqn == fqn
    assert isinstance(visitor.tracker, ScopeTracker)
    assert len(visitor.tracker.scopes) == 1  # Initial global scope
    assert visitor.deferred == []
    assert visitor.parent is None
    assert visitor.dep_graph == {}


@pytest.mark.parametrize(
    "code", ["global x", "def f():\n  global y", "def f():\n  x=1\n  def g():\n    nonlocal x"]
)
def test_visit_Global_Nonlocal_raises(code: str) -> None:
    """Test that visit_Global and visit_Nonlocal raise NotImplementedError."""
    tree = ast.parse(code)
    visitor = ScopeVisitor("test_mod")
    expected_error = NotImplementedError
    match_msg = (
        "Global statements are not supported"
        if "global" in code
        else "Nonlocal statements are not supported"
    )
    with pytest.raises(expected_error, match=match_msg):
        visitor.visit(tree)


@pytest.mark.parametrize(
    ("code", "expected_imports", "expected_imports_from"),
    [
        ("import os", {"os": "os"}, {}),
        ("import sys as system", {"system": "sys"}, {}),
        (
            "from collections import defaultdict",
            {},
            {"defaultdict": ("collections", "defaultdict")},
        ),
        ("from pathlib import Path as P", {}, {"P": ("pathlib", "Path")}),
        # Ensure tracker handles multiple imports added via visitor
        ("import os\nimport logging", {"os": "os", "logging": "logging"}, {}),
        ("from a import b\nfrom c import d as e", {}, {"b": ("a", "b"), "e": ("c", "d")}),
    ],
    ids=[
        "simple_import",
        "import_as",
        "from_import",
        "from_import_as",
        "multi_import",
        "multi_from_import",
    ],
)
def test_visit_Import_ImportFrom(
    code: str, expected_imports: dict[str, str], expected_imports_from: dict[str, tuple[str, str]]
) -> None:
    """Test that visiting imports updates the tracker correctly."""
    visitor = parse_and_visit(code)
    # Imports are added to the outermost scope
    outer_scope = visitor.tracker.scopes[0]
    assert outer_scope.imports == expected_imports
    assert outer_scope.imports_from == expected_imports_from

    # test that imports in scopes are only in this scope
    nested_code_lines = ["def func():"] + [f"  {line}" for line in code.splitlines()]
    nested_code = "\n".join(nested_code_lines)
    nested_visitor = parse_and_visit(nested_code)
    func_node = nested_visitor._visited_nodes[1]
    nested_outer_scope = nested_visitor.tracker.scopes[0]
    assert nested_outer_scope.imports == {}
    assert nested_outer_scope.imports_from == {}
    nested_inner_scope = nested_visitor.tracker.all_scopes[id(func_node)]
    assert nested_inner_scope.imports == expected_imports
    assert nested_inner_scope.imports_from == expected_imports_from


@pytest.mark.parametrize(
    ("code", "expected"), [("x = 1", {"x"}), ("del x", {"x"})], ids=["simple_name", "simple_del"]
)
def test_visit_Name_store_del(code: str, expected: set[str]) -> None:
    """Test that visiting imports updates the tracker correctly."""
    visitor = parse_and_visit(code)
    # Imports are added to the outermost scope
    outer_scope = visitor.tracker.scopes[0]
    assert set(outer_scope.names) == expected

    # test that imports in scopes are only in this scope
    nested_code_lines = ["def func():"] + [f"  {line}" for line in code.splitlines()]
    nested_code = "\n".join(nested_code_lines)
    nested_visitor = parse_and_visit(nested_code)
    func_node = nested_visitor._visited_nodes[1]
    nested_outer_scope = nested_visitor.tracker.scopes[0]
    assert set(nested_outer_scope.names) == set()  # outer scope should be empty
    nested_inner_scope = nested_visitor.tracker.all_scopes[id(func_node)]
    assert set(nested_inner_scope.names) == expected


@pytest.mark.parametrize(
    ("code", "expected", "dep_graph", "unresolved"),
    [
        ("x = 1\nx", {"x"}, {ModDef: {XDef}, XDef: set()}, []),
        ("del x", {"x"}, {ModDef: set(), XDef: set()}, []),
        ("x", set(), {ModDef: set()}, ["x"]),
        ("import os\nos.path", set(), {ModDef: {OsImp}}, []),
        ("import os\nx = 1\nx\nos.path", {"x"}, {ModDef: {OsImp, XDef}, XDef: set()}, []),
        # TODO(tihoph): Attribute access not implemented
        ("import importlib.util\nimportlib.spec", set(), {ModDef: set()}, []),
    ],
    ids=[
        "just_load",
        "just_del",
        "simple_unknown",
        "imported",
        "imported_and_load",
        "other_import",
    ],
)
def test_visit_Name_load(
    code: str,
    expected: set[str],
    dep_graph: dict[Definition, set[Definition | Import]],
    unresolved: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that visiting imports updates the tracker correctly."""
    with caplog.at_level(logging.DEBUG):
        visitor = parse_and_visit(code)
    for elem in unresolved:
        assert f"Could not resolve name '{elem}'." in caplog.text
    # Imports are added to the outermost scope
    outer_scope = visitor.tracker.scopes[0]
    assert set(outer_scope.names) == expected

    assert visitor.dep_graph == dep_graph

    # test that imports in scopes are only in this scope
    nested_code_lines = ["def func():"] + [f"  {line}" for line in code.splitlines()]
    nested_code = "\n".join(nested_code_lines)
    nested_visitor = parse_and_visit(nested_code)
    func_node = nested_visitor._visited_nodes[1]
    nested_outer_scope = nested_visitor.tracker.scopes[0]
    assert set(nested_outer_scope.names) == set()  # outer scope should be empty
    nested_inner_scope = nested_visitor.tracker.all_scopes[id(func_node)]
    assert set(nested_inner_scope.names) == expected

    assert nested_visitor.dep_graph == {ModDef: set(), FuncDef: set()}


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("@dec\ndef func(): pass", ["decorator0"]),
        ("@dec1\n@dec2\ndef func(): pass", ["decorator0", "decorator1"]),
        ("@dec\nclass Test: pass", ["decorator0", "stmt0"]),
        ("@dec1\n@dec2\nclass Test: pass", ["decorator0", "decorator1", "stmt0"]),
    ],
    ids=["decorated_func", "multidecorated_func", "decorated_class", "multidecorated_class"],
)
def test_visit_decorators(code: str, expected: list[str]) -> None:
    visitor = parse_and_visit(code, "comp_mod")
    string_nodes = [x for x in visitor._visited_nodes if isinstance(x, str)]
    assert string_nodes == expected


@pytest.mark.parametrize(
    ("code", "ix", "cls", "expected"),
    [
        ("def func(): pass", 1, ast.FunctionDef, []),
        ("async def func(): pass", 1, ast.AsyncFunctionDef, []),
        ("lambda: None", 2, ast.Lambda, []),  # expr before Lambda
        ("def func(x: int): pass", 1, ast.FunctionDef, ["arg0_ann"]),
        ("def func(x: int, y: int): pass", 1, ast.FunctionDef, ["arg0_ann", "arg1_ann"]),
        ("def func() -> None: pass", 1, ast.FunctionDef, ["returns"]),
    ],
    ids=[
        "simple_func",
        "simple_async",
        "simple_lambda",
        "annotated_func",
        "multiannotated_func",
        "returns_func",
    ],
)
def test_visit_FunctionDef_Async_Lambda(
    code: str, ix: int, cls: type[ast.AST], expected: list[str]
) -> None:
    """Test visiting function definition scope and visits components correctly."""
    visitor = parse_and_visit(code, "comp_mod")
    assert isinstance(visitor._visited_nodes[ix], cls)
    # verify the body was visited at the end
    assert isinstance(visitor._visited_nodes[-2], FunctionBodyWrapper)
    assert isinstance(visitor._visited_nodes[-1], (ast.Pass, ast.Constant))

    string_nodes = [x for x in visitor._visited_nodes[ix + 1 : -2] if isinstance(x, str)]
    assert string_nodes == expected
    # TODO(tihoph): test if names are added to scope, if decorator are run


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("class Test: pass", ["stmt0"]),
        ("class Test(Base): pass", ["base0", "stmt0"]),
        ("class Test(metaclass=Meta): pass", ["kwarg0", "stmt0"]),
        ("class Test(Base1, Base2): pass", ["base0", "base1", "stmt0"]),
        ("class Test(Base1, Base2, metaclass=Meta): pass", ["base0", "base1", "kwarg0", "stmt0"]),
        (
            "class Test(Base1, Base2, metaclass=Meta, option=True): pass",
            ["base0", "base1", "kwarg0", "kwarg1", "stmt0"],
        ),
        ("class Test:\n  def __init__(self): pass", ["stmt0"]),
        ("class Test:\n  def __init__(self): pass\n  def __repr__(self): pass", ["stmt0", "stmt1"]),
    ],
    ids=[
        "simple_class",
        "class_with_base",
        "class_with_metaclass",
        "multiple_bases",
        "bases_and_metaclass",
        "bases_metaclass_and_options",
        "class_with_content",
        "class_with_multiple",
    ],
)
def test_visit_ClassDef(code: str, expected: list[str]) -> None:
    """Test visiting class definition scope and visits components correctly."""
    visitor = parse_and_visit(code, "comp_mod")

    string_nodes = [x for x in visitor._visited_nodes[2:] if isinstance(x, str)]
    assert string_nodes == expected
    # TODO(tihoph): test if names are added to scope, if decorator are run


@pytest.mark.parametrize(
    ("code", "cls", "expected"),
    [
        ("[x for x in data]", ast.ListComp, ["generator0", "generator_expr0"]),
        ("{y for y in unknown}", ast.SetComp, ["generator0", "generator_expr0"]),
        (
            "{k: v for k, v in unknown}",
            ast.DictComp,
            ["generator0", "generator_expr0", "generator_expr1"],
        ),
        ("(z * 2 for z in generator)", ast.GeneratorExp, ["generator0", "generator_expr0"]),
        (
            "[x for x in data if x > 0]",
            ast.ListComp,
            ["generator0", "generator0_if0", "generator_expr0"],
        ),
        ("[x if x > 0 else 1 for x in data]", ast.ListComp, ["generator0", "generator_expr0"]),
        (
            "[1 for sublist in nested for x in sublist]",
            ast.ListComp,
            ["generator0", "generator1", "generator_expr0"],
        ),
        (
            "[1 for sublist in nested if sublist for x in sublist if x]",
            ast.ListComp,
            ["generator0", "generator1", "generator0_if0", "generator1_if0", "generator_expr0"],
        ),
    ],
    ids=[
        "listcomp",
        "setcomp",
        "dictcomp",
        "genexp",
        "listcomp_if",
        "listcomp_if_else",
        "listcomp_multiple",
        "listcomp_multiple_if",
    ],
)
def test_visit_Comprehension(code: str, cls: type[ast.AST], expected: list[str]) -> None:
    """Test visiting comprehensions handles scope and visits components correctly."""
    visitor = parse_and_visit(code, "comp_mod")

    assert isinstance(visitor._visited_nodes[2], cls)

    string_nodes = [x for x in visitor._visited_nodes[3:] if isinstance(x, str)]
    assert string_nodes == expected
    # TODO(tihoph): test if names are added to scope


@pytest.mark.parametrize(
    ("name", "log_text"),
    [
        ("func", "Visiting resolved function"),
        ("already", "has already been visited, skipping"),
        ("inner", "Visiting resolved function"),
        ("print", "is a built-in, skipping visit"),
        ("unknown", "not found in current scope"),
    ],
)
def test_resolve_and_visit(name: str, log_text: str, caplog: pytest.LogCaptureFixture) -> None:
    """Test resolve_and_visit method."""
    func_node = _make_func()
    inner_node = _make_func("inner")
    node = _make_name()
    visitor = ScopeVisitor("test_mod")
    visitor.tracker.add_func("func", func_node)

    if name == "already":
        visitor.tracker.mark_visited(func_node)
        name = "func"

    with visitor.tracker.scope(node), caplog.at_level(logging.DEBUG):
        visitor.tracker.add_func("inner", inner_node)
        visitor.resolve_and_visit(name)

    assert log_text in caplog.text
    if name == "func":
        assert visitor.tracker.is_visited(func_node)

    if name == "inner":  # after exiting the scope of 'func', 'inner' should be hidden
        assert visitor.tracker.is_visited(inner_node)
        with caplog.at_level(logging.DEBUG):
            visitor.resolve_and_visit(name)
            assert "not found in current scope" in caplog.text


def test_visit_deferred() -> None:
    """Test FunctionBodyWrapper accept method."""
    tree = ast.parse("def func():\n  def inner(): pass\ndef func2(): pass")
    func_node: ast.FunctionDef = tree.body[0]  # type: ignore[assignment]
    inner_node: ast.FunctionDef = func_node.body[0]  # type: ignore[assignment]
    func2_node: ast.FunctionDef = tree.body[1]  # type: ignore[assignment]
    visitor = ScopeVisitor("test_mod", debug=True)
    wrapper = FunctionBodyWrapper(func_node, visitor.tracker)
    wrapper2 = FunctionBodyWrapper(func2_node, visitor.tracker)

    visitor.deferred.append(wrapper)  # Add to deferred list
    visitor.deferred.append(wrapper2)  # Add another deferred function
    assert visitor.deferred == [wrapper, wrapper2]

    visitor.visit_deferred()

    assert len(visitor.deferred) == 3  # Check if inner function was added
    assert visitor.tracker.visited_funcs == [id(func_node), id(inner_node), id(func2_node)]
    assert visitor.deferred[-1].function == inner_node
    inner_wrapper = visitor.deferred[-1]
    assert visitor._visited_nodes == [
        wrapper,
        inner_node,
        inner_wrapper,
        inner_node.body[0],
        wrapper2,
        wrapper2.function.body[0],  # type: ignore[index]
    ]


@pytest.mark.parametrize(
    ("code", "add", "dep_graph", "unresolved"),
    [
        ("def func(): ...", [], {ModDef: set(), FuncDef: set()}, []),
        ("def func(): ...", ["func"], {ModDef: {FuncDef}, FuncDef: set()}, []),
        ("def func(): ...", ["test_module.func"], {ModDef: {FuncDef}, FuncDef: set()}, []),
        ("def func():\n  def inner(): pass", [], {ModDef: set(), FuncDef: set()}, []),
        ("def func():\n  def inner(): pass", ["func"], {ModDef: {FuncDef}, FuncDef: set()}, []),
        ("def func():\n  def inner(): pass", ["inner"], {ModDef: set(), FuncDef: set()}, ["inner"]),
        ("import os", [], {ModDef: set()}, []),
    ],
)
def test_add(
    code: str,
    add: list[str],
    dep_graph: dict[Definition, set[Definition | Import]],
    unresolved: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the ScopeVisitor class."""
    # Create a simple module with a function and an import
    visitor = parse_and_visit(code)
    with caplog.at_level(logging.DEBUG):
        for name in add:
            visitor.add(name)
    for elem in unresolved:
        assert f"Could not resolve name '{elem}'." in caplog.text
    assert visitor.dep_graph == dep_graph


def test_function_body_wrapper_init() -> None:
    """Test FunctionBodyWrapper initialization."""
    func_node = _make_func()
    mod = ast.Module([], [])
    func_node.parent = mod  # type: ignore[attr-defined]
    tracker = ScopeTracker()
    wrapper = FunctionBodyWrapper(func_node, tracker)
    assert wrapper.function == func_node
    assert wrapper.custom_name == "func"
    assert wrapper.custom_parent == mod
    assert wrapper.tracker == tracker


def test_function_body_wrapper_accept() -> None:
    """Test FunctionBodyWrapper accept method."""
    func_node: ast.FunctionDef = ast.parse("def func():\n  def inner(): pass").body[0]  # type: ignore[assignment]
    inner_node: ast.FunctionDef = func_node.body[0]  # type: ignore[assignment]
    visitor = ScopeVisitor("test_mod", debug=True)
    wrapper = FunctionBodyWrapper(func_node, visitor.tracker)
    visitor.deferred.append(wrapper)  # Add to deferred list
    assert visitor.deferred == [wrapper]
    visitor.tracker.mark_visited(wrapper.function)
    wrapper.accept(visitor)  # Call accept method

    # assert that the inner func was added to new deferred
    assert len(visitor.deferred) == 2
    assert visitor.deferred[0] == wrapper
    assert isinstance(visitor.deferred[1], FunctionBodyWrapper)
    inner_wrapper = visitor.deferred[1]
    assert inner_wrapper.function == inner_node
    # TODO(tihoph): Check if args were added to scope etc.

    assert visitor.tracker.visited_funcs == [id(func_node), id(inner_node)]
    assert visitor._visited_nodes == [inner_node, inner_wrapper, inner_node.body[0]]
