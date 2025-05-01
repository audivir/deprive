"""Test visit_* functions of visitor."""

# ruff: noqa: N802,SLF001
from __future__ import annotations

import ast
import logging
import textwrap
from collections import deque
from typing import TYPE_CHECKING

import pytest

from deprive.scope import ScopeTracker, add_parents
from deprive.visitor import Definition, FunctionBodyWrapper, Import, ScopeVisitor, get_args

if TYPE_CHECKING:
    from collections.abc import Sequence

ModDef = Definition("test_module", None)
FuncDef = Definition("test_module", "func")
AllDef = Definition("test_module", "__all__")
XDef = Definition("test_module", "x")
OsImp = Import("os")


def parse_and_visit(code: str | Sequence[str], module_fqn: str = "test_module") -> ScopeVisitor:
    """Parses code and runs ScopeVisitor on it."""
    visitor = ScopeVisitor(module_fqn, debug=True)
    if not isinstance(code, str):  # pragma: no cover
        code = "\n".join(code)
    visitor.run(textwrap.dedent(code))
    return visitor


def visited_before(visitor: ScopeVisitor, before: type[ast.AST], after: type[ast.AST]) -> bool:
    """Whether the class before was visited before the class after."""
    before_ix = next(
        iter(ix for ix, node in enumerate(visitor._visited_nodes) if isinstance(node, before))
    )
    after_ix = next(
        iter(ix for ix, node in enumerate(visitor._visited_nodes) if isinstance(node, after))
    )
    return before_ix < after_ix


def test_Import() -> None:
    os_imp = Import("os")
    assert os_imp.name == "os"
    assert os_imp.asname == "os"
    # comparable?
    assert Import("os") == os_imp
    assert Import("os") == Import("os", "os")
    # hashable?
    assert {os_imp} == {Import("os")}


def test_Definition() -> None:
    def_def = Definition("module", "name")
    assert def_def.module == "module"
    assert def_def.name == "name"
    # comparable?
    assert Definition("module", "name") == def_def
    # hashable?
    assert {def_def} == {Definition("module", "name")}


@pytest.mark.parametrize(
    ("snippet", "expected_args", "n_defaults"),
    [
        ("x", {"x"}, 0),
        ("x: int", {"x"}, 0),
        ("x, y", {"x", "y"}, 0),
        ("x: int, y: str", {"x", "y"}, 0),
        ("x, y=1", {"x", "y"}, 1),
        ("x, *y", {"x", "y"}, 0),
        ("x, **y", {"x", "y"}, 0),
        ("*x", {"x"}, 0),
        ("**x", {"x"}, 0),
        ("x, *, y", {"x", "y"}, 0),
        ("x, /", {"x"}, 0),
        ("x, y, /", {"x", "y"}, 0),
        ("x, y, /, z", {"x", "y", "z"}, 0),
        ("x, y, *, z", {"x", "y", "z"}, 0),
        ("x, y, /, z, *, w", {"x", "y", "z", "w"}, 0),
    ],
)
def test_get_args(snippet: str, expected_args: set[str], n_defaults: int) -> None:
    """Test get_args function."""
    code = f"def f({snippet}): pass"
    func_node: ast.FunctionDef = ast.parse(code).body[0]  # type: ignore[assignment]
    args, defaults = get_args(func_node)
    arg_names = {arg.arg for arg in args}
    assert arg_names == expected_args
    assert len(defaults) == n_defaults  # TODO(tihoph): test also defaults


def test_ScopeVisitor_init() -> None:
    """Test ScopeVisitor initialization."""
    fqn = "my.module"
    visitor = ScopeVisitor(fqn)
    assert visitor.module_fqn == fqn
    assert isinstance(visitor.tracker, ScopeTracker)
    assert len(visitor.tracker.scopes) == 1  # Initial global scope
    assert visitor.deferred == deque()
    assert visitor.parent is None
    assert visitor.dep_graph == {}
    assert visitor._visited_nodes == []
    assert visitor.all is None


def test_run_add_imports() -> None:
    code = "import os\nimport sys as sys"
    visitor = parse_and_visit(code)
    assert visitor.dep_graph == {Definition("test_module", None): set()}

    visitor = parse_and_visit(code, "test_module.__init__")
    assert visitor.dep_graph == {
        Definition("test_module.__init__", None): set(),
        Definition("test_module.__init__", "sys"): {Import("sys")},
    }

    # for redefinitions add both
    code = "import os\nif True: import sys as sys\nelse: import os as sys"
    visitor = parse_and_visit(code, "test_module.__init__")
    assert visitor.dep_graph == {
        Definition("test_module.__init__", None): set(),
        Definition("test_module.__init__", "sys"): {Import("sys"), Import("os", "sys")},
    }


def test_run_add_definitions() -> None:
    visitor = parse_and_visit("import sys\nx = sys.path")
    assert visitor.dep_graph == {
        Definition("test_module", None): set(),
        Definition("test_module", "x"): {Import("sys"), Import(("sys", "path"), None)},
    }
    visitor = parse_and_visit("import sys, os\nx = sys.path\nx = os.path")
    assert visitor.dep_graph == {
        Definition("test_module", None): set(),
        Definition("test_module", "x"): {
            Import("sys"),
            Import("os"),
            Import(("sys", "path"), None),
            Import(("os", "path"), None),
        },
    }


def test_run_overwrite_definitions() -> None:
    visitor = parse_and_visit("import sys, os\ndef os(): sys.path")
    assert visitor.dep_graph == {
        Definition("test_module", None): set(),
        Definition("test_module", "os"): {Import("sys"), Import(("sys", "path"), None)},
    }

    visitor = parse_and_visit("import sys, os\ndef other():\n  os = sys.path\n  print(os)")
    assert visitor.dep_graph == {
        Definition("test_module", None): set(),
        Definition("test_module", "other"): {
            Import("sys"),
            Import("os"),
            Import(("sys", "path"), None),
        },
    }

    visitor = parse_and_visit("import sys, os\ndef other():\n  print(os)\ndef other(): print(sys)")
    assert visitor.dep_graph == {
        Definition("test_module", None): set(),
        Definition("test_module", "other"): {Import("sys"), Import("os")},
    }


def test_visit_Global_fail() -> None:
    code = """\
    global x
    x = 1
    """
    with pytest.raises(ValueError, match="Global keyword in outer scope redundant"):
        parse_and_visit(code)


def test_visit_Global() -> None:
    code = """\
    x = 1
    def func():
        global x
    """
    visitor = parse_and_visit(code)
    _, inner_scope = visitor.tracker.all_scopes.popitem()
    inner_scope.global_names = {"x"}


@pytest.mark.parametrize("code", ["nonlocal x", "def func():\n  nonlocal x"])
def test_visit_Nonlocal_fail(code: str) -> None:
    with pytest.raises(ValueError, match="Nonlocal keyword must be used in nested scope"):
        parse_and_visit(code)


def test_visit_Nonlocal() -> None:
    code = """\
    def func():
        x = 1
        def inner():
            nonlocal x
    """
    visitor = parse_and_visit(code)
    _, inner_scope = visitor.tracker.all_scopes.popitem()
    inner_scope.nonlocal_names = {"x"}


@pytest.mark.parametrize(
    ("code", "expected_imports"),
    [
        ("import os", {"os": [("os", None)]}),
        ("import os, sys", {"os": [("os", None)], "sys": [("sys", None)]}),
        ("import sys as system", {"system": [("sys", "system")]}),
        ("import os, sys as system", {"os": [("os", None)], "system": [("sys", "system")]}),
        (
            "from collections import defaultdict",
            {"defaultdict": [("collections", "defaultdict", None)]},
        ),
        ("from pathlib import Path as P", {"P": [("pathlib", "Path", "P")]}),
        # Ensure tracker handles multiple imports added via visitor
        ("import os\nimport logging", {"os": [("os", None)], "logging": [("logging", None)]}),
        (
            "from a import b\nfrom c import d as e",
            {"b": [("a", "b", None)], "e": [("c", "d", "e")]},
        ),
        # redefining imports
        ("import os\nimport os", {"os": [("os", None), ("os", None)]}),
        ("import os\nimport sys as os", {"os": [("os", None), ("sys", "os")]}),
    ],
    ids=[
        "simple_import",
        "double_import",
        "import_as",
        "double_import_as",
        "from_import",
        "from_import_as",
        "multi_import",
        "multi_from_import",
        "redefining_import",
        "redefining_import_as",
    ],
)
def test_visit_Import_ImportFrom(
    code: str, expected_imports: dict[str, list[str | tuple[str, str]]]
) -> None:
    """Test that visiting imports updates the tracker correctly."""
    visitor = parse_and_visit(code)
    # Imports are added to the outermost scope
    outer_scope = visitor.tracker.scopes[0]
    assert outer_scope.imports == expected_imports

    # check the same for nested
    nested_code_lines = ["def func():"] + [f"  {line}" for line in code.splitlines()]
    nested_visitor = parse_and_visit(nested_code_lines)
    func_node = nested_visitor._visited_nodes[1]
    # assert outer scope is empty
    nested_outer_scope = nested_visitor.tracker.scopes[0]
    assert nested_outer_scope.imports == {}
    # assert inner scope is correct
    nested_inner_scope = nested_visitor.tracker.all_scopes[id(func_node)]
    assert nested_inner_scope.imports == expected_imports


def test_ImportFrom_star() -> None:
    with pytest.raises(ValueError, match="Star imports are not supported."):
        parse_and_visit("from test import *")


def test_ImportFrom_relative() -> None:
    with pytest.raises(ValueError, match="Relative import is deeper than module FQN."):
        parse_and_visit("from . import foo", "test_module")

    visitor = parse_and_visit("from . import foo\nfrom .other import bar", "pkg.test_module")
    outer_scope = visitor.tracker.scopes[0]
    assert outer_scope.imports == {
        "foo": [("pkg", "foo", None)],
        "bar": [("pkg.other", "bar", None)],
    }


@pytest.mark.parametrize(
    ("code", "expected"),
    [("x = 1", {"x"}), ("del x", {"x"}), ("x.a = 1", set())],
    ids=["simple_name", "simple_del", "store_attr"],
)
def test_visit_Name_store_del(code: str, expected: set[str]) -> None:
    """Test that visiting imports updates the tracker correctly."""
    visitor = parse_and_visit(code)
    # Imports are added to the outermost scope
    outer_scope = visitor.tracker.scopes[0]
    assert set(outer_scope.names) == expected

    # check the same for nested
    nested_code_lines = ["def func():"] + [f"  {line}" for line in code.splitlines()]
    nested_visitor = parse_and_visit(nested_code_lines)
    func_node = nested_visitor._visited_nodes[1]
    # assert outer scope is empty
    nested_outer_scope = nested_visitor.tracker.scopes[0]
    assert set(nested_outer_scope.names) == set()
    # assert inner scope is correct
    nested_inner_scope = nested_visitor.tracker.all_scopes[id(func_node)]
    assert set(nested_inner_scope.names) == expected


@pytest.mark.parametrize(
    ("code", "expected", "dep_graph", "unresolved"),
    [
        ("x = 1\nx", {"x"}, {ModDef: {XDef}, XDef: set()}, []),
        ("del x", {"x"}, {ModDef: set(), XDef: set()}, []),
        ("x", set(), {ModDef: set()}, ["x"]),
        ("x.a = 1\nx.a", set(), {ModDef: set()}, []),
        ("x.a.b = 1\nx.a.b", set(), {ModDef: set()}, []),
        ("x.a", set(), {ModDef: set()}, ["x"]),
        ("x.a.b", set(), {ModDef: set()}, ["x"]),
        ("x = 1\nx.a", {"x"}, {ModDef: {XDef}, XDef: set()}, []),
        ("x = 1\nx.a.b", {"x"}, {ModDef: {XDef}, XDef: set()}, []),
        ("import os\nos.path", set(), {ModDef: {OsImp, Import(("os", "path"), None)}}, []),
        (
            "import os\nx = 1\nx\nos.path",
            {"x"},
            {ModDef: {OsImp, XDef, Import(("os", "path"), None)}, XDef: set()},
            [],
        ),
        ("import importlib.util\nimportlib.spec", set(), {ModDef: set()}, []),
    ],
    ids=[
        "just_load",
        "just_del",
        "attr_load",
        "attr_load_deeper",
        "unknown_attr",
        "unknown_attr_deeper",
        "attr_load_defined",
        "attr_load_defined_deeper",
        "simple_unknown",
        "imported",
        "imported_and_load",
        "other_import",
    ],
)
def test_visit_Name_Attribute_load(
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

    # if the import is kept
    func_deps = (
        {OsImp, Import(("os", "path"), None)}
        if any(OsImp in v for v in dep_graph.values())
        else set()
    )
    assert nested_visitor.dep_graph == {ModDef: set(), FuncDef: func_deps}


def test_visit_Name_global() -> None:
    code = """\
    def func():
        global x
        x += 1
    """
    visitor = parse_and_visit(code)
    outer_scope = visitor.tracker.scopes[0]
    assert set(outer_scope.names) == {"x"}
    assert visitor.dep_graph == {ModDef: set(), XDef: set(), FuncDef: {XDef}}


def test_visit_Name_nonlocal() -> None:
    code = """\
    def func():
        def inner():
            nonlocal x
            x = 1
    """
    visitor = parse_and_visit(code)
    func_node = visitor._visited_nodes[1]
    func_scope = visitor.tracker.all_scopes[id(func_node)]
    assert set(func_scope.names) == {"x"}
    assert set(func_scope.functions) == {"inner"}
    assert visitor.dep_graph == {ModDef: set(), FuncDef: set()}


@pytest.mark.parametrize("code", ["""__all__ = ["x"]""", """__all__: list[str] = ["x"]"""])
def test_handle_all(code: str) -> None:
    visitor = parse_and_visit(f"x=3\n{code}")
    assert visitor.all == ["x"]
    assert visitor.dep_graph == {ModDef: set(), AllDef: set(), XDef: set()}

    visitor = parse_and_visit(f"import x\n{code}")
    assert visitor.all == ["x"]
    assert visitor.dep_graph == {ModDef: set(), AllDef: set(), XDef: {Import("x")}}

    with pytest.raises(ValueError, match="Name in __all__ neither defined nor an import"):
        parse_and_visit(code)


def test_handle_all_errors() -> None:
    with pytest.raises(TypeError, match=r"Expected a tuple or list of literal strings, got"):
        parse_and_visit("__all__ = [1, 2, 3]")

    with pytest.raises(ValueError, match="No value for __all__ assignment."):
        parse_and_visit("__all__: list[str]")

    with pytest.raises(ValueError, match="Unsupported __all__ definition."):
        parse_and_visit("__all__ += []")

    with pytest.raises(ValueError, match="Unsupported __all__ definition."):
        parse_and_visit("__all__, other = []")

    with pytest.raises(ValueError, match="__all__ must be defined at the module level."):
        parse_and_visit("def func():\n  __all__ = []")


def test_AugAssign() -> None:
    visitor = parse_and_visit("""x = 1""")
    assert visitor.dep_graph == {ModDef: set(), XDef: set()}

    # is also loaded with aug assign
    visitor = parse_and_visit("""x += 1""")
    assert visitor.dep_graph == {ModDef: set(), XDef: {XDef}}
    # TODO(tihoph): should raise error if x not in scope?

    visitor = parse_and_visit("x = 1\nx += 1")
    assert visitor.dep_graph == {ModDef: set(), XDef: {XDef}}

    visitor = parse_and_visit("x = 1\nx.a += 1")
    assert visitor.dep_graph == {ModDef: {XDef}, XDef: set()}

    visitor = parse_and_visit("x = 1\nx.y.a += 1")
    assert visitor.dep_graph == {ModDef: {XDef}, XDef: set()}


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


def test_visit_Call(caplog: pytest.LogCaptureFixture) -> None:
    """Test visiting calls handles scope and visits components correctly."""
    code = """\
    def func(a, b, c):
       pass
    x = 1
    """
    visitor = parse_and_visit(code)
    # function bodies are deferred
    assert visited_before(visitor, ast.Assign, ast.Pass)

    code = """\
    def func(a, b, c):
       pass
    func(1, 2, 3)
    x: int = 1
    """
    visitor = parse_and_visit(code)
    # function bodies are visited on calls
    assert visited_before(visitor, ast.Pass, ast.AnnAssign)

    with caplog.at_level(logging.DEBUG):
        visitor = parse_and_visit("""print("Test")""")
    assert "Name print is a built-in, skipping visit" in caplog.text

    with caplog.at_level(logging.DEBUG):
        visitor = parse_and_visit("""unknown("Hi")""")
    assert "Function unknown not found in current scope" in caplog.text

    # TODO(tihoph): check the branch
    visitor = parse_and_visit("attr()()")
    visitor = parse_and_visit("attr.call()")
    # attr and calls being called is just passed


@pytest.mark.parametrize(
    ("code", "log_text"),
    [
        ("def func(): pass\nfunc()", "Visiting resolved function func"),
        ("def func(): pass\ndef func2(): pass\nfunc2()", "Visiting resolved function func2"),
        ("def func(): pass\nfunc()\nfunc()", "has already been visited, skipping"),
        ("def func():\n  def inner(): pass\n  inner()", "Visiting resolved function inner"),
        ("print()", "is a built-in, skipping visit"),
        ("unknown()", "not found in current scope"),
    ],
)
def test_resolve_and_visit(code: str, log_text: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        parse_and_visit(code)
    assert log_text in caplog.text


def test_visit_deferred() -> None:
    """Test FunctionBodyWrapper accept method."""
    tree = ast.parse("def func():\n  def inner(): pass\ndef func2(): pass")
    add_parents(tree)
    func_node: ast.FunctionDef = tree.body[0]  # type: ignore[assignment]
    inner_node: ast.FunctionDef = func_node.body[0]  # type: ignore[assignment]
    func2_node: ast.FunctionDef = tree.body[1]  # type: ignore[assignment]
    visitor = ScopeVisitor("test_mod", debug=True)
    wrapper = FunctionBodyWrapper(func_node, visitor.tracker)
    wrapper2 = FunctionBodyWrapper(func2_node, visitor.tracker)

    visitor.deferred.append(wrapper)  # Add to deferred list
    visitor.deferred.append(wrapper2)  # Add another deferred function
    assert visitor.deferred == deque([wrapper, wrapper2])

    visitor.visit_deferred()

    assert len(visitor.deferred) == 0  # Check if inner function was added
    assert visitor.tracker.visited_funcs == [id(func_node), id(inner_node), id(func2_node)]
    inner_wrapper = visitor._visited_nodes[2]
    assert visitor._visited_nodes == [
        wrapper,
        inner_node,
        inner_wrapper,
        inner_node.body[0],
        wrapper2,
        wrapper2.function.body[0],  # type: ignore[index]
    ]


def test_function_body_wrapper_init() -> None:
    """Test FunctionBodyWrapper initialization."""
    tree = ast.parse("def func(): pass")
    add_parents(tree)
    func_node: ast.FunctionDef = tree.body[0]  # type: ignore[assignment]
    tracker = ScopeTracker()
    wrapper = FunctionBodyWrapper(func_node, tracker)
    assert wrapper.function == func_node
    assert wrapper.custom_name == "func"
    assert wrapper.custom_parent == tree
    assert wrapper.tracker == tracker


def test_function_body_wrapper_accept() -> None:
    """Test FunctionBodyWrapper accept method."""
    tree = ast.parse("def func():\n  def inner(): pass")
    add_parents(tree)
    func_node: ast.FunctionDef = tree.body[0]  # type: ignore[assignment]
    inner_node: ast.FunctionDef = func_node.body[0]  # type: ignore[assignment]
    visitor = ScopeVisitor("test_mod", debug=True)
    wrapper = FunctionBodyWrapper(func_node, visitor.tracker)
    visitor.deferred.append(wrapper)  # Add to deferred list
    assert visitor.deferred == deque([wrapper])
    wrapper = visitor.deferred.popleft()
    visitor.tracker.mark_visited(wrapper.function)
    wrapper.accept(visitor)  # Call accept method

    # assert that the inner func was added to new deferred
    assert len(visitor.deferred) == 0
    # TODO(tihoph): Check if args were added to scope etc.

    inner_wrapper = visitor._visited_nodes[1]
    assert visitor.tracker.visited_funcs == [id(func_node), id(inner_node)]
    assert visitor._visited_nodes == [inner_node, inner_wrapper, inner_node.body[0]]
