"""Test deprive.scope."""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Literal

import pytest

from deprive.scope import Scope, ScopeTracker


def _make_name(name: str = "x") -> ast.Name:
    """Create a name node."""
    return ast.parse(name).body[0].value  # type: ignore[attr-defined,no-any-return]


def _make_func(name: str = "func") -> ast.FunctionDef:
    """Create a function node with the given name."""
    return ast.parse(f"def {name}(): pass").body[0]  # type: ignore[return-value]


def _make_assign(annotated: bool = False) -> ast.Assign:
    """Create an assignment node."""
    if annotated:
        return ast.parse("x: int = 1").body[0]  # type: ignore[return-value]
    return ast.parse("x = 1").body[0]  # type: ignore[return-value]


def add_parents(node: ast.AST, parent: ast.AST | None = None) -> None:
    """Recursively add parent pointers to an AST."""
    node.parent = parent  # type: ignore[attr-defined]
    for child in ast.iter_child_nodes(node):
        add_parents(child, node)


def test_init() -> None:
    """Test ScopeTracker initialization."""
    tracker = ScopeTracker()
    assert len(tracker.scopes) == 1
    assert tracker.scopes[0] == Scope()
    assert tracker.visited_funcs == []
    assert tracker.all_nodes == {}
    assert tracker.all_scopes == {}


@pytest.mark.parametrize(
    ("setup_scopes", "name_to_check", "expected"),
    [
        # Test empty scope
        ([], "x", None),
        # Test name in outermost scope only
        ([{"names": {"x": _make_name()}}], "x", "outermost"),
        ([{"imports": {"y": "mod_y"}}], "y", "outermost"),
        ([{"imports_from": {"z": ("mod_z", "z_orig")}}], "z", "outermost"),
        ([{"functions": {"f": _make_func("f")}}], "f", "outermost"),
        # Test name not present
        ([{"names": {"x": _make_name()}}], "y", None),
        # Test name in inner scope
        ([{"names": {"x": _make_name()}}, {"names": {"y": _make_name("y")}}], "y", "inner"),
        # Test name shadowed in inner scope (should report inner)
        ([{"names": {"x": _make_name()}}, {"names": {"x": _make_name()}}], "x", "inner"),
        # Test name only in outer scope when checking from inner
        ([{"names": {"x": _make_name()}}, {"names": {"y": _make_name("y")}}], "x", "outermost"),
        # Test complex nesting
        (
            [
                {"imports": {"a": "mod_a"}},
                {"names": {"b": _make_name("b")}},
                {"functions": {"a": _make_func("a")}},
            ],
            "a",
            "inner",
        ),  # Function 'a' shadows import 'a'
        (
            [
                {"imports": {"a": "mod_a"}},
                {"names": {"b": _make_name("b")}},
                {"functions": {"c": _make_func("c")}},
            ],
            "a",
            "outermost",
        ),  # Import 'a' visible in inner scope
        (
            [
                {"imports": {"a": "mod_a"}},
                {"names": {"b": _make_name("b")}},
                {"functions": {"c": _make_func("c")}},
            ],
            "b",
            "inner",
        ),
        (
            [
                {"imports": {"a": "mod_a"}},
                {"names": {"b": _make_name("b")}},
                {"functions": {"c": _make_func("c")}},
            ],
            "c",
            "inner",
        ),
        (
            [
                {"imports": {"a": "mod_a"}},
                {"names": {"b": _make_name("b")}},
                {"functions": {"c": _make_func("c")}},
            ],
            "d",
            None,
        ),
    ],
    ids=[
        "empty",
        "outer_name",
        "outer_import",
        "outer_import_from",
        "outer_function",
        "not_present",
        "inner_name",
        "inner_shadows_outer",
        "outer_visible_from_inner",
        "complex_shadowing",
        "complex_outer_visible",
        "complex_inner_name",
        "complex_inner_func",
        "complex_not_present",
    ],
)
def test_is_in(
    setup_scopes: list[dict[str, Any]],
    name_to_check: str,
    expected: Literal["outermost", "inner"] | None,
) -> None:
    """Test ScopeTracker.is_in method."""
    tracker = ScopeTracker()
    tracker.scopes = []  # Clear initial scope
    for scope_data in setup_scopes:
        scope = Scope(
            imports=scope_data.get("imports", {}),
            imports_from=scope_data.get("imports_from", {}),
            functions=scope_data.get("functions", {}),
            names=scope_data.get("names", {}),
        )
        tracker.scopes.append(scope)

    # Ensure at least one scope exists if setup is empty
    if not tracker.scopes:
        tracker.scopes.append(Scope())

    inner_expected = expected == "inner"
    all_expected = expected is not None
    assert tracker.is_in(name_to_check) == all_expected
    assert tracker.is_in(name_to_check, inner_only=True) == inner_expected


@pytest.mark.parametrize(
    ("setup_scopes", "name_to_check", "expected"),
    [
        ([], "x", None),  # Empty scope
        ([{"imports": {"x": "mod_x"}}], "x", "mod_x"),  # Direct import
        ([{"imports_from": {"y": ("mod_y", "y_orig")}}], "y", ("mod_y", "y_orig")),  # From import
        ([{"names": {"z": _make_name("z")}}], "z", None),  # Regular name
        ([{"functions": {"f": _make_func("f")}}], "f", None),  # Function name
        # Check across scopes
        (
            [{"imports": {"x": "mod_x"}}, {"names": {"y": _make_name("y")}}],
            "x",
            "mod_x",
        ),  # Import in outer
        (
            [{"names": {"x": _make_name()}}, {"imports": {"y": "mod_y"}}],
            "y",
            "mod_y",
        ),  # Import in inner
        (
            [{"imports_from": {"x": ("mod_x", "x_orig")}}, {"names": {"y": _make_name("y")}}],
            "x",
            ("mod_x", "x_orig"),
        ),  # ImportFrom in outer
        (
            [{"names": {"x": _make_name()}}, {"imports_from": {"y": ("mod_y", "y_orig")}}],
            "y",
            ("mod_y", "y_orig"),
        ),  # ImportFrom in inner
        # Shadowing (is_import checks if *any* scope defines it as import)
        ([{"imports": {"x": "mod_x"}}, {"names": {"x": _make_name()}}], "x", "mod_x"),
        ([{"names": {"x": _make_name()}}, {"imports": {"x": "mod_x"}}], "x", "mod_x"),
    ],
    ids=[
        "empty",
        "direct_import",
        "from_import",
        "regular_name",
        "function_name",
        "import_in_outer",
        "import_in_inner",
        "import_from_in_outer",
        "import_from_in_inner",
        "shadowed_by_name",
        "shadowed_by_import",
    ],
)
def test_is_import(
    setup_scopes: list[dict[str, Any]], name_to_check: str, expected: str | None
) -> None:
    """Test ScopeTracker.is_import method."""
    tracker = ScopeTracker()
    tracker.scopes = []  # Clear initial scope
    for scope_data in setup_scopes:
        scope = Scope(
            imports=scope_data.get("imports", {}),
            imports_from=scope_data.get("imports_from", {}),
            functions=scope_data.get("functions", {}),
            names=scope_data.get("names", {}),
        )
        tracker.scopes.append(scope)

    if not tracker.scopes:
        tracker.scopes.append(Scope())

    assert tracker.is_import(name_to_check) == expected


@pytest.mark.parametrize(
    ("code", "target_node_path", "module_name", "expected_fqn_pattern"),
    [
        (
            "x = 1",
            [0, 0],  # Path to the Name node 'x' within the Assign node's targets
            "my_module",
            r"my_module\.x",
        ),
        (
            "def my_func():\n  pass",
            [0],  # Path to the FunctionDef node
            "my_package.my_mod",
            r"my_package\.my_mod\.my_func",
        ),
        (
            "def my_func():\n  y = 2",
            [0, 0, 0],  # Path to Name node 'y' inside Assign inside FunctionDef
            "another_module",
            r"another_module\.my_func\.y",
        ),
        (
            "def outer():\n  def inner():\n    z = 3",
            [0, 0, 0, 0],  # Path to Name node 'z' inside Assign inside inner FunctionDef
            "nested.mod",
            r"nested\.mod\.outer\.inner\.z",
        ),
        (
            "def comp_func():\n  vals = [x for x in range(10)]",
            [0, 0, 1, 0],  # Path to the Name node 'x' (the target in the comprehension)
            "comp_mod",
            # Expecting placeholder ID for comprehension
            r"comp_mod\.comp_func\.vals\.<\d+>\.<\d+>",
        ),
        (
            "def lambda_func():\n  f = lambda y: y + 1",
            [0, 0, 1, 0],  # Path to the Name node 'y' (arg in lambda)
            "lambda_mod",
            r"lambda_mod\.lambda_func\.f\.<\d+>\.<\d+>",  # Expecting placeholder ID for lambda
        ),
        (
            "import os",
            [0],  # Path to the Import node
            "import_test",
            r"import_test\.<\d+>",  # imports have no name, expect placeholder ID
        ),
        (
            "from sys import argv",
            [0],  # Path to the ImportFrom node
            "import_from_test",
            r"import_from_test\.<\d+>",  # imports have no name, expect placeholder ID
        ),
    ],
    ids=[
        "module_var",
        "module_func",
        "func_var",
        "nested_func_var",
        "comprehension_target",
        "lambda_arg",
        "import_node",
        "import_from_node",
    ],
)
def test_build_fqn(
    code: str, target_node_path: list[int], module_name: str, expected_fqn_pattern: str
) -> None:
    """Test ScopeTracker.build_fqn method."""
    tracker = ScopeTracker()
    module_node = ast.parse(code)

    add_parents(module_node)
    module_node.custom_name = module_name  # type: ignore[attr-defined]

    # Find the target node using the path
    target_node: ast.AST = module_node
    for index in target_node_path:
        # Need to handle different ways children are stored
        if isinstance(target_node, ast.Assign):
            # Path might point to target or value, assume target if first element
            target_node = target_node.targets[index] if index == 0 else target_node.value
        elif isinstance(target_node, ast.Lambda):
            target_node = (
                target_node.body if index > 0 else target_node.args.args[index]
            )  # Simplified: assumes path to args or body
        elif isinstance(target_node, ast.ListComp):
            # Simplified: path to elt or target in first generator
            target_node = target_node.elt if index == 0 else target_node.generators[0].target
        else:
            target_node = target_node.body[index]  # type: ignore[attr-defined]

    fqn = tracker.build_fqn(target_node)

    assert fqn is not None
    assert re.match(expected_fqn_pattern, fqn) is not None, (
        f"FQN '{fqn}' did not match expected pattern '{expected_fqn_pattern}'"
    )


def test_scope_context_manager() -> None:
    """Test the scope context manager."""
    tracker = ScopeTracker()
    node1 = _make_name()
    node2 = _make_name("y")
    func_node = _make_func("inner_func")

    assert len(tracker.scopes) == 1
    outer_scope = tracker.scopes[0]
    assert id(node1) not in tracker.all_nodes
    assert id(node1) not in tracker.all_scopes

    with tracker.scope(node1):
        assert len(tracker.scopes) == 2
        inner_scope1 = tracker.scopes[1]
        assert inner_scope1 is not outer_scope
        assert tracker.all_nodes[id(node1)] is node1
        assert tracker.all_scopes[id(node1)] is inner_scope1

        # Add something to the inner scope
        tracker.add_func("inner_func", func_node)
        assert "inner_func" in inner_scope1.functions
        assert "inner_func" not in outer_scope.functions

        # Nested scope
        with tracker.scope(node2):
            assert len(tracker.scopes) == 3
            inner_scope2 = tracker.scopes[2]
            assert tracker.all_nodes[id(node2)] is node2
            assert tracker.all_scopes[id(node2)] is inner_scope2
            assert inner_scope2 is not inner_scope1

        # Check scope popped correctly
        assert len(tracker.scopes) == 2
        assert tracker.scopes[-1] is inner_scope1

    # Check scope popped correctly
    assert len(tracker.scopes) == 1
    assert tracker.scopes[0] is outer_scope
    assert id(node1) in tracker.all_nodes  # Nodes/scopes remain tracked
    assert id(node1) in tracker.all_scopes
    assert id(node2) in tracker.all_nodes
    assert id(node2) in tracker.all_scopes


def test_push_pop() -> None:
    """Test push and pop methods directly."""
    tracker = ScopeTracker()
    node1 = _make_name()
    node2 = _make_name("y")

    initial_scope = tracker.scopes[0]
    assert len(tracker.scopes) == 1

    # Push first scope
    tracker.push(node1)
    assert len(tracker.scopes) == 2
    scope1 = tracker.scopes[1]
    assert scope1 is not initial_scope
    assert tracker.all_nodes == {id(node1): node1}
    assert tracker.all_scopes == {id(node1): scope1}

    # Push second scope
    tracker.push(node2)
    assert len(tracker.scopes) == 3
    scope2 = tracker.scopes[2]
    assert scope2 is not scope1
    assert tracker.all_nodes == {id(node1): node1, id(node2): node2}
    assert tracker.all_scopes == {id(node1): scope1, id(node2): scope2}

    # Pop second scope
    tracker.pop()
    assert len(tracker.scopes) == 2
    assert tracker.scopes[-1] is scope1

    # Pop first scope
    tracker.pop()
    assert len(tracker.scopes) == 1
    assert tracker.scopes[-1] is initial_scope

    # Test pushing existing node raises error
    tracker = ScopeTracker()
    tracker.push(node1)  # Push it once
    with pytest.raises(ValueError, match=f"Scope for node {node1} already exists"):
        tracker.push(node1)  # Try pushing again


def test_add_func() -> None:
    """Test adding a function to the current scope."""
    tracker = ScopeTracker()
    func_node = _make_func("my_func")

    tracker.add_func("my_func", func_node)

    assert len(tracker.scopes) == 1
    current_scope = tracker.scopes[0]
    assert current_scope.functions == {"my_func": func_node}
    assert "my_func" not in current_scope.names  # Should not add to names dict

    # Test in inner scope
    node1 = _make_name()
    func_node2 = _make_func("inner_func")
    with tracker.scope(node1):
        tracker.add_func("inner_func", func_node2)
        assert len(tracker.scopes) == 2
        inner_scope = tracker.scopes[1]
        assert inner_scope.functions == {"inner_func": func_node2}
        assert "inner_func" not in inner_scope.names
        # Outer scope should be unchanged
        assert current_scope.functions == {"my_func": func_node}


@pytest.mark.parametrize(
    ("name", "node", "initial_imports", "expected_names", "expected_exception"),
    [
        ("x", _make_assign(), {}, {"x": Ellipsis}, None),  # Single name
        (("y", "z"), _make_assign(), {}, {"y": Ellipsis, "z": Ellipsis}, None),  # Tuple of names
        (
            "a",
            _make_assign(annotated=True),
            {},
            {"a": Ellipsis},
            None,
        ),  # Single name, different node
        (None, _make_assign(), {}, {}, None),  # None name, should do nothing
        (
            "os",
            _make_assign(),
            {"imports": {"os": "os"}},
            {},
            NotImplementedError,
        ),  # Name conflicts with import
        (
            "m_path",
            _make_assign(),
            {"imports_from": {"m_path": ("os", "path")}},
            {},
            NotImplementedError,
        ),  # Name conflicts with import from
    ],
    ids=[
        "single_name",
        "tuple_name",
        "different_node",
        "none_name",
        "conflict_import",
        "conflict_import_from",
    ],
)
def test_add_name(
    name: str | tuple[str, ...] | None,
    node: ast.AST,
    initial_imports: dict[str, dict[str, Any]],
    expected_names: dict[str, ast.AST],
    expected_exception: type[Exception] | None,
) -> None:
    """Test adding names to the current scope."""
    tracker = ScopeTracker()
    # Set up initial imports if needed for conflict testing
    if "imports" in initial_imports:
        tracker.scopes[0].imports = initial_imports["imports"]
    if "imports_from" in initial_imports:
        tracker.scopes[0].imports_from = initial_imports["imports_from"]

    if expected_exception:
        with pytest.raises(expected_exception, match="Redefining imports is not supported yet"):
            tracker.add_name(name, node)
    else:
        tracker.add_name(name, node)
        current_scope = tracker.scopes[-1]
        # Use Ellipsis to check for presence and avoid exact node comparison if not needed
        assert len(current_scope.names) == len(expected_names)
        for n in expected_names:
            assert n in current_scope.names
            # Can optionally add assert current_scope.names[n] is node if needed


def test_resolve_func() -> None:
    """Test resolving function names across scopes."""
    tracker = ScopeTracker()
    func1_outer = _make_func("f1")
    func1_inner = _make_func("f1")
    func2_inner = _make_func("f2")
    scope_node = _make_name()

    # Add f1 to outer scope
    tracker.add_func("f1", func1_outer)
    assert tracker.resolve_func("f1") is func1_outer
    assert tracker.resolve_func("f2") is None
    assert tracker.resolve_func("f3") is None

    # Enter inner scope
    with tracker.scope(scope_node):
        # Add f2 and shadowed f1 to inner scope
        tracker.add_func("f2", func2_inner)
        tracker.add_func("f1", func1_inner)

        # Resolve from inner scope
        assert tracker.resolve_func("f1") is func1_inner  # Inner shadows outer
        assert tracker.resolve_func("f2") is func2_inner
        assert tracker.resolve_func("f3") is None

    # Back in outer scope
    assert tracker.resolve_func("f1") is func1_outer  # Original outer f1
    assert tracker.resolve_func("f2") is None  # f2 was only in inner scope
    assert tracker.resolve_func("f3") is None


def test_visited_funcs() -> None:
    """Test marking and checking visited functions."""
    tracker = ScopeTracker()
    func1 = _make_func("func1")
    func2: ast.FunctionDef = ast.parse("async def func2(): pass").body[0]  # type: ignore[assignment]

    assert not tracker.is_visited(func1)
    assert not tracker.is_visited(func2)
    assert tracker.visited_funcs == []

    tracker.mark_visited(func1)

    assert tracker.is_visited(func1)
    assert not tracker.is_visited(func2)
    assert tracker.visited_funcs == [id(func1)]

    tracker.mark_visited(func2)

    assert tracker.is_visited(func1)
    assert tracker.is_visited(func2)
    assert tracker.visited_funcs == [id(func1), id(func2)]


def test_add_import() -> None:
    """Test add_import delegates correctly."""
    tracker = ScopeTracker()

    # Test ast.Import
    node_import = ast.parse("import os as myos").body[0]
    assert isinstance(node_import, ast.Import)
    tracker.add_import(node_import)
    assert tracker.scopes[-1].imports == {"myos": "os"}
    assert tracker.scopes[-1].imports_from == {}

    # Reset scope imports for next test
    tracker = ScopeTracker()

    # Test ast.ImportFrom
    node_import_from = ast.parse("from sys import argv as a").body[0]
    assert isinstance(node_import_from, ast.ImportFrom)
    tracker.add_import(node_import_from)
    assert tracker.scopes[-1].imports == {}
    assert tracker.scopes[-1].imports_from == {"a": ("sys", "argv")}


@pytest.mark.parametrize(
    ("code", "expected", "log_check"),
    [
        # Success cases
        ("import os", {"imports": {"os": "os"}}, ("Found import:", "import os as os")),
        (
            "import sys as system",
            {"imports": {"system": "sys"}},
            ("Found import:", "import sys as system"),
        ),
        # Error cases
        ("import os, sys", (NotImplementedError, "Multiple imports.*not supported"), None),
        (
            "import os\nimport os",
            (NotImplementedError, "Redefining imports.*not supported"),
            None,
        ),  # Requires adding import first
        (
            "import sys\nimport os as sys",
            (NotImplementedError, "Redefining imports.*not supported"),
            None,
        ),  # Requires adding import first
    ],
    ids=["simple", "alias", "error_multiple", "error_redefine_direct", "error_redefine_alias"],
)
def test_handle_import(
    code: str,
    expected: dict[str, str] | tuple[type[Exception], str],
    log_check: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test ScopeTracker._handle_import method."""
    # Note: Redefining tests require manual setup before the tested call
    tracker = ScopeTracker()
    node = ast.parse(code).body[0]
    assert isinstance(node, ast.Import)

    if "Redefining" in str(expected):
        # Pre-add the conflicting import
        conflicting_name = "os" if "import os\nimport os" in code else "sys"
        tracker.scopes[-1].imports[conflicting_name] = (
            "some_module"  # Value doesn't strictly matter here
        )

    if isinstance(expected, tuple) and issubclass(expected[0], Exception):
        exc, msg = expected
        with pytest.raises(exc, match=msg):
            tracker._handle_import(node)  # noqa: SLF001
        return

    assert isinstance(expected, dict)
    with caplog.at_level(logging.DEBUG):
        tracker._handle_import(node)  # noqa: SLF001
    current_scope = tracker.scopes[-1]
    assert current_scope.imports == expected.get("imports", {})
    assert current_scope.imports_from == expected.get("imports_from", {})
    assert current_scope.functions == {}
    assert current_scope.names == {}
    assert log_check is not None  # should already be returned at error branch
    assert log_check[0] in caplog.text
    assert log_check[1] in caplog.text


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # Success cases
        ("from os import path", {"imports_from": {"path": ("os", "path")}}),
        ("from sys import argv as a", {"imports_from": {"a": ("sys", "argv")}}),
        (
            "from os import path, environ",
            {"imports_from": {"path": ("os", "path"), "environ": ("os", "environ")}},
        ),
        (
            "from os import path, environ as env",
            {"imports_from": {"path": ("os", "path"), "env": ("os", "environ")}},
        ),
        # Error cases
        ("from . import mymod", (NotImplementedError, "Relative imports.*not supported")),
        ("from .mymod import myfunc", (NotImplementedError, "Relative imports.*not supported")),
        ("from .. import mymod", (NotImplementedError, "Relative imports.*not supported")),
        ("from os import *", (NotImplementedError, "Star imports.*not supported")),
        (
            "from os import path\nfrom sys import path",
            (NotImplementedError, "Redefining imports.*not supported"),
        ),  # Requires adding import first
        (
            "from os import path\nfrom sys import argv as path",
            (NotImplementedError, "Redefining imports.*not supported"),
        ),  # Requires adding import first
    ],
    ids=[
        "simple",
        "alias",
        "multiple",
        "multiple_with_alias",
        "error_relative_dot",
        "error_relative_dot_mod",
        "error_relative_dot_dot",
        "error_star",
        "error_redefine_direct",
        "error_redefine_alias",
    ],
)
def test_handle_import_from(
    code: str,
    expected: dict[str, dict[str, str] | dict[str, tuple[str, str]]] | tuple[type[Exception], str],
) -> None:
    """Test ScopeTracker._handle_import_from method."""
    # Note: Redefining tests require manual setup before the tested call
    tracker = ScopeTracker()
    node = ast.parse(code).body[-1]
    assert isinstance(node, ast.ImportFrom)

    if "Redefining" in str(expected):
        # Pre-add the conflicting import
        tracker.scopes[-1].imports_from["path"] = (
            "some_module",
            "some_name",
        )  # Value doesn't strictly matter

    if isinstance(expected, tuple) and issubclass(expected[0], Exception):
        exc, msg = expected
        with pytest.raises(exc, match=msg):
            tracker._handle_import_from(node)  # noqa: SLF001
        return

    assert isinstance(expected, dict)
    tracker._handle_import_from(node)  # noqa: SLF001
    current_scope = tracker.scopes[-1]
    # Custom comparison for imports_from value tuple
    expected_imports_from = expected.get("imports_from", {})
    assert current_scope.imports_from.keys() == expected_imports_from.keys()
    for key, val in expected_imports_from.items():
        assert current_scope.imports_from[key] == val
    assert current_scope.imports == expected.get("imports", {})
    assert current_scope.functions == {}
    assert current_scope.names == {}
