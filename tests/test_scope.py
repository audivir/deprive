"""Test Scope utilities."""

# ruff: noqa: N802
from __future__ import annotations

import ast
import re
from typing import Any, Literal

import pytest

from deprive.scope import Scope, ScopeTracker, add_parents


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


def test_Scope_init() -> None:
    """Test Scope initialization."""
    scope = Scope()
    assert scope.imports == {}
    assert scope.functions == {}
    assert scope.names == {}
    assert scope.global_names == set()
    assert scope.nonlocal_names == set()
    assert scope.fields == ({}, {}, {})
    func_node = _make_func()
    scope = Scope(
        imports={"a": [("b", "a")], "c": [("d", "e", "c")]},
        functions={"f": func_node},
        names={"h": None},
        global_names={"j"},
        nonlocal_names={"k"},
    )
    assert scope.imports == {"a": [("b", "a")], "c": [("d", "e", "c")]}
    assert scope.functions == {"f": func_node}
    assert scope.names == {"h": None}
    assert scope.global_names == {"j"}
    assert scope.nonlocal_names == {"k"}
    assert scope.fields == (
        {"a": [("b", "a")], "c": [("d", "e", "c")]},
        {"f": func_node},
        {"h": None},
    )


def test_ScopeTracker_init() -> None:
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
        ([{"imports": {"y": [("mod_y", "y")]}}], "y", "outermost"),
        ([{"imports": {"z": [("mod_z", "z_orig", "z")]}}], "z", "outermost"),
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
                {"imports": {"a": [("mod_a", "a")]}},
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
    expected: Literal["inner", "outermost"] | None,
) -> None:
    """Test ScopeTracker.is_in method."""
    tracker = ScopeTracker()
    tracker.scopes = []  # Clear initial scope
    for scope_data in setup_scopes:
        scope = Scope(
            imports=scope_data.get("imports", {}),
            functions=scope_data.get("functions", {}),
            names=scope_data.get("names", {}),
        )
        tracker.scopes.append(scope)

    # Ensure at least one scope exists if setup is empty
    if not tracker.scopes:
        tracker.scopes.append(Scope())

    all_expected = expected is not None
    inner_expected = expected == "inner"
    assert tracker.is_in(name_to_check) == all_expected
    assert tracker.is_in(name_to_check, inner_only=True) == inner_expected
    assert tracker.is_local(name_to_check) == inner_expected

    if inner_expected:
        tracker.current_scope.global_names.add(name_to_check)
        assert not tracker.is_local(name_to_check)


@pytest.mark.parametrize(
    ("setup_scopes", "name_to_check", "expected"),
    [
        ([], "x", None),  # Empty scope
        ([{"imports": {"x": [("mod_x", "x")]}}], "x", [("mod_x", "x")]),  # Direct import
        (
            [{"imports": {"y": [("mod_y", "y_orig", "y")]}}],
            "y",
            [("mod_y", "y_orig", "y")],
        ),  # From import
        ([{"names": {"z": _make_name("z")}}], "z", None),  # Regular name
        ([{"functions": {"f": _make_func("f")}}], "f", None),  # Function name
        # Check across scopes
        (
            [{"imports": {"x": [("mod_x", "x")]}}, {"names": {"y": _make_name("y")}}],
            "x",
            [("mod_x", "x")],
        ),  # Import in outer
        (
            [{"names": {"x": _make_name()}}, {"imports": {"y": [("mod_y", "y")]}}],
            "y",
            [("mod_y", "y")],
        ),  # Import in inner
        (
            [{"imports": {"x": [("mod_x", "x_orig", "x")]}}, {"names": {"y": _make_name("y")}}],
            "x",
            [("mod_x", "x_orig", "x")],
        ),  # ImportFrom in outer
        (
            [{"names": {"x": _make_name()}}, {"imports": {"y": [("mod_y", "y_orig", "y")]}}],
            "y",
            [("mod_y", "y_orig", "y")],
        ),  # ImportFrom in inner
        # Shadowing (is_import checks if *any* scope defines it as import)
        (
            [{"imports": {"x": [("mod_x", "x")]}}, {"names": {"x": _make_name()}}],
            "x",
            [("mod_x", "x")],
        ),
        (
            [{"names": {"x": _make_name()}}, {"imports": {"x": [("mod_x", "x")]}}],
            "x",
            [("mod_x", "x")],
        ),
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
    setup_scopes: list[dict[str, Any]],
    name_to_check: str,
    expected: list[tuple[str, str | None] | tuple[str, str, str | None]] | None,
) -> None:
    """Test ScopeTracker.is_import method."""
    tracker = ScopeTracker()
    tracker.scopes = []  # Clear initial scope
    for scope_data in setup_scopes:
        scope = Scope(
            imports=scope_data.get("imports", {}),
            functions=scope_data.get("functions", {}),
            names=scope_data.get("names", {}),
        )
        tracker.scopes.append(scope)

    if not tracker.scopes:
        tracker.scopes.append(Scope())

    assert tracker.is_import(name_to_check) == expected
    outer_expected = expected if setup_scopes and "imports" in setup_scopes[0] else None
    assert tracker.is_import(name_to_check, outer_only=True) == outer_expected


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
        assert tracker.current_scope is inner_scope1

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
    assert tracker.current_scope is scope1

    # Pop first scope
    tracker.pop()
    assert len(tracker.scopes) == 1
    assert tracker.current_scope is initial_scope

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
    ("name", "node", "initial_imports", "expected_names"),
    [
        ("x", _make_assign(), {}, {"x": Ellipsis}),  # Single name
        (("y", "z"), _make_assign(), {}, {"y": Ellipsis, "z": Ellipsis}),  # Tuple of names
        ("a", _make_assign(annotated=True), {}, {"a": Ellipsis}),  # Single name, different node
        (None, _make_assign(), {}, {}),  # None name, should do nothing
        ("os", _make_assign(), {"os": [("os", None)]}, {"os"}),
        ("m_path", _make_assign(), {"m_path": [("os", "path", None)]}, {"m_path"}),
    ],
    ids=[
        "single_name",
        "tuple_name",
        "different_node",
        "none_name",
        "overwrite_import",
        "overwrite_import_from",
    ],
)
def test_add_name(
    name: str | tuple[str, ...] | None,
    node: ast.AST,
    initial_imports: dict[str, list[tuple[str, str | None] | tuple[str, str, str | None]]],
    expected_names: dict[str, ast.AST],
) -> None:
    """Test adding names to the current scope."""
    tracker = ScopeTracker()
    # Set up initial imports if needed for conflict testing
    tracker.scopes[0].imports = initial_imports

    tracker.add_name(name, node)
    current_scope = tracker.current_scope
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
    node_import: ast.Import = ast.parse("import os as myos").body[0]  # type: ignore[assignment]
    tracker.add_import(node_import.names[0], None)
    assert tracker.current_scope.imports == {"myos": [("os", "myos")]}

    tracker.add_import(node_import.names[0], None)
    assert tracker.current_scope.imports == {"myos": [("os", "myos"), ("os", "myos")]}

    # Reset scope imports for next test
    tracker = ScopeTracker()

    # Test ast.ImportFrom
    node_import_from: ast.ImportFrom = ast.parse("from sys import argv as a").body[0]  # type: ignore[assignment]
    tracker.add_import(node_import_from.names[0], node_import_from.module)
    assert tracker.current_scope.imports == {"a": [("sys", "argv", "a")]}

    tracker.add_import(node_import_from.names[0], None)
    assert tracker.current_scope.imports == {"a": [("sys", "argv", "a"), ("argv", "a")]}


def test_add_global() -> None:
    tracker = ScopeTracker()

    global_node: ast.Global = ast.parse("global x").body[0]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Global keyword in outer scope redundant"):
        tracker.add_global(global_node)

    name_node = _make_name()
    with tracker.scope(name_node):
        tracker.add_global(global_node)
        assert tracker.current_scope.global_names == {"x"}


def test_add_nonlocal() -> None:
    tracker = ScopeTracker()

    nonlocal_node: ast.Nonlocal = ast.parse("nonlocal x").body[0]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Nonlocal keyword must be used in nested scope"):
        tracker.add_nonlocal(nonlocal_node)

    name_node = _make_name()
    with tracker.scope(name_node):
        with pytest.raises(ValueError, match="Nonlocal keyword must be used in nested scope"):
            tracker.add_nonlocal(nonlocal_node)

        other_name_node = _make_name()
        with tracker.scope(other_name_node):
            tracker.add_nonlocal(nonlocal_node)
            assert tracker.current_scope.nonlocal_names == {"x"}


def test_global_nonlocal_exclusive() -> None:
    global_node: ast.Global = ast.parse("global x").body[0]  # type: ignore[assignment]
    nonlocal_node: ast.Nonlocal = ast.parse("nonlocal x").body[0]  # type: ignore[assignment]

    tracker = ScopeTracker()
    outer_name_node = _make_name()
    with tracker.scope(outer_name_node):
        name_node = _make_name()
        with tracker.scope(name_node):
            tracker.add_global(global_node)
            with pytest.raises(ValueError, match="Global and nonlocal are mutually exclusive"):
                tracker.add_nonlocal(nonlocal_node)

        other_name_node = _make_name()
        with tracker.scope(other_name_node):
            tracker.add_nonlocal(nonlocal_node)
            with pytest.raises(ValueError, match="Global and nonlocal are mutually exclusive"):
                tracker.add_global(global_node)


def test_add_parents() -> None:
    tree = ast.parse("def func(): pass")
    add_parents(tree)
    func_node: ast.FunctionDef = tree.body[0]  # type: ignore[assignment]
    pass_node: ast.Pass = func_node.body[0]  # type: ignore[assignment]
    assert pass_node.parent == func_node  # type: ignore[attr-defined]
    assert func_node.parent == tree  # type: ignore[attr-defined]
