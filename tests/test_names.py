"""Test deprive.names."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from deprive.names import get_attribute_parts, get_node_defined_names, path_to_fqn


@pytest.mark.parametrize(
    ("rel_path_str", "expected_fqn"),
    [
        ("mod.py", "mytestroot.mod"),
        ("pkg/sub.py", "mytestroot.pkg.sub"),
        ("pkg/__init__.py", "mytestroot.pkg"),
        ("__init__.py", "mytestroot"),
        ("pkg", "mytestroot.pkg"),  # Directory itself implies package
    ],
)
@pytest.mark.parametrize(
    "root_dir", [Path("/path/to/mytestroot"), Path("relative/path/to/mytestroot")]
)
def test_path_to_fqn(rel_path_str: str, expected_fqn: str, root_dir: Path) -> None:
    """Test conversion of path to fully qualified name."""
    file_path = root_dir / rel_path_str
    assert path_to_fqn(file_path, root_dir) == expected_fqn


def test_path_to_fqn_module_only() -> None:
    """Test conversion of path to fully qualified name without `root_dir`."""
    assert path_to_fqn(Path("/path/to/mytestroot/mod.py"), None) == "mod"


def test_path_to_fqn_outside_root() -> None:
    """Test error when path is not inside the root directory."""
    root_dir = Path("/path/to/mytestroot")
    outside_path = Path("/path/to/different") / "other_file.py"
    with pytest.raises(ValueError, match="test"):
        path_to_fqn(outside_path, root_dir)


@pytest.mark.parametrize(
    ("source", "expected_names"),
    [
        ("def my_func(): pass", "my_func"),
        ("async def my_async_func(): pass", "my_async_func"),
        ("class MyClass: pass", "MyClass"),
        ("MY_CONST = 1", "MY_CONST"),
        ("MY_VAR: int = 1", "MY_VAR"),
        ("x, y = 1, 2", ("x", "y")),
        ("(a, b), c = (1, 2), 3", ("a", "b", "c")),
        (
            "head, *tail = [1, 2, 3]",
            ("head", "tail"),
        ),  # Starred handled via _unpack_target if implemented
        ("my_list[0] = 1", None),  # Assign to subscript ignored
        ("obj.attr = 1", None),  # Assign to attribute ignored
        ("my_var: list[int]", "my_var"),  # AnnAssign without value *should* capture Name
        ("a = b = 1", ("a", "b")),  # Chained assignment - need to parse carefully
        ("def _internal(): pass", "_internal"),  # Names starting with _ are included
        ("def __dunder__(): pass", "__dunder__"),
        # TypeAlias example (if supported in the future)
        # ("MyType = list[int]", ("MyType",)), # noqa: ERA001
        # Complex target assignments
        ("x.y = 1", None),  # Attribute target is ignored
        ("a[0] = 1", None),  # Subscript target is ignored
    ],
)
def test_get_node_defined_names(source: str, expected_names: tuple[str, ...] | str | None) -> None:
    """Test extraction of defined names from various AST nodes."""
    node = ast.parse(source).body[0]
    assert get_node_defined_names(node) == expected_names


def test_get_node_defined_names_custom_name() -> None:
    """Test that custom names are returned correctly."""
    # Create a custom AST node with a custom name
    tree = ast.parse("pass")
    tree.custom_name = "custom_name"  # type: ignore[attr-defined]
    assert get_node_defined_names(tree) == "custom_name"


# Test edge case: node type not handled
def test_get_node_defined_names_unhandled_type() -> None:
    """Test that unhandled node types return an empty tuple."""
    node = ast.Expr(value=ast.Constant(value=1))  # An expression statement node
    assert get_node_defined_names(node) is None


@pytest.mark.parametrize(("name"), ["a.b.c"])
def test_get_attribute_parts(name: str) -> None:
    """Test extraction of attribute names from various AST nodes."""
    node: ast.Attribute = ast.parse(name).body[0].value  # type: ignore[attr-defined]
    assert get_attribute_parts(node) == name.split(".")
