"""Test the deprive.handler."""

from __future__ import annotations

import ast
import textwrap

import libcst as cst
import pytest

from deprive.handler import (
    _handle_if_else,
    get_names,
    get_node,
    handle_elem,
    handle_import,
    handle_module,
    to_ast,
)

EXPR_LINES = ['"""This is a test module."""', 'print("test")']
IMPORT_LINES = ["import os", "import pathlib as path", "from test import a, b as c"]
STMT_LINES = [
    "CONST, CONST_B = 1, 2",
    "CONST, CONST_B = 1, 2",
    "CONST, CONST_B = 1, 2",
    """\
    def func():
        print("Hello, world!")
    """,
    """\
    def outer_func():
        def inner_func():
            print("Inner function")
        return inner_func
    """,
    """\
    class MyClass:
        def method(self):
            pass
    """,
]
STMT_KEEP = [{"CONST"}, {"CONST_B"}, {"CONST", "CONST_B"}, {"func"}, {"outer_func"}, {"MyClass"}]
CODE_LINES = EXPR_LINES + IMPORT_LINES + STMT_LINES


def assert_node_equals(node: cst.CSTNode, code: str) -> None:
    """Check if two nodes are equal."""
    assert ast.unparse(to_ast(node)).strip() == textwrap.dedent(code).strip()


@pytest.mark.parametrize("code", CODE_LINES)
def test_to_ast(code: str) -> None:
    code = textwrap.dedent(code)
    node = ast.parse(code).body[0]
    cst_node = cst.parse_module(code).body[0]
    converted_node = to_ast(cst_node)
    assert ast.unparse(converted_node) == ast.unparse(node)


# TODO(tihoph): make as fixture
@pytest.mark.parametrize(
    ("code", "expected"),
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
        ("x.y: int  = 1", None),  # Attribute target is ignored
        ("a[0]: int = 1", None),  # Subscript target is ignored
        ("global x", ("x",)),
        ("global x, y", ("x", "y")),
        ("nonlocal x", ("x",)),
        ("nonlocal x, y", ("x", "y")),
    ],
)
def test_get_names(code: str, expected: str | tuple[str, ...] | None) -> None:
    """Test get_names function."""
    node = cst.parse_module(code).body[0]
    if isinstance(expected, str):
        expected = (expected,)
    if expected is None:
        with pytest.raises(ValueError, match="No names found in element:"):
            get_names(node)
    else:
        assert get_names(node) == expected


@pytest.mark.parametrize(
    ("code", "typ", "expected"),
    [
        (None, cst.Assign, False),
        ("x = 1", cst.Assign, True),
        ("x: int = 1", cst.AnnAssign, True),
        ("x = 1", cst.AnnAssign, False),
        ("x: int = 1", cst.Assign, False),
    ],
)
def test_get_node(code: str | None, typ: type[cst.CSTNode], expected: bool) -> None:
    """Test get_node function."""
    node: cst.SimpleStatementLine | None = cst.parse_statement(code) if code else None  # type: ignore[assignment]
    for visit_body in (False, True):
        subnode = node.body[0] if node and visit_body else node

        actual = get_node(subnode, typ)
        if expected:
            assert isinstance(actual, typ)
        else:
            assert actual is None


def test_get_node_multiple() -> None:
    node = cst.parse_module("x = 1; y = 2").body[0]
    with pytest.raises(
        NotImplementedError, match="Multiple statements per line are not supported."
    ):
        get_node(node, cst.Assign)


def test_handle_import() -> None:
    node: cst.SimpleStatementLine = cst.parse_statement("import os as myos, sys")  # type: ignore[assignment]
    elem: cst.Import = node.body[0]  # type: ignore[assignment]

    # removed
    assert handle_import(node, elem, set()) == []
    new_node = handle_import(node, elem, {"myos"})
    assert len(new_node) == 1
    assert_node_equals(new_node[0], "import os as myos")
    new_node = handle_import(node, elem, {"sys"})
    assert len(new_node) == 1
    assert_node_equals(new_node[0], "import sys")


def test_handle_import_star() -> None:
    node: cst.SimpleStatementLine = cst.parse_statement("from os import *")  # type: ignore[assignment]
    elem: cst.ImportFrom = node.body[0]  # type: ignore[assignment]
    with pytest.raises(NotImplementedError, match=r"Import \* is not supported."):
        handle_import(node, elem, set())


def test_handle_import_future() -> None:
    node: cst.SimpleStatementLine = cst.parse_statement("import __future__")  # type: ignore[assignment]
    elem: cst.Import = node.body[0]  # type: ignore[assignment]
    assert handle_import(node, elem, set()) == [node]

    node = cst.parse_statement("import __future__ as myfuture")  # type: ignore[assignment]
    elem = node.body[0]  # type: ignore[assignment]
    assert handle_import(node, elem, set()) == [node]

    node = cst.parse_statement("import __future__ as myfuture, other")  # type: ignore[assignment]
    elem = node.body[0]  # type: ignore[assignment]
    new_node = handle_import(node, elem, set())
    assert len(new_node) == 1
    assert_node_equals(new_node[0], "import __future__ as myfuture")

    node = cst.parse_statement("from __future__ import annotations")  # type: ignore[assignment]
    elem = node.body[0]  # type: ignore[assignment]
    assert handle_import(node, elem, set()) == [node]


@pytest.mark.parametrize("root", ["os", ".", ".other"])
def test_handle_import_from(root: str) -> None:
    node: cst.SimpleStatementLine = cst.parse_statement(f"from {root} import path as mypath, other")  # type: ignore[assignment]
    elem: cst.ImportFrom = node.body[0]  # type: ignore[assignment]
    assert handle_import(node, elem, set()) == []

    new_node = handle_import(node, elem, {"mypath"})
    assert len(new_node) == 1
    assert_node_equals(new_node[0], f"from {root} import path as mypath")

    new_node = handle_import(node, elem, {"other"})
    assert len(new_node) == 1
    assert_node_equals(new_node[0], f"from {root} import other")


@pytest.mark.parametrize(("code", "keep"), zip(STMT_LINES, STMT_KEEP))
def test_handle_elem(code: str, keep: set[str]) -> None:
    node = cst.parse_statement(textwrap.dedent(code))
    elem = node.body[0] if isinstance(node, cst.SimpleStatementLine) else node

    assert handle_elem(node, elem, set(), set()) == []  # type: ignore[arg-type]
    assert handle_elem(node, elem, set(), keep) == [node]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "code",
    [
        """__all__ = ("a", "b", "c")""",
        """__all__ = ["a", "b", "c"]""",
        """__all__: list[str] = ["a", "b", "c"]""",
    ],
)
def test_handle_elem_all(code: str) -> None:
    parens = "()" if "(" in code else "[]"
    p_open, p_close = parens[0], parens[1]
    comma = "," if "(" in code else ""

    intro, _ = code.split(" = ", 1)
    node: cst.SimpleStatementLine = cst.parse_statement(code)  # type: ignore[assignment]
    elem: cst.Assign | cst.AnnAssign = node.body[0]  # type: ignore[assignment]
    handle_elem(node, elem, set(), set())

    new_node = handle_elem(node, elem, set(), {"a"})
    assert len(new_node) == 1
    assert_node_equals(new_node[0], f"{intro} = {p_open}'a'{comma}{p_close}")

    new_node = handle_elem(node, elem, {"b"}, {"a"})
    assert len(new_node) == 1
    assert_node_equals(new_node[0], f"{intro} = {p_open}'a', 'b'{p_close}")

    new_node = handle_elem(node, elem, {"c"}, set())
    assert len(new_node) == 1
    assert_node_equals(new_node[0], f"{intro} = {p_open}'c'{comma}{p_close}")

    assert handle_elem(node, elem, {"c"}, {"a", "b"}) == [node]


def test_handle_elem_all_not_list_of_strings() -> None:
    node: cst.SimpleStatementLine = cst.parse_statement("__all__ = [1, 2, 3]")  # type: ignore[assignment]
    assign_node: cst.Assign = node.body[0]  # type: ignore[assignment]
    with pytest.raises(TypeError, match=r"Expected a tuple or list of literal strings, got"):
        handle_elem(node, assign_node, set(), set())


def test_handle_elem_all_no_value() -> None:
    node: cst.SimpleStatementLine = cst.parse_statement("""__all__: list[str]""")  # type: ignore[assignment]
    annassign_node: cst.AnnAssign = node.body[0]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="No value for __all__ assignment."):
        handle_elem(node, annassign_node, set(), set())


def test_handle_elem_all_unsupported() -> None:
    node: cst.SimpleStatementLine = cst.parse_statement("""__all__, other = []""")  # type: ignore[assignment]
    assign_node: cst.Assign = node.body[0]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported __all__ definition."):
        handle_elem(node, assign_node, set(), set())

    node = cst.parse_statement("""__all__ += []""")  # type: ignore[assignment]
    augassign_node: cst.AugAssign = node.body[0]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported __all__ definition."):
        handle_elem(node, augassign_node, set(), set())


@pytest.mark.parametrize(
    ("keep", "expected"),
    [
        (
            {"x"},
            """\
    if True:
        x = 1
    else:
        x = 2
    """,
        ),
        (
            {"y"},
            """\
    if True:
        pass
    else:
        y = 3
    """,
        ),
        (
            {"z"},
            """\
    if True:
        z = 2
    """,
        ),
    ],
)
def test_handle_if_else(keep: set[str], expected: str) -> None:
    node: cst.If = cst.parse_statement(  # type: ignore[assignment]
        textwrap.dedent("""\
    if True:
        x = 1
        z = 2
    else:
        x = 2
        y = 3
    """)
    )
    new_node = _handle_if_else(node, set(), keep)
    assert len(new_node) == 1
    assert_node_equals(new_node[0], expected)

    assert _handle_if_else(node, set(), set()) == []


def test_handle_if() -> None:
    code = """\
    if True:
        x = 1
    """
    node: cst.If = cst.parse_statement(textwrap.dedent(code))  # type: ignore[assignment]
    new_node = _handle_if_else(node, set(), {"x"})
    assert len(new_node) == 1
    assert_node_equals(new_node[0], code)
    assert _handle_if_else(node, set(), set()) == []


@pytest.mark.parametrize(
    ("code", "expected", "required", "keep"),
    [
        (
            '''\
"""This is a test module."""
import os
import sys
import pathlib as path
from test import a, b as c

if sys.platform == "win32":
    CONST, CONST_B = 1, 2
else:
    CONST, CONST_B = 3, 4

def func():
    print("Hello, world!")

def outer_func():
    def inner_func():
        print("Inner function")
    return inner_func

class MyClass:
    def method(self):
        pass

print("test")
''',
            '''\
"""This is a test module."""
import os
import sys
from test import a

if sys.platform == "win32":
    CONST, CONST_B = 1, 2
else:
    CONST, CONST_B = 3, 4

def func():
    print("Hello, world!")

print("test")
''',
            {"os", "sys", "a"},
            {"func", "CONST"},
        ),
        ("from __future__ import annotations\n", "from __future__ import annotations\n", {}, {}),
        ("import logging\nlogger = logging.getLogger(__name__)\n", None, {}, {}),
        (
            """\
import logging
logger = logging.getLogger(__name__)
def uses_logger(): logger.log("test")
""",
            None,
            {},
            {},
        ),
        (
            """\
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
def annotated() -> Any: pass
""",
            None,
            {},
            {},
        ),
        (
            """\
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
def annotated() -> Any: pass
""",
            """\
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
def annotated() -> Any: pass
""",
            {"Any", "TYPE_CHECKING"},
            {"annotated"},
        ),
    ],
)
def test_handle_module(code: str, expected: str | None, required: set[str], keep: set[str]) -> None:
    new_code = handle_module(code, required, keep)
    assert new_code == expected
