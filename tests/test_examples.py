"""Test for specific examples."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from deprive.visitor import Definition, Import, ScopeVisitor

if TYPE_CHECKING:
    from collections.abc import Sequence

ModDef = Definition("test_module", None)


def parse_and_visit(code: str | Sequence[str], module_fqn: str = "test_module") -> ScopeVisitor:
    """Parses code and runs ScopeVisitor on it."""
    visitor = ScopeVisitor(module_fqn, debug=True)
    if not isinstance(code, str):  # pragma: no cover
        code = "\n".join(code)
    visitor.run(textwrap.dedent(code))
    return visitor


def test_class_def() -> None:
    code = """\
    import pkg
    import pkg.module
    class A(pkg.module.B):
        pass
    """
    visitor = parse_and_visit(code)
    assert visitor.dep_graph == {
        ModDef: set(),
        Definition("test_module", "A"): {
            Import("pkg"),
            Import("pkg.module"),
            Import(("pkg.module", "B"), None),
        },
    }


def test_class_methods() -> None:
    code = """\
    def outer(): pass
    class A:
        def outer(self):
            outer()
    """
    visitor = parse_and_visit(code)
    assert visitor.dep_graph == {
        ModDef: set(),
        Definition("test_module", "outer"): set(),
        Definition("test_module", "A"): {Definition("test_module", "outer")},
    }


def test_function_defaults() -> None:
    code = """\
    from pkg import Default
    def outer(test = Default(None)): pass
    """
    visitor = parse_and_visit(code)
    assert visitor.dep_graph == {
        ModDef: set(),
        Definition("test_module", "outer"): {Import(("pkg", "Default"))},
    }


def test_imported_attr() -> None:
    code = """\
    import rich_utils
    rich_utils._get_rich_console(stderr=True)
    """
    visitor = parse_and_visit(code)
    assert visitor.dep_graph == {
        ModDef: {Import("rich_utils"), Import(("rich_utils", "_get_rich_console"), None)}
    }
