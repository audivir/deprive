"""Test the deprive.handler."""

from __future__ import annotations

import pytest

from deprive.handler import handle_module


@pytest.mark.parametrize(
    ("code", "expected", "required", "keep"),
    [
        (
            '''\
"""This is a test module."""
import os
import pathlib as path
from test import a, b as c

CONST, CONST_B = 1, 2

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
from test import a

CONST, CONST_B = 1, 2

def func():
    print("Hello, world!")

print("test")
''',
            {"os", "a"},
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
