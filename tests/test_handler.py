"""Test the deprive.handler."""

from __future__ import annotations

from deprive.handler import handle_module


def test_handle_module() -> None:
    code = '''\
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
'''
    expected = '''\
"""This is a test module."""
import os
from test import a

CONST, CONST_B = 1, 2

def func():
    print("Hello, world!")

print("test")
'''
    required = {"os", "a"}
    keep = {"func", "CONST"}
    new_code = handle_module(code, required, keep)
    assert new_code == expected
