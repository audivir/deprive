"""Test deprive.collect."""

from __future__ import annotations

from pathlib import Path

from deprive.collect import collect_module, collect_package
from deprive.visitor import Definition, Import

PROJ_PATH = Path(__file__).parent / "_assets" / "simple_proj"


# TODO(tihoph): Test coverage needs to be reached without this test
def _test_collect_module(module: str, file_path: Path, root_dir: Path | None) -> None:
    """Test the public collect_module function."""
    code = """
import os
import typing
import importlib.util
import test.subpkg
from multiprocessing import cpu_count, current_process
import platform

CONSTANT = cpu_count()
OTHER_CONSTANT = current_process()
print(platform.system())

def func() -> typing.Any:
    import tensorflow as tf
    def inner_func(x):
        return x + 1
    return inner_func(2)

def outer_func() -> os.PathLike:
    print("Hello, world!")
    def inner_func():
       print(typing.Any)

def other_func() -> None:
    print(importlib.util.find_spec)


class Test(test.subpkg.Class):
    import sys
    sys.path.append("/here") # should not be in dependency graph
    def __init__(self) -> None:
        import pathlib
        print(pathlib.Path) # should also not be in dependency graph

    def method(self) -> None:
        importlib.util.find_spec("tensorflow")

def use_constant():
    print(CONSTANT)

def empty_func():
    pass
"""

    def _get_curr_def(name: str | None = None) -> Definition:
        return Definition(module, name)

    expected = {
        _get_curr_def(): {Import("platform"), Import(("platform", "system"), None)},
        _get_curr_def("func"): {Import("typing"), Import(("typing", "Any"), None)},
        _get_curr_def("outer_func"): {
            Import("os"),
            Import("typing"),
            Import(("os", "PathLike"), None),
            Import(("typing", "Any"), None),
        },
        _get_curr_def("other_func"): {
            Import("importlib.util"),
            Import(("importlib.util", "find_spec"), None),
        },
        _get_curr_def("use_constant"): {_get_curr_def("CONSTANT")},
        _get_curr_def("empty_func"): set(),
        _get_curr_def("Test"): {
            Import("pathlib"),
            Import("sys"),
            Import("test.subpkg"),
            Import("importlib.util"),
            Import(("pathlib", "Path"), None),
            Import(("sys", "path"), None),
            Import(("test.subpkg", "Class"), None),
            Import(("importlib.util", "find_spec"), None),
        },
        _get_curr_def("CONSTANT"): {Import(("multiprocessing", "cpu_count"))},
        _get_curr_def("OTHER_CONSTANT"): {Import(("multiprocessing", "current_process"))},
    }
    file_path.write_text(code)

    dependencies = collect_module(file_path, root_dir)
    # missing keys
    assert set(dependencies) == set(expected)
    for key, value in expected.items():
        assert dependencies[key] == value


def test_collect_module_init(tmp_path: Path) -> None:
    root_dir = tmp_path / "test_module"
    root_dir.mkdir()
    file_path = root_dir / "__init__.py"
    _test_collect_module("test_module.__init__", file_path, root_dir)


def test_collect_module_single(tmp_path: Path) -> None:
    file_path = tmp_path / "test_module.py"
    _test_collect_module("test_module", file_path, None)


def test_collect_module_sub(tmp_path: Path) -> None:
    root_dir = tmp_path / "test_module"
    root_dir.mkdir()
    file_path = root_dir / "sub.py"
    _test_collect_module("test_module.sub", file_path, root_dir)


def test_collect_package_small() -> None:
    dependencies = collect_package(PROJ_PATH / "nested_pkg")

    expected = {
        Definition("nested_pkg.__init__", None): set(),
        Definition("nested_pkg.__init__", "nested_func"): {
            Import(name=("simple_proj.nested_pkg.nester", "nested_func"))
        },
        Definition("nested_pkg.__init__", "__all__"): set(),
        Definition("nested_pkg.nester", None): set(),
        Definition("nested_pkg.nester", "nested_func"): {
            Import(("simple_proj.utils", "helper_func"))
        },
    }

    assert dependencies == expected


def test_collect_package_large() -> None:
    dependencies = collect_package(PROJ_PATH)

    def _get_curr_def(suffix: str, name: str | None = None) -> Definition:
        return Definition(f"simple_proj.{suffix}", name)

    expected = {
        Definition("simple_proj.__init__", None): set(),
        _get_curr_def("main_module"): set(),
        _get_curr_def("main_module", "__all__"): set(),
        _get_curr_def("main_module", "MainClass"): {
            Import("pathlib", "pathlib_alias"),
            Import(("simple_proj.utils", "HelperClass")),
            Import(("simple_proj.utils", "CONST")),
            Import(("simple_proj.utils", "helper_func")),
            Import("json"),
            Import(("pathlib", "Path"), None),
            Import(("json", "dumps"), None),
        },
        _get_curr_def("main_module", "OTHER_CONST"): {Import(("simple_proj.utils", "CONST"))},
        _get_curr_def("main_module", "main_func"): {_get_curr_def("main_module", "MainClass")},
        _get_curr_def("utils"): set(),
        _get_curr_def("utils", "CONST"): set(),
        _get_curr_def("utils", "HelperClass"): set(),
        _get_curr_def("utils", "helper_func"): set(),
        _get_curr_def("utils", "_internal_var"): set(),
        _get_curr_def("utils", "_internal_func"): set(),
        _get_curr_def("independent"): {_get_curr_def("independent", "indep_func")},
        _get_curr_def("independent", "INDEP_CONST"): set(),
        _get_curr_def("independent", "indep_func"): {_get_curr_def("independent", "INDEP_CONST")},
        _get_curr_def("nested_pkg.__init__"): set(),
        _get_curr_def("nested_pkg.__init__", "nested_func"): {
            Import(name=("simple_proj.nested_pkg.nester", "nested_func"))
        },
        _get_curr_def("nested_pkg.__init__", "__all__"): set(),
        _get_curr_def("nested_pkg.nester"): set(),
        _get_curr_def("nested_pkg.nester", "nested_func"): {
            Import(("simple_proj.utils", "helper_func"))
        },
    }
    assert set(dependencies) == set(expected)
    for key, value in dependencies.items():
        assert value == expected[key]
