"""Test deprive.tracker."""

from __future__ import annotations

from pathlib import Path

import pytest

from deprive.collect import collect_package
from deprive.tracker import NoDefinitionError, track_dependencies
from deprive.visitor import Definition, Import

PROJ_PATH = Path(__file__).parent / "_assets" / "simple_proj"


def test_track_dependencies_fails() -> None:
    graph = collect_package(PROJ_PATH)
    with pytest.raises(NoDefinitionError, match="No matching definition or module found for"):
        track_dependencies("simple_proj", graph, ["simple_proj.unknown"])

    with pytest.raises(NoDefinitionError, match="No matching definition or module found for"):
        track_dependencies("simple_proj", graph, ["simple_proj.unknown.func"])

    with pytest.raises(ValueError, match="Required element must start with the module name"):
        track_dependencies("simple_proj", graph, ["other_proj"])


def test_track_dependencies() -> None:
    graph = collect_package(PROJ_PATH)
    tracked = track_dependencies(
        "simple_proj",
        graph,
        ["simple_proj.nested_pkg.nester.nested_func", "simple_proj.main_module.MainClass"],
    )
    expected = {
        Definition("simple_proj.__init__", None): set(),
        Definition("simple_proj.nested_pkg.__init__", None): set(),
        Definition("simple_proj.nested_pkg.nester", None): set(),
        Definition("simple_proj.nested_pkg.nester", "nested_func"): {
            Import(("simple_proj.utils", "helper_func"))
        },
        Definition("simple_proj.utils", None): set(),
        Definition("simple_proj.utils", "helper_func"): set(),
        Definition("simple_proj.utils", "HelperClass"): set(),
        Definition("simple_proj.utils", "CONST"): set(),
        Definition("simple_proj.main_module", None): set(),
        Definition("simple_proj.main_module", "MainClass"): {
            Import(("simple_proj.utils", "HelperClass")),
            Import(("simple_proj.utils", "CONST")),
            Import(("simple_proj.utils", "helper_func")),
            Import("json"),
            Import("pathlib", "pathlib_alias"),
            Import(("pathlib", "Path"), None),
            Import(("json", "dumps"), None),
        },
    }
    assert tracked == expected


def test_track_dependencies_root() -> None:
    graph = collect_package(PROJ_PATH)
    tracked = track_dependencies("simple_proj", graph, ["simple_proj"])
    expected: dict[Definition, set[Definition | Import]] = {
        Definition("simple_proj.__init__", None): set()
    }
    assert tracked == expected
