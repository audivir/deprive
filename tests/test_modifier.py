"""Test the modification of source files."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from deprive.modifier import _modify_module, modify_package
from deprive.visitor import Definition

PROJ_PATH = Path(__file__).parent / "_assets" / "simple_proj"


def test_modify_package_wrong_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Output .* must be a directory"):
        modify_package(
            PROJ_PATH, ["simple_proj.nested_pkg.nester.nested_func"], tmp_path / "file.py"
        )


def test_modify_package(tmp_path: Path) -> None:
    modify_package(
        PROJ_PATH,
        ["simple_proj.nested_pkg.nester.nested_func", "simple_proj.main_module.MainClass"],
        tmp_path,
    )
    for path in tmp_path.rglob("*.py"):
        rel_path = path.relative_to(tmp_path)
        expected_path = PROJ_PATH.parent / "simple_proj_minified" / rel_path
        assert path.read_text() == expected_path.read_text()


def test_modify_module_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    code = """import os"""
    root = tmp_path / "root"
    root.mkdir()
    (root / "module.py").write_text(code)
    with caplog.at_level(logging.DEBUG):
        _modify_module(root, "root", {}, tmp_path / "output", Definition("root.module", None))
    assert "Skipping empty module" in caplog.text
