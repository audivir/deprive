"""Main module for the sample project."""

# Standard library import
from __future__ import annotations

import json

# Alias import
import pathlib as pathlib_alias

from .utils import CONST, HelperClass, helper_func


class MainClass:
    """Main class that uses utility functions and constants."""

    def __init__(self) -> None:
        """Initializes the MainClass with a helper instance and a constant."""
        self.helper = HelperClass("Main")
        self.const_val = CONST

    def process(self, data: int) -> str:
        """Processes the input data using the helper function and returns a JSON string."""
        processed_data = helper_func(data)
        self.helper.greet()
        # Use aliased import
        pathlib_alias.Path("example")
        return json.dumps({"result": processed_data})


__all__ = ["MainClass"]  # fmt: skip
