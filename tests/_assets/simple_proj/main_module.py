"""Main module for the sample project."""

# Non-existent import - should be ignored if not used
# from .nonexistent import NonExistent
# Standard library import
from __future__ import annotations

import json

# Alias import
import pathlib as pathlib_alias

from simple_proj.utils import CONST, HelperClass, helper_func


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


def main_func() -> str:
    """Main function that creates an instance of MainClass and processes data."""
    instance = MainClass()
    return instance.process(5)


# Constant using another constant
OTHER_CONST = CONST + 1
