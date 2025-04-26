# Deprive

# Installation

```bash
pip install deprive
```

# Disabled Python Features
If one of the disabled features is used, a `NotImplementedError` will be raised:
1. `global` and `nonlocal` keywords.
2. Star imports (e.g., `from module import *`).
3. Relative imports (e.g., `from . import module`).
4. Multiple imports in a single statement (e.g., `import os, sys`).
5. Redefining imports (independent of the scope).
6. Multiple statements per line (e.g., `a = 1; b = 2`).

# Discouraged Python Features
These following features don't raise any warning or errors, but the correctness of the output is not guaranteed:
1. `importlib.import_module`, `__import__`, `eval`, `exec`, or similar dynamic functions.
2. Overwriting built-in functions.
3. Assigning functions to variables.
4. Namespace packages (e.g., no `__init__.py` in a package).

# TODO
- Rewrite `__all__`: eval original all and only set still valid ones.

# Changelog

## 0.1.0 - Initial Release
- Initial implementation of the deprive library.
- Support for parsing Python files and generating dependency graphs.
- Codes needs to be cleaned up and refactored.