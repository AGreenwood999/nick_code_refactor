"""
Imports full directory structure, every folder available
except those starting with "." or "__". For example, it
omits "__pycache__" and ".DS_Store" folders.
"""

import importlib as _importlib
import pathlib as _pathlib

MAGNOLIA_PATH = _pathlib.Path(__file__).expanduser().resolve().parent


def _import_recursively(loc: _pathlib.Path, module_prefix: str):
    for d in loc.glob("*"):
        if d.name[0] == "." or d.name[:2] == "__":
            continue
        if d.is_dir():
            new_module = f"{module_prefix}.{d.name}"
            _importlib.import_module(new_module)
            _import_recursively(d, new_module)


_import_recursively(_pathlib.Path(__file__).resolve().parent, __name__)
