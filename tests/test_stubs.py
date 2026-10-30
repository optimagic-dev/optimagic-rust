"""Test that the type stub _core.pyi matches the compiled extension module.

IDEs show the docstrings from the stub, ``help()`` shows the docstrings compiled from
the Rust doc comments. This test checks that they are identical. Signatures are checked
separately by ``pixi run -e type-checking stubtest``.

"""

import ast
import importlib.metadata
import inspect
from pathlib import Path

import pytest

import optimagic_rust
from optimagic_rust import _core

STUB_PATH = Path(optimagic_rust.__file__).parent / "_core.pyi"


def _collect_stub_docstrings() -> dict[str, str | None]:
    """Map names like "func" or "Class.attribute" to their docstring in the stub."""
    tree = ast.parse(STUB_PATH.read_text(encoding="utf-8"))
    docstrings = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            docstrings[node.name] = ast.get_docstring(node)
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    docstrings[f"{node.name}.{item.name}"] = ast.get_docstring(item)
    return docstrings


STUB_DOCSTRINGS = _collect_stub_docstrings()


@pytest.mark.parametrize("name", STUB_DOCSTRINGS)
def test_stub_docstring_equals_runtime_docstring(name):
    obj = _core
    for part in name.split("."):
        obj = getattr(obj, part)
    assert STUB_DOCSTRINGS[name] == inspect.getdoc(obj)


def test_public_api_is_exported():
    public_stub_names = {name for name in STUB_DOCSTRINGS if "." not in name}
    assert public_stub_names | {"__version__"} == set(optimagic_rust.__all__)


def test_version_equals_package_metadata():
    assert optimagic_rust.__version__ == importlib.metadata.version("optimagic-rust")
