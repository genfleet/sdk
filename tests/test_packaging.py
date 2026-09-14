"""Guards the PEP 420 namespace layout of the shared `genfleet` package.

The `genfleet` import namespace is shared by every genfleet-* distribution
(genfleet-sdk, genfleet-core, genfleet-engine, genfleet-loader, genfleet-runner,
genfleet-adapter, genfleet-protocols). If any one of them ships an
`genfleet/__init__.py`, `genfleet` becomes a *regular* package and its
`__path__` collapses to that single directory, making every sibling
distribution unimportable.
"""

import importlib
import pathlib

import genfleet
import genfleet.sdk


def test_genfleet_is_namespace_package():
    """`genfleet` must have no module file of its own."""
    assert getattr(genfleet, "__file__", None) is None, (
        "genfleet has a __file__, so it is a regular package. "
        "Delete genfleet/__init__.py — it shadows sibling distributions."
    )


def test_genfleet_path_is_namespace_path():
    assert type(genfleet.__path__).__name__ == "_NamespacePath", (
        f"genfleet.__path__ is {type(genfleet.__path__).__name__}, "
        "expected _NamespacePath"
    )


def test_genfleet_has_no_init_file_on_disk():
    """The source tree must not contain genfleet/__init__.py."""
    root = pathlib.Path(__file__).resolve().parent.parent
    assert not (root / "genfleet" / "__init__.py").exists()


def test_genfleet_sdk_is_a_regular_package():
    """The subpackage itself is a normal package and stays importable."""
    assert genfleet.sdk.__file__ is not None
    assert importlib.import_module("genfleet.sdk") is genfleet.sdk
