"""Guards the PEP 420 namespace layout of the shared `fleet` package.

The `fleet` import namespace is shared by every fleet-* distribution
(withfleet-sdk, withfleet-core, withfleet-engine, withfleet-loader, withfleet-runner,
withfleet-adapter, withfleet-protocols). If any one of them ships an
`fleet/__init__.py`, `fleet` becomes a *regular* package and its
`__path__` collapses to that single directory, making every sibling
distribution unimportable.
"""

import importlib
import pathlib

import fleet
import fleet.sdk


def test_fleet_is_namespace_package():
    """`fleet` must have no module file of its own."""
    assert getattr(fleet, "__file__", None) is None, (
        "fleet has a __file__, so it is a regular package. "
        "Delete fleet/__init__.py — it shadows sibling distributions."
    )


def test_fleet_path_is_namespace_path():
    assert type(fleet.__path__).__name__ == "_NamespacePath", (
        f"fleet.__path__ is {type(fleet.__path__).__name__}, "
        "expected _NamespacePath"
    )


def test_fleet_has_no_init_file_on_disk():
    """The source tree must not contain fleet/__init__.py."""
    root = pathlib.Path(__file__).resolve().parent.parent
    assert not (root / "fleet" / "__init__.py").exists()


def test_fleet_sdk_is_a_regular_package():
    """The subpackage itself is a normal package and stays importable."""
    assert fleet.sdk.__file__ is not None
    assert importlib.import_module("fleet.sdk") is fleet.sdk
