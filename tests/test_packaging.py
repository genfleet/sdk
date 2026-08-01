"""Guards the PEP 420 namespace layout of the shared `agentinc` package.

The `agentinc` import namespace is shared by every agentinc-* distribution
(agentinc-sdk, agentinc-core, agentinc-engine, agentinc-loader, agentinc-runner,
agentinc-adapter, agentinc-protocols). If any one of them ships an
`agentinc/__init__.py`, `agentinc` becomes a *regular* package and its
`__path__` collapses to that single directory, making every sibling
distribution unimportable.
"""

import importlib
import pathlib

import agentinc
import agentinc.sdk


def test_agentinc_is_namespace_package():
    """`agentinc` must have no module file of its own."""
    assert getattr(agentinc, "__file__", None) is None, (
        "agentinc has a __file__, so it is a regular package. "
        "Delete agentinc/__init__.py — it shadows sibling distributions."
    )


def test_agentinc_path_is_namespace_path():
    assert type(agentinc.__path__).__name__ == "_NamespacePath", (
        f"agentinc.__path__ is {type(agentinc.__path__).__name__}, "
        "expected _NamespacePath"
    )


def test_agentinc_has_no_init_file_on_disk():
    """The source tree must not contain agentinc/__init__.py."""
    root = pathlib.Path(__file__).resolve().parent.parent
    assert not (root / "agentinc" / "__init__.py").exists()


def test_agentinc_sdk_is_a_regular_package():
    """The subpackage itself is a normal package and stays importable."""
    assert agentinc.sdk.__file__ is not None
    assert importlib.import_module("agentinc.sdk") is agentinc.sdk
