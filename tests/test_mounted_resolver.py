"""
`MountedToolResolver` — the sandbox path, where tools arrive as directories.

The engine resolves each name to a digest on the *host*, verifies the bytes
and mounts them read-only; by the time this resolver runs there is nothing to
authenticate to. So every test here is a directory on disk, which is exactly
the resolver's whole world.
"""

from __future__ import annotations

import sys

import pytest

from agentinc.sdk.manifest import ToolRef
from agentinc.sdk.tools_loader import (
    LocalToolResolver,
    MountedToolResolver,
    ToolResolutionError,
    default_resolver,
    load_tools,
)


def mount_tool(root, slug, *, entry, body, package=False):
    """Write a tool artifact the way `deliver_tools` unpacks one."""
    scope, _, name = slug.lstrip("@").partition("/")
    directory = root / scope / name
    directory.mkdir(parents=True)
    (directory / "agentinc.toml").write_text(
        f'[tool]\nslug = "{slug}"\nversion = "1.0.0"\nentry = "{entry}"\n'
    )
    module_path, _, _ = entry.partition(":")
    if package:
        target = directory / "__init__.py"
    else:
        target = directory / f"{module_path.split('.')[-1]}.py"
    target.write_text(body)
    return directory


@pytest.fixture(autouse=True)
def _no_leaked_modules():
    """Each test gets a clean `sys.modules`.

    The resolver caches by module name so a tool imported twice in one process
    is one module. That is correct in a sandbox, which serves one agent, and
    it would silently make the second test in this file assert against the
    first test's code.
    """
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]


def test_it_resolves_a_tool_pushed_as_its_own_package_directory(tmp_path):
    # The shape `ag tools push src/agentinc_tools/echo` actually produces: the
    # archive root *is* the package, so the entry's leading segments have no
    # counterpart on disk.
    mount_tool(
        tmp_path,
        "@agentinc/echo",
        entry="agentinc_tools.echo:echo",
        body="def echo(text):\n    return text\n",
        package=True,
    )

    resolved = MountedToolResolver(tmp_path).resolve(ToolRef(slug="@agentinc/echo"))

    assert resolved("hi") == "hi"


def test_a_tool_that_really_ships_the_full_package_path_resolves_to_itself(tmp_path):
    # Longest-suffix-first is what keeps this a widening rather than a change
    # of meaning: an artifact containing `agentinc_tools/echo.py` must not be
    # answered by a root `__init__.py` that happens to sit beside it.
    directory = tmp_path / "agentinc" / "echo"
    (directory / "agentinc_tools").mkdir(parents=True)
    (directory / "agentinc.toml").write_text(
        '[tool]\nslug = "@agentinc/echo"\nversion = "1.0.0"\n'
        'entry = "agentinc_tools.echo:echo"\n'
    )
    (directory / "agentinc_tools" / "echo.py").write_text("def echo(t):\n    return 'inner'\n")
    (directory / "__init__.py").write_text("def echo(t):\n    return 'root'\n")

    resolved = MountedToolResolver(tmp_path).resolve(ToolRef(slug="@agentinc/echo"))

    assert resolved("x") == "inner"


def test_a_multi_file_tool_can_import_its_own_helpers(tmp_path):
    # `submodule_search_locations` is what makes this work. Without it the
    # loaded module is not a package and `from . import helpers` raises
    # ImportError, which is the ordinary way to write a tool of any size.
    directory = mount_tool(
        tmp_path,
        "@agentinc/calc",
        entry="agentinc_tools.calc:add",
        body="from . import helpers\n\ndef add(a, b):\n    return helpers.total(a, b)\n",
        package=True,
    )
    (directory / "helpers.py").write_text("def total(a, b):\n    return a + b\n")

    resolved = MountedToolResolver(tmp_path).resolve(ToolRef(slug="@agentinc/calc"))

    assert resolved(2, 3) == 5


def test_two_tools_whose_artifacts_share_a_filename_stay_distinct(tmp_path):
    # Both would claim the `sys.modules` key of their declared module path, so
    # the second import would silently return the first tool's module. Naming
    # by slug is what stops one tool answering for another.
    mount_tool(
        tmp_path, "@acme/one", entry="tool:run",
        body="def run():\n    return 'one'\n",
    )
    mount_tool(
        tmp_path, "@acme/two", entry="tool:run",
        body="def run():\n    return 'two'\n",
    )
    resolver = MountedToolResolver(tmp_path)

    assert resolver.resolve(ToolRef(slug="@acme/one"))() == "one"
    assert resolver.resolve(ToolRef(slug="@acme/two"))() == "two"


def test_a_tool_the_engine_did_not_mount_says_what_was(tmp_path):
    mount_tool(
        tmp_path, "@acme/present", entry="tool:run", body="def run():\n    pass\n",
    )

    with pytest.raises(ToolResolutionError) as excinfo:
        MountedToolResolver(tmp_path).resolve(ToolRef(slug="@acme/absent"))

    # The list matters more than the refusal: delivery and the manifest
    # disagreeing is an engine bug, and the message is the only evidence.
    assert "@acme/present" in str(excinfo.value)


def test_an_artifact_with_no_manifest_is_refused(tmp_path):
    (tmp_path / "acme" / "broken").mkdir(parents=True)

    with pytest.raises(ToolResolutionError, match="no usable manifest"):
        MountedToolResolver(tmp_path).resolve(ToolRef(slug="@acme/broken"))


def test_an_entry_naming_a_missing_attribute_names_the_attribute(tmp_path):
    mount_tool(
        tmp_path, "@acme/typo", entry="tool:run",
        body="def ran():\n    pass\n",
    )

    with pytest.raises(ToolResolutionError, match="'run'"):
        MountedToolResolver(tmp_path).resolve(ToolRef(slug="@acme/typo"))


def test_a_tool_that_raises_on_import_is_reported_and_not_left_in_sys_modules(tmp_path):
    # A half-initialised module left behind would be returned by the *next*
    # resolve of the same slug, turning a hard import error into a
    # AttributeError somewhere unrelated.
    mount_tool(
        tmp_path, "@acme/boom", entry="tool:run",
        body="raise RuntimeError('bad tool')\n",
    )

    with pytest.raises(ToolResolutionError, match="failed to import"):
        MountedToolResolver(tmp_path).resolve(ToolRef(slug="@acme/boom"))

    assert "agentinc_mounted_tools.acme_boom" not in sys.modules


def test_load_tools_uses_the_mount_when_the_engine_set_one(tmp_path, monkeypatch):
    # The whole point of selecting on the environment: agent code calls
    # `load_tools()` with no arguments, so nothing at the call site can choose.
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "agentinc.toml").write_text(
        '[agent]\nname = "a"\n\n[tools]\n"@agentinc/echo" = "*"\n'
    )
    mounted = tmp_path / "tools"
    mount_tool(
        mounted, "@agentinc/echo", entry="agentinc_tools.echo:echo",
        body="def echo(t):\n    return t\n", package=True,
    )

    monkeypatch.setenv("AGENTINC_AGENT_ROOT", str(agent))
    monkeypatch.setenv("AGENTINC_TOOLS_ROOT", str(mounted))

    tools = load_tools()

    assert [t("v") for t in tools] == ["v"]


def test_without_the_mount_variable_the_local_resolver_is_still_the_default(monkeypatch):
    monkeypatch.delenv("AGENTINC_TOOLS_ROOT", raising=False)
    assert isinstance(default_resolver(), LocalToolResolver)
