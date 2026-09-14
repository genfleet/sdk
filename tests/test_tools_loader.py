"""
Resolution is the seam the platform overrides, so it is exercised through a
real resolver rather than a mock — the Protocol is public API, and a test
implementation of it is the same thing the platform supplies.
"""

from __future__ import annotations

import pytest

from genfleet.sdk import tools_loader
from genfleet.sdk.manifest import ToolRef
from genfleet.sdk.tools_loader import (
    LocalToolResolver,
    ToolResolutionError,
    load_tools,
)


class StubResolver:
    """Returns a distinct callable per slug, and records what it was asked for."""

    def __init__(self, *, fails: set[str] | None = None) -> None:
        self.asked: list[str] = []
        self._fails = fails or set()

    def resolve(self, ref: ToolRef):
        self.asked.append(ref.slug)
        if ref.slug in self._fails:
            raise ToolResolutionError(f"{ref.slug} is not installed. Available: none")
        return lambda: ref.slug


def _manifest(tmp_path, body: str):
    (tmp_path / "genfleet.toml").write_text(body)
    return tmp_path


def test_tools_arrive_in_manifest_order(tmp_path):
    resolver = StubResolver()
    tools = load_tools(
        _manifest(
            tmp_path,
            '[agent]\nname = "a"\n\n[tools]\n'
            '"@acme/jira" = "*"\n"@acme/slack" = "*"\n',
        ),
        resolver=resolver,
    )

    assert resolver.asked == ["@acme/jira", "@acme/slack"]
    assert [t() for t in tools] == ["@acme/jira", "@acme/slack"]


def test_mcps_are_not_returned_as_callables(tmp_path):
    # They are configuration, and reach Agent through its own `mcps=` argument.
    # Returning them here would hand Agent a callable it cannot invoke.
    resolver = StubResolver()
    tools = load_tools(
        _manifest(
            tmp_path,
            '[agent]\nname = "a"\n\n[tools]\n"@acme/jira" = "*"\n'
            '\n[mcps]\n"@acme/gdrive" = "*"\n',
        ),
        resolver=resolver,
    )

    assert resolver.asked == ["@acme/jira"]
    assert len(tools) == 1


def test_agent_with_no_tools_resolves_to_nothing(tmp_path):
    assert load_tools(_manifest(tmp_path, '[agent]\nname = "solo"\n')) == []


def test_failure_names_the_agent_not_just_the_tool(tmp_path):
    # Several agents can share a repo; "not installed" alone does not say
    # which manifest to go fix.
    with pytest.raises(ToolResolutionError, match="support-triage: @acme/jira"):
        load_tools(
            _manifest(
                tmp_path,
                '[agent]\nname = "support-triage"\n\n[tools]\n"@acme/jira" = "*"\n',
            ),
            resolver=StubResolver(fails={"@acme/jira"}),
        )


def test_no_providers_installed_names_the_entry_point_group():
    # No tool provider is installed in the SDK's own environment — the
    # dependency runs the other way — so this is the real empty case.
    with pytest.raises(ToolResolutionError, match="genfleet.tools"):
        LocalToolResolver().resolve(ToolRef(slug="@acme/jira"))


def test_one_broken_provider_does_not_hide_the_others(monkeypatch, caplog):
    # A third-party package that fails to import would otherwise make every
    # tool on the machine unresolvable, with an error naming the wrong thing.
    class _Entry:
        def __init__(self, name, loader):
            self.name = name
            self._loader = loader

        def load(self):
            return self._loader

    def _explode():
        raise ImportError("provider is broken")

    def _working():
        return {"@acme/jira": _Stub()}

    class _Stub:
        def load(self):
            return lambda: "jira"

    monkeypatch.setattr(
        tools_loader,
        "entry_points",
        lambda group: [_Entry("broken", _explode), _Entry("good", _working)],
    )

    found = tools_loader.discover_installed_tools()

    assert list(found) == ["@acme/jira"]
    assert "broken" in caplog.text


def test_agent_root_env_overrides_the_hardcoded_path(tmp_path, monkeypatch):
    # An agent computes its root from __file__ when the code is written. Once
    # the runtime unpacks the artifact somewhere else, that path is a stale
    # guess and the environment is the authority. Raised in review on agents#3.
    relocated = tmp_path / "unpacked"
    relocated.mkdir()
    (relocated / "genfleet.toml").write_text(
        '[agent]\nname = "relocated"\n\n[tools]\n"@acme/jira" = "*"\n'
    )
    stale = tmp_path / "where-the-code-thinks-it-is"
    stale.mkdir()
    (stale / "genfleet.toml").write_text('[agent]\nname = "stale"\n')

    monkeypatch.setenv("GENFLEET_AGENT_ROOT", str(relocated))
    resolver = StubResolver()

    assert len(load_tools(stale, resolver=resolver)) == 1
    assert resolver.asked == ["@acme/jira"]


def test_an_explicit_manifest_needs_no_filesystem_at_all(tmp_path, monkeypatch):
    # The platform path: the pinned manifest comes from the published artifact
    # and may never touch this machine's disk.
    from genfleet.sdk.manifest import AgentManifest, ToolRef

    monkeypatch.setenv("GENFLEET_AGENT_ROOT", str(tmp_path / "does-not-exist"))
    resolver = StubResolver()

    tools = load_tools(
        manifest=AgentManifest(
            name="from-registry",
            tools=[ToolRef(slug="@acme/jira", version="1.0.0", digest="sha256:ab")],
        ),
        resolver=resolver,
    )

    assert resolver.asked == ["@acme/jira"]
    assert len(tools) == 1
