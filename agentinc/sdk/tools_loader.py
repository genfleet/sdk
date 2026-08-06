"""
`load_tools()` — turn an agent's manifest into callables it can use.

An agent names the tools it wants and receives them; it never imports them.
That indirection is the whole design: the same published agent runs
for two tenants against their own credentials, because what a name resolves to
is decided at run time rather than at `import` time.

Resolution is pluggable, and the two implementations differ only in where they
look:

* **local** (this module's default) — whatever tool providers are installed,
  secrets from the environment. What a developer gets on a laptop with no
  backend. Providers advertise themselves through an entry point group, so this
  SDK names no particular package.
* **platform** — the pinned digest from the registry, config and secrets from
  the tenant's installation. Supplied by the runtime, not from here.

The SDK ships only the local one. A resolver that reaches into tenant
installations belongs to the platform, and putting it here would drag the
registry across the open-source boundary.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .manifest import AgentManifest, ToolRef, load_agent_manifest


# Set by whatever unpacked the agent — the engine, a sandbox, a test harness.
# Exists because an agent's own idea of where it lives is computed when the
# code is written and can be wrong once the artifact is relocated.
AGENT_ROOT_ENV = "AGENTINC_AGENT_ROOT"

logger = logging.getLogger(__name__)


class ToolResolutionError(RuntimeError):
    """A named tool could not be turned into something callable."""


@runtime_checkable
class ToolResolver(Protocol):
    """How a reference becomes a callable. The platform supplies its own."""

    def resolve(self, ref: ToolRef) -> Callable: ...


# Any installed distribution can advertise tools by publishing an entry point
# in this group. Nothing here names a particular package on purpose: this SDK
# is open-source, and hardcoding one would point a public user at a repository
# they may not be able to install.
#
#     [project.entry-points."agentinc.tools"]
#     agentinc_tools = "agentinc_tools:discover"
#
# The named callable returns {slug: object with .load() -> Callable}.
TOOL_ENTRY_POINT_GROUP = "agentinc.tools"


class LocalToolResolver:
    """
    Resolve against whatever tool providers are installed.

    Deliberately ignores `ref.digest`. Local development is iteration — you are
    editing a tool and an agent together, and refusing to run because the
    working copy does not hash to a published digest would make that
    impossible. `ag agents pull` is the mode that does verify, for when
    the question is "what will the platform actually run".
    """

    def resolve(self, ref: ToolRef) -> Callable:
        installed = discover_installed_tools()
        if ref.slug not in installed:
            known = ", ".join(sorted(installed)) or "none"
            raise ToolResolutionError(
                f"{ref.slug} is not installed. Available: {known}. "
                "Tools come from a package that advertises the "
                f"'{TOOL_ENTRY_POINT_GROUP}' entry point group."
            )
        return installed[ref.slug].load()


def discover_installed_tools() -> dict[str, Any]:
    """
    Every tool advertised by an installed distribution, keyed by slug.

    A provider that fails to load must not hide the others: one broken package
    would otherwise make every tool on the machine unresolvable, and the error
    would name the wrong thing entirely. The failure is logged rather than
    raised, and the tools it would have supplied then fail individually with a
    message naming the tool actually being looked for.
    """
    found: dict[str, Any] = {}
    for entry_point in entry_points(group=TOOL_ENTRY_POINT_GROUP):
        try:
            provider = entry_point.load()
            found.update(provider())
        except Exception as exc:  # noqa: BLE001 — one bad provider, not all
            logger.warning(
                "tool provider %r failed to load and was skipped: %s",
                entry_point.name,
                exc,
            )
    return found


def load_tools(
    directory: Path | str | None = None,
    *,
    resolver: ToolResolver | None = None,
    manifest: AgentManifest | None = None,
) -> list[Callable]:
    """
    The tools named in an agent's `agentinc.toml`, ready to pass to `Agent`.

        from agentinc.sdk import Agent, load_tools

        def build_agent():
            return Agent(role="…", model={…}, tools=load_tools())

    Where the manifest comes from, in order:

    1. **`manifest=`** — the manifest object itself. This is the platform's
       path: it holds the *pinned* manifest from the published artifact, which
       may never touch this machine's filesystem at all.
    2. **`AGENTINC_AGENT_ROOT`** — set by whatever unpacked the agent. It wins
       over `directory` deliberately: after relocation the environment knows
       where the agent actually landed, and a path computed from `__file__`
       when the code was written is a guess that can be stale.
    3. **`directory`**, then the working directory — the local-development
       path, where the manifest really is sitting next to the code.

    `mcps` are *not* included: they are configuration rather than callables and
    reach `Agent` through its own `mcps=` argument.
    """
    if manifest is None:
        root = os.environ.get(AGENT_ROOT_ENV) or directory or "."
        spec = load_agent_manifest(root)
    else:
        spec = manifest
    active = resolver or LocalToolResolver()

    resolved: list[Callable] = []
    for ref in spec.tools:
        try:
            resolved.append(active.resolve(ref))
        except ToolResolutionError as exc:
            # Name the agent as well as the tool. With several agents in one
            # repo, "not installed" alone does not say which manifest to fix.
            raise ToolResolutionError(f"{spec.name}: {exc}") from exc
    return resolved
