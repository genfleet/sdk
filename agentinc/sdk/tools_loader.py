"""
`load_tools()` — turn an agent's manifest into callables it can use.

An agent names the tools it wants and receives them; it never imports them.
That indirection is the whole design: the same published agent runs
for two tenants against their own credentials, because what a name resolves to
is decided at run time rather than at `import` time.

Resolution is pluggable, and the two implementations differ only in where they
look:

* **local** (this module's default) — whatever is installed, secrets from the
  environment. What a developer gets on a laptop with no backend.
* **platform** — the pinned digest from the registry, config and secrets from
  the tenant's installation. Supplied by the runtime, not from here.

The SDK ships only the local one. A resolver that reaches into tenant
installations belongs to the platform, and putting it here would drag the
registry across the open-source boundary.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from .manifest import AgentManifest, ToolRef, load_agent_manifest


# Set by whatever unpacked the agent — the engine, a sandbox, a test harness.
# Exists because an agent's own idea of where it lives is computed when the
# code is written and can be wrong once the artifact is relocated.
AGENT_ROOT_ENV = "AGENTINC_AGENT_ROOT"


class ToolResolutionError(RuntimeError):
    """A named tool could not be turned into something callable."""


@runtime_checkable
class ToolResolver(Protocol):
    """How a reference becomes a callable. The platform supplies its own."""

    def resolve(self, ref: ToolRef) -> Callable: ...


class LocalToolResolver:
    """
    Resolve against the installed `agentinc_tools` package.

    Deliberately ignores `ref.digest`. Local development is iteration — you are
    editing a tool and an agent together, and refusing to run because the
    working copy does not hash to a published digest would make that
    impossible. `ag agents pull` is the mode that does verify, for when
    the question is "what will the platform actually run".
    """

    def resolve(self, ref: ToolRef) -> Callable:
        try:
            from agentinc_tools import discover
        except ImportError as exc:
            raise ToolResolutionError(
                f"{ref.slug} is referenced but agentinc-tools is not installed. "
                "Run `uv add agentinc-tools`."
            ) from exc

        installed = discover()
        if ref.slug not in installed:
            known = ", ".join(sorted(installed)) or "none"
            raise ToolResolutionError(
                f"{ref.slug} is not installed. Available: {known}"
            )
        return installed[ref.slug].load()


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
