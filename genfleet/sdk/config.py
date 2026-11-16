"""Reading what a workspace configured for this agent: settings, connectors and prompts.

An agent declares what can be configured in its ``agent.toml``:
``[settings.<key>]`` for values and ``[connectors.<slot>]`` for the
connections it needs. The workspace that installs the agent fills them in on
the agent's page, and may add prompts of its own. The agent reads all three
here::

    from genfleet.sdk.config import connector, setting, with_tenant_prompts

    def build_agent(remote_tools=()):
        erp = connector("erp")                 # None until the workspace picks one
        sink = setting("lead_sink", "views")   # the manifest default, else "views"
        return Agent(role=with_tenant_prompts(ROLE), ...)

On the platform the configuration is a JSON file whose path is in
``$GENFLEET_AGENT_CONFIG``. It is read once, when first asked for: the
platform restarts the agent when the workspace changes it. With the variable
unset (a local run) everything is empty: ``setting`` returns your default,
``connector`` returns ``None`` and there are no prompts. To try values
locally, write the file yourself and point the variable at it; its contract
(``version`` 1) is in ``tests/fixtures/agent-config.json``.

``setting`` returns the value the workspace saved, else the default declared
in the manifest, else the default you pass. ``is_set`` tells the first case
from the others. The platform checks saved values against the manifest's
types, so a value that is present has the declared type.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "AGENT_CONFIG_ENV",
    "AgentConfig",
    "AgentConfigError",
    "ConnectorBinding",
    "TenantPrompt",
    "agent_config",
    "connector",
    "is_set",
    "setting",
    "settings",
    "tenant_prompts",
    "with_tenant_prompts",
]

AGENT_CONFIG_ENV = "GENFLEET_AGENT_CONFIG"
_VERSION = 1
#: Refuse a file larger than this rather than read it into memory.
_MAX_BYTES = 2 * 1024 * 1024


class AgentConfigError(RuntimeError):
    """The configuration file is named but missing, too large, or not the expected shape."""


@dataclass(frozen=True)
class ConnectorBinding:
    """The connection a workspace chose for one of the agent's connector slots.

    ``connection`` is the name to pass to ``call_connection``. ``config`` is
    the connector's configuration for this agent: the connection's own, with
    the agent's overrides applied.
    """

    connection: str
    provider: str
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TenantPrompt:
    """A prompt the workspace added to the agent."""

    title: str
    body: str


@dataclass(frozen=True)
class AgentConfig:
    #: What the workspace saved.
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: The manifest's defaults.
    defaults: Mapping[str, Any] = field(default_factory=dict)
    connectors: Mapping[str, ConnectorBinding] = field(default_factory=dict)
    prompts: tuple[TenantPrompt, ...] = ()


def _object(value: Any, what: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AgentConfigError(f"{what} must be an object")
    return value


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise AgentConfigError(f"{what} must be a string")
    return value


def parse_agent_config(raw: Any) -> AgentConfig:
    """Turn the decoded file into an `AgentConfig`, refusing a shape it doesn't know."""
    document = _object(raw, "the agent configuration")
    version = document.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < _VERSION:
        raise AgentConfigError("the agent configuration has no version this SDK reads")

    connectors: dict[str, ConnectorBinding] = {}
    for slot, entry in _object(document.get("connectors"), "connectors").items():
        entry = _object(entry, f"connectors.{slot}")
        connectors[slot] = ConnectorBinding(
            connection=_text(entry.get("connection"), f"connectors.{slot}.connection"),
            provider=_text(entry.get("provider"), f"connectors.{slot}.provider"),
            config=_object(entry.get("config"), f"connectors.{slot}.config"),
        )

    prompts = document.get("prompts")
    if prompts is None:
        prompts = []
    if not isinstance(prompts, list):
        raise AgentConfigError("prompts must be a list")
    parsed = []
    for i, entry in enumerate(prompts):
        entry = _object(entry, f"prompts[{i}]")
        parsed.append(TenantPrompt(title=_text(entry.get("title", ""), f"prompts[{i}].title"), body=_text(entry.get("body"), f"prompts[{i}].body")))

    return AgentConfig(
        settings=_object(document.get("settings"), "settings"),
        defaults=_object(document.get("defaults"), "defaults"),
        connectors=connectors,
        prompts=tuple(parsed),
    )


@lru_cache(maxsize=4)
def _load(path: str) -> AgentConfig:
    file = Path(path)
    try:
        if file.stat().st_size > _MAX_BYTES:
            raise AgentConfigError(f"{path} is larger than {_MAX_BYTES} bytes")
        raw = json.loads(file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AgentConfigError(f"cannot read {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise AgentConfigError(f"{path} is not valid JSON") from exc
    return parse_agent_config(raw)


def agent_config() -> AgentConfig:
    """Everything the workspace configured. Empty when ``$GENFLEET_AGENT_CONFIG`` is unset."""
    path = os.environ.get(AGENT_CONFIG_ENV, "").strip()
    return _load(path) if path else AgentConfig()


def settings() -> Mapping[str, Any]:
    """Every setting that has a value: saved by the workspace, or the manifest's default."""
    config = agent_config()
    return {**config.defaults, **config.settings}


def setting(key: str, default: Any = None) -> Any:
    """One setting: the saved value, else the manifest's default, else ``default``."""
    config = agent_config()
    if key in config.settings:
        return config.settings[key]
    return config.defaults.get(key, default)


def is_set(key: str) -> bool:
    """Whether the workspace saved a value for ``key`` (a manifest default doesn't count)."""
    return key in agent_config().settings


def connector(slot: str) -> ConnectorBinding | None:
    """The connection filling ``slot``, or ``None`` when the workspace has not picked one."""
    return agent_config().connectors.get(slot)


def tenant_prompts() -> tuple[TenantPrompt, ...]:
    """The prompts the workspace added, in its order. Only the ones switched on."""
    return agent_config().prompts


def with_tenant_prompts(system: str) -> str:
    """``system`` followed by the workspace's prompts, each under its title.

    Returns ``system`` unchanged when the workspace added none.
    """
    added = []
    for prompt in tenant_prompts():
        body = prompt.body.strip()
        if body:
            added.append(f"## {prompt.title.strip()}\n\n{body}" if prompt.title.strip() else body)
    if not added:
        return system
    return "\n\n".join(([system.rstrip()] if system.strip() else []) + added)
