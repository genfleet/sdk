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

What you get back is yours to change: a list or table is a copy.

``setting`` returns the value the workspace saved, else the default declared
in the manifest, else the default you pass. ``is_set`` tells the first case
from the others. The platform checks saved values against the manifest's
types, so a value that is present has the declared type.

The file also lists the **connector tools** the agent may call: the verified
tools of each connection the workspace let it use, and the built-in tools of
the channels it is connected to. ``connector_tools()`` turns them into tools
for the model (see :mod:`genfleet.sdk.connector_calls`). A file written by an
older platform has no tools: ``connector_tools()`` is then empty.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .connector_calls import ConnectorTool

__all__ = [
    "AGENT_CONFIG_ENV",
    "AgentConfig",
    "AgentConfigError",
    "ConnectionGrant",
    "ConnectorBinding",
    "ConnectorToolSpec",
    "TenantPrompt",
    "agent_config",
    "connector",
    "connector_tools",
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

#: A tool name the model accepts: letters, digits, ``_`` and ``-``, at most 64.
_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_DEFINITION_HASH = re.compile(r"[0-9a-f]{64}")

log = logging.getLogger("genfleet.sdk.config")


class AgentConfigError(RuntimeError):
    """The configuration file is named but missing, too large, or not the expected shape."""


@dataclass(frozen=True)
class ConnectorToolSpec:
    """One connector tool the platform lists for the agent.

    ``name`` is what the model sees, and the name the workspace's audience
    and approval settings use: ``<connection>__<tool>`` for a tool of a
    connection, the bare name (``reply``) for a channel's built-in tool. It
    is used exactly as listed, never renamed or shortened. ``tool`` and
    ``definition_hash`` identify the tool and the exact definition the
    workspace verified when it is called.
    """

    name: str
    tool: str
    description: str
    input_schema: Mapping[str, Any]
    definition_hash: str


@dataclass(frozen=True)
class ConnectorBinding:
    """The connection a workspace chose for one of the agent's connector slots.

    ``connection`` is the name to pass to ``call_connection``. ``config`` is
    the connector's configuration for this agent: the connection's own, with
    the agent's overrides applied. ``tools`` are the connection's verified
    tools (empty on an older platform).
    """

    connection: str
    provider: str
    config: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[ConnectorToolSpec, ...] = ()


@dataclass(frozen=True)
class ConnectionGrant:
    """A connection the workspace let the agent use, in a slot or not, and its verified tools."""

    provider: str
    slot: str | None = None
    tools: tuple[ConnectorToolSpec, ...] = ()


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
    #: Every connection the agent may use, by connection name.
    connections: Mapping[str, ConnectionGrant] = field(default_factory=dict)
    #: The built-in tools of each channel the agent is connected to, by channel.
    channels: Mapping[str, tuple[ConnectorToolSpec, ...]] = field(default_factory=dict)


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


def _tool(entry: Any, where: str) -> ConnectorToolSpec | None:
    """One listed tool, or None (logged by its place, never its text) when it is unusable.

    A tool is skipped rather than failing the whole file: the agent keeps
    serving without it. A name the model would refuse is never renamed.
    """
    if not isinstance(entry, dict):
        log.warning("skipping a connector tool at %s: not an object", where)
        return None
    name, tool, description = entry.get("name"), entry.get("tool"), entry.get("description")
    schema, digest = entry.get("input_schema"), entry.get("definition_hash")
    if not (isinstance(name, str) and _TOOL_NAME.fullmatch(name)):
        log.warning("skipping a connector tool at %s: unusable name", where)
        return None
    if not (
        isinstance(tool, str) and tool
        and isinstance(description, str)
        and isinstance(schema, dict) and schema.get("type") == "object"
        and isinstance(digest, str) and _DEFINITION_HASH.fullmatch(digest)
    ):
        log.warning("skipping connector tool %s at %s: malformed entry", name, where)
        return None
    return ConnectorToolSpec(
        name=name,
        tool=tool,
        description=description,
        input_schema=MappingProxyType(copy.deepcopy(schema)),
        definition_hash=digest,
    )


def _tools(value: Any, where: str) -> tuple[ConnectorToolSpec, ...]:
    """A ``tools`` list: absent is none; each usable entry once, by name, first kept."""
    if value is None:
        return ()
    if not isinstance(value, list):
        log.warning("skipping the connector tools at %s: not a list", where)
        return ()
    seen: set[str] = set()
    tools = []
    for i, entry in enumerate(value):
        spec = _tool(entry, f"{where}[{i}]")
        if spec is not None and spec.name not in seen:
            seen.add(spec.name)
            tools.append(spec)
    return tuple(tools)


def _listing(value: Any, what: str) -> dict[str, Any]:
    """``connections`` or ``channels``: only tools hang off them, so a broken
    one is skipped (logged) rather than failing settings, connectors and prompts."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        log.warning("skipping %s: not an object", what)
        return {}
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
            config=MappingProxyType(_object(entry.get("config"), f"connectors.{slot}.config")),
            tools=_tools(entry.get("tools"), f"connectors.{slot}.tools"),
        )

    connections: dict[str, ConnectionGrant] = {}
    for name, entry in _listing(document.get("connections"), "connections").items():
        granted_slot = entry.get("slot") if isinstance(entry, dict) else None
        if not (isinstance(entry, dict) and isinstance(entry.get("provider"), str)
                and (granted_slot is None or isinstance(granted_slot, str))):
            log.warning("skipping connections.%s: malformed entry", name)
            continue
        connections[name] = ConnectionGrant(
            provider=entry["provider"],
            slot=granted_slot,
            tools=_tools(entry.get("tools"), f"connections.{name}.tools"),
        )

    channels: dict[str, tuple[ConnectorToolSpec, ...]] = {}
    for provider, entry in _listing(document.get("channels"), "channels").items():
        if not isinstance(entry, dict):
            log.warning("skipping channels.%s: malformed entry", provider)
            continue
        channels[provider] = _tools(entry.get("tools"), f"channels.{provider}.tools")

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
        settings=MappingProxyType(_object(document.get("settings"), "settings")),
        defaults=MappingProxyType(_object(document.get("defaults"), "defaults")),
        connectors=MappingProxyType(connectors),
        prompts=tuple(parsed),
        connections=MappingProxyType(connections),
        channels=MappingProxyType(channels),
    )


@lru_cache(maxsize=4)
def _load(path: str) -> AgentConfig:
    try:
        with Path(path).open("rb") as handle:
            # One bounded read: the size cannot change between a check and the read.
            data = handle.read(_MAX_BYTES + 1)
    except OSError as exc:
        raise AgentConfigError(f"cannot read {path}: {exc.strerror or exc}") from exc
    if len(data) > _MAX_BYTES:
        raise AgentConfigError(f"{path} is larger than {_MAX_BYTES} bytes")
    try:
        raw = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise AgentConfigError(f"{path} is not valid JSON") from exc
    return parse_agent_config(raw)


def agent_config() -> AgentConfig:
    """Everything the workspace configured. Empty when ``$GENFLEET_AGENT_CONFIG`` is unset."""
    path = os.environ.get(AGENT_CONFIG_ENV, "").strip()
    return _load(path) if path else AgentConfig()


def settings() -> Mapping[str, Any]:
    """Every setting that has a value: saved by the workspace, or the manifest's default."""
    config = agent_config()
    return copy.deepcopy({**config.defaults, **config.settings})


def setting(key: str, default: Any = None) -> Any:
    """One setting: the saved value, else the manifest's default, else ``default``."""
    config = agent_config()
    if key in config.settings:
        return copy.deepcopy(config.settings[key])
    if key in config.defaults:
        return copy.deepcopy(config.defaults[key])
    return default


def is_set(key: str) -> bool:
    """Whether the workspace saved a value for ``key`` (a manifest default doesn't count)."""
    return key in agent_config().settings


def connector(slot: str) -> ConnectorBinding | None:
    """The connection filling ``slot``, or ``None`` when the workspace has not picked one."""
    return agent_config().connectors.get(slot)


def connector_tools(slot: str | None = None) -> list[ConnectorTool]:
    """The connector tools for the model: every connection's, or one slot's.

    See :func:`genfleet.sdk.connector_calls.connector_tools`.
    """
    from .connector_calls import connector_tools as _connector_tools

    return _connector_tools(slot)


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
        # A title is one line: a line break in it would start a heading of its own.
        title = " ".join(prompt.title.split())
        if body:
            added.append(f"## {title}\n\n{body}" if title else body)
    if not added:
        return system
    return "\n\n".join(([system.rstrip()] if system.strip() else []) + added)
