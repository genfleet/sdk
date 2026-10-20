"""`ToolContext` — a tool's own settings and credentials, without reading env.

A tool declares the settings and secrets it needs in its `genfleet.toml`
(`[config]`, `[secrets]`). Whoever runs the tool supplies the values, and the
tool reads them through a context it asks for by annotation::

    from genfleet.sdk import ToolContext, tool

    @tool
    async def create_issue(title: str, ctx: ToolContext) -> str:
        client = Jira(ctx.config["base_url"], token=ctx.secret("api_token"))
        ...

A parameter annotated `ToolContext` is not part of the tool's JSON schema, so
the model never sees it and cannot fill it in: the wrapper builds it at call
time, from the tool's slug.

Where the values come from:

* **On the platform**, the runtime puts each mounted tool's values in the
  environment under a namespace derived from its slug (`tool_env_namespace`):
  `<NS>__CONFIG` holds the config as a JSON object and `<NS>__SECRETS` the
  secrets as a JSON object. The context reads only its own namespace. Nothing
  else is consulted, so a tool never picks up an unrelated variable that
  happens to share a secret's name.
* **Locally**, the same two variables work, so a developer can reproduce the
  platform exactly. A secret missing from them falls back to an environment
  variable with the secret's own name (`api_token`, then `API_TOKEN`), which is
  what a laptop usually has.

This module only reads the environment. It never calls a registry or a
secrets service, and holds no credential of its own.

**What this is not.** A tool loaded with `load_tools()` runs in the agent's own
process, so these variables are readable by the agent's code and by every
other tool in that process. The namespace keeps tools from colliding; it is
not an isolation boundary.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

#: Every tool namespace starts with this.
TOOL_ENV_PREFIX = "GENFLEET_TOOL_"

#: Suffixes under a namespace. Distinct final letters, so a variable of one
#: namespace can never be spelled as another namespace plus a suffix.
CONFIG_SUFFIX = "__CONFIG"
SECRETS_SUFFIX = "__SECRETS"

# Set by the engine when it mounted tools (see `tools_loader.TOOLS_ROOT_ENV`).
# Its presence means "the platform supplied these values": the local fallback
# to bare variable names is then off.
_TOOLS_ROOT_ENV = "GENFLEET_TOOLS_ROOT"

_NON_NAME = re.compile(r"[^A-Z0-9_]")


class ToolSecretError(RuntimeError):
    """A tool asked for a secret it was not given. Names it, never a value."""


def tool_env_namespace(slug: str) -> str:
    """The environment prefix holding ``slug``'s config and secrets.

    ``@acme/jira-cloud`` → ``GENFLEET_TOOL_ACME__JIRA_CLOUD``: the ``@`` is
    dropped, ``/`` becomes ``__``, anything else that is not a letter, digit or
    ``_`` becomes ``_``, and the result is uppercased. Two slugs can map to
    one namespace (``@a/b-c`` and ``@a/b.c``); the runtime refuses to serve an
    agent whose tools collide rather than letting one read the other's values.
    """
    body = slug.removeprefix("@").upper().replace("/", "__")
    return TOOL_ENV_PREFIX + _NON_NAME.sub("_", body)


def _json_object(raw: str | None, *, what: str, slug: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        # The variable's name only: its content may be a credential.
        raise ToolSecretError(f"{slug or 'tool'}: {what} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise ToolSecretError(f"{slug or 'tool'}: {what} is not a JSON object.")
    return value


class ToolContext:
    """One tool's config and secrets, read from its own namespace.

    ``config`` is a read-only mapping. ``secret(name)`` returns the value or
    raises `ToolSecretError` naming the secret. Values are read when asked
    for, not copied in at construction.
    """

    def __init__(self, slug: str | None, env: Mapping[str, str] | None = None) -> None:
        self.slug = slug
        self._env = os.environ if env is None else env
        self._namespace = tool_env_namespace(slug) if slug else None

    @property
    def on_platform(self) -> bool:
        """Whether the runtime mounted this tool (no local fallbacks then)."""
        return bool(self._env.get(_TOOLS_ROOT_ENV))

    @property
    def config(self) -> Mapping[str, Any]:
        """The tool's settings: the tenant's installation on the platform."""
        if self._namespace is None:
            return MappingProxyType({})
        raw = self._env.get(self._namespace + CONFIG_SUFFIX)
        return MappingProxyType(_json_object(raw, what="config", slug=self.slug))

    def secret(self, name: str) -> str:
        """The value bound to the secret ``name`` this tool declares."""
        if self._namespace is not None:
            raw = self._env.get(self._namespace + SECRETS_SUFFIX)
            value = _json_object(raw, what="secrets", slug=self.slug).get(name)
            if isinstance(value, str):
                return value
        if not self.on_platform:
            for candidate in (name, name.upper()):
                value = self._env.get(candidate)
                if value is not None:
                    return value
        where = (
            "Bind it in the tool's settings, and for a marketplace agent grant it "
            "the tool's secrets."
            if self.on_platform
            else f"Set {self._namespace + SECRETS_SUFFIX if self._namespace else name} "
            f"or {name} in the environment."
        )
        raise ToolSecretError(f"{self.slug or 'tool'}: secret {name!r} is not available. {where}")

    def __repr__(self) -> str:
        # Never the values: a context ends up in tracebacks and logs.
        return f"ToolContext(slug={self.slug!r})"


__all__ = [
    "CONFIG_SUFFIX",
    "SECRETS_SUFFIX",
    "TOOL_ENV_PREFIX",
    "ToolContext",
    "ToolSecretError",
    "tool_env_namespace",
]
