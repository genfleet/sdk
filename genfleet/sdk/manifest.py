"""
`genfleet.toml` — what an agent depends on, and what a tool offers.

Two manifests share this module because they are two halves of one contract:
an agent names the tools it wants, a tool declares what it needs to run. The
parser is here rather than in the platform because developers author these
files by hand and must be able to validate them offline.

The central rule, from ADR-0007: **an agent's manifest names tools, and the
publisher records digests.** A developer writes `"@acme/jira" = "*"`; `push`
resolves that to a concrete version and digest and records both in the
published artifact. Nobody types a hash, and what ships is still pinned — the
same split as `package.json` and its lockfile.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MANIFEST_NAME = "genfleet.toml"

# `@scope/name` — the scope is the publishing tenant's namespace, which the
# backend already enforces at submit time (backend#74).
SLUG_PATTERN = re.compile(r"^@[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9._-]*$")

AgentKind = Literal["declarative", "code", "native_sdk"]
ToolKind = Literal["cli", "http", "mcp_stdio", "mcp_http"]


class ManifestError(ValueError):
    """Raised with the file path attached, because these are hand-authored."""

    def __init__(self, path: Path | str, problem: str) -> None:
        super().__init__(f"{path}: {problem}")
        self.path = str(path)
        self.problem = problem


class ToolRef(BaseModel):
    """
    One entry in an agent's `[tools]` or `[mcps]` table.

    `spec` is what the developer wrote — `"*"`, `"1.2.x"`, an exact version.
    `version` and `digest` are empty until `push` resolves them, and are what
    the published artifact carries. An unresolved ref is fine on disk and
    invalid in a published artifact; `is_pinned` is the difference.
    """

    slug: str
    spec: str = "*"
    version: str | None = None
    digest: str | None = None

    @field_validator("slug")
    @classmethod
    def _slug_is_well_formed(cls, value: str) -> str:
        if not SLUG_PATTERN.match(value):
            raise ValueError(
                f"'{value}' is not a tool slug. Expected @scope/name, "
                "lowercase, e.g. @acme/jira"
            )
        return value

    @property
    def is_pinned(self) -> bool:
        return self.version is not None and self.digest is not None


class ConfigField(BaseModel):
    """A value the installing tenant supplies — a URL, a project key, a region."""

    type: Literal["string", "number", "boolean"] = "string"
    required: bool = False
    description: str = ""
    default: Any = None


class SecretField(BaseModel):
    """
    A credential the installing tenant binds, declared by **name only**.

    There is deliberately no `value` field. A tool manifest is committed to a
    repository and published to a registry readable by every tenant, so a
    manifest that could carry a secret would eventually carry one. The parser
    rejects the key outright rather than trusting that nobody tries.
    """

    description: str = ""
    required: bool = True


class AgentManifest(BaseModel):
    """The `genfleet.toml` sitting next to an agent's entry point."""

    name: str
    kind: AgentKind = "code"
    description: str = ""
    entry: str = "agent:build_agent"
    tools: list[ToolRef] = Field(default_factory=list)
    mcps: list[ToolRef] = Field(default_factory=list)

    @property
    def is_fully_pinned(self) -> bool:
        """Every dependency resolved — the precondition for publishing."""
        return all(ref.is_pinned for ref in (*self.tools, *self.mcps))


class ToolManifest(BaseModel):
    """The `genfleet.toml` sitting next to a tool's implementation."""

    slug: str
    version: str
    kind: ToolKind = "cli"
    description: str = ""
    entry: str
    config: dict[str, ConfigField] = Field(default_factory=dict)
    secrets: dict[str, SecretField] = Field(default_factory=dict)

    @field_validator("slug")
    @classmethod
    def _slug_is_well_formed(cls, value: str) -> str:
        if not SLUG_PATTERN.match(value):
            raise ValueError(f"'{value}' is not a tool slug. Expected @scope/name")
        return value


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(path, f"no {MANIFEST_NAME} here")
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(path, f"not valid TOML — {exc}") from exc


def _parse_refs(table: Any, section: str, path: Path) -> list[ToolRef]:
    """
    Turn a `[tools]` / `[mcps]` table into refs.

    Two spellings are accepted because both are natural to write:

        "@acme/jira" = "*"                        # just a version spec
        "@acme/jira" = { spec = "1.2.x" }         # room for more keys later

    A bare string is the common case and stays the short one.
    """
    if not table:
        return []
    if not isinstance(table, dict):
        raise ManifestError(path, f"[{section}] must be a table")

    refs: list[ToolRef] = []
    for slug, value in table.items():
        if isinstance(value, str):
            fields: dict[str, Any] = {"spec": value}
        elif isinstance(value, dict):
            fields = dict(value)
        else:
            raise ManifestError(
                path, f"[{section}] '{slug}' must be a version string or a table"
            )
        try:
            refs.append(ToolRef(slug=slug, **fields))
        except ValueError as exc:
            raise ManifestError(path, f"[{section}] {exc}") from exc
    return refs


def load_agent_manifest(directory: Path | str) -> AgentManifest:
    """Read and validate the `genfleet.toml` in an agent directory."""
    path = Path(directory) / MANIFEST_NAME
    raw = _read_toml(path)

    agent = raw.get("agent")
    if not isinstance(agent, dict):
        raise ManifestError(path, "missing an [agent] section")

    try:
        return AgentManifest(
            **agent,
            tools=_parse_refs(raw.get("tools"), "tools", path),
            mcps=_parse_refs(raw.get("mcps"), "mcps", path),
        )
    except ValueError as exc:
        raise ManifestError(path, str(exc)) from exc


def load_tool_manifest(directory: Path | str) -> ToolManifest:
    """Read and validate the `genfleet.toml` in a tool directory."""
    path = Path(directory) / MANIFEST_NAME
    raw = _read_toml(path)

    tool = raw.get("tool")
    if not isinstance(tool, dict):
        raise ManifestError(path, "missing a [tool] section")

    secrets = raw.get("secrets") or {}
    for name, declaration in secrets.items():
        # See SecretField: a manifest is committed and published, so a secret
        # value in one is a leak the moment it is written, not when it is used.
        if isinstance(declaration, dict) and "value" in declaration:
            raise ManifestError(
                path,
                f"[secrets] '{name}' has a value. Manifests declare secret "
                "names only — the tenant binds the value at install time.",
            )

    try:
        return ToolManifest(
            **tool,
            config=raw.get("config") or {},
            secrets=secrets,
        )
    except ValueError as exc:
        raise ManifestError(path, str(exc)) from exc
