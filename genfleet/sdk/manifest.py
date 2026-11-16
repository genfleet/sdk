"""
`agent.toml` and a tool's `genfleet.toml` — what an agent is and depends on, and what a tool offers.

An agent's manifest is `agent.toml`. It was `genfleet.toml` before 0.28.0, and
that name is still read: a directory holding both is an error.

Two manifests share this module because they are two halves of one contract:
an agent names the tools it wants, a tool declares what it needs to run. The
parser is here rather than in the platform because developers author these
files by hand and must be able to validate them offline.

The central rule: **an agent's manifest names tools, and the
publisher records digests.** A developer writes `"@acme/jira" = "*"`; when
the agent is submitted to the marketplace, that spec is resolved to a concrete
version and digest, recorded alongside the published artifact. Nobody types a
hash, and what ships is still pinned — the same split as `package.json` and
its lockfile.
"""

from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, ValidationError, field_validator, model_validator

from .content import content_glob_problem

#: A tool's manifest, and the name an agent's manifest had before 0.28.0.
MANIFEST_NAME = "genfleet.toml"
#: An agent's manifest.
AGENT_MANIFEST_NAME = "agent.toml"

#: At most this many globs in ``[data] data``.
MAX_CONTENT_DATA_GLOBS = 100
#: At most this many ``[connectors.<slot>]`` tables.
MAX_CONNECTOR_SLOTS = 8
#: At most this many ``[settings.<key>]`` tables.
MAX_SETTINGS = 40
#: Limits on a setting's value, by type.
MAX_SETTING_STRING = 500
MAX_SETTING_TEXT = 20_000
MAX_SETTING_ITEMS = 200
MAX_SETTING_OPTIONS = 50

# A connector slot, a setting key, and a connector provider share one shape.
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# `@scope/name` — the scope is the publishing tenant's namespace, which the
# publishing service already enforces at submit time.
SLUG_PATTERN = re.compile(r"^@[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9._-]*$")

AgentKind = Literal["declarative", "code", "native_sdk"]
ToolKind = Literal["cli", "http", "mcp_stdio", "mcp_http"]


#: A platform model id.
_CATALOG_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9.-]{1,63}")
_MAX_MODELS = 20


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
    `version` and `digest` are empty until marketplace submission resolves
    them, and are what the published artifact carries. An unresolved ref is fine on disk and
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


class ContentSpec(BaseModel):
    """The agent's `[data]` table (`[content]` before 0.28.0): whether it needs content, and which files its code reads.

    A generic agent has no business of its own: the workspace that installs
    it supplies its content (a persona, knowledge, rules). With
    ``required = true`` the platform does not serve the agent until that
    content is in place. ``data`` lists globs, relative to the content root,
    of the files the agent's code reads by path (``knowledge.json``,
    ``flows/**``): the workspace can't change how those are used, and they
    are never returned by ``genfleet.sdk.content.loaded_files``. Unknown keys
    are refused, so a misspelt ``required`` is an error rather than a silent
    ``false``.
    """

    model_config = ConfigDict(extra="forbid")

    required: StrictBool = False
    data: list[StrictStr] = Field(default_factory=list, max_length=MAX_CONTENT_DATA_GLOBS)

    @field_validator("data")
    @classmethod
    def _data_globs_are_valid(cls, value: list[str]) -> list[str]:
        for glob in value:
            problem = content_glob_problem(glob)
            if problem is not None:
                raise ValueError(f"glob {glob!r} {problem}")
        if len(set(value)) != len(value):
            raise ValueError("lists a glob twice")
        return value


SettingType = Literal["string", "text", "integer", "number", "boolean", "enum", "list", "map"]


def _is_number(value: Any) -> bool:
    """A finite int or float. Not a bool, and not NaN or an infinity (neither is JSON)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _plain(number: float) -> str:
    """A bound as its author wrote it: ``3`` not ``3.0``, ``1000000`` not ``1e+06``."""
    return str(int(number)) if float(number).is_integer() else repr(float(number))


def setting_value_problem(spec: SettingSpec, value: Any) -> str | None:
    """Why ``value`` doesn't fit the setting, or ``None`` when it does.

    The platform applies the same rules when a workspace saves a value.
    """
    kind = spec.type
    if kind in ("string", "text"):
        limit = MAX_SETTING_STRING if kind == "string" else MAX_SETTING_TEXT
        if not isinstance(value, str):
            return "must be a string"
        if len(value) > limit:
            return f"must be at most {limit} characters"
    elif kind == "boolean":
        if not isinstance(value, bool):
            return "must be true or false"
    elif kind in ("integer", "number"):
        if not _is_number(value) or (kind == "integer" and not isinstance(value, int)):
            return "must be a whole number" if kind == "integer" else "must be a number"
        if spec.min is not None and value < spec.min:
            return f"must be at least {_plain(spec.min)}"
        if spec.max is not None and value > spec.max:
            return f"must be at most {_plain(spec.max)}"
    elif kind == "enum":
        if value not in (spec.options or []):
            return "must be one of " + ", ".join(repr(o) for o in spec.options or [])
    elif kind == "list":
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return "must be a list of strings"
        if len(value) > MAX_SETTING_ITEMS or any(len(item) > MAX_SETTING_STRING for item in value):
            return f"must hold at most {MAX_SETTING_ITEMS} strings of at most {MAX_SETTING_STRING} characters"
    elif kind == "map":
        if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
            return "must be a table of strings"
        if len(value) > MAX_SETTING_ITEMS or any(len(k) > MAX_SETTING_STRING or len(v) > MAX_SETTING_STRING for k, v in value.items()):
            return f"must hold at most {MAX_SETTING_ITEMS} entries of at most {MAX_SETTING_STRING} characters"
    return None


class SettingSpec(BaseModel):
    """One `[settings.<key>]` table: a value the installing workspace sets on the agent's page.

    The agent reads it with ``genfleet.sdk.config.setting("<key>")``. ``type``
    is ``string`` (one line), ``text`` (long), ``integer``, ``number``,
    ``boolean``, ``enum`` (one of ``options``), ``list`` (of strings) or
    ``map`` (string to string). A setting never holds a credential: declare
    those as secrets.
    """

    model_config = ConfigDict(extra="forbid")

    type: SettingType = "string"
    label: StrictStr = Field(default="", max_length=60)
    description: StrictStr = Field(default="", max_length=300)
    group: StrictStr = Field(default="", max_length=40)
    required: StrictBool = False
    default: Any = None
    options: list[StrictStr] | None = None
    min: StrictInt | StrictFloat | None = None
    max: StrictInt | StrictFloat | None = None

    @model_validator(mode="after")
    def _fields_fit_the_type(self) -> SettingSpec:
        if self.type == "enum":
            if not self.options or len(self.options) > MAX_SETTING_OPTIONS or len(set(self.options)) != len(self.options):
                raise ValueError(f"an enum needs 'options': 1 to {MAX_SETTING_OPTIONS} distinct strings")
            if any(not option or len(option) > MAX_SETTING_STRING for option in self.options):
                raise ValueError(f"each of 'options' is 1 to {MAX_SETTING_STRING} characters")
        elif self.options is not None:
            raise ValueError("'options' is only for type = \"enum\"")
        if self.type not in ("integer", "number") and (self.min is not None or self.max is not None):
            raise ValueError("'min' and 'max' are only for integer and number settings")
        for name, bound in (("min", self.min), ("max", self.max)):
            if bound is None:
                continue
            if not math.isfinite(bound):
                raise ValueError(f"'{name}' must be a finite number")
            if self.type == "integer" and not isinstance(bound, int):
                raise ValueError(f"'{name}' of an integer setting must be a whole number")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("'min' is above 'max'")
        if self.default is not None:
            problem = setting_value_problem(self, self.default)
            if problem is not None:
                raise ValueError(f"'default' {problem}")
        return self


class ConnectorSlot(BaseModel):
    """One `[connectors.<slot>]` table: a connection the agent needs from its workspace.

    The workspace picks which of its connections fills the slot, and the agent
    asks for it by slot: ``genfleet.sdk.config.connector("<slot>")``.
    ``provider`` is the kind of connection, e.g. ``custom_api``.
    """

    model_config = ConfigDict(extra="forbid")

    provider: StrictStr
    required: StrictBool = True
    description: StrictStr = Field(default="", max_length=300)

    @field_validator("provider")
    @classmethod
    def _provider_is_a_key(cls, value: str) -> str:
        if not _KEY.fullmatch(value):
            raise ValueError("provider must be a connector id such as 'custom_api'")
        return value


_VIEW_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_VIEW_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._~-]+$")
_ENTRY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
MAX_VIEWS = 8


class ViewSpec(BaseModel):
    """One `[[views]]` entry: a page the agent serves, shown to the workspace in the app.

    ``id`` names it; ``title`` is the tab label (the id when left out);
    ``path`` is the page, and everything it loads lives under it;
    ``min_role`` is who may open it: ``owner``, ``admin`` (owners and admins,
    the default) or ``member`` (every member); ``editable`` lists the
    workspace content files the view may read and save through the platform
    (exact paths, at most 16; saving is for owners and admins). The platform
    checks the same rules when the agent is published.
    """

    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    title: StrictStr | None = None
    path: StrictStr
    min_role: Literal["owner", "admin", "member"] = "admin"
    editable: list[StrictStr] = []

    @field_validator("id")
    @classmethod
    def _id_is_a_slug(cls, value: str) -> str:
        if not _VIEW_ID.match(value):
            raise ValueError("id must be 1-40 lowercase letters, digits or '-'")
        return value

    @field_validator("title")
    @classmethod
    def _title_fits(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or len(value) > 60):
            raise ValueError("title must be 1-60 characters")
        return value.strip() if value is not None else None

    @field_validator("path")
    @classmethod
    def _path_is_clean(cls, value: str) -> str:
        if not value.startswith("/") or len(value) > 200:
            raise ValueError("path must start with '/' and be at most 200 characters")
        if value != "/":
            segments = value[1:].rstrip("/").split("/")
            if any(not _VIEW_PATH_SEGMENT.match(s) or s in (".", "..") for s in segments):
                raise ValueError("path may hold only letters, digits and . _ ~ - between single slashes")
        if value.lower() == "/__gf" or value.lower().startswith("/__gf/"):
            raise ValueError("path may not start with /__gf (the platform's own)")
        return value

    @field_validator("editable")
    @classmethod
    def _editable_are_content_paths(cls, value: list[str]) -> list[str]:
        if len(value) > 16:
            raise ValueError("editable lists at most 16 content files")
        for path in value:
            if not path or len(path) > 200 or any(not _VIEW_PATH_SEGMENT.match(s) or s in (".", "..") for s in path.split("/")):
                raise ValueError("editable entries are content paths: letters, digits and . _ ~ - between single slashes, no leading slash")
        if len(set(value)) != len(value):
            raise ValueError("editable lists a path twice")
        return value


class AgentManifest(BaseModel):
    """The `agent.toml` sitting next to an agent's entry point."""

    name: str
    kind: AgentKind = "code"
    description: str = ""
    entry: str = "agent:build_agent"
    tools: list[ToolRef] = Field(default_factory=list)
    mcps: list[ToolRef] = Field(default_factory=list)
    #: Who the agent serves on channels: ``operator`` (tenant staff),
    #: ``customer`` (the tenant's customers), or both. ``None`` when the
    #: manifest does not say; the platform then serves operators only.
    audiences: list[Literal["operator", "customer"]] | None = None
    #: The ``[data]`` table. Absent means the agent needs no content.
    content: ContentSpec = Field(default_factory=ContentSpec)
    #: The ``[connectors.<slot>]`` tables, by slot.
    connectors: dict[str, ConnectorSlot] = Field(default_factory=dict)
    #: The ``[settings.<key>]`` tables, by key.
    settings: dict[str, SettingSpec] = Field(default_factory=dict)
    #: ``module:symbol`` of the ASGI app serving the agent's views
    #: (``genfleet.sdk.views.Views``), next to the manifest.
    views_app: str | None = None
    #: The ``[[views]]`` list. Needs ``views_app``.
    views: list[ViewSpec] = Field(default_factory=list, max_length=MAX_VIEWS)

    @field_validator("views_app")
    @classmethod
    def _views_app_is_an_entry(cls, value: str | None) -> str | None:
        if value is not None and not _ENTRY.match(value):
            raise ValueError("views_app must be 'module:symbol', e.g. 'sales_views:views'")
        return value

    @field_validator("views")
    @classmethod
    def _view_ids_are_unique(cls, value: list[ViewSpec]) -> list[ViewSpec]:
        if len({v.id for v in value}) != len(value):
            raise ValueError("[[views]] gives two views the same id")
        return value

    @model_validator(mode="after")
    def _views_need_an_app(self) -> "AgentManifest":
        if self.views and self.views_app is None:
            raise ValueError("[[views]] needs [agent] views_app, the app that serves them")
        return self

    @field_validator("audiences")
    @classmethod
    def _audiences_are_distinct(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("audiences: list at least one of 'operator', 'customer', or leave it out")
        return list(dict.fromkeys(value))

    #: The agent reads its model id from the ``AGENT_MODEL`` environment
    #: variable, so a workspace owner or admin may choose the model on the
    #: agent's page. Leave it ``False`` when the code ignores ``AGENT_MODEL``:
    #: a choice would then change nothing.
    model_choice: StrictBool = False

    #: Platform model ids the agent calls. In hosted
    #: execution the platform scopes the agent's model credential to these.
    #: A ``provider/model`` id is the agent's own business and isn't listed.
    models: list[str] = Field(default_factory=list)

    @field_validator("models")
    @classmethod
    def _models_are_catalog_ids(cls, value: list[str]) -> list[str]:
        if len(value) > _MAX_MODELS:
            raise ValueError(f"models: at most {_MAX_MODELS} entries")
        out: list[str] = []
        for entry in value:
            model = entry.strip().lower()
            if not _CATALOG_MODEL_ID.fullmatch(model):
                raise ValueError(
                    "models: each entry is a platform model id such as 'catalog-model' "
                    "(lowercase letters, digits, '.' and '-'; no 'provider/' prefix)"
                )
            if model not in out:
                out.append(model)
        return out

    @property
    def is_fully_pinned(self) -> bool:
        """Every dependency resolved, as marketplace submission leaves it."""
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
    #: The author's marking. A call waits for an owner's
    #: approval. The platform reads this at publish as the tool's default and
    #: sends the effective flag on every turn (`genfleet.sensitive_tools`).
    sensitive: bool = False

    @field_validator("slug")
    @classmethod
    def _slug_is_well_formed(cls, value: str) -> str:
        if not SLUG_PATTERN.match(value):
            raise ValueError(f"'{value}' is not a tool slug. Expected @scope/name")
        return value


#: Top-level tables of an agent manifest: never keys of ``[agent]``.
_OWN_TABLES = frozenset({"tools", "mcps", "content", "connectors", "settings", "views"})


def agent_manifest_path(directory: Path | str) -> Path:
    """The agent manifest in ``directory``: `agent.toml`, else the older `genfleet.toml`.

    Raises `ManifestError` when the directory holds both or neither.
    """
    current = Path(directory) / AGENT_MANIFEST_NAME
    legacy = Path(directory) / MANIFEST_NAME
    if current.is_file() and legacy.is_file():
        raise ManifestError(current, f"this directory also has a {MANIFEST_NAME}; keep only {AGENT_MANIFEST_NAME}")
    if current.is_file() or not legacy.is_file():
        return current
    return legacy


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(path, f"no {path.name} here")
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
    """Read and validate the `agent.toml` (or older `genfleet.toml`) in an agent directory."""
    path = agent_manifest_path(directory)
    raw = _read_toml(path)

    agent = raw.get("agent")
    if not isinstance(agent, dict):
        raise ManifestError(path, "missing an [agent] section")
    misplaced = sorted(_OWN_TABLES & agent.keys())
    if misplaced:
        raise ManifestError(path, f"[agent] has {misplaced[0]!r}, which is its own table: write [{misplaced[0]}] at the top level")

    try:
        return AgentManifest(
            **agent,
            tools=_parse_refs(raw.get("tools"), "tools", path),
            mcps=_parse_refs(raw.get("mcps"), "mcps", path),
            content=_parse_data(raw, path),
            connectors=_parse_keyed(raw.get("connectors"), "connectors", ConnectorSlot, MAX_CONNECTOR_SLOTS, path),
            settings=_parse_keyed(raw.get("settings"), "settings", SettingSpec, MAX_SETTINGS, path),
            views=_parse_views(raw.get("views"), path),
        )
    except ManifestError:
        raise
    except ValueError as exc:
        raise ManifestError(path, str(exc)) from exc


def _parse_views(table: Any, path: Path) -> list[ViewSpec]:
    if table is None:
        return []
    if not isinstance(table, list):
        raise ManifestError(path, "[[views]] must be an array of tables")
    if len(table) > MAX_VIEWS:
        raise ManifestError(path, f"[[views]] lists more than {MAX_VIEWS} views")
    views = []
    for i, entry in enumerate(table, start=1):
        if not isinstance(entry, dict):
            raise ManifestError(path, f"[[views]] #{i} must be a table")
        try:
            views.append(ViewSpec(**entry))
        except ValidationError as exc:
            problems = "; ".join(
                f"unknown key {e['loc'][0]!r}" if e["type"] == "extra_forbidden" else f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                for e in exc.errors()
            )
            raise ManifestError(path, f"[[views]] #{i}: {problems}") from None
    return views


def _data_problem(error: Any) -> str:
    if error["type"] == "value_error":
        return "'data' " + str(error["ctx"]["error"])
    if error["type"] == "too_long":
        return f"'data' lists more than {MAX_CONTENT_DATA_GLOBS} globs"
    return "'data' must be a list of glob strings"


def _problems(exc: ValidationError) -> str:
    out = []
    for error in exc.errors():
        key = ".".join(str(part) for part in error["loc"])
        if error["type"] == "extra_forbidden":
            out.append(f"unknown key {key!r}")
        elif error["type"] == "value_error":
            out.append(str(error["ctx"]["error"]))
        else:
            out.append(f"{key}: {error['msg']}" if key else error["msg"])
    return "; ".join(out)


def _parse_keyed(table: Any, section: str, model: type[BaseModel], limit: int, path: Path) -> dict[str, Any]:
    """Turn `[connectors.<slot>]` / `[settings.<key>]` into models, by key."""
    if table is None:
        return {}
    if not isinstance(table, dict):
        raise ManifestError(path, f"[{section}] must be a table of tables")
    if len(table) > limit:
        raise ManifestError(path, f"[{section}] has more than {limit} entries")
    out: dict[str, Any] = {}
    for key, entry in table.items():
        if not _KEY.fullmatch(key):
            raise ManifestError(path, f"[{section}.{key}]: the name must be 1-32 lowercase letters, digits or '_', starting with a letter")
        if not isinstance(entry, dict):
            raise ManifestError(path, f"[{section}.{key}] must be a table")
        try:
            out[key] = model(**entry)
        except ValidationError as exc:
            raise ManifestError(path, f"[{section}.{key}]: {_problems(exc)}") from None
    return out


def _parse_data(raw: dict[str, Any], path: Path) -> ContentSpec:
    """`[data]`, or the older `[content]`; never both."""
    if "data" in raw and "content" in raw:
        raise ManifestError(path, "has both [data] and [content]; keep only [data]")
    if "data" in raw:
        return _parse_content(raw["data"], path, "data")
    return _parse_content(raw.get("content"), path, "content")


def _parse_content(table: Any, path: Path, section: str = "content") -> ContentSpec:
    if table is None:
        return ContentSpec()
    if not isinstance(table, dict):
        raise ManifestError(path, f"[{section}] must be a table")
    try:
        return ContentSpec(**table)
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            key = ".".join(str(part) for part in error["loc"]) or section
            if error["type"] == "extra_forbidden":
                problems.append(f"unknown key {key!r} (the table takes only 'required' and 'data')")
            elif error["loc"] and error["loc"][0] == "data":
                problems.append(_data_problem(error))
            else:
                problems.append(f"{key!r} must be true or false")
        raise ManifestError(path, f"[{section}] " + "; ".join(problems)) from None


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
