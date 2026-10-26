# Genfleet SDK - Complete API Reference

Matches genfleet-sdk 0.21.x.

## Table of Contents

1. [Agent](#agent)
2. [Config TypedDicts](#config-typeddicts)
3. [Schemas](#schemas)
4. [Parts and files](#parts-and-files)
5. [Protocols](#protocols)
6. [Tools](#tools)
7. [Skills](#skills)
8. [Workflow state](#workflow-state)
9. [Channels](#channels)
10. [Manifests](#manifests)
11. [RawAdapter (removed)](#rawadapter-removed)
12. [Serve Module](#serve-module)

---

## Agent

```python
class Agent:
    def __init__(
        self,
        role:    str,
        model:   ModelConfig,
        tools:   list[Callable | ToolProtocol] | None = None,
        mcps:    list[MCPConfig] | None = None,
        memory:  MemoryConfig | Literal["platform"] | Memory | None = None,
        context: str | None = None,
        audit:   AuditConfig | None = None,
        model_client: ModelClient | None = None,
        skills:  list[Skill | str | Path] | None = None,
        state:   StateStore | Literal["platform"] | None = None,
        accepts: list[str] | None = None,
        files:   FileResolver | None = None,
        max_images_per_turn:      int = 8,
        max_image_bytes_per_turn: int = 20 * 1024 * 1024,
        max_history_turns:        int | None = None,
    ) -> None: ...

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]: ...
```

Implements `AgentProtocol`. Pass directly to `serve()` or `create_app()`. Properties: `agent.memory` (the `Memory` backend or `None`), `agent.state` (the `StateStore` or `None`), `agent.skills` (tuple of `Skill`).

**Parameters:**

- `role` - System prompt / persona. Prepended to every LLM call.
- `model` - Provider credentials. See [ModelConfig](#modelconfig).
- `tools` - Plain Python functions, `ToolWrapper` instances (`@tool`), or objects implementing `ToolProtocol` (`schema()` and async `call()`, such as peer agents in `remote_tools`). Plain functions are wrapped, with the schema derived from type hints and the docstring; a `ToolProtocol` object is used as it is, operator-only and not sensitive unless the platform says otherwise. Anything else raises `TypeError`. A function or MCP tool keeps its name: an object whose name a function or MCP tool has, in either order, or that an earlier object took, is skipped with a logged warning, as is one whose name starts with `@` or is reserved. Among functions a later one replaces an earlier one with the same name. Names starting with `skill:` are reserved, and `read_skill` is reserved when skills are configured.
- `mcps` - MCP server connections. Tool schemas are fetched on first `run()` and merged with local tools. An MCP tool whose name is taken by another tool is skipped.
- `memory` - `"platform"` (the hosted agent's store; fails at construction when not hosted), a [MemoryConfig](#memoryconfig) dict for Redis, or any object implementing the `Memory` protocol (`async load / save / clear`; an optional `append` makes saves incremental). Left out: a hosted agent gets the platform store anyway, anywhere else the agent is stateless and the caller manages history through `AgentInput.history`.
- `context` - Extra string appended to the system prompt at runtime.
- `audit` - Structured audit logging. See [AuditConfig](#auditconfig). `None` (default) adds no overhead.
- `model_client` - Explicit client for the model calls (`ModelClient` from `genfleet.sdk.models`). Wins over an installed client registered under the `genfleet.model_clients` entry-point group, which wins over the direct provider. For tests use `genfleet.sdk.models.ScriptedModelClient(steps)`: it answers each call with the next scripted step (plain text, or tool calls that drive the agent's tool loop) and records the calls. With an installed client `model` may be a plain id and `api_key` may be omitted.
- `skills` - `Skill` objects, or repository sources pinned to a 40-character commit (`"owner/repo@<sha>"`). See [Skills](#skills).
- `state` - A `StateStore`, or `"platform"` for the hosted store. See [Workflow state](#workflow-state).
- `accepts` - MIME types the agent takes besides text (`"image/*"`, `"application/pdf"`, `"*/*"`). Becomes the agent's `accepted_input_types`, which `serve` advertises and enforces. Default: text only.
- `files` - A `FileResolver` that reads file references. Without one only inline files can be read.
- `max_images_per_turn`, `max_image_bytes_per_turn` - Limits on the images one turn sends to a vision model. Negative values raise `ValueError`.
- `max_history_turns` - How many past turns go to the model; `None` (default) sends the whole thread. A turn is a user message plus everything up to the next user message, so the cut never separates a tool call from its result. The history sent always starts on a user message, so a thread with none sends nothing. When the last turns are very long, fewer than N may be sent. Applies to loaded memory and to a caller-supplied `history`. Only what is sent is cut: memory still stores every turn. Anything but an `int` of at least 1 raises `ValueError`.

---

## Config TypedDicts

These are plain Python `TypedDict` classes. Pass as regular dicts.

### ModelConfig

```python
class ModelConfig(TypedDict, total=False):
    model:           Required[str]   # "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-6", "gemini/gemini-1.5-pro", "openrouter/google/gemini-2.5-flash"
    api_key:         str             # a direct provider's key; an installed model client may need none
    base_url:        str             # optional, for OpenAI-compatible endpoints
    default_headers: dict[str, str]  # optional, sent on every request (OpenAI-compatible providers)
    options:         dict[str, Any]  # optional, passed to a model client unchanged; the SDK never reads it
    vision:          bool            # optional, the model takes images: Agent sends image parts as images
```

Uses explicit `provider/model-name` format:

| Provider prefix | Provider | Install |
|--------|----------|---------|
| `openai/` | OpenAI (+ any compatible endpoint) | `pip install 'genfleet-sdk[openai] @ git+https://github.com/genfleet/sdk@dev'` |
| `anthropic/` | Anthropic | `pip install 'genfleet-sdk[anthropic] @ git+https://github.com/genfleet/sdk@dev'` |
| `gemini/` | Google Gemini | `pip install 'genfleet-sdk[gemini] @ git+https://github.com/genfleet/sdk@dev'` |
| `openrouter/<vendor>/<model>` | OpenRouter gateway, any vendor, one key | `pip install 'genfleet-sdk[openai] @ git+https://github.com/genfleet/sdk@dev'` |

### MemoryConfig

```python
class MemoryConfig(TypedDict, total=False):
    type:       Required[Literal["redis", "platform"]]
    connection: str   # redis only, e.g. "redis://localhost:6379/0" (required for redis)
    user:       str
    password:   str
```

`{"type": "platform"}` equals `memory="platform"`. Redis needs `pip install 'genfleet-sdk[memory] @ git+https://github.com/genfleet/sdk@dev'`; its key pattern is `genfleet:session:{session_id}:history` with a 24h TTL.

Memory is keyed by `input.metadata["session_id"]`; a turn without one gets a per-request UUID (stateless). How `serve` fills `session_id` is in `serve.md` under "Sessions". `metadata["genfleet.memory"] == "off"` makes the Agent neither load nor save for that turn. On an append-capable store (the platform store, or a custom `Memory` with `append`) a retried turn is stored twice unless the request metadata names it with `turn_id`, `request_id` or `message_id`, which becomes the idempotency key. Redis rewrites the whole thread on each save, so a retry does not duplicate. The platform store also has `agent.memory.search(subject, query)` and `agent.memory.purge(subject)`.

With `max_history_turns`, a store that implements both `append` and `WindowedMemory` (`async load_newest(session_id, limit) -> list[Message]`: the newest `limit` messages, oldest first; it may leave out leading tool results whose call fell outside them) is read through `load_newest`. The platform store does. Any other store, Redis included, is loaded whole and cut before the model call: a store that saves rewrites the thread from what was loaded, so it must load all of it.

A custom store implements the protocol:

```python
class Memory(Protocol):
    async def load(self, session_id: str) -> list[Message]: ...
    async def save(self, session_id: str, history: list[Message]) -> None: ...
    async def clear(self, session_id: str) -> None: ...
```

### MCPConfig

```python
class MCPStdioConfig(TypedDict):
    type:    Literal["stdio"]
    command: str
    args:    list[str]
    customer_safe_tools: NotRequired[list[str]]   # offered to customers too (both audiences)
    sensitive_tools:     NotRequired[list[str]]   # wait for an owner's approval

class MCPSseConfig(TypedDict):
    type: Literal["sse"]
    url:  str
    customer_safe_tools: NotRequired[list[str]]
    sensitive_tools:     NotRequired[list[str]]

MCPConfig = MCPStdioConfig | MCPSseConfig
```

Requires `pip install 'genfleet-sdk[mcp] @ git+https://github.com/genfleet/sdk@dev'`. Unlisted MCP tools are operator-only. There is no customer-only form in the config; the platform's per-tool setting can make one.

### AuditConfig

```python
class AuditConfig(TypedDict, total=False):
    backend:            Required[Literal["console", "file", "callback"]]
    file_path:          str           # default: "audit.jsonl" (file backend)
    callback:           Callable      # sync or async handler (callback backend)
    max_content_length: int           # truncation limit, default 500 (0 = unlimited)
    events:             list[str]     # filter which event types to emit (default: all)
    agent_name:         str           # name tag included in every event
```

Built-in backends:
- **console** — structured JSON via `logging.getLogger("genfleet.audit")`
- **file** — appends JSONL to `file_path`
- **callback** — calls your sync or async function with an `AuditEvent`

Events emitted: `invocation.start`, `llm.request`, `llm.response`, `tool.call`, `tool.result`, `invocation.end`, `invocation.error`.

---

## Schemas

All schemas are Pydantic v2 `BaseModel` subclasses from `genfleet.sdk.schemas`.

### AgentInput

```python
class AgentInput(BaseModel):
    message:      str = ""
    parts:        list[Part]       = Field(default_factory=list)
    history:      list[Message]    = Field(default_factory=list)
    tool_schemas: list[ToolSchema] = Field(default_factory=list)
    metadata:     dict[str, Any]   = Field(default_factory=dict)
```

- `message` - The turn's text. Left empty, it is the text parts joined; given with no text part in `parts`, it is added as the first part. A text-only agent reads `message` and ignores `parts`.
- `parts` - The whole turn: text and files. See [Parts and files](#parts-and-files).
- `history` - Previous turns. With memory configured, persisted history is loaded and used when this is empty; a non-empty caller `history` takes precedence.
- `tool_schemas` - Caller-supplied tool schemas (for `AgentProtocol` implementations). `Agent` manages its own tools.
- `metadata` - Free-form. Keys the SDK reads: `session_id` (memory key), `genfleet.memory` (`"off"` skips memory for the turn), `genfleet.caller` (hosted turns: `{role, id, name, channel, private, legacy_tools}`, `name` untrusted; read with `caller_of(metadata)`), `genfleet.tool_audiences` (the platform's per-tool audiences, `tool_audiences_of(metadata)`), `genfleet.history` (prior turns; only honoured by `serve(accept_history=True)`, and removed before the agent sees metadata).

### AgentOutput

```python
class AgentOutput(BaseModel):
    content:     str | None      = None
    parts:       list[Part]      = Field(default_factory=list)
    tool_calls:  list[ToolCall]  = Field(default_factory=list)
    done:        bool            = False
    metadata:    dict[str, Any]  = Field(default_factory=dict)
    token_usage: TokenUsage | None = None
```

- `content` - Text chunk. Left out, it is the text parts joined; when set, text parts are not sent again.
- `parts` - Typed results (files, images) alongside the text. `serve` sends them as A2A file parts.
- `tool_calls` - Calls the agent wants run. `Agent` dispatches its own tools internally. **`serve` does not run `tool_calls` from a custom `AgentProtocol` and does not return them**; run tools inside `run()`.
- `done` - Final chunk signal. The serve layer stops reading after `done=True`.
- `token_usage` - Token counts (input, output, total), on the final chunk of each LLM turn.
- `metadata["tool_event"]` (`TOOL_EVENT_KEY`) - a *tool event*: `Agent` reports each tool it runs, with `content=""` and `done=False`:
  - `{"type": "tool_call", "id", "name", "arguments"}` once the call's input is complete;
  - `{"type": "tool_result", "id", "name", "ok", "output"}` when the tool returns (`ok` is false for an unknown tool or a failed MCP call). For a `sensitive` call awaiting owner approval the result carries `"status": "pending"` and `ok` is false; the tool did not run.

  Values are raw; the platform redacts and truncates before showing them in chat. A consumer that only reads `content` can ignore these outputs. `serve` relays them on `working` status updates under `metadata.tool_event`: `arguments` redacted and capped at 8 KB, a failed result's `output` replaced by `"The tool failed."`, and a successful result's `output` sent **raw**, capped at 4000 chars. A direct caller of a self-hosted `serve` sees that raw output; only the platform scrubs it.

### Message

```python
class Message(BaseModel):
    role:         Literal["user", "assistant", "tool"]
    content:      str
    tool_call_id: str | None      = None
    tool_calls:   list[ToolCall]  = Field(default_factory=list)
```

### ToolCall

```python
class ToolCall(BaseModel):
    id:                str
    name:              str
    arguments:         dict[str, Any] = Field(default_factory=dict)
    thought_signature: str | None     = None
```

- `thought_signature` - Gemini 3.x only. Opaque per-call signature, base64-encoded, that Gemini requires echoed back when the call is replayed in history. Set by the Gemini provider and threaded through memory; leave it alone.

### ToolSchema

```python
class ToolSchema(BaseModel):
    name:        str
    description: str
    parameters:  dict[str, Any]
```

---

## Parts and files

Exported from `genfleet.sdk`.

| Symbol | Purpose |
|--------|---------|
| `TextPart(text)` | A text part (`type="text"`) |
| `ImagePart`, `AudioPart`, `VideoPart`, `FilePart` | Media parts. Fields: `mime_type`, `name` (display only, directories and control characters stripped), `size`, `uri`, `data` (inline bytes) |
| `MediaPart` | Union of the four media parts; `Part` is `TextPart` or any media part, discriminated by `type` |
| `part_for(mime_type, *, data=None, uri=None, name=None, size=None)` | Builds the part of the kind the MIME type names: `image/*` is `ImagePart`, `audio/*` `AudioPart`, `video/*` `VideoPart`, anything else `FilePart` |
| `MAX_INLINE_BYTES` | 1 MiB; larger files go by reference (`uri`) |
| `UnsupportedInput` | Raised by `Agent.run` when a turn has a file type outside `accepts`; has `.unsupported` and `.accepted` |
| `FileResolver` | Protocol: `handles(uri) -> bool`, `async read(uri, *, max_bytes, metadata) -> bytes`. The platform's resolver checks access and expiry for the running turn; never grant access from ids in `metadata`, much of it is caller-supplied |
| `read_part(part, *, max_bytes, metadata, resolver)` | The part's bytes: inline `data`, else `uri` through the resolver |
| `InMemoryFileResolver` | For tests only: `put(data, *, mime_type, name=None, ttl=None, session_id=None)` returns a part referencing `mem://...` |
| `FileAccessError` and `FileNotFound`, `FileExpired`, `FileTooLarge`, `UnsupportedReference` | Errors a resolver raises; each has a `reason` safe to show a model |

```python
from genfleet.sdk import Agent, InMemoryFileResolver, AgentInput

files = InMemoryFileResolver()
part = files.put(pdf_bytes, mime_type="application/pdf", name="invoice.pdf")
agent = Agent(role="...", model={"model": "openai/gpt-4o-mini", "api_key": "...", "vision": True},
              accepts=["image/*", "application/pdf"], files=files)
turn = AgentInput(message="Summarise this", parts=[part])
```

Model handling: with `"vision": True`, PNG, JPEG, GIF and WebP images up to 5 MiB whose bytes match the declared type are sent as images (at most `max_images_per_turn`, `max_image_bytes_per_turn` in total, a repeated image once). Everything else is described to the model in one line (kind, name, type, size); file bytes never enter the prompt or memory.

---

## Protocols

Both are `@runtime_checkable` - `isinstance()` checks work at runtime.

### AgentProtocol

```python
@runtime_checkable
class AgentProtocol(Protocol):
    def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]: ...
```

`Agent` satisfies this. For framework integrations (LangChain, CrewAI), implement it directly. A custom agent may set an `accepted_input_types` and an `output_types` attribute (tuples or lists of MIME patterns) for `serve` to advertise.

### ToolProtocol

```python
@runtime_checkable
class ToolProtocol(Protocol):
    def schema(self) -> ToolSchema: ...
    async def call(self, tool_call: ToolCall) -> str: ...
```

---

## Tools

### @tool and ToolWrapper

```python
def tool(fn=None, *, name=None, description="", audiences=None, sensitive=False, customer_safe=None): ...

class ToolWrapper:
    def __init__(self, fn, schema: ToolSchema, *, customer_safe=None, audiences=None, slug=None, sensitive=False) -> None: ...
    audiences: frozenset[Literal["operator", "customer"]]
    sensitive: bool
    slug:      str | None            # manifest slug for a tool from load_tools()
    def schema(self) -> ToolSchema: ...
    async def call(self, tool_call: ToolCall) -> str: ...
    async def __call__(self, **kwargs) -> str: ...
```

`call()` and `__call__()` handle sync and async functions and always return `str`.

- `audiences=["operator", "customer"]` - who the tool is offered to. Default `["operator"]`. Empty or unknown values raise `ValueError`. The author's marking is the default; on the platform the tenant owner's per-tool setting replaces it, wider or narrower. An entry the SDK cannot read offers the tool to no one.
- `sensitive=True` - the call does not run when the model makes it. A workspace owner or admin approves the exact call, then it runs once in a follow-up turn; the model reports the outcome. Refused on a call from another agent. The owner's platform setting wins over this marking.
- `customer_safe=True` - **deprecated since 0.18, removed in 1.0.** Meant both audiences; emits `DeprecationWarning`. Passing it together with `audiences` raises `ValueError`. `ToolWrapper.customer_safe` is a read-only property (true when the audience is both) and assigning to it is deprecated.
- `name` / `description` default to the function name and docstring.
- A parameter annotated `ToolContext` (also `ToolContext | None`, `Optional[ToolContext]`) is filled by the wrapper and left out of the JSON schema.

### Caller helpers

`caller_of(metadata) -> Caller | None` (`None` only outside the platform), `Caller(role, id, name, channel, private, legacy_tools)`, `may_offer(caller, audiences)`, `tool_audiences_of(metadata)`, `is_hosted()`, `Audience`, `EVERYONE`, `OPERATOR_ONLY`, `CALLER_METADATA_KEY`, `TOOL_AUDIENCES_METADATA_KEY`.

Rules: an operator in a private chat gets tools whose audience includes `operator`; a customer, or an operator in a group, gets those that include `customer`. A tool not offered is refused as "not found" at dispatch too. Without a caller key, a local agent is unfiltered; a hosted agent is treated as a customer in a group (narrowest). `legacy_tools` offers every tool. `may_use` and `Caller.full_toolset` are deprecated and removed in 1.0.

### ToolContext

```python
class ToolContext:
    slug:   str | None
    config: Mapping[str, Any]            # read-only; the tenant's installation on the platform
    def secret(self, name: str) -> str   # raises ToolSecretError naming the secret, never a value
    on_platform: bool
```

Declared in the tool's `genfleet.toml` with `[config.<name>]` and `[secrets.<name>]`. Local use: set `tool_env_namespace(slug) + "__CONFIG"` and `+ "__SECRETS"` (JSON objects) in the environment, or a variable named like the secret (`api_token` or `API_TOKEN`). Build one by hand for a hand-written `ToolProtocol` class: `ToolContext(slug)`. `tool_env_namespace("@acme/jira")` gives the prefix for that slug. Tools from `load_tools()` share the agent process, so this is not an isolation boundary.

### load_tools

```python
load_tools(directory=None, *, resolver=None, manifest=None) -> list[Callable]
```

Returns the tools named in `[tools]` of the agent's `genfleet.toml`, each carrying its manifest slug. The manifest comes from `manifest=`, else the agent root the runtime sets, else `directory`, else the working directory. Locally, tools resolve from installed packages advertising the `genfleet.tools` entry-point group (`LocalToolResolver`); a missing one raises `ToolResolutionError`. `[mcps]` entries are not included: they go to `Agent(mcps=...)`.

The runtime also hands the agent factory `remote_tools` (other agents and remote tool servers it discovered). Merge them:

```python
def build_agent(remote_tools=()):
    return Agent(role="...", model={...}, tools=[*load_tools(), *remote_tools])
```

(An older spelling of the parameter, `tools`, still works; write `remote_tools` in new code.)

---

## Skills

```python
class Skill(BaseModel, frozen=True):
    name: str                 # lowercase slug, max 64 chars
    description: str          # max 1024 chars
    instructions: str         # max 32,000 chars
    version: str = "1"
    references: dict[str, str] = {}      # text files under references/
    source_revision: str | None = None   # commit, when loaded from a repository
    audiences: frozenset[str] = {"operator"}

Skill.from_file(path)                        # SKILL.md with name/description/version frontmatter + its text references
discover_skills(source) -> list[SkillOption] # name, description, version, source_revision
load_skills(source, *, names=None) -> list[Skill]   # all skills, or the named ones
```

`source` is `owner/repo`, an HTTPS Git URL (optionally `@<40-character-commit>`), or a local `Path`. Discovery finds `SKILL.md` in the repository root, its folders, and under `skills/`, `.agents/skills/`, `.claude/skills/`, `.cursor/skills/`, `agent/skills/`; invalid files are logged and skipped. `Agent(skills=[...])` accepts `Skill` objects and repository sources that end in a 40-character commit (an unpinned string raises `ValueError`; use `load_skills` for interactive discovery). Skill names must be unique within an agent.

The model sees an index of skills and calls the `read_skill(name, path=None)` tool. Skills grant no tool access. Skill bodies are kept out of audit events, tool events and saved memory. The platform can override each skill's audience through the `skill:<name>` key and can disable `read_skill`. Repository code and scripts are never executed.

---

## Workflow state

Separate from conversation memory. `StateStore` is an async protocol; `LocalStateStore(path)` (SQLite on a worker thread, refuses to open on a hosted agent) and the hosted store implement it. `Agent(state="platform")` selects the hosted store and fails when it is unavailable; `state_for(...)` is the helper behind it.

| Method | Purpose |
|--------|---------|
| `create(agent, owner, state, *, subject=None)` | Start a run, returns `RunState(run_id, status, state, version)` |
| `get(run_id, *, agent, owner)` | Read a run |
| `checkpoint(run_id, *, agent, owner, version, state)` | Update a running run |
| `pause(run_id, *, agent, owner, version, state, ttl_seconds=3600)` | Returns `(RunState, resume_token)`; a pause lasts at most 30 days |
| `resume(run_id, *, agent, owner, token)` | Consume the token once |
| `complete(...)` / `cancel(...)` | Finish or close a run |
| `purge(*, agent, owner)` / `purge_subject(*, agent, subject)` | Delete runs |
| `claim_effect(run_id, effect_id, *, agent, owner)` | Record intent before an external side effect; returns `EffectState` with a stable `idempotency_key`; a second claim is rejected |
| `complete_effect(run_id, effect_id, *, agent, owner, result)` / `get_effect(...)` | Record or read the outcome; a `claimed` effect after a crash means the remote outcome is uncertain |

- Writes need the last returned `version`. State and result payloads are capped at 64 KiB (`StateTooLarge`); `StateQuotaExceeded` for a hosted quota; all are `StateError`. A hosted failure is also a `PlatformStateError` with the HTTP `status` (0 when unreachable).
- Locally `agent` and `owner` are yours: use a stable agent id and the verified caller id (`caller_of(input.metadata).id`). Hosted, the platform stamps the scope itself, `owner` is ignored, and a store holds one `agent` value. Never take the owner from an unverified payload.
- **Pause is not an approval.** Whoever holds the resume token in the same owner scope can resume, including the requester. Human approval of an action is `sensitive=True` on a tool.

---

## Channels

```python
async def send_message(to: str, text: str | None = None, *, template: str | None = None,
                       params: list[str] | None = None, language: str = "en") -> dict
send_message_tool   # the same as a ToolWrapper for the model; failures return as text
class ChannelSendError(RuntimeError): status: int; code: str | None; delivery_unknown: bool
```

Pass exactly one of `text` or `template` (else `ValueError`). `to` is `"<channel>:<address>"`, e.g. `"whatsapp:201001234567"`; replying to the current contact is `send_message(input.metadata["session_id"], ...)`. Free text reaches only a contact who wrote in the last 24 hours; a template reaches anyone the binding may contact. Returns `{"sent": True, "messageIds": [...]}`. `ChannelSendError.code` values: `outside_window` (send a template instead), `cold_send_disabled`, `no_binding`, `invalid_request`, `send_rejected`, `not_in_sandbox` (the agent is not hosted); `None` for anything else. A coded refusal means nothing was sent; with no code, `delivery_unknown` is true unless the status was a 4xx. `send_message_tool` is operator-only by default (like any unmarked tool).

---

## Manifests

`genfleet.toml` next to the agent. Validate offline with `load_agent_manifest(dir)` / `load_tool_manifest(dir)`; problems raise `ManifestError` with the file path.

```toml
[agent]
name = "support"                       # required
kind = "code"                          # "declarative" | "code" | "native_sdk"; default "code"
description = "Answers order questions"
entry = "agent:build_agent"            # default

[tools]                                # @scope/name = version spec ("*", "1.2.x", exact)
"@acme/jira" = "*"
"@acme/crm" = { spec = "1.2.x" }

[mcps]                                 # same form

[content]
required = true                        # don't serve until the workspace supplies content
data = ["knowledge.json", "flows/**"]  # globs of files the code reads by path
```

Content: `genfleet.sdk.content.loaded_files(root=None, *, exclude=())` returns the workspace's loaded text files as sorted `[(path, text)]` from `$CONTENT_DIR` (else `/content`); `loaded_prompt(...)` joins them under `--- path ---` headers. `data` files are never returned. Without the platform's modes file every text file counts as loaded, so pass `exclude=` your `data` globs. Globs: `*` within a name, `?` one character, `**` any folders; relative, `/`-separated, no brackets or braces. A malformed modes file or a path outside the folder raises `ContentError`.

`push` resolves each spec to a version and digest recorded in the published artifact; nobody types a hash. `AgentManifest.is_fully_pinned` says whether every dependency is resolved.

A tool's manifest has `[tool]` (`slug`, `version`, `kind` = `cli | http | mcp_stdio | mcp_http`, `description`, `entry`, `sensitive`), `[config.<name>]` (`type` string/number/boolean, `required`, `description`, `default`) and `[secrets.<name>]` (`description`, `required`). Secrets are declared by name only; a `value` key is rejected, because manifests are committed and published.

---

## RawAdapter (removed)

> **Removed in v0.4.0**, deprecated since v0.2. `from genfleet.sdk import RawAdapter` now raises `ImportError`.
> Migrate to `Agent()`. See `references/raw-adapter.md` for the old signature patterns and migration steps.

---

## Serve Module

Exported from `genfleet.sdk.serve`. Requires `pip install 'genfleet-sdk[serve] @ git+https://github.com/genfleet/sdk@dev'`.

```python
def serve(agent, *, name="agent", description="", host="0.0.0.0", port=8000,
          public_url=None, accept_history=False, accepted_input_types=None,
          output_types=None, max_body_bytes=8 * 1024 * 1024) -> None: ...

def create_app(agent, *, name="agent", description="", public_url=None,
               accept_history=False, accepted_input_types=None,
               output_types=None, max_body_bytes=8 * 1024 * 1024) -> FastAPI: ...
```

`serve` blocks (`uvicorn.run`) and binds `0.0.0.0` by default. It has no authentication: put your own auth in front before exposing it publicly. `create_app` returns a FastAPI app for custom ASGI deployments. Option semantics are in `serve.md`.
