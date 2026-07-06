# Agentinc SDK — Complete API Reference

## Table of Contents

1. [Agent](#agent)
2. [Config TypedDicts](#config-typeddicts)
3. [Schemas](#schemas)
4. [Protocols](#protocols)
5. [ToolWrapper](#toolwrapper)
6. [RawAdapter (deprecated)](#rawadapter-deprecated)
7. [Serve Module](#serve-module)

---

## Agent

```python
class Agent:
    def __init__(
        self,
        role:    str,
        model:   ModelConfig,
        tools:   list[Callable] = [],
        mcps:    list[MCPConfig] = [],
        memory:  MemoryConfig | None = None,
        context: str | None = None,
        data:    DataConfig | None = None,
        audit:   AuditConfig | None = None,
    ) -> None: ...

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]: ...
```

Implements `AgentProtocol`. Pass directly to `serve()` or `create_app()`.

**Parameters:**

- `role` — System prompt / persona. Prepended to every LLM call.
- `model` — Provider credentials. See [ModelConfig](#modelconfig).
- `tools` — Plain Python functions or `ToolWrapper` instances. Auto-wrapped if plain functions. Schema is derived from type hints and docstring.
- `mcps` — MCP server connections. Tool schemas are fetched on first `run()` call and merged with local tools.
- `memory` — Redis-backed session memory. See [MemoryConfig](#memoryconfig). When absent the agent is stateless (caller manages history via `AgentInput.history`).
- `context` — Optional extra string appended to the system prompt at runtime.
- `data` — RAG configuration. Accepted but not yet implemented — reserved for a future release.
- `audit` — Structured audit logging. See [AuditConfig](#auditconfig). When `None` (default), no audit overhead.

---

## Config TypedDicts

These are plain Python `TypedDict` classes. Pass as regular dicts.

### ModelConfig

```python
class ModelConfig(TypedDict, total=False):
    model:    Required[str]   # e.g. "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-6", "gemini/gemini-1.5-pro"
    api_key:  Required[str]
    base_url: str             # optional — for OpenAI-compatible endpoints
```

Uses explicit `provider/model-name` format:

| Provider prefix | Provider | Install |
|--------|----------|---------|
| `openai/` | OpenAI (+ any compatible endpoint) | `pip install 'agentinc-sdk[openai]'` |
| `anthropic/` | Anthropic | `pip install 'agentinc-sdk[anthropic]'` |
| `gemini/` | Google Gemini | `pip install 'agentinc-sdk[gemini]'` |

### MemoryConfig

```python
class MemoryConfig(TypedDict, total=False):
    type:       Required[Literal["redis"]]
    connection: Required[str]   # Redis URL, e.g. "redis://localhost:6379/0"
    user:       str
    password:   str
```

Requires `pip install 'agentinc-sdk[memory]'`.

Session key pattern: `agentinc:session:{session_id}:history` (TTL 24h).  
Session ID from `input.metadata["session_id"]` — falls back to a per-request UUID.

### MCPConfig

```python
# stdio
class MCPStdioConfig(TypedDict):
    type:    Literal["stdio"]
    command: str         # e.g. "npx"
    args:    list[str]   # e.g. ["-y", "@modelcontextprotocol/server-filesystem"]

# SSE
class MCPSseConfig(TypedDict):
    type: Literal["sse"]
    url:  str            # e.g. "http://localhost:3001/sse"

MCPConfig = MCPStdioConfig | MCPSseConfig
```

Requires `pip install 'agentinc-sdk[mcp]'`.

### DataConfig

```python
class DataConfig(TypedDict, total=False):
    type:        str   # e.g. "lightrag"
    storage_dir: str
    mode:        str   # "naive" | "local" | "hybrid"
```

Reserved for native RAG in a future release. The `data=` parameter is accepted by `Agent()` but currently ignored.

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
- **console** — structured JSON via `logging.getLogger("agentinc.audit")`
- **file** — appends JSONL to `file_path`
- **callback** — calls your sync or async function with an `AuditEvent`

Events emitted: `invocation.start`, `llm.request`, `llm.response`, `tool.call`, `tool.result`, `invocation.end`, `invocation.error`.

---

## Schemas

All schemas are Pydantic v2 `BaseModel` subclasses from `agentinc.sdk.schemas`.

### AgentInput

```python
class AgentInput(BaseModel):
    message:      str
    history:      list[Message]   = Field(default_factory=list)
    tool_schemas: list[ToolSchema] = Field(default_factory=list)
    metadata:     dict[str, Any]  = Field(default_factory=dict)
```

- `message` — The user's text input.
- `history` — Previous conversation turns. When `Agent` memory is configured, persisted history is loaded from Redis and merged with this.
- `tool_schemas` — Caller-supplied tool schemas (used when implementing AgentProtocol directly). `Agent` manages its own tools internally.
- `metadata` — `session_id` key used by Redis memory backend.

### AgentOutput

```python
class AgentOutput(BaseModel):
    content:     str | None      = None
    tool_calls:  list[ToolCall]  = Field(default_factory=list)
    done:        bool            = False
    metadata:    dict[str, Any]  = Field(default_factory=dict)
    token_usage: TokenUsage | None = None
```

- `content` — Text chunk.
- `tool_calls` — Tool calls to dispatch. `Agent` handles dispatch internally.
- `done` — Final chunk signal. The serve layer stops reading after `done=True`.
- `token_usage` — Token counts from the provider (input, output, total). Populated on the final chunk of each LLM turn.

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
    id:        str
    name:      str
    arguments: dict[str, Any] = Field(default_factory=dict)
```

### ToolSchema

```python
class ToolSchema(BaseModel):
    name:        str
    description: str
    parameters:  dict[str, Any]
```

---

## Protocols

Both are `@runtime_checkable` — `isinstance()` checks work at runtime.

### AgentProtocol

```python
@runtime_checkable
class AgentProtocol(Protocol):
    def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]: ...
```

`Agent` satisfies this. For framework integrations (LangChain, CrewAI), implement directly.

### ToolProtocol

```python
@runtime_checkable
class ToolProtocol(Protocol):
    def schema(self) -> ToolSchema: ...
    async def call(self, tool_call: ToolCall) -> str: ...
```

---

## ToolWrapper

```python
class ToolWrapper:
    def __init__(self, fn: Callable[..., Any], schema: ToolSchema) -> None: ...
    def schema(self) -> ToolSchema: ...
    async def call(self, tool_call: ToolCall) -> str: ...
    async def __call__(self, **kwargs: Any) -> str: ...
```

Created by `@tool`. Satisfies `ToolProtocol`. Both `call()` and `__call__()` handle sync and async functions.

---

## RawAdapter (deprecated)

> **Deprecated in v0.2.** Emits `DeprecationWarning` on instantiation. Will be removed in v0.4.  
> Migrate to `Agent()`. See `references/raw-adapter.md` for the old signature patterns.

```python
class RawAdapter:
    def __init__(self, fn: Any) -> None: ...          # emits DeprecationWarning
    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]: ...
```

---

## Serve Module

Exported from `agentinc.sdk.serve`. Requires `pip install 'agentinc-sdk[serve]'`.

### serve()

```python
def serve(
    agent:       AgentProtocol,
    *,
    name:        str = "agent",
    description: str = "",
    host:        str = "0.0.0.0",
    port:        int = 8000,
) -> None: ...
```

Blocking. Calls `uvicorn.run()` internally.

### create_app()

```python
def create_app(
    agent:       AgentProtocol,
    *,
    name:        str = "agent",
    description: str = "",
) -> FastAPI: ...
```

Returns a FastAPI app for custom ASGI deployments.
