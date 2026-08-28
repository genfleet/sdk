---
name: withfleet-sdk
description: "How to build and serve AI agents using the withfleet-sdk Python package. Use this skill whenever the user is working with fleet, withfleet-sdk, AgentProtocol, Agent(), RawAdapter, or wants to create agents for the Fleet marketplace. Also trigger when you see imports from fleet.sdk, files referencing fleet agent patterns, Agent() constructor with role=/model=/tools=/mcps=/memory= params, or when the user asks about wrapping OpenAI/Anthropic/Gemini/LangChain/CrewAI agents into a universal protocol. Even if the user doesn't say 'fleet' explicitly, trigger if they're working in a project that has withfleet-sdk as a dependency or has fleet/ in its import paths."
---

# Fleet SDK

The withfleet-sdk is the developer interface for the Fleet agent marketplace. Developers declare an agent using `Agent()`, and the SDK handles provider selection, tool dispatch, memory, and MCP connections automatically.

The SDK is **open-source**. Core (`pydantic>=2.7`) has zero LLM dependencies. Provider libraries are installed as optional extras.

## Installation

```bash
pip install withfleet-sdk                    # core only (pydantic)
pip install 'withfleet-sdk[openai,serve]'    # OpenAI + A2A server
pip install 'withfleet-sdk[anthropic,serve]' # Anthropic + A2A server
pip install 'withfleet-sdk[all]'             # everything
```

Requires **Python 3.12+**.

## The Core Pattern

```
Agent(role, model, tools, mcps, memory) → AgentProtocol → serve()
```

1. Declare your agent with `Agent()` — specify the role, model, tools, and optional memory/MCP
2. Serve it over A2A with `serve()` or `create_app()`

No provider client setup, no manual tool dispatch loop, no history wiring needed.

## Quick Reference

### All Exports

```python
from fleet.sdk import (
    Agent,          # Main class — declare and run agents
    AgentProtocol,  # Protocol — implement run(input) -> AsyncIterator[AgentOutput]
    ToolProtocol,   # Protocol — implement schema() + call()
    AgentFactory,   # Type alias — callable returning AgentProtocol
    AgentInput,     # Pydantic model — agent invocation input
    AgentOutput,    # Pydantic model — single output chunk
    Message,        # Pydantic model — conversation history entry
    ToolCall,       # Pydantic model — tool invocation request
    ToolSchema,     # Pydantic model — tool JSON Schema description
    TokenUsage,     # Pydantic model — token counts (input, output, total)
    AuditEvent,     # Pydantic model — structured audit event
    ToolWrapper,    # Class — wraps callable as ToolProtocol
    tool,           # Decorator — function → ToolWrapper
    # Config TypedDicts
    ModelConfig,
    MemoryConfig,
    MCPConfig,
    DataConfig,
    AuditConfig,    # Audit backend config
)

# Serve module (requires [serve] extra)
from fleet.sdk.serve import create_app, serve
```

### Minimal Agent (4 lines)

```python
import os
from fleet.sdk import Agent
from fleet.sdk.serve import serve

serve(
    Agent(role="You are a helpful assistant.", model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}),
    name="assistant", port=8000,
)
```

## Agent Constructor

```python
Agent(
    role:    str,                      # system prompt / persona
    model:   ModelConfig,              # provider + credentials (dict)
    tools:   list[Callable] = [],      # plain Python functions, auto-wrapped
    mcps:    list[MCPConfig] = [],     # MCP server connections
    memory:  MemoryConfig | None = None,  # Redis-backed session memory
    context: str | None = None,        # extra context appended to system prompt
    data:    DataConfig | None = None, # RAG config (reserved, not yet implemented)
    audit:   AuditConfig | None = None, # structured audit logging
)
```

`Agent` implements `AgentProtocol` — pass it directly to `serve()` or `create_app()`.

### ModelConfig

```python
{
    "model":    str,       # "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-6", "gemini/gemini-1.5-pro"
    "api_key":  str,
    "base_url": str,       # optional — for OpenAI-compatible endpoints (DeepSeek, Groq, Ollama, …)
}
```

Uses explicit `provider/model-name` format.

| Provider prefix | Provider | Extra needed |
|---|---|---|
| `openai/` | OpenAI (+ any compatible endpoint) | `[openai]` |
| `anthropic/` | Anthropic | `[anthropic]` |
| `gemini/` | Google Gemini | `[gemini]` |

### MemoryConfig

```python
{
    "type":       "redis",
    "connection": "redis://localhost:6379/0",
    "user":       str | None,
    "password":   str | None,
}
```

Session ID is read from `input.metadata["session_id"]`. Falls back to a per-request UUID when absent (stateless).

### MCPConfig

```python
# stdio transport
{"type": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}

# SSE transport
{"type": "sse", "url": "http://localhost:3001/sse"}
```

MCP tool schemas are fetched and merged with local tools on first `run()` call.

## Examples

### OpenAI with tools

```python
import os
from fleet.sdk import Agent
from fleet.sdk.serve import serve

def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"72°F and sunny in {city}"

agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    tools=[get_weather],
)
serve(agent, name="assistant", port=8000)
```

### Anthropic with memory

```python
import os
from fleet.sdk import Agent
from fleet.sdk.serve import serve

agent = Agent(
    role="You are a customer support agent.",
    model={"model": "anthropic/claude-sonnet-4-6", "api_key": os.environ["ANTHROPIC_API_KEY"]},
    memory={"type": "redis", "connection": "redis://localhost:6379", "password": "secret"},
)
serve(agent, name="support", port=8000)
```

### DeepSeek (OpenAI-compatible endpoint)

```python
agent = Agent(
    role="You are a helpful assistant.",
    model={
        "model":    "openai/deepseek-chat",
        "api_key":  os.environ["DEEPSEEK_API_KEY"],
        "base_url": "https://api.deepseek.com",
    },
)
```

### With MCP server

```python
agent = Agent(
    role="You are a file assistant.",
    model={"model": "openai/gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]},
    mcps=[{"type": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}],
)
```

## Schemas

### AgentInput

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `message` | `str` | *required* | The user's message |
| `history` | `list[Message]` | `[]` | Conversation history |
| `tool_schemas` | `list[ToolSchema]` | `[]` | Available tools (caller-supplied) |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata; `session_id` key used by memory |

### AgentOutput

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `content` | `str \| None` | `None` | Text content |
| `tool_calls` | `list[ToolCall]` | `[]` | Tool calls the agent wants to make |
| `done` | `bool` | `False` | Whether this is the final chunk |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary output metadata |

## @tool Decorator

Turns any function into a `ToolProtocol` with auto-generated JSON Schema from type hints. When using `Agent(tools=[...])`, plain functions are auto-wrapped — `@tool` is optional but useful for adding descriptions.

```python
from fleet.sdk import tool, ToolCall

@tool(description="adds two numbers")
def add(a: float, b: float) -> str:
    return str(a + b)

# Direct call
result = await add(a=3, b=4)

# Via ToolCall (how the platform dispatches)
result = await add.call(ToolCall(id="1", name="add", arguments={"a": 3, "b": 4}))
```

## AgentProtocol — Direct Implementation

For framework integrations (LangChain, CrewAI) where you manage the LLM yourself, implement `AgentProtocol` directly instead of using `Agent()`:

```python
from fleet.sdk import AgentInput, AgentOutput, AgentProtocol

class MyAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content=f"Got: {input.message}", done=True)

assert isinstance(MyAgent(), AgentProtocol)  # passes
serve(MyAgent(), name="my-agent", port=8000)
```

## Serving Over A2A

```python
from fleet.sdk.serve import serve, create_app

# Blocking (dev/scripts)
serve(agent, name="my-agent", description="Does things", host="0.0.0.0", port=8000)

# Returns FastAPI app (for custom ASGI deployments)
app = create_app(agent, name="my-agent")
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/.well-known/agent.json` | Agent card |
| `POST` | `/` | JSON-RPC 2.0 (`tasks/send`, `tasks/sendSubscribe`) |

```bash
# Request/response
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"hello"}]}}}'

# Streaming (SSE)
curl -N -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"hello"}]}}}'
```

## Key Rules

1. **SDK imports only from itself.** Never add imports from `fleet.core`, `fleet.runner`, `fleet.loader`, `fleet.engine`, or `fleet.protocols`. Those are platform internals.
2. **Use `Agent()` as the default path.** Only implement `AgentProtocol` directly for framework integrations (LangChain, CrewAI) that manage their own LLM calls.
3. **`RawAdapter` was removed in 0.4.0.** Deprecated since 0.2. Use `Agent()`; see `references/raw-adapter.md` to migrate.
4. **Python 3.12+ required.** Always use `--python 3.12` when creating venvs with `uv`.
5. **Provider extras are required.** `Agent` lazy-imports the provider library — install the matching extra (`[openai]`, `[anthropic]`, `[gemini]`) or the import will fail with a clear error.
6. **Tool functions return `str`.** `ToolWrapper.call()` always returns `str`.

## Reference Files

- `references/api.md` — Complete field-level reference for Agent, all schemas, and protocols
- `references/raw-adapter.md` — RawAdapter (removed in 0.4.0) — signature patterns for migration reference
- `references/frameworks.md` — Framework integration examples (LangChain, CrewAI)
- `references/serve.md` — A2A serve module: endpoints, JSON-RPC methods, SSE streaming format
