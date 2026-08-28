# withfleet-sdk

The developer SDK for the [Fleet](https://withfleet.com) agent marketplace platform.

Declare an agent with `Agent()` — give it a role, model, tools, memory, or MCP connections — and serve it over [A2A](https://google.github.io/A2A/). The SDK handles provider selection, tool dispatch, session memory, and streaming automatically.

## Install

```bash
pip install "withfleet-sdk @ git+https://github.com/withfleet/sdk@dev"                    # core (pydantic only)
pip install "withfleet-sdk[openai,serve] @ git+https://github.com/withfleet/sdk@dev"      # OpenAI + A2A server
pip install "withfleet-sdk[anthropic,serve] @ git+https://github.com/withfleet/sdk@dev"   # Anthropic + A2A server
pip install "withfleet-sdk[all] @ git+https://github.com/withfleet/sdk@dev"               # everything
```

Installed from GitHub rather than PyPI: the package is not published yet, so
the git reference is the supported way to get it. Pin a tag or commit instead
of `@dev` if you need a fixed version.

Requires **Python 3.12+**.

## Agent Skill

Install the withfleet-sdk skill so your coding agent understands the SDK and can help you build agents:

```bash
npx skills add fleet/sdk
```

Your coding agent will automatically use it when working with `Agent()`, `AgentProtocol`, `@tool`, `serve()`, and all framework integration patterns.

## Quickstart

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

serve(agent, name="my-agent", port=8000)
```

```bash
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"What is the weather in Paris?"}]}}}'
```

## Agent Constructor

```python
Agent(
    role:    str,                      # system prompt / persona
    model:   ModelConfig,              # provider + credentials
    tools:   list[Callable] = [],      # plain Python functions — auto-wrapped
    mcps:    list[MCPConfig] = [],     # MCP server connections
    memory:  MemoryConfig | None = None,  # Redis-backed session memory
    context: str | None = None,        # extra context appended to system prompt
    data:    DataConfig | None = None, # RAG config (reserved, not yet implemented)
    audit:   AuditConfig | None = None, # structured audit logging
)
```

### ModelConfig — explicit `provider/model-name` format

```python
{"model": "openai/gpt-4o-mini",       "api_key": "sk-..."}     # OpenAI
{"model": "anthropic/claude-sonnet-4-6", "api_key": "sk-ant-..."} # Anthropic
{"model": "gemini/gemini-1.5-pro",     "api_key": "..."}        # Gemini
{"model": "openai/deepseek-chat",      "api_key": "sk-...", "base_url": "https://api.deepseek.com"}  # any OpenAI-compatible
```

### With Redis memory

```python
agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    memory={
        "type":       "redis",
        "connection": "redis://localhost:6379",
    },
)
```

Pass `session_id` in request metadata to persist history across turns:

```bash
curl -X POST http://localhost:8000 \
  -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","metadata":{"session_id":"user-123"},"message":{"role":"user","parts":[{"type":"text","text":"My name is Alice"}]}}}'
```

### With MCP server

```python
agent = Agent(
    role="You are a file assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    mcps=[{
        "type":    "stdio",
        "command": "npx",
        "args":    ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    }],
)
```

### With audit logging

```python
agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    audit={
        "backend": "console",       # "console", "file", or "callback"
        "agent_name": "my-agent",
    },
)
```

**Console backend** — emits structured JSON to the `fleet.audit` logger:

```python
audit={"backend": "console"}
```

**File backend** — appends JSONL to a file:

```python
audit={"backend": "file", "file_path": "audit.jsonl"}
```

**Callback backend** — calls your function (sync or async) for each event:

```python
async def my_handler(event):
    print(event.event_type, event.data)

audit={"backend": "callback", "callback": my_handler}
```

**AuditConfig options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | `str` | required | `"console"`, `"file"`, or `"callback"` |
| `file_path` | `str` | `"audit.jsonl"` | Output path (file backend only) |
| `callback` | `Callable` | required for callback | Handler function |
| `max_content_length` | `int` | `500` | Truncation limit (0 = unlimited) |
| `events` | `list[str]` | all | Filter which event types to emit |
| `agent_name` | `str` | `None` | Name tag included in every event |

**Audit events emitted:**

| Event | When | Key data |
|-------|------|----------|
| `invocation.start` | `run()` called | message, session_id |
| `llm.request` | Before LLM call | model, message_count, tool_count |
| `llm.response` | After LLM responds | token_usage, latency_ms |
| `tool.call` | Before tool dispatch | tool_name, arguments |
| `tool.result` | After tool returns | tool_name, result, latency_ms |
| `invocation.end` | `run()` completes | total_latency_ms, total_token_usage |
| `invocation.error` | Exception in `run()` | error_type, message |

Token usage (input/output/total tokens) is tracked automatically for OpenAI, Anthropic, and Gemini providers and included in `llm.response` and `invocation.end` events.

## What's in the SDK

| Export | Type | Description |
|--------|------|-------------|
| `Agent` | Class | Main developer-facing class — wires provider, tools, memory, MCP, audit |
| `AgentProtocol` | Protocol | Universal agent contract — implement `run()` |
| `ToolProtocol` | Protocol | Tool contract — implement `schema()` + `call()` |
| `AgentInput` | Model | Input to every agent invocation |
| `AgentOutput` | Model | Output chunk yielded by agents |
| `Message` | Model | Conversation history entry |
| `ToolCall` | Model | Tool invocation request |
| `ToolSchema` | Model | Tool JSON Schema description |
| `TokenUsage` | Model | Token counts (input, output, total) |
| `AuditEvent` | Model | Structured audit event |
| `ModelConfig` | TypedDict | Provider + credentials config |
| `MemoryConfig` | TypedDict | Redis memory config |
| `AuditConfig` | TypedDict | Audit backend config |
| `MCPConfig` | TypedDict | MCP server connection config |
| `DataConfig` | TypedDict | RAG config (reserved) |
| `ToolWrapper` | Class | Wraps any callable as a `ToolProtocol` |
| `@tool` | Decorator | Function → `ToolWrapper` with auto-generated schema |

## @tool decorator

Plain functions passed to `tools=` are auto-wrapped. Use `@tool` when you want an explicit name or description:

```python
from fleet.sdk import tool, ToolCall

@tool(name="add", description="Adds two numbers")
def add(a: float, b: float) -> str:
    return str(a + b)

result = await add.call(ToolCall(id="1", name="add", arguments={"a": 3, "b": 4}))
# "7.0"
```

## AgentProtocol — direct implementation

For framework integrations (LangChain, CrewAI) that manage their own LLM calls, implement `AgentProtocol` directly:

```python
from fleet.sdk import AgentInput, AgentOutput, AgentProtocol
from fleet.sdk.serve import serve

class MyAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content=f"Got: {input.message}", done=True)

assert isinstance(MyAgent(), AgentProtocol)  # passes
serve(MyAgent(), name="my-agent", port=8000)
```

## Package extras

| Extra | Installs | Use for |
|-------|----------|---------|
| `openai` | `openai>=1.0` | OpenAI + any OpenAI-compatible endpoint |
| `anthropic` | `anthropic>=0.25` | Anthropic Claude models |
| `gemini` | `google-genai>=1.0` | Google Gemini models |
| `memory` | `redis>=5.0` | Redis-backed session memory |
| `mcp` | `mcp>=1.0` | MCP server connections |
| `serve` | fastapi, uvicorn, sse-starlette | A2A HTTP server |
| `all` | all of the above | Full install |

## Examples

See [`examples/`](examples/) for complete runnable agents:

| File | Description |
|------|-------------|
| [echo_agent.py](examples/echo_agent.py) | Minimal A2A agent (no LLM) |
| [streaming_agent.py](examples/streaming_agent.py) | SSE streaming |
| [tool_agent.py](examples/tool_agent.py) | `@tool` decorator demo |
| [openai_agent.py](examples/openai_agent.py) | OpenAI GPT-4o-mini with tools |
| [anthropic_agent.py](examples/anthropic_agent.py) | Anthropic Claude |
| [langchain_agent.py](examples/langchain_agent.py) | LangChain via AgentProtocol |
| [crewai_agent.py](examples/crewai_agent.py) | CrewAI via AgentProtocol |
| [agent_with_tools.py](examples/agent_with_tools.py) | Multi-tool agent |
| [memory_agent.py](examples/memory_agent.py) | Redis-backed session memory |
| [mcp_agent.py](examples/mcp_agent.py) | MCP filesystem server |
| [rag_agent.py](examples/rag_agent.py) | RAG with LightRAG |

## Requirements

- Python 3.12+
- `pydantic >= 2.7`
- Provider extras: `[openai]`, `[anthropic]`, `[gemini]`
- `[serve]` extra: `fastapi`, `uvicorn`, `sse-starlette`
- `[memory]` extra: `redis`
- `[mcp]` extra: `mcp`

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
