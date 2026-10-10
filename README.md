# genfleet-sdk

The developer SDK for the [Genfleet](https://genfleet.ai) agent marketplace platform.

Declare an agent with `Agent()` — give it a role, model, tools, memory, or MCP connections — and serve it over [A2A](https://google.github.io/A2A/). The SDK handles provider selection, tool dispatch, session memory, and streaming automatically.

## Install

```bash
pip install "genfleet-sdk @ git+https://github.com/genfleet/sdk@dev"                    # core (pydantic only)
pip install "genfleet-sdk[openai,serve] @ git+https://github.com/genfleet/sdk@dev"      # OpenAI + A2A server
pip install "genfleet-sdk[anthropic,serve] @ git+https://github.com/genfleet/sdk@dev"   # Anthropic + A2A server
pip install "genfleet-sdk[all] @ git+https://github.com/genfleet/sdk@dev"               # everything
```

Installed from GitHub rather than PyPI: the package is not published yet, so
the git reference is the supported way to get it. Pin a tag or commit instead
of `@dev` if you need a fixed version.

Requires **Python 3.12+**.

## Agent Skill

Install the genfleet-sdk skill so your coding agent understands the SDK and can help you build agents:

```bash
npx skills add genfleet/sdk
```

Your coding agent will automatically use it when working with `Agent()`, `AgentProtocol`, `@tool`, `serve()`, and all framework integration patterns.

## Quickstart

```python
import os
from genfleet.sdk import Agent
from genfleet.sdk.serve import serve

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
  -H "Content-Type: application/json" -H 'A2A-Version: 1.0' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"What is the weather in Paris?"}]}}}'
```

## Publish on Genfleet (private beta)

Invited to the beta? The step-by-step guide — write an agent with a tool,
test it locally, `gf agents push` it, run it in your workspace and chat with
it — is the **Publish on Genfleet** section of the docs:
[`docs/index.html#marketplace`](docs/index.html#marketplace).

## Agent Constructor

```python
Agent(
    role:    str,                      # system prompt / persona
    model:   ModelConfig,              # provider + credentials
    tools:   list[Callable] = [],      # plain Python functions — auto-wrapped
    mcps:    list[MCPConfig] = [],     # MCP server connections
    memory:  MemoryConfig | Memory | "platform" | None = None,  # episodic memory (see below)
    skills:  list[Skill | str] | None = None,  # Skill objects or Git repositories
    context: str | None = None,        # extra context appended to system prompt
    audit:   AuditConfig | None = None, # structured audit logging
)
```

### ModelConfig — explicit `provider/model-name` format

```python
{"model": "openai/gpt-4o-mini",       "api_key": "sk-..."}     # OpenAI
{"model": "anthropic/claude-sonnet-4-6", "api_key": "sk-ant-..."} # Anthropic
{"model": "gemini/gemini-1.5-pro",     "api_key": "..."}        # Gemini
{"model": "openrouter/google/gemini-2.5-flash", "api_key": "sk-or-..."}  # any vendor via OpenRouter
{"model": "openai/deepseek-chat",      "api_key": "sk-...", "base_url": "https://api.deepseek.com"}  # any OpenAI-compatible
```

### Model clients — `genfleet.model_clients`

An agent's model calls go through a *model client*. The direct providers above
are the default. A package can register its own under the entry-point group
`genfleet.model_clients`: a factory `(ModelConfig) -> ModelClient | None` that
returns `None` when it doesn't apply in the current environment. Installed
factories are tried in name order and the first active one is used (the SDK
logs which). An explicit `Agent(model_client=...)` wins over both, which is how
tests use `genfleet.sdk.models.FakeModelClient`.

With a model client installed, `model` may be a plain id (no provider prefix)
and `api_key` may be left out. `ModelConfig["options"]` is passed to the client
unchanged; the SDK never reads it. During a turn,
`genfleet.sdk.turn.current_turn()` returns the turn's `AgentInput.metadata`.

### Memory

Memory is episodic: one thread of messages per `session_id`, loaded before a
turn and appended to after it.

- **On the platform** you need not name a backend. A hosted agent is given
  `GENFLEET_MEMORY_URL` / `GENFLEET_MEMORY_TOKEN` by its sandbox and uses the
  platform's store, scoped to that agent instance. `memory="platform"` says so
  explicitly and fails fast outside a sandbox. `agent.memory.search(subject, query)`
  and `agent.memory.purge(subject)` are available on it.
- **Anywhere else** pass your own Redis, or nothing for a stateless agent.
- Both sandbox variables travel together. With only `GENFLEET_MEMORY_URL` set,
  the agent raises at construction; with only `GENFLEET_MEMORY_TOKEN`, there is
  nothing to dial, so it runs stateless and logs a warning.
- A retried turn is appended twice unless the caller names it: `turn_id`,
  `request_id` or `message_id` in the request metadata is passed to the store
  as the turn's idempotency key.

### Local workflow state

Workflow state is separate from conversation memory. A hosted `Agent` selects
`PlatformStateStore` when its sandbox receives `GENFLEET_STATE_URL` and
`GENFLEET_STATE_TOKEN`; `agent.state` then uses the engine proxy and a
PostgreSQL store scoped to that tenant and agent instance. The SDK sends no
tenant or agent identifier to the proxy. `Agent(state="platform")` requires
the environment pair and fails if it is absent.

For local/self-hosted work, use `LocalStateStore(path)` as the async backend.
It wraps `SQLiteStateStore` on a worker thread and refuses to open in a hosted
sandbox. Both implement the same `StateStore` interface. This state API is
not an approval or authorization service.

Give it a persistent SQLite path and a stable agent ID plus a caller ID
verified by your ingress. The store treats these IDs as scopes; it does not
authenticate callers or approvers.

```python
from genfleet.sdk import LocalStateStore

store = LocalStateStore("/var/lib/my-agent/runs.sqlite3")
run = await store.create("orders-agent", verified_caller_id, {"order_id": order_id}, subject=customer_id)
# Save progress before pausing for an external event.
run, resume_token = await store.pause(
    run.run_id, agent="orders-agent", owner=verified_caller_id,
    version=run.version, state={"order_id": order_id, "step": "waiting"},
    ttl_seconds=3600,
)
# A later process can atomically consume the token once.
run = await store.resume(
    run.run_id, agent="orders-agent", owner=verified_caller_id,
    token=resume_token,
)
effect = await store.claim_effect(
    run.run_id, "charge-order", agent="orders-agent", owner=verified_caller_id,
)
receipt = payment_client.charge(order_id, idempotency_key=effect.idempotency_key)
await store.complete_effect(
    run.run_id, "charge-order", agent="orders-agent",
    owner=verified_caller_id, result={"receipt": receipt.id},
)
run = await store.complete(
    run.run_id, agent="orders-agent", owner=verified_caller_id,
    version=run.version, state={"order_id": order_id, "step": "done"},
)
```

`checkpoint` updates a running run without pausing. State writes require the
last returned `version`; effect claims are separately deduplicated by
`effect_id`. Keep resume tokens private. `cancel` closes an expired or rejected
run, and `purge` removes a caller's runs and effect records. `purge_subject`
deletes one subject's runs within the agent. State and result payloads are
limited to 64 KiB. For an external side effect,
`claim_effect` records intent before the call and gives a stable idempotency
key to pass to a remote API that supports it. A second claim is rejected,
including after a restart. If a worker dies between the claim and
`complete_effect`, `get_effect` reports `claimed`: the remote outcome is
uncertain and must be reconciled before proceeding. The store never silently
replays that call. No local checkpoint can make a remote side effect and its
SQLite commit one atomic transaction.

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

Send the same `contextId` on each message to persist history across turns (it becomes the agent's `session_id`):

```bash
curl -X POST http://localhost:8000 -H 'A2A-Version: 1.0' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","contextId":"user-123","parts":[{"text":"My name is Alice"}]}}}'
```

### Runtime skills

`Skill` holds versioned instructions. The model sees each skill's name and description, then can call `read_skill` to load its full instructions for that turn. Skill bodies are not copied into audit results, tool events, or saved conversation memory. Skills never grant access to tools; the existing tool audience rules still apply. Do not put secrets in skill instructions.

```python
from genfleet.sdk import Agent, Skill

agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": "..."},
    skills=[Skill(
        name="concise",
        version="1",
        description="Give short answers",
        instructions="Answer in one sentence unless the user asks for detail.",
    )],
)
```

`Skill.from_file("SKILL.md")` reads `name`, `description`, and optional `version` from simple frontmatter, followed by the Markdown instructions. The version is author-defined; file contents are captured when the agent is constructed. For a local, credential-free walkthrough of skills, tools, and memory, use [`examples/skills_memory.ipynb`](examples/skills_memory.ipynb).

To load all skills from a repository, pass its GitHub handle or HTTPS Git URL. The SDK discovers `SKILL.md` in root skill folders and under `skills/`, `.agents/skills/`, `.claude/skills/`, `.cursor/skills/`, and `agent/skills/` (up to three nested directories), plus text files under each skill's `references/`. It clones once when `Agent` is constructed and records the source commit on each `Skill.source_revision`.

```python
from genfleet.sdk import Agent

agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": "..."},
    skills=["vercel-labs/agent-skills@<40-character-commit>"],
)
```

To see the choices first and select a subset:

```python
from genfleet.sdk import Agent, discover_skills, load_skills

repo = "mmedhat1910/skills"
for option in discover_skills(repo):
    print(option.name, "—", option.description)

selected = load_skills(repo, names=["handoff", "handoff-proceed"])
agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": "..."},
    skills=selected,
)
# Omit names to load every skill: load_skills(repo)
```

Pass a selected list as `skills=selected`; combine it with other sources using `skills=[*selected, other]`. Direct repository sources in `Agent(skills=...)` must end in `@<40-character-commit>` so published agents use a fixed revision. For deployment, load and package the selected `Skill` objects during the build to avoid a Git fetch at agent startup. Use `discover_skills` and `load_skills` for interactive selection; pass a `Path` for a local repository. Discovery logs and skips invalid skill files, then returns valid skills in the repository.

The model uses `read_skill(name)` for instructions and `read_skill(name, path="references/guide.md")` for a text reference. Skills default to the `operator` audience; set `Skill(audiences={"operator", "customer"}, ...)` to offer one to customers. The platform can override each skill through the `skill:<name>` audience key and can disable the `read_skill` tool. Git authentication uses your configured Git credentials. Repository code and scripts are not executed by the loader; a skill can only use tools already offered to the agent.

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

`serve(...)` / `create_app(...)` options for an agent you host yourself: `public_url="https://…/"` fixes the URL the agent card advertises (otherwise it follows the request's `Host` and is not cached), and `accept_history=True` lets a trusted caller send prior turns in `metadata["genfleet.history"]` (off by default).


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

**Console backend** — emits structured JSON to the `genfleet.audit` logger:

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
| `Skill` | Class | Versioned instructions loaded by the agent on demand |
| `SkillOption` | Class | Name, description, version, and source commit returned by discovery |
| `discover_skills` | Function | List skills available in a Git repository |
| `load_skills` | Function | Discover and snapshot skills from a Git repository |
| `Memory` | Protocol | Contract for a caller-provided episodic memory backend |
| `AgentProtocol` | Protocol | Universal agent contract — implement `run()` |
| `ToolProtocol` | Protocol | Tool contract — implement `schema()` + `call()` |
| `AgentInput` | Model | Input to every agent invocation |
| `AgentOutput` | Model | Output chunk yielded by agents |
| `Message` | Model | Conversation history entry |
| `ToolCall` | Model | Tool invocation request |
| `ToolSchema` | Model | Tool JSON Schema description |
| `TokenUsage` | Model | Token counts (input, output, total) |
| `TOOL_EVENT_KEY` | Constant | `AgentOutput.metadata` key on the outputs `Agent` yields to report a tool call and its result |
| `AuditEvent` | Model | Structured audit event |
| `ModelConfig` | TypedDict | Provider + credentials config |
| `MemoryConfig` | TypedDict | Memory config — `{"type": "redis", ...}` or `{"type": "platform"}` |
| `AuditConfig` | TypedDict | Audit backend config |
| `MCPConfig` | TypedDict | MCP server connection config |
| `ToolWrapper` | Class | Wraps any callable as a `ToolProtocol` |
| `@tool` | Decorator | Function → `ToolWrapper` with auto-generated schema |

## @tool decorator

Plain functions passed to `tools=` are auto-wrapped. Use `@tool` when you want an explicit name or description:

```python
from genfleet.sdk import tool, ToolCall

@tool(name="add", description="Adds two numbers")
def add(a: float, b: float) -> str:
    return str(a + b)

result = await add.call(ToolCall(id="1", name="add", arguments={"a": 3, "b": 4}))
# "7.0"
```

### Operators and customers

On a hosted agent the platform tells each turn who is writing, as `AgentInput.metadata["genfleet.caller"]` (`CALLER_METADATA_KEY`, from `genfleet.sdk.caller`): `role` (`"operator"` or `"customer"`), `id` (the verified sender), `channel`, `private` (a one-to-one chat), `legacy_tools`, and `name`. `name` is the sender's own display name and is untrusted; don't base decisions on it. Use `caller_of(input.metadata)` to read it.

Every tool has an **audience**: who it is offered to. Mark it on the tool:

```python
@tool(audiences=["operator", "customer"])
def order_status(order_id: str) -> str: ...

@tool(audiences=["customer"])               # customers only: staff shouldn't trigger it
def start_return(order_id: str) -> str: ...
```

An unmarked tool is `["operator"]`. An operator in a private chat gets the tools whose audience includes `operator`. A customer, or an operator writing in a group, gets those whose audience includes `customer`. So a `["customer"]` tool is hidden from staff in a private chat. A call to a tool that wasn't offered is refused as "not found". For MCP servers, the config's `customer_safe_tools: [names]` (the 0.17 form) marks tools for both audiences; there is no customer-only form in the config, but the platform's setting can make one by name.

**Your marking is the default, and the platform's setting wins.** The tenant owner sets each tool's audience in the dashboard. The platform sends it as `metadata["genfleet.tool_audiences"]` (`TOOL_AUDIENCES_METADATA_KEY`), keyed by tool name, or by manifest slug for tools from `load_tools()`. Each tool it returns carries its slug, on a fresh proxy that behaves like the original. That setting replaces your marking, and can widen it as well as narrow it. An entry the SDK can't read is offered to no one. `Agent` applies this to every tool it holds: its own functions, `load_tools()` tools, remote tools passed in as `tools=`, and MCP tools. Tools the engine keeps itself are filtered by the engine. Read the effective rule with `may_offer(caller, audiences)`.

`customer_safe=True` (0.17) still works and means both audiences, but it is deprecated and removed in 1.0, as are `may_use` and `Caller.full_toolset`.

**Sensitive tools.** Mark a tool whose call needs a workspace owner's approval:

```python
@tool(audiences=["operator", "customer"], sensitive=True)
def issue_refund(order_id: str, amount: float) -> str: ...
```

When the model calls it, it doesn't run. The model is told the action awaits approval, and the platform shows the owner or admin the exact call to approve. Once approved, a follow-up turn runs that one call, exactly once, and the model tells the user the outcome. The engine signs the approved call with a key only this sandbox knows (`GENFLEET_APPROVAL_KEY`), so another agent or a direct caller can't forge one. On a call from another agent, a sensitive tool is refused. The owner can mark or unmark any tool in the dashboard, and that setting wins.

An agent the platform did not spawn (a local `serve`, a test) has no roles: a turn without the caller key gets every tool. In a platform sandbox (`GENFLEET_HOSTED` set; on older engines, `GENFLEET_A2A_TOKEN`) a missing caller gets the narrowest set, as a malformed one always does. `legacy_tools` gets every tool. An MCP tool whose name is already taken by another tool is skipped.

Declare who the agent serves in `genfleet.toml`:

```toml
[agent]
name = "support"
audiences = ["operator", "customer"]   # default when absent: operators only
```

## AgentProtocol — direct implementation

For framework integrations (LangChain, CrewAI) that manage their own LLM calls, implement `AgentProtocol` directly:

```python
from genfleet.sdk import AgentInput, AgentOutput, AgentProtocol
from genfleet.sdk.serve import serve

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
