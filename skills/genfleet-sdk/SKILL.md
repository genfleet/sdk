---
name: genfleet-sdk
description: "How to build and serve AI agents using the genfleet-sdk Python package. Use this skill whenever the user is working with genfleet, genfleet-sdk, AgentProtocol, Agent(), RawAdapter, or wants to create agents for the Genfleet marketplace. Also trigger when you see imports from genfleet.sdk, files referencing genfleet agent patterns, Agent() constructor with role=/model=/tools=/mcps=/memory= params, or when the user asks about wrapping OpenAI/Anthropic/Gemini/LangChain/CrewAI agents into a universal protocol. Even if the user doesn't say 'genfleet' explicitly, trigger if they're working in a project that has genfleet-sdk as a dependency or has genfleet/ in its import paths."
---

# Genfleet SDK

The genfleet-sdk is the developer interface for the Genfleet agent marketplace. Developers declare an agent with `Agent()`, and the SDK handles provider selection, tool dispatch, memory, skills, files and MCP connections.

The SDK is **open-source**. Core (`pydantic>=2.7`) has zero LLM dependencies. Provider libraries are installed as optional extras. This skill matches 0.21.x.

## Installation

```bash
pip install "genfleet-sdk @ git+https://github.com/genfleet/sdk@dev"                    # core (pydantic only)
pip install "genfleet-sdk[openai,serve] @ git+https://github.com/genfleet/sdk@dev"      # OpenAI + A2A server
pip install "genfleet-sdk[anthropic,serve] @ git+https://github.com/genfleet/sdk@dev"   # Anthropic + A2A server
pip install "genfleet-sdk[all] @ git+https://github.com/genfleet/sdk@dev"               # everything
```

Installed from GitHub rather than PyPI: the package is not published yet, so the git reference is the supported way to get it. Pin a tag or commit instead of `@dev` if you need a fixed version. Requires **Python 3.12+**. Extras: `openai`, `anthropic`, `gemini`, `memory` (Redis), `mcp`, `serve`, `all`.

## The Core Pattern

```
Agent(role, model, tools, ...) -> AgentProtocol -> serve()
```

1. Declare the agent with `Agent()`: role, model, tools, and optionally memory, skills, MCP servers, accepted files.
2. Serve it over A2A with `serve()` or `create_app()`.

No provider client setup, no manual tool dispatch loop, no history wiring.

### Minimal agent

```python
import os
from genfleet.sdk import Agent
from genfleet.sdk.serve import serve

serve(
    Agent(role="You are a helpful assistant.",
          model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}),
    name="assistant", port=8000,
)
```

## Agent Constructor

```python
Agent(
    role:    str,                        # system prompt / persona
    model:   ModelConfig,                # provider + credentials (dict)
    tools:   list[Callable] | None,      # plain functions or @tool wrappers
    mcps:    list[MCPConfig] | None,     # MCP server connections
    memory:  MemoryConfig | Memory | "platform" | None = None,
    context: str | None = None,          # extra text appended to the system prompt
    audit:   AuditConfig | None = None,  # structured audit logging
    model_client: ModelClient | None = None,   # explicit client (tests: FakeModelClient)
    skills:  list[Skill | str | Path] | None = None,   # runtime skills
    state:   StateStore | "platform" | None = None,    # workflow checkpoints
    accepts: list[str] | None = None,    # file MIME types taken besides text
    files:   FileResolver | None = None, # reads file references
    max_images_per_turn: int = 8,
    max_image_bytes_per_turn: int = 20 * 1024 * 1024,
)
```

`Agent` implements `AgentProtocol`: pass it straight to `serve()` or `create_app()`. Field-level detail for every argument is in `references/api.md`.

### ModelConfig

```python
{"model": "openai/gpt-4o-mini", "api_key": "..."}                       # OpenAI
{"model": "anthropic/claude-sonnet-4-6", "api_key": "..."}              # Anthropic
{"model": "gemini/gemini-1.5-pro", "api_key": "..."}                    # Gemini
{"model": "openrouter/google/gemini-2.5-flash", "api_key": "..."}       # any vendor via OpenRouter
{"model": "openai/deepseek-chat", "api_key": "...", "base_url": "https://api.deepseek.com"}  # OpenAI-compatible
{"model": "openai/gpt-4o-mini", "api_key": "...", "vision": True}       # model takes images
```

Always the explicit `provider/model-name` form. Optional keys: `base_url`, `default_headers`, `options`, `vision`. Prefixes: `openai/` (also any OpenAI-compatible endpoint), `anthropic/`, `gemini/`, `openrouter/<vendor>/<model>` (the last also needs the `[openai]` extra). For free trial runs use an OpenRouter `:free` model, e.g. `openrouter/nex-agi/nex-n2.5-mini:free`.

### Memory

Episodic memory is one thread of messages per `session_id`, loaded before a turn and saved after it.

```python
memory="platform"                                          # a hosted agent's platform store
memory={"type": "redis", "connection": "redis://localhost:6379/0"}   # your own Redis ([memory] extra)
memory=MyMemory()                                          # any object with async load/save/clear (the Memory protocol)
```

- Omit `memory` and a hosted agent still gets the platform store; run anywhere else it is stateless. `memory="platform"` says so explicitly and fails at construction when the agent is not hosted.
- Redis keeps 24 hours of history per session. `agent.memory.search(subject, query)` and `agent.memory.purge(subject)` exist on the platform store.
- `metadata["genfleet.memory"] == "off"` makes the Agent skip loading and saving for that turn.
- Where `session_id` comes from over A2A: see "Sessions" in `references/serve.md`. A turn without one is stateless.

### MCP servers

```python
{"type": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}
{"type": "sse", "url": "http://localhost:3001/sse"}
```

Tool schemas are fetched on the first `run()` and merged with local tools. Either form accepts `customer_safe_tools: [names]` (offered to customers and operators, the 0.17 form) and `sensitive_tools: [names]` (owner approval before the call runs). Needs the `[mcp]` extra.

## Tools

Plain functions in `tools=` are auto-wrapped; the schema comes from type hints and the docstring. Tool results are always `str`.

```python
from genfleet.sdk import tool

@tool(description="adds two numbers")
def add(a: float, b: float) -> str:
    return str(a + b)
```

### Audiences and sensitive tools

On a hosted agent each turn carries `metadata["genfleet.caller"]` (read it with `caller_of(input.metadata)`): `role` (`operator` or `customer`), `id`, `channel`, `private`, `legacy_tools`, `name` (untrusted, never base a decision on it).

```python
@tool                                              # default: operators only
def internal_report() -> str: ...

@tool(audiences=["operator", "customer"])          # both
def order_status(order_id: str) -> str: ...

@tool(audiences=["customer"])                      # customers only, staff should not trigger it
def start_return(order_id: str) -> str: ...

@tool(audiences=["operator", "customer"], sensitive=True)   # waits for owner approval
def issue_refund(order_id: str, amount: float) -> str: ...
```

- An operator in a private chat is offered the tools whose audience includes `operator`; a customer, or an operator in a group chat, the ones that include `customer`. A tool that was not offered is refused as "not found".
- Your marking is only the default: the tenant owner sets each tool's audience on the platform, and that setting wins, wider or narrower. It reaches the agent as `metadata["genfleet.tool_audiences"]`.
- `sensitive=True`: the call does not run when the model makes it. The model is told it awaits approval, a workspace owner or admin approves the exact call, then it runs once in a follow-up turn. A sensitive tool is refused on a call from another agent. The owner can also mark or unmark tools on the platform.
- `customer_safe=True` is **deprecated** (0.18, removed in 1.0): it meant both audiences. Use `audiences=[...]`. Do not combine the two.
- Outside the platform there are no roles and nothing is filtered. Declare who the agent serves under `[agent] audiences` in `genfleet.toml` (see below).

### ToolContext: tool settings and secrets (0.21)

A tool that needs a setting or a credential declares it in its own `genfleet.toml` (`[config.<name>]`, `[secrets.<name>]`) and asks for a `ToolContext` by annotation. The parameter is left out of the tool schema, so the model never sees or fills it.

```python
from genfleet.sdk import ToolContext, tool

@tool
def create_issue(title: str, ctx: ToolContext) -> str:
    client = Jira(project=ctx.config["project"], token=ctx.secret("api_token"))
    ...
```

- `ctx.config` is a read-only mapping; `ctx.secret(name)` returns the value or raises `ToolSecretError` (naming the secret, never a value).
- **On the platform** the values are the tenant's installation of the tool. A secret that was not bound is unavailable; nothing else is consulted.
- **Locally** set `tool_env_namespace(slug) + "__CONFIG"` and `"__SECRETS"` (each a JSON object), or a plain variable named like the secret (`api_token` or `API_TOKEN`).
- Annotate as `ToolContext`, `ToolContext | None` or `Optional[ToolContext]`; any other spelling that names it raises `TypeError`. A tool from `load_tools()` shares the agent process, so its settings are not isolated from the agent's code.

### Sending messages: `send_message`

For alerts to staff, follow-ups and templates (the normal reply is sent automatically):

```python
from genfleet.sdk import ChannelSendError, send_message, send_message_tool

await send_message("whatsapp:201001234567", "Order shipped")
await send_message("whatsapp:201001234567", template="order_shipped", params=["A-1042"], language="en")
agent = Agent(..., tools=[send_message_tool])   # model-facing form: failures come back as text
```

`to` is `"<channel>:<address>"` (the inbound turn's `session_id` has that shape). Free text reaches only a contact who wrote in the last 24 hours; otherwise send an approved `template`. `ChannelSendError.code` names a refusal the agent can act on: `outside_window` (send a template), `cold_send_disabled`, `no_binding`, `invalid_request`, `send_rejected`, and `not_in_sandbox` when the agent is not hosted. A refusal means nothing was sent; with no `code`, `delivery_unknown` says the message may have gone out, so do not blindly resend.

## Files and parts (0.20)

A turn is a list of typed parts: `AgentInput.parts` and `AgentOutput.parts` hold `TextPart`, `ImagePart`, `AudioPart`, `VideoPart`, `FilePart`. `message` and `content` stay the turn's text, so a text-only agent never looks at `parts`. Build one with `part_for(mime_type, data=... | uri=...)`.

```python
agent = Agent(
    role="You read invoices.",
    model={"model": "openai/gpt-4o-mini", "api_key": "...", "vision": True},
    accepts=["image/*", "application/pdf"],
)
```

- Text is always accepted; files are refused unless declared via `accepts=`. Over A2A a refused type is error `-32005`; `Agent.run` raises `UnsupportedInput`.
- `"vision": True` sends PNG, JPEG, GIF and WebP images (up to 5 MiB each, within `max_images_per_turn` / `max_image_bytes_per_turn`) to the model as images. Any other file is described to the model in one line; its bytes never go into the prompt or memory.
- Bytes are inline (`data`, at most `MAX_INLINE_BYTES` = 1 MiB) or by reference (`uri`). A reference is read through a `FileResolver` (`Agent(files=...)`, `read_part`); the SDK never fetches a URL itself. `InMemoryFileResolver` is for tests.
- Return files from your own agent in `AgentOutput.parts`; declare their types with an `output_types` attribute or `serve(output_types=...)`.

## Skills (runtime instructions)

`Skill` holds versioned instructions. The model sees each skill's name and description and calls `read_skill(name)` (or `read_skill(name, path="references/x.md")`) to load one. Skills never grant tool access, and bodies are kept out of audit logs and saved memory. Do not put secrets in them.

```python
from genfleet.sdk import Agent, Skill, discover_skills, load_skills

Agent(..., skills=[Skill(name="concise", description="Short answers", instructions="One sentence.")])
Agent(..., skills=[Skill.from_file("skills/concise/SKILL.md")])      # frontmatter: name, description, version
Agent(..., skills=["owner/repo@<40-character-commit>"])               # pinned repository source

for option in discover_skills("owner/repo"):                          # list, then pick
    print(option.name, option.description)
selected = load_skills("owner/repo", names=["handoff"])               # omit names for all
```

A repository source passed directly to `Agent(skills=...)` must end in `@<40-character-commit>`; for deployment prefer loading the `Skill` objects at build time. Local repositories are passed as `Path`. Skills default to the `operator` audience; use `Skill(audiences={"operator", "customer"}, ...)`.

## Workflow state

State is separate from conversation memory: checkpoints, pause/resume and effect claims for long-running work.

```python
from genfleet.sdk import Agent, LocalStateStore

store = LocalStateStore("runs.sqlite3")               # local / self-hosted (SQLite file)
agent = Agent(..., state=store)                       # or state="platform" when hosted
run = await store.create("orders", owner_id, {"step": "start"})
run, token = await store.pause(run.run_id, agent="orders", owner=owner_id, version=run.version, state={"step": "wait"})
```

- Calls take `agent` and `owner`. Locally you choose both (use the verified caller id as `owner`); hosted, the platform sets the scope and ignores `owner`.
- Also `checkpoint`, `resume`, `complete`, `cancel`, `purge`, `purge_subject`, `claim_effect` / `complete_effect` (a stable idempotency key for an external side effect). Payloads are capped at 64 KiB (`StateTooLarge`); `StateQuotaExceeded` and `StateError` for the rest.
- **Pause is not an approval.** Whoever holds the resume token within the same owner scope can resume. Use `sensitive=True` tools for human approval.
- `state="platform"` fails when the agent is not hosted. `LocalStateStore` refuses to open on a hosted agent.

## Serving over A2A

```python
from genfleet.sdk.serve import serve, create_app

serve(agent, name="my-agent", description="Does things", host="0.0.0.0", port=8000,
      public_url=None, accept_history=False, accepted_input_types=None,
      output_types=None, max_body_bytes=8 * 1024 * 1024)
app = create_app(agent, name="my-agent")     # same options; returns a FastAPI app
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/.well-known/agent-card.json` | Agent card |
| `POST` | `/` | JSON-RPC 2.0, A2A v1.0 (`SendMessage`, `SendStreamingMessage`); 0.x `tasks/send` / `tasks/sendSubscribe` still work for one minor release |

```bash
curl -X POST http://localhost:8000 -H "Content-Type: application/json" -H 'A2A-Version: 1.0' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"hello"}]}}}'
```

Option meanings, sessions, files on the wire and the error table are in `references/serve.md`.

## AgentProtocol: direct implementation

For frameworks that run their own LLM loop (LangChain, CrewAI), implement `run` yourself:

```python
class MyAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content=f"Got: {input.message}", done=True)
```

`serve` does **not** execute `tool_calls` that such an agent yields: it only relays text, files and tool events. Run your tools inside `run` and yield the final text. See `references/frameworks.md`.

## genfleet.toml

```toml
[agent]
name = "support"                 # required
kind = "code"                    # "declarative" | "code" | "native_sdk" (default "code")
description = "Answers order questions"
entry = "agent:build_agent"      # module:factory (default)
audiences = ["operator", "customer"]   # who the agent serves; absent = operators only

[tools]
"@acme/jira" = "*"               # slug = version spec; push pins version and digest
```

`[mcps]` takes the same slug form. `load_agent_manifest(dir)` and `load_tool_manifest(dir)` validate a manifest offline and raise `ManifestError` with the file path.

## Key Rules

1. **SDK imports only from itself.** Never import `genfleet.core`, `genfleet.runner`, `genfleet.loader`, `genfleet.engine` or `genfleet.protocols`; those are platform internals.
2. **Use `Agent()` as the default path.** Implement `AgentProtocol` directly only for frameworks that manage their own LLM calls.
3. **Entry factory takes `remote_tools`.** For a platform agent write `def build_agent(remote_tools=())` and merge it: `tools=[*load_tools(), *remote_tools]`. `load_tools()` returns the tools the manifest names; `remote_tools` is what the runtime discovered (other agents, remote tool servers) and no manifest can name. Dropping it silently loses those tools.
4. **Audiences, not `customer_safe`.** Mark tools with `audiences=[...]`, and `sensitive=True` for anything with a consequence. An unmarked tool is operator-only.
5. **`RawAdapter` was removed in 0.4.0.** Use `Agent()`; see `references/raw-adapter.md`.
6. **Python 3.12+.** Use `--python 3.12` when creating venvs with `uv`.
7. **Provider extras are required.** `Agent` lazy-imports the provider library; install `[openai]`, `[anthropic]` or `[gemini]`.
8. **Tools return `str`.** `ToolWrapper.call()` always returns `str`.
9. **No secrets in code or manifests.** Use `ToolContext` for tool credentials; manifests declare secret names only.

```python
from genfleet.sdk import Agent, load_tools

def build_agent(remote_tools=()):
    return Agent(
        role="...",
        model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
        tools=[*load_tools(), *remote_tools],
    )
```

## Exports

```python
from genfleet.sdk import (
    # Agents and protocols
    Agent, AgentProtocol, ToolProtocol, AgentFactory,
    # Turn schemas
    AgentInput, AgentOutput, Message, ToolCall, ToolSchema, TokenUsage, TOOL_EVENT_KEY,
    # Parts and files
    Part, TextPart, ImagePart, AudioPart, VideoPart, FilePart, MediaPart, MAX_INLINE_BYTES,
    part_for, UnsupportedInput, FileResolver, InMemoryFileResolver, read_part,
    FileAccessError, FileNotFound, FileExpired, FileTooLarge, UnsupportedReference,
    # Tools
    tool, ToolWrapper, ToolContext, ToolSecretError, tool_env_namespace,
    load_tools, ToolResolver, LocalToolResolver, ToolResolutionError,
    # Audiences
    Caller, caller_of, is_hosted, may_offer, tool_audiences_of, Audience, EVERYONE, OPERATOR_ONLY,
    CALLER_METADATA_KEY, TOOL_AUDIENCES_METADATA_KEY,
    # Memory, skills, state, channels
    Memory, Skill, SkillOption, discover_skills, load_skills,
    StateStore, LocalStateStore, state_for, RunState, EffectState,
    StateError, StateTooLarge, StateQuotaExceeded,
    send_message, send_message_tool, ChannelSendError,
    # Config and audit
    ModelConfig, MemoryConfig, MCPConfig, AuditConfig, AuditEvent,
    # Manifests
    AgentManifest, ToolManifest, ToolRef, ConfigField, SecretField,
    ManifestError, load_agent_manifest, load_tool_manifest,
)
from genfleet.sdk.serve import create_app, serve    # [serve] extra
```

`may_use` and `Caller.full_toolset` are deprecated and removed in 1.0. `PlatformStateStore` / `PlatformStateError` are also exported; you normally reach them through `state="platform"`.

## Reference Files

- `references/api.md` - field-level reference: Agent, config types, schemas, parts, tools, skills, state, manifests
- `references/serve.md` - A2A serve module: options, sessions, files on the wire, endpoints, error codes
- `references/frameworks.md` - provider examples and LangChain / CrewAI via `AgentProtocol`
- `references/raw-adapter.md` - RawAdapter (removed in 0.4.0), migration reference only
