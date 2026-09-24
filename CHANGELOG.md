# Changelog

All notable changes to `withfleet-sdk` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.1] - 2026-09-24

### Fixed
- **`invocation.end` is emitted for served turns.** It was emitted after the final `done` yield, and a consumer that stops reading at `done` — `serve` does — closed the generator first, so every served turn had an `invocation.start` and no `invocation.end` (and no total token usage in the audit trail). It is now emitted before the `done` chunk, like the memory write fixed in 0.10.0.

## [0.11.0] - 2026-09-22

### Added
- **`send_message`** (ADR-0019 §5). `await send_message("whatsapp:2010…", "text")` or `send_message(to, template="name", params=[...])` sends on one of the agent's channel bindings through the sandbox's channels proxy (`GENFLEET_CHANNELS_URL` / `GENFLEET_CHANNELS_TOKEN`, set by the platform). The agent never holds the channel token; the platform picks the binding, enforces the 24-hour window and cold-send policy, and says which rule refused a send via `ChannelSendError.code` (`outside_window`, `cold_send_disabled`, `no_binding`, `invalid_request`, `send_rejected`). The reply to an inbound message is still sent by the platform — this is for everything else.
- `send_message_tool`: the same as a model-callable tool (with `language` for templates), answering `"sent"`, `"not sent: <reason>"` for a refusal, or `"delivery unknown …; do not resend without checking"` when the platform could not confirm the outcome — so a refusal is something the model can act on, and a timeout does not become a duplicate message.
- `ChannelSendError.delivery_unknown`; `code="not_in_sandbox"` when the channels variables are absent.

### Fixed
- **Tool schemas typed `X | None` parameters as `{}`.** `@tool` only recognised `typing.Union`, and `str | None` is a `types.UnionType`, so every optional parameter written in the modern spelling reached the model with no type. Both spellings are now read.

## [0.10.0] - 2026-09-18

### Added
- **Platform episodic memory** (ADR-0018). `Agent(memory="platform")` — or `memory` omitted, when the sandbox provides `GENFLEET_MEMORY_URL` / `GENFLEET_MEMORY_TOKEN` — selects `PlatformMemory`, an HTTP client for the engine's memory proxy. It never sends a tenant or agent id; the proxy's per-spawn token *is* the scope. Stdlib HTTP, no new dependency.
- `AppendableMemory`: a backend that can add one turn without rewriting the thread. `Agent.run` appends the turn on such a backend and only falls back to whole-thread `save` on others (Redis). `subject` and `channel` from the input metadata are kept with the session.
- **Turn idempotency.** `AppendableMemory.append` takes a `turn_id`, and `Agent.run` fills it from the input metadata (`turn_id`, `request_id` or `message_id`), falling back to the invocation id. A store that honours it drops a replayed turn instead of storing it twice.
- `PlatformMemory.search(subject, query)` and `.purge(subject)`; `Agent.memory` exposes the backend.

### Fixed
- **A served agent now persists its turn.** `Agent.run` wrote to memory after yielding `AgentOutput(done=True)`, and a consumer that stops reading at `done` — `serve` does — closed the generator first, so nothing was ever written. The turn is written before the `done` yield. Present since Redis memory shipped.
- **A provider's `done` no longer ends the turn early.** A content chunk that arrived with `done=True` (the non-streaming provider path) was forwarded as-is, so a consumer stopped reading before the agent's own final chunk — and before any tool round still to come. `done` is now set once, by the agent, at the end of the turn.

### Removed
- **`Agent(data=...)` and `DataConfig`** — reserved for RAG since 0.1 and never implemented. RAG is the semantic layer of memory and arrives through the same backend. Passing `data=` is now a `TypeError`. **No deprecation cycle**, unlike `RawAdapter` in 0.2 → 0.4: the parameter was accepted and silently ignored, so nothing depended on its behaviour, and 0.x allows the break.

### Changed
- `MemoryConfig.type` accepts `"platform"`; `connection` is required only for `"redis"`, and a `"redis"` config without it now raises a `ValueError` naming the key instead of a bare `KeyError`.
- `PlatformMemory` pins one response shape per call: `search` requires `{"results": [...]}` with snake_case keys, and a non-JSON or unexpected body raises `PlatformMemoryError` rather than `JSONDecodeError`/`AttributeError`. `PlatformMemoryError` is exported from `genfleet.sdk.memory`.
- `memory` is typed `MemoryConfig | Literal["platform"] | None`, so a mistyped backend name is a type error rather than a runtime one.

## [0.9.1] - 2026-09-17

### Added
- `openrouter/` requests carry OpenRouter's attribution headers (`HTTP-Referer: https://genfleet.ai`, `X-OpenRouter-Title: Genfleet`) when they go to the gateway. `ModelConfig.default_headers` lets a caller add or override headers on any OpenAI-compatible provider.

## [0.9.0] - 2026-09-16

### Added

- **`openrouter/<vendor>/<model>` provider prefix** — routes to the OpenAI provider pointed at the OpenRouter gateway (`https://openrouter.ai/api/v1`), passing the vendor id through intact, so one key reaches every vendor OpenRouter fronts.
  ```python
  {"model": "openrouter/google/gemini-2.5-flash", "api_key": "sk-or-..."}
  ```
  A non-empty `base_url` in the config still wins; a missing, `None` or empty `base_url` falls back to the gateway. The prefix requires the vendor segment — `openrouter/<model>` raises `ValueError`.

## [0.5.0] – [0.8.2]

See git history.

## [0.4.1] - 2026-08-01

### Fixed

- **`fleet` is once again a PEP 420 implicit namespace package** — the SDK shipped an empty `fleet/__init__.py`, which made `fleet` a *regular* package in this distribution. Installed alongside any sibling (`withfleet-core`, `withfleet-engine`, `withfleet-loader`, `withfleet-runner`, `withfleet-adapter`, `withfleet-protocols`), `fleet.__path__` collapsed to the single site-packages directory and every sibling became unimportable with `ModuleNotFoundError: No module named 'fleet.core'`. The file is removed and the wheel build now targets `fleet/sdk` directly, so no `fleet/__init__.py` is synthesized. No public API changed.

## [0.4.0] - 2026-07-28

### Breaking Changes

- **`RawAdapter` removed** — deprecated since 0.2 with a `DeprecationWarning` announcing removal in v0.4. Use `Agent()` instead. Migration guidance is kept in `skills/withfleet-sdk/references/raw-adapter.md`.
  ```python
  # before
  from fleet.sdk import RawAdapter
  serve(RawAdapter(my_fn), name="agent", port=8000)

  # after
  from fleet.sdk import Agent
  serve(Agent(role="...", model={...}), name="agent", port=8000)
  ```

### Added

- **`ToolCall.thought_signature`** — optional `str | None` field (base64-encoded), defaulting to `None`. Set only by the Gemini provider on Gemini 3.x; every other provider leaves it unset and it is never emitted into their payloads.

### Fixed

- **Gemini 3.x tool calling** — agents on Gemini 3.x models failed on the second LLM turn with `400 INVALID_ARGUMENT: Function call is missing a thought_signature in functionCall parts`. Gemini 3.x issues an opaque signature per function call and requires it echoed back verbatim when that call is replayed in history. The signature lives on the enclosing `Part`, not on `FunctionCall`, so the flattened `response.function_calls` accessor could not see it; the provider now walks `candidates[].content.parts[]` and pairs each function call with its sibling signature, replaying it on the `Part` when converting history. Signatures also survive the memory round-trip, so turn 3+ works for Redis-backed agents. Gemini 2.5 and all other providers are unaffected; histories written before this change decode without error.

## [0.3.1] - 2026-07-07

### Fixed

- **Gemini tool-use conversation history** — assistant `tool_calls` are now converted to Gemini `function_call` parts and tool results resolve their function name via `tool_call_id`; previously the follow-up request after any tool call sent an orphan `function_response` with an empty name, which Gemini rejected with a 400 `ClientError`

## [0.3.0] - 2026-07-06

### Breaking Changes

- **Model format** — model names now use explicit `provider/model-name` format instead of auto-detection from prefix.
  - `"gpt-4o-mini"` → `"openai/gpt-4o-mini"`
  - `"claude-sonnet-4-6"` → `"anthropic/claude-sonnet-4-6"`
  - `"gemini-1.5-pro"` → `"gemini/gemini-1.5-pro"`
  - OpenAI-compatible endpoints: `"deepseek-chat"` → `"openai/deepseek-chat"` (with `base_url`)
  - Supported providers: `openai`, `anthropic`, `gemini`

### Added

- **Audit system** — opt-in structured audit logging via `audit=` parameter on `Agent()`.
  - Three built-in backends: `console` (structured JSON logger), `file` (JSONL), `callback` (custom function)
  - Seven event types: `invocation.start`, `llm.request`, `llm.response`, `tool.call`, `tool.result`, `invocation.end`, `invocation.error`
  - Configurable content truncation (`max_content_length`, default 500; set 0 for unlimited)
  - Event filtering via `events` list
  - New exports: `AuditConfig`, `AuditEvent`
- **Token usage tracking** — native token counting across all providers.
  - `TokenUsage` model with `input_tokens`, `output_tokens`, `total_tokens`
  - Extracted automatically from OpenAI, Anthropic, and Gemini API responses
  - `token_usage` field added to `AgentOutput`
  - Cumulative token usage included in `llm.response` and `invocation.end` audit events
- **LLM response content in audit** — `llm.response` events include actual response text and tool calls
- **Example** — `examples/audit_agent.py` demonstrating all three audit backends

### Fixed

- **A2A session forwarding** — the serve endpoint now forwards `sessionId`, `metadata`, and `history` from A2A params into `AgentInput` (both `tasks/send` and `tasks/sendSubscribe`), enabling session memory over A2A; `sessionId` populates `metadata["session_id"]` unless the caller set it explicitly, and invalid `history` is ignored rather than failing the request
- **Memory persistence produces valid replayed histories** — assistant tool-call turns are persisted with their `tool_calls` (JSON-encoded arguments, not Python repr), tool results carry `tool_call_id`, and the assistant's final reply (including text streamed alongside tool calls) is saved; previously the second turn of a session with tools failed with a provider 400 on the orphan `tool` message
- **JSON-RPC input validation** — the A2A endpoint returns a proper `-32600 Invalid Request` for non-dict request bodies (arrays, strings) and normalizes non-dict `params` to `{}` instead of raising HTTP 500
- **OpenAI streaming token usage** — usage chunk arrives after the `finish_reason` chunk; the stream is now fully consumed before yielding the final output
- **Anthropic tool dispatch** — messages are now properly converted from OpenAI format to Anthropic's `tool_use`/`tool_result` content block format, fixing `Unexpected role "tool"` errors

## [0.2.0] - 2025-06-01

### Added

- **Agent class** — high-level developer-facing class wiring provider, tools, memory, and MCP into a single `AgentProtocol`-compatible object
- **Built-in providers** — OpenAI, Anthropic, and Gemini with auto-detection from model name prefix
- **Redis memory** — `MemoryConfig` for persistent session history with 24-hour TTL
- **MCP integration** — stdio and SSE transports for Model Context Protocol servers
- **Examples** — OpenAI tools, Anthropic, memory, MCP, multi-tool, and RAG agents

### Deprecated

- **RawAdapter** — use `Agent()` instead; will be removed in v0.4

## [0.1.0] - 2025-05-13

### Added

- **AgentProtocol** — runtime-checkable protocol defining `run(input) -> AsyncIterator[AgentOutput]`
- **ToolProtocol** — runtime-checkable protocol defining `schema()` and `call()`
- **Schemas** — `AgentInput`, `AgentOutput`, `Message`, `ToolCall`, `ToolSchema` (Pydantic v2 models)
- **@tool decorator** — turns any sync/async function into a `ToolProtocol` with auto-generated JSON Schema
- **ToolWrapper** — class backing `@tool`, callable directly or via `ToolCall` dispatch
- **RawAdapter** — wraps any callable as `AgentProtocol`, auto-detecting 6 signature patterns
- **A2A serve module** — `create_app()` and `serve()` helpers to expose agents over A2A (JSON-RPC 2.0 + SSE)
- **Examples** — echo, streaming, tool, OpenAI, Anthropic, LangChain, and CrewAI agents
- **Documentation** — single-page HTML docs with full API reference
