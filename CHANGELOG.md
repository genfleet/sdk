# Changelog

All notable changes to `withfleet-sdk` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Add a shared async workflow state interface with a hosted proxy client and a local SQLite backend. Runs support bounded checkpoints, one-time resume claims, cancellation, caller/subject purge, and external-effect claims. Hosted sandboxes refuse SQLite because their files are ephemeral.
- Add repository skill discovery and selection, including pinned `owner/repo@commit` sources for `Agent`. Skill instructions are read on demand, filtered by caller audience, and omitted from audit events and persisted memory.
- Accept caller-provided `Memory` implementations for local development and custom stores.
- Add a notebook covering skills, tools, memory, and an optional free OpenRouter model.


## 0.19.0

- **Sensitive tools (ADR-0028 §8a).** `@tool(sensitive=True)` (or `sensitive = true` in a tool's manifest, or an MCP config's `sensitive_tools`) marks a tool whose call waits for a workspace owner's or admin's approval. The platform's per-tool flags (`metadata["genfleet.sensitive_tools"]`, `{key: bool}`, keyed like audiences) are **add-only**: `true` marks a tool sensitive, `false` never unmarks one its author marked. The map is unsigned, so a forged one can't switch an approval off. `remember` is never sensitive.
  - When the model calls one, it does **not run**. The model is told the action awaits approval, and the tool result event has `ok: false` and `status: "pending"`. The request (`{tool key, arguments}`) is reported on the turn's final chunk, `metadata["genfleet.confirmation_requests"]`.
  - On a turn another agent started (`genfleet.peer_turn`), a sensitive tool is refused outright.
- **Approved calls.** A follow-up turn can carry `metadata["genfleet.approved_call"]`, signed by the engine with a per-spawn key in `GENFLEET_APPROVAL_KEY`. `Agent` runs it **first and exactly once**, before the model, and only if all of these hold:
  - the HMAC-SHA256 signature over `canonical_json` of `{confirmation_id, tool, arguments, args_hash, expires_at}` verifies (`hmac.compare_digest`);
  - it hasn't expired;
  - its `confirmation_id` hasn't already run in this process;
  - the tool is offered to the caller.

  Otherwise nothing runs and the model is told so. A verified call is used up even when it can't run (unknown tool, not offered now), so it can never run later. A tool that raises becomes a failed result the model reports, instead of ending the turn. The key sits in the agent's own environment, so the signature stops other agents and direct callers, not the agent's code. For code that doesn't use `Agent`, this is advisory.
  - The turn's final chunk says what became of it, under `metadata["genfleet.approved_call_result"]` (`APPROVED_CALL_RESULT_KEY`): `"ran"`, `"failed"` (the tool raised, or an MCP call failed; a local tool that returns an error string counts as `"ran"`) or `"refused"` (no key, malformed, a bad signature, expired, already run, an unknown tool, or not offered to the caller). A turn without an approved call has no such key.
  - It is agent-reported, so it is advisory. A turn that errors before its final chunk (the provider, the audit sink) sends no result, even if the tool ran: read a missing result as "unconfirmed", not "did not run". It reaches the platform only through the non-streaming invoke (`serve`'s streaming paths drop final-chunk metadata).
  - Only the tool's own exception becomes a failed result; one from the audit sink after the tool ran ends the turn as before.
- An approved call is verified **and claimed in one step**. A signature that isn't 64 lowercase hex characters, and arguments with NaN, infinities or lone surrogates (`canonical_json` uses `allow_nan=False`), are malformed and refused without an error. It runs through the same audited dispatch as any tool call (`tool.call` / `tool.result`).
- Exported for the engine: `APPROVAL_KEY_ENV`, `PEER_TURN_KEY`, `canonical_json`, `sign_approved_call` and the metadata keys. The agent-side helpers in `genfleet.sdk.confirmations` are private.
- MCP tools whose names start with `@` are skipped: that's a slug key's form.

## 0.18.1

- **Tool keys never mix (ADR-0028 §5).** The platform's per-tool setting (`genfleet.tool_audiences`) has two key spaces:
  - A manifest tool, one that carries a `slug` from `load_tools()`, matches **only** its slug key: the slug with a leading `@`, from the new `slug_key()`. So `@acme/crm` stays as is, and a platform tool `search` is `@search`. A bare name never reaches it.
  - Any other tool (code-defined, MCP, peer) matches only its name.

  In 0.18.0 a manifest tool matched its name first. The owner's setting for an installed `search` then also opened a code or MCP tool called `search`, and the other way round.

## 0.18.0

- **Tool audiences (ADR-0028).** A tool is offered by its audience, set with `@tool(audiences=[...])`: `["operator"]` (the default), `["operator", "customer"]` or `["customer"]`. `ToolWrapper.audiences` holds the set. An empty or unknown audience list is a `ValueError`.
  - An operator in a private chat gets the tools for `operator`. A customer, or anyone in a group, gets the tools for `customer`.
  - **New:** a `["customer"]` tool is hidden from staff in a private chat.
  - `legacy_tools`, and an agent the platform didn't spawn, get every tool.
  - A hosted turn with a missing or malformed caller is a customer in a group. It now also gets `["customer"]` tools, never staff tools.
- **The platform's per-tool setting wins.** On a hosted turn, `metadata["genfleet.tool_audiences"]` (`TOOL_AUDIENCES_METADATA_KEY`) holds the tenant owner's audience per tool, keyed by tool name or manifest slug. It replaces the author's marking, wider or narrower (`effective_audiences`: name, then slug, then the marking).
  - It covers every tool an `Agent` holds: its own functions, `load_tools()` tools, remote tools passed in as `tools=`, and MCP tools (by name). Tools the engine keeps itself, such as runner-held peers and `remember`, are filtered by the engine.
  - **Fail closed:** an entry whose value can't be read is offered to no one. A value that isn't a mapping is ignored.
  - **Deploy gate:** the engine must replace or strip this key on every turn before a runtime image ships 0.18. An engine that passes request metadata through would let an invoke set audiences. See the engine PR for backend#248.
- **`load_tools()` tools carry their manifest slug.** A function or method comes back as a fresh `functools.wraps` proxy that behaves exactly like it: sync stays sync and the result is unchanged. Any other callable (a builtin, a partial, an `itemgetter`, a callable instance) comes back as a `ToolWrapper`, named by its slug if it has no `__name__`. `ToolWrapper`s come back as copies (`with_slug`). The loaded objects are never modified. Schemas for callables without type hints no longer fail; their parameters are left untyped.
- New in `genfleet.sdk`: `may_offer`, `tool_audiences_of`, `Audience`, `EVERYONE`, `OPERATOR_ONLY` and `TOOL_AUDIENCES_METADATA_KEY`, plus `ToolWrapper.with_slug` and `genfleet.sdk.tool.wrap_tool`.
- **Deprecated, removed in 1.0 (backend#252), each with a `DeprecationWarning`:**
  - `customer_safe=` on `@tool` / `ToolWrapper` (it means both audiences);
  - `may_use(caller, customer_safe=…)` (use `may_offer`);
  - `Caller.full_toolset` (use `may_offer(caller, OPERATOR_ONLY)`).
  - `ToolWrapper.customer_safe` reads true only for both audiences. Assigning to it still works for this release, with a warning.
  - Deprecation warnings point at the caller's code.
  - MCP configs' `customer_safe_tools` stays, as the legacy marking for both audiences.

## 0.17.1

- **A sandbox is hosted when `GENFLEET_HOSTED` is set.** The engine sets it in every sandbox it spawns, process sandboxes included. Before, the SDK keyed on `GENFLEET_A2A_TOKEN`, which a process sandbox never gets, so a process-hosted agent reached without `genfleet.caller` offered every tool. `GENFLEET_A2A_TOKEN` still counts as hosted, for engines that don't set the new marker yet. `HOSTED_ENV` is now `"GENFLEET_HOSTED"`.

## 0.17.0

- **Operators and customers (ADR-0028).** On a hosted turn the platform sets `metadata["genfleet.caller"]` = `{role, id, name, channel, private, legacy_tools}`, resolved from the provider-verified sender and overwritten from any value a caller sends. `role` is `"operator"` (tenant staff) or `"customer"`. `name` is the sender's own display name: untrusted text, never an authorization input. `genfleet.sdk.caller` adds `CALLER_METADATA_KEY`, `Caller`, `caller_of(metadata)` and `may_use(caller, customer_safe=…)`.
- **Tools are operator-only unless marked customer-safe.** `@tool(customer_safe=True)` marks a tool; an MCP config lists its customer-safe tools in `customer_safe_tools`. On a customer's turn, or anyone's turn in a group chat (where everyone sees the reply), `Agent` offers only customer-safe tools, and a call to any other tool is refused as unknown. An operator in a private chat gets every tool; so does every turn of an agent the platform migrated with the `legacy_tools` grace flag. A turn without `genfleet.caller` is not filtered only when the platform did not spawn the agent (a local `serve`, a test); in a platform sandbox (`GENFLEET_A2A_TOKEN` set) it gets the narrowest set. An MCP tool whose name is already taken (by a local tool or another server) is skipped, so a server cannot lend its mark to another tool. **Breaking for hosted customer-facing agents:** mark the tools customers may use before the platform turns roles on.
- **`audiences` in `genfleet.toml`** (`[agent]`): `["operator"]`, `["customer"]` or both. Left out, the platform serves operators only. Entries are deduplicated; an empty list is a `ManifestError`.
- The 0.x A2A card path and methods, announced for removal in this minor, stay one more minor (sdk#47).

## 0.16.0

- **`serve` speaks A2A v1.0** (JSON-RPC binding, ADR-0027). The card is at `/.well-known/agent-card.json` with `supportedInterfaces`, `capabilities.pushNotifications: false` and `capabilities.extendedAgentCard: false`. `SendMessage` returns a completed task whose reply is artifact `result`; `SendStreamingMessage` streams `TASK_STATE_WORKING`, appended `result` chunks and `TASK_STATE_COMPLETED` (or `TASK_STATE_FAILED`). The message's `contextId` becomes `metadata["session_id"]` (over a `session_id` in request metadata); a message with no text part is refused with `-32005`; prior turns in request `metadata["genfleet.history"]` reach the agent only with the new `accept_history=True` (off by default: the key is stripped). A new `public_url=` sets the URL the card advertises; without it the card follows the request's `Host` and is sent `Cache-Control: no-store`. `GetTask`, `ListTasks`, `CancelTask` and `SubscribeToTask` answer UnsupportedOperation (`serve` keeps no tasks); push notifications are not supported; an `A2A-Version` other than 0.x or 1.x is refused.
- **Breaking for self-hosted `serve`: caller history is off by default.** The 0.x `params.history` (and the v1.0 `metadata["genfleet.history"]`) no longer reaches the agent unless the app sets `accept_history=True`; the `genfleet.history` key is always stripped from the metadata, on both paths. A bad `public_url` (not an absolute http(s) URL, or with a query or fragment) is refused when the app is built.
- **`Agent` honours a memory-off turn:** with `metadata["genfleet.memory"] == "off"` (set by the platform for a turn from its public A2A edge) the session is neither loaded nor written.
- **For one minor release** the 0.x card path `/.well-known/agent.json` and the 0.x methods `tasks/send` / `tasks/sendSubscribe` keep working. They will be removed in the next minor.

## [0.15.0] - 2026-10-07

### Added
- **`models` in `genfleet.toml`** (`[agent]`): the platform catalog model ids the agent calls, e.g. `models = ["general-fast"]`. In hosted execution the platform scopes the agent's model credential to these. Entries are lowercased and deduplicated, at most 20; a `provider/model` id or anything but a catalog id (lowercase letters, digits, `.` and `-`) is a `ManifestError`. `AgentManifest.models` holds them.

## [0.14.0] - 2026-10-07

### Added
- **Pluggable model clients** (`genfleet.sdk.models`): an agent's model calls go through a `ModelClient` (the provider protocol, promoted). Packages register a factory `(ModelConfig) -> ModelClient | None` under the entry-point group `genfleet.model_clients`; installed factories are tried in name order and the first active one is used, else the direct `provider/model` provider. `Agent(model_client=...)` overrides both. `FakeModelClient` for tests.
- `ModelConfig.options`, passed to the model client unchanged (the SDK never reads it); `ModelConfig.api_key` is optional (a direct provider still refuses a missing one with a clear error; an empty string passes through as before).
- `genfleet.sdk.turn.current_turn()`: the running turn's `AgentInput.metadata`, read-only, visible only while the turn's model client or a tool runs (never to the caller or to another turn).

## [0.13.0] - 2026-10-06

### Added
- **`egress` in `genfleet.toml`** (ADR-0021 §3): the hosts an agent or tool reaches, declared in `[agent]` or `[tool]` as `egress = ["api.github.com", "*.googleapis.com", "hooks.example.com:8443"]`. A hosted sandbox leaves only through the platform's egress proxy, and the hosts a manifest declares are one of the lists the proxy allows. An agent is allowed its own entries plus those of every tool it pins. `AgentManifest.egress` and `ToolManifest.egress` hold the entries validated, lowercased, without a trailing dot or a default `:443`, and deduplicated. A bad entry is a `ManifestError` naming the file.
- `genfleet.sdk.egress`: `parse_egress_entry` / `parse_egress` (an `EgressEntry` with `canonical` and `matches(host, port)`), raising `EgressEntryError` with a `reason` of `invalid_host`, `ip_not_allowed` or `limit`. An entry is a hostname of at least two labels, optionally starting with `*.` (any subdomain at any depth, never the bare domain) and optionally ending with `:port` (default 443). IP literals (including resolver and URL-parser shorthands such as `127.1` and `1.2.3.0x`), a bare `*`, any non-ASCII or internal whitespace, and more than 100 entries are refused; ASCII whitespace at the ends is trimmed. Every wildcard carries `EgressEntry.review_warning`; there is no public-suffix list, so `*.github.io` is valid and marketplace review decides. `tests/fixtures/egress-hosts.cases.json` pins the rules; the platform tests its own validators against a copy.

## [0.12.0] - 2026-10-04

### Added
- **`Agent.run` reports the tools it runs** (sdk#39), so a hosted agent's tool calls show in chat. Next to the text it yields, per tool round: one `AgentOutput(content="", done=False, metadata={"tool_event": {"type": "tool_call", "id", "name", "arguments"}})` per call once its input is complete, then one `{"type": "tool_result", "id", "name", "ok", "output"}` per call as its tool returns. `ok` is false for an unknown tool or a failed MCP call. The shape is the one the platform runner emits for tools it dispatches (genfleet/engine#155); values are raw, and the platform redacts and truncates them before a user sees them. Text chunks are unchanged, and a consumer that only reads `content` sees the same text as before.
- `TOOL_EVENT_KEY` (`"tool_event"`), exported from `genfleet.sdk`.
- `serve` relays tool events on non-final `working` status updates under `metadata.tool_event`, as the platform's A2A server does, with a tool result's `output` capped at 4000 characters. A failed result's `output` goes out as the fixed `"The tool failed."`: it is error text, and `serve` never sends exception text to its caller (sdk#20). A tool call's `arguments` are redacted (tool-call stream contract v1.1 redaction v2: secret-looking keys, header-style `{name: "Authorization", value}` pairs, and credential-looking values such as `Bearer …` or `sk-…`) and then capped at 8 KB of JSON, past which they become `{"_truncated": true, "preview": …}`.

### Changed
- **A self-hosted `serve` now sends each successful tool's raw output to its caller** on `tasks/sendSubscribe`, up to 4000 characters per result. Before 0.12.0 a caller got only the model's text. A tool that reads a database row, a file or an internal API exposes that data to whoever calls your `serve` app directly. On the platform the engine scrubs tool output before a user sees it; a `serve` you host yourself does not. `tasks/send` still returns only the text.

### Fixed
- An MCP tool call that fails without raising is reported with `ok: false`: a result flagged `isError` by the MCP server, and an unsupported transport. Before, both said `ok: true`, so `serve` relayed their error text.

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
