# Changelog

All notable changes to `agentinc-sdk` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
