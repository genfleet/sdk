# Graph Report - .  (2026-07-25)

## Corpus Check
- Corpus is ~23,343 words - fits in a single context window. You may not need a graph.

## Summary
- 448 nodes · 1030 edges · 18 communities
- Extraction: 90% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 94 edges (avg confidence: 0.69)
- Token cost: 192,058 input · 0 output

## Community Hubs (Navigation)
- Agent Core, Memory & MCP
- RawAdapter & Example Agents
- Audit Backends & Events
- API Reference & Schemas
- Release History & Publishing
- A2A Serve Layer & Tool Examples
- Tool Decorator & Agent Tests
- Serve Session Forwarding Tests
- Gemini Provider & Conversion
- Favicon Icon Set
- Brand Logo System
- LightRAG Retrieval Agent
- Anthropic Provider
- CI Test Matrix & Extras
- Vault Documentation Rules
- Tool Protocol Surface

## God Nodes (most connected - your core abstractions)
1. `AgentInput` - 59 edges
2. `AgentOutput` - 59 edges
3. `ToolCall` - 38 edges
4. `ToolSchema` - 28 edges
5. `AuditEvent` - 23 edges
6. `Agent` - 22 edges
7. `Message` - 19 edges
8. `audit_for()` - 18 edges
9. `AgentProtocol` - 16 edges
10. `_make_agent()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `agentinc CLI (login/publish/agents list)` --semantically_similar_to--> `Publish to PyPI Workflow`  [INFERRED] [semantically similar]
  docs/index.html → .github/workflows/publish.yml
- `Publish to PyPI Workflow` --semantically_similar_to--> `Ship to the Marketplace flow`  [INFERRED] [semantically similar]
  .github/workflows/publish.yml → docs/index.html
- `agentinc-sdk coding-agent skill` --semantically_similar_to--> `Agentinc SDK Developer Documentation (single-page)`  [INFERRED] [semantically similar]
  README.md → docs/index.html
- `test_message_with_tool_call_id()` --calls--> `Message`  [EXTRACTED]
  tests/test_schemas.py → agentinc/sdk/schemas.py
- `test_tool_schema()` --calls--> `ToolSchema`  [EXTRACTED]
  tests/test_schemas.py → agentinc/sdk/schemas.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Agent wires provider, memory, MCP, and audit configs** — readme_agent, readme_modelconfig, readme_memoryconfig, readme_mcpconfig, readme_auditconfig [EXTRACTED 1.00]
- **Tool-call round-trip history conversion fixes across providers** — changelog_gemini_tool_history_fix, changelog_anthropic_tool_dispatch_fix, changelog_memory_replay_fix, changelog_openai_streaming_usage_fix [INFERRED 0.85]
- **Marketplace publishing flow: manifest, entrypoint, CLI, secret injection** — docs_index_marketplace_publishing, docs_index_agent_json_manifest, docs_index_entrypoint_agent_variable, docs_index_agentinc_cli, docs_index_runtime_secret_injection [EXTRACTED 1.00]
- **Declare Agent -> AgentProtocol -> serve over A2A** — skills_agentinc_sdk_references_api_agent, skills_agentinc_sdk_references_api_agentprotocol, skills_agentinc_sdk_references_api_serve, skills_agentinc_sdk_references_serve_tasks_send, skills_agentinc_sdk_references_serve_agent_card [EXTRACTED 1.00]
- **Tool dispatch data model (@tool, ToolWrapper, ToolSchema, ToolCall, ToolProtocol)** — skills_agentinc_sdk_skill_tool_decorator, skills_agentinc_sdk_references_api_toolwrapper, skills_agentinc_sdk_references_api_toolschema, skills_agentinc_sdk_references_api_toolcall, skills_agentinc_sdk_references_api_toolprotocol [EXTRACTED 1.00]
- **Provider backends selected via ModelConfig prefix** — skills_agentinc_sdk_references_api_modelconfig, skills_agentinc_sdk_references_frameworks_openai, skills_agentinc_sdk_references_frameworks_anthropic, skills_agentinc_sdk_references_frameworks_gemini, skills_agentinc_sdk_references_frameworks_openai_compatible [EXTRACTED 1.00]

## Communities (18 total, 0 thin omitted)

### Community 0 - "Agent Core, Memory & MCP"
Cohesion: 0.06
Nodes (45): Agent, _fetch_mcp_tools(), _invoke_mcp_tool(), Connect to an MCP server and return its tool schemas., High-level agent that wires together a provider, tools, MCP servers,     Redis m, Memory, Protocol, memory_for() (+37 more)

### Community 1 - "RawAdapter & Example Agents"
Cohesion: 0.07
Nodes (44): _chunk_to_output(), _history_to_dicts(), _is_str_param(), Any, .. deprecated::         RawAdapter is deprecated. Use :class:`agentinc.sdk.Agent, RawAdapter, _resolve_arg(), _schemas_to_dicts() (+36 more)

### Community 2 - "Audit Backends & Events"
Cohesion: 0.09
Nodes (37): AuditBackend, CallbackAuditBackend, ConsoleAuditBackend, FileAuditBackend, Protocol, audit_for(), Auditor, Any (+29 more)

### Community 3 - "API Reference & Schemas"
Cohesion: 0.06
Nodes (48): Agent class (API reference), AgentInput, AgentOutput, AgentProtocol (runtime_checkable), Audit event taxonomy, AuditConfig, create_app() signature, DataConfig (reserved RAG config) (+40 more)

### Community 4 - "Release History & Publishing"
Cohesion: 0.07
Nodes (46): Split build/publish jobs with artifact handoff, Publish to PyPI Workflow, OIDC Trusted Publishing (id-token: write), Tenant scoping invariant, Open-source boundary (sdk never imports platform internals), A2A session forwarding fix, Anthropic tool_use/tool_result conversion fix, Explicit provider/model-name format (+38 more)

### Community 5 - "A2A Serve Layer & Tool Examples"
Cohesion: 0.07
Nodes (33): _build_agent_input(), _collect_output(), create_app(), _jsonrpc_error(), _jsonrpc_result(), Any, Build a FastAPI app that serves an agent over the A2A protocol., Run an agent as an A2A server (blocking). (+25 more)

### Community 6 - "Tool Decorator & Agent Tests"
Cohesion: 0.16
Nodes (21): _wrap_tool(), add(), main(), multiply(), Demonstrates the @tool decorator and Agent with tools running locally (no HTTP s, _make_agent(), Tests for the Agent class — Tier 1 (no network, no LLM)., test_agent_context_optional() (+13 more)

### Community 7 - "Serve Session Forwarding Tests"
Cohesion: 0.32
Nodes (17): TestClient, client(), _params(), Tests for the A2A serve module — session/metadata/history forwarding., Stub agent that records the AgentInput it receives., RecordingAgent, _send(), test_explicit_metadata_session_id_wins() (+9 more)

### Community 8 - "Gemini Provider & Conversion"
Cohesion: 0.21
Nodes (10): GeminiProvider, Convert OpenAI-style messages to Gemini contents + system instruction.      Gemi, _to_gemini_messages(), Tests for the Gemini provider's OpenAI-format → Gemini contents conversion., test_assistant_text(), test_assistant_text_alongside_tool_calls(), test_assistant_tool_calls_become_function_call_parts(), test_empty_assistant_message_skipped() (+2 more)

### Community 9 - "Favicon Icon Set"
Cohesion: 0.23
Nodes (13): Puzzle Piece Mark (Android Chrome 192x192), Android Chrome Home-Screen Icon (192px), Agentinc SDK Favicon Icon Set, Puzzle Piece Mark (Android Chrome 512x512), Android PWA Splash / Maskable Icon (512px), Puzzle Piece Mark (Apple Touch Icon), iOS Safari Home-Screen Icon (180px), Puzzle Piece Mark (Favicon 16x16) (+5 more)

### Community 10 - "Brand Logo System"
Cohesion: 0.29
Nodes (12): Chevron Apex Stroke (outer A silhouette), Inner Crossbar Stroke (short A leg), Near-Black Ink Color (#231f20), Agentinc "A" Monogram Logo Mark, Red Accent Sliver (#ed1c24), Black Fill (#000000) for Light Backgrounds, Agentinc Brand Logo System (light/dark variants), Agentinc Pinwheel Glyph (black variant) (+4 more)

### Community 11 - "LightRAG Retrieval Agent"
Cohesion: 0.36
Nodes (7): _build_rag(), get_rag(), index(), RAGAgent, RAG-powered agent using LightRAG for knowledge retrieval.  Native RAG support vi, Index a text document into the RAG storage., LightRAG

### Community 12 - "Anthropic Provider"
Cohesion: 0.39
Nodes (3): AnthropicProvider, _convert_messages(), Convert OpenAI-format messages to Anthropic format.      - Strips system message

### Community 13 - "CI Test Matrix & Extras"
Cohesion: 0.67
Nodes (4): CI Workflow (test matrix), Editable install with [dev,all] extras, Python 3.12/3.13 Test Matrix, Optional-dependency extras (openai/anthropic/gemini/memory/mcp/serve/all)

### Community 14 - "Vault Documentation Rules"
Cohesion: 0.67
Nodes (3): Code-wins-over-note precedence, Docs-with-code same-PR invariant, Vault-first / resolver-before-scan rule

### Community 15 - "Tool Protocol Surface"
Cohesion: 0.67
Nodes (3): @tool decorator, ToolProtocol, ToolWrapper

## Ambiguous Edges - Review These
- `Agentinc "A" Monogram Logo Mark` → `Agentinc Brand Logo System (light/dark variants)`  [AMBIGUOUS]
  docs/images/agentinc/logo.svg · relation: conceptually_related_to
- `Agentinc "A" Monogram Logo Mark` → `Agentinc Pinwheel Glyph (black variant)`  [AMBIGUOUS]
  docs/images/agentinc/logo.svg · relation: semantically_similar_to
- `Chevron Apex Stroke (outer A silhouette)` → `Red Accent Sliver (#ed1c24)`  [AMBIGUOUS]
  docs/images/agentinc/logo.svg · relation: conceptually_related_to
- `Puzzle Piece Metaphor (Pluggable Agent / Interoperability)` → `Agentinc SDK Favicon Icon Set`  [AMBIGUOUS]
  docs/images/favicon/favicon-32x32.png · relation: rationale_for

## Knowledge Gaps
- **26 isolated node(s):** `Open-source boundary (sdk never imports platform internals)`, `Release 0.3.1`, `Keep a Changelog + SemVer convention`, `agentinc-sdk coding-agent skill`, `ToolProtocol` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Agentinc "A" Monogram Logo Mark` and `Agentinc Brand Logo System (light/dark variants)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Agentinc "A" Monogram Logo Mark` and `Agentinc Pinwheel Glyph (black variant)`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **What is the exact relationship between `Chevron Apex Stroke (outer A silhouette)` and `Red Accent Sliver (#ed1c24)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Puzzle Piece Metaphor (Pluggable Agent / Interoperability)` and `Agentinc SDK Favicon Icon Set`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **Why does `AgentOutput` connect `RawAdapter & Example Agents` to `Agent Core, Memory & MCP`, `Audit Backends & Events`, `A2A Serve Layer & Tool Examples`, `Tool Decorator & Agent Tests`, `Serve Session Forwarding Tests`, `Gemini Provider & Conversion`, `LightRAG Retrieval Agent`, `Anthropic Provider`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `AgentInput` connect `RawAdapter & Example Agents` to `Agent Core, Memory & MCP`, `Audit Backends & Events`, `A2A Serve Layer & Tool Examples`, `Tool Decorator & Agent Tests`, `Serve Session Forwarding Tests`, `LightRAG Retrieval Agent`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Why does `ToolCall` connect `Agent Core, Memory & MCP` to `RawAdapter & Example Agents`, `Audit Backends & Events`, `Tool Decorator & Agent Tests`, `Gemini Provider & Conversion`, `Anthropic Provider`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._