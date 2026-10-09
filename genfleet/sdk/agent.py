from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any, AsyncIterator, Literal

from .audit import Auditor, audit_for
from .audit.schemas import AuditConfig
from .memory import memory_for

#: ``AgentInput.metadata`` key the platform sets per turn: ``"off"`` means the
#: Agent neither loads nor writes its session memory for that turn.
MEMORY_METADATA_KEY = "genfleet.memory"
from .memory.base import AppendableMemory, Memory
from . import turn as _turn
from .models import ModelClient, discover_model_client
from .providers import provider_for
from .schemas import (
    TOOL_EVENT_KEY,
    AgentInput,
    AgentOutput,
    MCPConfig,
    MemoryConfig,
    Message,
    ModelConfig,
    TokenUsage,
    ToolCall,
    ToolSchema,
)
from .caller import (
    Audience,
    Caller,
    caller_of,
    effective_audiences,
    marked,
    slug_key,
    may_offer,
    tool_audiences_of,
)
from .confirmations import (
    _PEER_REFUSAL,
    _PENDING_RESULT,
    CONFIRMATION_REQUESTS_KEY,
    _claim_approved_call,
    _is_peer_turn,
    _is_sensitive,
    _sensitive_flags_of,
)
from .tool import ToolWrapper, wrap_tool

log = logging.getLogger("genfleet.sdk.agent")


def _wrap_tool(fn: Callable) -> ToolWrapper:
    return wrap_tool(fn)


def _tool_call_to_dict(tc: ToolCall) -> dict:
    """Serialize a ToolCall into the OpenAI-style message-history shape.

    ``thought_signature`` is only emitted when set (Gemini 3.x), so OpenAI and
    Anthropic payloads are byte-for-byte unchanged. It rides alongside the
    ``function`` block rather than inside it so providers that pass the block
    straight through never see an unexpected key.
    """
    d: dict = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
    }
    if tc.thought_signature:
        d["thought_signature"] = tc.thought_signature
    return d


def _tool_event(payload: dict[str, Any]) -> AgentOutput:
    """An output reporting one tool event (see ``schemas.TOOL_EVENT_KEY``)."""
    return AgentOutput(content="", done=False, metadata={TOOL_EVENT_KEY: payload})


def _model_client_for(config: ModelConfig, explicit: ModelClient | None) -> ModelClient:
    """An explicit client, else an installed one that is active, else the direct provider."""
    if explicit is not None:
        return explicit
    found = discover_model_client(config)
    if found is not None:
        name, client = found
        log.info("model client %s active for %s", name, config["model"])
        return client
    return provider_for(config)


class Agent:
    """
    High-level agent that wires together a provider, tools, MCP servers,
    episodic memory, and optional context into a single AgentProtocol-compatible object.

    Usage::

        agent = Agent(
            role="You are a helpful assistant.",
            model={"model": "gpt-4o-mini", "api_key": "sk-…"},
            tools=[my_function],
            memory="platform",   # the platform's store when hosted; or {"type": "redis", ...}
        )
        serve(agent, name="my-agent", port=8000)

    ``memory`` may be left out: hosted on the platform the agent then gets
    the platform store anyway; on a laptop it runs without memory.
    """

    def __init__(
        self,
        role: str,
        model: ModelConfig,
        tools: list[Callable] | None = None,
        mcps: list[MCPConfig] | None = None,
        memory: MemoryConfig | Literal["platform"] | None = None,
        context: str | None = None,
        audit: AuditConfig | None = None,
        model_client: ModelClient | None = None,
    ) -> None:
        self._role = role
        self._context = context
        self._model_config = model

        self._provider: ModelClient = _model_client_for(model, model_client)
        self._memory: Memory | None = memory_for(memory)
        self._auditor: Auditor | None = audit_for(audit)
        self._mcp_configs: list[MCPConfig] = mcps or []

        # Wrap plain functions as ToolWrapper instances
        self._local_tools: dict[str, ToolWrapper] = {
            w.schema().name: w for w in (_wrap_tool(t) for t in (tools or []))
        }

        # MCP tool registry — populated on first run()
        self._mcp_tools: dict[str, Any] = {}
        self._mcp_ready = False

    @property
    def memory(self) -> Memory | None:
        """The episodic backend, for calls beyond the turn loop — a
        :class:`~genfleet.sdk.memory.PlatformMemory` exposes ``search`` and
        ``purge``. ``None`` when the agent runs without memory."""
        return self._memory

    # ------------------------------------------------------------------
    # AgentProtocol
    # ------------------------------------------------------------------

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]:
        invocation_id = str(uuid.uuid4())
        invocation_start = time.monotonic()
        auditor = self._auditor
        total_usage = TokenUsage()

        if self._mcp_configs and not self._mcp_ready:
            await self._init_mcp()

        session_id = input.metadata.get("session_id") or str(uuid.uuid4())
        # The platform turns memory off for a turn from its public A2A edge
        # (ADR-0027 §2c): the session is then neither loaded nor written.
        use_memory = self._memory is not None and input.metadata.get(MEMORY_METADATA_KEY) != "off"

        if auditor:
            await auditor.emit(
                "invocation.start",
                invocation_id=invocation_id,
                session_id=session_id,
                data={"message": auditor.truncate(input.message)},
            )

        try:
            # Load history from memory backend (empty list if no memory configured)
            persisted: list[Message] = []
            if use_memory:
                persisted = await self._memory.load(session_id)

            # Merge caller-supplied history (takes precedence) with persisted history
            history = persisted if not input.history else list(input.history)

            # Build system prompt
            system_parts = [self._role]
            if self._context:
                system_parts.append(self._context)

            messages: list[dict] = [{"role": "system", "content": "\n\n".join(system_parts)}]
            for msg in history:
                messages.append(self._message_to_dict(msg))
            messages.append({"role": "user", "content": input.message})

            # The tools this caller may use (ADR-0028): by each tool's
            # audience, the platform's per-tool setting first. A tool left out
            # here is also refused at dispatch, whatever the model asks for.
            caller = caller_of(input.metadata)
            offered = self._offered_tools(caller, tool_audiences_of(input.metadata))
            tool_schemas = [w.schema() for name, w in self._local_tools.items() if name in offered]
            tool_schemas += [
                ToolSchema(name=name, description=spec["description"], parameters=spec["parameters"])
                for name, spec in self._mcp_tools.items()
                if name in offered
            ]

            # ADR-0028 §8a: sensitive tools wait for an owner's approval; an
            # approved call (signed by the engine) runs first, exactly once.
            sensitive = self._sensitive_tools(_sensitive_flags_of(input.metadata))
            peer_turn = _is_peer_turn(input.metadata)
            requests: list[dict[str, Any]] = []
            new_messages: list[dict] = []
            # Verified and claimed in one step: it can't run twice, nor later.
            approved, refused = _claim_approved_call(input.metadata)
            if refused:
                log.warning("approved call not run: %s", refused)
                messages[0]["content"] += (
                    "\n\nAn approval arrived with this message but could not be verified, so nothing ran. "
                    "Tell the user the action did not run."
                )
            elif approved:
                name = self._tool_by_key(approved.tool)
                if name is None or name not in offered:
                    messages[0]["content"] += (
                        "\n\nAn approved action can't run for this caller now, so nothing ran. Tell the user."
                    )
                else:
                    tc = ToolCall(id=f"approved-{approved.confirmation_id}", name=name, arguments=approved.arguments)
                    yield _tool_event({"type": "tool_call", "id": tc.id, "name": tc.name, "arguments": tc.arguments})
                    try:
                        result, ok = await self._audited_dispatch(tc, offered, input, invocation_id, session_id)
                    except Exception as exc:  # noqa: BLE001 — the user must still hear the outcome
                        log.exception("approved call %s failed", approved.confirmation_id)
                        result, ok = f"Error: the approved action failed ({type(exc).__name__}).", False
                    new_messages.append({"role": "assistant", "content": "", "tool_calls": [_tool_call_to_dict(tc)]})
                    new_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    yield _tool_event({"type": "tool_result", "id": tc.id, "name": tc.name, "ok": ok, "output": result})
                    messages[0]["content"] += (
                        f"\n\nA workspace owner or admin approved `{name}`, and it has run (its result follows). "
                        "Tell the user the outcome."
                    )

            # Agentic loop
            while True:
                tool_calls_batch: list[ToolCall] = []

                if auditor:
                    await auditor.emit(
                        "llm.request",
                        invocation_id=invocation_id,
                        session_id=session_id,
                        data={
                            "model": self._model_config["model"],
                            "message_count": len(messages) + len(new_messages),
                            "tool_count": len(tool_schemas),
                        },
                    )

                llm_start = time.monotonic()
                last_token_usage: TokenUsage | None = None
                response_content_parts: list[str] = []

                # The model client reads this turn's metadata through
                # `current_turn()`, set only while it runs (never across a yield).
                async for chunk in _turn.within(
                    input.metadata, self._provider.complete(messages + new_messages, tool_schemas)
                ):
                    if chunk.token_usage:
                        last_token_usage = chunk.token_usage
                    if chunk.tool_calls:
                        tool_calls_batch.extend(chunk.tool_calls)
                    elif chunk.content:
                        response_content_parts.append(chunk.content)
                        # ``done`` describes the turn, not a provider chunk: the
                        # loop may still have a tool round to run, and a consumer
                        # stops reading at the first ``done`` it sees. Only the
                        # yield at the bottom of the loop ends the turn.
                        yield chunk.model_copy(update={"done": False}) if chunk.done else chunk
                    if chunk.done:
                        break

                llm_latency = (time.monotonic() - llm_start) * 1000

                if last_token_usage:
                    total_usage.input_tokens += last_token_usage.input_tokens
                    total_usage.output_tokens += last_token_usage.output_tokens
                    total_usage.total_tokens += last_token_usage.total_tokens

                if auditor:
                    response_text = "".join(response_content_parts)
                    await auditor.emit(
                        "llm.response",
                        invocation_id=invocation_id,
                        session_id=session_id,
                        data={
                            "content": auditor.truncate(response_text) if response_text else None,
                            "tool_calls": [
                                {"name": tc.name, "arguments": tc.arguments}
                                for tc in tool_calls_batch
                            ] if tool_calls_batch else None,
                            "token_usage": last_token_usage.model_dump() if last_token_usage else None,
                        },
                        latency_ms=llm_latency,
                    )

                if not tool_calls_batch:
                    final_text = "".join(response_content_parts)
                    if final_text:
                        new_messages.append({"role": "assistant", "content": final_text})
                    # Persist *before* announcing the turn is done. A consumer
                    # stops reading at ``done`` — ``serve`` breaks out of the
                    # ``async for`` — which closes this generator at the yield
                    # below, so anything after it never runs.
                    await self._persist_turn(
                        session_id, input, history, new_messages, invocation_id
                    )
                    # Same reason as the persist: after the ``done`` yield this
                    # generator may already be closed, and the audit trail
                    # would have a start with no end for every served turn.
                    if auditor:
                        await auditor.emit(
                            "invocation.end",
                            invocation_id=invocation_id,
                            session_id=session_id,
                            data={"total_token_usage": total_usage.model_dump()},
                            latency_ms=(time.monotonic() - invocation_start) * 1000,
                        )
                    # The requests ride on the final chunk: a non-streaming
                    # invoke (every channel turn) returns only its metadata.
                    yield AgentOutput(
                        content="",
                        done=True,
                        metadata={CONFIRMATION_REQUESTS_KEY: requests} if requests else {},
                    )
                    break

                # Append assistant tool-call turn (keep any text streamed
                # alongside the tool calls — the user already saw it)
                new_messages.append({
                    "role": "assistant",
                    "content": "".join(response_content_parts),
                    "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls_batch],
                })

                # Report each call once its input is complete, then run them.
                # Same order as the platform runner: every call of the round
                # first, then each result as its tool returns.
                for tc in tool_calls_batch:
                    yield _tool_event({
                        "type": "tool_call",
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    })

                # Dispatch all tool calls and append results
                for tc in tool_calls_batch:
                    status: str | None = None
                    if tc.name in offered and tc.name in sensitive:
                        # Not run: a peer can't be approved for; anyone else waits for an owner.
                        if peer_turn:
                            result, ok = _PEER_REFUSAL, False
                        else:
                            requests.append({"tool": self._key_of(tc.name), "arguments": tc.arguments})
                            result, ok, status = _PENDING_RESULT, False, "pending"
                    else:
                        result, ok = await self._audited_dispatch(tc, offered, input, invocation_id, session_id)

                    new_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    yield _tool_event({
                        "type": "tool_result",
                        "id": tc.id,
                        "name": tc.name,
                        "ok": ok,
                        "output": result,
                        **({"status": status} if status else {}),
                    })

        except Exception as exc:
            if auditor:
                await auditor.emit(
                    "invocation.error",
                    invocation_id=invocation_id,
                    session_id=session_id,
                    data={"error_type": type(exc).__name__, "message": str(exc)},
                    latency_ms=(time.monotonic() - invocation_start) * 1000,
                    status="error",
                )
            raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _persist_turn(
        self,
        session_id: str,
        input: AgentInput,
        history: list[Message],
        new_messages: list[dict],
        invocation_id: str,
    ) -> None:
        """Write this turn to memory — keep the full assistant/tool structure so
        replayed histories remain valid provider payloads (tool messages must
        follow an assistant turn carrying the matching tool_calls ids).
        """
        if not self._memory or input.metadata.get(MEMORY_METADATA_KEY) == "off":
            return
        turn = [Message(role="user", content=input.message)]
        for m in new_messages:
            if m["role"] == "assistant":
                turn.append(Message(
                    role="assistant",
                    content=m.get("content", "") or "",
                    tool_calls=[
                        ToolCall(
                            id=t["id"],
                            name=t["function"]["name"],
                            arguments=json.loads(t["function"]["arguments"] or "{}"),
                            thought_signature=t.get("thought_signature"),
                        )
                        for t in m.get("tool_calls", [])
                    ],
                ))
            elif m["role"] == "tool":
                turn.append(Message(
                    role="tool",
                    content=m.get("content", "") or "",
                    tool_call_id=m.get("tool_call_id"),
                ))
        if isinstance(self._memory, AppendableMemory):
            # One turn, not the whole thread: the store already holds
            # what ``load`` returned. ``subject`` and the session's
            # metadata ride on the input — a channel ingress sets them
            # (ADR-0019 §4); a plain caller may leave them out.
            await self._memory.append(
                session_id,
                turn,
                subject=_optional_str(input.metadata.get("subject")),
                metadata=_session_metadata(input.metadata),
                turn_id=_turn_id(input.metadata, invocation_id),
            )
        else:
            await self._memory.save(session_id, list(history) + turn)

    def _offered_tools(self, caller: Caller | None, overrides: dict[str, frozenset[Audience]]) -> set[str]:
        """Names of the tools offered on this caller's turn, by ``effective_audiences``."""
        names = {
            n for n, w in self._local_tools.items()
            if may_offer(caller, effective_audiences(n, w.slug, w.audiences, overrides))
        }
        names |= {
            n for n, spec in self._mcp_tools.items()
            # MCP: `customer_safe_tools` is the config's (legacy) marking; the
            # platform's setting names an MCP tool by name only.
            if may_offer(
                caller, effective_audiences(n, None, marked(n in spec["config"].get("customer_safe_tools", [])), overrides)
            )
        }
        return names

    def _key_of(self, name: str) -> str:
        """The platform's key for a tool: a manifest tool's slug key, else its name."""
        wrapper = self._local_tools.get(name)
        return slug_key(wrapper.slug) if wrapper is not None and wrapper.slug else name

    def _tool_by_key(self, key: str) -> str | None:
        """The tool a platform key names, by the same rule as audiences (§5)."""
        for name in [*self._local_tools, *self._mcp_tools]:
            if self._key_of(name) == key:
                return name
        return None

    def _sensitive_tools(self, flags: dict[str, bool]) -> set[str]:
        """Names of the sensitive tools: the author's marking, plus the platform's ``true`` (add-only)."""
        names = {n for n, w in self._local_tools.items() if _is_sensitive(self._key_of(n), w.sensitive, flags)}
        names |= {
            n for n, spec in self._mcp_tools.items()
            if _is_sensitive(n, n in spec["config"].get("sensitive_tools", []), flags)
        }
        return names

    async def _audited_dispatch(
        self, tc: ToolCall, offered: set[str], input: AgentInput, invocation_id: str, session_id: str
    ) -> tuple[str, bool]:
        """One tool call, inside the turn, with its ``tool.call`` / ``tool.result`` audit events."""
        auditor = self._auditor
        if auditor:
            await auditor.emit(
                "tool.call",
                invocation_id=invocation_id,
                session_id=session_id,
                data={"tool_name": tc.name, "arguments": auditor.truncate(str(tc.arguments))},
            )
        tool_start = time.monotonic()
        result, ok = await _turn.call_within(input.metadata, self._dispatch_tool(tc, offered))
        if auditor:
            await auditor.emit(
                "tool.result",
                invocation_id=invocation_id,
                session_id=session_id,
                data={"tool_name": tc.name, "result": auditor.truncate(result)},
                latency_ms=(time.monotonic() - tool_start) * 1000,
            )
        return result, ok

    async def _dispatch_tool(self, tc: ToolCall, offered: set[str] | None = None) -> tuple[str, bool]:
        """The tool's result for the model, and whether the tool ran.

        ``ok`` is false when the tool is unknown, not offered on this turn, or
        an MCP call failed; the result is then the error text the model sees.
        A local tool that raises still ends the run, as it always has.
        """
        if offered is not None and tc.name not in offered:
            # Same answer as an unknown tool: a customer's turn learns nothing
            # about the operator-only tools it cannot see.
            return f"Error: tool '{tc.name}' not found", False
        if tc.name in self._local_tools:
            return await self._local_tools[tc.name].call(tc), True
        if tc.name in self._mcp_tools:
            return await self._call_mcp_tool(tc)
        return f"Error: tool '{tc.name}' not found", False

    async def _init_mcp(self) -> None:
        for config in self._mcp_configs:
            try:
                tools = await _fetch_mcp_tools(config)
                # An MCP server names its own tools. One that reuses a local
                # tool's name is dropped: dispatch goes by name, so it could
                # otherwise lend its customer-safe mark to an operator-only tool.
                for name in [n for n in tools if n in self._local_tools or n in self._mcp_tools]:
                    log.warning("MCP tool %r skipped: the name is already taken by another tool", name)
                    del tools[name]
                # A name starting with `@` is a slug key's form (ADR-0028 §5):
                # an MCP tool can't take one and borrow a manifest tool's settings.
                for name in [n for n in tools if n.startswith("@")]:
                    log.warning("MCP tool %r skipped: a tool name can't start with '@'", name)
                    del tools[name]
                self._mcp_tools.update(tools)
            except Exception:
                log.exception("Failed to initialise MCP server: %s", config)
        self._mcp_ready = True

    async def _call_mcp_tool(self, tc: ToolCall) -> tuple[str, bool]:
        spec = self._mcp_tools[tc.name]
        try:
            return await _invoke_mcp_tool(spec["config"], tc.name, tc.arguments)
        except Exception as exc:
            return f"Error calling MCP tool '{tc.name}': {exc}", False

    @staticmethod
    def _message_to_dict(msg: Message) -> dict:
        d: dict = {"role": msg.role, "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [_tool_call_to_dict(tc) for tc in msg.tool_calls]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        return d


# ------------------------------------------------------------------
# MCP helpers (lazy-import mcp package)
# ------------------------------------------------------------------

async def _fetch_mcp_tools(config: MCPConfig) -> dict[str, dict]:
    """Connect to an MCP server and return its tool schemas."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise ImportError(
            "MCP support requires the mcp extra: pip install 'genfleet-sdk[mcp]'"
        )

    tools: dict[str, dict] = {}

    if config["type"] == "stdio":
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for tool in result.tools:
                    tools[tool.name] = {
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        "config": config,
                    }

    elif config["type"] == "sse":
        async with sse_client(config["url"]) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for tool in result.tools:
                    tools[tool.name] = {
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        "config": config,
                    }

    return tools


async def _invoke_mcp_tool(config: MCPConfig, name: str, arguments: dict) -> tuple[str, bool]:
    """The MCP tool's result text, and whether the call succeeded.

    A tool that fails on the server's side does not raise: per the MCP spec
    ``call_tool`` returns a result with ``isError`` set, and that is a failure.
    """
    if config["type"] not in ("stdio", "sse"):
        return "Error: unsupported MCP transport", False

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    if config["type"] == "stdio":
        transport = stdio_client(StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
        ))
    else:
        transport = sse_client(config["url"])
    async with transport as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(name, arguments)
        return str(result.content), not result.isError


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


#: Input-metadata keys that describe the *session* rather than the turn, and
#: are worth keeping with it. Everything else in ``metadata`` is per request.
#: Deliberately one key: ``channel`` says which ingress a thread arrived on
#: (ADR-0019 §4) and identifies nobody. The party is already named by
#: ``subject``, an opaque id; ``contact`` — a phone number or an address — is
#: customer PII and is not copied into the store until a spec asks for it.
_SESSION_METADATA_KEYS = ("channel",)

#: Input-metadata keys, in order of preference, that an ingress uses to name a
#: turn. A retry that replays the same id lets an idempotent backend drop the
#: duplicate; see :class:`~genfleet.sdk.memory.base.AppendableMemory`.
_TURN_ID_KEYS = ("turn_id", "request_id", "message_id")


def _session_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    kept = {k: metadata[k] for k in _SESSION_METADATA_KEYS if metadata.get(k) is not None}
    return kept or None


def _turn_id(metadata: dict[str, Any], invocation_id: str) -> str:
    """The idempotency key for this turn.

    A caller's own id is what makes a retry recognisable — the invocation id
    is fresh on every attempt, so the fallback only dedupes a redelivery of
    the same in-flight call.
    """
    for key in _TURN_ID_KEYS:
        value = _optional_str(metadata.get(key))
        if value:
            return value
    return invocation_id
