from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any, AsyncIterator

from .audit import Auditor, audit_for
from .audit.schemas import AuditConfig
from .memory import memory_for
from .memory.base import Memory
from .providers import provider_for
from .providers.base import Provider
from .schemas import (
    AgentInput,
    AgentOutput,
    DataConfig,
    MCPConfig,
    MemoryConfig,
    Message,
    ModelConfig,
    TokenUsage,
    ToolCall,
    ToolSchema,
)
from .tool import ToolWrapper, _build_schema

log = logging.getLogger("agentinc.sdk.agent")


def _wrap_tool(fn: Callable) -> ToolWrapper:
    if isinstance(fn, ToolWrapper):
        return fn
    schema = _build_schema(fn, name=fn.__name__, description=fn.__doc__ or "")
    return ToolWrapper(fn=fn, schema=schema)


class Agent:
    """
    High-level agent that wires together a provider, tools, MCP servers,
    Redis memory, and optional context into a single AgentProtocol-compatible object.

    Usage::

        agent = Agent(
            role="You are a helpful assistant.",
            model={"model": "gpt-4o-mini", "api_key": "sk-…"},
            tools=[my_function],
            memory={"type": "redis", "connection": "redis://localhost:6379"},
        )
        serve(agent, name="my-agent", port=8000)
    """

    def __init__(
        self,
        role: str,
        model: ModelConfig,
        tools: list[Callable] | None = None,
        mcps: list[MCPConfig] | None = None,
        memory: MemoryConfig | None = None,
        context: str | None = None,
        data: DataConfig | None = None,
        audit: AuditConfig | None = None,
    ) -> None:
        self._role = role
        self._context = context
        self._model_config = model

        self._provider: Provider = provider_for(model)
        self._memory: Memory | None = memory_for(memory) if memory else None
        self._auditor: Auditor | None = audit_for(audit)
        self._mcp_configs: list[MCPConfig] = mcps or []

        # Wrap plain functions as ToolWrapper instances
        self._local_tools: dict[str, ToolWrapper] = {
            w.schema().name: w for w in (_wrap_tool(t) for t in (tools or []))
        }

        # MCP tool registry — populated on first run()
        self._mcp_tools: dict[str, Any] = {}
        self._mcp_ready = False

        if data is not None:
            log.debug("data= parameter accepted but not yet implemented (reserved for RAG)")

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
            if self._memory:
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

            # Collect all tool schemas
            tool_schemas = [w.schema() for w in self._local_tools.values()]
            tool_schemas += [
                ToolSchema(name=name, description=spec["description"], parameters=spec["parameters"])
                for name, spec in self._mcp_tools.items()
            ]

            # Agentic loop
            new_messages: list[dict] = []
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

                async for chunk in self._provider.complete(messages + new_messages, tool_schemas):
                    if chunk.token_usage:
                        last_token_usage = chunk.token_usage
                    if chunk.tool_calls:
                        tool_calls_batch.extend(chunk.tool_calls)
                    elif chunk.content:
                        response_content_parts.append(chunk.content)
                        yield chunk
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
                    yield AgentOutput(content="", done=True)
                    break

                # Append assistant tool-call turn (keep any text streamed
                # alongside the tool calls — the user already saw it)
                new_messages.append({
                    "role": "assistant",
                    "content": "".join(response_content_parts),
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in tool_calls_batch
                    ],
                })

                # Dispatch all tool calls and append results
                for tc in tool_calls_batch:
                    if auditor:
                        await auditor.emit(
                            "tool.call",
                            invocation_id=invocation_id,
                            session_id=session_id,
                            data={
                                "tool_name": tc.name,
                                "arguments": auditor.truncate(str(tc.arguments)),
                            },
                        )

                    tool_start = time.monotonic()
                    result = await self._dispatch_tool(tc)
                    tool_latency = (time.monotonic() - tool_start) * 1000

                    if auditor:
                        await auditor.emit(
                            "tool.result",
                            invocation_id=invocation_id,
                            session_id=session_id,
                            data={
                                "tool_name": tc.name,
                                "result": auditor.truncate(result),
                            },
                            latency_ms=tool_latency,
                        )

                    new_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            # Persist updated history — keep the full assistant/tool structure so
            # replayed histories remain valid provider payloads (tool messages must
            # follow an assistant turn carrying the matching tool_calls ids).
            if self._memory:
                updated = list(history) + [Message(role="user", content=input.message)]
                for m in new_messages:
                    if m["role"] == "assistant":
                        updated.append(Message(
                            role="assistant",
                            content=m.get("content", "") or "",
                            tool_calls=[
                                ToolCall(
                                    id=t["id"],
                                    name=t["function"]["name"],
                                    arguments=json.loads(t["function"]["arguments"] or "{}"),
                                )
                                for t in m.get("tool_calls", [])
                            ],
                        ))
                    elif m["role"] == "tool":
                        updated.append(Message(
                            role="tool",
                            content=m.get("content", "") or "",
                            tool_call_id=m.get("tool_call_id"),
                        ))
                await self._memory.save(session_id, updated)

            if auditor:
                await auditor.emit(
                    "invocation.end",
                    invocation_id=invocation_id,
                    session_id=session_id,
                    data={"total_token_usage": total_usage.model_dump()},
                    latency_ms=(time.monotonic() - invocation_start) * 1000,
                )

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

    async def _dispatch_tool(self, tc: ToolCall) -> str:
        if tc.name in self._local_tools:
            return await self._local_tools[tc.name].call(tc)
        if tc.name in self._mcp_tools:
            return await self._call_mcp_tool(tc)
        return f"Error: tool '{tc.name}' not found"

    async def _init_mcp(self) -> None:
        for config in self._mcp_configs:
            try:
                tools = await _fetch_mcp_tools(config)
                self._mcp_tools.update(tools)
            except Exception:
                log.exception("Failed to initialise MCP server: %s", config)
        self._mcp_ready = True

    async def _call_mcp_tool(self, tc: ToolCall) -> str:
        spec = self._mcp_tools.get(tc.name)
        if spec is None:
            return f"Error: MCP tool '{tc.name}' not found"
        try:
            result = await _invoke_mcp_tool(spec["config"], tc.name, tc.arguments)
            return result
        except Exception as exc:
            return f"Error calling MCP tool '{tc.name}': {exc}"

    @staticmethod
    def _message_to_dict(msg: Message) -> dict:
        d: dict = {"role": msg.role, "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]
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
        from mcp.client.stdio import stdio_client
        from mcp.client.sse import sse_client
    except ImportError:
        raise ImportError(
            "MCP support requires the mcp extra: pip install 'agentinc-sdk[mcp]'"
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


async def _invoke_mcp_tool(config: MCPConfig, name: str, arguments: dict) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client

    if config["type"] == "stdio":
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return str(result.content)

    elif config["type"] == "sse":
        async with sse_client(config["url"]) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return str(result.content)

    return "Error: unsupported MCP transport"
