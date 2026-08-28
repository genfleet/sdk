from __future__ import annotations

import logging
from typing import AsyncIterator

from ..schemas import AgentOutput, ModelConfig, TokenUsage, ToolCall, ToolSchema

log = logging.getLogger("fleet.sdk.providers.anthropic")

_SYSTEM_KEY = "__system__"


def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    - Strips system message (returned separately)
    - Converts assistant tool_calls → assistant with tool_use content blocks
    - Converts role:"tool" → role:"user" with tool_result content blocks
    """
    system = ""
    result: list[dict] = []

    i = 0
    while i < len(messages):
        m = messages[i]

        if m["role"] == "system":
            system = m.get("content", "")
            i += 1
            continue

        if m["role"] == "assistant" and "tool_calls" in m:
            content: list[dict] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn.get("name", ""),
                    "input": args,
                })
            result.append({"role": "assistant", "content": content})
            i += 1
            continue

        if m["role"] == "tool":
            tool_results: list[dict] = []
            while i < len(messages) and messages[i]["role"] == "tool":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": messages[i].get("tool_call_id", ""),
                    "content": messages[i].get("content", ""),
                })
                i += 1
            result.append({"role": "user", "content": tool_results})
            continue

        if m["role"] == "assistant":
            result.append({"role": "assistant", "content": m.get("content", "")})
        else:
            result.append({"role": "user", "content": m.get("content", "")})
        i += 1

    return system, result


class AnthropicProvider:
    def __init__(self, config: ModelConfig) -> None:
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the anthropic extra: "
                "pip install 'withfleet-sdk[anthropic]'"
            )
        self._client = self._anthropic.AsyncAnthropic(api_key=config["api_key"])
        self._model = config["model"]

    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        stream: bool = True,
    ) -> AsyncIterator[AgentOutput]:
        system, msgs = _convert_messages(messages)

        anthropic_tools = (
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
            if tools
            else []
        )

        kwargs: dict = dict(
            model=self._model,
            max_tokens=4096,
            messages=msgs,
        )
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if stream:
            async for chunk in self._stream(kwargs):
                yield chunk
        else:
            async for chunk in self._blocking(kwargs):
                yield chunk

    @staticmethod
    def _extract_usage(msg) -> TokenUsage | None:
        if hasattr(msg, "usage") and msg.usage:
            inp = getattr(msg.usage, "input_tokens", 0) or 0
            out = getattr(msg.usage, "output_tokens", 0) or 0
            return TokenUsage(input_tokens=inp, output_tokens=out, total_tokens=inp + out)
        return None

    async def _stream(self, kwargs: dict) -> AsyncIterator[AgentOutput]:
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield AgentOutput(content=text, done=False)

            msg = await stream.get_final_message()

        usage = self._extract_usage(msg)
        if msg.stop_reason == "tool_use":
            tool_calls = []
            for block in msg.content:
                if block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                    )
            yield AgentOutput(tool_calls=tool_calls, done=False, token_usage=usage)
        else:
            yield AgentOutput(content="", done=True, token_usage=usage)

    async def _blocking(self, kwargs: dict) -> AsyncIterator[AgentOutput]:
        msg = await self._client.messages.create(**kwargs)
        usage = self._extract_usage(msg)

        if msg.stop_reason == "tool_use":
            tool_calls = []
            for block in msg.content:
                if block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=block.input or {})
                    )
            yield AgentOutput(tool_calls=tool_calls, done=False, token_usage=usage)
        else:
            text = "".join(b.text for b in msg.content if hasattr(b, "text"))
            yield AgentOutput(content=text, done=True, token_usage=usage)
