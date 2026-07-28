from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from ..schemas import AgentOutput, ModelConfig, TokenUsage, ToolCall, ToolSchema

log = logging.getLogger("agentinc.sdk.providers.openai")

# Keys carried in message history for other providers' benefit, which the
# OpenAI chat-completions API rejects as unknown fields. Messages are passed
# through verbatim, so they must be stripped here: a session started on Gemini
# and resumed on an OpenAI model replays Gemini-shaped history.
_FOREIGN_TOOL_CALL_KEYS = ("thought_signature",)


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Drop non-OpenAI keys from assistant tool_calls, copying only when needed."""
    cleaned: list[dict] = []
    for m in messages:
        tool_calls = m.get("tool_calls")
        if not tool_calls or not any(
            k in tc for tc in tool_calls for k in _FOREIGN_TOOL_CALL_KEYS
        ):
            cleaned.append(m)
            continue
        cleaned.append({
            **m,
            "tool_calls": [
                {k: v for k, v in tc.items() if k not in _FOREIGN_TOOL_CALL_KEYS}
                for tc in tool_calls
            ],
        })
    return cleaned


class OpenAIProvider:
    """Covers OpenAI and any OpenAI-compatible endpoint (DeepSeek, Groq, Ollama, …)."""

    def __init__(self, config: ModelConfig) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai extra: "
                "pip install 'agentinc-sdk[openai]'"
            )
        self._client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
        )
        self._model = config["model"]

    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        stream: bool = True,
    ) -> AsyncIterator[AgentOutput]:
        openai_tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if tools
            else None
        )

        messages = _sanitize_messages(messages)

        if stream:
            async for chunk in self._stream(messages, openai_tools):
                yield chunk
        else:
            async for chunk in self._blocking(messages, openai_tools):
                yield chunk

    async def _stream(self, messages: list[dict], openai_tools) -> AsyncIterator[AgentOutput]:
        accumulated: dict[int, dict] = {}

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=openai_tools,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage: TokenUsage | None = None
        finish_reason: str | None = None
        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )

            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                yield AgentOutput(content=delta.content, done=False)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated:
                        accumulated[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        accumulated[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            accumulated[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            accumulated[idx]["arguments"] += tc.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if finish_reason == "tool_calls":
            tool_calls = [
                ToolCall(
                    id=v["id"],
                    name=v["name"],
                    arguments=json.loads(v["arguments"]) if v["arguments"] else {},
                )
                for v in accumulated.values()
            ]
            yield AgentOutput(tool_calls=tool_calls, done=False, token_usage=usage)
        else:
            yield AgentOutput(content="", done=True, token_usage=usage)

    async def _blocking(self, messages: list[dict], openai_tools) -> AsyncIterator[AgentOutput]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=openai_tools,
            stream=False,
        )
        choice = response.choices[0]
        msg = choice.message

        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]
            yield AgentOutput(tool_calls=tool_calls, done=False, token_usage=usage)
        else:
            yield AgentOutput(content=msg.content or "", done=True, token_usage=usage)
