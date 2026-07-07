from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from ..schemas import AgentOutput, ModelConfig, TokenUsage, ToolCall, ToolSchema

log = logging.getLogger("agentinc.sdk.providers.gemini")


def _to_gemini_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-style messages to Gemini contents + system instruction.

    Gemini requires every function_response part to follow a model turn with
    the matching function_call part, so assistant tool_calls are converted to
    function_call parts and tool names are resolved via tool_call_id.
    """
    system = ""
    contents = []
    tool_names: dict[str, str] = {}  # tool_call_id -> function name
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if role == "system":
            system = content
        elif role == "assistant":
            parts: list[dict] = []
            if content:
                parts.append({"text": content})
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                tool_names[tc["id"]] = fn["name"]
                try:
                    args = json.loads(fn["arguments"]) if fn.get("arguments") else {}
                except (TypeError, ValueError):
                    args = {}
                parts.append({"function_call": {"name": fn["name"], "args": args}})
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            name = tool_names.get(m.get("tool_call_id") or "") or m.get("name", "")
            contents.append({
                "role": "user",
                "parts": [{"function_response": {
                    "name": name,
                    "response": {"result": content},
                }}],
            })
        else:
            contents.append({"role": "user", "parts": [{"text": content}]})
    return system, contents


class GeminiProvider:
    def __init__(self, config: ModelConfig) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
            self._genai = genai
            self._types = genai_types
        except ImportError:
            raise ImportError(
                "Gemini provider requires the gemini extra: "
                "pip install 'agentinc-sdk[gemini]'"
            )
        self._client = genai.Client(api_key=config["api_key"])
        self._model = config["model"]

    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        stream: bool = True,
    ) -> AsyncIterator[AgentOutput]:
        system, contents = _to_gemini_messages(messages)

        gemini_tools = None
        if tools:
            function_declarations = [
                self._types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                )
                for t in tools
            ]
            gemini_tools = [self._types.Tool(function_declarations=function_declarations)]

        config_kwargs: dict = {}
        if system:
            config_kwargs["system_instruction"] = system
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

        gen_config = self._types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        if stream:
            async for chunk in self._stream(contents, gen_config):
                yield chunk
        else:
            async for chunk in self._blocking(contents, gen_config):
                yield chunk

    @staticmethod
    def _extract_usage(response) -> TokenUsage | None:
        meta = getattr(response, "usage_metadata", None)
        if meta:
            inp = getattr(meta, "prompt_token_count", 0) or 0
            out = getattr(meta, "candidates_token_count", 0) or 0
            total = getattr(meta, "total_token_count", 0) or (inp + out)
            return TokenUsage(input_tokens=inp, output_tokens=out, total_tokens=total)
        return None

    async def _stream(self, contents, gen_config) -> AsyncIterator[AgentOutput]:
        kwargs = {"model": self._model, "contents": contents}
        if gen_config:
            kwargs["config"] = gen_config

        last_chunk = None
        async for chunk in await self._client.aio.models.generate_content_stream(**kwargs):
            last_chunk = chunk
            if chunk.function_calls:
                tool_calls = [
                    ToolCall(id=fc.id or fc.name, name=fc.name, arguments=dict(fc.args or {}))
                    for fc in chunk.function_calls
                ]
                usage = self._extract_usage(chunk)
                yield AgentOutput(tool_calls=tool_calls, done=False, token_usage=usage)
                return
            if chunk.text:
                yield AgentOutput(content=chunk.text, done=False)

        usage = self._extract_usage(last_chunk) if last_chunk else None
        yield AgentOutput(content="", done=True, token_usage=usage)

    async def _blocking(self, contents, gen_config) -> AsyncIterator[AgentOutput]:
        kwargs = {"model": self._model, "contents": contents}
        if gen_config:
            kwargs["config"] = gen_config

        response = await self._client.aio.models.generate_content(**kwargs)
        usage = self._extract_usage(response)

        if response.function_calls:
            tool_calls = [
                ToolCall(id=fc.id or fc.name, name=fc.name, arguments=dict(fc.args or {}))
                for fc in response.function_calls
            ]
            yield AgentOutput(tool_calls=tool_calls, done=False, token_usage=usage)
        else:
            yield AgentOutput(content=response.text or "", done=True, token_usage=usage)
