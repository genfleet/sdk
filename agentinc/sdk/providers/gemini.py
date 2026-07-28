from __future__ import annotations

import base64
import json
import logging
from typing import AsyncIterator

from ..schemas import AgentOutput, ModelConfig, TokenUsage, ToolCall, ToolSchema

log = logging.getLogger("agentinc.sdk.providers.gemini")


def _decode_signature(value: object) -> bytes | None:
    """Decode a base64 thought_signature back to the bytes Gemini expects.

    Tolerates None/absent (pre-signature history rows) and malformed values —
    a live conversation must not start erroring, and a bad signature is better
    dropped than raised.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (TypeError, ValueError):
        log.warning("discarding malformed thought_signature")
        return None


def _tool_calls_from_response(response) -> list[ToolCall]:
    """Collect function calls from a response/chunk, keeping thought signatures.

    Walks candidates -> content -> parts rather than using the flattened
    ``.function_calls`` accessor: that accessor yields bare FunctionCall
    objects and discards the enclosing Part, which is where Gemini 3.x puts
    ``thought_signature``.
    """
    tool_calls: list[ToolCall] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            fc = getattr(part, "function_call", None)
            if fc is None or not fc.name:
                continue
            raw_signature = getattr(part, "thought_signature", None)
            tool_calls.append(ToolCall(
                id=fc.id or fc.name,
                name=fc.name,
                arguments=dict(fc.args or {}),
                thought_signature=(
                    base64.b64encode(raw_signature).decode("ascii")
                    if raw_signature else None
                ),
            ))
    return tool_calls


def _to_gemini_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-style messages to Gemini contents + system instruction.

    Gemini requires every function_response part to follow a model turn with
    the matching function_call part, so assistant tool_calls are converted to
    function_call parts and tool names are resolved via tool_call_id.

    Gemini 3.x additionally requires each replayed function_call to carry the
    thought_signature it was issued with. The signature belongs on the Part,
    not on the function_call itself. Histories written before signatures were
    captured have none, and must still convert without raising.
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
                part: dict = {"function_call": {"name": fn["name"], "args": args}}
                signature = _decode_signature(tc.get("thought_signature"))
                if signature:
                    part["thought_signature"] = signature
                parts.append(part)
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
            tool_calls = _tool_calls_from_response(chunk)
            if tool_calls:
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

        tool_calls = _tool_calls_from_response(response)
        if tool_calls:
            yield AgentOutput(tool_calls=tool_calls, done=False, token_usage=usage)
        else:
            yield AgentOutput(content=response.text or "", done=True, token_usage=usage)
