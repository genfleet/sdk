from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from ..schemas import AgentOutput, ToolSchema


@runtime_checkable
class Provider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        tools: list[ToolSchema],
        stream: bool = True,
    ) -> AsyncIterator[AgentOutput]: ...


def split_data_url(url: str) -> tuple[str, str]:
    """``data:<mime>;base64,<data>`` as ``(mime, base64 data)``.

    The `Agent` sends images to a model as data URLs (OpenAI's ``image_url``
    form); providers with their own image block convert them with this.
    """
    header, sep, data = url.partition(",")
    if not sep or not header.startswith("data:") or not header.endswith(";base64"):
        raise ValueError("expected a base64 data URL")
    return header[len("data:"):-len(";base64")], data
