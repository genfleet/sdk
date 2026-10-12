"""``Agent`` with files: images to vision models, a descriptor for everything else."""

from __future__ import annotations

import base64

import pytest

from genfleet.sdk import Agent, AgentInput, InMemoryFileResolver, TextPart, UnsupportedInput, part_for
from genfleet.sdk.models import FakeModelClient
from genfleet.sdk.providers.anthropic import _convert_messages
from genfleet.sdk.providers.gemini import _to_gemini_messages

PNG = b"\x89PNG\r\n\x1a\nfake"
JPEG = b"\xff\xd8\xff\xe0fake"


def _agent(
    *, vision: bool = False, accepts: list[str] | None = None, files=None, memory=None, **limits
) -> tuple[Agent, FakeModelClient]:
    client = FakeModelClient(["ok"])
    model = {"model": "general-fast", "vision": vision} if vision else {"model": "general-fast"}
    agent = Agent(role="r", model=model, model_client=client, accepts=accepts or ["*/*"], files=files, **limits)
    agent._memory = memory
    return agent, client


async def _run(agent: Agent, turn: AgentInput) -> str:
    return "".join([o.content or "" async for o in agent.run(turn)])


async def test_a_text_turn_is_sent_as_before() -> None:
    agent, client = _agent(vision=True)
    await _run(agent, AgentInput(message="hi"))
    assert client.calls[0][-1] == {"role": "user", "content": "hi"}


async def test_a_vision_model_gets_the_image_and_a_line_naming_it() -> None:
    agent, client = _agent(vision=True)
    await _run(agent, AgentInput(message="what is it?", parts=[part_for("image/png", data=PNG, name="cat.png")]))
    content = client.calls[0][-1]["content"]
    assert content == [
        {"type": "text", "text": 'what is it?\n[Attached image: name="cat.png", type=image/png, size=12 bytes]'},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}},
    ]


async def test_an_image_by_reference_is_read_through_the_resolver() -> None:
    files = InMemoryFileResolver()
    agent, client = _agent(vision=True, files=files)
    image = files.put(JPEG, mime_type="image/jpeg", session_id="s1")
    await _run(agent, AgentInput(parts=[image], metadata={"session_id": "s1"}))
    content = client.calls[0][-1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_without_vision_an_image_is_only_described() -> None:
    agent, client = _agent(vision=False)
    await _run(agent, AgentInput(parts=[part_for("image/png", data=PNG)]))
    assert client.calls[0][-1]["content"] == "[Attached image: type=image/png, size=12 bytes]"


async def test_a_file_only_turn_describes_the_file_and_never_sends_its_bytes() -> None:
    agent, client = _agent(vision=True)
    reply = await _run(agent, AgentInput(parts=[part_for("application/pdf", data=b"%PDF-SECRET", name="q.pdf")]))
    assert reply == "ok"
    content = client.calls[0][-1]["content"]
    assert content == '[Attached file: name="q.pdf", type=application/pdf, size=11 bytes]'
    assert "SECRET" not in str(client.calls)


@pytest.mark.parametrize(("setup", "reason"), [
    ("expired", "expired"),
    ("other-session", "not found or not accessible"),
    ("no-resolver", "not readable here"),
    ("too-large", "too large"),
])
async def test_an_unreadable_image_is_described_with_the_reason(setup: str, reason: str) -> None:
    now = [0.0]
    files = InMemoryFileResolver(clock=lambda: now[0])
    image = files.put(PNG, mime_type="image/png", ttl=10, session_id="s1")
    if setup == "expired":
        now[0] = 11
    if setup == "too-large":
        image = part_for("image/png", uri=image.uri, size=50 * 1024 * 1024)
    agent, client = _agent(vision=True, files=None if setup == "no-resolver" else files)
    session = "s2" if setup == "other-session" else "s1"
    await _run(agent, AgentInput(message="see", parts=[image], metadata={"session_id": session}))
    assert client.calls[0][-1]["content"] == f"see\n[Attached image: type=image/png, size={image.size} bytes] (unavailable: {reason})"


async def test_an_unaccepted_type_is_refused_before_the_model_is_called() -> None:
    agent, client = _agent(accepts=["image/*"])
    with pytest.raises(UnsupportedInput, match="video/mp4"):
        await _run(agent, AgentInput(parts=[part_for("video/mp4", uri="mem://v")]))
    assert client.calls == []
    default = Agent(role="r", model={"model": "general-fast"}, model_client=FakeModelClient(["ok"]))
    assert default.accepted_input_types == ()
    with pytest.raises(UnsupportedInput):
        await _run(default, AgentInput(parts=[part_for("image/png", data=PNG)]))


async def test_memory_keeps_the_description_not_the_bytes() -> None:
    stored: dict[str, list] = {}

    class FakeMemory:
        async def load(self, session_id):
            return stored.get(session_id, [])

        async def save(self, session_id, history):
            stored[session_id] = history

        async def clear(self, session_id):
            pass

    agent, _ = _agent(vision=True, memory=FakeMemory())
    await _run(agent, AgentInput(message="hi", parts=[part_for("image/png", data=PNG)], metadata={"session_id": "s"}))
    user = stored["s"][0]
    assert user.content == "hi\n[Attached image: type=image/png, size=12 bytes]"


def test_anthropic_gets_an_image_block() -> None:
    url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    _, messages = _convert_messages([{"role": "user", "content": [
        {"type": "text", "text": "t"}, {"type": "image_url", "image_url": {"url": url}},
    ]}])
    assert messages == [{"role": "user", "content": [
        {"type": "text", "text": "t"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}},
    ]}]


def test_gemini_gets_inline_data() -> None:
    url = "data:image/webp;base64," + base64.b64encode(PNG).decode()
    _, contents = _to_gemini_messages([{"role": "user", "content": [
        {"type": "text", "text": "t"}, {"type": "image_url", "image_url": {"url": url}},
    ]}])
    assert contents == [{"role": "user", "parts": [{"text": "t"}, {"inline_data": {"mime_type": "image/webp", "data": PNG}}]}]


def test_text_parts_alone_need_no_declared_types() -> None:
    turn = AgentInput(parts=[TextPart(text="a"), TextPart(text="b")])
    assert turn.message == "ab"


def test_a_served_agent_takes_a_file_only_request() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from genfleet.sdk.serve import create_app

    agent, client = _agent(accepts=["application/pdf"])
    app = TestClient(create_app(agent))
    assert app.get("/.well-known/agent-card.json").json()["defaultInputModes"] == ["text/plain", "application/pdf"]
    body = app.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": {
        "messageId": "m", "role": "ROLE_USER", "parts": [{"url": "genfleet-file://f/1", "mediaType": "application/pdf", "filename": "r.pdf"}],
    }}}).json()
    assert body["result"]["task"]["artifacts"][0]["parts"] == [{"text": "ok"}]
    assert client.calls[0][-1]["content"] == '[Attached file: name="r.pdf", type=application/pdf]'
    refused = app.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": {
        "messageId": "m", "role": "ROLE_USER", "parts": [{"url": "genfleet-file://f/1", "mediaType": "image/png"}],
    }}}).json()
    assert refused["error"]["code"] == -32005


class CountingResolver(InMemoryFileResolver):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def read(self, uri, *, max_bytes, metadata):
        self.reads += 1
        return await super().read(uri, max_bytes=max_bytes, metadata=metadata)


async def test_a_repeated_reference_is_read_and_sent_once() -> None:
    files = CountingResolver()
    agent, client = _agent(vision=True, files=files)
    image = files.put(PNG, mime_type="image/png")
    await _run(agent, AgentInput(parts=[image] * 50))
    content = client.calls[0][-1]["content"]
    assert files.reads == 1
    assert len([b for b in content if b["type"] == "image_url"]) == 1
    assert content[0]["text"].count("(repeated; shown once)") == 49


async def test_repeated_inline_bytes_are_sent_once() -> None:
    agent, client = _agent(vision=True)
    await _run(agent, AgentInput(parts=[part_for("image/png", data=PNG), part_for("image/png", data=PNG)]))
    assert len(client.calls[0][-1]["content"]) == 2  # the text block and one image


async def test_images_past_the_count_limit_are_described() -> None:
    files = CountingResolver()
    agent, client = _agent(vision=True, files=files, max_images_per_turn=2)
    parts = [files.put(PNG + bytes([i]), mime_type="image/png") for i in range(5)]
    await _run(agent, AgentInput(parts=parts))
    content = client.calls[0][-1]["content"]
    assert files.reads == 2
    assert len(content) == 3
    assert content[0]["text"].count("(not shown: image limit for this turn reached)") == 3


async def test_images_past_the_byte_budget_are_described_not_loaded() -> None:
    files = CountingResolver()
    agent, client = _agent(vision=True, files=files, max_image_bytes_per_turn=36)
    first = files.put(PNG + b"x" * 10, mime_type="image/png")  # 22 bytes
    second = files.put(PNG + b"y" * 10, mime_type="image/png")  # 22 more: over the 14 left
    third = files.put(PNG, mime_type="image/png", name="small.png")  # 12: fits what's left
    await _run(agent, AgentInput(parts=[first, second, third]))
    content = client.calls[0][-1]["content"]
    assert len(content) == 3
    assert "size=22 bytes] (unavailable: too large)" in content[0]["text"]
    assert files.reads == 2  # `second` refused on its declared size, before a read


def test_image_limits_must_not_be_negative() -> None:
    with pytest.raises(ValueError):
        _agent(max_images_per_turn=-1)


@pytest.mark.parametrize(("mime", "data"), [("image/png", b"GIF89a..."), ("image/jpeg", b"<svg/>"), ("image/webp", PNG)])
async def test_an_image_whose_bytes_are_not_its_type_is_described(mime: str, data: bytes) -> None:
    agent, client = _agent(vision=True)
    reply = await _run(agent, AgentInput(parts=[part_for(mime, data=data)]))
    assert reply == "ok"
    assert client.calls[0][-1]["content"] == (
        f"[Attached image: type={mime}, size={len(data)} bytes] (unavailable: content does not match its type)"
    )


async def test_every_supported_image_signature_is_sent() -> None:
    agent, client = _agent(vision=True)
    parts = [
        part_for("image/png", data=PNG),
        part_for("image/jpeg", data=JPEG),
        part_for("image/gif", data=b"GIF87a..."),
        part_for("image/webp", data=b"RIFF\x00\x00\x00\x00WEBPVP8 "),
    ]
    await _run(agent, AgentInput(parts=parts))
    assert len(client.calls[0][-1]["content"]) == 5


def test_serve_override_reaches_a_real_agent() -> None:
    """`serve(accepted_input_types=…)` and `Agent.run` accept the same types."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from genfleet.sdk.serve import create_app

    client = FakeModelClient(["ok"])
    agent = Agent(role="r", model={"model": "general-fast"}, model_client=client)  # no accepts=
    app = TestClient(create_app(agent, accepted_input_types=["Image/*"]))
    assert agent.accepted_input_types == ("image/*",)
    assert app.get("/.well-known/agent-card.json").json()["defaultInputModes"] == ["text/plain", "image/*"]
    body = app.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": {
        "messageId": "m", "role": "ROLE_USER",
        "parts": [{"text": "see"}, {"raw": base64.b64encode(PNG).decode(), "mediaType": "image/png"}],
    }}}).json()
    assert body["result"]["task"]["artifacts"][0]["parts"] == [{"text": "ok"}]
    refused = app.post("/", json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": {
        "messageId": "m", "role": "ROLE_USER", "parts": [{"url": "genfleet-file://f", "mediaType": "application/pdf"}],
    }}}).json()
    assert refused["error"]["code"] == -32005
