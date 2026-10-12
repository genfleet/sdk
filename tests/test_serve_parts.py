"""Files through ``serve``: A2A file/data parts in, typed results out (rich input and output)."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from genfleet.sdk import MAX_INLINE_BYTES, AgentInput, AgentOutput, FilePart, ImagePart, TextPart, part_for
from genfleet.sdk.serve import create_app

PNG = b"\x89PNG\r\n\x1a\nfake"
B64 = base64.b64encode(PNG).decode()


class Recorder:
    """Accepts what it is told to; replies with a CSV file and a sentence."""

    def __init__(self, accepts: list[str] | None = None) -> None:
        self.accepted_input_types = accepts or []
        self.inputs: list[AgentInput] = []

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]:
        self.inputs.append(input)
        yield AgentOutput(content="Here ")
        yield AgentOutput(parts=[TextPart(text="it is."), part_for("text/csv", data=b"a,b\n1,2", name="out.csv")])
        yield AgentOutput(parts=[part_for("image/png", uri="genfleet-file://f/1", name="chart.png", size=42)], done=True)


def _rpc(method: str, parts: list[dict], **params) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {"message": {"messageId": "m", "role": "ROLE_USER", "parts": parts}, **params},
    }


def _events(resp) -> list[dict]:
    return [json.loads(line[6:])["result"] for line in resp.text.splitlines() if line.startswith("data: ")]


def test_the_card_lists_the_accepted_types() -> None:
    card = TestClient(create_app(Recorder(["image/*", "application/pdf"]))).get("/.well-known/agent-card.json").json()
    assert card["defaultInputModes"] == ["text/plain", "image/*", "application/pdf"]
    card = TestClient(create_app(Recorder())).get("/.well-known/agent-card.json").json()
    assert card["defaultInputModes"] == ["text/plain"]


def test_an_explicit_list_overrides_the_agent_attribute() -> None:
    app = create_app(Recorder(["image/*"]), accepted_input_types=["Application/PDF"])
    assert TestClient(app).get("/.well-known/agent-card.json").json()["defaultInputModes"] == [
        "text/plain",
        "application/pdf",
    ]
    with pytest.raises(TypeError):
        create_app(Recorder(), accepted_input_types="image/*")
    with pytest.raises(ValueError):
        create_app(Recorder(), accepted_input_types=["not a type"])


def test_mixed_text_and_files_reach_the_agent_as_typed_parts() -> None:
    agent = Recorder(["*/*"])
    body = TestClient(create_app(agent)).post("/", json=_rpc("SendMessage", [
        {"text": "describe these"},
        {"raw": B64, "mediaType": "image/png", "filename": "../../a.png"},
        {"url": "genfleet-file://f/9", "mediaType": "application/pdf", "filename": "r.pdf", "metadata": {"size": 2048}},
        {"data": {"k": 1}},
    ])).json()
    assert "error" not in body
    turn = agent.inputs[0]
    assert turn.message == "describe these"
    text, image, pdf, data = turn.parts
    assert text == TextPart(text="describe these")
    assert isinstance(image, ImagePart) and image.data == PNG and image.name == "a.png" and image.size == len(PNG)
    assert isinstance(pdf, FilePart) and pdf.uri == "genfleet-file://f/9" and pdf.data is None and pdf.size == 2048
    assert isinstance(data, FilePart) and data.mime_type == "application/json" and json.loads(data.data) == {"k": 1}


@pytest.mark.parametrize("method", ["SendMessage", "SendStreamingMessage"])
def test_a_file_only_request_runs_and_returns_typed_results(method: str) -> None:
    agent = Recorder(["application/pdf"])
    resp = TestClient(create_app(agent)).post("/", json=_rpc(method, [
        {"url": "genfleet-file://f/9", "mediaType": "application/pdf"},
    ]))
    assert agent.inputs[0].message == ""
    if method == "SendMessage":
        parts = resp.json()["result"]["task"]["artifacts"][0]["parts"]
    else:
        events = _events(resp)
        parts = [p for e in events if "artifactUpdate" in e for p in e["artifactUpdate"]["artifact"]["parts"]]
        assert events[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    texts = "".join(p["text"] for p in parts if "text" in p)
    assert texts == "Here it is."
    files = [p for p in parts if "text" not in p]
    assert files == [
        {"mediaType": "text/csv", "raw": base64.b64encode(b"a,b\n1,2").decode(), "filename": "out.csv", "metadata": {"size": 7}},
        {"mediaType": "image/png", "url": "genfleet-file://f/1", "filename": "chart.png", "metadata": {"size": 42}},
    ]


def test_output_files_round_trip_into_a_typed_input() -> None:
    """What one agent returns is what the next accepts: the same part, both ways."""
    out = TestClient(create_app(Recorder())).post("/", json=_rpc("SendMessage", [{"text": "go"}])).json()
    returned = [p for p in out["result"]["task"]["artifacts"][0]["parts"] if "text" not in p]
    agent = Recorder(["*/*"])
    TestClient(create_app(agent)).post("/", json=_rpc("SendMessage", returned))
    csv, chart = agent.inputs[0].parts
    assert csv == part_for("text/csv", data=b"a,b\n1,2", name="out.csv")
    assert chart == part_for("image/png", uri="genfleet-file://f/1", name="chart.png", size=42)


@pytest.mark.parametrize("method", ["SendMessage", "SendStreamingMessage", "tasks/send"])
def test_an_unsupported_type_is_refused_by_name(method: str) -> None:
    agent = Recorder(["image/*"])
    parts = [{"text": "hi"}, {"url": "genfleet-file://f/9", "mediaType": "video/mp4"}]
    if method == "tasks/send":
        parts = [{"type": "text", "text": "hi"}, {"type": "file", "file": {"uri": "genfleet-file://f/9", "mimeType": "video/mp4"}}]
    error = TestClient(create_app(agent)).post("/", json=_rpc(method, parts)).json()["error"]
    assert error["code"] == -32005
    assert error["message"] == "Unsupported input type: video/mp4. This agent accepts: text/plain, image/*."
    assert error["data"] == {"unsupported": ["video/mp4"], "accepted": ["text/plain", "image/*"]}
    assert agent.inputs == []


def test_by_default_a_file_is_refused() -> None:
    agent = Recorder()
    error = TestClient(create_app(agent)).post("/", json=_rpc("SendMessage", [
        {"text": "hi"}, {"raw": B64, "mediaType": "image/png"},
    ])).json()["error"]
    assert error["code"] == -32005
    assert agent.inputs == []


def test_an_oversized_inline_part_is_refused() -> None:
    agent = Recorder(["*/*"])
    big = base64.b64encode(b"x" * (MAX_INLINE_BYTES + 1)).decode()
    error = TestClient(create_app(agent)).post("/", json=_rpc("SendMessage", [
        {"raw": big, "mediaType": "application/octet-stream"},
    ])).json()["error"]
    assert error["code"] == -32602
    assert "by reference" in error["message"]
    assert agent.inputs == []


def test_an_oversized_body_is_refused_before_parsing() -> None:
    agent = Recorder(["*/*"])
    app = create_app(agent, max_body_bytes=1000)
    error = TestClient(app).post("/", json=_rpc("SendMessage", [{"text": "x" * 2000}])).json()["error"]
    assert error["code"] == -32600
    assert "by reference" in error["message"]
    assert agent.inputs == []


@pytest.mark.parametrize("part", [
    {"raw": "not base64!!", "mediaType": "image/png"},
    {"url": "relative/path", "mediaType": "image/png"},
    {"url": "genfleet-file://x", "mediaType": "no slash"},
    {"metadata": {}},
    "a string",
])
def test_a_malformed_part_is_invalid_params(part) -> None:
    agent = Recorder(["*/*"])
    error = TestClient(create_app(agent)).post("/", json=_rpc("SendMessage", [part])).json()["error"]
    assert error["code"] == -32602
    assert agent.inputs == []


def test_v0_file_and_data_parts_are_typed_and_files_go_back_out() -> None:
    agent = Recorder(["*/*"])
    body = TestClient(create_app(agent)).post("/", json={
        "jsonrpc": "2.0", "id": 1, "method": "tasks/send", "params": {"message": {"role": "user", "parts": [
            {"type": "text", "text": "a"},
            {"kind": "file", "file": {"bytes": B64, "mimeType": "image/png", "name": "p.png"}},
            {"type": "data", "data": {"x": [1]}},
            {"type": "text", "text": "b"},
        ]}},
    }).json()
    turn = agent.inputs[0]
    assert turn.message == "a b"
    assert [type(p) for p in turn.parts] == [TextPart, ImagePart, FilePart, TextPart]
    parts = body["result"]["artifacts"][0]["parts"]
    assert parts[0] == {"type": "text", "text": "Here it is."}
    assert parts[1] == {"type": "file", "file": {"mimeType": "text/csv", "bytes": base64.b64encode(b"a,b\n1,2").decode(), "name": "out.csv"}}
    assert parts[2] == {"type": "file", "file": {"mimeType": "image/png", "uri": "genfleet-file://f/1", "name": "chart.png"}}


def test_the_card_lists_the_output_types() -> None:
    class Producer(Recorder):
        output_types = ["text/csv", "image/*"]

    card = TestClient(create_app(Producer())).get("/.well-known/agent-card.json").json()
    assert card["defaultOutputModes"] == ["text/plain", "text/csv", "image/*"]
    card = TestClient(create_app(Recorder(), output_types=["application/pdf"])).get("/.well-known/agent-card.json").json()
    assert card["defaultOutputModes"] == ["text/plain", "application/pdf"]
    assert TestClient(create_app(Recorder())).get("/.well-known/agent-card.json").json()["defaultOutputModes"] == [
        "text/plain"
    ]


def test_an_override_leaves_the_agent_unchanged() -> None:
    class Frozen:
        __slots__ = ("inputs",)

        def __init__(self) -> None:
            self.inputs = []

        async def run(self, input):
            self.inputs.append(input)
            yield AgentOutput(content="ok", done=True)

    agent = Frozen()
    app = TestClient(create_app(agent, accepted_input_types=["image/*"]))
    assert not hasattr(agent, "accepted_input_types")
    app.post("/", json=_rpc("SendMessage", [{"raw": B64, "mediaType": "image/png"}]))
    assert agent.inputs[0]._served_input_types == ("image/*",)


def test_served_types_never_come_from_the_request() -> None:
    agent = Recorder(["image/*"])
    TestClient(create_app(agent)).post("/", json=_rpc("SendMessage", [{"text": "hi"}], metadata={
        "_served_input_types": ["*/*"], "served_input_types": ["*/*"],
    }))
    turn = agent.inputs[0]
    assert turn._served_input_types == ("image/*",)
    assert "_served_input_types" not in turn.model_dump()
    assert AgentInput.model_validate({"message": "x", "_served_input_types": ["*/*"]})._served_input_types is None
