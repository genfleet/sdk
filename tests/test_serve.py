"""Tests for the A2A serve module — session/metadata/history forwarding."""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from genfleet.sdk import AgentInput, AgentOutput
from genfleet.sdk.serve import create_app


class RecordingAgent:
    """Stub agent that records the AgentInput it receives."""

    def __init__(self) -> None:
        self.inputs: list[AgentInput] = []

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]:
        self.inputs.append(input)
        yield AgentOutput(content="ok", done=True)


def _send(client: TestClient, params: dict) -> dict:
    resp = client.post("/", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks/send",
        "params": params,
    })
    assert resp.status_code == 200
    return resp.json()


def _params(text: str = "hello", **extra) -> dict:
    return {
        "id": "t1",
        "message": {"role": "user", "parts": [{"type": "text", "text": text}]},
        **extra,
    }


@pytest.fixture()
def agent() -> RecordingAgent:
    return RecordingAgent()


@pytest.fixture()
def client(agent: RecordingAgent) -> TestClient:
    return TestClient(create_app(agent, name="test-agent"))


def test_message_text_forwarded(agent: RecordingAgent, client: TestClient) -> None:
    body = _send(client, _params("hi there"))
    assert body["result"]["status"]["state"] == "completed"
    assert agent.inputs[0].message == "hi there"


def test_session_id_populates_metadata(agent: RecordingAgent, client: TestClient) -> None:
    _send(client, _params(sessionId="sess-42"))
    assert agent.inputs[0].metadata["session_id"] == "sess-42"


def test_metadata_passthrough(agent: RecordingAgent, client: TestClient) -> None:
    _send(client, _params(metadata={"channel": "whatsapp", "tenant": "heatec"}))
    assert agent.inputs[0].metadata["channel"] == "whatsapp"
    assert agent.inputs[0].metadata["tenant"] == "heatec"


def test_explicit_metadata_session_id_wins(agent: RecordingAgent, client: TestClient) -> None:
    _send(client, _params(sessionId="outer", metadata={"session_id": "inner"}))
    assert agent.inputs[0].metadata["session_id"] == "inner"


def test_non_dict_metadata_ignored(agent: RecordingAgent, client: TestClient) -> None:
    _send(client, _params(metadata="not-a-dict", sessionId="s1"))
    assert agent.inputs[0].metadata == {"session_id": "s1"}


def test_history_forwarded(agent: RecordingAgent, client: TestClient) -> None:
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    _send(client, _params(history=history))
    got = agent.inputs[0].history
    assert [(m.role, m.content) for m in got] == [
        ("user", "earlier question"),
        ("assistant", "earlier answer"),
    ]


def test_invalid_history_ignored(agent: RecordingAgent, client: TestClient) -> None:
    body = _send(client, _params(history=[{"role": "alien", "content": 5}]))
    assert body["result"]["status"]["state"] == "completed"
    assert agent.inputs[0].history == []


def test_non_dict_params_handled(agent: RecordingAgent, client: TestClient) -> None:
    resp = client.post("/", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks/send",
        "params": [1, 2, 3],
    })
    assert resp.status_code == 200
    assert resp.json()["result"]["status"]["state"] == "completed"
    assert agent.inputs[0].message == ""


def test_non_dict_body_rejected(client: TestClient) -> None:
    resp = client.post("/", json=[1, 2, 3])
    assert resp.status_code == 200
    assert resp.json()["error"]["code"] == -32600


def test_send_subscribe_forwards_session(agent: RecordingAgent, client: TestClient) -> None:
    resp = client.post("/", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks/sendSubscribe",
        "params": _params(sessionId="sse-sess"),
    })
    assert resp.status_code == 200
    assert "completed" in resp.text
    assert agent.inputs[0].metadata["session_id"] == "sse-sess"


# --------------------------------------------------------------------------
# sdk#20 — the agent's exception must not cross the network
# --------------------------------------------------------------------------

#: Stands in for what providers actually put in exception text: OpenAI echoes
#: the submitted key, others embed request payloads, file paths, console URLs.
LEAKY_MESSAGE = (
    "Incorrect API key provided: sk-proj-REDACTME. "
    'File "/home/dev/.venv/lib/site-packages/openai/_client.py", line 1'
)


class ExplodingAgent:
    """Raises the way a provider SDK does — with secrets in the message."""

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]:
        raise RuntimeError(LEAKY_MESSAGE)
        yield  # pragma: no cover — unreachable, keeps this an async generator


@pytest.fixture()
def exploding_client() -> TestClient:
    return TestClient(create_app(ExplodingAgent(), name="boom"))


def test_send_does_not_return_the_exception_text(exploding_client: TestClient) -> None:
    body = _send(exploding_client, _params())

    message = body["error"]["message"]
    assert "sk-proj-REDACTME" not in message
    assert "site-packages" not in message
    assert body["error"]["code"] == -32603


def test_stream_does_not_return_the_exception_text(exploding_client: TestClient) -> None:
    """The flag named only tasks/send; this path leaked the same way."""
    resp = exploding_client.post("/", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks/sendSubscribe",
        "params": _params(),
    })

    assert resp.status_code == 200
    assert "sk-proj-REDACTME" not in resp.text
    assert "site-packages" not in resp.text
    # Not `"failed" in resp.text`: OPAQUE_FAILURE_MESSAGE contains the word
    # "failed", so that assertion passes even on a completed task.
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    status = events[-1]["result"]["status"]
    assert status["state"] == "failed"
    # A2A shape: the message is a Message, so the id is reachable by a
    # spec-conforming client rather than only by reading the raw stream.
    text = "".join(p["text"] for p in status["message"]["parts"] if p["type"] == "text")
    assert "error id: " in text


def test_failure_carries_an_id_the_operator_can_correlate(
    exploding_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """An opaque message is only useful if the detail is findable somewhere."""
    with caplog.at_level("ERROR", logger="genfleet.sdk.serve"):
        message = _send(exploding_client, _params())["error"]["message"]

    error_id = message.rsplit("error id: ", 1)[1].rstrip(")")
    assert error_id in caplog.text
    # The operator's own log is where the detail is allowed to be.
    assert "sk-proj-REDACTME" in caplog.text
