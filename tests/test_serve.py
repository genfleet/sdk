"""Tests for the A2A serve module — session/metadata/history forwarding."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from agentinc.sdk import AgentInput, AgentOutput
from agentinc.sdk.serve import create_app


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
    _send(client, _params(metadata={"channel": "whatsapp", "tenant": "acme"}))
    assert agent.inputs[0].metadata["channel"] == "whatsapp"
    assert agent.inputs[0].metadata["tenant"] == "acme"


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
