"""A2A v1.0 in ``serve`` (sdk#46, ADR-0027): the card, SendMessage, SendStreamingMessage."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from genfleet.sdk import TOOL_EVENT_KEY, AgentInput, AgentOutput
from genfleet.sdk.serve import HISTORY_METADATA_KEY, create_app


class Agent:
    def __init__(self) -> None:
        self.inputs: list[AgentInput] = []

    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]:
        self.inputs.append(input)
        yield AgentOutput(content="Hel")
        yield AgentOutput(content="", metadata={TOOL_EVENT_KEY: {"type": "tool_call", "id": "1", "name": "t", "arguments": {}}})
        yield AgentOutput(content="lo", done=True)


class Failing:
    async def run(self, input: AgentInput) -> AsyncIterator[AgentOutput]:
        raise RuntimeError("provider said sk-secret")
        yield  # pragma: no cover


def _rpc(method: str, params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}


def _message(text: str = "hi", **extra) -> dict:
    return {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": text}], **extra}


def _events(resp) -> list[dict]:
    return [json.loads(line[6:])["result"] for line in resp.text.splitlines() if line.startswith("data: ")]


def test_the_card_is_v1_at_both_paths() -> None:
    client = TestClient(create_app(Agent(), name="bot", description="d"))
    new, old = client.get("/.well-known/agent-card.json").json(), client.get("/.well-known/agent.json").json()
    assert new == old
    assert new["supportedInterfaces"] == [
        {"url": "http://testserver/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
    ]
    assert new["capabilities"] == {"streaming": True, "pushNotifications": False, "extendedAgentCard": False}
    assert "url" not in new


def test_send_message_returns_a_completed_task() -> None:
    body = TestClient(create_app(Agent())).post("/", json=_rpc("SendMessage", {"message": _message()})).json()
    task = body["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"] == [{"artifactId": "result", "parts": [{"text": "Hello"}]}]


def test_context_id_is_the_session_and_history_comes_from_metadata() -> None:
    agent = Agent()
    TestClient(create_app(agent)).post("/", json=_rpc("SendMessage", {
        "message": _message(contextId="ctx-1"),
        "metadata": {"channel": "web", HISTORY_METADATA_KEY: [{"role": "user", "content": "earlier"}]},
    }))
    assert agent.inputs[0].metadata == {"channel": "web", "session_id": "ctx-1"}
    assert [m.content for m in agent.inputs[0].history] == ["earlier"]


def test_a_malformed_message_is_invalid_params() -> None:
    body = TestClient(create_app(Agent())).post("/", json=_rpc("SendMessage", {"message": "hi"})).json()
    assert body["error"]["code"] == -32602


def test_the_stream_is_working_chunks_then_completed() -> None:
    resp = TestClient(create_app(Agent())).post("/", json=_rpc("SendStreamingMessage", {"message": _message()}))
    events = _events(resp)
    assert events[0]["statusUpdate"]["status"]["state"] == "TASK_STATE_WORKING"
    chunks = [e["artifactUpdate"] for e in events if "artifactUpdate" in e]
    assert [(c["append"], c["lastChunk"], c["artifact"]["parts"][0]["text"]) for c in chunks] == [
        (False, False, "Hel"), (True, True, "lo"),
    ]
    assert any(TOOL_EVENT_KEY in (e.get("statusUpdate", {}).get("metadata") or {}) for e in events)
    assert events[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert all("kind" not in e and "final" not in json.dumps(e) for e in events)


def test_a_failed_stream_is_opaque() -> None:
    resp = TestClient(create_app(Failing())).post("/", json=_rpc("SendStreamingMessage", {"message": _message()}))
    last = _events(resp)[-1]["statusUpdate"]["status"]
    assert last["state"] == "TASK_STATE_FAILED"
    assert last["message"]["role"] == "ROLE_AGENT"
    assert "sk-secret" not in resp.text


@pytest.mark.parametrize(
    ("method", "code"),
    [("GetTask", -32004), ("CancelTask", -32004), ("CreateTaskPushNotificationConfig", -32003),
     ("GetExtendedAgentCard", -32007)],
)
def test_what_a_stateless_server_refuses(method: str, code: int) -> None:
    assert TestClient(create_app(Agent())).post("/", json=_rpc(method, {"id": "t"})).json()["error"]["code"] == code


def test_an_unsupported_version_is_refused() -> None:
    resp = TestClient(create_app(Agent())).post(
        "/", json=_rpc("SendMessage", {"message": _message()}), headers={"A2A-Version": "2.0"}
    )
    assert resp.json()["error"]["code"] == -32009
