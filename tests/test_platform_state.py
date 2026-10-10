"""Platform state uses the proxy wire contract and never sends its own scope."""

import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from genfleet.sdk import turn as sdk_turn
from genfleet.sdk.platform_state import PlatformStateError, PlatformStateStore
from genfleet.sdk.state import StateError, StateQuotaExceeded, StateTooLarge
from genfleet.sdk.state_store import state_for


class Proxy(BaseHTTPRequestHandler):
    calls: list = []
    #: op -> (status, body) overriding the default answer.
    answers: dict = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.calls.append((self.path, body, self.headers.get("authorization")))
        op = self.path.rsplit("/", 1)[-1]
        if op in self.answers:
            status, data = self.answers[op]
        elif op == "create":
            status, data = 200, {"run_id": "run-1", "status": "running", "state": body["state"], "version": 1}
        elif op == "pause":
            status, data = 200, {"run_id": body["run_id"], "status": "paused", "state": body["state"],
                                 "version": 2, "token": "resume-1"}
        elif op == "resume":
            status, data = 200, {"run_id": body["run_id"], "status": "running", "state": {"step": 1}, "version": 3}
        elif op == "get":
            status, data = 200, {"run_id": body["run_id"], "status": "running", "state": {}, "version": 1}
        elif op in ("purge", "purge_subject"):
            status, data = 200, {"runs": 2}
        else:
            status, data = 400, {"error": "unknown"}
        raw = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_):
        pass


@pytest.fixture
def proxy():
    Proxy.calls = []
    Proxy.answers = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/state"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_create_pause_resume_through_proxy(proxy):
    store = PlatformStateStore(proxy, "spawn-token")
    run = await store.create("agent-name", "caller-1", {"step": 0}, subject="customer-1")
    paused, token = await store.pause(run.run_id, agent="agent-name", owner="caller-1",
                                      version=run.version, state={"step": 1})
    resumed = await store.resume(run.run_id, agent="agent-name", owner="caller-1", token=token)
    assert (paused.status, resumed.version) == ("paused", 3)
    assert [call[2] for call in Proxy.calls] == ["Bearer spawn-token"] * 3


async def test_scope_and_owner_are_never_sent(proxy):
    """The engine stamps tenant, agent and owner; the SDK sends none of them."""
    store = PlatformStateStore(proxy, "spawn-token")
    run = await store.create("agent-name", "caller-1", {"step": 0})
    await store.get(run.run_id, agent="agent-name", owner="someone-else")
    await store.purge(agent="agent-name", owner="caller-1")
    for _, body, _ in Proxy.calls:
        assert not {"owner", "agent", "tenantId", "agentName", "tenant_id", "agent_name"} & body.keys()


async def test_the_running_turn_is_named_so_the_engine_can_stamp_its_caller(proxy):
    store = PlatformStateStore(proxy, "spawn-token")
    metadata = {"session_id": "s-1", "turn_id": "t-1", "genfleet.caller": {"role": "customer", "id": "x"}}
    await sdk_turn.call_within(metadata, store.create("agent-name", "ignored", {}))
    await store.create("agent-name", "ignored", {})  # outside a turn: the background scope
    (_, in_turn, _), (_, outside, _) = Proxy.calls
    assert (in_turn["session_id"], in_turn["turn_id"]) == ("s-1", "t-1")
    assert "session_id" not in outside and "turn_id" not in outside
    assert "genfleet.caller" not in in_turn


async def test_a_second_agent_name_is_refused_rather_than_sharing_runs(proxy):
    store = PlatformStateStore(proxy, "spawn-token")
    await store.create("orders", "c", {})
    with pytest.raises(StateError, match="one scope per agent"):
        await store.purge(agent="refunds", owner="c")
    assert [path for path, _, _ in Proxy.calls] == ["/state/create"]


@pytest.mark.parametrize(
    ("status", "error"),
    [(409, PlatformStateError), (404, PlatformStateError), (502, PlatformStateError),
     (413, StateTooLarge), (429, StateQuotaExceeded)],
)
async def test_proxy_refusals_are_state_errors(proxy, status, error):
    Proxy.answers["checkpoint"] = (status, {"error": "state checkpoint rejected"})
    store = PlatformStateStore(proxy, "spawn-token")
    with pytest.raises(error) as raised:
        await store.checkpoint("run-1", agent="a", owner="o", version=1, state={})
    assert isinstance(raised.value, PlatformStateError)
    assert isinstance(raised.value, StateError)
    assert (raised.value.op, raised.value.status) == ("checkpoint", status)


async def test_an_unreachable_proxy_is_a_state_error():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    store = PlatformStateStore(f"http://127.0.0.1:{port}/state", "spawn-token")
    with pytest.raises(PlatformStateError) as raised:
        await store.get("run-1", agent="a", owner="o")
    assert raised.value.status == 0


@pytest.mark.parametrize(
    ("op", "answer"),
    [("get", {"status": "running"}), ("get", {"run_id": "r", "status": "running", "state": {}, "version": "x"}),
     ("pause", {"run_id": "r", "status": "paused", "state": {}, "version": 2}),
     ("claim_effect", {"effect_id": "e"}), ("purge", {}), ("get", []), ("get", b"not json")],
)
async def test_a_malformed_reply_is_a_state_error_not_a_key_error(proxy, op, answer):
    Proxy.answers[op] = (200, answer)
    store = PlatformStateStore(proxy, "spawn-token")
    calls = {
        "get": lambda: store.get("r", agent="a", owner="o"),
        "pause": lambda: store.pause("r", agent="a", owner="o", version=1, state={}),
        "claim_effect": lambda: store.claim_effect("r", "e", agent="a", owner="o"),
        "purge": lambda: store.purge(agent="a", owner="o"),
    }
    with pytest.raises(PlatformStateError):
        await calls[op]()


def test_hosted_state_selected_from_env(monkeypatch):
    monkeypatch.setenv("GENFLEET_STATE_URL", "http://engine/state")
    monkeypatch.setenv("GENFLEET_STATE_TOKEN", "spawn-token")
    assert isinstance(state_for(), PlatformStateStore)
    monkeypatch.delenv("GENFLEET_STATE_TOKEN")
    with pytest.raises(RuntimeError, match="GENFLEET_STATE_TOKEN"):
        state_for()
