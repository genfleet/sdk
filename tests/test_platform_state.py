"""Platform state uses the proxy wire contract and never sends tenant scope."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from genfleet.sdk.platform_state import PlatformStateStore, PlatformStateError
from genfleet.sdk.state_store import state_for


class Proxy(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.calls.append((self.path, body, self.headers.get("authorization")))
        if self.path.endswith("/create"):
            data = {"run_id": "run-1", "status": "running", "state": body["state"], "version": 1}
        elif self.path.endswith("/pause"):
            data = {"run_id": body["run_id"], "status": "paused", "state": body["state"],
                    "version": 2, "token": "resume-1"}
        elif self.path.endswith("/resume"):
            data = {"run_id": body["run_id"], "status": "running", "state": {"step": 1}, "version": 3}
        else:
            data = {"error": "unknown"}
        self.send_response(200 if "error" not in data else 400)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, *_):
        pass


@pytest.fixture
def proxy():
    Proxy.calls = []
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
    assert all("agent" not in body and "tenantId" not in body for _, body, _ in Proxy.calls)


def test_hosted_state_selected_from_env(monkeypatch):
    monkeypatch.setenv("GENFLEET_STATE_URL", "http://engine/state")
    monkeypatch.setenv("GENFLEET_STATE_TOKEN", "spawn-token")
    assert isinstance(state_for(), PlatformStateStore)
    monkeypatch.delenv("GENFLEET_STATE_TOKEN")
    with pytest.raises(RuntimeError, match="GENFLEET_STATE_TOKEN"):
        state_for()
