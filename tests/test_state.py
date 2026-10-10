"""Durable state survives process boundaries and enforces transition claims."""

import pytest
from concurrent.futures import ThreadPoolExecutor

from genfleet.sdk.state import SQLiteStateStore, StateError


def test_restart_and_resume_once(tmp_path):
    path = tmp_path / "runs.sqlite3"
    first = SQLiteStateStore(path)
    run = first.create("agent-1", "verified-caller-1", {"step": "start"})
    run = first.checkpoint(run.run_id, agent="agent-1", owner="verified-caller-1",
                           version=run.version, state={"step": "charged", "receipt": "r-1"})
    run, token = first.pause(run.run_id, agent="agent-1", owner="verified-caller-1",
                             version=run.version, state={"step": "approval", "receipt": "r-1"})

    restarted = SQLiteStateStore(path)
    assert restarted.get(run.run_id, agent="agent-1", owner="verified-caller-1") == run
    resumed = restarted.resume(run.run_id, agent="agent-1", owner="verified-caller-1", token=token)
    assert resumed.state["receipt"] == "r-1"
    with pytest.raises(StateError, match="invalid or expired"):
        first.resume(run.run_id, agent="agent-1", owner="verified-caller-1", token=token)
    done = restarted.complete(run.run_id, agent="agent-1", owner="verified-caller-1",
                              version=resumed.version, state={"result": "ok"})
    assert done.status == "completed"


def test_caller_and_agent_scope(tmp_path):
    store = SQLiteStateStore(tmp_path / "runs.db")
    run = store.create("agent-1", "alice", {})
    run, token = store.pause(run.run_id, agent="agent-1", owner="alice",
                             version=run.version, state={})
    for agent, owner in [("agent-1", "bob"), ("agent-2", "alice")]:
        with pytest.raises(StateError):
            store.get(run.run_id, agent=agent, owner=owner)
        with pytest.raises(StateError):
            store.resume(run.run_id, agent=agent, owner=owner, token=token)
    assert store.resume(run.run_id, agent="agent-1", owner="alice", token=token).status == "running"


def test_expired_approval_and_stale_checkpoint(tmp_path, monkeypatch):
    import genfleet.sdk.state as state_module

    store = SQLiteStateStore(tmp_path / "runs.db")
    run = store.create("agent", "alice", {"step": 0})
    newer = store.checkpoint(run.run_id, agent="agent", owner="alice",
                             version=run.version, state={"step": 1})
    with pytest.raises(StateError, match="version changed"):
        store.checkpoint(run.run_id, agent="agent", owner="alice",
                         version=run.version, state={"step": 0})
    monkeypatch.setattr(state_module.time, "time", lambda: 100.0)
    paused, token = store.pause(run.run_id, agent="agent", owner="alice",
                                version=newer.version, state={"step": 1}, ttl_seconds=5)
    monkeypatch.setattr(state_module.time, "time", lambda: 105.0)
    with pytest.raises(StateError, match="expired"):
        store.resume(run.run_id, agent="agent", owner="alice", token=token)
    assert store.get(run.run_id, agent="agent", owner="alice") == paused


def test_concurrent_resume_has_one_winner(tmp_path):
    path = tmp_path / "runs.db"
    store = SQLiteStateStore(path)
    run = store.create("agent", "alice", {"step": "approval"})
    run, token = store.pause(run.run_id, agent="agent", owner="alice",
                             version=run.version, state=run.state)

    def claim():
        try:
            return SQLiteStateStore(path).resume(
                run.run_id, agent="agent", owner="alice", token=token
            ).status
        except StateError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: claim(), range(2))) == ["rejected", "running"]


def test_effect_claim_survives_restart_and_never_replays(tmp_path):
    path = tmp_path / "runs.db"
    store = SQLiteStateStore(path)
    run = store.create("agent", "alice", {})
    effect = store.claim_effect(run.run_id, "charge-order-1", agent="agent", owner="alice")
    assert effect.status == "claimed"
    assert len(effect.idempotency_key) == 64

    restarted = SQLiteStateStore(path)
    assert restarted.get_effect(run.run_id, "charge-order-1", agent="agent", owner="alice") == effect
    with pytest.raises(StateError, match="already claimed"):
        restarted.claim_effect(run.run_id, "charge-order-1", agent="agent", owner="alice")

    completed = restarted.complete_effect(run.run_id, "charge-order-1", agent="agent",
                                          owner="alice", result={"receipt": "r-1"})
    assert restarted.get_effect(run.run_id, "charge-order-1", agent="agent", owner="alice") == completed
    with pytest.raises(StateError, match="already completed"):
        restarted.complete_effect(run.run_id, "charge-order-1", agent="agent",
                                  owner="alice", result={"receipt": "r-2"})


def test_effect_claim_is_caller_scoped_and_atomic(tmp_path):
    path = tmp_path / "runs.db"
    store = SQLiteStateStore(path)
    run = store.create("agent", "alice", {})

    def claim():
        try:
            return SQLiteStateStore(path).claim_effect(
                run.run_id, "send-email", agent="agent", owner="alice"
            ).status
        except StateError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: claim(), range(2))) == ["claimed", "rejected"]
    with pytest.raises(StateError):
        store.get_effect(run.run_id, "send-email", agent="agent", owner="bob")
    with pytest.raises(StateError):
        store.complete_effect(run.run_id, "send-email", agent="agent",
                              owner="bob", result="sent")


def test_hosted_sandbox_refuses_ephemeral_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("GENFLEET_HOSTED", "1")
    with pytest.raises(RuntimeError, match="hosted sandbox"):
        SQLiteStateStore(tmp_path / "runs.db")


def test_cancel_expired_run_and_purge_caller_data(tmp_path, monkeypatch):
    import genfleet.sdk.state as state_module

    store = SQLiteStateStore(tmp_path / "runs.db")
    run = store.create("agent", "alice", {"order": "one"})
    store.claim_effect(run.run_id, "charge", agent="agent", owner="alice")
    monkeypatch.setattr(state_module.time, "time", lambda: 100)
    paused, token = store.pause(run.run_id, agent="agent", owner="alice",
                                version=run.version, state=run.state, ttl_seconds=1)
    monkeypatch.setattr(state_module.time, "time", lambda: 101)
    with pytest.raises(StateError):
        store.resume(run.run_id, agent="agent", owner="alice", token=token)
    cancelled = store.cancel(run.run_id, agent="agent", owner="alice", version=paused.version)
    assert cancelled.status == "cancelled"
    assert store.purge(agent="agent", owner="bob") == 0
    assert store.purge(agent="agent", owner="alice") == 1
    with pytest.raises(StateError):
        store.get_effect(run.run_id, "charge", agent="agent", owner="alice")


def test_payload_limit_and_file_permissions(tmp_path):
    store = SQLiteStateStore(tmp_path / "runs.db")
    assert (tmp_path / "runs.db").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="64 KiB"):
        store.create("agent", "alice", {"large": "x" * 65536})


def test_confirmed_effect_can_be_recorded_after_pause(tmp_path):
    store = SQLiteStateStore(tmp_path / "runs.db")
    run = store.create("agent", "alice", {})
    store.claim_effect(run.run_id, "send-email", agent="agent", owner="alice")
    store.pause(run.run_id, agent="agent", owner="alice", version=run.version,
                state={"step": "waiting"})
    effect = store.complete_effect(run.run_id, "send-email", agent="agent",
                                   owner="alice", result={"message_id": "m-1"})
    assert effect.result == {"message_id": "m-1"}
