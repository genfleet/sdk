"""Local workflow checkpoints using Python's SQLite library.

The caller identity must come from trusted ingress (for example the platform's
verified sender), never from a resume request's unverified payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .caller import _hosted

_MAX_JSON_BYTES = 64 * 1024


class StateError(RuntimeError):
    """A run cannot make the requested state transition."""


@dataclass(frozen=True)
class RunState:
    run_id: str
    status: str
    state: Any
    version: int


@dataclass(frozen=True)
class EffectState:
    effect_id: str
    idempotency_key: str
    status: str
    result: Any = None


class SQLiteStateStore:
    """A local or self-hosted workflow store on a persistent filesystem.

    Each transition opens its own connection and commits atomically. ``owner``
    is an opaque, verified caller ID and ``agent`` is the agent's stable ID.
    Neither identifier is accepted as proof of identity by this class.
    """

    def __init__(self, path: str | Path):
        if _hosted():
            raise RuntimeError("SQLiteStateStore cannot persist in a hosted sandbox; use platform state")
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("durable state requires a file path, not :memory:")
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_runs (
                run_id TEXT PRIMARY KEY, agent TEXT NOT NULL, owner TEXT NOT NULL,
                status TEXT NOT NULL, state TEXT NOT NULL, version INTEGER NOT NULL,
                token_hash TEXT, expires_at REAL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_effects (
                run_id TEXT NOT NULL, effect_id TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT,
                PRIMARY KEY (run_id, effect_id),
                FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
            )""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=1)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _identity(agent: str, owner: str) -> None:
        if not agent or not owner:
            raise ValueError("agent and verified owner are required")

    @staticmethod
    def _json(state: Any) -> str:
        encoded = json.dumps(state, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
            raise ValueError("state or result exceeds 64 KiB")
        return encoded

    def create(self, agent: str, owner: str, state: Any) -> RunState:
        self._identity(agent, owner)
        run_id = secrets.token_urlsafe(24)
        with self._connect() as db:
            db.execute("""INSERT INTO agent_runs
                (run_id, agent, owner, status, state, version, token_hash, expires_at)
                VALUES (?, ?, ?, 'running', ?, 1, NULL, NULL)""",
                       (run_id, agent, owner, self._json(state)))
        return RunState(run_id, "running", state, 1)

    def get(self, run_id: str, *, agent: str, owner: str) -> RunState:
        self._identity(agent, owner)
        with self._connect() as db:
            row = db.execute("SELECT status, state, version FROM agent_runs WHERE run_id=? AND agent=? AND owner=?",
                             (run_id, agent, owner)).fetchone()
        if row is None:
            raise StateError("run not found")
        return RunState(run_id, row[0], json.loads(row[1]), row[2])

    def checkpoint(self, run_id: str, *, agent: str, owner: str, version: int, state: Any) -> RunState:
        self._identity(agent, owner)
        with self._connect() as db:
            cursor = db.execute("""UPDATE agent_runs SET state=?, version=version+1
                WHERE run_id=? AND agent=? AND owner=? AND status='running' AND version=?""",
                (self._json(state), run_id, agent, owner, version))
            if cursor.rowcount != 1:
                raise StateError("run missing, not running, or version changed")
        return RunState(run_id, "running", state, version + 1)

    def pause(self, run_id: str, *, agent: str, owner: str, version: int,
              state: Any, ttl_seconds: float = 3600) -> tuple[RunState, str]:
        self._identity(agent, owner)
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as db:
            cursor = db.execute("""UPDATE agent_runs SET status='paused', state=?,
                version=version+1, token_hash=?, expires_at=?
                WHERE run_id=? AND agent=? AND owner=? AND status='running' AND version=?""",
                (self._json(state), digest, time.time() + ttl_seconds,
                 run_id, agent, owner, version))
            if cursor.rowcount != 1:
                raise StateError("run missing, not running, or version changed")
        return RunState(run_id, "paused", state, version + 1), token

    def resume(self, run_id: str, *, agent: str, owner: str, token: str) -> RunState:
        """Claim a pause exactly once. A duplicate or expired claim fails."""
        self._identity(agent, owner)
        if not token:
            raise StateError("invalid or expired resume token")
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as db:
            cursor = db.execute("""UPDATE agent_runs SET status='running',
                version=version+1, token_hash=NULL, expires_at=NULL
                WHERE run_id=? AND agent=? AND owner=? AND status='paused'
                AND token_hash=? AND expires_at>?""",
                (run_id, agent, owner, digest, time.time()))
            if cursor.rowcount != 1:
                raise StateError("invalid or expired resume token")
            row = db.execute("SELECT state, version FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
        return RunState(run_id, "running", json.loads(row[0]), row[1])

    def complete(self, run_id: str, *, agent: str, owner: str, version: int,
                 state: Any) -> RunState:
        self._identity(agent, owner)
        with self._connect() as db:
            cursor = db.execute("""UPDATE agent_runs SET status='completed', state=?,
                version=version+1 WHERE run_id=? AND agent=? AND owner=?
                AND status='running' AND version=?""",
                (self._json(state), run_id, agent, owner, version))
            if cursor.rowcount != 1:
                raise StateError("run missing, not running, or version changed")
        return RunState(run_id, "completed", state, version + 1)

    def cancel(self, run_id: str, *, agent: str, owner: str, version: int) -> RunState:
        """Close a running or paused run, including an expired pause."""
        self._identity(agent, owner)
        with self._connect() as db:
            cursor = db.execute("""UPDATE agent_runs SET status='cancelled',
                version=version+1, token_hash=NULL, expires_at=NULL
                WHERE run_id=? AND agent=? AND owner=? AND version=?
                AND status IN ('running', 'paused')""",
                (run_id, agent, owner, version))
            if cursor.rowcount != 1:
                raise StateError("run missing, closed, or version changed")
            row = db.execute("SELECT state FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
        return RunState(run_id, "cancelled", json.loads(row[0]), version + 1)

    def purge(self, *, agent: str, owner: str) -> int:
        """Delete this caller's runs and their effect records."""
        self._identity(agent, owner)
        with self._connect() as db:
            db.execute("""DELETE FROM agent_effects WHERE run_id IN
                (SELECT run_id FROM agent_runs WHERE agent=? AND owner=?)""", (agent, owner))
            cursor = db.execute("DELETE FROM agent_runs WHERE agent=? AND owner=?", (agent, owner))
        return cursor.rowcount

    @staticmethod
    def _effect_key(run_id: str, effect_id: str) -> str:
        return hashlib.sha256(f"genfleet-effect-v1:{run_id}:{effect_id}".encode()).hexdigest()

    def claim_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState:
        """Record intent before an external call; refuse every duplicate claim.

        Pass the returned key to a remote API that supports idempotency. A
        duplicate means the remote outcome may be unknown: inspect it before
        deciding whether to retry or compensate.
        """
        self._identity(agent, owner)
        if not effect_id:
            raise ValueError("effect_id is required")
        with self._connect() as db:
            cursor = db.execute("""INSERT OR IGNORE INTO agent_effects
                (run_id, effect_id, status, result)
                SELECT run_id, ?, 'claimed', NULL FROM agent_runs
                WHERE run_id=? AND agent=? AND owner=? AND status='running'""",
                (effect_id, run_id, agent, owner))
            if cursor.rowcount != 1:
                raise StateError("effect already claimed or run not running")
        return EffectState(effect_id, self._effect_key(run_id, effect_id), "claimed")

    def get_effect(self, run_id: str, effect_id: str, *, agent: str, owner: str) -> EffectState:
        self._identity(agent, owner)
        with self._connect() as db:
            row = db.execute("""SELECT e.status, e.result FROM agent_effects e
                JOIN agent_runs r ON r.run_id=e.run_id
                WHERE e.run_id=? AND e.effect_id=? AND r.agent=? AND r.owner=?""",
                (run_id, effect_id, agent, owner)).fetchone()
        if row is None:
            raise StateError("effect not found")
        return EffectState(effect_id, self._effect_key(run_id, effect_id),
                           row[0], json.loads(row[1]) if row[1] is not None else None)

    def complete_effect(self, run_id: str, effect_id: str, *, agent: str,
                        owner: str, result: Any) -> EffectState:
        """Record a confirmed external result without allowing it to change."""
        self._identity(agent, owner)
        with self._connect() as db:
            cursor = db.execute("""UPDATE agent_effects SET status='completed', result=?
                WHERE run_id=? AND effect_id=? AND status='claimed' AND EXISTS (
                    SELECT 1 FROM agent_runs WHERE run_id=? AND agent=? AND owner=?)""",
                (self._json(result), run_id, effect_id, run_id, agent, owner))
            if cursor.rowcount != 1:
                raise StateError("effect missing or already completed")
        return EffectState(effect_id, self._effect_key(run_id, effect_id), "completed", result)
