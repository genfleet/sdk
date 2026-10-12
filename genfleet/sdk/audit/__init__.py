from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from .backends import (
    AuditBackend,
    CallbackAuditBackend,
    ConsoleAuditBackend,
    FileAuditBackend,
)
from .schemas import AuditConfig, AuditEvent


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class Auditor:
    def __init__(self, backend: AuditBackend, config: AuditConfig) -> None:
        self._backend = backend
        self._max_content_length = config.get("max_content_length", 500)
        self._events_filter: set[str] | None = (
            set(config["events"]) if "events" in config else None
        )
        self._agent_name = config.get("agent_name")

    def truncate(self, text: str) -> str:
        return _truncate(text, self._max_content_length)

    async def emit(
        self,
        event_type: str,
        *,
        invocation_id: str = "",
        session_id: str = "",
        data: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        status: str = "ok",
    ) -> None:
        if self._events_filter and event_type not in self._events_filter:
            return

        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC),
            invocation_id=invocation_id,
            session_id=session_id,
            agent_name=self._agent_name,
            data=data or {},
            latency_ms=latency_ms,
            status=status,
        )
        await self._backend.emit(event)


def audit_for(config: AuditConfig | None) -> Auditor | None:
    if config is None:
        return None

    backend_type = config["backend"]
    backend: AuditBackend
    if backend_type == "console":
        backend = ConsoleAuditBackend()
    elif backend_type == "file":
        backend = FileAuditBackend(config.get("file_path", "audit.jsonl"))
    elif backend_type == "callback":
        if "callback" not in config:
            raise ValueError("callback backend requires a 'callback' key in AuditConfig")
        backend = CallbackAuditBackend(config["callback"])
    else:
        raise ValueError(f"Unknown audit backend: {backend_type}")

    return Auditor(backend, config)
