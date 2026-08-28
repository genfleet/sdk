from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, Required, TypedDict

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    invocation_id: str = ""
    session_id: str = ""
    agent_name: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    status: Literal["ok", "error"] = "ok"


class AuditConfig(TypedDict, total=False):
    backend: Required[Literal["console", "file", "callback"]]
    file_path: str
    callback: Callable
    max_content_length: int
    events: list[str]
    agent_name: str
