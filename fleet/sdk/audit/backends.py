from __future__ import annotations

import asyncio
import json
import logging
from typing import Protocol, runtime_checkable

from .schemas import AuditEvent

log = logging.getLogger("fleet.audit")


@runtime_checkable
class AuditBackend(Protocol):
    async def emit(self, event: AuditEvent) -> None: ...


class ConsoleAuditBackend:
    async def emit(self, event: AuditEvent) -> None:
        log.info(event.model_dump_json())


class FileAuditBackend:
    def __init__(self, file_path: str = "audit.jsonl") -> None:
        self._file_path = file_path

    async def emit(self, event: AuditEvent) -> None:
        line = event.model_dump_json() + "\n"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._append, line)

    def _append(self, line: str) -> None:
        with open(self._file_path, "a") as f:
            f.write(line)


class CallbackAuditBackend:
    def __init__(self, callback) -> None:
        self._callback = callback

    async def emit(self, event: AuditEvent) -> None:
        result = self._callback(event)
        if asyncio.iscoroutine(result):
            await result
