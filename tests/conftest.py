"""Shared fixtures."""

from __future__ import annotations

import socket
import threading

import pytest


class RawServer:
    """A TCP server that answers every connection with ``answer`` (raw bytes,
    not necessarily valid HTTP) and records what each request carried."""

    def __init__(self) -> None:
        self.answer = b""
        self.requests: list[bytes] = []
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen()
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                conn.settimeout(2)
                data = b""
                try:
                    while b"\r\n\r\n" not in data:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                except OSError:
                    pass
                self.requests.append(data)
                try:
                    conn.sendall(self.answer)
                except OSError:
                    pass

    def close(self) -> None:
        self._sock.close()


@pytest.fixture
def raw_server():
    server = RawServer()
    yield server
    server.close()
