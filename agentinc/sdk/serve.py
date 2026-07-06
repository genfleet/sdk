from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

from .protocol import AgentProtocol
from .schemas import AgentInput, AgentOutput, Message

log = logging.getLogger("agentinc.sdk.serve")

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse
except ImportError as exc:
    raise ImportError(
        "The serve helper requires the serve extra. "
        "Install it with: pip install 'agentinc-sdk[serve]'"
    ) from exc


def _build_agent_input(message_text: str, params: dict[str, Any]) -> AgentInput:
    """Build an AgentInput from A2A params, forwarding session and state.

    - ``params.metadata`` (dict) is passed through as ``AgentInput.metadata``.
    - ``params.sessionId`` (A2A-spec field) populates ``metadata["session_id"]``
      unless the caller already set that key explicitly.
    - ``params.history`` is forwarded when it validates as Message models;
      invalid history is ignored rather than failing the request.
    """
    metadata = params.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}

    session_id = params.get("sessionId")
    if session_id and "session_id" not in metadata:
        metadata["session_id"] = session_id

    history: list[Message] = []
    raw_history = params.get("history")
    if isinstance(raw_history, list):
        try:
            history = [Message(**m) for m in raw_history]
        except Exception:
            log.warning("Ignoring invalid history in A2A request")
            history = []

    return AgentInput(message=message_text, history=history, metadata=metadata)


def _jsonrpc_error(req_id: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    })


def _jsonrpc_result(req_id: Any, result: Any) -> JSONResponse:
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    })


async def _collect_output(agent: AgentProtocol, agent_input: AgentInput) -> AgentOutput:
    content_parts: list[str] = []
    last_output: AgentOutput | None = None
    async for output in agent.run(agent_input):
        last_output = output
        if output.content:
            content_parts.append(output.content)
        if output.done:
            break
    if last_output is None:
        return AgentOutput(content="", done=True)
    return AgentOutput(
        content="".join(content_parts),
        tool_calls=last_output.tool_calls,
        done=True,
        metadata=last_output.metadata,
    )


async def _stream_output(
    agent: AgentProtocol, agent_input: AgentInput, req_id: Any, task_id: str
) -> AsyncIterator[dict[str, str]]:
    def _event(payload: dict[str, Any]) -> dict[str, str]:
        return {"data": json.dumps({"jsonrpc": "2.0", "id": req_id, "result": payload})}

    yield _event({"id": task_id, "status": {"state": "working"}, "final": False})

    try:
        async for output in agent.run(agent_input):
            if output.content:
                yield _event({
                    "id": task_id,
                    "artifact": {"parts": [{"type": "text", "text": output.content}]},
                    "final": False,
                })
            if output.done:
                yield _event({"id": task_id, "status": {"state": "completed"}, "final": True})
                return
    except Exception as exc:
        log.exception("streaming failed")
        yield _event({
            "id": task_id,
            "status": {"state": "failed", "message": str(exc)},
            "final": True,
        })


def create_app(
    agent: AgentProtocol,
    *,
    name: str = "agent",
    description: str = "",
) -> FastAPI:
    """Build a FastAPI app that serves an agent over the A2A protocol."""
    app = FastAPI(title=name, docs_url=None, redoc_url=None)

    card = {
        "name": name,
        "description": description,
        "url": "/",
        "capabilities": {"streaming": True},
    }

    @app.get("/.well-known/agent.json")
    async def agent_card() -> dict[str, Any]:
        return card

    @app.post("/")
    async def jsonrpc(request: Request):  # type: ignore[return]
        try:
            body = await request.json()
        except Exception:
            return _jsonrpc_error(None, -32700, "Parse error")
        if not isinstance(body, dict):
            return _jsonrpc_error(None, -32600, "Invalid Request")

        req_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})
        if not isinstance(params, dict):
            params = {}
        message_text = ""
        msg = params.get("message", {})
        if isinstance(msg, dict):
            parts = msg.get("parts", [])
            message_text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
        elif isinstance(msg, str):
            message_text = msg

        agent_input = _build_agent_input(message_text, params)
        task_id = params.get("id", str(uuid.uuid4()))

        if method == "tasks/send":
            try:
                output = await _collect_output(agent, agent_input)
            except Exception as exc:
                log.exception("tasks/send failed")
                return _jsonrpc_error(req_id, -32603, str(exc))

            return _jsonrpc_result(req_id, {
                "id": task_id,
                "status": {"state": "completed"},
                "artifacts": [{"parts": [{"type": "text", "text": output.content or ""}]}]
                if output.content else [],
            })

        if method == "tasks/sendSubscribe":
            return EventSourceResponse(_stream_output(agent, agent_input, req_id, task_id))

        return _jsonrpc_error(req_id, -32601, f"Unknown method: {method}")

    return app


def serve(
    agent: AgentProtocol,
    *,
    name: str = "agent",
    description: str = "",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Run an agent as an A2A server (blocking)."""
    import uvicorn

    app = create_app(agent, name=name, description=description)
    uvicorn.run(app, host=host, port=port)
