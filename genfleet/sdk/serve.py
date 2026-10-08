from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

from ._redaction import safe_arguments
from .protocol import AgentProtocol
from .schemas import TOOL_EVENT_KEY, AgentInput, AgentOutput, Message

log = logging.getLogger("genfleet.sdk.serve")

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse
except ImportError as exc:
    raise ImportError(
        "The serve helper requires the serve extra. "
        "Install it with: pip install 'genfleet-sdk[serve]'"
    ) from exc


#: A2A v1.0 (the JSON-RPC binding, A2A v1.0.1 proto in ProtoJSON form): the
#: card path, version header and the error codes this server answers with.
PROTOCOL_VERSION = "1.0"
VERSION_HEADER = "A2A-Version"
CARD_PATH = "/.well-known/agent-card.json"
#: The 0.x card path, served as an alias for one minor release.
LEGACY_CARD_PATH = "/.well-known/agent.json"
#: Request ``metadata`` key for the turn's prior messages (``Message`` dicts).
#: Not an A2A field; a conforming peer ignores it. Removed from the metadata
#: the agent sees.
HISTORY_METADATA_KEY = "genfleet.history"
_RESULT_ARTIFACT = "result"

_INVALID_PARAMS = -32602
_CONTENT_TYPE_NOT_SUPPORTED = -32005
_UNSUPPORTED_OPERATION = -32004
_PUSH_NOT_SUPPORTED = -32003
_EXTENDED_CARD_NOT_CONFIGURED = -32007
_VERSION_NOT_SUPPORTED = -32009
_TASK_STORE_METHODS = frozenset({"GetTask", "ListTasks", "CancelTask", "SubscribeToTask"})
_PUSH_METHODS = frozenset({
    "CreateTaskPushNotificationConfig",
    "GetTaskPushNotificationConfig",
    "ListTaskPushNotificationConfigs",
    "DeleteTaskPushNotificationConfig",
})


def _history(raw: Any) -> list[Message]:
    if not isinstance(raw, list):
        return []
    try:
        return [Message(**m) for m in raw]
    except Exception:
        log.warning("Ignoring invalid history in A2A request")
        return []


def _build_v1_input(params: dict[str, Any]) -> AgentInput | None:
    """A v1.0 ``SendMessageRequest`` as the agent's turn, or ``None`` if malformed.

    The message's text parts are the turn's text; request ``metadata`` passes
    through; the message's ``contextId`` is ``metadata["session_id"]``, over
    any ``session_id`` in the metadata (a gateway in front may scope
    ``contextId`` per caller; metadata must not route around it), which is
    used only when there is no ``contextId``; ``metadata["genfleet.history"]``
    is forwarded as the turn's history and removed from the metadata.
    """
    message = params.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("parts"), list):
        return None
    text = "".join(p["text"] for p in message["parts"] if isinstance(p, dict) and isinstance(p.get("text"), str))
    metadata = params.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    history = _history(metadata.pop(HISTORY_METADATA_KEY, None))
    context_id = message.get("contextId")
    if isinstance(context_id, str) and context_id:
        metadata["session_id"] = context_id
    return AgentInput(message=text, history=history, metadata=metadata)


def _no_text(params: dict[str, Any]) -> bool:
    """Parts, but no text part: the agent would run on an empty turn."""
    parts = params["message"]["parts"]
    return bool(parts) and not any(isinstance(p, dict) and isinstance(p.get("text"), str) for p in parts)


def _v1_ids(params: dict[str, Any]) -> tuple[str, str]:
    message = params.get("message") or {}
    task_id, context_id = message.get("taskId"), message.get("contextId")
    return (
        task_id if isinstance(task_id, str) and task_id else str(uuid.uuid4()),
        context_id if isinstance(context_id, str) and context_id else str(uuid.uuid4()),
    )


def _supported_version(value: str | None) -> bool:
    """1.x, or empty/0.x (a 0.3 client sends none; the 0.x methods stay one release)."""
    return not value or value.strip().split(".", 1)[0] in ("0", "1")


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

    return AgentInput(message=message_text, history=_history(params.get("history")), metadata=metadata)


#: What a caller is told when the agent raised. Deliberately fixed text: the
#: alternative — scanning the exception for provider markers and redacting
#: matches — is a denylist, and a denylist is a list of the leaks someone
#: remembered. The next provider's error format is exposed by default.
OPAQUE_FAILURE_MESSAGE = "The agent failed to handle this request."


def _opaque_failure(context: str) -> str:
    """Log the active exception; return what the caller is told instead.

    `serve` is the public package's network surface, running on machines the
    platform does not operate, so the exception text never crosses it: provider
    errors carry request payloads, file paths, console URLs, and — observed on
    OpenAI — an echo of the submitted API key (sdk#20). The detail stays in the
    operator's own log, reachable from the id quoted back to the caller.

    Both failure paths go through here so the two cannot drift into saying
    different amounts about the same failure.
    """
    error_id = uuid.uuid4().hex[:12]
    log.exception("%s [error_id=%s]", context, error_id)
    return f"{OPAQUE_FAILURE_MESSAGE} (error id: {error_id})"


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


#: Cap on a relayed tool result's ``output``, matching the platform's A2A
#: server: enough for the receiver to see a long output was cut and shape its
#: own preview, while a megabyte result stays off the wire.
TOOL_EVENT_OUTPUT_MAX_CHARS = 4000


#: What a failed tool result says on the wire. A failed call's output is
#: error text (an MCP failure carries the exception's message), and `serve`
#: never sends exception text to its caller (sdk#20).
TOOL_FAILED_OUTPUT = "The tool failed."


def _relayable_tool_event(output: AgentOutput) -> dict[str, Any] | None:
    """The tool event on ``output`` (``TOOL_EVENT_KEY``), made safe for the wire.

    A call's ``arguments`` are redacted and capped at 8 KB (``_redaction``),
    a failed result's ``output`` is replaced by ``TOOL_FAILED_OUTPUT``, and a
    successful result's ``output`` is sent as the tool returned it, cut to
    ``TOOL_EVENT_OUTPUT_MAX_CHARS``.
    """
    event = output.metadata.get(TOOL_EVENT_KEY)
    if not isinstance(event, dict) or event.get("type") not in ("tool_call", "tool_result"):
        return None
    relayed = dict(event)
    if relayed["type"] == "tool_call":
        relayed["arguments"] = safe_arguments(relayed.get("arguments"))
    elif relayed.get("ok") is not True:
        relayed["output"] = TOOL_FAILED_OUTPUT
    elif isinstance(text := relayed.get("output"), str):
        relayed["output"] = text[:TOOL_EVENT_OUTPUT_MAX_CHARS]
    return relayed


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
            # A tool event rides on a non-final ``working`` status update under
            # ``metadata.tool_event``, as the platform's A2A server sends it;
            # a reader that does not know the key sees one more progress event.
            tool_event = _relayable_tool_event(output)
            if tool_event is not None:
                yield _event({
                    "id": task_id,
                    "status": {"state": "working"},
                    "final": False,
                    "metadata": {TOOL_EVENT_KEY: tool_event},
                })
                continue
            if output.content:
                yield _event({
                    "id": task_id,
                    "artifact": {"parts": [{"type": "text", "text": output.content}]},
                    "final": False,
                })
            if output.done:
                yield _event({"id": task_id, "status": {"state": "completed"}, "final": True})
                return
    except Exception:
        # The same leak as `tasks/send`, on the streaming path. The flag
        # (sdk#20) named only the other one; both reach the network.
        message = _opaque_failure("streaming failed")
        yield _event({
            "id": task_id,
            "status": {
                "state": "failed",
                # A2A types `TaskStatus.message` as a Message, not a string.
                # It was emitted as a bare string here, so a spec-conforming
                # reader — including the platform's own A2A client, which does
                # `status_msg.get("parts")` — raised AttributeError instead of
                # reporting the failure. That made the error id unreachable on
                # exactly the path this fix exists to cover.
                "message": {
                    "role": "agent",
                    "parts": [{
                        "type": "text",
                        "text": message,
                    }],
                },
            },
            "final": True,
        })


async def _v1_stream(
    agent: AgentProtocol, agent_input: AgentInput, req_id: Any, task_id: str, context_id: str
) -> AsyncIterator[dict[str, str]]:
    """``SendStreamingMessage``: working, appended ``result`` chunks, then completed or failed."""

    def _event(payload: dict[str, Any]) -> dict[str, str]:
        return {"data": json.dumps({"jsonrpc": "2.0", "id": req_id, "result": payload})}

    def _status(state: str, *, text: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, str]:
        status: dict[str, Any] = {"state": state}
        if text:
            status["message"] = {"messageId": str(uuid.uuid4()), "role": "ROLE_AGENT", "parts": [{"text": text}]}
        update: dict[str, Any] = {"taskId": task_id, "contextId": context_id, "status": status}
        if metadata:
            update["metadata"] = metadata
        return _event({"statusUpdate": update})

    chunks = 0

    def _chunk(text: str, *, last: bool) -> dict[str, str]:
        nonlocal chunks
        chunks += 1
        return _event({"artifactUpdate": {
            "taskId": task_id,
            "contextId": context_id,
            "artifact": {"artifactId": _RESULT_ARTIFACT, "parts": [{"text": text}]},
            "append": chunks > 1,
            "lastChunk": last,
        }})

    yield _status("TASK_STATE_WORKING")
    try:
        async for output in agent.run(agent_input):
            tool_event = _relayable_tool_event(output)
            if tool_event is not None:
                yield _status("TASK_STATE_WORKING", metadata={TOOL_EVENT_KEY: tool_event})
                continue
            if output.content:
                yield _chunk(output.content, last=output.done)
            if output.done:
                yield _status("TASK_STATE_COMPLETED")
                return
        yield _status("TASK_STATE_COMPLETED")
    except Exception:
        yield _status("TASK_STATE_FAILED", text=_opaque_failure("streaming failed"))


def create_app(
    agent: AgentProtocol,
    *,
    name: str = "agent",
    description: str = "",
) -> FastAPI:
    """Build a FastAPI app that serves an agent over A2A v1.0 (JSON-RPC binding).

    The card is at ``/.well-known/agent-card.json`` (and, for one minor
    release, ``/.well-known/agent.json``). ``SendMessage`` returns a completed
    task with the reply as artifact ``result``; ``SendStreamingMessage``
    streams it. The server keeps no tasks, so ``GetTask``, ``ListTasks``,
    ``CancelTask`` and ``SubscribeToTask`` answer UnsupportedOperation, and
    push notifications are not supported. The 0.x ``tasks/send`` and
    ``tasks/sendSubscribe`` still work for one minor release.

    ``SendStreamingMessage`` (and ``tasks/sendSubscribe``) relays each tool call and result the agent reports
    (``metadata.tool_event``) on a ``working`` status update. Tool calls go out
    with their arguments redacted and capped at 8 KB, and a failed result says
    only "The tool failed.". **A successful tool's output is sent raw**, up to
    4000 characters, to whoever calls this app: a tool that reads a database
    row, a file or an internal API exposes that data to a direct caller. On the
    platform the engine scrubs it before a user sees it; when you host ``serve``
    yourself, nothing does. ``SendMessage`` returns only the agent's text.
    """
    app = FastAPI(title=name, docs_url=None, redoc_url=None)

    def card(base_url: str) -> dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "supportedInterfaces": [
                {"url": base_url, "protocolBinding": "JSONRPC", "protocolVersion": PROTOCOL_VERSION}
            ],
            "version": "1.0.0",
            "capabilities": {"streaming": True, "pushNotifications": False, "extendedAgentCard": False},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [],
        }

    @app.get(CARD_PATH)
    async def agent_card(request: Request) -> dict[str, Any]:
        return card(str(request.base_url))

    # The 0.x path, for one minor release: the same card.
    app.get(LEGACY_CARD_PATH)(agent_card)

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

        if not _supported_version(request.headers.get(VERSION_HEADER)):
            return _jsonrpc_error(req_id, _VERSION_NOT_SUPPORTED, "Supported A2A versions: 1.0.")
        if method in ("SendMessage", "SendStreamingMessage"):
            v1_input = _build_v1_input(params)
            if v1_input is None:
                return _jsonrpc_error(req_id, _INVALID_PARAMS, "Invalid request parameters: check message.")
            if _no_text(params):
                return _jsonrpc_error(req_id, _CONTENT_TYPE_NOT_SUPPORTED, "Only text parts are supported.")
            task_id, context_id = _v1_ids(params)
            if method == "SendStreamingMessage":
                return EventSourceResponse(_v1_stream(agent, v1_input, req_id, task_id, context_id))
            try:
                output = await _collect_output(agent, v1_input)
            except Exception:
                return _jsonrpc_error(req_id, -32603, _opaque_failure("SendMessage failed"))
            return _jsonrpc_result(req_id, {"task": {
                "id": task_id,
                "contextId": context_id,
                "status": {"state": "TASK_STATE_COMPLETED"},
                "artifacts": (
                    [{"artifactId": _RESULT_ARTIFACT, "parts": [{"text": output.content}]}] if output.content else []
                ),
            }})
        if method in _TASK_STORE_METHODS:
            return _jsonrpc_error(req_id, _UNSUPPORTED_OPERATION, "This agent keeps no tasks.")
        if method in _PUSH_METHODS:
            return _jsonrpc_error(req_id, _PUSH_NOT_SUPPORTED, "Push notifications are not supported.")
        if method == "GetExtendedAgentCard":
            return _jsonrpc_error(req_id, _EXTENDED_CARD_NOT_CONFIGURED, "No extended agent card.")

        # 0.x, for one minor release.
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
            except Exception:
                message = _opaque_failure("tasks/send failed")
                return _jsonrpc_error(req_id, -32603, message)

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
    """Run an agent as an A2A server (blocking).

    The app is ``create_app``'s; its docstring says what tool events a caller
    sees, including that a successful tool's output is sent raw.
    """
    import uvicorn

    app = create_app(agent, name=name, description=description)
    uvicorn.run(app, host=host, port=port)
