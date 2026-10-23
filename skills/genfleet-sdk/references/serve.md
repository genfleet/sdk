# A2A Serve Module — Detailed Reference

The serve module (`genfleet.sdk.serve`) exposes any `AgentProtocol` as an A2A-compliant HTTP endpoint using FastAPI. It supports both synchronous request/response and streaming via Server-Sent Events (SSE).

## Prerequisites

Install the `[serve]` extra:

```bash
pip install "genfleet-sdk[serve] @ git+https://github.com/genfleet/sdk@dev"
```

This adds: `fastapi>=0.115`, `uvicorn[standard]>=0.34`, `sse-starlette>=2.0`.

If you import from `genfleet.sdk.serve` without these installed, you get a clear `ImportError` telling you what to install.

## Functions

### serve()

```python
def serve(
    agent: AgentProtocol,
    *,
    name: str = "agent",
    description: str = "",
    host: str = "0.0.0.0",
    port: int = 8000,
    public_url: str | None = None,
    accept_history: bool = False,
    accepted_input_types: Iterable[str] | None = None,
    output_types: Iterable[str] | None = None,
    max_body_bytes: int = 8 * 1024 * 1024,
) -> None:
```

Blocking function that starts a Uvicorn server. Ideal for scripts and local development.

### create_app()

```python
def create_app(
    agent: AgentProtocol,
    *,
    name: str = "agent",
    description: str = "",
    public_url: str | None = None,
    accept_history: bool = False,
    accepted_input_types: Iterable[str] | None = None,
    output_types: Iterable[str] | None = None,
    max_body_bytes: int = 8 * 1024 * 1024,
) -> FastAPI:
```

Returns a FastAPI app with the same options as `serve()`. Use when you need to:
- Add middleware (CORS, auth, logging)
- Mount under a larger application
- Run with a custom ASGI server (Hypercorn, Daphne)
- Write tests with `httpx.AsyncClient` + ASGI transport

### Options

- `public_url` - the URL the card advertises (`supportedInterfaces[0].url`). Must be an absolute http(s) URL with no query or fragment, else `ValueError`. Without it the card names each request's own URL, built from the caller's `Host` header (so any host can be advertised), and is sent with `Cache-Control: no-store`; with it, `max-age=300`. Set it whenever the app is reachable from outside.
- `accept_history` - forward caller-supplied prior turns (`metadata["genfleet.history"]`, or 0.x `params.history`) to the agent as its history. Off by default: whoever can call the app could otherwise put words in the agent's earlier turns. Turn it on only for a trusted caller that keeps the conversation itself. The `genfleet.history` key is removed from the metadata the agent sees either way; other request `metadata` passes through.
- `accepted_input_types` - MIME types (`"image/*"`, `"application/pdf"`, `"*/*"`) of the files the app takes besides text. Defaults to the agent's `accepted_input_types` attribute (`Agent(accepts=...)` sets it), else none. Given here, it applies to this app only: the agent is unchanged and `Agent.run` accepts what this app admitted. The card lists them in `defaultInputModes`.
- `output_types` - MIME types of files the agent may return, listed in the card's `defaultOutputModes` after `text/plain`. Defaults to the agent's `output_types` attribute, else none.
- `max_body_bytes` - the largest request body read (default 8 MiB). A larger one is refused before it is parsed with `-32600`; send big files by reference.

## Endpoints (A2A v1.0, JSON-RPC binding)

### GET /.well-known/agent-card.json

Returns the agent card (also served at the 0.x path `/.well-known/agent.json`
for one minor release):

```json
{
  "name": "my-agent",
  "description": "Does useful things",
  "supportedInterfaces": [
    {"url": "http://localhost:8000/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
  ],
  "version": "1.0.0",
  "capabilities": {"streaming": true, "pushNotifications": false, "extendedAgentCard": false},
  "defaultInputModes": ["text/plain"],      // plus accepted_input_types
  "defaultOutputModes": ["text/plain"],     // plus output_types
  "skills": []
}
```

### POST /

JSON-RPC 2.0. Send `A2A-Version: 1.0`; a version other than 0.x or 1.x is
refused (`-32009`).

#### SendMessage — synchronous

Collects the full agent output and returns a completed task.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "m1",
      "role": "ROLE_USER",
      "contextId": "user-123",
      "parts": [{"text": "Hello, agent!"}]
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "task": {
      "id": "…",
      "contextId": "user-123",
      "status": {"state": "TASK_STATE_COMPLETED"},
      "artifacts": [{"artifactId": "result", "parts": [{"text": "Hello! How can I help?"}]}]
    }
  }
}
```

#### Sessions

`contextId` becomes `AgentInput.metadata["session_id"]`, over any `session_id` in the request `metadata` (which is used only when there is no `contextId`, so a gateway in front can scope `contextId` per caller and metadata cannot route around it). `Agent` memory is keyed by `session_id`; without one each request is stateless. The 0.x `params.sessionId` is honoured only when metadata has no `session_id`.

A message may carry text, files, or both; a file-only request is a turn with an empty `message`. A part that is neither text nor a recognisable file or data part is `-32602`.

#### SendStreamingMessage — streaming (SSE)

Same request as `SendMessage`. Each SSE event is a JSON-RPC response whose
`result` is a `StreamResponse`:

1. Working: `{"statusUpdate": {"taskId": "…", "contextId": "…", "status": {"state": "TASK_STATE_WORKING"}}}`
2. Content chunks: `{"artifactUpdate": {"taskId": "…", "contextId": "…", "artifact": {"artifactId": "result", "parts": [{"text": "Hello"}]}, "append": false, "lastChunk": false}}` — later chunks have `"append": true`.
3. Tool events (when the agent reports them): a working `statusUpdate` with `metadata.tool_event`.
4. Completion: `{"statusUpdate": {…, "status": {"state": "TASK_STATE_COMPLETED"}}}`
5. Failure: `{"statusUpdate": {…, "status": {"state": "TASK_STATE_FAILED", "message": {"messageId": "…", "role": "ROLE_AGENT", "parts": [{"text": "The agent failed to handle this request. (error id: 9f2c1ab40e7d)"}]}}}}`

#### Other methods

`serve` keeps no tasks: `GetTask`, `ListTasks`, `CancelTask` and
`SubscribeToTask` answer `-32004` (UnsupportedOperation). Push notifications
are not supported (`-32003`), and there is no extended card (`-32007`).

The 0.x methods `tasks/send` and `tasks/sendSubscribe` still work for one
minor release.

## Error Codes

| Code | Meaning |
|------|---------|
| -32700 | Parse error (malformed JSON) |
| -32600 | Invalid request: body larger than `max_body_bytes` (or not a JSON object) |
| -32601 | Method not found |
| -32602 | Invalid params (no `message.parts`, malformed part, inline file over 1 MiB) |
| -32005 | Content type not supported: a file whose type the app does not accept; `data` names `unsupported` and `accepted` |
| -32603 | Internal error (agent raised an exception) |
| -32004 | Unsupported operation (task methods) |
| -32003 | Push notifications not supported |
| -32007 | No extended agent card |
| -32009 | A2A version not supported |

## Failure detail is not returned to the caller

When your agent raises, both `SendMessage` and `SendStreamingMessage` return a
fixed message plus a random **error id**. The exception — type, message,
traceback — goes to the `genfleet.sdk.serve` logger under that same id.

This is deliberate. `serve` puts your agent on the network, and provider
exceptions routinely carry request payloads, file paths, console URLs, and (on
OpenAI) an echo of the API key you submitted. Returning them would hand those
to whoever called you.

To debug a failure, take the error id from the response and grep your own logs:

```
grep 9f2c1ab40e7d agent.log
```

## Message Parsing and Files

The serve module maps every A2A part into `AgentInput.parts`:

- `{"text": "..."}` becomes a `TextPart`; the text parts are joined into `AgentInput.message`.
- `{"raw": "<base64>", "mediaType": "image/png", "filename": "a.png"}` (inline, at most 1 MiB decoded) or `{"url": "...", "mediaType": "..."}` (by reference) becomes the media part for that MIME type (`ImagePart`, `FilePart`, ...). The 0.x form is `{"type": "file", "file": {"bytes" | "uri", "mimeType", "name"}}`.
- `{"data": {...}}` (a data part) becomes an `application/json` file part.

A file whose type is not in the accepted types is refused with `-32005` before the agent runs. `serve` never fetches a `url`: the agent reads references through a `FileResolver` (`Agent(files=...)`), which is where access and expiry are checked.

Files the agent returns in `AgentOutput.parts` go out as file parts of the `result` artifact, beside the text (`raw` for inline bytes, `url` for references; 0.x callers get the `file` form).

`SendMessage` returns the whole reply (text and files only); `SendStreamingMessage` streams it and is the only method that relays tool events (from `Agent`). In both, `tool_calls` yielded by a custom `AgentProtocol` are **not executed and not returned**.

## Testing with httpx (unit tests)

```python
import httpx
from genfleet.sdk import AgentInput, AgentOutput
from genfleet.sdk.serve import create_app

class EchoAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content=f"Echo: {input.message}", done=True)

app = create_app(EchoAgent(), name="test")

async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
    resp = await client.post("http://test/", json={
        "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
        "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hi"}]}}
    })
    data = resp.json()
    assert data["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
```
