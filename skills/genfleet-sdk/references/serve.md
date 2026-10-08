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
) -> FastAPI:
```

Returns a FastAPI app. Use when you need to:
- Add middleware (CORS, auth, logging)
- Mount under a larger application
- Run with a custom ASGI server (Hypercorn, Daphne)
- Write tests with `httpx.AsyncClient` + ASGI transport

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
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
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

`contextId` becomes `AgentInput.metadata["session_id"]`, over any
`session_id` in request `metadata` (which is used only when there is no
`contextId`). A message with no text part is refused (`-32005`).

Options on `create_app` / `serve`:

- `public_url="https://agents.example.com/bot/"` — the URL the card
  advertises. Without it the card names each request's own URL, built from the
  caller's `Host` header (so any host can be advertised), and is sent with
  `Cache-Control: no-store`; with it, `max-age=300`.
- `accept_history=True` — forward caller-supplied prior turns
  (`metadata["genfleet.history"]`, or 0.x `params.history`) to the agent.
  Off by default: the key is stripped and ignored, because whoever can call the
  app could otherwise put words in the agent's earlier turns. Turn it on only
  for a trusted caller that keeps the conversation itself. Request `metadata` passes through to the agent; the
`metadata["genfleet.history"]` key is always removed (see `accept_history`).

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
| -32601 | Method not found |
| -32602 | Invalid params (e.g. no `message.parts`) |
| -32005 | Content type not supported (no text part) |
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

## Message Parsing

The serve module extracts text from A2A message parts:

```python
# From the request params:
params.message.parts → [{"text": "..."}]
# Concatenated into: "..." and passed as AgentInput(message="...")
```

Only `text` parts are extracted. Non-text parts (file, data) are ignored by the default serve module.

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
