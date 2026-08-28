# A2A Serve Module — Detailed Reference

The serve module (`fleet.sdk.serve`) exposes any `AgentProtocol` as an A2A-compliant HTTP endpoint using FastAPI. It supports both synchronous request/response and streaming via Server-Sent Events (SSE).

## Prerequisites

Install the `[serve]` extra:

```bash
pip install withfleet-sdk[serve]
```

This adds: `fastapi>=0.115`, `uvicorn[standard]>=0.34`, `sse-starlette>=2.0`.

If you import from `fleet.sdk.serve` without these installed, you get a clear `ImportError` telling you what to install.

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

## Endpoints

### GET /.well-known/agent.json

Returns the agent card:

```json
{
  "name": "my-agent",
  "description": "Does useful things",
  "url": "/",
  "capabilities": {"streaming": true}
}
```

### POST /

JSON-RPC 2.0 endpoint. Supports two methods:

#### tasks/send — Synchronous

Collects the full agent output and returns it as a complete response.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tasks/send",
  "params": {
    "id": "task-123",
    "message": {
      "role": "user",
      "parts": [{"type": "text", "text": "Hello, agent!"}]
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
    "id": "task-123",
    "status": {"state": "completed"},
    "artifacts": [
      {"parts": [{"type": "text", "text": "Hello! How can I help?"}]}
    ]
  }
}
```

#### tasks/sendSubscribe — Streaming (SSE)

Returns a Server-Sent Events stream with incremental updates.

**Request:** Same format as `tasks/send` but with method `tasks/sendSubscribe`.

**SSE Events:**

1. Working status:
```
data: {"jsonrpc":"2.0","id":1,"result":{"id":"task-123","status":{"state":"working"},"final":false}}
```

2. Content chunks:
```
data: {"jsonrpc":"2.0","id":1,"result":{"id":"task-123","artifact":{"parts":[{"type":"text","text":"Hello"}]},"final":false}}
```

3. Completion:
```
data: {"jsonrpc":"2.0","id":1,"result":{"id":"task-123","status":{"state":"completed"},"final":true}}
```

4. Error (if agent fails):
```
data: {"jsonrpc":"2.0","id":1,"result":{"id":"task-123","status":{"state":"failed","message":{"role":"agent","parts":[{"type":"text","text":"The agent failed to handle this request. (error id: 9f2c1ab40e7d)"}]}},"final":true}}
```

`status.message` is a Message object, like any other A2A message — not a bare
string.

## Error Codes

| Code | Meaning |
|------|---------|
| -32700 | Parse error (malformed JSON) |
| -32601 | Method not found (not `tasks/send` or `tasks/sendSubscribe`) |
| -32603 | Internal error (agent raised an exception) |

## Failure detail is not returned to the caller

When your agent raises, both `tasks/send` and `tasks/sendSubscribe` return a
fixed message plus a random **error id**. The exception — type, message,
traceback — goes to the `fleet.sdk.serve` logger under that same id.

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
params.message.parts → [{"type": "text", "text": "..."}]
# Concatenated into: "..." and passed as AgentInput(message="...")
```

Only `text` parts are extracted. Non-text parts (file, data) are ignored by the default serve module.

## Testing with httpx (unit tests)

```python
import httpx
from fleet.sdk import AgentInput, AgentOutput
from fleet.sdk.serve import create_app

class EchoAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content=f"Echo: {input.message}", done=True)

app = create_app(EchoAgent(), name="test")

async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
    resp = await client.post("http://test/", json={
        "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
        "params": {"id": "t1", "message": {"role": "user", "parts": [{"type": "text", "text": "hi"}]}}
    })
    data = resp.json()
    assert data["result"]["status"]["state"] == "completed"
```
