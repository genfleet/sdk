# Writing a connector (a marketplace tool)

A connector is a tool you publish once, for many workspaces to use. Each
workspace installs it with its own settings and its own credentials, and
agents name it in their manifest's `[tools]` table. The same published code
then runs for every workspace, against that workspace's account.

This guide builds `@acme/tickets`, a tool that opens a ticket in a helpdesk.

## Layout

```text
acme-tickets/
  genfleet.toml        # the tool manifest
  acme_tickets.py      # the tool
  local_provider.py    # optional: lets load_tools() find it on your machine
  pyproject.toml       # optional: for local installs and tests
  tests/
    test_tool.py
```

`gf tools push acme-tickets` uploads this directory. The manifest must be at
its root.

## The manifest

```toml
# acme-tickets/genfleet.toml
[tool]
slug = "@acme/tickets"
version = "1.0.0"
kind = "http"
description = "Create a ticket in Acme Helpdesk."
entry = "acme_tickets:create_ticket"
sensitive = false

[config.base_url]
type = "string"
required = true
description = "Your helpdesk URL, e.g. https://acme.example.com"

[config.queue]
type = "string"
default = "support"
description = "Queue new tickets land in"

[secrets.api_token]
description = "Helpdesk API token"
```

- `[config.*]` are settings the installing workspace fills in. `[secrets.*]` are
  credentials it binds. Secrets are declared by name only: a `value` key is
  rejected, because the manifest is public.
- `entry` is `module:callable`, relative to this directory.
- `sensitive = true` would make every call wait for a workspace owner's
  approval (see [approvals](approvals-and-messaging.md)).

Every key is described in the [manifest reference](manifest.md#tool-manifest).

## The tool and `ToolContext`

A tool gets its settings and secrets by declaring a parameter annotated
`ToolContext`. That parameter is left out of the tool's schema, so the model
never sees it and can't fill it in. The SDK builds it on every call.

```python
# acme-tickets/acme_tickets.py
import asyncio
import json
import urllib.request

from genfleet.sdk import ToolContext, tool


@tool(audiences=["operator", "customer"])
async def create_ticket(subject: str, body: str, ctx: ToolContext) -> str:
    """Open a helpdesk ticket and return its id."""
    request = urllib.request.Request(
        f"{ctx.config['base_url'].rstrip('/')}/api/tickets",
        data=json.dumps({
            "subject": subject,
            "body": body,
            "queue": ctx.config.get("queue", "support"),
        }).encode(),
        headers={
            "Authorization": f"Bearer {ctx.secret('api_token')}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    # urllib blocks; run it off the event loop so other turns keep going.
    return await asyncio.to_thread(_send, request)


def _send(request: urllib.request.Request) -> str:
    with urllib.request.urlopen(request, timeout=20) as response:
        return f"created ticket {json.load(response)['id']}"
```

- `ctx.config` is a read-only mapping of the settings the tool was given.
  Read optional ones with a fallback, as `queue` is read above.
- `ctx.secret("api_token")` returns the value, or raises `ToolSecretError`
  naming the secret. The error never includes a value.
- Accept `ToolContext`, `ToolContext | None` or `Optional[ToolContext]`. Any
  other hint naming `ToolContext` raises `TypeError` when the tool is wrapped.
- `audiences` is the tool's default audience. Without it, only operators get the
  tool. The installing workspace's own setting wins. See "Operators and
  customers" in the [README](../../README.md#operators-and-customers).
- A synchronous tool runs on the agent's event loop, so a blocking call in it
  holds up every other turn. Make the tool `async` and push blocking work to
  `asyncio.to_thread`, as above.

A tool loaded with `load_tools()` runs in the agent's own process, so its
settings and secrets are readable by the agent's code and by every other tool
in that process. Each tool's values are kept under their own name so tools
don't collide, but that is not an isolation boundary. Treat everything you
read from `ctx` as something the agent's code could also see.

## What a hosted agent can import

On the platform an agent, and every tool it loads, can import **the SDK and
the Python standard library only**. Your `pyproject.toml` dependencies are
installed on your machine, not on the platform. That's why the tool above uses
`urllib.request` instead of `requests` or `httpx`.

Heavier work (PDF parsing, OCR, spreadsheets, data libraries, a vendor SDK) has
to run outside the agent's process. Two ways to do it:

- **An MCP server.** Run the work behind an MCP server and connect the agent to
  it with `Agent(mcps=[...])`.
- **An HTTP service.** Run the work behind your own HTTP API and write a small
  connector, like the one above, that calls it with `urllib` and the
  workspace's credentials.

Either way, the agent's code stays plain Python with no dependencies.

## Testing locally

### Supplying settings and secrets

On your machine, `ToolContext` reads the same two environment variables the
platform sets, both holding JSON objects, under a namespace derived from the
slug:

```python
from genfleet.sdk import tool_env_namespace

tool_env_namespace("@acme/tickets")   # "GENFLEET_TOOL_ACME__TICKETS"
```

```bash
export GENFLEET_TOOL_ACME__TICKETS__CONFIG='{"base_url": "https://acme.example.com"}'
export GENFLEET_TOOL_ACME__TICKETS__SECRETS='{"api_token": "…"}'
```

Locally only, a secret missing from `…__SECRETS` falls back to an environment
variable with the secret's own name, `api_token` and then `API_TOKEN`:

```bash
export API_TOKEN=…
```

On the platform there is no fallback. A secret the workspace didn't bind isn't
available.

The namespace comes from the slug: the `@` is dropped, `/` becomes `__`, any
other character that isn't a letter, digit or `_` becomes `_`, and the result
is uppercased.

### Unit tests

`ToolContext` needs the tool's slug to find its namespace. `load_tools()`
attaches it, and in a test `with_slug` does the same:

```python
# acme-tickets/tests/test_tool.py
import asyncio
import io
import json
import urllib.request
from pathlib import Path

import pytest

from acme_tickets import create_ticket
from genfleet.sdk import ToolCall, ToolSecretError, load_tool_manifest, tool_env_namespace

SLUG = "@acme/tickets"
NS = tool_env_namespace(SLUG)  # "GENFLEET_TOOL_ACME__TICKETS"
CALL = ToolCall(id="1", name="create_ticket", arguments={"subject": "Hi", "body": "Help"})


def run(tool):
    return asyncio.run(tool.call(CALL))


def test_manifest_is_valid():
    manifest = load_tool_manifest(Path(__file__).parent.parent)
    assert manifest.slug == SLUG
    assert set(manifest.secrets) == {"api_token"}


def test_the_model_never_sees_the_context():
    assert set(create_ticket.schema().parameters["properties"]) == {"subject", "body"}


def test_creates_a_ticket(monkeypatch):
    monkeypatch.setenv(f"{NS}__CONFIG", json.dumps({"base_url": "https://helpdesk.test"}))
    monkeypatch.setenv(f"{NS}__SECRETS", json.dumps({"api_token": "t0ken"}))
    sent = {}

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["auth"] = request.get_header("Authorization")
        return io.BytesIO(b'{"id": 42}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # load_tools() gives each tool its slug; do the same here.
    assert run(create_ticket.with_slug(SLUG)) == "created ticket 42"
    assert sent == {"url": "https://helpdesk.test/api/tickets", "auth": "Bearer t0ken"}


def test_a_missing_secret_is_named(monkeypatch):
    monkeypatch.setenv(f"{NS}__CONFIG", json.dumps({"base_url": "https://helpdesk.test"}))
    for name in (f"{NS}__SECRETS", "api_token", "API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ToolSecretError, match="api_token"):
        run(create_ticket.with_slug(SLUG))
```

### Using a tool from an agent locally

An agent gets the tools its manifest names from `load_tools()`. On your
machine, `load_tools()` looks them up among installed packages that advertise
tools under the `genfleet.tools` entry-point group. The entry point names a
function that returns `{slug: object with a load() method}`, and `load()`
returns the tool:

```python
# acme-tickets/local_provider.py
"""Lets `load_tools()` find this tool on your machine. Not used on the platform."""

from acme_tickets import create_ticket


class _Installed:
    def __init__(self, fn):
        self._fn = fn

    def load(self):
        return self._fn


def discover():
    return {"@acme/tickets": _Installed(create_ticket)}
```

```toml
# acme-tickets/pyproject.toml
[project]
name = "acme-tickets"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = ["genfleet-sdk @ git+https://github.com/genfleet/sdk@dev"]

[project.entry-points."genfleet.tools"]
acme_tickets = "local_provider:discover"

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["acme_tickets", "local_provider"]
```

Install it into the agent's environment (for example
`uv pip install -e ../acme-tickets` from the agent's project), set the
variables above, and the agent's `load_tools()` returns `create_ticket` for
`"@acme/tickets" = "1.x"`. Locally, `load_tools()` ignores versions and
digests, so you can change the tool and the agent together. A slug nobody
provides raises `ToolResolutionError`, naming the agent, the slug, and the
slugs that are installed.

For an agent's unit tests you don't need to install anything: pass
`load_tools(..., resolver=...)` an object whose `resolve(ref)` returns a stand-in.
The [agent factory guide](agent-factory.md#testing-it) shows one.

## Publishing

```bash
gf tools digest acme-tickets                     # the digest, computed offline
gf tools push acme-tickets                       # validate, archive, store
gf tools push acme-tickets --tool-id <tool id>   # store and submit for review
```

`gf tools push` validates the manifest before uploading anything. It archives
the directory deterministically and checks that the registry stored the same
digest. Without `--tool-id` the archive is only stored: no agent can name it
yet. With `--tool-id`, it's submitted for review as the `version` in your
manifest.

What happens next:

1. **Review.** A Genfleet reviewer approves the submission before anything can
   use it. Only then does the slug resolve: an agent's `[tools]` spec can match
   this version from that point on.
2. **Pinning.** When an agent that names `@acme/tickets` is submitted, its spec
   is resolved to an exact version and digest. The agent keeps running that
   version until the agent is submitted again. See [what gets pinned](manifest.md#what-gets-pinned).
3. **Mounting.** On the platform the reviewed archive is mounted next to the
   agent, and `load_tools()` imports `entry` from it. Until the agent's own
   submission is approved, the agent runs without the tool.
4. **Settings and secrets.** Each workspace that installs the tool fills in its
   `[config]` and binds its `[secrets]` in the tool's settings. A workspace that
   installs a marketplace *agent* that uses your tool also has to grant that
   agent the tool's secrets. Until it does, `ctx.secret(...)` raises
   `ToolSecretError`, saying to bind the secret in the tool's settings and
   grant it to the agent. An exception from a tool ends the agent's turn, so
   if you want the model to explain the problem instead, catch it and return
   the message:

   ```python
   try:
       token = ctx.secret("api_token")
   except ToolSecretError as exc:
       return f"Not configured: {exc}"
   ```

   Never ask the user to paste a credential into the chat.

To ship a fix, bump `version`, push, and submit again. Agents whose spec
matches the new version (`"1.x"` matches `1.0.1`) get it when they are next
pinned.
