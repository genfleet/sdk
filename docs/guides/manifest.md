# `genfleet.toml` reference

Every agent and every marketplace tool has a `genfleet.toml`. An agent's
manifest says what the agent is and which tools it depends on. A tool's
manifest says what the tool is and which settings and credentials it needs to
run. The SDK parses both, so you can check a manifest offline:

```python
from genfleet.sdk import load_agent_manifest, load_tool_manifest

agent = load_agent_manifest("support_agent")   # the directory holding genfleet.toml
tool = load_tool_manifest("acme-tickets")
```

Both raise `ManifestError` when the file is missing, isn't valid TOML, or has a
bad value. The message starts with the file's path, and `.path` and `.problem`
hold the two parts. `gf agents push` and `gf tools push` run the same parser
before they upload anything, so a bad manifest never reaches the registry.

## Agent manifest

The agent's `genfleet.toml` sits inside the agent's Python package, next to
the module that holds `build_agent`:

```text
support-agent/
  pyproject.toml
  support_agent/
    __init__.py
    genfleet.toml
    agent.py
```

```toml
[agent]
name = "support"
kind = "code"
description = "Answers order questions and opens helpdesk tickets."
entry = "support_agent.agent:build_agent"
audiences = ["operator", "customer"]

[tools]
"@acme/tickets" = "1.x"

[mcps]
"@acme/files" = { spec = "*" }
```

### `[agent]`

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `name` | yes | | The agent's name. `gf agents push --create` uses it as the agent's name in your workspace unless you pass `--name`. |
| `kind` | no | `"code"` | One of `"code"`, `"declarative"`, `"native_sdk"`. See below. |
| `description` | no | `""` | A one-line description. `gf agents push --create` uses it as the agent's description. |
| `entry` | no | `"agent:build_agent"` | `module:callable` for the agent factory. See the [agent factory guide](agent-factory.md). |
| `audiences` | no | operators only | Who the agent serves on channels: a list of `"operator"` (your workspace's staff) and/or `"customer"` (your customers). An empty list is an error: leave the key out instead. Duplicates are dropped. |

**`kind`.** The parser accepts exactly these three values and rejects any
other. It doesn't attach behaviour to them. An agent written as a Python
package with a `build_agent` factory, like every example in this repository,
is `"code"`, which is the default.

**`audiences`.** When the key is absent, the platform serves the agent to
operators only. Add `"customer"` before you connect the agent to a channel your
customers write to. Which *tools* each audience gets is decided per tool; see
"Operators and customers" in the [README](../../README.md#operators-and-customers).

### `[tools]` and `[mcps]`

Each key is a tool slug and each value is a version spec. Both tables use the
same format, and both spellings below are equivalent:

```toml
[tools]
"@acme/tickets" = "1.x"                  # just the spec
"@acme/crm" = { spec = "2.3.x" }         # a table, with room for more keys later
```

- **Slug.** `@scope/name`, lowercase. The scope is a letter or digit followed
  by letters, digits and `-`; the name is a letter or digit followed by
  letters, digits, `.`, `_` and `-`. The scope is the publishing workspace's
  namespace. Quote the key in TOML, because of the `@` and `/`.
- **Spec.** `"*"` for the newest published version, an exact version such as
  `"1.4.2"`, or a prefix ending in `x` such as `"1.x"` or `"1.4.x"`, which
  matches the newest published version starting with `1.` or `1.4.`. A missing
  spec in the table form means `"*"`.

`load_tools()` returns the `[tools]` entries as callables. `[mcps]` entries are
not returned by `load_tools()`: MCP servers reach an `Agent` through its `mcps=`
argument.

### What gets pinned

You never write a hash. Two things are pinned for you:

- **The agent's code.** `gf agents push <dir>` validates the manifest, packs
  the directory into a deterministic archive and uploads it. The registry
  stores it under its `sha256:` digest, the CLI checks that the registry
  computed the same digest, and `gf agents create` and `--create` refer to the
  agent by that digest. `gf agents digest <dir>` prints the same digest without
  uploading anything.
- **The tools it names.** When you submit the agent to the marketplace, each
  `[tools]` spec is resolved to one exact version and that version's digest.
  These are recorded alongside the agent's archive, not written into your
  `genfleet.toml`, so the archive's digest still covers exactly the code you
  pushed. A spec that matches no published version, or a tool your workspace's
  plan can't use, fails the submission. The tools are mounted for the agent only
  after a reviewer approves the submission; until then the agent runs without
  them.

In the SDK, a `ToolRef` read from your manifest has `spec` set and `version`
and `digest` empty (`is_pinned` is `False`). A resolved one carries all three.
`AgentManifest.is_fully_pinned` is `True` when every `[tools]` and `[mcps]` entry
is pinned.

Locally nothing is pinned. `load_tools()` uses whatever tool provider is
installed on your machine and ignores digests, so you can edit an agent and a
tool together.

## Tool manifest

A marketplace tool (a connector) has its own `genfleet.toml` at the root of
the directory you pass to `gf tools push`:

```text
acme-tickets/
  genfleet.toml
  acme_tickets.py
```

```toml
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

### `[tool]`

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `slug` | yes | | `@scope/name`, with the same rules as an agent's `[tools]` keys. Agents name the tool by this slug. |
| `version` | yes | | The version you are publishing, as `MAJOR.MINOR.PATCH`. Agents' specs match against it. `gf tools push --tool-id` submits this exact version. |
| `kind` | no | `"cli"` | One of `"cli"`, `"http"`, `"mcp_stdio"`, `"mcp_http"`. See below. |
| `description` | no | `""` | What the tool does. |
| `entry` | yes | | `module:callable`, the function (or `@tool` object) `load_tools()` hands to the agent. The module path is relative to the tool's directory: `acme_tickets:create_ticket` is `create_ticket` in `acme_tickets.py`. |
| `sensitive` | no | `false` | `true` makes every call to this tool wait for a workspace owner's or admin's approval. See [approvals](approvals-and-messaging.md). |

**`kind`.** The parser accepts exactly these four values and rejects any
other. `load_tools()` loads every tool the same way, whatever its `kind`: it
imports `entry` and returns the callable.

**`sensitive`.** The platform reads this when the tool is published and uses
it as the tool's default on every turn. It can only add the marking: a tool
whose code says `@tool(sensitive=True)` stays sensitive even if the manifest
says `false`. The SDK itself doesn't read this key on a local run. To test the
approval flow locally, mark the function with `@tool(sensitive=True)`.

### `[config.<name>]`

Settings the installing workspace fills in: a URL, a project key, a region.
The tool reads them as `ctx.config["<name>"]` (see the
[connectors guide](connectors.md)).

| Key | Default | Meaning |
|-----|---------|---------|
| `type` | `"string"` | `"string"`, `"number"` or `"boolean"`. |
| `required` | `false` | Whether the installation must set it. |
| `description` | `""` | Shown to whoever installs the tool. |
| `default` | none | A default declared for the setting. `ctx.config` holds only the values the tool is given, so read optional ones with a fallback: `ctx.config.get("queue", "support")`. |

### `[secrets.<name>]`

Credentials the installing workspace binds. The tool reads them with
`ctx.secret("<name>")`.

| Key | Default | Meaning |
|-----|---------|---------|
| `description` | `""` | Shown to whoever installs the tool. |
| `required` | `true` | Whether the installation must bind it. |

A secret is declared by **name only**. A manifest is committed to a repository
and published to a registry every workspace can read, so the parser rejects a
`value` key under `[secrets]` outright:

```text
acme-tickets/genfleet.toml: [secrets] 'api_token' has a value. Manifests declare secret names only — the tenant binds the value at install time.
```

## Checking a manifest in a test

```python
from pathlib import Path

from genfleet.sdk import load_agent_manifest

PACKAGE = Path(__file__).parent.parent / "support_agent"


def test_manifest():
    manifest = load_agent_manifest(PACKAGE)
    assert manifest.name == "support"
    assert manifest.audiences == ["operator", "customer"]
    assert [ref.slug for ref in manifest.tools] == ["@acme/tickets"]
```
