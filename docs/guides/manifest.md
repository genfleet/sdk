# `agent.toml` and tool manifest reference

Every agent has an `agent.toml` and every marketplace tool has a
`genfleet.toml`. An agent's manifest says what the agent is, which tools it
depends on, and what the workspace that installs it can configure. A tool's
manifest says what the tool is and which settings and credentials it needs to
run. The SDK parses both, so you can check a manifest offline:

```python
from genfleet.sdk import load_agent_manifest, load_tool_manifest

agent = load_agent_manifest("support_agent")   # the directory holding agent.toml
tool = load_tool_manifest("acme-tickets")
```

Both raise `ManifestError` when the file is missing, isn't valid TOML, or has a
bad value. The message starts with the file's path, and `.path` and `.problem`
hold the two parts. `gf agents push` and `gf tools push` run the same parser
before they upload anything, so a manifest that fails these checks never
reaches the registry. The parser checks the slug, `kind` and the
shape of each table; it does not check the format of `version` or `entry`, so
check those yourself.

## Agent manifest

The agent's `agent.toml` sits inside the agent's Python package, next to
the module that holds `build_agent`:

```text
support-agent/
  pyproject.toml
  support_agent/
    __init__.py
    agent.toml
    agent.py
```

```toml
[agent]
name = "support"
kind = "code"
description = "Answers order questions and opens helpdesk tickets."
entry = "support_agent.agent:build_agent"

[tools]
"@acme/tickets" = "1.x"

[mcps]
"@acme/files" = { spec = "*" }
```

**The old name.** Before 0.28.0 the agent's manifest was also called
`genfleet.toml`. That name is still read, so an existing agent keeps working.
Rename the file when you next touch it: `git mv genfleet.toml agent.toml`. A
directory holding both is an error. `agent.toml` needs `gf` 0.7.0 or later to
push.

### `[agent]`

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `name` | yes | | The agent's name. `gf agents push --create` uses it as the agent's name in your workspace unless you pass `--name`. |
| `kind` | no | `"code"` | One of `"code"`, `"declarative"`, `"native_sdk"`. See below. |
| `description` | no | `""` | A one-line description. `gf agents push --create` uses it as the agent's description. |
| `entry` | no | `"agent:build_agent"` | `module:callable` for the agent factory. See the [agent factory guide](agent-factory.md). |
| `model_choice` | no | `false` | `true` when the agent reads its model id from the `AGENT_MODEL` environment variable. Workspace owners and admins can then choose its model on the agent's page. See below. |

**`kind`.** The parser accepts exactly these three values and rejects any
other. It doesn't attach behaviour to them. An agent written as a Python
package with a `build_agent` factory, like every example in this repository,
is `"code"`, which is the default.

**Who the agent serves.** The manifest doesn't say. The workspace owner sets
it in the dashboard, and by default an agent serves operators only. Which
*tools* each audience gets is decided per tool; see "Operators and customers"
in the [README](../../README.md#operators-and-customers).

The SDK parser still accepts an `audiences` key for compatibility, but the
platform doesn't use it, so set who an agent serves in the dashboard.

**Letting the workspace choose the model.** Read the model id from
`AGENT_MODEL` when the agent is built, with your own default when it is unset,
and set `model_choice = true`:

```python
import os

DEFAULT_MODEL = "your-default-model"

def build_agent():
    return Agent(
        role="Support",
        model={"model": os.environ.get("AGENT_MODEL") or DEFAULT_MODEL},
    )
```

```toml
[agent]
name = "support"
model_choice = true
```

The workspace picks from the models the platform offers, and the agent
restarts on its new model. With no choice, `AGENT_MODEL` is unset and your
default applies. Leave `model_choice` out when the code ignores `AGENT_MODEL`:
the page then shows no model picker, since a choice would change nothing.

### `[data]`

```toml
[data]
required = true
data = ["knowledge.json", "flows/**"]
```

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `required` | no | `false` | `true` when the agent can't serve without content supplied by the workspace that installs it. |
| `data` | no | `[]` | Globs, relative to the content folder, of the files your code reads by path. |

Use it for a generic agent whose business comes from its installer: a persona,
knowledge, rules. With `required = true` the platform doesn't serve the agent
until that content is in place. `required` must be `true` or `false` (not a
string), and any other key in `[data]` is an error, which also means an SDK
release that adds a key is needed before a manifest can use it. The table was
called `[content]` before 0.28.0; that name is still read, but not together
with `[data]`.

**Data files.** The workspace chooses, per file, whether it is *loaded* (meant
to go into the agent's context whole) or *retrieved* (searched when needed,
not in the content folder). A file your code opens by name, such as a JSON
knowledge base or a graph, must always be there, so list it in `data`: the
workspace sees it as used by the agent and can't change it, and it is never
returned as a loaded file. Data files have their own size limit.

In a glob, `*` matches any characters within one name, `?` one character, and
a `**` segment any number of folders (`flows/**`, `**/*.json`). Matching is
case-sensitive. A glob must be relative, use `/` between folders, and have no
`.` or `..` folders; brackets and braces aren't supported. At most 100 globs.

**Reading loaded files.** `genfleet.sdk.content.loaded_files()` returns the
loaded text files as `LoadedFile(path, text)` tuples, sorted by path, and
`loaded_prompt()` renders them as one block with a `--- path ---` header per
file, ready for a system prompt:

```python
from genfleet.sdk.content import loaded_prompt

def build_agent(remote_tools=()):
    return Agent(role=loaded_prompt(exclude=["knowledge.json", "flows/**"]), ...)
```

They read the content folder: the `root` you pass, else `$CONTENT_DIR`, else
`/content`. Only that last default may be missing (no files); a folder you
named that doesn't exist is an error. Text files are `.md`, `.markdown`,
`.txt`, `.json`, `.toml`, `.yaml`, `.yml`, `.csv`, `.html` and `.htm` that are
valid UTF-8; other files are never returned (a text file that isn't UTF-8 is
logged). On the platform only the files the workspace set to loaded are
returned, and `data` files are already left out, so `exclude=` only matters
for local runs against a plain folder, where every text file counts as
loaded: pass it the same globs as `data`. Paths and globs are compared
NFC-normalised. A loaded file over 4 MiB is left out with a logged warning,
and reading stops (with a warning) at the first file that would take the
loaded files past 8 MiB together, so you always get a prefix of the sorted
eligible files (skipped files can sit in between); both caps are far above
what fits in an agent's context.

### `[settings.<key>]`

Values the installing workspace sets on the agent's page, in a form built
from these tables. Use them for anything the workspace decides and your code
branches on: where leads go, a time window, a mapping between their names and
yours.

```toml
[settings.lead_sink]
type = "enum"
options = ["erp", "views"]
default = "views"
label = "Where leads go"
group = "Leads"

[settings.open_lead_window_days]
type = "integer"
min = 0
max = 365
default = 30

[settings.source_map]
type = "map"
description = "Channel name to the lead source in your system."
```

| Key | Default | Meaning |
|-----|---------|---------|
| `type` | `"string"` | `"string"` (one line, 500 characters), `"text"` (20,000), `"integer"`, `"number"`, `"boolean"`, `"enum"`, `"list"` (of strings) or `"map"` (string to string). A list or map holds at most 200 entries. |
| `label` | `""` | The field's label, at most 60 characters. The key is shown when left out. |
| `description` | `""` | Help text under the field, at most 300 characters. |
| `group` | `""` | A heading that settings with the same group share. |
| `required` | `false` | The agent's page marks the agent as needing setup until it has a value. |
| `default` | none | The value in effect until the workspace saves one. It must fit the type. |
| `options` | | The choices of an `enum`: 1 to 50 distinct strings. Only for `enum`. |
| `min`, `max` | none | Bounds of an `integer` or `number`. |

The key is 1 to 32 lowercase letters, digits or `_`, starting with a letter;
at most 40 settings. Any other key in the table is an error.

**A setting never holds a credential.** There is no secret type. Use a
connector for an API you call, or a tool's `[secrets]`.

Read them with `genfleet.sdk.config`:

```python
from genfleet.sdk.config import is_set, setting

sink = setting("lead_sink")                 # the saved value, else the manifest default
days = setting("open_lead_window_days", 30)  # ... else this default
if is_set("source_map"):                    # True only when the workspace saved one
    ...
```

Values are read once, when the agent starts; the platform restarts the agent
when the workspace changes them. On a local run nothing is set and `setting`
returns the default you pass. To try values locally, write the file yourself
and point `GENFLEET_AGENT_CONFIG` at it:

```json
{"version": 1, "settings": {"lead_sink": "erp"}, "defaults": {"open_lead_window_days": 30}}
```

### `[connectors.<slot>]`

A connection the agent needs from its workspace. The workspace picks which of
its connections fills the slot on the agent's page, and your code asks for the
slot, so it never hard-codes a connection's name:

```toml
[connectors.erp]
provider = "custom_api"
description = "The ERP that receives leads."
```

| Key | Default | Meaning |
|-----|---------|---------|
| `provider` | required | The kind of connection, e.g. `"custom_api"`. |
| `required` | `true` | The agent's page marks the agent as needing setup until the slot is filled. |
| `description` | `""` | Shown next to the slot, at most 300 characters. |

The slot name follows the same rule as a setting key; at most 8 slots.

```python
from genfleet.sdk import call_connection
from genfleet.sdk.config import connector

erp = connector("erp")          # None until the workspace fills the slot
if erp is not None:
    response = await call_connection(erp.connection, "GET", "/sales_people")
```

`connector()` returns a `ConnectorBinding` with `connection` (the name to
call), `provider`, `config` and `tools` (the connection's verified tools). An
agent can only call a connection the workspace has let it use; any other call
raises `ConnectionCallError` with code `connection_not_allowed`.

`connector_tools("erp")` (in `genfleet.sdk` and `genfleet.sdk.config`) turns
the slot's verified tools into tools for the model; see
[Connector tools and `reply`](approvals-and-messaging.md#connector-tools-and-reply).

### Prompts the workspace adds

The workspace can add prompts of its own to an agent, on the agent's page.
Put them after your system prompt with `with_tenant_prompts`:

```python
from genfleet.sdk.config import with_tenant_prompts

def build_agent(remote_tools=()):
    return Agent(role=with_tenant_prompts(ROLE), ...)
```

It returns your prompt followed by each of the workspace's prompts under its
title, or your prompt unchanged when there are none. `tenant_prompts()`
returns them as `TenantPrompt(title, body)` if you want to place them
yourself.

### `[tools]` and `[mcps]`

> **Beta:** for now, keep your tools in your agent's own package, as the
> [publish guide](../../README.md#publish-on-genfleet-private-beta) does. An
> agent that names a marketplace tool here can't be started until the agent is
> submitted to the marketplace and approved, with that tool pinned. A tool that
> isn't published yet can't be pinned, so the submission fails.

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
  uploading anything. `gf agents push --update <dir>` runs a new push on the
  agent that already exists, so it keeps its id and history.
- **The tools it names.** When you submit the agent to the marketplace, each
  `[tools]` spec is resolved to one exact version and that version's digest.
  These are recorded alongside the agent's archive, not written into your
  `agent.toml`, so the archive's digest still covers exactly the code you
  pushed. A spec that matches no published version, or a tool your workspace's
  plan can't use, fails the submission. An agent that names a
  marketplace tool can't be started until the agent is submitted and approved,
  with that tool pinned.

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
| `version` | yes | | The version you are publishing. Write it as `MAJOR.MINOR.PATCH`: agents' specs (`1.x`, `1.4.x`) match it by prefix. The parser accepts any string. `gf tools push --tool-id` submits this exact version. |
| `kind` | no | `"cli"` | One of `"cli"`, `"http"`, `"mcp_stdio"`, `"mcp_http"`. See below. |
| `description` | no | `""` | What the tool does. |
| `entry` | yes | | `module:callable`, the function (or `@tool` object) `load_tools()` hands to the agent. The module path is relative to the tool's directory: `acme_tickets:create_ticket` is `create_ticket` in `acme_tickets.py`. |
| `sensitive` | no | `false` | `true` makes every call to this tool wait for a workspace owner's or admin's approval. See [approvals](approvals-and-messaging.md). |

**`kind`.** The parser accepts exactly these four values and rejects any
other. On the platform, `load_tools()` loads every tool the same way,
whatever its `kind`: it imports `entry` from the tool's directory and returns
the callable. Locally it uses whatever the installed tool package provides
(see the [connectors guide](connectors.md#using-a-tool-from-an-agent-locally)).

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
    assert [ref.slug for ref in manifest.tools] == ["@acme/tickets"]
```

## `[[views]]` and `views_app`

Pages the agent serves to the workspace. `[agent] views_app` names the ASGI
app (`module:symbol`); each `[[views]]` table has `id`, `title`, `path` and
`min_role`. See [Agent views](views.md).
