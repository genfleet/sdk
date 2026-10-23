# Approvals and outbound messages

Two features that reach beyond a single reply:

- **Sensitive tools.** A tool call that waits for a workspace owner or admin to
  approve it, then runs exactly once.
- **`send_message`.** Sending a message to someone other than the person the
  agent is replying to, or sending it later than the reply.

Both are handled by `Agent`, and both only do their real work on the
platform. This guide covers what your code and your model see at each step,
and how to test it locally.

## Sensitive tools

### Marking a tool

A call to a sensitive tool doesn't run when the model makes it. Any of these
makes a tool sensitive:

| Where | How |
|-------|-----|
| Your code | `@tool(sensitive=True)` |
| A connector's manifest | `sensitive = true` under `[tool]` (see the [manifest reference](manifest.md#tool)) |
| An MCP server's config | `"sensitive_tools": ["delete_file", ...]` in its `MCPConfig` |
| The workspace | The owner marks the tool sensitive in the dashboard |

The markings only add up. The workspace's setting can make a tool sensitive,
but a `false` from the platform never unmarks a tool its author marked.

```python
from genfleet.sdk import tool


@tool(audiences=["operator", "customer"], sensitive=True)
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"
```

Mark a tool sensitive when a mistaken call would be costly or hard to undo:
money moving, data deleted, a message going to many people.

### What happens, step by step

1. **The model calls the tool.** It doesn't run. In its place, the model gets
   a tool result saying the action hasn't run yet because it needs approval
   from a workspace owner or admin, that a request was sent, and that it runs
   once approved. It is told to let the person it's talking to know it is
   waiting. The turn's tool event (`AgentOutput.metadata[TOOL_EVENT_KEY]`)
   for that call has `"ok": False` and `"status": "pending"`.
2. **The request leaves with the turn.** The turn's final chunk carries the
   request in its metadata under `CONFIRMATION_REQUESTS_KEY`
   (`"genfleet.confirmation_requests"`): a list of `{"tool": ..., "arguments": {...}}`.
   The platform records it and shows the owner or admin the exact call, tool
   and arguments, to approve.
3. **An owner or admin approves it.** The platform then starts a follow-up
   turn that carries the approved call, signed by the platform. Before the
   model does anything, `Agent` checks the signature and the expiry, makes sure
   this call hasn't already run, and runs it, once. The model sees the call and
   its result in the conversation, along with an instruction to tell the
   person it's talking to what happened.
4. **The outcome is reported.** The follow-up turn's final chunk carries
   `APPROVED_CALL_RESULT_KEY` (`"genfleet.approved_call_result"`):

| Value | Meaning |
|-------|---------|
| `"ran"` | The tool ran and returned. |
| `"failed"` | The tool raised, or the MCP call failed. The model gets the error as the tool's result (for your own tool, `Error: the approved action failed (<exception type>).`), and the turn goes on. The model is told the action failed and isn't confirmed as done, and that someone should check before asking for it again. |
| `"refused"` | Nothing ran. The approval couldn't be verified (missing, malformed, a bad signature, expired, or already run), or the tool is unknown or isn't offered to this turn's caller. The model is told that it did not run, and to say so. |

The platform treats this report as advisory and checks it against the audit
trail (`tool.call` and `tool.result` events). A turn that ends without the key
leaves the outcome unconfirmed, not "did not run".

A call that is **rejected, or never approved**, never reaches your agent: no
follow-up turn carries it. An approval that arrives after it has expired is
refused.

### Writing tools and prompts for it

- **Put everything the call needs in its arguments.** The approved call runs in
  a different turn, possibly in a different process, maybe hours later. It gets
  exactly the arguments the model chose, and nothing from the original turn's
  local variables.
- **Check state again when it runs.** Between the request and the approval,
  the order may have been refunded by hand. Make the tool look before it acts
  ("already refunded: nothing to do"), and pass a stable idempotency key to the
  remote API when it supports one.
- **Tell the model not to claim success early.** The pending result already
  tells it to say the action is waiting for approval. Add a line to `role` that
  fits your agent, for example *"When an action is waiting for approval, say
  that it will run once a manager approves it. Never say it is done."*
- **Approval needs `Agent`.** The sensitive flow is implemented by `Agent`. A
  hand-written `AgentProtocol` that dispatches tools itself has to honour the
  markings itself. If it doesn't, keep sensitive actions out of it.

### Peer turns

When another agent calls yours, the turn is marked as a peer turn
(`PEER_TURN_KEY`, `"genfleet.peer_turn"`). On a peer turn every sensitive tool
is **refused outright**, and no approval request is made, because nobody can
approve a call on another agent's behalf. The model gets:

```text
Error: this tool needs an owner's approval and can't be used on a call from another agent.
```

If a peer needs that action, it has to ask a person to make the request
directly.

### Trying it locally

Locally nothing is approved, but you can see steps 1 and 2 and the peer
refusal. This script uses `ScriptedModelClient` in place of a real model:

```python
import asyncio

from genfleet.sdk import PEER_TURN_KEY, TOOL_EVENT_KEY, Agent, AgentInput, tool
from genfleet.sdk.models import ScriptedModelClient, Step


@tool(audiences=["operator"], sensitive=True)
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


async def turn(metadata):
    client = ScriptedModelClient([
        Step(tool_calls=[{"name": "issue_refund", "arguments": {"order_id": "A-1", "amount": 20.0}}]),
        "Your refund is waiting for approval.",
    ])
    agent = Agent(role="You handle refunds.", model={"model": "scripted"},
                  tools=[issue_refund], model_client=client)
    async for out in agent.run(AgentInput(message="Refund order A-1", metadata=metadata)):
        event = out.metadata.get(TOOL_EVENT_KEY)
        if event and event["type"] == "tool_result":
            print("tool event:", event)
        if out.done:
            print("final metadata:", out.metadata)
    print("model saw:", client.calls[1].tool_results[0]["content"])


asyncio.run(turn({}))                     # pending, with a confirmation request
asyncio.run(turn({PEER_TURN_KEY: True}))  # refused, no request
```

An approved call can't be faked locally. One the platform didn't sign is
reported as `"refused"`, and nothing runs.

## Sensitive tools vs `StateStore.pause`

Both stop work and pick it up later. They answer different questions:

| | Sensitive tool | `StateStore.pause` / `resume` |
|---|---|---|
| Question | "May this action happen?" | "Where was I?" |
| Who continues | An owner or admin, who approves in the dashboard | Whoever holds the resume token, in the same owner scope, **including the original requester** |
| What runs | Exactly the call that was approved, once | Whatever your code does after `resume` |
| Use it for | Refunds, deletions, anything that needs a second person | Waiting for an external event (a payment webhook, a reply), multi-step workflows |

**Don't build an approval gate on `pause`.** The person who asked can resume
their own run, so a pause proves nothing about anyone else agreeing. An
approval by someone other than the requester goes through sensitive tools.
Durable resume after a human approval is tracked in
[genfleet/sdk#59](https://github.com/genfleet/sdk/issues/59).

You can combine them. A workflow can checkpoint its progress with the state
store and still put its one risky step behind a sensitive tool.

## Sending messages: `send_message`

An agent's reply to an inbound message needs nothing from here. Whatever the
agent returns is sent back to the contact by the platform. `send_message` is
for everything else: an alert to staff, a follow-up, a pre-approved template.
The platform still does the sending. Your code names a recipient and the
words, it never sees the channel's credentials, and the platform picks which
of the agent's channel connections the message goes out on.

```python
from genfleet.sdk import send_message

await send_message("whatsapp:201001234567", "Your order has shipped.")

await send_message(
    "whatsapp:201001234567",
    template="order_shipped",       # a template approved for your channel
    params=["A-1", "Tuesday"],      # one string per placeholder
    language="en",
)
```

`send_message(to, text=None, *, template=None, params=None, language="en")` is
async. It returns the platform's answer (`{"sent": True, "messageIds": [...]}`)
or raises `ChannelSendError`.

- **Recipients.** `to` is `"<channel>:<address>"`, for example
  `"whatsapp:201001234567"` (the number with its country code). That's the same
  shape as the `session_id` of an inbound channel turn, so writing to the
  current contact outside the reply is
  `send_message(input.metadata["session_id"], ...)`. Inside a tool, the turn's
  metadata is `genfleet.sdk.turn.current_turn()`.
- **Text or template.** Pass exactly one of `text` and `template`, or it raises
  `ValueError`. `params` must be a list (a bare string raises `ValueError`
  rather than being split into characters).
- **The 24-hour window.** Free text only reaches a contact who wrote to the
  agent in the last 24 hours. A template reaches anyone the channel connection
  may contact.

### `ChannelSendError`

| Attribute | Meaning |
|-----------|---------|
| `code` | A refusal you can act on, or `None` |
| `status` | The HTTP status, `0` when nothing answered |
| `delivery_unknown` | `True` when the message may have been sent: no `code`, and not a 4xx status |

| `code` | What it means | What to do |
|--------|---------------|------------|
| `outside_window` | The contact hasn't written in 24 hours | Send a template instead |
| `cold_send_disabled` | The contact never wrote, and this channel connection doesn't allow first contact | Don't retry; have the contact write first |
| `no_binding` | The agent has no channel connection to send this on | Connect the agent to the channel; don't retry |
| `invalid_request` | The platform couldn't accept the request as sent | Fix the arguments |
| `send_rejected` | The send was refused | Don't retry unchanged |
| `not_in_sandbox` | The agent isn't running on the platform (a local run, a test) | Expected locally |

With a `code`, nothing was sent. Without one (a timeout, the platform
unreachable, a server error), the message may or may not have gone out, and
`delivery_unknown` is `True`. **Don't resend blindly** in that case: the
recipient may get it twice.

### The model-facing tool: `send_message_tool`

`send_message_tool` wraps `send_message` as a tool named `send_message`, for
handing to the model directly:

```python
from genfleet.sdk import Agent, send_message_tool

agent = Agent(role="…", model={…}, tools=[send_message_tool])
```

It never raises. It returns `"sent"`, `"not sent: <reason>"`, or
`"delivery unknown (<reason>); do not resend without checking"`, so the model
can explain a refusal and doesn't repeat a message that may have arrived. Its
description tells the model to address WhatsApp numbers, to use `text` within
24 hours and a template otherwise, and not to use it for the normal reply.

It has the default audience, **operators only**. A tool that lets the model
message any number is not something to offer customers. For customer-facing
agents, write a narrow tool with a fixed recipient instead:

```python
import os

from genfleet.sdk import ChannelSendError, send_message, tool


async def _alert(to: str, summary: str) -> None:
    try:
        await send_message(to, f"Needs attention: {summary}")
    except ChannelSendError as exc:
        if exc.code != "outside_window":
            raise
        # Free text only reaches someone who wrote in the last 24 hours.
        await send_message(to, template="staff_alert", params=[summary[:200]], language="en")


@tool(audiences=["operator", "customer"])
async def alert_staff(summary: str) -> str:
    """Tell the on-call staff member about something that needs a human."""
    to = os.environ["STAFF_CONTACT"]  # e.g. "whatsapp:201001234567"
    try:
        await _alert(to, summary)
    except ChannelSendError as exc:
        if exc.delivery_unknown:
            return "The alert may already have been delivered. Don't send it again."
        return f"Couldn't alert staff ({exc.code or exc.status})."
    return "Staff were alerted."
```

Run locally, it returns `Couldn't alert staff (not_in_sandbox).`: there are
no channels outside the platform, and `send_message` says so instead of
pretending to send.
