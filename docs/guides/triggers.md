# Scheduled and event-driven work

**The SDK has no scheduler and no event subscriptions.** An agent built with
it does work only when a turn arrives: a chat message, a channel message, or an
A2A request to `serve()`. Nothing in the SDK wakes an agent at 08:00 or when a
row changes in your database.

You can still run scheduled or event-driven jobs with the pieces that exist.
Something outside the agent sends it a fixed message when the time comes or
the event happens, and an operator-only tool does the work. This guide shows
that pattern, how to make it safe to run twice, and where its state lives.

## The pattern

1. **An operator tool that does the job.** It takes the job's key, a date or
   an event id, as an argument. Its audience is `operator`, so customers never
   get it.
2. **A role line that maps the trigger message to the tool.** *"When asked to
   send the daily report for a date, call send_daily_report with that date."*
3. **An external caller** that sends the same message each time, with the key
   filled in: *"Send the daily report for 2026-10-23."*
4. **Idempotency on the key.** The tool records the key before it acts and
   does nothing when the key has already been handled, so a retried or
   duplicated trigger can't repeat the job.

> **Self-served agents have no roles.** On an agent you run with `serve()`,
> a turn without a verified caller gets every tool, `send_daily_report`
> included. The example binds to `127.0.0.1` for that reason. If you expose it,
> put your own authentication in front, and have that proxy strip every
> `genfleet.*` key from the request's `metadata` (for example
> `genfleet.caller`, `genfleet.peer_turn`, `genfleet.approved_call`):
> `serve()` passes request metadata through to the agent unchanged, so a caller
> who can set those keys can claim a role.

```python
# report_agent.py
import datetime as dt
import os
import sqlite3
from pathlib import Path

from genfleet.sdk import Agent, ModelConfig, tool
from genfleet.sdk.serve import serve

LEDGER = Path(os.environ.get("REPORT_LEDGER", "reports.sqlite3"))
DEFAULT_MODEL = "openrouter/nex-agi/nex-n2.5-mini:free"


def _db() -> sqlite3.Connection:
    db = sqlite3.connect(LEDGER)
    db.execute("CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, status TEXT, at TEXT)")
    return db


def claim(key: str) -> str | None:
    """Claim `key`. None if this run owns it now, else the status it already has."""
    with _db() as db:
        now = dt.datetime.now(dt.UTC).isoformat()
        cursor = db.execute("INSERT OR IGNORE INTO jobs VALUES (?, 'claimed', ?)", (key, now))
        if cursor.rowcount == 1:
            return None
        return db.execute("SELECT status FROM jobs WHERE key = ?", (key,)).fetchone()[0]


def mark(key: str, status: str) -> None:
    with _db() as db:
        db.execute("UPDATE jobs SET status = ? WHERE key = ?", (status, key))


def release(key: str) -> None:
    """Forget `key`, so a later trigger runs the job again."""
    with _db() as db:
        db.execute("DELETE FROM jobs WHERE key = ?", (key,))


def build_report(date: str) -> str:
    return f"Sales for {date}: ..."  # read-only work: safe to repeat


def send_report(report: str) -> None:
    pass  # the side effect: an email, an API call


@tool(audiences=["operator"])
def send_daily_report(date: str) -> str:
    """Build and send the daily sales report for `date` (YYYY-MM-DD)."""
    try:
        dt.date.fromisoformat(date)
    except ValueError:
        return f"{date!r} is not a date (YYYY-MM-DD); nothing was done."
    key = f"daily-report:{date}"
    status = claim(key)
    if status == "done":
        return f"The report for {date} was already sent; nothing was done."
    if status is not None:
        return (
            f"An earlier run for {date} did not finish ({status}). Check whether the report "
            "went out before running it again; nothing was done."
        )
    try:
        report = build_report(date)
    except Exception:
        release(key)  # nothing was sent yet: the next trigger can retry
        raise
    try:
        send_report(report)
    except Exception:
        mark(key, "uncertain")  # it may have gone out: a person checks before a retry
        raise
    mark(key, "done")
    return f"The report for {date} was sent."


def model_config() -> ModelConfig:
    """As in the agent factory guide: a key only when the environment has one."""
    model = os.environ.get("REPORT_MODEL", DEFAULT_MODEL)
    key = os.environ.get("OPENROUTER_API_KEY")
    return {"model": model, "api_key": key} if key else {"model": model}


def build_agent(remote_tools=()):
    return Agent(
        role=(
            "You run back-office jobs. When asked to send the daily report for a date, "
            "call send_daily_report with that date and reply with its result."
        ),
        model=model_config(),
        tools=[send_daily_report, *remote_tools],
    )


if __name__ == "__main__":
    # Local only: a self-served agent offers every tool to anyone who reaches it.
    serve(build_agent(), name="reports", host="127.0.0.1", port=8000)
```

The job is split at the side effect, and each failure is handled by where it
happened:

| Where it stopped | Ledger | Next trigger |
|------------------|--------|--------------|
| Finished | `done` | Does nothing |
| `build_report` raised (nothing sent) | released | Runs the job again |
| `send_report` raised (it may have gone out) | `uncertain` | Does nothing, and says a person should check |
| The process died between `claim` and the end | stays `claimed` | Does nothing, and says a person should check |

The last two rows need a person. Check whether the report actually went out,
then either mark the key `done` or `release` it so the next trigger sends it.
Two triggers racing for the same date can't both run the job: the second
insert is ignored. Where the destination accepts an idempotency key, pass the
job key (`daily-report:2026-10-23`) to it as well. A repeated send is then
harmless even in the uncertain cases.

The tool is tested without a model or a server:

```python
# test_report.py
import asyncio

import pytest

from genfleet.sdk import ToolCall

import report_agent


def trigger(date):
    call = ToolCall(id="1", name="send_daily_report", arguments={"date": date})
    return asyncio.run(report_agent.send_daily_report.call(call))


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(report_agent, "LEDGER", tmp_path / "ledger.sqlite3")


def test_second_trigger_for_the_same_date_does_nothing():
    assert trigger("2026-10-23") == "The report for 2026-10-23 was sent."
    assert trigger("2026-10-23") == "The report for 2026-10-23 was already sent; nothing was done."


def test_a_failure_before_sending_can_be_retried(monkeypatch):
    working = report_agent.build_report

    def broken(date):
        raise RuntimeError("database down")

    monkeypatch.setattr(report_agent, "build_report", broken)
    with pytest.raises(RuntimeError):
        trigger("2026-10-24")
    monkeypatch.setattr(report_agent, "build_report", working)
    assert trigger("2026-10-24") == "The report for 2026-10-24 was sent."


def test_a_failure_while_sending_is_not_retried_blindly(monkeypatch):
    def timeout(report):
        raise TimeoutError("no answer from the mail server")

    monkeypatch.setattr(report_agent, "send_report", timeout)
    with pytest.raises(TimeoutError):
        trigger("2026-10-25")
    assert "did not finish (uncertain)" in trigger("2026-10-25")
```

## The external caller

### An agent you serve yourself

`serve()` exposes the agent over A2A, so a cron job on the same machine can
post the trigger message. Give scheduled jobs their own `contextId`, so their
history doesn't mix with anyone's conversation. A JSON-RPC error still comes
back as HTTP 200, so check the body, not just the status:

```bash
# crontab: 06:00 every day
0 6 * * * /usr/local/bin/send-daily-report.sh
```

```bash
#!/bin/sh
# send-daily-report.sh
set -eu
DATE=$(date +%F)
RESPONSE=$(curl -sS -X POST http://127.0.0.1:8000 \
  -H 'Content-Type: application/json' -H 'A2A-Version: 1.0' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"SendMessage\",\"params\":{
        \"message\":{\"messageId\":\"daily-report-$DATE\",\"role\":\"ROLE_USER\",
                   \"contextId\":\"jobs:daily-report\",
                   \"parts\":[{\"text\":\"Send the daily report for $DATE.\"}]}}}")
# Fails (exit 1) on a JSON-RPC error or a task that didn't complete.
echo "$RESPONSE" | jq -e '.error == null and .result.task.status.state == "TASK_STATE_COMPLETED"' >/dev/null
# The agent's reply, for the cron log: check it says the report was sent.
echo "$RESPONSE" | jq -r '.result.task.artifacts[0].parts[0].text'
```

A retried request can't run the job twice, because the ledger guards it.
`Agent` hands a `turn_id` from the request's `metadata` to its memory as the
turn's idempotency key, but only the platform's memory backend uses it. For a
self-served agent the ledger is the only protection.

### An agent hosted on the platform

Send the trigger as a workspace member with the `gf` CLI:

```bash
0 6 * * * gf agents chat reports "Send the daily report for $(date +\%F)."
```

The agent sees that turn as an operator in a private chat, so it is offered
its operator tools. The agent must serve operators, which it does unless the workspace owner
has set it to serve customers only in the dashboard.

The machine running cron holds a logged-in `gf` session, which is a long-lived
credential, stored under `~/.genfleet/` (the CLI writes it readable only by
its owner). Use a dedicated account for it, not a person's login, with the
lowest workspace role that can chat with the agent, and run the cron job as an
OS user nobody else shares.

A hosted agent has no durable local files, so the SQLite ledger above doesn't
fit there. Keep the idempotency where the effect lands:

- pass the key (`daily-report:2026-10-23`) as the remote API's idempotency key,
  when it accepts one, or
- look first ("is there already a report for 2026-10-23?") and skip if so, or
- keep the ledger in a service of your own that the tool calls.

The workflow state store (`agent.state`) is durable on the platform, but it
finds runs by `run_id`, not by your key. On its own it can't answer "did I
already handle 2026-10-23?".

## Choosing the key

| Trigger | Key | Example |
|---------|-----|---------|
| Once per day | The date, in one fixed timezone | `daily-report:2026-10-23` |
| Once per period | The period | `weekly-digest:2026-W43`, `invoice-run:2026-10` |
| An external event | The event's own id from the sender | `order-paid:evt_8f2c…` |

Use the sender's event id, not a time you compute. A webhook delivered twice
carries the same id but arrives at two different times. Put the job's name in
the key so two jobs never share one.

## Workflow state and the background scope

When the job has several steps, or waits on something, keep its progress in
the workflow state store (`Agent(state=...)`; see "Workflow state" in the
[README](../../README.md#workflow-state-checkpoints-pauseresume-effect-claims)).
On the platform, the store decides whose run it is, the **scope**, from the
turn it is called in:

- **Inside a turn**, the scope is that turn's verified caller. For a job
  triggered with `gf agents chat`, that's the member who sent it. Their
  operator runs and a customer's runs never mix, and a call naming a turn that
  has already ended is refused.
- **Outside any turn**, the scope is the agent's own **background scope**.
  That covers code started outside every turn: work begun at startup, or a
  thread of your own. No customer turn can read the background scope.

`Agent` makes the turn visible to the store while its model client or a tool
runs, so your tool code is inside the turn without doing anything. Code that
runs elsewhere but should count as part of a turn can wrap the call in
`genfleet.sdk.turn.call_within(input.metadata, ...)`.

**A task started from a tool still belongs to the turn.** `asyncio.create_task`
copies the current context, turn included. Once the turn has ended, that
task's state calls name an ended turn and are refused. They do not fall back
to the background scope. For work that outlives the turn, start the task in a
fresh context, so it runs in the background scope explicitly:

```python
import asyncio
import contextvars

_background: set[asyncio.Task] = set()  # keeps tasks alive until they finish


def start_in_background(coro) -> None:
    """Run `coro` outside the current turn, in the background scope."""
    task = asyncio.create_task(coro, context=contextvars.Context())
    _background.add(task)
    task.add_done_callback(_background.discard)
```

Locally (`LocalStateStore`) there is no stamping. The scope is the `agent` and
`owner` values you pass, so use a fixed owner such as `"jobs"` for scheduled
work.

## What this pattern doesn't give you

- **Exactly-once timing.** If the caller doesn't run (the machine is off, `gf`
  isn't logged in), nothing happens. Monitor the caller like any cron job.
- **A guarantee that the model calls the tool.** The model reads the trigger
  message, and a vague message or a weak model may answer in text instead.
  Keep the message fixed and literal, check the reply in the caller, and
  prefer a model that is reliable at tool calling.
- **Customer-facing triggers.** Keep job tools operator-only. A customer who
  could send "send the daily report for …" should not get the tool.
