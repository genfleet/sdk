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
   duplicated trigger is harmless.

```python
# report_agent.py
import datetime as dt
import os
import sqlite3
from pathlib import Path

from genfleet.sdk import Agent, tool
from genfleet.sdk.serve import serve

LEDGER = Path(os.environ.get("REPORT_LEDGER", "reports.sqlite3"))


def claim(key: str) -> bool:
    """Record `key` as handled. False if it already was: the run is a duplicate."""
    with sqlite3.connect(LEDGER) as db:
        db.execute("CREATE TABLE IF NOT EXISTS handled (key TEXT PRIMARY KEY, at TEXT)")
        cursor = db.execute(
            "INSERT OR IGNORE INTO handled VALUES (?, ?)",
            (key, dt.datetime.now(dt.UTC).isoformat()),
        )
        return cursor.rowcount == 1


def release(key: str) -> None:
    """Forget `key`, so a later trigger can try again."""
    with sqlite3.connect(LEDGER) as db:
        db.execute("DELETE FROM handled WHERE key = ?", (key,))


def build_and_send_report(date: str) -> str:
    return f"report for {date} sent"  # your real work goes here


@tool(audiences=["operator"])
def send_daily_report(date: str) -> str:
    """Build and send the daily sales report for `date` (YYYY-MM-DD)."""
    try:
        dt.date.fromisoformat(date)
    except ValueError:
        return f"{date!r} is not a date (YYYY-MM-DD); nothing was done."
    key = f"daily-report:{date}"
    if not claim(key):
        return f"The report for {date} was already sent; nothing was done."
    try:
        return build_and_send_report(date)
    except Exception:
        release(key)  # it raised: let the next trigger try again
        raise


def build_agent(remote_tools=()):
    return Agent(
        role=(
            "You run back-office jobs. When asked to send the daily report for a date, "
            "call send_daily_report with that date and reply with its result."
        ),
        model={
            "model": "openrouter/nex-agi/nex-n2.5-mini:free",
            "api_key": os.environ["OPENROUTER_API_KEY"],
        },
        tools=[send_daily_report, *remote_tools],
    )


if __name__ == "__main__":
    serve(build_agent(), name="reports", port=8000)
```

The key goes into the ledger *before* the work starts. Two triggers racing for
the same date can't both win, because the second insert is ignored. If the work
raises, the key is released so the next trigger retries. Whether a retry is
safe depends on how far the work got, so make the work itself safe to repeat
where you can (see below).

The tool is tested without a model or a server:

```python
# test_report.py
import asyncio

import pytest

from genfleet.sdk import ToolCall

import report_agent


def test_second_trigger_for_the_same_date_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(report_agent, "LEDGER", tmp_path / "ledger.sqlite3")
    call = ToolCall(id="1", name="send_daily_report", arguments={"date": "2026-10-23"})
    first = asyncio.run(report_agent.send_daily_report.call(call))
    second = asyncio.run(report_agent.send_daily_report.call(call))
    assert first == "report for 2026-10-23 sent"
    assert second == "The report for 2026-10-23 was already sent; nothing was done."


def test_a_failed_run_can_be_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(report_agent, "LEDGER", tmp_path / "ledger.sqlite3")

    def boom(date):
        raise RuntimeError("mail server down")

    monkeypatch.setattr(report_agent, "build_and_send_report", boom)
    call = ToolCall(id="1", name="send_daily_report", arguments={"date": "2026-10-24"})
    with pytest.raises(RuntimeError):
        asyncio.run(report_agent.send_daily_report.call(call))
    assert report_agent.claim("daily-report:2026-10-24")
```

## The external caller

### An agent you serve yourself

`serve()` exposes the agent over A2A, so a cron job can post the trigger
message itself. Send the key in the request's `metadata` as `turn_id`: `Agent`
hands it to its memory backend as the turn's idempotency key, and a backend
that honours it keeps the first copy of a retried turn and drops the replay.
The Redis backend doesn't, so the ledger is still what keeps the job from
running twice. Give scheduled jobs their own `contextId`, so their history
doesn't mix with anyone's conversation.

```bash
# crontab: 06:00 every day
0 6 * * * /usr/local/bin/send-daily-report.sh
```

```bash
#!/bin/sh
# send-daily-report.sh
DATE=$(date +%F)
curl -sf -X POST http://localhost:8000 \
  -H 'Content-Type: application/json' -H 'A2A-Version: 1.0' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"SendMessage\",\"params\":{
        \"message\":{\"messageId\":\"daily-report-$DATE\",\"role\":\"ROLE_USER\",
                   \"contextId\":\"jobs:daily-report\",
                   \"parts\":[{\"text\":\"Send the daily report for $DATE.\"}]},
        \"metadata\":{\"turn_id\":\"daily-report:$DATE\"}}}"
```

An agent you serve yourself has no roles: a turn without a verified caller
gets every tool, `send_daily_report` included. Keep that endpoint off the
public internet, or put your own authentication in front of it.

### An agent hosted on the platform

Send the trigger as a workspace member with the `gf` CLI, from any machine
where `gf login` has been run:

```bash
0 6 * * * gf agents chat reports "Send the daily report for $(date +\%F)."
```

The agent sees that turn as an operator in a private chat, so it is offered
its operator tools. The agent must serve operators, which it does unless its
manifest's `audiences` leaves `"operator"` out.

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
- **Outside any turn**, the scope is the agent's own **background scope**. That
  covers code that isn't running inside a turn's model call or tool, such as a
  task your process started on its own. No customer turn can read the
  background scope.

`Agent` makes the turn visible to the store while its model client or a tool
runs. Your tool code is therefore inside the turn without doing anything. Code
that runs elsewhere but should count as part of a turn can wrap the call in
`genfleet.sdk.turn.call_within(input.metadata, ...)`.

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
