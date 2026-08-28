"""
Agent with audit logging — demonstrates all three audit backends.

Requires:
    pip install 'withfleet-sdk[openai,serve]'

Run:
    export OPENAI_API_KEY=sk-...
    python examples/audit_agent.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"What is the weather in Tokyo?"}]}}}' | python -m json.tool

Check audit output:
    cat audit.jsonl | python -m json.tool --json-lines
"""

import logging
import os

from fleet.sdk import Agent, AuditEvent
from fleet.sdk.serve import serve
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"72°F and sunny in {city}"


def calculate(expression: str) -> str:
    """Evaluates a math expression."""
    return str(eval(expression))


# --- Option 1: Console backend (structured JSON to logger) ---
# audit={"backend": "console", "agent_name": "weather-bot"}

# --- Option 2: File backend (JSONL) ---
# audit={"backend": "file", "file_path": "audit.jsonl", "agent_name": "weather-bot"}

# --- Option 3: Callback backend (custom handler) ---
def on_audit_event(event: AuditEvent) -> None:
    import json
    latency = f" | {event.latency_ms:.0f}ms" if event.latency_ms else ""
    print(f"\n[AUDIT] {event.event_type}{latency}")
    print(f"        {json.dumps(event.data, indent=2, default=str)}")


agent = Agent(
    role="You are a helpful assistant with access to weather and calculator tools.",
    model={"model": "anthropic/claude-haiku-4-5", "api_key": os.environ["ANTHROPIC_API_KEY"]},
    tools=[get_weather, calculate],
    audit={
        "backend": "callback",
        "callback": on_audit_event,
        "agent_name": "audit-demo",
        "max_content_length": 0,
    },
)

if __name__ == "__main__":
    serve(agent, name="audit-agent", description="Agent with audit logging", port=8000)
