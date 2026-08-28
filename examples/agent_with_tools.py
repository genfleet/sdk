"""
Agent with multiple tools using Agent(), served over A2A.

Demonstrates automatic tool wrapping — plain Python functions are passed
directly to Agent(); no @tool decorator required.

Requires:
    pip install 'withfleet-sdk[openai,serve]'

Run:
    export OPENAI_API_KEY=sk-...
    python examples/agent_with_tools.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"What is the weather in Paris and what is 25 * 4?"}]}}}' | python -m json.tool
"""

import os

from fleet.sdk import Agent
from fleet.sdk.serve import serve


def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"72°F and sunny in {city}"


def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    try:
        return str(eval(expression))  # noqa: S307 — example only
    except Exception as e:
        return f"Error: {e}"


def lookup_person(name: str) -> str:
    """Looks up a person's profile by name."""
    profiles = {
        "alice": "Alice — Senior Engineer, joined 2021, works on infra",
        "bob": "Bob — Product Manager, joined 2023, owns the marketplace",
    }
    return profiles.get(name.lower(), f"No profile found for {name}")


agent = Agent(
    role="You are a helpful assistant. Use the tools available to answer questions accurately.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    tools=[get_weather, calculator, lookup_person],
)

if __name__ == "__main__":
    serve(agent, name="smart-agent", description="Agent with weather, calculator, and lookup tools", port=8000)
