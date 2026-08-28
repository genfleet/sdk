"""
Agent using OpenAI via Agent(), served over A2A.

Requires:
    pip install 'withfleet-sdk[openai,serve]'

Run:
    export OPENAI_API_KEY=sk-...
    python examples/openai_agent.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"What is the capital of France?"}]}}}' | python -m json.tool

    # Streaming
    curl -N -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"Write a haiku about code"}]}}}'
"""

import os

from fleet.sdk import Agent
from fleet.sdk.serve import serve


def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"72°F and sunny in {city}"


agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    tools=[get_weather],
)

if __name__ == "__main__":
    serve(agent, name="openai-agent", description="Chat agent powered by OpenAI GPT-4o-mini", port=8000)
