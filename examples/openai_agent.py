"""
Agent using OpenAI via Agent(), served over A2A.

Requires:
    pip install 'genfleet-sdk[openai,serve]'

Run:
    export OPENAI_API_KEY=sk-...
    python examples/openai_agent.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" -H 'A2A-Version: 1.0' \
      -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"What is the capital of France?"}]}}}' | python -m json.tool

    # Streaming
    curl -N -X POST http://localhost:8000 \
      -H "Content-Type: application/json" -H 'A2A-Version: 1.0' \
      -d '{"jsonrpc":"2.0","id":1,"method":"SendStreamingMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"Write a haiku about code"}]}}}'
"""

import os

from genfleet.sdk import Agent
from genfleet.sdk.serve import serve


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
