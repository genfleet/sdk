"""
Agent using Anthropic Claude via Agent(), served over A2A.

Requires:
    pip install 'withfleet-sdk[anthropic,serve]'

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/anthropic_agent.py

Test:
    curl -s -X POST http://localhost:8002 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"Explain quantum computing in 2 sentences"}]}}}' | python -m json.tool

    # Streaming
    curl -N -X POST http://localhost:8002 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"Write a limerick about Python"}]}}}'
"""

import os

from fleet.sdk import Agent
from fleet.sdk.serve import serve


def lookup(topic: str) -> str:
    """Looks up a fact in the knowledge base."""
    return f"Here's what we know about {topic}: it's fascinating."


agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "anthropic/claude-sonnet-4-6", "api_key": os.environ["ANTHROPIC_API_KEY"]},
    tools=[lookup],
)

if __name__ == "__main__":
    serve(agent, name="anthropic-agent", description="Chat agent powered by Claude Sonnet", port=8002)
