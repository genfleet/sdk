"""
Agent with Redis-backed session memory for persistent multi-turn conversations.

Each request that includes a `session_id` in metadata will have its history
automatically loaded from Redis and saved back after the response.

Requires:
    pip install 'withfleet-sdk[openai,memory,serve]'

Start Redis:
    docker run -d -p 6379:6379 redis:7

Run:
    export OPENAI_API_KEY=sk-...
    python examples/memory_agent.py

Test (turn 1 — introduce yourself):
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{
        "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
        "params": {
          "id": "t1",
          "metadata": {"session_id": "demo-session"},
          "message": {"role": "user", "parts": [{"type": "text", "text": "My name is Alice"}]}
        }
      }' | python -m json.tool

Test (turn 2 — verify memory):
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{
        "jsonrpc": "2.0", "id": 2, "method": "tasks/send",
        "params": {
          "id": "t2",
          "metadata": {"session_id": "demo-session"},
          "message": {"role": "user", "parts": [{"type": "text", "text": "What is my name?"}]}
        }
      }' | python -m json.tool
"""

import os

from fleet.sdk import Agent
from fleet.sdk.serve import serve


agent = Agent(
    role="You are a friendly assistant with a long memory. Remember everything the user tells you.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    memory={
        "type":       "redis",
        "connection": os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        "password":   os.environ.get("REDIS_PASSWORD"),
    },
)

if __name__ == "__main__":
    serve(agent, name="memory-agent", description="Stateful agent with Redis-backed session memory", port=8000)
