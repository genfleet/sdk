"""
Streaming agent that yields word-by-word chunks over A2A SSE.
Implements AgentProtocol directly — no LLM required.

Run:
    python examples/streaming_agent.py

Test (streaming):
    curl -N -X POST http://localhost:8001 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/sendSubscribe","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"tell me a story"}]}}}'
"""

import asyncio

from fleet.sdk import AgentInput, AgentOutput
from fleet.sdk.serve import serve


class StorytellerAgent:
    async def run(self, input: AgentInput):
        story = f"Once upon a time, someone said: {input.message}. The end."
        for word in story.split():
            yield AgentOutput(content=word + " ", done=False)
            await asyncio.sleep(0.1)
        yield AgentOutput(content="", done=True)


if __name__ == "__main__":
    serve(StorytellerAgent(), name="storyteller", description="Streams a short story", port=8001)
