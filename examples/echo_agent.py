"""
Minimal echo agent served over A2A.
Implements AgentProtocol directly — no LLM required.

Run:
    python examples/echo_agent.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"hello world"}]}}}' | python -m json.tool

    curl -s http://localhost:8000/.well-known/agent.json | python -m json.tool
"""

from fleet.sdk import AgentInput, AgentOutput, AgentProtocol
from fleet.sdk.serve import serve


class EchoAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content=f"Echo: {input.message}", done=True)


assert isinstance(EchoAgent(), AgentProtocol)

if __name__ == "__main__":
    serve(EchoAgent(), name="echo-agent", description="Echoes your message back", port=8000)
