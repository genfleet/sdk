"""
Minimal echo agent served over A2A.
Implements AgentProtocol directly — no LLM required.

Run:
    python examples/echo_agent.py

Test:
    curl -s -X POST http://localhost:8000 \
      -H "Content-Type: application/json" -H 'A2A-Version: 1.0' \
      -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","parts":[{"text":"hello world"}]}}}' | python -m json.tool

    curl -s http://localhost:8000/.well-known/agent-card.json | python -m json.tool
"""

from genfleet.sdk import AgentInput, AgentOutput, AgentProtocol
from genfleet.sdk.serve import serve


class EchoAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content=f"Echo: {input.message}", done=True)


assert isinstance(EchoAgent(), AgentProtocol)

if __name__ == "__main__":
    serve(EchoAgent(), name="echo-agent", description="Echoes your message back", port=8000)
