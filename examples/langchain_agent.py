"""
LangChain agent served over A2A via AgentProtocol.

For framework integrations where you manage the LLM call yourself,
implement AgentProtocol directly (a class with async run()).

Requires:
    pip install langchain-openai langchain-core 'withfleet-sdk[serve]'

Run:
    export OPENAI_API_KEY=sk-...
    python examples/langchain_agent.py

Test:
    curl -s -X POST http://localhost:8003 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"What is 25 * 4?"}]}}}' | python -m json.tool
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

from fleet.sdk import AgentInput, AgentOutput
from fleet.sdk.serve import serve


@lc_tool
def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    try:
        return str(eval(expression))  # noqa: S307 — example only
    except Exception as e:
        return f"Error: {e}"


llm = ChatOpenAI(model="gpt-4o-mini", streaming=True).bind_tools([calculator])


class LangChainAgent:
    async def run(self, input: AgentInput):
        messages = [SystemMessage(content="You are a helpful assistant with a calculator tool.")]
        for msg in input.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content or ""))
        messages.append(HumanMessage(content=input.message))

        response = await llm.ainvoke(messages)

        if response.tool_calls:
            yield AgentOutput(
                tool_calls=[
                    {"id": tc["id"], "name": tc["name"], "arguments": tc["args"]}
                    for tc in response.tool_calls
                ],
                done=False,
            )
        else:
            yield AgentOutput(content=response.content, done=True)


if __name__ == "__main__":
    serve(LangChainAgent(), name="langchain-agent", description="LangChain agent with calculator", port=8003)
