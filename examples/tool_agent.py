"""
Demonstrates the @tool decorator and Agent with tools running locally (no HTTP server).

Run:
    export OPENAI_API_KEY=sk-...
    python examples/tool_agent.py
"""

import asyncio
import os

from fleet.sdk import Agent, AgentInput, ToolCall, tool


@tool(description="adds two numbers")
def add(a: float, b: float) -> str:
    return str(a + b)


@tool(description="multiplies two numbers")
def multiply(a: float, b: float) -> str:
    return str(a * b)


async def main():
    print("=== Tool schemas ===")
    print(f"  add:      {add.schema().model_dump()}")
    print(f"  multiply: {multiply.schema().model_dump()}")

    print("\n=== Direct tool calls ===")
    result = await add.call(ToolCall(id="1", name="add", arguments={"a": 3, "b": 4}))
    print(f"  add(3, 4) = {result}")

    result = await multiply.call(ToolCall(id="2", name="multiply", arguments={"a": 5, "b": 6}))
    print(f"  multiply(5, 6) = {result}")

    print("\n=== Agent with tools ===")
    agent = Agent(
        role="You are a calculator assistant. Use the tools to compute answers.",
        model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
        tools=[add, multiply],
    )

    inp = AgentInput(message="What is 3 + 4, and then multiply that result by 6?")
    async for output in agent.run(inp):
        if output.content:
            print(f"  agent: {output.content}", end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
