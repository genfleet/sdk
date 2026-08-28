"""
CrewAI agent served over A2A via AgentProtocol.

For framework integrations where you manage the LLM call yourself,
implement AgentProtocol directly (a class with async run()).

Requires:
    pip install crewai 'withfleet-sdk[serve]'

Run:
    export OPENAI_API_KEY=sk-...
    python examples/crewai_agent.py

Test:
    curl -s -X POST http://localhost:8004 \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","message":{"role":"user","parts":[{"type":"text","text":"Research the benefits of TypeScript over JavaScript"}]}}}' | python -m json.tool
"""

import asyncio

from crewai import Agent as CrewAgent, Crew, Task

from fleet.sdk import AgentInput, AgentOutput
from fleet.sdk.serve import serve


researcher = CrewAgent(
    role="Senior Research Analyst",
    goal="Provide thorough, well-structured research on any topic",
    backstory="You are an experienced analyst who excels at breaking down complex topics.",
    verbose=False,
    allow_delegation=False,
)


class CrewAIAgent:
    async def run(self, input: AgentInput):
        task = Task(
            description=input.message,
            expected_output="A clear, concise analysis",
            agent=researcher,
        )
        crew = Crew(agents=[researcher], tasks=[task], verbose=False)
        result = await asyncio.to_thread(crew.kickoff)
        yield AgentOutput(content=str(result), done=True)


if __name__ == "__main__":
    serve(CrewAIAgent(), name="crewai-agent", description="CrewAI research analyst agent", port=8004)
