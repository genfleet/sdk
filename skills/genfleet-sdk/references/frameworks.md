# Framework Integration Examples

For LLM providers supported natively by the SDK (OpenAI, Anthropic, Gemini, and any vendor behind OpenRouter), use `Agent()` directly.

For external orchestration frameworks (LangChain, CrewAI) that manage their own LLM calls, implement `AgentProtocol` directly.

## Table of Contents

1. [OpenAI](#openai)
2. [Anthropic](#anthropic)
3. [Gemini](#gemini)
4. [OpenRouter](#openrouter)
5. [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
6. [LangChain](#langchain)
7. [CrewAI](#crewai)

---

## OpenAI

**Install:** `pip install 'genfleet-sdk[openai,serve]'`

```python
import os
from genfleet.sdk import Agent
from genfleet.sdk.serve import serve

def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"72°F and sunny in {city}"

def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    return str(eval(expression))

agent = Agent(
    role="You are a helpful assistant. Use tools to answer questions accurately.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    tools=[get_weather, calculator],
)

serve(agent, name="openai-agent", port=8000)
```

**With memory (multi-turn sessions):**
```python
agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    memory={"type": "redis", "connection": "redis://localhost:6379"},
)
```

Pass `session_id` in request metadata to enable persistent sessions:
```bash
curl -X POST http://localhost:8000 \
  -d '{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"id":"t1","metadata":{"session_id":"user-123"},"message":{"role":"user","parts":[{"type":"text","text":"remember my name is Alice"}]}}}'
```

---

## Anthropic

**Install:** `pip install 'genfleet-sdk[anthropic,serve]'`

```python
import os
from genfleet.sdk import Agent
from genfleet.sdk.serve import serve

def lookup(topic: str) -> str:
    """Looks up a fact."""
    return f"Here's what we know about {topic}: it's fascinating."

agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "anthropic/claude-sonnet-4-6", "api_key": os.environ["ANTHROPIC_API_KEY"]},
    tools=[lookup],
)

serve(agent, name="anthropic-agent", port=8002)
```

---

## Gemini

**Install:** `pip install 'genfleet-sdk[gemini,serve]'`

```python
import os
from genfleet.sdk import Agent
from genfleet.sdk.serve import serve

agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "gemini/gemini-1.5-pro", "api_key": os.environ["GOOGLE_API_KEY"]},
)

serve(agent, name="gemini-agent", port=8005)
```

---

## OpenRouter

**Install:** `pip install 'genfleet-sdk[openai,serve]'`

`openrouter/<vendor>/<model>` routes through the OpenRouter gateway, so one key reaches every vendor it fronts. The vendor id is passed through intact — no `base_url` needed:

```python
agent = Agent(
    role="You are a helpful assistant.",
    model={
        "model":   "openrouter/google/gemini-2.5-flash",
        "api_key": os.environ["OPENROUTER_API_KEY"],
    },
)
```

The vendor segment is required: `openrouter/gemini-2.5-flash` raises `ValueError`. A non-empty `base_url` still wins if you need a different gateway.

---

## OpenAI-Compatible Endpoints

Any provider with an OpenAI-compatible API (DeepSeek, Groq, Ollama, Together, etc.) works with `openai/` prefix + `base_url`:

```python
# DeepSeek
agent = Agent(
    role="You are a helpful assistant.",
    model={
        "model":    "openai/deepseek-chat",
        "api_key":  os.environ["DEEPSEEK_API_KEY"],
        "base_url": "https://api.deepseek.com",
    },
)

# Groq
agent = Agent(
    role="You are a helpful assistant.",
    model={
        "model":    "openai/llama-3.3-70b-versatile",
        "api_key":  os.environ["GROQ_API_KEY"],
        "base_url": "https://api.groq.com/openai/v1",
    },
)

# Local Ollama
agent = Agent(
    role="You are a helpful assistant.",
    model={
        "model":    "openai/llama3.2",
        "api_key":  "ollama",          # any non-empty string
        "base_url": "http://localhost:11434/v1",
    },
)
```

---

## LangChain

For LangChain, implement `AgentProtocol` directly since LangChain manages its own LLM invocations.

**Install:** `pip install langchain-openai langchain-core 'genfleet-sdk[serve]'`

```python
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

from genfleet.sdk import AgentInput, AgentOutput
from genfleet.sdk.serve import serve


@lc_tool
def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    return str(eval(expression))


llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([calculator])


class LangChainAgent:
    async def run(self, input: AgentInput):
        messages = [SystemMessage(content="You are a helpful assistant with a calculator.")]
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


serve(LangChainAgent(), name="langchain-agent", port=8003)
```

**Key notes:**
- Use LangChain's own `@tool` decorator for LangChain tools
- LangChain tool calls use `tc["args"]` (not `tc["arguments"]`) — map accordingly
- Convert genfleet `Message` objects to LangChain message types when building history

---

## CrewAI

For CrewAI, implement `AgentProtocol` directly since CrewAI manages its own orchestration.

**Install:** `pip install crewai 'genfleet-sdk[serve]'`

```python
import asyncio
from crewai import Agent as CrewAgent, Crew, Task

from genfleet.sdk import AgentInput, AgentOutput
from genfleet.sdk.serve import serve


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
        # CrewAI is synchronous — wrap to avoid blocking the event loop
        result = await asyncio.to_thread(crew.kickoff)
        yield AgentOutput(content=str(result), done=True)


serve(CrewAIAgent(), name="crewai-agent", port=8004)
```

**Key notes:**
- `Crew.kickoff()` is synchronous and blocking — always wrap with `asyncio.to_thread()`
- CrewAI manages its own LLM calls and tool schemas internally
