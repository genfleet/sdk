# Framework Integration Examples

For LLM providers supported natively by the SDK (OpenAI, Anthropic, Gemini, and any vendor behind OpenRouter), use `Agent()` directly.

For external orchestration frameworks (LangChain, CrewAI) that manage their own LLM calls, implement `AgentProtocol` directly.

**Who runs the tools.** `serve` does not execute `tool_calls` that a custom `AgentProtocol` yields, and it does not return them to the caller: `SendMessage` carries only the text and files you yield, and `SendStreamingMessage` adds the tool events an `Agent` reports. A custom agent must run its own tools inside `run()` and yield the final text. (`Agent()` does this loop for you.)

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

**Install:** `pip install 'genfleet-sdk[openai,serve] @ git+https://github.com/genfleet/sdk@dev'`

```python
import os
from genfleet.sdk import Agent
from genfleet.sdk.serve import serve

def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    return f"72°F and sunny in {city}"

def add(a: float, b: float) -> str:
    """Adds two numbers."""
    return str(a + b)

agent = Agent(
    role="You are a helpful assistant. Use tools to answer questions accurately.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    tools=[get_weather, add],
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

Send the same `contextId` on each message to enable persistent sessions (how it becomes the agent's `session_id` is in `serve.md`, under "Sessions"). On a hosted agent `memory="platform"` (or leaving `memory` out) uses the platform's store instead of Redis:
```bash
curl -X POST http://localhost:8000 -H 'A2A-Version: 1.0' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"m1","role":"ROLE_USER","contextId":"user-123","parts":[{"text":"remember my name is Alice"}]}}}'
```

---

## Anthropic

**Install:** `pip install 'genfleet-sdk[anthropic,serve] @ git+https://github.com/genfleet/sdk@dev'`

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

**Install:** `pip install 'genfleet-sdk[gemini,serve] @ git+https://github.com/genfleet/sdk@dev'`

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

**Install:** `pip install 'genfleet-sdk[openai,serve] @ git+https://github.com/genfleet/sdk@dev'`

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

Requests to the gateway carry OpenRouter's attribution headers (`HTTP-Referer: https://genfleet.ai`, `X-OpenRouter-Title: Genfleet`). Set `default_headers` in the model config to add or override them; an explicit `base_url` to another endpoint sends none.

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

For LangChain, implement `AgentProtocol` directly since LangChain manages its own LLM invocations. `serve` will not run the tool calls the model asks for, so the agent runs them itself and loops until the model answers in text.

**Install:** `pip install langchain-openai langchain-core 'genfleet-sdk[serve] @ git+https://github.com/genfleet/sdk@dev'`

```python
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

from genfleet.sdk import AgentInput, AgentOutput
from genfleet.sdk.serve import serve

MAX_STEPS = 5


@lc_tool
def add(a: float, b: float) -> str:
    """Adds two numbers."""
    return str(a + b)


tools = {t.name: t for t in [add]}
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(list(tools.values()))


class LangChainAgent:
    async def run(self, input: AgentInput):
        messages = [SystemMessage(content="You are a helpful assistant with an adder.")]
        for msg in input.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content or ""))
        messages.append(HumanMessage(content=input.message))

        for _ in range(MAX_STEPS):
            response = await llm.ainvoke(messages)
            if not response.tool_calls:
                yield AgentOutput(content=str(response.content), done=True)
                return
            messages.append(response)
            for call in response.tool_calls:  # run the tools here: serve will not
                result = await tools[call["name"]].ainvoke(call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        yield AgentOutput(content="I could not finish within the step limit.", done=True)


serve(LangChainAgent(), name="langchain-agent", port=8003)
```

**Key notes:**
- Use LangChain's own `@tool` decorator for LangChain tools; the loop above calls them with `ainvoke`
- LangChain tool calls carry `call["args"]` (not `arguments`) and an `id` to echo back in the `ToolMessage`
- Tool results never reach the A2A caller unless you put them in `content`; to report a tool run as a tool event, use `Agent()` instead
- Convert genfleet `Message` objects to LangChain message types when building history. Only user and assistant turns are mapped here; `tool` messages are skipped
- Per-tool `audiences` and `sensitive` approval are enforced by `Agent`, not by `serve`: tools that need them belong in an `Agent()`

---

## CrewAI

For CrewAI, implement `AgentProtocol` directly since CrewAI manages its own orchestration.

**Install:** `pip install crewai 'genfleet-sdk[serve] @ git+https://github.com/genfleet/sdk@dev'`

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
- CrewAI manages its own LLM calls and runs its own tools internally, so there are no `tool_calls` for `serve` to dispatch
