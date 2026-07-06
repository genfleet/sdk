# RawAdapter — Migration Reference

> ⚠️ **Deprecated in v0.2.** `RawAdapter` emits a `DeprecationWarning` on instantiation and will be removed in v0.4.
>
> **Migrate to `Agent()`:**
> ```python
> # Before
> serve(RawAdapter(my_fn), name="agent", port=8000)
>
> # After
> serve(Agent(role="...", model={...}, tools=[...]), name="agent", port=8000)
> ```

This file is kept as a reference for teams migrating from `RawAdapter` patterns.

---

## Migration Guide

### Simple function → Agent

```python
# Before
async def my_agent(message: str) -> str:
    return f"You said: {message}"

serve(RawAdapter(my_agent), name="agent", port=8000)

# After (no LLM — implement AgentProtocol directly)
class MyAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content=f"You said: {input.message}", done=True)

serve(MyAgent(), name="agent", port=8000)
```

### OpenAI function → Agent

```python
# Before
async def openai_agent(message: str, history: list, tools: list):
    ...  # manual OpenAI wiring
    yield response.choices[0].message.content

serve(RawAdapter(openai_agent), ...)

# After
agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "openai/gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    tools=[my_tool_fn],
)
serve(agent, ...)
```

### Anthropic function → Agent

```python
# Before
async def anthropic_agent(message: str, history: list, tools: list):
    ...  # manual Anthropic wiring
    yield block.text

serve(RawAdapter(anthropic_agent), ...)

# After
agent = Agent(
    role="You are a helpful assistant.",
    model={"model": "anthropic/claude-sonnet-4-6", "api_key": os.environ["ANTHROPIC_API_KEY"]},
)
serve(agent, ...)
```

### Framework integrations (LangChain, CrewAI) → AgentProtocol

For framework integrations that manage their own LLM calls, implement `AgentProtocol` directly:

```python
# Before (LangChain)
async def langchain_agent(message: str, history: list):
    ...
serve(RawAdapter(langchain_agent), ...)

# After
class LangChainAgent:
    async def run(self, input: AgentInput):
        response = await llm.ainvoke(messages)
        yield AgentOutput(content=response.content, done=True)

serve(LangChainAgent(), ...)
```

---

## Old Signature Detection Rules (reference only)

The following describes how `RawAdapter` used to work. This is preserved for migration reference.

RawAdapter checked the function signature in this order:

**Just message:**
```python
async def agent(message: str) -> str: ...
```

**Message + history:**
```python
async def agent(message: str, history: list) -> str:
    # history was a list[dict]: {"role": ..., "content": ..., "tool_calls": ..., "tool_call_id": ...}
```

**Message + history + tools:**
```python
async def agent(message: str, history: list, tools: list):
    # tools was a list[dict]: {"name": ..., "description": ..., "parameters": {...}}
```

**Full AgentInput:**
```python
async def agent(input: AgentInput) -> AgentOutput: ...
```

Tool calls were signalled by yielding a dict:
```python
yield {
    "tool_calls": [{"id": "1", "name": "search", "arguments": {"q": message}}],
    "done": False,
}
```
