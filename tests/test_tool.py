import pytest
from fleet.sdk import tool, ToolWrapper, ToolCall


def test_tool_decorator_bare():
    @tool
    def greet(name: str) -> str:
        return f"hi {name}"

    assert isinstance(greet, ToolWrapper)
    assert greet.schema().name == "greet"
    assert "name" in greet.schema().parameters["properties"]


def test_tool_decorator_with_args():
    @tool(name="adder", description="adds two numbers")
    def add(a: float, b: float) -> str:
        return str(a + b)

    assert add.schema().name == "adder"
    assert add.schema().description == "adds two numbers"
    assert set(add.schema().parameters.get("required", [])) == {"a", "b"}


@pytest.mark.asyncio
async def test_tool_call():
    @tool
    def multiply(x: int, y: int) -> str:
        return str(x * y)

    tc = ToolCall(id="1", name="multiply", arguments={"x": 3, "y": 4})
    result = await multiply.call(tc)
    assert result == "12"


@pytest.mark.asyncio
async def test_tool_async_fn():
    @tool
    async def fetch(url: str) -> str:
        return f"fetched {url}"

    tc = ToolCall(id="1", name="fetch", arguments={"url": "http://example.com"})
    result = await fetch.call(tc)
    assert result == "fetched http://example.com"


def test_tool_optional_param():
    @tool
    def search(query: str, limit: int = 10) -> str:
        return query

    schema = search.schema()
    assert "query" in schema.parameters.get("required", [])
    assert "limit" not in schema.parameters.get("required", [])
