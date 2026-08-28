from fleet.sdk import AgentInput, AgentOutput, Message, ToolCall, ToolSchema


def test_agent_input_minimal():
    inp = AgentInput(message="hello")
    assert inp.message == "hello"
    assert inp.history == []
    assert inp.tool_schemas == []
    assert inp.metadata == {}


def test_agent_input_with_history():
    msg = Message(role="user", content="hi")
    inp = AgentInput(message="hello", history=[msg])
    assert len(inp.history) == 1
    assert inp.history[0].role == "user"


def test_agent_output_defaults():
    out = AgentOutput(content="reply")
    assert out.content == "reply"
    assert out.tool_calls == []
    assert out.done is False


def test_agent_output_with_tool_calls():
    tc = ToolCall(id="1", name="search", arguments={"q": "test"})
    out = AgentOutput(tool_calls=[tc], done=False)
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "search"


def test_tool_schema():
    ts = ToolSchema(name="add", description="adds", parameters={"type": "object"})
    assert ts.name == "add"


def test_message_with_tool_call_id():
    msg = Message(role="tool", content="result", tool_call_id="tc-1")
    assert msg.tool_call_id == "tc-1"
