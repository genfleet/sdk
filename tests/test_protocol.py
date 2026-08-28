from fleet.sdk import AgentInput, AgentOutput, AgentProtocol, ToolProtocol, ToolCall, ToolSchema


class MyAgent:
    async def run(self, input: AgentInput):
        yield AgentOutput(content="hello", done=True)


class MyTool:
    def schema(self) -> ToolSchema:
        return ToolSchema(name="test", description="test", parameters={})

    async def call(self, tool_call: ToolCall) -> str:
        return "ok"


def test_agent_protocol_check():
    agent = MyAgent()
    assert isinstance(agent, AgentProtocol)


def test_tool_protocol_check():
    t = MyTool()
    assert isinstance(t, ToolProtocol)
