from __future__ import annotations

from collections.abc import AsyncIterator
import subprocess

import pytest

from genfleet.sdk import Agent, AgentInput, AgentOutput, Message, Skill, ToolCall, discover_skills, load_skills
from genfleet.sdk.skills import _repository_url


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], list]] = []

    async def complete(self, messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
        self.calls.append((messages, tools))
        if len(self.calls) == 1:
            yield AgentOutput(tool_calls=[ToolCall(id="skill-1", name="read_skill", arguments={"name": "brief"})], done=True)
        else:
            yield AgentOutput(content="A concise answer.", done=True)


class DictMemory:
    def __init__(self) -> None:
        self.sessions: dict[str, list[Message]] = {}

    async def load(self, session_id: str) -> list[Message]:
        return self.sessions.get(session_id, [])

    async def save(self, session_id: str, history: list[Message]) -> None:
        self.sessions[session_id] = history

    async def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


@pytest.mark.asyncio
async def test_skill_is_read_on_demand_and_saved_with_session():
    model = RecordingModel()
    memory = DictMemory()
    agent = Agent(
        role="Answer questions.",
        model={"model": "fake"},
        model_client=model,
        memory=memory,
        skills=[Skill(name="brief", description="Short answers", instructions="Answer in one sentence.", version="2")],
    )
    output = [chunk async for chunk in agent.run(AgentInput(message="Hello", metadata={"session_id": "s1"}))]
    assert output[-1].done
    assert "Answer in one sentence." not in model.calls[0][0][0]["content"]
    assert "brief" in model.calls[0][0][0]["content"]
    assert any(tool.name == "read_skill" for tool in model.calls[0][1])
    assert any(
        "Answer in one sentence." in message.get("content", "")
        for messages, _ in model.calls for message in messages
    )
    assert [message.role for message in memory.sessions["s1"]] == ["user", "assistant", "tool", "assistant"]
    assert memory.sessions["s1"][2].content == "Skill loaded"
    assert all("Answer in one sentence." not in str(chunk.metadata) for chunk in output)


def test_skill_file_loads_instructions(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text("---\nname: brief\ndescription: Short answers\nversion: 3\n---\nAnswer briefly.\n")
    skill = Skill.from_file(path)
    assert skill.instructions == "Answer briefly."
    assert skill.version == "3"


def test_duplicate_skill_names_are_rejected():
    skill = Skill(name="brief", description="Short answers", instructions="Answer briefly.")
    with pytest.raises(ValueError, match="unique"):
        Agent(role="a", model={"model": "fake"}, model_client=RecordingModel(), skills=[skill, skill])


def test_selected_skill_list_can_be_nested_once():
    selected_skills = [Skill(name="brief", description="Short answers", instructions="Answer briefly.")]
    agent = Agent(
        role="a", model={"model": "fake"}, model_client=RecordingModel(),
        skills=[selected_skills],
    )
    assert [skill.name for skill in agent.skills] == ["brief"]


@pytest.mark.asyncio
async def test_unknown_skill_does_not_expose_other_skill_body():
    class UnknownSkillModel:
        async def complete(self, messages, tools, stream=True) -> AsyncIterator[AgentOutput]:
            if messages[-1]["role"] == "tool":
                yield AgentOutput(content="No skill found.", done=True)
            else:
                yield AgentOutput(
                    tool_calls=[ToolCall(id="1", name="read_skill", arguments={"name": "missing"})],
                    done=True,
                )

    agent = Agent(
        role="a", model={"model": "fake"}, model_client=UnknownSkillModel(),
        skills=[Skill(name="brief", description="Short answers", instructions="Private instructions.")],
    )
    chunks = [chunk async for chunk in agent.run(AgentInput(message="Hello"))]
    events = [chunk.metadata["tool_event"] for chunk in chunks if "tool_event" in chunk.metadata]
    assert events[-1]["ok"] is False
    assert events[-1]["output"] == "Error: skill not found"
    assert all("Private instructions." not in str(chunk) for chunk in chunks)


def test_github_shorthand_becomes_clone_url():
    assert _repository_url("acme/agent-skills") == "https://github.com/acme/agent-skills.git"
    assert _repository_url("acme/agent-skills.git") == "https://github.com/acme/agent-skills.git"


@pytest.fixture
def skill_repository(tmp_path):
    repository = tmp_path / "skill-repo"
    (repository / "skills" / "brief" / "references").mkdir(parents=True)
    (repository / "skills" / "formal").mkdir()
    (repository / "handoff").mkdir()
    (repository / "skills" / "brief" / "SKILL.md").write_text(
        "---\nname: brief\ndescription: >-\n  Keep answers\n  short\nversion: 3\nmetadata:\n  owner: acme\n---\nAnswer briefly.\n"
    )
    (repository / "skills" / "brief" / "references" / "examples.md").write_text("A short example.")
    (repository / "skills" / "formal" / "SKILL.md").write_text(
        "---\nname: formal\ndescription: Formal answers\n---\nUse formal language.\n"
    )
    (repository / "handoff" / "SKILL.md").write_text(
        "---\nname: handoff\ndescription: Write a handoff\n---\nSummarize the work.\n"
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "add skills"],
        check=True,
    )
    return repository


def test_repository_discovery_lists_root_and_nested_skills(skill_repository):
    options = discover_skills(skill_repository)
    assert {option.name for option in options} == {"brief", "formal", "handoff"}
    assert all(option.source_revision for option in options)
    assert next(option for option in options if option.name == "brief").description == "Keep answers short"


def test_repository_selection_loads_named_skills(skill_repository):
    assert {skill.name for skill in load_skills(skill_repository)} == {"brief", "formal", "handoff"}
    assert [skill.name for skill in load_skills(skill_repository, names=["handoff"])] == ["handoff"]
    with pytest.raises(ValueError, match="unknown skills: missing"):
        load_skills(skill_repository, names=["missing"])


@pytest.mark.asyncio
async def test_repository_skill_reference_can_be_read(skill_repository):
    agent = Agent(role="a", model={"model": "fake"}, model_client=RecordingModel(), skills=[skill_repository])
    reference, ok = await agent._dispatch_tool(
        ToolCall(id="1", name="read_skill", arguments={"name": "brief", "path": "references/examples.md"})
    )
    assert ok and reference == "A short example."


def test_selecting_one_skill_ignores_other_skills_large_references(skill_repository):
    (skill_repository / "skills" / "formal" / "references").mkdir()
    (skill_repository / "skills" / "formal" / "references" / "large.md").write_text("x" * 32_001)
    subprocess.run(["git", "-C", str(skill_repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(skill_repository), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "add large reference"],
        check=True,
    )
    assert {option.name for option in discover_skills(skill_repository)} == {"brief", "formal", "handoff"}
    assert [skill.name for skill in load_skills(skill_repository, names=["handoff"])] == ["handoff"]
    with pytest.raises(ValueError, match="reference is too large"):
        load_skills(skill_repository)
