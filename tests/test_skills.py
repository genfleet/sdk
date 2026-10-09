from __future__ import annotations

from collections.abc import AsyncIterator
import subprocess

import pytest

from genfleet.sdk import Agent, AgentInput, AgentOutput, Message, Skill, ToolCall, discover_skills, load_skills
from genfleet.sdk.skills import _frontmatter, _repository_url, _skill_paths


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
    assert memory.sessions["s1"][2].content == "[skill body omitted; call read_skill again to use it]"
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


def test_selected_skill_list_is_passed_flat():
    selected_skills = [Skill(name="brief", description="Short answers", instructions="Answer briefly.")]
    agent = Agent(
        role="a", model={"model": "fake"}, model_client=RecordingModel(),
        skills=selected_skills,
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


def test_existing_and_reserved_local_layouts_are_not_github_handles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "owner" / "repo").mkdir(parents=True)
    with pytest.raises(ValueError, match="Path"):
        _repository_url("owner/repo")
    with pytest.raises(ValueError):
        _repository_url("agent/skills")


def test_agent_requires_pinned_remote_skills():
    with pytest.raises(ValueError, match="pinned"):
        Agent(role="a", model={"model": "fake"}, model_client=RecordingModel(), skills=["acme/skills"])


@pytest.mark.asyncio
async def test_customer_cannot_read_operator_skill():
    class CustomerModel:
        def __init__(self):
            self.calls = []

        async def complete(self, messages, tools, stream=True):
            self.calls.append((messages, tools))
            if messages[-1]["role"] == "tool":
                yield AgentOutput(content="done", done=True)
            else:
                yield AgentOutput(tool_calls=[ToolCall(id="1", name="read_skill", arguments={"name": "private"})], done=True)

    model = CustomerModel()
    agent = Agent(role="a", model={"model": "fake"}, model_client=model,
                  skills=[Skill(name="private", description="Private skill", instructions="Secret instructions.")])
    chunks = [chunk async for chunk in agent.run(AgentInput(message="Hello", metadata={
        "genfleet.caller": {"role": "customer", "private": True},
    }))]
    assert "private" not in model.calls[0][0][0]["content"]
    assert not any(tool.name == "read_skill" for tool in model.calls[0][1])
    assert all("Secret instructions." not in str(chunk) for chunk in chunks)


@pytest.mark.asyncio
async def test_customer_cannot_request_hidden_skill_when_reader_is_offered():
    class MixedModel:
        async def complete(self, messages, tools, stream=True):
            if messages[-1]["role"] == "tool":
                yield AgentOutput(content="done", done=True)
            else:
                assert next(tool for tool in tools if tool.name == "read_skill").parameters["properties"]["name"]["enum"] == ["public"]
                yield AgentOutput(tool_calls=[ToolCall(id="1", name="read_skill", arguments={"name": "private"})], done=True)

    agent = Agent(role="a", model={"model": "fake"}, model_client=MixedModel(), skills=[
        Skill(name="private", description="Private", instructions="Secret instructions."),
        Skill(name="public", description="Public", instructions="Public instructions.", audiences={"customer"}),
    ])
    chunks = [chunk async for chunk in agent.run(AgentInput(message="Hello", metadata={
        "genfleet.caller": {"role": "customer", "private": True},
    }))]
    events = [chunk.metadata["tool_event"] for chunk in chunks if "tool_event" in chunk.metadata]
    assert events[-1]["output"] == "Error: skill not found"
    assert all("Secret instructions." not in str(chunk) for chunk in chunks)


def test_skill_file_rejects_large_body_before_read(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text("x" * 32_001)
    with pytest.raises(ValueError, match="too large"):
        Skill.from_file(path)


def test_yaml_single_quote_and_invalid_quote():
    fields, _ = _frontmatter("---\nname: brief\ndescription: 'it''s short'\n---\nDo it.\n")
    assert fields["description"] == "it's short"
    with pytest.raises(ValueError, match="invalid quoted"):
        _frontmatter("---\nname: 'unfinished\n---\nDo it.\n")


def test_symlinked_skill_root_is_not_walked(tmp_path):
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\nname: escape\ndescription: Escape\n---\nNo.\n")
    (root / "skills").symlink_to(outside, target_is_directory=True)
    assert _skill_paths(root) == []


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
