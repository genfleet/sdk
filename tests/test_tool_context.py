"""`ToolContext`: a tool reads its own config and secrets (backend#250)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from genfleet.sdk import (
    Agent,
    ToolCall,
    ToolContext,
    ToolSecretError,
    tool,
    tool_env_namespace,
)
from genfleet.sdk.manifest import AgentManifest, ToolRef
from genfleet.sdk.tool import wrap_tool
from genfleet.sdk.tools_loader import MountedToolResolver, load_tools

CASES = json.loads((Path(__file__).parent / "fixtures" / "tool-env-namespace.cases.json").read_text())["cases"]

NS = "GENFLEET_TOOL_ACME__JIRA"


@pytest.mark.parametrize("case", CASES, ids=[c["slug"] for c in CASES])
def test_namespace_matches_the_shared_table(case):
    assert tool_env_namespace(case["slug"]) == case["namespace"]


def platform_env(**extra: str) -> dict[str, str]:
    return {"GENFLEET_TOOLS_ROOT": "/tools", **extra}


def test_config_and_secret_come_from_the_tools_own_namespace():
    env = platform_env(**{
        f"{NS}__CONFIG": json.dumps({"base_url": "https://acme.example", "retries": 2}),
        f"{NS}__SECRETS": json.dumps({"api_token": "tok-$1"}),
    })
    ctx = ToolContext("@acme/jira", env)

    assert ctx.config == {"base_url": "https://acme.example", "retries": 2}
    assert ctx.secret("api_token") == "tok-$1"


def test_config_is_read_only():
    ctx = ToolContext("@acme/jira", platform_env(**{f"{NS}__CONFIG": '{"a": 1}'}))
    with pytest.raises(TypeError):
        ctx.config["a"] = 2  # type: ignore[index]


def test_another_tools_namespace_is_not_read():
    env = platform_env(GENFLEET_TOOL_ACME__GITHUB__SECRETS=json.dumps({"api_token": "gh"}))
    with pytest.raises(ToolSecretError) as exc:
        ToolContext("@acme/jira", env).secret("api_token")
    assert "api_token" in str(exc.value)


def test_on_the_platform_a_bare_variable_is_not_a_fallback():
    # The agent's own env may hold an API_TOKEN; a tool must not pick it up.
    env = platform_env(API_TOKEN="agent-value", api_token="agent-value")
    with pytest.raises(ToolSecretError) as exc:
        ToolContext("@acme/jira", env).secret("api_token")
    assert "agent-value" not in str(exc.value)
    assert "grant" in str(exc.value)


@pytest.mark.parametrize("marker", ["GENFLEET_HOSTED", "GENFLEET_A2A_TOKEN"])
def test_a_hosted_agent_without_mounted_tools_gets_no_bare_fallback(marker):
    # The engine sets GENFLEET_TOOLS_ROOT only when it mounted tools; an inline
    # tool of a hosted agent must still not read the agent's own variables.
    env = {marker: "1", "API_TOKEN": "agent-value"}
    with pytest.raises(ToolSecretError) as exc:
        ToolContext("@acme/jira", env).secret("api_token")
    assert "agent-value" not in str(exc.value)
    assert "grant" in str(exc.value)


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        (None, "Set api_token or API_TOKEN in the environment."),
        ("@acme/jira", f"Set {NS}__SECRETS or api_token or API_TOKEN in the environment."),
    ],
)
def test_the_local_error_names_each_variable_once(slug, expected):
    with pytest.raises(ToolSecretError) as exc:
        ToolContext(slug, {}).secret("api_token")
    assert str(exc.value).endswith(expected)


def test_an_upper_case_secret_name_is_named_once():
    with pytest.raises(ToolSecretError) as exc:
        ToolContext(None, {}).secret("TOKEN")
    assert str(exc.value).endswith("Set TOKEN in the environment.")


def test_locally_a_bare_variable_is_the_fallback():
    assert ToolContext("@acme/jira", {"API_TOKEN": "dev"}).secret("api_token") == "dev"
    assert ToolContext("@acme/jira", {"api_token": "dev2"}).secret("api_token") == "dev2"
    assert ToolContext(None, {"api_token": "dev3"}).secret("api_token") == "dev3"


def test_locally_the_namespace_wins_over_a_bare_variable():
    env = {"API_TOKEN": "bare", f"{NS}__SECRETS": json.dumps({"api_token": "ns"})}
    assert ToolContext("@acme/jira", env).secret("api_token") == "ns"


def test_malformed_values_name_the_variable_and_never_its_content():
    ctx = ToolContext("@acme/jira", platform_env(**{f"{NS}__SECRETS": "sk-not-json"}))
    with pytest.raises(ToolSecretError) as exc:
        ctx.secret("api_token")
    assert "sk-not-json" not in str(exc.value)
    with pytest.raises(ToolSecretError):
        _ = ToolContext("@acme/jira", platform_env(**{f"{NS}__CONFIG": "[1]"})).config


def test_repr_has_no_values():
    ctx = ToolContext("@acme/jira", platform_env(**{f"{NS}__SECRETS": '{"k": "secret-value"}'}))
    assert "secret-value" not in repr(ctx)


def test_a_context_parameter_is_not_in_the_schema():
    @tool
    def create(title: str, ctx: ToolContext) -> str:
        return title

    params = create.schema().parameters
    assert set(params["properties"]) == {"title"}
    assert params["required"] == ["title"]


def test_an_optional_or_string_annotation_counts():
    def a(x: str, ctx: ToolContext | None = None) -> str:
        return x

    def b(x: str, ctx: "ToolContext") -> str:  # noqa: UP037 - the string form is the case under test
        return x

    def c(x: str, ctx: "Optional[ToolContext]" = None) -> str:  # noqa: F821, UP037 - deliberately unresolvable
        return x

    for fn in (a, b, c):
        assert set(wrap_tool(fn).schema().parameters["properties"]) == {"x"}


def test_an_unsupported_context_annotation_is_refused_at_wrap_time():
    # Each names ToolContext but in a form the wrapper does not fill: refusing
    # beats offering the parameter to the model and never filling it.
    def a(x: str, ctx: Optional["ToolContext"] = None) -> str:  # noqa: F821, UP037 - the quoted inner hint is the case under test
        return x

    def b(x: str, ctx: "typing.Union[ToolContext, None]" = None) -> str:  # noqa: F821, UP037
        return x

    def c(x: str, ctxs: list[ToolContext]) -> str:
        return x

    for fn in (a, b, c):
        with pytest.raises(TypeError, match="not a supported form"):
            wrap_tool(fn)
        with pytest.raises(TypeError, match="not a supported form"):
            tool(fn)


@pytest.mark.asyncio
async def test_the_context_is_injected_with_the_tools_slug(monkeypatch):
    monkeypatch.setenv("GENFLEET_TOOLS_ROOT", "/tools")
    monkeypatch.setenv(f"{NS}__SECRETS", json.dumps({"api_token": "t"}))
    seen = {}

    def create(title: str, ctx: ToolContext) -> str:
        seen["ctx"] = ctx
        return ctx.secret("api_token")

    wrapper = wrap_tool(create, slug="@acme/jira")
    # A model-supplied value under the context's name is overwritten.
    out = await wrapper.call(ToolCall(id="1", name="create", arguments={"title": "x", "ctx": "forged"}))

    assert out == "t"
    assert isinstance(seen["ctx"], ToolContext)
    assert seen["ctx"].slug == "@acme/jira"


class _CallableTool:
    def __call__(self, title: str, ctx: ToolContext) -> str:
        return ctx.secret("api_token")


class _Holder:
    def create(self, title: str, ctx: ToolContext) -> str:
        return ctx.secret("api_token")


async def _async_create(title: str, ctx: ToolContext) -> str:
    return ctx.secret("api_token")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fn",
    [_CallableTool(), _Holder().create, _async_create],
    ids=["callable-instance", "bound-method", "async"],
)
async def test_every_tool_style_drops_and_injects_the_context(fn, monkeypatch):
    monkeypatch.setenv("GENFLEET_TOOLS_ROOT", "/tools")
    monkeypatch.setenv(f"{NS}__SECRETS", json.dumps({"api_token": "t"}))
    wrapper = wrap_tool(fn, slug="@acme/jira")

    assert set(wrapper.schema().parameters["properties"]) == {"title"}
    assert await wrapper.call(ToolCall(id="1", name="create", arguments={"title": "x"})) == "t"


@pytest.mark.asyncio
async def test_a_tool_without_a_context_is_unchanged():
    @tool
    def add(a: int, b: int) -> str:
        return str(a + b)

    assert await add.call(ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})) == "3"
    assert add.with_slug("@acme/add")._context_params == ()


@pytest.fixture
def _clean_modules():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]


@pytest.mark.asyncio
async def test_a_mounted_tool_gets_its_context_through_load_tools(tmp_path, monkeypatch, _clean_modules):
    # The platform path end to end: a mounted artifact whose function asks for
    # a context, resolved by `load_tools`, wrapped by `Agent`, called.
    directory = tmp_path / "acme" / "jira"
    directory.mkdir(parents=True)
    (directory / "genfleet.toml").write_text(
        '[tool]\nslug = "@acme/jira"\nversion = "1.0.0"\nentry = "jira:create"\n'
        '[secrets.api_token]\ndescription = "token"\n'
    )
    (directory / "jira.py").write_text(
        "from __future__ import annotations\n"
        "from genfleet.sdk import ToolContext\n"
        "def create(title: str, ctx: ToolContext) -> str:\n"
        "    return ctx.config['project'] + ':' + ctx.secret('api_token')\n"
    )
    monkeypatch.setenv("GENFLEET_TOOLS_ROOT", str(tmp_path))
    monkeypatch.setenv(f"{NS}__CONFIG", json.dumps({"project": "OPS"}))
    monkeypatch.setenv(f"{NS}__SECRETS", json.dumps({"api_token": "tok"}))

    [fn] = load_tools(
        manifest=AgentManifest(name="a", tools=[ToolRef(slug="@acme/jira", digest="sha256:" + "0" * 64)]),
        resolver=MountedToolResolver(tmp_path),
    )
    agent = Agent(role="r", model={"model": "openai/gpt-4o-mini", "api_key": "sk-test"}, tools=[fn])
    [wrapper] = agent._local_tools.values()

    assert set(wrapper.schema().parameters["properties"]) == {"title"}
    assert await wrapper.call(ToolCall(id="1", name="create", arguments={"title": "t"})) == "OPS:tok"
