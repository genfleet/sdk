"""genfleet.sdk.config: the agent configuration file and its readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genfleet.sdk import config
from genfleet.sdk.config import AGENT_CONFIG_ENV, AgentConfigError

CASES = json.loads((Path(__file__).parent / "fixtures" / "agent-config.json").read_text())["cases"]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    config._load.cache_clear()
    monkeypatch.delenv(AGENT_CONFIG_ENV, raising=False)
    yield
    config._load.cache_clear()


def _write(tmp_path, monkeypatch, file) -> None:
    path = tmp_path / "agent-config.json"
    path.write_text(file if isinstance(file, str) else json.dumps(file))
    monkeypatch.setenv(AGENT_CONFIG_ENV, str(path))


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_shared_contract(case, tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, case["file"])
    if case.get("error"):
        with pytest.raises(AgentConfigError):
            config.agent_config()
        return
    assert dict(config.settings()) == case["settings"]
    assert sorted(k for k in case["settings"] if config.is_set(k)) == case.get("saved", sorted(case["file"].get("settings", {})))
    got = {slot: {"connection": b.connection, "provider": b.provider, "config": dict(b.config)} for slot, b in config.agent_config().connectors.items()}
    assert got == case["connectors"]
    assert [[p.title, p.body] for p in config.tenant_prompts()] == case["prompts"]
    assert config.with_tenant_prompts("ROLE") == case["with_prompts"]


def test_unset_variable_means_empty():
    assert config.setting("lead_sink", "views") == "views"
    assert config.connector("erp") is None
    assert config.tenant_prompts() == ()
    assert config.with_tenant_prompts("ROLE\n") == "ROLE\n"


def test_setting_and_connector_lookups(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, CASES[0]["file"])
    assert config.setting("lead_sink") == "erp"
    assert config.setting("missing") is None
    assert config.setting("currency", "USD") == "EGP"
    assert config.is_set("lead_sink") and not config.is_set("currency")
    assert config.setting("strict", False) is True
    assert config.connector("erp").connection == "heatec-erp"
    assert config.connector("crm") is None


def test_a_named_file_that_is_missing_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv(AGENT_CONFIG_ENV, str(tmp_path / "gone.json"))
    with pytest.raises(AgentConfigError, match="cannot read"):
        config.agent_config()


def test_an_oversized_file_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "big.json"
    path.write_text(" " * (config._MAX_BYTES + 1))
    monkeypatch.setenv(AGENT_CONFIG_ENV, str(path))
    with pytest.raises(AgentConfigError, match="larger"):
        config.agent_config()


def test_empty_system_prompt_takes_only_the_tenant_prompts(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"version": 1, "prompts": [{"title": "Tone", "body": " Be brief. "}, {"title": "x", "body": "  "}]})
    assert config.with_tenant_prompts("") == "## Tone\n\nBe brief."
