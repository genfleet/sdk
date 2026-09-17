"""`provider_for` routes by prefix; `openrouter/` is the OpenAI provider
pointed at the gateway with the vendor id passed through.

The OpenAI client is stubbed out, so these assert on what the SDK hands the
client (api_key, base_url, model) rather than on any provider internals.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from genfleet.sdk.providers import _OPENROUTER_BASE_URL, _OPENROUTER_HEADERS, provider_for
from genfleet.sdk.providers import openai as openai_provider
from genfleet.sdk.providers.openai import OpenAIProvider

MODEL = "openrouter/google/gemini-2.5-flash"
VENDOR_MODEL = "google/gemini-2.5-flash"

_REPLY = SimpleNamespace(
    choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
    usage=None,
)


class _StubClient:
    """Records the kwargs the SDK passes to `AsyncOpenAI(...)` and to `create(...)`."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.create_kwargs: dict = {}
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.create_kwargs = kwargs
        return _REPLY


@pytest.fixture
def clients(monkeypatch):
    """Every OpenAI client built during a test, in creation order."""
    created: list[_StubClient] = []

    def factory(**kwargs):
        client = _StubClient(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(openai_provider, "AsyncOpenAI", factory)
    return created


async def _drive_one_completion(provider) -> None:
    """Run one non-streaming completion so the stub records the routed model."""
    async for _ in provider.complete([{"role": "user", "content": "hi"}], [], stream=False):
        pass


async def test_openrouter_selects_the_openai_provider_at_the_gateway(clients):
    provider = provider_for({"model": MODEL, "api_key": "sk-or-k"})

    assert isinstance(provider, OpenAIProvider)
    await _drive_one_completion(provider)
    assert clients[0].init_kwargs == {
        "api_key": "sk-or-k",
        "base_url": _OPENROUTER_BASE_URL,
        "default_headers": _OPENROUTER_HEADERS,
    }
    assert clients[0].create_kwargs["model"] == VENDOR_MODEL


@pytest.mark.parametrize("base_url", [None, ""], ids=["none", "empty"])
async def test_a_blank_base_url_still_falls_back_to_the_gateway(clients, base_url):
    # Without this an `sk-or-` key would be sent to api.openai.com or $OPENAI_BASE_URL.
    provider_for({"model": MODEL, "api_key": "sk-or-k", "base_url": base_url})

    assert clients[0].init_kwargs["base_url"] == _OPENROUTER_BASE_URL


async def test_an_explicit_base_url_beats_the_gateway_default(clients):
    provider = provider_for(
        {"model": MODEL, "api_key": "k", "base_url": "http://localhost:11434/v1"}
    )

    await _drive_one_completion(provider)
    assert clients[0].init_kwargs["base_url"] == "http://localhost:11434/v1"
    # Attribution headers are OpenRouter's; a Groq or Ollama endpoint gets none.
    assert clients[0].init_kwargs.get("default_headers") is None
    # The vendor id survives the override — only the endpoint changes.
    assert clients[0].create_kwargs["model"] == VENDOR_MODEL


def test_openrouter_without_a_vendor_segment_is_rejected():
    with pytest.raises(ValueError, match=r"openrouter/<vendor>/<model>"):
        provider_for({"model": "openrouter/m", "api_key": "k"})


def test_unknown_prefix_lists_the_supported_ones():
    with pytest.raises(ValueError, match="openrouter"):
        provider_for({"model": "groq/llama", "api_key": "k"})


def test_a_model_without_a_prefix_is_rejected():
    with pytest.raises(ValueError, match="Expected 'provider/model-name'"):
        provider_for({"model": "gpt-4o-mini", "api_key": "k"})


async def test_caller_headers_are_kept_alongside_the_attribution_ones(clients):
    provider = provider_for({
        "model": f"openrouter/{VENDOR_MODEL}",
        "api_key": "k",
        "default_headers": {"X-OpenRouter-Title": "My App", "X-Custom": "1"},
    })
    await _drive_one_completion(provider)
    sent = clients[0].init_kwargs["default_headers"]
    assert sent["X-OpenRouter-Title"] == "My App"      # the caller's value wins
    assert sent["HTTP-Referer"] == _OPENROUTER_HEADERS["HTTP-Referer"]
    assert sent["X-Custom"] == "1"
