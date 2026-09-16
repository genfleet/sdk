"""`provider_for` routes by prefix; `openrouter/` is the OpenAI provider
pointed at the gateway with the vendor id passed through."""
from __future__ import annotations

import pytest

from genfleet.sdk.providers import OPENROUTER_BASE_URL, _parse_model, provider_for


def test_openrouter_prefix_keeps_the_vendor_id_intact():
    assert _parse_model("openrouter/google/gemini-2.5-flash") == ("openrouter", "google/gemini-2.5-flash")


def test_openrouter_selects_the_openai_provider_at_the_gateway():
    pytest.importorskip("openai")
    provider = provider_for({"model": "openrouter/google/gemini-2.5-flash", "api_key": "k"})
    assert type(provider).__name__ == "OpenAIProvider"
    assert str(provider._client.base_url).rstrip("/") == OPENROUTER_BASE_URL
    assert provider._model == "google/gemini-2.5-flash"


def test_an_explicit_base_url_beats_the_gateway_default():
    pytest.importorskip("openai")
    provider = provider_for({"model": "openrouter/m", "api_key": "k", "base_url": "http://localhost:11434/v1"})
    assert str(provider._client.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_unknown_prefix_lists_the_supported_ones():
    with pytest.raises(ValueError, match="openrouter"):
        _parse_model("groq/llama")
