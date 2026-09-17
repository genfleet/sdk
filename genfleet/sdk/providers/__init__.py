from __future__ import annotations

from ..schemas import ModelConfig
from .base import Provider

_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "openrouter"})

#: OpenRouter is an OpenAI-compatible gateway in front of every vendor.
#: `openrouter/<vendor>/<model>` selects the OpenAI provider with this base
#: URL and passes the vendor id through intact, so the same agent runs on any
#: model with one key. A non-empty `base_url` in the config still wins.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: OpenRouter's optional attribution headers (https://openrouter.ai/docs):
#: they credit the app in OpenRouter's rankings and are harmless anywhere
#: else. Sent only when the request goes to the gateway, so an explicit
#: `base_url` pointing at Groq or Ollama does not carry them.
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://genfleet.ai",
    "X-OpenRouter-Title": "Genfleet",
}


def _parse_model(model: str) -> tuple[str, str]:
    """Parse 'provider/model-name' into (provider, model_name)."""
    if "/" not in model:
        raise ValueError(
            f"Invalid model format: '{model}'. "
            f"Expected 'provider/model-name', e.g. 'openai/gpt-4o-mini', "
            f"'anthropic/claude-sonnet-4-6', 'gemini/gemini-1.5-pro', "
            f"'openrouter/google/gemini-2.5-flash'."
        )
    provider, _, model_name = model.partition("/")
    provider = provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Supported providers: {', '.join(sorted(_PROVIDERS))}."
        )
    if provider == "openrouter" and "/" not in model_name:
        raise ValueError(
            f"Invalid OpenRouter model: '{model}'. "
            f"Expected 'openrouter/<vendor>/<model>', "
            f"e.g. 'openrouter/google/gemini-2.5-flash'."
        )
    return provider, model_name


def provider_for(config: ModelConfig) -> Provider:
    """Return the correct Provider instance based on the provider/model format."""
    provider_key, model_name = _parse_model(config["model"])
    resolved = {**config, "model": model_name}

    if provider_key == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(resolved)

    if provider_key == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(resolved)

    if provider_key == "openrouter" and not resolved.get("base_url"):
        # A missing, None or empty base_url all mean "use the gateway"; without
        # this an `sk-or-` key would be sent to api.openai.com or $OPENAI_BASE_URL.
        resolved["base_url"] = _OPENROUTER_BASE_URL
    if provider_key == "openrouter" and resolved.get("base_url") == _OPENROUTER_BASE_URL:
        resolved["default_headers"] = {**_OPENROUTER_HEADERS, **(resolved.get("default_headers") or {})}

    from .openai import OpenAIProvider
    return OpenAIProvider(resolved)


__all__ = ["Provider", "provider_for"]
