from __future__ import annotations

from ..schemas import ModelConfig
from .base import Provider

PROVIDER_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "openrouter": "openai",
}

#: OpenRouter is an OpenAI-compatible gateway in front of every vendor.
#: `openrouter/<vendor>/<model>` selects the OpenAI provider with this base
#: URL and passes the vendor id through intact, so the same agent runs on any
#: model with one key. An explicit `base_url` in the config still wins.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
    if provider not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Supported providers: {', '.join(sorted(PROVIDER_MAP))}."
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

    if provider_key == "openrouter":
        resolved.setdefault("base_url", OPENROUTER_BASE_URL)

    from .openai import OpenAIProvider
    return OpenAIProvider(resolved)


__all__ = ["OPENROUTER_BASE_URL", "Provider", "provider_for"]
