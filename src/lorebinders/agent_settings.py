"""Set model settings for agents."""

from pydantic_ai import ModelSettings
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_ai.models.groq import GroqModelSettings
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.models.openrouter import OpenRouterModelSettings


def _openai_settings() -> ModelSettings:
    return OpenAIChatModelSettings(openai_reasoning_effort="low")


def _anthropic_settings() -> ModelSettings:
    return AnthropicModelSettings(anthropic_thinking={"type": "disabled"})


def _google_settings() -> ModelSettings:
    return GoogleModelSettings(
        google_thinking_config={"include_thoughts": False}
    )


def _groq_settings() -> ModelSettings:
    return GroqModelSettings(groq_reasoning_format="hidden")


def _openrouter_settings() -> ModelSettings:
    return OpenRouterModelSettings(openrouter_reasoning={"effort": "low"})


def get_model_settings(model_provider: str) -> ModelSettings:
    """Set model settings for agents.

    Args:
        model_provider (str): The model provider to use.

    Returns:
        ModelSettings: The model settings for the specified model provider.
    """
    configs = {
        "openai": _openai_settings,
        "anthropic": _anthropic_settings,
        "google-gla": _google_settings,
        "google-vertex": _google_settings,
        "groq": _groq_settings,
        "openrouter": _openrouter_settings,
    }

    if model_provider in configs:
        return configs[model_provider]()

    return ModelSettings()
