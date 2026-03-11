import pytest

from lorebinders.agent_settings import get_model_settings as settings_config


def test_settings_config_openai() -> None:
    settings = settings_config("openai")

    assert settings.get("openai_reasoning_effort") == "low"


def test_settings_config_anthropic() -> None:
    settings = settings_config("anthropic")

    assert settings.get("anthropic_thinking") == {"type": "disabled"}


@pytest.mark.parametrize("provider", ["google-gla", "google-vertex"])
def test_settings_config_google(provider: str) -> None:
    settings = settings_config(provider)

    assert settings.get("google_thinking_config") == {"include_thoughts": False}


def test_settings_config_groq() -> None:
    settings = settings_config("groq")

    assert settings.get("groq_reasoning_format") == "hidden"


def test_settings_config_openrouter() -> None:
    settings = settings_config("openrouter")

    assert settings.get("openrouter_reasoning") == {"effort": "low"}


def test_settings_config_unknown_fallback() -> None:
    settings = settings_config("mistral")

    assert settings == {}
