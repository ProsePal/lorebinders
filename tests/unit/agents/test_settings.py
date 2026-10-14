import pytest
from pydantic import ValidationError

from lorebinders.agent_settings import get_model_settings as settings_config
from lorebinders.settings import Settings


def test_limits_are_opt_in_for_library_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "LOREBINDERS_SPEND_CEILING",
        "LOREBINDERS_MAX_OUTPUT_TOKENS",
        "LOREBINDERS_EXTRACTION_MAX_OUTPUT_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.spend_ceiling is None
    assert settings.max_output_tokens is None
    assert settings.extraction_max_output_tokens is None
    assert "max_tokens" not in settings.model_settings_for("openrouter:test")


@pytest.mark.parametrize(
    "field, value",
    [
        ("spend_ceiling", 0),
        ("spend_ceiling", -1),
        ("spend_ceiling", float("inf")),
        ("max_output_tokens", 0),
        ("extraction_max_output_tokens", -1),
    ],
)
def test_cost_controls_reject_invalid_limits(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_cost_controls_load_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOREBINDERS_SPEND_CEILING", "1.75")
    monkeypatch.setenv("LOREBINDERS_MAX_OUTPUT_TOKENS", "18000")
    monkeypatch.setenv("LOREBINDERS_EXTRACTION_MAX_OUTPUT_TOKENS", "3000")
    settings = Settings()
    assert settings.spend_ceiling == 1.75
    assert (
        settings.model_settings_for("openrouter:z-ai/glm-4.5")["max_tokens"]
        == 18000
    )
    assert settings.extraction_max_output_tokens == 3000


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

    assert settings == {"timeout": 600.0}
