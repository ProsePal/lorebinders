from typing import Any

from lorebinders.agent.factory import (
    create_alias_resolution_agent,
    create_analysis_agent,
    create_extraction_agent,
    create_summarization_agent,
)
from lorebinders.settings import get_settings


def _get_settings(agent: Any) -> Any:
    model = agent.model
    if hasattr(model, "models"):
        model = model.models[0]
    # some models use _settings because settings is a property
    return getattr(model, "settings", getattr(model, "_settings", None))


def test_factories_carry_model_settings() -> None:
    settings = get_settings()

    extraction_agent = create_extraction_agent(settings)
    ms = _get_settings(extraction_agent)
    assert ms is not None
    assert "openrouter_reasoning" in ms or "openai_reasoning_effort" in ms

    analysis_agent = create_analysis_agent(settings)
    assert _get_settings(analysis_agent) is not None

    alias_agent = create_alias_resolution_agent(settings)
    assert _get_settings(alias_agent) is not None

    summary_agent = create_summarization_agent(settings)
    assert _get_settings(summary_agent) is not None


def test_unprefixed_model_string() -> None:
    settings = get_settings()
    settings.extraction_model = "z-ai/glm-4.5"  # no prefix
    settings.extraction_fallback_model = None
    agent = create_extraction_agent(settings)
    ms = _get_settings(agent)
    assert ms is not None
    assert "openrouter_reasoning" in ms
