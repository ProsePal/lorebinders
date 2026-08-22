from lorebinders.agent.factory import (
    create_alias_resolution_agent,
    create_analysis_agent,
    create_extraction_agent,
    create_summarization_agent,
)
from lorebinders.settings import get_settings


def test_factories_carry_model_settings():
    settings = get_settings()
    
    extraction_agent = create_extraction_agent(settings)
    assert extraction_agent.model_settings is not None
    assert ("openrouter_reasoning" in extraction_agent.model_settings or
            "openai_reasoning_effort" in extraction_agent.model_settings)

    analysis_agent = create_analysis_agent(settings)
    assert analysis_agent.model_settings is not None
    
    alias_agent = create_alias_resolution_agent(settings)
    assert alias_agent.model_settings is not None
    
    summary_agent = create_summarization_agent(settings)
    assert summary_agent.model_settings is not None

def test_unprefixed_model_string():
    settings = get_settings()
    settings.extraction_model = "z-ai/glm-4.5" # no prefix
    settings.extraction_fallback_model = None
    agent = create_extraction_agent(settings)
    assert agent.model_settings is not None
    assert "openrouter_reasoning" in agent.model_settings
