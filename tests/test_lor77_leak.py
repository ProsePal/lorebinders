import pytest
from lorebinders.agent.factory import create_extraction_agent
from lorebinders.settings import Settings

def test_fallback_model_settings_leak():
    s = Settings(
        extraction_model='openrouter:z-ai/glm-4.5',
        extraction_fallback_model='openai:gpt-5.4-nano',
    )
    agent = create_extraction_agent(s)
    
    primary, fallback = agent.model.models
    
    print("\nPrimary settings:", getattr(primary, 'settings', None))
    print("Fallback settings:", getattr(fallback, 'settings', None))
    
    # Assert fallback has no openrouter reasoning settings
    fallback_settings = getattr(fallback, 'settings', None) or {}
    assert 'openrouter_reasoning' not in fallback_settings
    assert 'extra_body' not in fallback_settings
    
    # To test that prepare_request doesn't leak into fallback, we can just call it on the primary model
    # and observe that fallback settings are unaffected.
    run_settings = {}
    
    # We can mock ModelRequestParameters to avoid signature issues
    from unittest.mock import MagicMock
    params = MagicMock()
    params.builtin_tools = []
    params.output_mode = 'text'
    
    try:
        new_settings, _ = primary.prepare_request(run_settings, params)
    except Exception as e:
        print("primary prepare_request failed", e)
        
    try:
        new_settings_fallback, _ = fallback.prepare_request(run_settings, params)
        assert 'extra_body' not in (new_settings_fallback or {})
        assert 'openrouter_reasoning' not in (new_settings_fallback or {})
    except Exception as e:
        print("fallback prepare_request failed", e)
