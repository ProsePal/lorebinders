from pydantic_ai.models.fallback import FallbackModel

from lorebinders.agent.factory import create_extraction_agent
from lorebinders.settings import Settings


def test_fallback_model_does_not_inherit_primary_settings():  # type: ignore
    """A cross-provider fallback must not receive the primary's model_settings.

    Regression test for the leak where model_settings attached at the Agent
    level was shared across every candidate in a FallbackModel chain, so an
    OpenRouter-specific reasoning setting mutated in place by the primary's
    prepare_request would reach a non-OpenRouter fallback's request body.
    """
    settings = Settings(
        extraction_model="openrouter:z-ai/glm-4.5",
        extraction_fallback_model="openai:gpt-5.4-nano",
    )
    agent = create_extraction_agent(settings)
    assert isinstance(agent.model, FallbackModel)
    primary, fallback = agent.model.models

    assert primary.settings is not None
    assert "openrouter_reasoning" in primary.settings
    assert (
        fallback.settings is None
        or "openrouter_reasoning" not in fallback.settings
    )

    primary_merged, _ = primary.prepare_request(None, _fake_request_params())  # type: ignore
    fallback_merged, _ = fallback.prepare_request(None, _fake_request_params())  # type: ignore

    assert primary_merged is not None
    assert "extra_body" in primary_merged
    assert (fallback_merged or {}).get("extra_body") != primary_merged[
        "extra_body"
    ]
    assert "openrouter_reasoning" not in (fallback_merged or {})
    assert "extra_body" not in (fallback_merged or {})


def _fake_request_params():  # type: ignore
    from pydantic_ai.models import ModelRequestParameters

    return ModelRequestParameters()
