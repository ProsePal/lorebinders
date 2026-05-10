import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import PromptedOutput

from lorebinders.agent.factory import (
    _is_moderation_error,
    create_agent,
    create_extraction_agent,
    create_summarization_agent,
    load_prompt_from_assets,
    run_agent_async,
)
from lorebinders.models import (
    AgentDeps,
    ExtractionResult,
    ObservationEvent,
    ObservationType,
    SummarizerResult,
)
from lorebinders.settings import get_settings


def test_is_moderation_error_true() -> None:
    exc = ModelHTTPError(status_code=403, model_name="test/model")
    assert _is_moderation_error(exc) is True


def test_is_moderation_error_false_wrong_status() -> None:
    exc = ModelHTTPError(status_code=500, model_name="test/model")
    assert _is_moderation_error(exc) is False


def test_is_moderation_error_false_wrong_type() -> None:
    assert _is_moderation_error(ValueError("boom")) is False


def test_create_agent_no_fallback_returns_plain_agent() -> None:
    primary = TestModel()
    agent = create_agent(
        primary,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
    )
    assert not isinstance(agent.model, FallbackModel)


def test_create_agent_with_fallback_wraps_in_fallback_model() -> None:
    primary = TestModel()
    fallback = TestModel()
    agent = create_agent(
        primary,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
        fallback=fallback,
    )
    assert isinstance(agent.model, FallbackModel)


def test_create_extraction_agent_accepts_output_type_override() -> None:
    agent = create_extraction_agent(
        output_type=PromptedOutput(ExtractionResult)
    )

    assert agent is not None


def test_create_summarization_agent_accepts_output_type_override() -> None:
    agent = create_summarization_agent(
        output_type=PromptedOutput(SummarizerResult)
    )

    assert agent is not None


@pytest.mark.anyio
async def test_run_agent_async_emits_metric_event() -> None:
    """Test that run_agent_async emits a METRIC event with token counts."""
    observations: list[ObservationEvent] = []

    def on_observe(event: ObservationEvent) -> None:
        observations.append(event)

    agent = create_extraction_agent()
    agent.model = TestModel()

    deps = AgentDeps(
        settings=get_settings(),
        prompt_loader=load_prompt_from_assets,
    )

    await run_agent_async(
        agent, "test prompt", deps=deps, on_observe=on_observe
    )

    metric_events = [
        o for o in observations if o.type == ObservationType.METRIC
    ]
    assert len(metric_events) == 1
    meta = metric_events[0].metadata
    assert isinstance(meta["input_tokens"], int)
    assert isinstance(meta["output_tokens"], int)
    assert isinstance(meta["total_tokens"], int)
    assert meta["input_tokens"] >= 0
    assert meta["output_tokens"] >= 0
    assert meta["total_tokens"] >= 0
    assert meta["total_tokens"] == meta["input_tokens"] + meta["output_tokens"]
    assert "model" in meta
