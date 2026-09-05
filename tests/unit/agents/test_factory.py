import logging
from typing import Any
from unittest.mock import patch

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


def test_create_agent_fallback_settings_independence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify primary and fallback get independent, provider-appropriate
    settings dicts that do not share object identity, ensuring mutation of
    one does not affect the other.
    """
    from lorebinders.settings import Settings

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    settings = Settings(
        extraction_model="openai:gpt-4",
        extraction_fallback_model="anthropic:claude-3-haiku",
    )

    agent = create_extraction_agent(settings=settings)
    assert isinstance(agent.model, FallbackModel)

    primary_model = agent.model.models[0]
    fallback_model = agent.model.models[1]

    assert primary_model.settings is not None
    assert fallback_model.settings is not None
    assert primary_model.settings.get("openai_reasoning_effort") == "low"
    assert fallback_model.settings.get("anthropic_thinking") == {
        "type": "disabled"
    }

    assert primary_model.settings is not fallback_model.settings

    primary_model.settings["timeout"] = 999.0
    assert fallback_model.settings.get("timeout") == 600.0


def test_create_agent_fallback_settings_same_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify primary and fallback do not share the same settings dict
    even if using the same provider.
    """
    from lorebinders.settings import Settings

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = Settings(
        extraction_model="openai:gpt-4",
        extraction_fallback_model="openai:gpt-3.5-turbo",
    )

    agent = create_extraction_agent(settings=settings)
    assert isinstance(agent.model, FallbackModel)

    primary_model = agent.model.models[0]
    fallback_model = agent.model.models[1]

    assert primary_model.settings is not None
    assert fallback_model.settings is not None

    assert primary_model.settings is not fallback_model.settings

    primary_model.settings["timeout"] = 999.0
    assert fallback_model.settings.get("timeout") == 600.0


@pytest.mark.anyio
async def test_run_agent_async_with_fallback_model_initial_model_name() -> None:
    """Verify that run_agent_async resolves initial model name to primary
    model.
    """
    observations: list[ObservationEvent] = []

    def on_observe(event: ObservationEvent) -> None:
        observations.append(event)

    primary = TestModel()
    fallback = TestModel()
    agent = create_agent(
        primary,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
        fallback=fallback,
    )

    deps = AgentDeps(
        settings=get_settings(),
        prompt_loader=load_prompt_from_assets,
    )

    await run_agent_async(
        agent, "test prompt", deps=deps, on_observe=on_observe
    )

    start_events = [
        o for o in observations if o.type == ObservationType.AGENT_RUN_STARTED
    ]
    assert len(start_events) == 1
    assert start_events[0].metadata.get("model") == primary.model_name
    assert not str(start_events[0].metadata.get("model", "")).startswith(
        "fallback:"
    )


@pytest.mark.anyio
async def test_run_agent_async_actual_model_rederivation_failure_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify exceptions in actual-model re-derivation log a warning and
    fall back to model.
    """
    observations: list[ObservationEvent] = []

    def on_observe(event: ObservationEvent) -> None:
        observations.append(event)

    primary = TestModel()
    fallback = TestModel()
    agent = create_agent(
        primary,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
        fallback=fallback,
    )

    deps = AgentDeps(
        settings=get_settings(),
        prompt_loader=load_prompt_from_assets,
    )

    original_run = agent.run

    async def mock_run(*args: Any, **kwargs: Any) -> Any:
        res = await original_run(*args, **kwargs)
        res.all_messages = lambda: (_ for _ in ()).throw(
            RuntimeError("message history unavailable")
        )
        return res

    with patch.object(agent, "run", side_effect=mock_run):
        with caplog.at_level(logging.WARNING):
            await run_agent_async(
                agent, "test prompt", deps=deps, on_observe=on_observe
            )

    assert "Failed to re-derive actual model" in caplog.text
    completed_events = [
        o for o in observations if o.type == ObservationType.AGENT_RUN_COMPLETED
    ]
    assert len(completed_events) == 1
    assert (
        completed_events[0].message
        == f"Agent run completed with model {primary.model_name}"
    )
