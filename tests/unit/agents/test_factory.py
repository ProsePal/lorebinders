import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import PromptedOutput
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from lorebinders.agent.factory import (
    ConfiguredModel,
    _is_moderation_error,
    create_agent,
    create_extraction_agent,
    create_summarization_agent,
    load_prompt_from_assets,
    run_agent_async,
)
from lorebinders.agent.spend import Spend, estimate_cost
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


def test_create_agent_accepts_model_without_settings_setter() -> None:
    """Configured agents support models without a mutable settings field."""

    class ImmutableSettingsModel(TestModel):
        def __init__(self) -> None:
            self._settings_mutable = True
            super().__init__()
            self._settings_mutable = False

        def __setattr__(self, name: str, value: object) -> None:
            if name == "_settings" and not getattr(
                self, "_settings_mutable", True
            ):
                raise AttributeError(
                    "settings are immutable after construction"
                )
            super().__setattr__(name, value)

    agent = create_agent(
        ImmutableSettingsModel(),
        deps_type=AgentDeps,
        output_type=ExtractionResult,
        model_settings={"timeout": 60.0},
    )

    assert isinstance(agent.model, ConfiguredModel)
    configured_settings, _ = agent.model.prepare_request(
        None, ModelRequestParameters()
    )

    assert configured_settings == {"timeout": 60.0}


@pytest.mark.anyio
async def test_configured_model_request_merges_settings_sent_to_model() -> None:
    """Removing settings merging from ConfiguredModel.request breaks this."""
    received_settings: list[ModelSettings | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received_settings.append(info.model_settings)
        return ModelResponse(parts=[TextPart(content="complete")])

    agent = create_agent(
        FunctionModel(respond, settings={"temperature": 0.3}),
        deps_type=AgentDeps,
        output_type=str,
        model_settings={"timeout": 60.0},
        fallback=TestModel(),
    )

    result = await agent.run("test", model_settings={"max_tokens": 100})

    assert result.output == "complete"
    assert received_settings == [
        {"temperature": 0.3, "timeout": 60.0, "max_tokens": 100}
    ]


@pytest.mark.anyio
async def test_configured_model_stream_request_merges_settings() -> None:
    """Removing settings merging from request_stream breaks this."""
    received_settings: list[ModelSettings | None] = []

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        received_settings.append(info.model_settings)
        yield "complete"

    agent = create_agent(
        FunctionModel(stream_function=respond, settings={"temperature": 0.3}),
        deps_type=AgentDeps,
        output_type=str,
        model_settings={"timeout": 60.0},
        fallback=TestModel(),
    )

    async with agent.run_stream(
        "test", model_settings={"max_tokens": 100}
    ) as result:
        assert await result.get_output() == "complete"

    assert received_settings == [
        {"temperature": 0.3, "timeout": 60.0, "max_tokens": 100}
    ]


def test_configured_model_settings_merge_wrapped_and_configured_values() -> (
    None
):
    """Configured defaults supplement advertised wrapped settings."""
    model = ConfiguredModel(
        TestModel(settings={"temperature": 0.3}), {"timeout": 60.0}
    )

    assert model.settings == {"temperature": 0.3, "timeout": 60.0}


@pytest.mark.anyio
async def test_configured_model_count_tokens_delegates_to_wrapped_model() -> (
    None
):
    """ConfiguredModel preserves the wrapped model's token-counting support."""
    received_settings: list[ModelSettings | None] = []

    class CountingModel(TestModel):
        async def count_tokens(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> RequestUsage:
            received_settings.append(model_settings)
            return RequestUsage(input_tokens=7)

    model = ConfiguredModel(CountingModel(), {"timeout": 60.0})

    usage = await model.count_tokens([], None, ModelRequestParameters())

    assert usage.input_tokens == 7
    assert received_settings == [{"timeout": 60.0}]


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

    primary = TestModel(model_name="primary-test-model")
    fallback = TestModel(model_name="fallback-test-model")
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
    assert start_events[0].metadata.get("model") == "primary-test-model"
    assert start_events[0].metadata.get("model") == primary.model_name
    assert not str(start_events[0].metadata.get("model", "")).startswith(
        "fallback:"
    )

    completed_events = [
        o for o in observations if o.type == ObservationType.AGENT_RUN_COMPLETED
    ]
    assert len(completed_events) == 1
    assert (
        completed_events[0].message
        == f"Agent run completed with model {primary.model_name}"
    )
    assert completed_events[0].metadata.get("model") == "primary-test-model"


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

    primary = TestModel(model_name="primary-test-model")
    fallback = TestModel(model_name="fallback-test-model")
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
    assert completed_events[0].metadata.get("model") == "primary-test-model"


@pytest.mark.anyio
async def test_run_agent_async_fallback_serves_request_updates_model_and_spend(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify that when fallback serves the request, observations and spend
    reflect the fallback model.
    """
    observations: list[ObservationEvent] = []

    def on_observe(event: ObservationEvent) -> None:
        observations.append(event)

    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise ModelHTTPError(status_code=403, model_name="openai:gpt-5.4-nano")

    primary = FunctionModel(fail)
    fallback = TestModel(model_name="openrouter:bytedance-seed/seed-1.6")
    agent = create_agent(
        primary,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
        fallback=fallback,
    )

    spend = Spend()
    deps = AgentDeps(
        settings=get_settings(),
        prompt_loader=load_prompt_from_assets,
        spend=spend,
    )

    with caplog.at_level(logging.WARNING):
        await run_agent_async(
            agent, "test prompt", deps=deps, on_observe=on_observe
        )

    assert "Failed to re-derive actual model" not in caplog.text

    start_events = [
        o for o in observations if o.type == ObservationType.AGENT_RUN_STARTED
    ]
    assert len(start_events) == 1
    assert start_events[0].metadata.get("model") == "function:fail:"

    completed_events = [
        o for o in observations if o.type == ObservationType.AGENT_RUN_COMPLETED
    ]
    assert len(completed_events) == 1
    assert (
        completed_events[0].message
        == "Agent run completed with model openrouter:bytedance-seed/seed-1.6"
    )
    assert (
        completed_events[0].metadata.get("model")
        == "openrouter:bytedance-seed/seed-1.6"
    )

    metric_events = [
        o for o in observations if o.type == ObservationType.METRIC
    ]
    assert len(metric_events) == 1
    assert (
        metric_events[0].metadata.get("model")
        == "openrouter:bytedance-seed/seed-1.6"
    )

    in_tokens = int(metric_events[0].metadata.get("input_tokens") or 0)
    out_tokens = int(metric_events[0].metadata.get("output_tokens") or 0)
    expected_cost = estimate_cost(
        "openrouter:bytedance-seed/seed-1.6",
        in_tokens,
        out_tokens,
    )
    generic_cost = estimate_cost(
        "openai:gpt-5.4-nano",
        in_tokens,
        out_tokens,
    )
    assert spend.total == expected_cost
    assert spend.total < generic_cost


@pytest.mark.anyio
async def test_run_agent_async_no_model_response_in_history_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify warning is logged when message history has no ModelResponse with
    model_name.
    """
    observations: list[ObservationEvent] = []

    def on_observe(event: ObservationEvent) -> None:
        observations.append(event)

    primary = TestModel(model_name="primary-test-model")
    fallback = TestModel(model_name="fallback-test-model")
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
        res.all_messages = lambda: [
            ModelRequest(parts=[]),
        ]
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
    assert completed_events[0].metadata.get("model") == "primary-test-model"


@pytest.mark.anyio
async def test_run_agent_async_no_all_messages_attribute_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify warning is logged when result lacks all_messages attribute."""
    observations: list[ObservationEvent] = []

    def on_observe(event: ObservationEvent) -> None:
        observations.append(event)

    primary = TestModel(model_name="primary-test-model")
    agent = create_agent(
        primary,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
    )

    deps = AgentDeps(
        settings=get_settings(),
        prompt_loader=load_prompt_from_assets,
    )

    mock_res = MagicMock(spec=["output", "usage"])
    mock_res.output = ExtractionResult(results=[])
    mock_res.usage.return_value = MagicMock(input_tokens=10, output_tokens=5)

    async def mock_run(*args: Any, **kwargs: Any) -> Any:
        return mock_res

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
    assert completed_events[0].metadata.get("model") == "primary-test-model"
