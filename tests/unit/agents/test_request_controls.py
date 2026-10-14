import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import PromptedOutput
from pydantic_ai.usage import RequestUsage

from lorebinders.agent.extraction import _perform_extraction
from lorebinders.agent.factory import create_agent, run_agent_async
from lorebinders.agent.spend import Spend, SpendError
from lorebinders.models import (
    AgentDeps,
    BookInput,
    Chapter,
    ExtractionResult,
    NarratorConfig,
    ObservationEvent,
    ObservationType,
    RunConfiguration,
)
from lorebinders.settings import Settings


def deps_for(spend: Spend | None = None, **limits: int) -> AgentDeps:
    return AgentDeps(
        settings=Settings(**limits),
        prompt_loader=lambda _: "prompt",
        spend=spend,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("override", [None, 1000])
async def test_original_agent_cap_cannot_be_removed(
    dynamic: bool, override: int | None
) -> None:
    received: list[int | None] = []
    settings_calls = 0

    def settings_callback(ctx: RunContext[AgentDeps]) -> dict[str, int]:
        nonlocal settings_calls
        settings_calls += 1
        return {"max_tokens": 20}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append((info.model_settings or {}).get("max_tokens"))
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(respond),
        deps_type=AgentDeps,
        model_settings=settings_callback if dynamic else {"max_tokens": 20},
    )
    assert (
        await run_agent_async(
            agent, "prompt", deps_for(), model_settings={"max_tokens": override}
        )
        == "done"
    )
    assert received == [20]
    assert settings_calls == (1 if dynamic else 0)


@pytest.mark.anyio
async def test_metric_callback_failure_does_not_discard_cost() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart("done")], usage=RequestUsage(cost=Decimal("0.3"))
        )

    def observe(event: ObservationEvent) -> None:
        if event.type == ObservationType.METRIC:
            raise RuntimeError("metric sink unavailable")

    spend = Spend(limit=1)
    agent = Agent(FunctionModel(respond), deps_type=AgentDeps)
    assert (
        await run_agent_async(
            agent, "prompt", deps_for(spend), on_observe=observe
        )
        == "done"
    )
    assert spend.total == pytest.approx(0.3)
    assert spend.reserved == 0


@pytest.mark.anyio
async def test_retry_checks_shared_budget_before_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[TextPart("invalid")], usage=RequestUsage(cost=Decimal("0.6"))
        )

    def fixed_price(*args: Any) -> float:
        return 0.5

    monkeypatch.setattr("lorebinders.agent.spend.estimate_cost", fixed_price)
    spend = Spend(limit=0.8)
    agent = Agent(
        FunctionModel(respond), deps_type=AgentDeps, output_type=int, retries=3
    )
    with pytest.raises(SpendError):
        await run_agent_async(agent, "prompt", deps_for(spend))
    assert calls == 1
    assert spend.total == pytest.approx(0.6)
    assert spend.reserved == 0


@pytest.mark.anyio
async def test_unpriced_response_uses_responding_model_rate() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart("done")],
            usage=RequestUsage(input_tokens=1000, output_tokens=1000),
        )

    agent = create_agent(
        FunctionModel(respond, model_name="seed-1.6-flash"),
        deps_type=AgentDeps,
        output_type=str,
        fallback=FunctionModel(respond, model_name="claude-3-5-sonnet"),
    )
    spend = Spend(limit=1)
    await run_agent_async(agent, "prompt", deps_for(spend))
    assert spend.total == pytest.approx(0.0002)


@pytest.mark.anyio
@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_cancellation_while_settling_releases_reservation(
    cancel_count: int,
) -> None:
    release_started = asyncio.Event()

    class BlockingSpend(Spend):
        async def release(
            self, estimated_cost: float = 0, actual_cost: float | None = None
        ) -> None:
            release_started.set()
            await super().release(estimated_cost, actual_cost)

    spend = BlockingSpend(limit=1)

    async def respond(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        await spend._lock.acquire()
        return ModelResponse(
            parts=[TextPart("done")], usage=RequestUsage(cost=Decimal("0.3"))
        )

    task = asyncio.create_task(
        run_agent_async(
            Agent(FunctionModel(respond), deps_type=AgentDeps),
            "prompt",
            deps_for(spend),
        )
    )
    await asyncio.wait_for(release_started.wait(), 2)
    for _ in range(cancel_count):
        task.cancel()
        await asyncio.sleep(0)
    spend._lock.release()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert spend.reserved == 0
    assert spend.total == pytest.approx(0.3)


@pytest.mark.anyio
async def test_custom_extraction_agent_obeys_stage_cap() -> None:
    received: list[int | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append((info.model_settings or {}).get("max_tokens"))
        return ModelResponse(parts=[TextPart('{"results": []}')])

    agent = Agent(
        FunctionModel(respond),
        deps_type=AgentDeps,
        output_type=PromptedOutput(ExtractionResult),
    )
    config = RunConfiguration(
        series_title="Story",
        books=[BookInput(path=Path("story.txt"), title="Story")],
        author_name="Author",
        narrator_config=NarratorConfig(is_1st_person=False),
    )
    await _perform_extraction(
        Chapter(number=1, title="One", content="story"),
        agent,
        deps_for(extraction_max_output_tokens=16),
        ["Characters"],
        config,
        asyncio.Semaphore(1),
        None,
    )
    assert received == [16]


@pytest.mark.anyio
async def test_billable_primary_blocks_fallback_when_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append("primary")
        return ModelResponse(
            parts=[TextPart("rejected")],
            usage=RequestUsage(cost=Decimal("0.6")),
        )

    def fallback(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        calls.append("fallback")
        return ModelResponse(parts=[TextPart("done")])

    def reject(response: ModelResponse) -> bool:
        return True

    def fixed_price(*args: Any) -> float:
        return 0.5

    monkeypatch.setattr("lorebinders.agent.spend.estimate_cost", fixed_price)
    agent = Agent(
        FallbackModel(
            FunctionModel(primary), FunctionModel(fallback), fallback_on=reject
        ),
        deps_type=AgentDeps,
    )
    spend = Spend(limit=0.8)
    with pytest.raises(SpendError):
        await run_agent_async(agent, "prompt", deps_for(spend))
    assert calls == ["primary"]
    assert spend.total == pytest.approx(0.6)
    assert spend.reserved == 0


@pytest.mark.anyio
async def test_existing_agent_overrides_remain_effective() -> None:
    received: list[int | None] = []

    def original(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        raise AssertionError("Original model must not run")

    def overridden(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        received.append((info.model_settings or {}).get("max_tokens"))
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(original), deps_type=AgentDeps)
    with agent.override(
        model=FunctionModel(overridden), model_settings={"max_tokens": 10}
    ):
        assert await run_agent_async(agent, "prompt", deps_for()) == "done"
    assert received == [10]


@pytest.mark.anyio
async def test_capability_cannot_raise_configured_limit() -> None:
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.settings import ModelSettings

    class LargerLimit(AbstractCapability[AgentDeps]):
        def get_model_settings(self) -> ModelSettings:
            return {"max_tokens": 1000}

    received: list[int | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append((info.model_settings or {}).get("max_tokens"))
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(respond),
        deps_type=AgentDeps,
        capabilities=[LargerLimit()],
    )
    await run_agent_async(agent, "prompt", deps_for(max_output_tokens=20))
    assert received == [20]
