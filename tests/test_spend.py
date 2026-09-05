import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from lorebinders.agent.analysis import analyze_entities, analyze_entity_results
from lorebinders.agent.extraction import extract_book
from lorebinders.agent.factory import run_agent_async
from lorebinders.agent.spend import Spend, SpendError, estimate_cost
from lorebinders.agent.summarization import summarize_binder
from lorebinders.models import (
    AgentDeps,
    Binder,
    Book,
    CategoryRecord,
    Chapter,
    ChapterAppearances,
    EntityAppearance,
    EntityRecord,
    NarratorConfig,
    ObservationEvent,
    ObservationType,
    RunConfiguration,
)
from lorebinders.refinement.alias_resolution import resolve_aliases
from lorebinders.settings import Settings


@pytest.mark.anyio
async def test_spend_limit() -> None:
    """Verify Spend enforces ceiling threshold on reservation and release."""
    spend = Spend(limit=1.0)

    await spend.reserve(0.5)
    assert spend.reserved == 0.5
    await spend.release(0.5, actual_cost=0.5)
    assert spend.total == 0.5
    assert spend.reserved == 0.0

    await spend.reserve(0.5)
    assert spend.reserved == 0.5
    await spend.release(0.5, actual_cost=0.5)
    assert spend.total == 1.0
    assert spend.reserved == 0.0

    with pytest.raises(SpendError):
        await spend.reserve(0.1)


@pytest.mark.anyio
async def test_spend_no_limit() -> None:
    """Verify Spend allows unlimited spend when limit is None."""
    spend = Spend(limit=None)
    await spend.reserve(1000.0)
    assert spend.reserved == 1000.0
    await spend.release(1000.0, actual_cost=1000.0)
    assert spend.total == 1000.0
    assert spend.reserved == 0.0


@pytest.mark.anyio
async def test_run_agent_async_spend_error_propagates() -> None:
    """Verify SpendError propagates out of run_agent_async."""
    spend = Spend(limit=0.0)
    settings = Settings()
    deps = AgentDeps(
        settings=settings, prompt_loader=lambda x: "prompt", spend=spend
    )

    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    with patch("lorebinders.agent.spend.estimate_cost", return_value=1.0):
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            await run_agent_async(agent, "test prompt", deps)


def test_estimate_cost_slug_match() -> None:
    """Verify model cost estimation rates."""
    cost = estimate_cost("deepseek-v3.2", 1_000_000, 1_000_000)
    assert cost == 0.42

    cost_flash = estimate_cost("seed-1.6-flash", 1_000_000, 1_000_000)
    assert cost_flash == 0.20

    cost_unknown = estimate_cost("something-unknown", 1_000_000, 1_000_000)
    assert cost_unknown == 4.0


@pytest.mark.anyio
async def test_spend_reserve_and_release() -> None:
    """Verify reserve and release ceiling validation."""
    spend = Spend(limit=1.0)

    await spend.reserve(0.5)
    assert spend.reserved == 0.5

    with pytest.raises(
        SpendError, match="Spend ceiling exceeded: \\$1.10 of \\$1.00"
    ):
        await spend.reserve(0.6)

    await spend.release(0.5, actual_cost=0.5)
    assert spend.reserved == 0.0
    assert spend.total == 0.5

    with pytest.raises(
        SpendError, match="Spend ceiling exceeded: \\$1.10 of \\$1.00"
    ):
        await spend.reserve(0.6)

    unlimited = Spend(limit=None)
    await unlimited.reserve(50_000.0)
    await unlimited.release(50_000.0, actual_cost=50_000.0)
    assert unlimited.total == 50_000.0
    assert unlimited.reserved == 0.0


@pytest.mark.anyio
async def test_spend_release_ceiling_check_on_actual_cost() -> None:
    """Verify release enforces spend ceiling when recording actual cost."""
    spend = Spend(limit=1.0)
    await spend.reserve(0.5)
    with pytest.raises(
        SpendError, match="Spend ceiling exceeded: \\$1.20 of \\$1.00"
    ):
        await spend.release(estimated_cost=0.5, actual_cost=1.2)
    assert spend.total == 1.2
    assert spend.reserved == 0.0


@pytest.mark.anyio
async def test_spend_release_accounts_for_reserved() -> None:
    """Verify release ceiling check accounts for remaining reserved spend."""
    spend = Spend(limit=1.0)
    await spend.reserve(0.6)
    await spend.reserve(0.3)
    with pytest.raises(
        SpendError, match="Spend ceiling exceeded: \\$1.10 of \\$1.00"
    ):
        await spend.release(estimated_cost=0.3, actual_cost=0.5)
    assert spend.total == 0.5
    assert spend.reserved == pytest.approx(0.6)


@pytest.mark.anyio
async def test_spend_release_clamps_reserved_at_zero() -> None:
    """Verify release clamps reserved spend at zero instead of negative."""
    spend = Spend(limit=1.0)
    await spend.reserve(0.2)
    await spend.release(estimated_cost=0.5)
    assert spend.reserved == 0.0

    clean_spend = Spend(limit=1.0)
    await clean_spend.release(estimated_cost=0.5)
    assert clean_spend.reserved == 0.0


@pytest.mark.anyio
async def test_spend_check_ceiling_effective_reports_total() -> None:
    """Verify error reports total spend when total alone exceeds limit."""
    spend = Spend(limit=1.0)
    spend.total = 1.5
    spend.reserved = 0.5
    with pytest.raises(
        SpendError, match="Spend ceiling exceeded: \\$1.50 of \\$1.00"
    ):
        await spend.reserve(0.2)


@pytest.mark.anyio
async def test_run_agent_async_pre_dispatch_check() -> None:
    """Verify run_agent_async checks spend before dispatching agent.run."""
    spend = Spend(limit=1.0)
    spend.total = 1.5
    settings = Settings()
    deps = AgentDeps(
        settings=settings, prompt_loader=lambda x: "prompt", spend=spend
    )

    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    with patch.object(agent, "run") as mock_run:
        with pytest.raises(
            SpendError, match="Spend ceiling exceeded: \\$1.50 of \\$1.00"
        ):
            await run_agent_async(agent, "test prompt", deps)
        mock_run.assert_not_called()


@pytest.mark.anyio
async def test_concurrent_tasks_pre_dispatch_prevents_overshoot() -> None:
    """Verify pre-dispatch reservation prevents concurrent overshoot."""
    spend = Spend(limit=1.0)
    settings = Settings(max_concurrency=4)
    deps = AgentDeps(
        settings=settings, prompt_loader=lambda x: "prompt", spend=spend
    )
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)
    dispatch_count = 0
    original_run = agent.run

    async def mock_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal dispatch_count
        dispatch_count += 1
        await asyncio.sleep(0.02)
        return await original_run(*args, **kwargs)

    with patch.object(agent, "run", side_effect=mock_run):
        with patch("lorebinders.agent.spend.estimate_cost", return_value=1.0):
            results = await asyncio.gather(
                *(
                    run_agent_async(agent, f"prompt {i}", deps)
                    for i in range(8)
                ),
                return_exceptions=True,
            )

    assert dispatch_count == 1
    assert spend.total == 1.0
    assert spend.reserved == 0.0

    spend_errors = [r for r in results if isinstance(r, SpendError)]
    successes = [r for r in results if not isinstance(r, BaseException)]
    assert len(spend_errors) == 7
    assert len(successes) == 1


@pytest.mark.anyio
async def test_run_agent_async_exception_releases_reservation() -> None:
    """Verify reservation is released when agent execution fails."""
    spend = Spend(limit=1.0)
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda x: "prompt", spend=spend
    )
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    with patch.object(
        agent, "run", side_effect=RuntimeError("execution error")
    ):
        with patch("lorebinders.agent.spend.estimate_cost", return_value=0.5):
            with pytest.raises(RuntimeError, match="execution error"):
                await run_agent_async(agent, "prompt", deps)

    assert spend.reserved == 0.0
    assert spend.total == 0.0


@pytest.mark.anyio
async def test_run_agent_async_observation_failure_no_leak() -> None:
    """Verify observation failure does not leak spend reservation."""
    spend = Spend(limit=0.05)
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda x: "prompt", spend=spend
    )
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    def failing_observer(event: ObservationEvent) -> None:
        raise RuntimeError("observer error")

    for _ in range(5):
        with patch("lorebinders.agent.spend.estimate_cost", return_value=0.01):
            with pytest.raises(RuntimeError, match="observer error"):
                await run_agent_async(
                    agent, "prompt", deps, on_observe=failing_observer
                )

    assert spend.reserved == 0.0
    assert spend.total == 0.0

    with patch("lorebinders.agent.spend.estimate_cost", return_value=0.01):
        await run_agent_async(agent, "prompt", deps)

    assert spend.reserved == 0.0
    assert spend.total == 0.01


@pytest.mark.anyio
async def test_run_agent_async_deepcopy_failure_no_leak() -> None:
    """Verify settings deepcopy failure does not leak spend reservation."""
    spend = Spend(limit=0.05)
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda x: "prompt", spend=spend
    )
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    with patch("lorebinders.agent.spend.estimate_cost", return_value=0.01):
        with patch("copy.deepcopy", side_effect=RuntimeError("deepcopy error")):
            with pytest.raises(RuntimeError, match="deepcopy error"):
                await run_agent_async(
                    agent, "prompt", deps, model_settings={"temperature": 0.7}
                )

    assert spend.reserved == 0.0
    assert spend.total == 0.0


@pytest.mark.anyio
async def test_run_agent_async_reservation_accounts_for_output_tokens() -> None:
    """Verify reservation accounts for estimated output tokens."""
    spend = Spend(limit=1.0)
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda x: "prompt", spend=spend
    )
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    prompt = "a" * 400
    in_only_cost = estimate_cost(model.model_name, 100, 0)

    reserved_amount = 0.0
    original_run = agent.run

    async def mock_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal reserved_amount
        reserved_amount = spend.reserved
        return await original_run(*args, **kwargs)

    with patch.object(agent, "run", side_effect=mock_run):
        await run_agent_async(agent, prompt, deps)

    expected_cost = estimate_cost(model.model_name, 100, 100)
    assert reserved_amount == expected_cost
    assert reserved_amount > in_only_cost


@pytest.mark.anyio
async def test_run_agent_async_empty_prompt_reserves_minimum_tokens() -> None:
    """Verify empty prompt reserves minimum token count."""
    spend = Spend(limit=1.0)
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda x: "prompt", spend=spend
    )
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    reserved_amount = 0.0
    original_run = agent.run

    async def mock_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal reserved_amount
        reserved_amount = spend.reserved
        return await original_run(*args, **kwargs)

    with patch.object(agent, "run", side_effect=mock_run):
        await run_agent_async(agent, "", deps)

    expected_min_cost = estimate_cost(model.model_name, 1, 1)
    assert reserved_amount == expected_min_cost
    assert reserved_amount > 0.0


@pytest.mark.anyio
async def test_run_agent_async_prevents_double_release() -> None:
    """Verify agent run does not double release spend in finally block."""
    spend = Spend(limit=10.0)
    await spend.reserve(0.5)
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda x: "prompt", spend=spend
    )
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    await run_agent_async(agent, "test prompt", deps)

    assert spend.reserved == pytest.approx(0.5)


@pytest.mark.anyio
async def test_run_agent_async_prevents_double_release_on_error() -> None:
    """Verify release error does not trigger a duplicate release in finally."""
    spend = Spend(limit=0.5)
    await spend.reserve(0.3)
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda x: "prompt", spend=spend
    )
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    with patch("lorebinders.agent.spend.estimate_cost", side_effect=[0.1, 0.5]):
        with pytest.raises(SpendError):
            await run_agent_async(agent, "test prompt", deps)

    assert spend.reserved == pytest.approx(0.3)


@pytest.mark.anyio
async def test_concurrent_real_agent_runs_do_not_overshoot_ceiling() -> None:
    """Verify concurrent real agent runs do not exceed the spend ceiling."""
    model = TestModel()
    prompt = "a" * 208
    single_res = estimate_cost(model.model_name, 52, 52)
    ceiling = single_res * 2
    spend = Spend(limit=ceiling)
    settings = Settings(max_concurrency=10)
    deps = AgentDeps(
        settings=settings, prompt_loader=lambda x: "prompt", spend=spend
    )
    agent = Agent(model, deps_type=AgentDeps)

    results = await asyncio.gather(
        *(run_agent_async(agent, prompt, deps) for _ in range(10)),
        return_exceptions=True,
    )

    spend_errors = [r for r in results if isinstance(r, SpendError)]
    successes = [r for r in results if not isinstance(r, BaseException)]
    assert len(spend_errors) == 8
    assert len(successes) == 2
    assert spend.total <= ceiling
    assert spend.reserved == 0.0


@pytest.mark.anyio
async def test_cached_items_bypass_spend_ceiling() -> None:
    """Verify cached items succeed without model calls when ceiling exceeded."""
    binder = Binder(
        categories={
            "Characters": CategoryRecord(
                name="Characters",
                entities={
                    "Hero": EntityRecord(
                        name="Hero",
                        category="Characters",
                        appearances={
                            "Book 1": ChapterAppearances(
                                chapters={1: EntityAppearance()}
                            )
                        },
                    )
                },
            )
        }
    )
    storage = MagicMock()
    storage.summary_exists.return_value = True
    storage.load_summary.return_value = "Cached summary of Hero"

    spend = Spend(limit=1.0)
    spend.total = 1.5
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda x: "prompt", spend=spend
    )
    agent = AsyncMock()

    await summarize_binder(
        binder=binder,
        storage=storage,
        agent=agent,
        deps=deps,
    )

    assert binder.categories["Characters"].entities["Hero"].summary == (
        "Cached summary of Hero"
    )
    agent.run.assert_not_called()


@pytest.mark.anyio
async def test_extract_book_propagates_spend_error() -> None:
    """Verify that extract_book re-raises SpendError."""
    book = Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            Chapter(number=1, title="Ch1", content=""),
            Chapter(number=2, title="Ch2", content=""),
        ],
    )
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "prompt",
        spend=Spend(limit=1.0),
    )
    config = RunConfiguration(
        books=[],
        series_title="Test Series",
        author_name="Test Author",
        narrator_config=NarratorConfig(name="Narrator"),
    )
    storage = MagicMock()

    with patch(
        "lorebinders.agent.extraction._extract_chapter", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.side_effect = [
            SpendError("Spend ceiling exceeded"),
            (2, {}),
        ]
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            await extract_book(
                book=book,
                agent=AsyncMock(),
                deps=deps,
                categories=["Characters"],
                config=config,
                storage=storage,
            )


@pytest.mark.anyio
async def test_extract_book_spend_abort_emits_failure_metric() -> None:
    """Verify extract_book emits failure metric before re-raising SpendError."""
    book = Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            Chapter(number=1, title="Ch1", content=""),
            Chapter(number=2, title="Ch2", content=""),
        ],
    )
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "prompt",
        spend=Spend(limit=1.0),
    )
    config = RunConfiguration(
        books=[],
        series_title="Test Series",
        author_name="Test Author",
        narrator_config=NarratorConfig(name="Narrator"),
    )
    storage = MagicMock()
    observations: list[ObservationEvent] = []

    with patch(
        "lorebinders.agent.extraction._extract_chapter", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.side_effect = [
            RuntimeError("Chapter 1 failed"),
            SpendError("Spend ceiling exceeded"),
        ]
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            await extract_book(
                book=book,
                agent=AsyncMock(),
                deps=deps,
                categories=["Characters"],
                config=config,
                storage=storage,
                on_observe=observations.append,
            )

    metric_events = [
        o for o in observations if o.type == ObservationType.METRIC
    ]
    assert len(metric_events) == 1
    assert metric_events[0].metadata["stage"] == "extraction"
    assert metric_events[0].metadata["failed_count"] == 1
    assert metric_events[0].metadata["total_count"] == 2


@pytest.mark.anyio
async def test_analyze_entities_propagates_spend_error() -> None:
    """Verify that analyze_entities re-raises SpendError."""
    book = Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            Chapter(number=1, title="Ch1", content=""),
            Chapter(number=2, title="Ch2", content=""),
        ],
    )
    entities = {"Characters": {"Hero": [1], "Villain": [2]}}
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "prompt",
        spend=Spend(limit=1.0),
    )
    storage = MagicMock()

    with patch(
        "lorebinders.agent.analysis._analyze_chapter_block",
        new_callable=AsyncMock,
    ) as mock_analyze:
        mock_analyze.side_effect = [
            SpendError("Spend ceiling exceeded"),
            [],
        ]
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            await analyze_entities(
                entities=entities,
                book=book,
                agent=AsyncMock(),
                deps=deps,
                effective_traits={"Characters": []},
                storage=storage,
            )


@pytest.mark.anyio
async def test_analyze_entities_spend_abort_emits_failure_metric() -> None:
    """Verify analyze_entities emits failure metric before SpendError."""
    book = Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            Chapter(number=1, title="Ch1", content=""),
            Chapter(number=2, title="Ch2", content=""),
        ],
    )
    entities = {"Characters": {"Hero": [1], "Villain": [2]}}
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "prompt",
        spend=Spend(limit=1.0),
    )
    storage = MagicMock()
    observations: list[ObservationEvent] = []

    with patch(
        "lorebinders.agent.analysis._analyze_chapter_block",
        new_callable=AsyncMock,
    ) as mock_analyze:
        mock_analyze.side_effect = [
            RuntimeError("Chapter 1 analysis error"),
            SpendError("Spend ceiling exceeded"),
        ]
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            await analyze_entities(
                entities=entities,
                book=book,
                agent=AsyncMock(),
                deps=deps,
                effective_traits={"Characters": []},
                storage=storage,
                on_observe=observations.append,
            )

    metric_events = [
        o for o in observations if o.type == ObservationType.METRIC
    ]
    assert len(metric_events) == 1
    assert metric_events[0].metadata["stage"] == "analysis"
    assert metric_events[0].metadata["failed_count"] == 1
    assert metric_events[0].metadata["total_count"] == 2


@pytest.mark.anyio
async def test_analyze_entity_results_propagates_spend_error() -> None:
    """Verify analyze_entity_results re-raises SpendError."""
    book = Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            Chapter(number=1, title="Ch1", content=""),
        ],
    )
    entities = {"Characters": {"Hero": [1]}}
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "prompt",
        spend=Spend(limit=1.0),
    )

    with patch(
        "lorebinders.agent.analysis._analyze_chapter_results_block",
        new_callable=AsyncMock,
    ) as mock_analyze_block:
        mock_analyze_block.side_effect = SpendError("Spend ceiling exceeded")
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            await analyze_entity_results(
                entities=entities,
                book=book,
                agent=AsyncMock(),
                deps=deps,
                effective_traits={"Characters": []},
                raise_on_error=False,
            )


@pytest.mark.anyio
async def test_summarize_binder_propagates_spend_error() -> None:
    """Verify that summarize_binder re-raises SpendError."""
    binder = Binder(
        categories={
            "Characters": CategoryRecord(
                name="Characters",
                entities={
                    "Hero": EntityRecord(
                        name="Hero",
                        category="Characters",
                        appearances={
                            "Test Book": ChapterAppearances(
                                chapters={1: EntityAppearance()}
                            )
                        },
                    )
                },
            )
        }
    )
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "prompt",
        spend=Spend(limit=1.0),
    )
    storage = MagicMock()

    with patch(
        "lorebinders.agent.summarization._summarize_entity",
        new_callable=AsyncMock,
    ) as mock_summarize:
        mock_summarize.side_effect = SpendError("Spend ceiling exceeded")
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            await summarize_binder(
                binder=binder,
                storage=storage,
                agent=AsyncMock(),
                deps=deps,
            )


@pytest.mark.anyio
async def test_summarize_binder_spend_abort_emits_failure_metric() -> None:
    """Verify summarize_binder emits failure metric before SpendError."""
    binder = Binder(
        categories={
            "Characters": CategoryRecord(
                name="Characters",
                entities={
                    "Hero": EntityRecord(
                        name="Hero",
                        category="Characters",
                        appearances={
                            "Test Book": ChapterAppearances(
                                chapters={1: EntityAppearance()}
                            )
                        },
                    ),
                    "Villain": EntityRecord(
                        name="Villain",
                        category="Characters",
                        appearances={
                            "Test Book": ChapterAppearances(
                                chapters={1: EntityAppearance()}
                            )
                        },
                    ),
                },
            )
        }
    )
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "prompt",
        spend=Spend(limit=1.0),
    )
    storage = MagicMock()
    observations: list[ObservationEvent] = []

    with patch(
        "lorebinders.agent.summarization._summarize_entity",
        new_callable=AsyncMock,
    ) as mock_summarize:
        mock_summarize.side_effect = [
            RuntimeError("Hero summary failed"),
            SpendError("Spend ceiling exceeded"),
        ]
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            await summarize_binder(
                binder=binder,
                storage=storage,
                agent=AsyncMock(),
                deps=deps,
                on_observe=observations.append,
            )

    metric_events = [
        o for o in observations if o.type == ObservationType.METRIC
    ]
    assert len(metric_events) == 1
    assert metric_events[0].metadata["stage"] == "summarization"
    assert metric_events[0].metadata["failed_count"] == 1
    assert metric_events[0].metadata["total_count"] == 2


@pytest.mark.anyio
async def test_resolve_aliases_propagates_spend_error() -> None:
    """Verify that resolve_aliases re-raises SpendError."""
    binder = Binder(
        categories={
            "Characters": CategoryRecord(
                name="Characters",
                entities={
                    "Hero": EntityRecord(name="Hero", category="Characters"),
                    "Brave Hero": EntityRecord(
                        name="Brave Hero", category="Characters"
                    ),
                },
            )
        }
    )
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "prompt",
        spend=Spend(limit=1.0),
    )

    with patch(
        "lorebinders.refinement.alias_resolution._resolve_category_aliases",
        new_callable=AsyncMock,
    ) as mock_resolve:
        mock_resolve.side_effect = SpendError("Spend ceiling exceeded")
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            await resolve_aliases(
                binder=binder,
                agent=AsyncMock(),
                deps=deps,
            )
