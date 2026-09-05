from unittest.mock import patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from lorebinders.agent.factory import run_agent_async
from lorebinders.agent.spend import Spend, SpendError, estimate_cost
from lorebinders.models import AgentDeps
from lorebinders.settings import Settings


@pytest.mark.anyio
async def test_spend_limit() -> None:
    spend = Spend(limit=1.0)

    # At or below shouldn't raise
    await spend.add(0.5)
    assert spend.total == 0.5
    await spend.add(0.5)
    assert spend.total == 1.0

    # Exceeding should raise SpendError
    with pytest.raises(SpendError):
        await spend.add(0.1)


@pytest.mark.anyio
async def test_spend_no_limit() -> None:
    spend = Spend(limit=None)
    # Should never raise
    await spend.add(1000.0)
    assert spend.total == 1000.0


@pytest.mark.anyio
async def test_run_agent_async_spend_error_propagates() -> None:
    spend = Spend(limit=0.0)
    settings = Settings()
    deps = AgentDeps(
        settings=settings, prompt_loader=lambda x: "prompt", spend=spend
    )

    # Use a TestModel and mock usage so that cost > 0
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    with patch("lorebinders.agent.spend.estimate_cost", return_value=1.0):
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            # Mock estimate_cost to return 1.0 > 0.0 limit
            #
            # and estimate_cost will return 1.0, which > 0.0 limit
            await run_agent_async(agent, "test prompt", deps)


def test_estimate_cost_slug_match() -> None:
    cost = estimate_cost("deepseek-v3.2", 1_000_000, 1_000_000)
    assert cost == 0.42

    cost_flash = estimate_cost("seed-1.6-flash", 1_000_000, 1_000_000)
    assert cost_flash == 0.20

    cost_unknown = estimate_cost("something-unknown", 1_000_000, 1_000_000)
    assert cost_unknown == 4.0


@pytest.mark.anyio
async def test_spend_check() -> None:
    """Verify that Spend.check raises only when the ceiling is exceeded."""
    spend = Spend(limit=1.0)
    await spend.check()

    await spend.add(0.5)
    await spend.check()

    await spend.add(0.5)
    await spend.check()

    spend.total = 1.05
    with pytest.raises(SpendError, match="Spend ceiling exceeded"):
        await spend.check()

    unlimited = Spend(limit=None)
    unlimited.total = 100_000.0
    await unlimited.check()


def test_spend_exceeded_property() -> None:
    """Verify the exceeded property reflects ceiling state."""
    spend = Spend(limit=2.0)
    assert not spend.exceeded

    spend.total = 2.0
    assert not spend.exceeded

    spend.total = 2.01
    assert spend.exceeded

    unlimited = Spend(limit=None)
    unlimited.total = 999.0
    assert not unlimited.exceeded


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
async def test_concurrent_tasks_pre_dispatch_aborts_without_dispatch() -> None:
    """Verify queued tasks fail pre-dispatch once ceiling is exceeded."""
    import asyncio

    spend = Spend(limit=1.0)
    settings = Settings(max_concurrency=1)
    deps = AgentDeps(
        settings=settings, prompt_loader=lambda x: "prompt", spend=spend
    )
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)
    semaphore = asyncio.Semaphore(1)
    dispatch_count = 0

    async def worker(cost: float) -> str:
        nonlocal dispatch_count
        async with semaphore:
            if deps.spend is not None:
                await deps.spend.check()
            dispatch_count += 1
            with patch(
                "lorebinders.agent.spend.estimate_cost", return_value=cost
            ):
                return await run_agent_async(agent, "prompt", deps)

    results = await asyncio.gather(
        worker(1.5),
        worker(0.5),
        return_exceptions=True,
    )

    assert dispatch_count == 1
    assert isinstance(results[0], SpendError)
    assert isinstance(results[1], SpendError)


@pytest.mark.anyio
async def test_extract_book_propagates_spend_error() -> None:
    """Verify that extract_book re-raises SpendError."""
    from unittest.mock import AsyncMock, MagicMock

    from lorebinders.agent.extraction import extract_book
    from lorebinders.models import (
        Book,
        Chapter,
        NarratorConfig,
        RunConfiguration,
    )

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
async def test_analyze_entities_propagates_spend_error() -> None:
    """Verify that analyze_entities re-raises SpendError."""
    from unittest.mock import AsyncMock, MagicMock

    from lorebinders.agent.analysis import analyze_entities
    from lorebinders.models import Book, Chapter

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
async def test_summarize_binder_propagates_spend_error() -> None:
    """Verify that summarize_binder re-raises SpendError."""
    from unittest.mock import AsyncMock, MagicMock

    from lorebinders.agent.summarization import summarize_binder
    from lorebinders.models import (
        Binder,
        CategoryRecord,
        ChapterAppearances,
        EntityAppearance,
        EntityRecord,
    )

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
async def test_resolve_aliases_propagates_spend_error() -> None:
    """Verify that resolve_aliases re-raises SpendError."""
    from unittest.mock import AsyncMock

    from lorebinders.models import Binder, CategoryRecord, EntityRecord
    from lorebinders.refinement.alias_resolution import resolve_aliases

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
