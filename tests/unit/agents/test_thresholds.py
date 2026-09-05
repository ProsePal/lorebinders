from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from lorebinders import models
from lorebinders.agent.analysis import analyze_entities
from lorebinders.agent.extraction import extract_book
from lorebinders.agent.summarization import summarize_binder
from lorebinders.settings import Settings


@pytest.fixture
def base_deps() -> models.AgentDeps:
    return models.AgentDeps(
        settings=Settings(
            failure_threshold=0.2,
            failure_threshold_min_count=2,
            max_concurrency=10,
        ),
        prompt_loader=lambda x: "Prompt",
    )


@pytest.fixture
def run_config() -> models.RunConfiguration:
    return models.RunConfiguration(
        series_title="Test Series",
        books=[models.BookInput(path=Path("dummy.txt"), title="Test Book")],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(is_1st_person=False),
    )


@pytest.fixture
def mock_storage() -> Any:
    from unittest.mock import MagicMock

    return MagicMock()


@pytest.mark.anyio
async def test_extraction_threshold_raises(
    base_deps: models.AgentDeps,
    run_config: models.RunConfiguration,
    mock_storage: Any,
) -> None:
    book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=1, title="Ch1", content=""),
            models.Chapter(number=2, title="Ch2", content=""),
            models.Chapter(number=3, title="Ch3", content=""),
            models.Chapter(number=4, title="Ch4", content=""),
            models.Chapter(number=5, title="Ch5", content=""),
        ],
    )

    with patch(
        "lorebinders.agent.extraction._extract_chapter", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.side_effect = [
            (1, {}),
            RuntimeError("Fail 1"),
            RuntimeError("Fail 2"),
            (4, {}),
            (5, {}),
        ]

        with pytest.raises(RuntimeError, match="exceeding 20% threshold"):
            await extract_book(
                book=book,
                agent=AsyncMock(),
                deps=base_deps,
                categories=["Characters"],
                config=run_config,
                storage=mock_storage,
            )


@pytest.mark.anyio
async def test_extraction_threshold_passes(
    base_deps: models.AgentDeps,
    run_config: models.RunConfiguration,
    mock_storage: Any,
) -> None:
    book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=1, title="Ch1", content=""),
            models.Chapter(number=2, title="Ch2", content=""),
            models.Chapter(number=3, title="Ch3", content=""),
            models.Chapter(number=4, title="Ch4", content=""),
            models.Chapter(number=5, title="Ch5", content=""),
        ],
    )

    with patch(
        "lorebinders.agent.extraction._extract_chapter", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.side_effect = [
            (1, {}),
            RuntimeError("Fail 1"),
            (3, {}),
            (4, {}),
            (5, {}),
        ]

        results = await extract_book(
            book=book,
            agent=AsyncMock(),
            deps=base_deps,
            categories=["Characters"],
            config=run_config,
            storage=mock_storage,
        )
        assert len(results) == 4
        assert 1 in results
        assert 3 in results
        assert 4 in results
        assert 5 in results


@pytest.mark.anyio
async def test_analysis_threshold_raises(
    base_deps: models.AgentDeps,
    run_config: models.RunConfiguration,
    mock_storage: Any,
) -> None:
    book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=i, title=f"Ch{i}", content="")
            for i in range(1, 6)
        ],
    )

    entities = {"Characters": {f"Ent{i}": [i] for i in range(1, 6)}}

    with patch(
        "lorebinders.agent.analysis._analyze_chapter_block",
        new_callable=AsyncMock,
    ) as mock_analyze:
        mock_analyze.side_effect = [
            [],
            RuntimeError("Fail 1"),
            RuntimeError("Fail 2"),
            [],
            [],
        ]

        with pytest.raises(RuntimeError, match="exceeding 20% threshold"):
            await analyze_entities(
                entities=entities,
                book=book,
                agent=AsyncMock(),
                deps=base_deps,
                effective_traits={"Characters": []},
                storage=mock_storage,
            )


@pytest.mark.anyio
async def test_analysis_threshold_passes(
    base_deps: models.AgentDeps,
    run_config: models.RunConfiguration,
    mock_storage: Any,
) -> None:
    book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=i, title=f"Ch{i}", content="")
            for i in range(1, 6)
        ],
    )

    entities = {"Characters": {f"Ent{i}": [i] for i in range(1, 6)}}

    with patch(
        "lorebinders.agent.analysis._analyze_chapter_block",
        new_callable=AsyncMock,
    ) as mock_analyze:
        mock_analyze.side_effect = [[], RuntimeError("Fail 1"), [], [], []]

        results = await analyze_entities(
            entities=entities,
            book=book,
            agent=AsyncMock(),
            deps=base_deps,
            effective_traits={"Characters": []},
            storage=mock_storage,
        )
        assert isinstance(results, list)


@pytest.mark.anyio
async def test_summarization_threshold_raises(
    base_deps: models.AgentDeps,
    run_config: models.RunConfiguration,
    mock_storage: Any,
) -> None:
    binder = models.Binder(
        categories={
            "Characters": models.CategoryRecord(
                name="Characters",
                entities={
                    f"Ent{i}": models.EntityRecord(
                        name=f"Ent{i}",
                        category="Characters",
                        appearances={
                            "Test Book": models.ChapterAppearances(
                                chapters={1: models.EntityAppearance()}
                            )
                        },
                    )
                    for i in range(1, 6)
                },
            )
        }
    )

    with patch(
        "lorebinders.agent.summarization._summarize_entity",
        new_callable=AsyncMock,
    ) as mock_summarize:
        mock_summarize.side_effect = [
            "Sum",
            RuntimeError("Fail 1"),
            RuntimeError("Fail 2"),
            "Sum",
            "Sum",
        ]

        with pytest.raises(RuntimeError, match="exceeding 20% threshold"):
            await summarize_binder(
                binder=binder,
                agent=AsyncMock(),
                deps=base_deps,
                storage=mock_storage,
            )


@pytest.mark.anyio
async def test_summarization_threshold_passes(
    base_deps: models.AgentDeps,
    run_config: models.RunConfiguration,
    mock_storage: Any,
) -> None:
    binder = models.Binder(
        categories={
            "Characters": models.CategoryRecord(
                name="Characters",
                entities={
                    f"Ent{i}": models.EntityRecord(
                        name=f"Ent{i}",
                        category="Characters",
                        appearances={
                            "Test Book": models.ChapterAppearances(
                                chapters={1: models.EntityAppearance()}
                            )
                        },
                    )
                    for i in range(1, 6)
                },
            )
        }
    )

    with patch(
        "lorebinders.agent.summarization._summarize_entity",
        new_callable=AsyncMock,
    ) as mock_summarize:
        mock_summarize.side_effect = [
            "Sum",
            RuntimeError("Fail 1"),
            "Sum",
            "Sum",
            "Sum",
        ]

        await summarize_binder(
            binder=binder,
            agent=AsyncMock(),
            deps=base_deps,
            storage=mock_storage,
        )
        assert binder.categories["Characters"].entities["Ent1"].summary == "Sum"


@pytest.mark.anyio
@pytest.mark.parametrize(("min_count", "aborts"), [(1, True), (2, False)])
async def test_extraction_min_count_floor_governs_small_books(
    min_count: int,
    aborts: bool,
    base_deps: models.AgentDeps,
    run_config: models.RunConfiguration,
    mock_storage: Any,
) -> None:
    """One failure in three chapters is 33%, over the 20% ratio.

    Only the min-count floor prevents the abort, so lowering the floor
    to 1 must make the same scenario raise.
    """
    base_deps.settings.failure_threshold_min_count = min_count
    book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=1, title="Ch1", content=""),
            models.Chapter(number=2, title="Ch2", content=""),
            models.Chapter(number=3, title="Ch3", content=""),
        ],
    )

    with patch(
        "lorebinders.agent.extraction._extract_chapter", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.side_effect = [
            (1, {}),
            RuntimeError("Fail 1"),
            (3, {}),
        ]

        if aborts:
            with pytest.raises(RuntimeError, match="exceeding 20% threshold"):
                await extract_book(
                    book=book,
                    agent=AsyncMock(),
                    deps=base_deps,
                    categories=["Characters"],
                    config=run_config,
                    storage=mock_storage,
                )
        else:
            results = await extract_book(
                book=book,
                agent=AsyncMock(),
                deps=base_deps,
                categories=["Characters"],
                config=run_config,
                storage=mock_storage,
            )
            assert len(results) == 2
            assert 1 in results
            assert 3 in results


@pytest.mark.anyio
async def test_small_n_loss_compounds_across_all_stages(
    base_deps: models.AgentDeps,
    run_config: models.RunConfiguration,
    mock_storage: Any,
) -> None:
    """A 3-chapter book can lose all content while passing gates.

    Compounding loss across extraction (33%), analysis (50%), and
    summarization (100%) can discard the entire book without triggering
    any per-stage threshold.
    """
    book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=1, title="Ch1", content=""),
            models.Chapter(number=2, title="Ch2", content=""),
            models.Chapter(number=3, title="Ch3", content=""),
        ],
    )

    with patch(
        "lorebinders.agent.extraction._extract_chapter", new_callable=AsyncMock
    ) as mock_extract:
        mock_extract.side_effect = [
            (
                1,
                {
                    "Characters": [
                        models.ExtractedEntity(
                            name="Ent1",
                            mentions=[1],
                            presence_type="literal_entity",
                        )
                    ]
                },
            ),
            RuntimeError("Fail Ch2 extraction"),
            (
                3,
                {
                    "Characters": [
                        models.ExtractedEntity(
                            name="Ent3",
                            mentions=[3],
                            presence_type="literal_entity",
                        )
                    ]
                },
            ),
        ]
        extraction_results = await extract_book(
            book=book,
            agent=AsyncMock(),
            deps=base_deps,
            categories=["Characters"],
            config=run_config,
            storage=mock_storage,
        )
        assert len(extraction_results) == 2
        assert 1 in extraction_results
        assert 3 in extraction_results

    surviving_entities = {"Characters": {"Ent1": [1], "Ent3": [3]}}

    with patch(
        "lorebinders.agent.analysis._analyze_chapter_block",
        new_callable=AsyncMock,
    ) as mock_analyze:
        mock_analyze.side_effect = [
            RuntimeError("Fail Ch1 analysis"),
            [
                models.EntityProfile(
                    name="Ent3",
                    category="Characters",
                    chapter_number=3,
                    book_title="Test Book",
                    traits={"Mood": ["Calm"]},
                )
            ],
        ]
        analysis_results = await analyze_entities(
            entities=surviving_entities,
            book=book,
            agent=AsyncMock(),
            deps=base_deps,
            effective_traits={"Characters": ["Mood"]},
            storage=mock_storage,
        )
        assert len(analysis_results) == 1
        assert analysis_results[0].name == "Ent3"

    binder = models.Binder(
        categories={
            "Characters": models.CategoryRecord(
                name="Characters",
                entities={
                    "Ent3": models.EntityRecord(
                        name="Ent3",
                        category="Characters",
                        appearances={
                            "Test Book": models.ChapterAppearances(
                                chapters={3: models.EntityAppearance()}
                            )
                        },
                    )
                },
            )
        }
    )

    with patch(
        "lorebinders.agent.summarization._summarize_entity",
        new_callable=AsyncMock,
    ) as mock_summarize:
        mock_summarize.side_effect = RuntimeError("Fail summarize Ent3")
        await summarize_binder(
            binder=binder,
            agent=AsyncMock(),
            deps=base_deps,
            storage=mock_storage,
        )
        assert binder.categories["Characters"].entities["Ent3"].summary is None
