"""Unit tests for the workflow module."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lorebinders import models
from lorebinders.models import ChapterAppearances
from lorebinders.settings import Settings, get_settings
from lorebinders.workflow import (
    _aggregate_to_binder,
    build_binder,
    extraction_categories,
    merge_traits,
)


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Fixture providing a temporary workspace directory."""
    ws = tmp_path / "work"
    ws.mkdir()
    os.environ["LOREBINDERS_WORKSPACE_BASE_PATH"] = str(ws)
    get_settings.cache_clear()
    return ws


@pytest.fixture
def book_file(tmp_path: Path) -> Path:
    """Fixture providing a temporary book file."""
    path = tmp_path / "book.txt"
    path.write_text("Chapter 1\nAlice.")
    return path


@pytest.fixture
def run_config(book_file: Path) -> models.RunConfiguration:
    """Fixture providing a standard run configuration."""
    return models.RunConfiguration(
        series_title="Test Series",
        books=[models.BookInput(path=book_file, title="Test Book")],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )


def _make_fake_book() -> models.Book:
    return models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=1, title="Ch1", content="Alice content")
        ],
    )


def test_aggregate_to_binder_structure() -> None:
    """Test that profiles are aggregated into binder structure."""
    profiles = [
        models.EntityProfile(
            name="Alice",
            category="Characters",
            chapter_number=1,
            book_title="Book 1",
            traits={"Role": "Hero"},
        ),
        models.EntityProfile(
            name="Alice",
            category="Characters",
            chapter_number=2,
            book_title="Book 1",
            traits={"Age": "20"},
        ),
    ]

    binder = _aggregate_to_binder(profiles, tracking="nested")

    assert "Characters" in binder.categories
    alice = binder.categories["Characters"].entities["Alice"]
    book_app = alice.appearances["Book 1"]
    assert isinstance(book_app, ChapterAppearances)
    assert book_app.chapters[1].traits == {"Role": "Hero"}
    assert book_app.chapters[2].traits == {"Age": "20"}


def test_merge_traits_includes_allusion_traits() -> None:
    """Test that allusions use their configured default traits."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )

    traits = merge_traits(Settings(), config)

    assert traits["Allusions"] == [
        "Invoked by",
        "Rhetorical significance",
        "What it reveals about the invoker",
    ]


def test_extraction_categories_excludes_allusions() -> None:
    """Test that Allusions is never sent to the extraction agent."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )

    categories = extraction_categories(Settings(), config)

    assert categories == ["Characters", "Locations"]
    assert "Allusions" not in categories


def test_extraction_categories_includes_custom_categories_and_traits() -> None:
    """Test that custom categories and custom trait keys are included."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
        custom_categories=["Factions"],
        custom_traits={"Objects": ["Origin"]},
    )

    categories = extraction_categories(Settings(), config)

    assert categories == ["Characters", "Locations", "Factions", "Objects"]


@pytest.mark.anyio
async def test_build_binder_orchestration(
    temp_workspace: Path,
    run_config: models.RunConfiguration,
) -> None:
    """Test end-to-end binder build orchestration with patched collaborators."""
    fake_book = _make_fake_book()
    fake_profiles = [
        models.EntityProfile(
            name="Alice",
            category="Characters",
            chapter_number=1,
            book_title="Test Book",
            traits={"Role": "Hero"},
        )
    ]

    fake_storage = MagicMock()
    fake_storage.path = temp_workspace / "Test_Author" / "Test_Series"
    fake_storage.extraction_exists.return_value = False
    fake_storage.profile_exists.return_value = False

    with (
        patch(
            "lorebinders.workflow.convert_to_text",
            return_value="Chapter 1\nAlice content",
        ) as mock_convert,
        patch(
            "lorebinders.workflow.ingest", return_value=fake_book
        ) as mock_ingest,
        patch(
            "lorebinders.workflow.extract_book",
            new_callable=AsyncMock,
            return_value={
                1: {
                    "Characters": [
                        models.ExtractedEntity(
                            name="Alice", presence_type="literal_entity"
                        )
                    ]
                }
            },
        ) as mock_extract,
        patch(
            "lorebinders.workflow.analyze_entities",
            new_callable=AsyncMock,
            return_value=fake_profiles,
        ),
        patch(
            "lorebinders.workflow.summarize_binder",
            new_callable=AsyncMock,
        ),
        patch("lorebinders.workflow.generate_pdf_report") as mock_report,
        patch("lorebinders.workflow.get_storage", return_value=fake_storage),
    ):
        result = await build_binder(run_config)

    mock_convert.assert_called_once_with(run_config.books[0].path)
    mock_ingest.assert_called_once_with(
        "Chapter 1\nAlice content", run_config.books[0].title
    )
    extraction_call_categories = mock_extract.call_args.args[3]
    assert "Allusions" not in extraction_call_categories
    mock_report.assert_called_once()
    assert (
        result
        == temp_workspace
        / "Test_Author"
        / "Test_Series"
        / "Test_Series_story_bible.pdf"
    )


@pytest.mark.anyio
async def test_build_binder_forwards_user_id(
    temp_workspace: Path,
    book_file: Path,
) -> None:
    """Test that config.user_id is forwarded to storage.set_workspace."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[models.BookInput(path=book_file, title="Test Book")],
        author_name="Test Author",
        user_id="user_123",
        narrator_config=models.NarratorConfig(),
    )
    fake_book = _make_fake_book()
    fake_storage = MagicMock()
    fake_storage.path = (
        temp_workspace / "user_123" / "Test_Author" / "Test_Series"
    )

    with (
        patch("lorebinders.workflow.convert_to_text", return_value=""),
        patch("lorebinders.workflow.ingest", return_value=fake_book),
        patch(
            "lorebinders.workflow.extract_book",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "lorebinders.workflow.analyze_entities",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("lorebinders.workflow.summarize_binder", new_callable=AsyncMock),
        patch("lorebinders.workflow.generate_pdf_report"),
        patch("lorebinders.workflow.get_storage", return_value=fake_storage),
    ):
        await build_binder(config)

    fake_storage.set_workspace.assert_called_once_with(
        "Test Author", "Test Series", user_id="user_123"
    )


@pytest.mark.anyio
async def test_build_binder_accumulates_stage_failures(
    temp_workspace: Path,
    book_file: Path,
) -> None:
    """Test that stage failures correctly accumulate across multiple books."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[
            models.BookInput(path=book_file, title="Book 1"),
            models.BookInput(path=book_file, title="Book 2"),
        ],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )
    fake_book = _make_fake_book()
    fake_storage = MagicMock()
    fake_storage.path = temp_workspace / "Test_Author" / "Test_Series"

    from typing import Any
    async def mock_extract(*args: Any, **kwargs: Any) -> dict[Any, Any]:
        on_obs = kwargs.get("on_observe") or args[7]
        on_obs(
            models.ObservationEvent(
                type=models.ObservationType.METRIC,
                stage="extraction",
                message="failed",
                metadata={
                    "failed_count": 1,
                    "total_count": 5,
                    "stage": "extraction",
                },
            )
        )
        return {}

    async def mock_analyze(*args: Any, **kwargs: Any) -> list[Any]:
        on_obs = kwargs.get("on_observe") or args[7]
        on_obs(
            models.ObservationEvent(
                type=models.ObservationType.METRIC,
                stage="analysis",
                message="failed",
                metadata={
                    "failed_count": 2,
                    "total_count": 10,
                    "stage": "analysis",
                },
            )
        )
        return []

    async def mock_summarize(*args: Any, **kwargs: Any) -> None:
        on_obs = kwargs.get("on_observe") or args[5]
        on_obs(
            models.ObservationEvent(
                type=models.ObservationType.METRIC,
                stage="summarization",
                message="failed",
                metadata={
                    "failed_count": 3,
                    "total_count": 15,
                    "stage": "summarization",
                },
            )
        )
        return None

    with (
        patch("lorebinders.workflow.convert_to_text", return_value=""),
        patch("lorebinders.workflow.ingest", return_value=fake_book),
        patch("lorebinders.workflow.extract_book", side_effect=mock_extract),
        patch(
            "lorebinders.workflow.analyze_entities", side_effect=mock_analyze
        ),
        patch(
            "lorebinders.workflow.summarize_binder", side_effect=mock_summarize
        ),
        patch("lorebinders.workflow.generate_pdf_report") as mock_report,
        patch("lorebinders.workflow.get_storage", return_value=fake_storage),
    ):
        await build_binder(config)

    # Verify that the PDF generator was called with accumulated counts
    binder = mock_report.call_args.args[0]
    
    # 2 books * 1 failed extraction each = 2 failed extractions
    assert binder.stage_failures["extraction"]["failed_count"] == 2
    assert binder.stage_failures["extraction"]["total_count"] == 10
    
    # 2 books * 2 failed analyses each = 4 failed analyses
    assert binder.stage_failures["analysis"]["failed_count"] == 4
    assert binder.stage_failures["analysis"]["total_count"] == 20
    
    # 1 summarize_binder call * 3 failed summarizations = 3
    assert binder.stage_failures["summarization"]["failed_count"] == 3
    assert binder.stage_failures["summarization"]["total_count"] == 15
