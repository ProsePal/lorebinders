"""Unit tests for the workflow module."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lorebinders import models
from lorebinders.workflow import (
    _aggregate_to_binder,
    build_binder,
)


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Fixture providing a temporary workspace directory."""
    from lorebinders.settings import get_settings

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
        book_path=book_file,
        author_name="Test Author",
        book_title="Test Book",
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
            traits={"Role": "Hero"},
        ),
        models.EntityProfile(
            name="Alice",
            category="Characters",
            chapter_number=2,
            traits={"Age": "20"},
        ),
    ]

    binder = _aggregate_to_binder(profiles)

    assert "Characters" in binder.categories
    alice = binder.categories["Characters"].entities["Alice"]
    assert alice.appearances[1].traits == {"Role": "Hero"}
    assert alice.appearances[2].traits == {"Age": "20"}


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
            traits={"Role": "Hero"},
        )
    ]

    fake_storage = MagicMock()
    fake_storage.path = temp_workspace / "Test_Author" / "Test_Book"
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
            return_value={1: {"Characters": ["Alice"]}},
        ),
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

    mock_convert.assert_called_once_with(run_config.book_path)
    mock_ingest.assert_called_once_with(
        "Chapter 1\nAlice content", run_config.book_path.stem
    )
    mock_report.assert_called_once()
    assert (
        result
        == temp_workspace
        / "Test_Author"
        / "Test_Book"
        / "Test_Book_story_bible.pdf"
    )
