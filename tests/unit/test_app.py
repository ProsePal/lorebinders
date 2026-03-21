from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import Agent

from lorebinders import models
from lorebinders.app import run
from lorebinders.storage import FilesystemStorage, StorageProvider


@pytest.fixture
def run_config(tmp_path: Path) -> models.RunConfiguration:
    return models.RunConfiguration(
        series_title="Test Series",
        books=[models.BookInput(path=tmp_path / "book.txt", title="Test Book")],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )


def test_run_returns_path(run_config: models.RunConfiguration) -> None:
    """Test that run delegates to build_binder and returns a Path."""
    fake_path = Path("/fake/output/Test_Series_story_bible.pdf")

    async def mock_coroutine(
        config: models.RunConfiguration,
        progress: Callable[[models.ProgressUpdate], None] | None = None,
        on_observe: Callable[[models.ObservationEvent], None] | None = None,
        extraction_agent: Agent[models.AgentDeps, models.ExtractionResult]
        | None = None,
        analysis_agent: Agent[models.AgentDeps, list[models.AnalysisResult]]
        | None = None,
        summarization_agent: Agent[models.AgentDeps, models.SummarizerResult]
        | None = None,
        provider: type[StorageProvider] = FilesystemStorage,
    ) -> Path:
        return fake_path

    with patch(
        "lorebinders.app.build_binder",
        side_effect=mock_coroutine,
    ) as mock_build:
        result = run(run_config)

    assert mock_build.called
    assert result == fake_path


def test_run_passes_optional_args(run_config: models.RunConfiguration) -> None:
    """Test that agent overrides and progress callbacks are forwarded."""
    fake_path = Path("/fake/output/Test_Series_story_bible.pdf")
    fake_agent = MagicMock()

    def fake_progress(update: models.ProgressUpdate) -> None:
        pass

    async def mock_coroutine(
        config: models.RunConfiguration,
        progress: Callable[[models.ProgressUpdate], None] | None = None,
        on_observe: Callable[[models.ObservationEvent], None] | None = None,
        extraction_agent: Agent[models.AgentDeps, models.ExtractionResult]
        | None = None,
        analysis_agent: Agent[models.AgentDeps, list[models.AnalysisResult]]
        | None = None,
        summarization_agent: Agent[models.AgentDeps, models.SummarizerResult]
        | None = None,
        provider: type[StorageProvider] = FilesystemStorage,
    ) -> Path:
        return fake_path

    with patch(
        "lorebinders.app.build_binder",
        side_effect=mock_coroutine,
    ) as mock_build:
        result = run(
            run_config,
            progress=fake_progress,
            extraction_agent=fake_agent,
            analysis_agent=fake_agent,
            summarization_agent=fake_agent,
        )

    assert mock_build.called

    args, kwargs = mock_build.call_args
    assert args[0] == run_config
    assert kwargs["progress"] == fake_progress
    assert kwargs["extraction_agent"] == fake_agent
    assert kwargs["analysis_agent"] == fake_agent
    assert kwargs["summarization_agent"] == fake_agent
    assert result == fake_path
