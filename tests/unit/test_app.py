from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lorebinders import models
from lorebinders.app import run


@pytest.fixture
def run_config(tmp_path: Path) -> models.RunConfiguration:
    return models.RunConfiguration(
        book_path=tmp_path / "book.txt",
        author_name="Test Author",
        book_title="Test Book",
        narrator_config=models.NarratorConfig(),
    )


def test_run_returns_path(run_config: models.RunConfiguration) -> None:
    """Test that run delegates to build_binder and returns a Path."""
    fake_path = Path("/fake/output/Test_Book_story_bible.pdf")

    async def mock_coro(*args: Any, **kwargs: Any) -> Path:
        return fake_path

    with patch(
        "lorebinders.app.build_binder",
        side_effect=mock_coro,
    ) as mock_build:
        result = run(run_config)

    assert mock_build.called
    assert result == fake_path


def test_run_passes_optional_args(run_config: models.RunConfiguration) -> None:
    """Test that agent overrides and progress callbacks are forwarded."""
    fake_path = Path("/fake/output/Test_Book_story_bible.pdf")
    fake_agent = MagicMock()

    def fake_progress(update: models.ProgressUpdate) -> None:
        pass

    async def mock_coro(*args: Any, **kwargs: Any) -> Path:
        return fake_path

    with patch(
        "lorebinders.app.build_binder",
        side_effect=mock_coro,
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
