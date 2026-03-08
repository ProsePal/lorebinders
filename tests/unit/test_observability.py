"""Unit tests for the observability and monitoring system."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from lorebinders import models
from lorebinders.agent import (
    create_analysis_agent,
    create_extraction_agent,
    create_summarization_agent,
)
from lorebinders.workflow import build_binder


@pytest.fixture
def run_config(tmp_path: Path) -> models.RunConfiguration:
    """Fixture providing a standard run configuration."""
    book_file = tmp_path / "book.txt"
    book_file.write_text("Chapter 1\nAlice.")
    return models.RunConfiguration(
        book_path=book_file,
        author_name="Test Author",
        book_title="Test Book",
        narrator_config=models.NarratorConfig(),
    )


@pytest.mark.anyio
async def test_build_binder_emits_both_callbacks_with_agents(
    run_config: models.RunConfiguration,
    tmp_path: Path,
) -> None:
    """Test that build_binder emits both progress and observation events.

    Uses real agents with TestModel.
    """
    progress_updates = []
    observations = []

    def handle_progress(update: models.ProgressUpdate) -> None:
        progress_updates.append(update)

    def handle_observation(event: models.ObservationEvent) -> None:
        observations.append(event)

    model = TestModel()

    ext_agent = create_extraction_agent()
    ext_agent.model = model

    ana_agent = create_analysis_agent()
    ana_agent.model = model

    sum_agent = create_summarization_agent()
    sum_agent.model = model

    fake_book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=1, title="Ch1", content="Alice content")
        ],
    )

    fake_storage = MagicMock()

    fake_storage.extraction_exists.return_value = False
    fake_storage.profile_exists.return_value = False
    fake_storage.summary_exists.return_value = False

    with (
        patch(
            "lorebinders.workflow.convert_to_text",
            return_value="Chapter 1\nAlice content",
        ),
        patch("lorebinders.workflow.ingest", return_value=fake_book),
        patch("lorebinders.workflow.generate_pdf_report"),
        patch("lorebinders.workflow.get_storage", return_value=fake_storage),
        patch(
            "lorebinders.workflow.ensure_workspace",
            return_value=tmp_path / "workspace",
        ),
    ):
        await build_binder(
            run_config,
            progress=handle_progress,
            on_observe=handle_observation,
            extraction_agent=ext_agent,
            analysis_agent=ana_agent,
            summarization_agent=sum_agent,
        )

    obs_stages = {o.stage for o in observations}
    assert "ingestion" in obs_stages
    assert "extraction" in obs_stages
    assert "reporting" in obs_stages
    assert "workflow" in obs_stages

    obs_types = {o.type for o in observations}
    assert models.ObservationType.STAGE_STARTED in obs_types
    assert models.ObservationType.STAGE_COMPLETED in obs_types

    assert any(
        o.type == models.ObservationType.AGENT_RUN_STARTED for o in observations
    )
