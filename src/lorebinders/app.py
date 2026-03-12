import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from pydantic_ai import Agent

from lorebinders import models
from lorebinders.workflow import build_binder

logger = logging.getLogger(__name__)


def run(
    config: models.RunConfiguration,
    progress: Callable[[models.ProgressUpdate], None] | None = None,
    on_observe: Callable[[models.ObservationEvent], None] | None = None,
    extraction_agent: (
        Agent[models.AgentDeps, models.ExtractionResult] | None
    ) = None,
    analysis_agent: (
        Agent[models.AgentDeps, list[models.AnalysisResult]] | None
    ) = None,
    summarization_agent: (
        Agent[models.AgentDeps, models.SummarizerResult] | None
    ) = None,
) -> Path:
    """Execute the LoreBinders build pipeline.

    Args:
        config: The run configuration containing book path, author, title, etc.
        progress: Optional callback to report progress.
        on_observe: Optional callback for rich observation events.
        extraction_agent: Optional agent override.
        analysis_agent: Optional agent override.
        summarization_agent: Optional agent override.

    Returns:
        The path to the generated PDF report.
    """
    return asyncio.run(
        build_binder(
            config,
            progress=progress,
            on_observe=on_observe,
            extraction_agent=extraction_agent,
            analysis_agent=analysis_agent,
            summarization_agent=summarization_agent,
        )
    )
