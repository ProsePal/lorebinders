import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from pydantic_ai import Agent

from lorebinders.models import (
    ObservationEvent,
    ProgressUpdate,
    RunConfiguration,
)
from lorebinders.workflow import build_binder

logger = logging.getLogger(__name__)


def run(
    config: RunConfiguration,
    progress: Callable[[ProgressUpdate], None] | None = None,
    on_observe: Callable[[ObservationEvent], None] | None = None,
    log_file: Path | None = None,
    extraction_agent: Agent | None = None,
    analysis_agent: Agent | None = None,
    summarization_agent: Agent | None = None,
) -> Path:
    """Execute the LoreBinders build pipeline.

    Args:
        config: The run configuration containing book path, author, title, etc.
        progress: Optional callback to report progress.
        on_observe: Optional callback for rich observation events.
        log_file: Optional path to log file.
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
