"""Entity extraction using AI agents."""

import asyncio
import logging
from collections.abc import Callable

from pydantic_ai import Agent

from lorebinders import models
from lorebinders.agent.factory import (
    build_extraction_user_prompt,
    run_agent_async,
)
from lorebinders.storage.provider import StorageProvider

logger = logging.getLogger(__name__)


async def _extract_chapter(
    chapter: models.Chapter,
    agent: Agent[models.AgentDeps, models.ExtractionResult],
    deps: models.AgentDeps,
    categories: list[str],
    config: models.RunConfiguration,
    idx: int,
    total: int,
    semaphore: asyncio.Semaphore,
    storage: StorageProvider,
    progress: Callable[[models.ProgressUpdate], None] | None = None,
    on_observe: Callable[[models.ObservationEvent], None] | None = None,
) -> tuple[int, dict[str, list[str]]]:
    """Extract entities from a chapter.

    Uses throttling and abstracted storage.

    Args:
        chapter: The chapter to extract from.
        agent: The extraction agent.
        deps: Dependencies for the agent.
        categories: List of categories to extract.
        config: The run configuration.
        idx: The current chapter index.
        total: The total number of chapters.
        semaphore: Semaphore for concurrency control.
        storage: The storage provider for persistence.
        progress: Optional callback for progress updates.
        on_observe: Optional callback for rich observation events.

    Returns:
        A tuple of (chapter_number, extraction_data).
    """
    if progress:
        progress(
            models.ProgressUpdate(
                stage="extraction",
                current=idx,
                total=total,
                message=f"Extracting chapter {chapter.number}: {chapter.title}",
            )
        )

    if storage.extraction_exists(chapter.number):
        logger.info(f"Loading cached extraction for chapter {chapter.number}")
        result = storage.load_extraction(chapter.number)
    else:
        async with semaphore:
            prompt = build_extraction_user_prompt(
                text=chapter.content,
                categories=categories,
                narrator=config.narrator_config,
            )
            raw_result = await run_agent_async(
                agent, prompt, deps=deps, on_observe=on_observe
            )
            result = raw_result.to_dict()
            storage.save_extraction(chapter.number, result)

    return chapter.number, result


async def extract_book(
    book: models.Book,
    agent: Agent[models.AgentDeps, models.ExtractionResult],
    deps: models.AgentDeps,
    categories: list[str],
    config: models.RunConfiguration,
    storage: StorageProvider,
    progress: Callable[[models.ProgressUpdate], None] | None = None,
    on_observe: Callable[[models.ObservationEvent], None] | None = None,
) -> dict[int, dict[str, list[str]]]:
    """Extract entities from all chapters in parallel with throttling.

    Args:
        book: The book to extract from.
        agent: The extraction agent.
        deps: Dependencies for the agent.
        categories: List of categories to extract.
        config: The run configuration.
        storage: The storage provider for persistence.
        progress: Optional callback for progress updates.
        on_observe: Optional callback for rich observation events.

    Returns:
        A dictionary mapping chapter numbers to extraction data.
    """
    total_chapters = len(book.chapters)
    logger.info(f"Extracting entities from {total_chapters} chapters")

    semaphore = asyncio.Semaphore(10)

    tasks = [
        _extract_chapter(
            chapter,
            agent,
            deps,
            categories,
            config,
            idx,
            total_chapters,
            semaphore,
            storage,
            progress,
            on_observe,
        )
        for idx, chapter in enumerate(book.chapters, 1)
    ]

    results = await asyncio.gather(*tasks)
    return dict(results)
