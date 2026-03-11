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


def _report_extraction_progress(
    progress: Callable[[models.ProgressUpdate], None] | None,
    chapter: models.Chapter,
    idx: int,
    total: int,
) -> None:
    """Helper to report extraction progress.

    Args:
        progress: Progress callback.
        chapter: The chapter model.
        idx: Current index.
        total: Total count.
    """
    if not progress:
        return
    progress(
        models.ProgressUpdate(
            stage="extraction",
            current=idx,
            total=total,
            message=f"Extracting chapter {chapter.number}: {chapter.title}",
        )
    )


async def _perform_extraction(
    chapter: models.Chapter,
    agent: Agent[models.AgentDeps, models.ExtractionResult],
    deps: models.AgentDeps,
    categories: list[str],
    config: models.RunConfiguration,
    semaphore: asyncio.Semaphore,
    on_observe: Callable[[models.ObservationEvent], None] | None,
) -> dict[str, list[str]]:
    """Helper to perform extraction with concurrency limit.

    Args:
        chapter: The chapter model.
        agent: The AI agent.
        deps: Agent dependencies.
        categories: Categories to extract.
        config: Run configuration.
        semaphore: Concurrency semaphore.
        on_observe: Optional observation callback.

    Returns:
        A dictionary mapping categories to lists of extracted entity names.
    """
    async with semaphore:
        prompt = build_extraction_user_prompt(
            text=chapter.content,
            categories=categories,
            narrator=config.narrator_config,
        )
        raw = await run_agent_async(
            agent, prompt, deps=deps, on_observe=on_observe
        )
        return raw.to_dict()


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
    """Extract entities from a chapter with throttling and storage.

    Args:
        chapter: The chapter model.
        agent: The AI agent.
        deps: Agent dependencies.
        categories: Categories to extract.
        config: Run configuration.
        idx: Chapter index.
        total: Total chapters.
        semaphore: Concurrency semaphore.
        storage: Storage provider.
        progress: Optional progress callback.
        on_observe: Optional observation callback.

    Returns:
        A tuple of (chapter_number, extraction_data).
    """
    _report_extraction_progress(progress, chapter, idx, total)

    if storage.extraction_exists(chapter.number):
        logger.info(f"Loading cached extraction for chapter {chapter.number}")
        return chapter.number, storage.load_extraction(chapter.number)

    result = await _perform_extraction(
        chapter, agent, deps, categories, config, semaphore, on_observe
    )
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
        book: The book model.
        agent: The AI agent.
        deps: Agent dependencies.
        categories: Categories to extract.
        config: Run configuration.
        storage: Storage provider.
        progress: Optional progress callback.
        on_observe: Optional observation callback.

    Returns:
        A dictionary mapping chapter numbers to their extraction data.
    """
    total = len(book.chapters)
    logger.info(f"Extracting entities from {total} chapters")
    semaphore = asyncio.Semaphore(deps.settings.max_concurrency)

    tasks: list[asyncio.Task[tuple[int, dict[str, list[str]]]]] = []
    for i, chap in enumerate(book.chapters, 1):
        task = asyncio.create_task(
            _extract_chapter(
                chap,
                agent,
                deps,
                categories,
                config,
                i,
                total,
                semaphore,
                storage,
                progress,
                on_observe,
            )
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    extracted: dict[int, dict[str, list[str]]] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Extraction task failed: {r}")
            continue
        if isinstance(r, tuple):
            chapter_num, data = r
            extracted[chapter_num] = data

    return extracted
