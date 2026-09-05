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
) -> dict[str, list[models.ExtractedEntity]]:
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
        A dictionary mapping categories to lists of extracted entities.
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
    book_title: str,
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
) -> tuple[int, dict[str, list[models.ExtractedEntity]]]:
    """Extract entities from a chapter with throttling and storage.

    Args:
        chapter: The chapter model.
        book_title: The book title.
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

    if storage.extraction_exists(chapter.number, book_title):
        logger.info(f"Loading cached extraction for chapter {chapter.number}")
        return chapter.number, storage.load_extraction(
            chapter.number, book_title
        )

    result = await _perform_extraction(
        chapter, agent, deps, categories, config, semaphore, on_observe
    )
    storage.save_extraction(chapter.number, result, book_title)
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
) -> dict[int, dict[str, list[models.ExtractedEntity]]]:
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

    Note:
        Failure thresholds are per-stage and independent. The actual
        tolerated per-stage loss is max(threshold, (min_count-1)/N). For
        large N, up to ~1-(1-threshold)^3 total content loss can occur
        across the full pipeline. However, for a small number of tasks (N),
        the min-count floor dominates: e.g. at 3 tasks up to 33% loss is
        tolerated, at 2 tasks up to 50%, and at 1 task a 100% loss is
        tolerated silently. Compounded across all three stages
        (extraction, analysis, summarization), a 3-chapter book can lose
        the entire book (33% → 50% → 100%) while every individual per-stage
        gate passes. Output is only guaranteed up to this bound,
        not guaranteed to be complete.
    """
    total = len(book.chapters)
    logger.info(f"Extracting entities from {total} chapters")
    semaphore = asyncio.Semaphore(deps.settings.max_concurrency)

    tasks: list[
        asyncio.Task[tuple[int, dict[str, list[models.ExtractedEntity]]]]
    ] = []
    for i, chap in enumerate(book.chapters, 1):
        task = asyncio.create_task(
            _extract_chapter(
                chap,
                book.title,
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
    extracted: dict[int, dict[str, list[models.ExtractedEntity]]] = {}
    failed_count = 0
    for r in results:
        if isinstance(r, BaseException):
            if isinstance(
                r, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)
            ):
                raise r
            logger.error(f"Extraction task failed: {r}")
            failed_count += 1
            continue
        if isinstance(r, tuple):
            chapter_num, data = r
            extracted[chapter_num] = data

    if len(tasks) > 0:
        ratio = failed_count / len(tasks)
        models.emit_failure_metric(
            on_observe, "extraction", failed_count, len(tasks)
        )
        if (
            ratio > deps.settings.failure_threshold
            and failed_count >= deps.settings.failure_threshold_min_count
        ):
            raise RuntimeError(
                f"Extraction failed: {failed_count}/{len(tasks)} tasks failed, "
                f"exceeding {deps.settings.failure_threshold * 100:.0f}% "
                "threshold."
            )

    return extracted
