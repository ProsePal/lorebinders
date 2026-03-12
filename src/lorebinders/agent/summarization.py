"""Entity summarization using AI agents."""

import asyncio
import logging
from collections.abc import Callable

from pydantic_ai import Agent

from lorebinders import models
from lorebinders.agent.factory import (
    build_summarization_user_prompt,
    run_agent_async,
)
from lorebinders.storage.provider import StorageProvider

logger = logging.getLogger(__name__)


def _format_traits(traits: dict[str, list[str] | str]) -> list[str]:
    """Helper to format traits for context.

    Args:
        traits: The traits dictionary.

    Returns:
        A list of formatted trait strings.
    """
    lines = []
    for trait, value in traits.items():
        val_str = ", ".join(value) if isinstance(value, list) else str(value)
        lines.append(f"  - {trait}: {val_str}")
    return lines


def _format_context(details: dict[int, models.EntityAppearance]) -> str:
    """Format the entity data into a readable string for the AI.

    Args:
        details: Mapping of chapter numbers to entity appearances.

    Returns:
        A formatted string containing entity traits across chapters.
    """
    lines = []
    for chap_num, appearance in details.items():
        lines.append(f"Chapter {chap_num}:")
        lines.extend(_format_traits(appearance.traits))
    return "\n".join(lines)


async def _summarize_entity(
    category: str,
    name: str,
    agent: Agent[models.AgentDeps, models.SummarizerResult],
    prompt: str,
    storage: StorageProvider,
    deps: models.AgentDeps,
    on_observe: Callable[[models.ObservationEvent], None] | None = None,
) -> str:
    """Summarize an entity using the AI agent with abstracted storage.

    Args:
        category: The entity category.
        name: The entity name.
        agent: The summarization agent.
        prompt: The user prompt.
        storage: The storage provider.
        deps: Agent dependencies.
        on_observe: Optional observation callback.

    Returns:
        The generated summary text.
    """
    if storage.summary_exists(category, name):
        logger.debug(f"Loading cached summary for {category}: {name}")
        return storage.load_summary(category, name)

    logger.info(f"Summarizing {category}: {name}")
    try:
        result_data = await run_agent_async(
            agent, prompt, deps=deps, on_observe=on_observe
        )
        summary_text: str = result_data.summary
        storage.save_summary(category, name, summary_text)
        logger.debug(f"Summary saved for {category}: {name}")
        return summary_text
    except Exception as e:
        logger.error(f"Failed to summarize {name}: {e}")
        raise


def _update_progress(
    progress: Callable[[models.ProgressUpdate], None] | None,
    state: list[int],
    total: int,
    category: str,
    name: str,
) -> None:
    """Helper to update progress state.

    Args:
        progress: Progress callback.
        state: Shared state for progress tracking.
        total: Total entities.
        category: Entity category.
        name: Entity name.
    """
    if not progress:
        return
    state[0] += 1
    progress(
        models.ProgressUpdate(
            stage="summarization",
            current=state[0],
            total=total,
            message=f"Summarized {category}: {name}",
        )
    )


async def _throttled_summarize(
    entity: models.EntityRecord,
    prompt: str,
    total: int,
    semaphore: asyncio.Semaphore,
    agent: Agent[models.AgentDeps, models.SummarizerResult],
    storage: StorageProvider,
    deps: models.AgentDeps,
    on_observe: Callable[[models.ObservationEvent], None] | None,
    progress: Callable[[models.ProgressUpdate], None] | None,
    progress_state: list[int],
) -> str:
    """Run summary with concurrency limit and progress tracking.

    Args:
        entity: The entity record.
        prompt: The user prompt.
        total: Total entities.
        semaphore: Concurrency semaphore.
        agent: The AI agent.
        storage: The storage provider.
        deps: Agent dependencies.
        on_observe: Observation callback.
        progress: Progress callback.
        progress_state: Shared state for progress tracking.

    Returns:
        The generated summary text.
    """
    async with semaphore:
        res = await _summarize_entity(
            entity.category,
            entity.name,
            agent,
            prompt,
            storage,
            deps,
            on_observe=on_observe,
        )
        _update_progress(
            progress, progress_state, total, entity.category, entity.name
        )
        return res


def _collect_tasks_from_category(
    tasks: list[tuple[models.EntityRecord, str]],
    category_record: models.CategoryRecord,
) -> None:
    """Helper to collect tasks from a category record.

    Args:
        tasks: List to accumulate tasks.
        category_record: The category record.
    """
    for entity in category_record.entities.values():
        if entity.summary or not entity.appearances:
            continue
        context_str = _format_context(entity.appearances)
        prompt = build_summarization_user_prompt(
            entity_name=entity.name,
            category=entity.category,
            context_data=context_str,
        )
        tasks.append((entity, prompt))


def _collect_tasks(
    binder: models.Binder,
) -> list[tuple[models.EntityRecord, str]]:
    """Collect all entities that need summarization.

    Args:
        binder: The full binder model.

    Returns:
        A list of tuples containing the entity record and its summary prompt.
    """
    tasks: list[tuple[models.EntityRecord, str]] = []
    for category_record in binder.categories.values():
        _collect_tasks_from_category(tasks, category_record)
    return tasks


async def summarize_binder(
    binder: models.Binder,
    storage: StorageProvider,
    agent: Agent[models.AgentDeps, models.SummarizerResult],
    deps: models.AgentDeps,
    progress: Callable[[models.ProgressUpdate], None] | None = None,
    on_observe: Callable[[models.ObservationEvent], None] | None = None,
) -> None:
    """Summarize entities in the binder asynchronously in-place.

    Args:
        binder: The binder model to update.
        storage: The storage provider.
        agent: The agent instance to use.
        deps: The agent dependencies.
        progress: Optional progress callback.
        on_observe: Optional observation callback.
    """
    tasks = _collect_tasks(binder)
    if not tasks:
        return

    logger.info(f"Summarizing {len(tasks)} entities...")
    semaphore = asyncio.Semaphore(deps.settings.max_concurrency)
    progress_state = [0]

    chapter_tasks: list[asyncio.Task[str]] = []
    for e, p in tasks:
        task = asyncio.create_task(
            _throttled_summarize(
                e,
                p,
                len(tasks),
                semaphore,
                agent,
                storage,
                deps,
                on_observe,
                progress,
                progress_state,
            )
        )
        chapter_tasks.append(task)

    results = await asyncio.gather(*chapter_tasks, return_exceptions=True)
    for i, res in enumerate(results):
        entity, _ = tasks[i]
        if isinstance(res, Exception):
            logger.error(f"Summarization failed for {entity.name}: {res}")
            entity.summary = "Error during summarization."
        elif isinstance(res, str):
            entity.summary = res
        else:
            entity.summary = "Unexpected error."
