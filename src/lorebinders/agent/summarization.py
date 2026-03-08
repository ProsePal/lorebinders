"""Entity summarization using AI agents."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from lorebinders.agent.factory import (
    build_summarization_user_prompt,
    create_summarization_agent,
    load_prompt_from_assets,
    run_agent_async,
)
from lorebinders.models import (
    AgentDeps,
    Binder,
    EntityRecord,
    ObservationEvent,
    ProgressUpdate,
    SummarizerResult,
)
from lorebinders.settings import get_settings

if TYPE_CHECKING:
    from lorebinders.storage.provider import StorageProvider

logger = logging.getLogger(__name__)


def _format_context(details: dict) -> str:
    """Format the entity data into a readable string for the AI.

    Args:
        details: The entity details containing appearances.

    Returns:
        str: The formatted data.
    """
    lines = []
    for chap_num, appearance in details.items():
        lines.append(f"Chapter {chap_num}:")
        for trait, value in appearance.traits.items():
            val_str = (
                ", ".join(value) if isinstance(value, list) else str(value)
            )
            lines.append(f"  - {trait}: {val_str}")
    return "\n".join(lines)


async def _summarize_entity(
    category: str,
    name: str,
    agent: Agent[AgentDeps, SummarizerResult],
    prompt: str,
    storage: "StorageProvider",
    deps: AgentDeps,
    on_observe: Callable[[ObservationEvent], None] | None = None,
) -> str:
    """Summarize an entity using the AI agent with abstracted storage.

    Args:
        category: The category of the entity.
        name: The name of the entity.
        agent: The agent to use for summarization.
        prompt: The prompt to use for summarization.
        storage: The storage provider for persistence.
        deps: The dependencies to inject into the agent.
        on_observe: Optional callback for rich observation events.

    Returns:
        str: The summary text.
    """
    if storage.summary_exists(category, name):
        logger.debug(f"Loading cached summary for {category}: {name}")
        return storage.load_summary(category, name)

    logger.info(f"Summarizing {category}: {name}")
    try:
        result_data = await run_agent_async(
            agent, prompt, deps=deps, on_observe=on_observe
        )
        summary_text = result_data.summary
        storage.save_summary(category, name, summary_text)
        logger.debug(f"Summary saved for {category}: {name}")
        return summary_text

    except Exception as e:
        logger.error(f"Failed to summarize {name}: {e}")
        raise


async def summarize_binder(
    binder: Binder,
    storage: "StorageProvider",
    agent: Agent[AgentDeps, SummarizerResult] | None = None,
    deps: AgentDeps | None = None,
    progress: Callable[[ProgressUpdate], None] | None = None,
    on_observe: Callable[[ObservationEvent], None] | None = None,
) -> None:
    """Summarize entities in the binder asynchronously in-place.

    Includes throttling and abstracted storage.

    Args:
        binder: The refined binder model.
        storage: The storage provider for persistence.
        agent: The agent to use for summarization.
        deps: Optional dependencies for the agent.
        progress: Optional callback for progress updates.
        on_observe: Optional callback for rich observation events.
    """
    import asyncio

    if agent is None:
        agent = create_summarization_agent()

    if deps is None:
        deps = AgentDeps(
            settings=get_settings(),
            prompt_loader=load_prompt_from_assets,
        )

    semaphore = asyncio.Semaphore(10)
    tasks = []
    progress_state = [0]

    async def _throttled_summarize(
        e: "EntityRecord", p: str, total: int
    ) -> str:
        async with semaphore:
            res = await _summarize_entity(
                e.category,
                e.name,
                agent,
                p,
                storage,
                deps,
                on_observe=on_observe,
            )
            if progress:
                progress_state[0] += 1
                progress(
                    ProgressUpdate(
                        stage="summarization",
                        current=progress_state[0],
                        total=total,
                        message=f"Summarized {e.category}: {e.name}",
                    )
                )
            return res

    for category_record in binder.categories.values():
        for entity in category_record.entities.values():
            if not entity.summary and entity.appearances:
                context_str = _format_context(entity.appearances)
                prompt = build_summarization_user_prompt(
                    entity_name=entity.name,
                    category=entity.category,
                    context_data=context_str,
                )
                tasks.append((entity, prompt))

    total_tasks = len(tasks)
    if tasks:
        logger.info(f"Summarizing {total_tasks} entities...")
        actual_tasks = [
            _throttled_summarize(e, p, total_tasks) for e, p in tasks
        ]
        summaries = await asyncio.gather(*actual_tasks)
        for (entity, _), summary in zip(tasks, summaries, strict=True):
            entity.summary = summary
