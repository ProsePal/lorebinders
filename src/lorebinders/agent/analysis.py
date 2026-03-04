"""Entity analysis using AI agents."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable

from pydantic_ai import Agent

from lorebinders import models
from lorebinders.agent.factory import build_analysis_user_prompt
from lorebinders.storage.provider import StorageProvider
from lorebinders.types import SortedExtractions

logger = logging.getLogger(__name__)


async def _analyze_batch(
    target_categories: list[models.CategoryTarget],
    chapter: models.Chapter,
    agent: Agent[models.AgentDeps, list[models.AnalysisResult]],
    deps: models.AgentDeps,
    effective_traits: dict[str, list[str]],
    storage: StorageProvider,
) -> list[models.EntityProfile]:
    """Analyze a batch of entities with abstracted storage.

    Args:
        target_categories: The categories and entities to analyze.
        chapter: The chapter context for analysis.
        agent: The analysis agent.
        deps: Dependencies for the agent.
        effective_traits: Map of category to traits.
        storage: The storage provider for persistence.

    Returns:
        A list of analyzed entity profiles.
    """
    profiles: list[models.EntityProfile] = []
    to_analyze: list[models.CategoryTarget] = []

    for cat_target in target_categories:
        category = cat_target.name
        cached_names = [
            n
            for n in cat_target.entities
            if storage.profile_exists(chapter.number, category, n)
        ]
        run_names = [
            n
            for n in cat_target.entities
            if not storage.profile_exists(chapter.number, category, n)
        ]

        profiles.extend(
            storage.load_profile(chapter.number, category, n)
            for n in cached_names
        )

        if run_names:
            c_traits = effective_traits.get(category) or ["Description", "Role"]
            to_analyze.append(
                models.CategoryTarget(
                    name=category, entities=run_names, traits=c_traits
                )
            )

    if not to_analyze:
        return profiles

    full_prompt = build_analysis_user_prompt(
        context_text=chapter.content,
        categories=to_analyze,
    )
    result = await agent.run(full_prompt, deps=deps)

    for r in result.output:
        profile_traits: models.EntityTraits = {
            trait.trait: trait.value for trait in r.traits
        }
        p = models.EntityProfile(
            name=r.entity_name,
            category=r.category,
            chapter_number=chapter.number,
            traits=profile_traits,
            confidence_score=deps.settings.confidence_threshold,
        )
        storage.save_profile(chapter.number, p)
        profiles.append(p)

    return profiles


async def _analyze_chapter_block(
    chapter: models.Chapter,
    cat_map: dict[str, list[str]],
    agent: Agent[models.AgentDeps, list[models.AnalysisResult]],
    deps: models.AgentDeps,
    effective_traits: dict[str, list[str]],
    storage: StorageProvider,
    semaphore: asyncio.Semaphore,
    progress: Callable[[models.ProgressUpdate], None] | None,
    progress_state: list[int],
    total_tasks: int,
) -> list[models.EntityProfile]:
    """Analyze chapter categories sequentially for prefix caching.

    By processing a chapter's categories sequentially, the LLM provider's prefix
    cache is primed by the first category and hit by subsequent categories,
    since they share the identical chapter context prefix.
    """
    profiles = []

    async with semaphore:
        for category, names in cat_map.items():
            batch_targets = [
                models.CategoryTarget(name=category, entities=names)
            ]

            batch_profiles = await _analyze_batch(
                batch_targets, chapter, agent, deps, effective_traits, storage
            )
            profiles.extend(batch_profiles)

            if progress:
                progress_state[0] += 1
                msg = (
                    f"Analyzing batch {progress_state[0]}/{total_tasks} "
                    f"(Chapter {chapter.number})"
                )
                progress(
                    models.ProgressUpdate(
                        stage="analysis",
                        current=progress_state[0],
                        total=total_tasks,
                        message=msg,
                    )
                )

    return profiles


async def analyze_entities(
    entities: SortedExtractions,
    book: models.Book,
    agent: Agent[models.AgentDeps, list[models.AnalysisResult]],
    deps: models.AgentDeps,
    effective_traits: dict[str, list[str]],
    storage: StorageProvider,
    progress: Callable[[models.ProgressUpdate], None] | None = None,
) -> list[models.EntityProfile]:
    """Analyze all entities in parallel by chapter, preserving prefix caching.

    Args:
        entities: The sorted extractions to analyze.
        book: The book context.
        agent: The analysis agent.
        deps: Dependencies for the agent.
        effective_traits: Map of category to traits.
        storage: The storage provider for persistence.
        progress: Optional callback for progress updates.

    Returns:
        A list of all analyzed entity profiles.
    """
    chapter_map = {ch.number: ch for ch in book.chapters}

    chapter_entities: dict[int, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for category, entity_chapters in entities.items():
        for entity_name, chapters in entity_chapters.items():
            for chapter_num in chapters:
                chapter_entities[chapter_num][category].append(entity_name)

    total_tasks = sum(len(cat_map) for cat_map in chapter_entities.values())
    logger.info(
        f"Analyzing {total_tasks} entity batches "
        f"grouped into {len(chapter_entities)} parallel chapter blocks"
    )

    semaphore = asyncio.Semaphore(10)
    progress_state = [0]
    chapter_tasks = []

    for chapter_num, cat_map in chapter_entities.items():
        if not (chapter := chapter_map.get(chapter_num)):
            continue

        task = _analyze_chapter_block(
            chapter=chapter,
            cat_map=cat_map,
            agent=agent,
            deps=deps,
            effective_traits=effective_traits,
            storage=storage,
            semaphore=semaphore,
            progress=progress,
            progress_state=progress_state,
            total_tasks=total_tasks,
        )
        chapter_tasks.append(task)

    results = await asyncio.gather(*chapter_tasks)

    profiles = []
    for chapter_profiles in results:
        profiles.extend(chapter_profiles)

    return profiles
