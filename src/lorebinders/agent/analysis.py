"""Entity analysis using AI agents."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from pydantic_ai import Agent

from lorebinders import models
from lorebinders.agent.factory import (
    build_analysis_user_prompt,
    run_agent_async,
)
from lorebinders.storage.provider import StorageProvider
from lorebinders.types import SortedExtractions

if TYPE_CHECKING:
    from lorebinders.settings import Settings

logger = logging.getLogger(__name__)

_AnalysisAgent: TypeAlias = Agent[models.AgentDeps, list[models.AnalysisResult]]
_ProgressCb: TypeAlias = Callable[[models.ProgressUpdate], None] | None
_ObserveCb: TypeAlias = Callable[[models.ObservationEvent], None] | None


def _get_target_names(
    storage: StorageProvider,
    ch_num: int,
    category: str,
    entities: list[str],
) -> tuple[list[str], list[str]]:
    """Helper to split entities into cached and to-run.

    Args:
        storage: The storage provider.
        ch_num: The chapter number.
        category: The entity category.
        entities: List of entity names.

    Returns:
        A tuple of (cached_names, run_names).
    """
    return storage.filter_cached_profiles(ch_num, category, entities)


async def _analyze_batch(
    target_categories: list[models.CategoryTarget],
    chapter: models.Chapter,
    agent: _AnalysisAgent,
    deps: models.AgentDeps,
    effective_traits: dict[str, list[str]],
    storage: StorageProvider,
    on_observe: _ObserveCb = None,
) -> list[models.EntityProfile]:
    """Analyze a batch of entities with abstracted storage.

    Args:
        target_categories: Categories and entities to analyze.
        chapter: The chapter model.
        agent: The AI agent for analysis.
        deps: Agent dependencies.
        effective_traits: Trait mapping for categories.
        storage: The storage provider.
        on_observe: Optional observation callback.

    Returns:
        A list of analyzed entity profiles.
    """
    profiles: list[models.EntityProfile] = []
    to_analyze: list[models.CategoryTarget] = []

    for cat_target in target_categories:
        cat = cat_target.name
        cached, run = _get_target_names(
            storage, chapter.number, cat, cat_target.entities
        )
        profiles.extend(
            storage.load_profile(chapter.number, cat, n) for n in cached
        )
        _prepare_run_targets(to_analyze, cat, run, effective_traits)

    if not to_analyze:
        return profiles

    prompt = build_analysis_user_prompt(
        context_text=chapter.content, categories=to_analyze
    )
    result = await run_agent_async(
        agent, prompt, deps=deps, on_observe=on_observe
    )
    _process_analysis_results(
        result, profiles, chapter.number, storage, deps.settings
    )
    return profiles


def _prepare_run_targets(
    targets: list[models.CategoryTarget],
    cat: str,
    names: list[str],
    traits_map: dict[str, list[str]],
) -> None:
    if not names:
        return
    traits = traits_map.get(cat) or ["Description", "Role"]
    targets.append(
        models.CategoryTarget(name=cat, entities=names, traits=traits)
    )


def _process_analysis_results(
    results: list[models.AnalysisResult],
    profiles: list[models.EntityProfile],
    ch_num: int,
    storage: StorageProvider,
    settings: "Settings",
) -> None:
    for r in results:
        profile_traits: dict[str, str | list[str]] = {
            t.trait: t.value for t in r.traits
        }
        p = models.EntityProfile(
            name=r.entity_name,
            category=r.category,
            chapter_number=ch_num,
            traits=profile_traits,
            confidence_score=settings.confidence_threshold,
        )
        storage.save_profile(ch_num, p)
        profiles.append(p)


def _update_analysis_progress(
    progress: _ProgressCb,
    state: list[int],
    total: int,
    ch_num: int,
) -> None:
    """Helper to update analysis progress."""
    if not progress:
        return
    state[0] += 1
    progress(
        models.ProgressUpdate(
            stage="analysis",
            current=state[0],
            total=total,
            message=f"Analyzing batch {state[0]}/{total} (Chapter {ch_num})",
        )
    )


async def _analyze_category_sequential(
    chapter: models.Chapter,
    cat_map: dict[str, list[str]],
    agent: _AnalysisAgent,
    deps: models.AgentDeps,
    traits: dict[str, list[str]],
    storage: StorageProvider,
    progress: _ProgressCb,
    state: list[int],
    total: int,
    on_observe: _ObserveCb,
) -> list[models.EntityProfile]:
    """Analyze categories in a chapter sequentially for prefix caching.

    Args:
        chapter: The chapter model.
        cat_map: Mapping of categories to entity names.
        agent: The AI agent.
        deps: Agent dependencies.
        traits: Trait mapping.
        storage: The storage provider.
        progress: Progress callback.
        state: Shared state for progress tracking.
        total: Total number of batches.
        on_observe: Observation callback.

    Returns:
        A list of analyzed entity profiles for the chapter.
    """
    profiles: list[models.EntityProfile] = []
    for category, names in cat_map.items():
        batch_targets = [models.CategoryTarget(name=category, entities=names)]
        batch_profiles = await _analyze_batch(
            batch_targets,
            chapter,
            agent,
            deps,
            traits,
            storage,
            on_observe=on_observe,
        )
        profiles.extend(batch_profiles)
        _update_analysis_progress(progress, state, total, chapter.number)
    return profiles


async def _analyze_chapter_block(
    chapter: models.Chapter,
    cat_map: dict[str, list[str]],
    agent: _AnalysisAgent,
    deps: models.AgentDeps,
    traits: dict[str, list[str]],
    storage: StorageProvider,
    semaphore: asyncio.Semaphore,
    progress: _ProgressCb,
    state: list[int],
    total: int,
    on_observe: _ObserveCb = None,
) -> list[models.EntityProfile]:
    """Analyze chapter categories with concurrency control.

    Args:
        chapter: The chapter model.
        cat_map: Mapping of categories to entity names.
        agent: The AI agent.
        deps: Agent dependencies.
        traits: Trait mapping.
        storage: The storage provider.
        semaphore: Concurrency semaphore.
        progress: Progress callback.
        state: Shared state for progress tracking.
        total: Total number of batches.
        on_observe: Observation callback.

    Returns:
        A list of analyzed entity profiles for the chapter.
    """
    async with semaphore:
        return await _analyze_category_sequential(
            chapter,
            cat_map,
            agent,
            deps,
            traits,
            storage,
            progress,
            state,
            total,
            on_observe,
        )


def _add_entity_to_chapters(
    ch_entities: dict[int, dict[str, list[str]]],
    category: str,
    name: str,
    chapters: list[int],
) -> None:
    for ch_num in chapters:
        ch_entities[ch_num][category].append(name)


def _group_entities_by_chapter(
    entities: SortedExtractions,
) -> dict[int, dict[str, list[str]]]:
    """Group extracted entities by chapter number.

    Args:
        entities: The sorted extractions mapping.

    Returns:
        A dictionary mapping chapter numbers to categories and entity names.
    """
    ch_entities: dict[int, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for category, ent_chapters in entities.items():
        for ent_name, chapters in ent_chapters.items():
            _add_entity_to_chapters(ch_entities, category, ent_name, chapters)
    return dict(ch_entities)


async def analyze_entities(
    entities: SortedExtractions,
    book: models.Book,
    agent: _AnalysisAgent,
    deps: models.AgentDeps,
    effective_traits: dict[str, list[str]],
    storage: StorageProvider,
    progress: _ProgressCb = None,
    on_observe: _ObserveCb = None,
) -> list[models.EntityProfile]:
    """Analyze all entities in parallel by chapter.

    Args:
        entities: Sorted extractions to analyze.
        book: The full book model.
        agent: The AI agent.
        deps: Agent dependencies.
        effective_traits: Trait mapping for all categories.
        storage: The storage provider.
        progress: Optional progress callback.
        on_observe: Optional observation callback.

    Returns:
        A complete list of analyzed entity profiles for the book.
    """
    ch_map = {ch.number: ch for ch in book.chapters}
    ch_entities = _group_entities_by_chapter(entities)

    total_tasks = sum(len(cat_map) for cat_map in ch_entities.values())
    logger.info(
        f"Analyzing {total_tasks} batches across {len(ch_entities)} chapters"
    )

    semaphore = asyncio.Semaphore(deps.settings.max_concurrency)
    state = [0]
    chapter_tasks: list[asyncio.Task[list[models.EntityProfile]]] = []

    for ch_num, cat_map in ch_entities.items():
        chapter = ch_map.get(ch_num)
        if not chapter:
            continue
        task = asyncio.create_task(
            _analyze_chapter_block(
                chapter,
                cat_map,
                agent,
                deps,
                effective_traits,
                storage,
                semaphore,
                progress,
                state,
                total_tasks,
                on_observe,
            )
        )
        chapter_tasks.append(task)

    results = await asyncio.gather(*chapter_tasks, return_exceptions=True)
    profiles: list[models.EntityProfile] = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Analysis task failed: {r}")
            continue
        if isinstance(r, list):
            profiles.extend(r)
    return profiles
