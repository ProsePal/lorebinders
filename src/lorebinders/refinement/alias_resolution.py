"""LLM alias resolution for entities the rule-based pass cannot match.

Rule-based deduplication only merges names that overlap as strings, so
semantic aliases ("The Dark Lord" for "Sauron") survive as separate entries.
This pass hands each category's surviving entities, with a short excerpt of
their notes, to a model that groups them under a canonical name.
"""

import asyncio
import logging
from collections.abc import Callable

from pydantic_ai import Agent

from lorebinders.agent.factory import (
    build_alias_resolution_user_prompt,
    run_agent_async,
)
from lorebinders.models import (
    AgentDeps,
    AliasGroup,
    AliasResolution,
    AppearanceValue,
    Binder,
    CategoryRecord,
    ChapterAppearances,
    EntityRecord,
    ObservationEvent,
    SingleAppearance,
)
from lorebinders.refinement.deduplication import merge_entities
from lorebinders.types import EntityTraits

logger = logging.getLogger(__name__)

AliasAgent = Agent[AgentDeps, AliasResolution]

_MAX_CONTEXT_CHARS = 400


def _trait_lines(traits: EntityTraits) -> list[str]:
    """Render an appearance's traits as ``trait: value`` fragments."""
    return [
        f"{trait}: {', '.join(value) if isinstance(value, list) else value}"
        for trait, value in traits.items()
    ]


def _appearance_lines(value: AppearanceValue) -> list[str]:
    """Render one appearance value, nested or flat, as trait fragments."""
    match value:
        case ChapterAppearances(chapters=chapters):
            return [
                line
                for appearance in chapters.values()
                for line in _trait_lines(appearance.traits)
            ]
        case SingleAppearance(appearance=appearance):
            return _trait_lines(appearance.traits)
        case _:
            return []


def _single_line(text: str) -> str:
    """Collapse whitespace so an excerpt cannot break the prompt's list."""
    return " ".join(text.split())


def entity_context(entity: EntityRecord) -> str:
    """Condense an entity's notes into a short excerpt for the model.

    Args:
        entity: The entity record to describe.

    Returns:
        A truncated single-line excerpt, empty when nothing is recorded.
    """
    if entity.summary:
        return _single_line(entity.summary)[:_MAX_CONTEXT_CHARS]

    lines = [
        line
        for value in entity.appearances.values()
        for line in _appearance_lines(value)
    ]
    return _single_line("; ".join(lines))[:_MAX_CONTEXT_CHARS]


def apply_alias_group(category: CategoryRecord, group: AliasGroup) -> int:
    """Merge one alias group into its canonical entity in-place.

    Names the model invented, already-merged names, repeated aliases within
    the group, and self-references are ignored rather than treated as errors,
    so a partially hallucinated response still contributes its usable groups.

    Args:
        category: The category record to update.
        group: The canonical name and aliases proposed by the model.

    Returns:
        The number of entities removed from the category.
    """
    lookup = {name.strip().lower(): name for name in category.entities}
    canonical = lookup.get(group.canonical_name.strip().lower())
    if canonical is None:
        logger.debug(
            "Ignoring alias group for unknown canonical name %r",
            group.canonical_name,
        )
        return 0

    target = category.entities[canonical]
    merged = 0
    for alias in group.aliases:
        name = lookup.get(alias.strip().lower())
        if name is None or name == canonical:
            logger.debug("Ignoring unusable alias %r", alias)
            continue
        source = category.entities.pop(name, None)
        if source is None:
            logger.debug("Ignoring repeated alias %r", alias)
            continue
        merge_entities(target, source)
        merged += 1

    return merged


async def _resolve_category_aliases(
    category: CategoryRecord,
    agent: AliasAgent,
    deps: AgentDeps,
    semaphore: asyncio.Semaphore,
    on_observe: Callable[[ObservationEvent], None] | None,
) -> None:
    """Run the alias pass for a single category and apply its groups."""
    if len(category.entities) < 2:
        return

    prompt = build_alias_resolution_user_prompt(
        category.name,
        [
            (name, entity_context(entity))
            for name, entity in category.entities.items()
        ],
    )

    async with semaphore:
        resolution = await run_agent_async(
            agent, prompt, deps=deps, on_observe=on_observe
        )

    merged = sum(
        apply_alias_group(category, group) for group in resolution.groups
    )
    logger.info(f"Alias resolution merged {merged} entities in {category.name}")


async def resolve_aliases(
    binder: Binder,
    agent: AliasAgent,
    deps: AgentDeps,
    on_observe: Callable[[ObservationEvent], None] | None = None,
) -> Binder:
    """Merge semantic aliases within each category of a Binder.

    Categories are resolved concurrently. A category whose model call fails
    keeps the entities the rule-based pass produced, so alias resolution
    never fails the pipeline.

    Args:
        binder: The Binder model to resolve, updated in-place.
        agent: The alias resolution agent.
        deps: Agent dependencies.
        on_observe: Optional observation callback.

    Returns:
        The Binder model with confirmed aliases merged.
    """
    semaphore = asyncio.Semaphore(deps.settings.max_concurrency)
    categories = list(binder.categories.values())

    results = await asyncio.gather(
        *(
            _resolve_category_aliases(
                category, agent, deps, semaphore, on_observe
            )
            for category in categories
        ),
        return_exceptions=True,
    )

    for category, result in zip(categories, results, strict=True):
        if isinstance(result, BaseException):
            logger.error(
                f"Alias resolution failed for {category.name}: {result}"
            )

    return binder


__all__ = ["apply_alias_group", "entity_context", "resolve_aliases"]
