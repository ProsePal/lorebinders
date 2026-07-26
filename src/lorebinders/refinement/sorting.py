"""Logic for sorting and deduplicating raw extractions."""

import logging
from collections import defaultdict

from lorebinders import models
from lorebinders.refinement.deduplication import is_similar_key
from lorebinders.refinement.normalization import clean_entity_name
from lorebinders.refinement.patterns import NARRATOR_PATTERN
from lorebinders.types import SortedExtractions

logger = logging.getLogger(__name__)


def _replace_narrator_in_categories(
    categories: dict[str, list[models.ExtractedEntity]], narrator_name: str
) -> dict[str, list[models.ExtractedEntity]]:
    """Replace narrator references in category data.

    Args:
        categories: The categories mapping.
        narrator_name: The name of the narrator.

    Returns:
        A dictionary with narrator placeholders replaced by the narrator name.
    """
    result: dict[str, list[models.ExtractedEntity]] = {
        category: [
            ent.model_copy(
                update={"name": NARRATOR_PATTERN.sub(narrator_name, ent.name)}
            )
            for ent in entities
        ]
        for category, entities in categories.items()
    }
    return result


def _update_aggregated(
    aggregated: SortedExtractions,
    category: str,
    name: str,
    chapter_num: int,
) -> None:
    """Helper to update aggregated extractions dictionary."""
    if name not in aggregated[category]:
        aggregated[category][name] = []
    if chapter_num not in aggregated[category][name]:
        aggregated[category][name].append(chapter_num)


def _find_similar_in_canonical(name: str, canonical: list[str]) -> int:
    """Find index of similar name in canonical list.

    Args:
        name: The name to find.
        canonical: List of canonical names.

    Returns:
        The index of the similar name, or -1 if no match is found.
    """
    return next(
        (
            i
            for i, existing in enumerate(canonical)
            if is_similar_key(name, existing)
        ),
        -1,
    )


def _deduplicate_entities(
    entities: list[models.ExtractedEntity], category: str
) -> list[models.ExtractedEntity]:
    """Clean and deduplicate a list of entities.

    Args:
        entities: List of raw extracted entities.
        category: The entity category.

    Returns:
        A deduplicated list of cleaned entities.
    """
    cleaned: list[models.ExtractedEntity] = []
    for ent in entities:
        c = clean_entity_name(ent.name, category)
        if c and len(c) >= 1:
            cleaned.append(ent.model_copy(update={"name": c}))

    canonical: list[models.ExtractedEntity] = []
    for ent in cleaned:
        canonical_names = [c.name for c in canonical]
        idx = _find_similar_in_canonical(ent.name, canonical_names)
        if idx == -1:
            canonical.append(ent)
        elif len(ent.name) > len(canonical[idx].name):
            canonical[idx] = ent

    canonical.sort(key=lambda x: x.name)
    return canonical


def _process_chapter_extractions(
    aggregated: SortedExtractions,
    chapter_num: int,
    categories: dict[str, list[models.ExtractedEntity]],
) -> None:
    """Process all categories in a chapter extraction."""
    for category, entities in categories.items():
        deduped = _deduplicate_entities(entities, category)
        for ent in deduped:
            target_category = (
                "Allusions"
                if ent.presence_type == "allusive_figure"
                else category
            )
            _update_aggregated(
                aggregated, target_category, ent.name, chapter_num
            )


def sort_extractions(
    raw_extractions: dict[int, dict[str, list[models.ExtractedEntity]]],
    narrator_name: str | None = None,
) -> SortedExtractions:
    """Aggregates, cleans, and deduplicates raw extractions.

    Args:
        raw_extractions: Mapping of chapter number to extractions.
        narrator_name: Optional name of the narrator.

    Returns:
        A SortedExtractions mapping of category -> name -> [chapters].
    """
    aggregated: SortedExtractions = defaultdict(dict)
    for chapter_num, categories in raw_extractions.items():
        effective_cats = (
            _replace_narrator_in_categories(categories, narrator_name)
            if narrator_name
            else categories
        )
        _process_chapter_extractions(aggregated, chapter_num, effective_cats)

    return dict(aggregated)
