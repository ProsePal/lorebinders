"""Logic for sorting and deduplicating raw extractions."""

import logging
from collections import defaultdict

from lorebinders.refinement.deduplication import is_similar_key
from lorebinders.refinement.normalization import clean_entity_name
from lorebinders.refinement.patterns import NARRATOR_PATTERN
from lorebinders.types import SortedExtractions

logger = logging.getLogger(__name__)


def _replace_narrator_in_categories(
    categories: dict[str, list[str]], narrator_name: str
) -> dict[str, list[str]]:
    """Replace narrator references in category data.

    Args:
        categories: The categories mapping.
        narrator_name: The name of the narrator.

    Returns:
        A dictionary with narrator placeholders replaced by the narrator name.
    """
    result: dict[str, list[str]] = {
        category: [NARRATOR_PATTERN.sub(narrator_name, n) for n in names]
        for category, names in categories.items()
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


def _deduplicate_entity_names(names: list[str], category: str) -> list[str]:
    """Clean and deduplicate a list of entity names.

    Args:
        names: List of raw entity names.
        category: The entity category.

    Returns:
        A deduplicated list of cleaned entity names.
    """
    cleaned = [
        c
        for n in names
        if (c := clean_entity_name(n, category)) and len(c) >= 1
    ]

    canonical: list[str] = []
    for name in cleaned:
        idx = _find_similar_in_canonical(name, canonical)
        if idx == -1:
            canonical.append(name)
        elif len(name) > len(canonical[idx]):
            canonical[idx] = name

    return sorted(list(set(canonical)))


def _process_chapter_extractions(
    aggregated: SortedExtractions,
    chapter_num: int,
    categories: dict[str, list[str]],
) -> None:
    """Process all categories in a chapter extraction."""
    for category, names in categories.items():
        deduped = _deduplicate_entity_names(names, category)
        for name in deduped:
            _update_aggregated(aggregated, category, name, chapter_num)


def sort_extractions(
    raw_extractions: dict[int, dict[str, list[str]]],
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
