"""Entity cleaning logic for refinement using Binder models."""

import logging

from lorebinders.models import (
    Binder,
    CategoryRecord,
    EntityRecord,
)
from lorebinders.refinement.normalization import clean_entity_name
from lorebinders.refinement.patterns import NARRATOR_PATTERN
from lorebinders.types import EntityTraits

logger = logging.getLogger(__name__)


def clean_str(text: str) -> str:
    """Clean 'none found' from strings.

    Args:
        text: The string to clean.

    Returns:
        The cleaned string or an empty string if 'none found'.
    """
    return "" if text.lower().strip() == "none found" else text


def _clean_trait_value(value: str | list[str]) -> str | list[str] | None:
    """Helper to clean a single trait value.

    Args:
        value: The trait value (string or list of strings).

    Returns:
        The cleaned trait value or None.
    """
    if isinstance(value, str):
        return clean_str(value)
    if isinstance(value, list):
        cleaned = [clean_str(v) for v in value if clean_str(v)]
        return cleaned or None
    return None


def clean_traits(traits: EntityTraits) -> EntityTraits:
    """Recursively clean traits dictionary.

    Args:
        traits: The traits dictionary to clean.

    Returns:
        A new dictionary with cleaned trait values.
    """
    cleaned: EntityTraits = {}
    for key, value in traits.items():
        if c_val := _clean_trait_value(value):
            cleaned[key] = c_val
    return cleaned


def _replace_narrator_text(
    traits: EntityTraits, narrator_name: str
) -> EntityTraits:
    """Helper to replace narrator placeholders in traits.

    Args:
        traits: The traits dictionary.
        narrator_name: The name of the narrator.

    Returns:
        A new dictionary with narrator placeholders replaced.
    """
    final: EntityTraits = {}
    for k, v in traits.items():
        if isinstance(v, str):
            final[k] = NARRATOR_PATTERN.sub(narrator_name, v)
        elif isinstance(v, list):
            final[k] = [NARRATOR_PATTERN.sub(narrator_name, item) for item in v]
    return final


def _process_entity(
    cat_name: str,
    entity: EntityRecord,
    narrator_name: str | None,
) -> None:
    """Process a single entity during binder cleaning.

    Args:
        cat_name: The name of the category.
        entity: The entity record to process.
        narrator_name: Optional name of the narrator.
    """
    entity.name = clean_entity_name(entity.name, cat_name)

    if narrator_name:
        if NARRATOR_PATTERN.match(entity.name):
            entity.name = narrator_name

        for appearance in entity.appearances.values():
            appearance.traits = _replace_narrator_text(
                appearance.traits, narrator_name
            )


def _process_category(
    category: CategoryRecord,
    narrator_name: str | None,
) -> None:
    """Clean all entities in a category.

    Args:
        category: The category record to process.
        narrator_name: Optional name of the narrator.
    """
    entities_list = list(category.entities.values())
    category.entities.clear()

    for entity in entities_list:
        _process_entity(category.name, entity, narrator_name)

        category.entities[entity.name] = entity


def clean_binder(binder: Binder, narrator_name: str | None) -> Binder:
    """Full cleaning pipeline using Binder model.

    Args:
        binder: The binder model to clean.
        narrator_name: Optional name of the narrator.

    Returns:
        A new Binder instance with cleaned data.
    """
    logger.debug("Starting binder cleaning...")
    new_binder = Binder()

    for cat_name, category in binder.categories.items():
        _process_category(category, narrator_name)
        new_binder.categories[cat_name] = category

    return new_binder
