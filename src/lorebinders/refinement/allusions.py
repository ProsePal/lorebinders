"""Resolution of allusive figures into the entities that invoke them."""

from lorebinders.models import (
    AppearanceValue,
    Binder,
    ChapterAppearances,
    EntityAppearance,
    EntityRecord,
    SingleAppearance,
)
from lorebinders.refinement.deduplication import is_similar_key
from lorebinders.refinement.normalization import merge_values
from lorebinders.types import EntityTraits

_ALLUSIONS_CATEGORY = "Allusions"
_INVOKED_BY_TRAIT = "Invoked by"
_RHETORICAL_TRAITS = (
    "Rhetorical significance",
    "What it reveals about the invoker",
)


def _find_invoker(binder: Binder, invoked_by: str) -> EntityRecord | None:
    """Find the entity record matching an "Invoked by" attribution.

    Args:
        binder: The Binder to search, excluding the Allusions category.
        invoked_by: The name the allusion was attributed to.

    Returns:
        The matching EntityRecord, or None if no entity matches.
    """
    return next(
        (
            entity
            for category_name, category in binder.categories.items()
            if category_name != _ALLUSIONS_CATEGORY
            for name, entity in category.entities.items()
            if is_similar_key(name, invoked_by)
        ),
        None,
    )


def _rhetorical_traits(traits: EntityTraits) -> EntityTraits:
    """Extract only the rhetorical traits worth attaching to an invoker."""
    return {key: traits[key] for key in _RHETORICAL_TRAITS if key in traits}


def _attach_to_flat_appearance(
    invoker: EntityRecord, key: str, traits: EntityTraits
) -> None:
    """Merge traits into an invoker's flat-tracked appearance in-place."""
    existing = invoker.appearances.get(key)
    if isinstance(existing, SingleAppearance):
        existing.appearance.traits = merge_values(
            existing.appearance.traits, traits
        )
    else:
        invoker.appearances[key] = SingleAppearance(
            appearance=EntityAppearance(traits=traits)
        )


def _attach_to_nested_appearance(
    invoker: EntityRecord, book_title: str, chapter: int, traits: EntityTraits
) -> None:
    """Merge traits into an invoker's nested-tracked appearance in-place."""
    existing = invoker.appearances.get(book_title)
    if not isinstance(existing, ChapterAppearances):
        existing = ChapterAppearances(chapters={})
        invoker.appearances[book_title] = existing

    if chapter in existing.chapters:
        existing.chapters[chapter].traits = merge_values(
            existing.chapters[chapter].traits, traits
        )
    else:
        existing.chapters[chapter] = EntityAppearance(traits=traits)


def _resolve_appearance(
    binder: Binder, key: str, value: AppearanceValue
) -> None:
    """Resolve a single allusion appearance (nested or flat) into invoker."""
    match value:
        case ChapterAppearances(chapters=chapters):
            for chapter, appearance in chapters.items():
                invoked_by = appearance.traits.get(_INVOKED_BY_TRAIT)
                rhetorical = _rhetorical_traits(appearance.traits)
                if not isinstance(invoked_by, str) or not rhetorical:
                    continue
                invoker = _find_invoker(binder, invoked_by)
                if invoker is not None:
                    _attach_to_nested_appearance(
                        invoker, key, chapter, rhetorical
                    )
        case SingleAppearance(appearance=appearance):
            invoked_by = appearance.traits.get(_INVOKED_BY_TRAIT)
            rhetorical = _rhetorical_traits(appearance.traits)
            if not isinstance(invoked_by, str) or not rhetorical:
                return
            invoker = _find_invoker(binder, invoked_by)
            if invoker is not None:
                _attach_to_flat_appearance(invoker, key, rhetorical)


def resolve_allusions(binder: Binder) -> Binder:
    """Merge allusive figures' rhetorical traits into their invoking entities.

    Args:
        binder: The Binder model to resolve.

    Returns:
        The Binder model with the "Allusions" category removed.
    """
    allusions = binder.categories.get(_ALLUSIONS_CATEGORY)
    if allusions is None:
        return binder

    for entity in allusions.entities.values():
        for key, value in entity.appearances.items():
            _resolve_appearance(binder, key, value)

    del binder.categories[_ALLUSIONS_CATEGORY]
    return binder


__all__ = ["resolve_allusions"]
