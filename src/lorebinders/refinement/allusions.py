"""Resolution of allusive figures into the entities that invoke them."""

import logging
from dataclasses import dataclass, field

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

logger = logging.getLogger(__name__)

_ALLUSIONS_CATEGORY = "Allusions"
_INVOKED_BY_TRAIT = "Invoked by"
_RHETORICAL_TRAITS = (
    "Rhetorical significance",
    "What it reveals about the invoker",
)


@dataclass
class _InvokerIndex:
    """Cached, deterministic lookup of invokers by "Invoked by" attribution.

    Candidate entities are flattened once per resolution run and every
    resolved attribution is memoized, so repeated attributions cost one
    scan rather than one scan per allusion appearance.
    """

    candidates: list[tuple[str, EntityRecord]]
    resolved: dict[str, EntityRecord | None] = field(default_factory=dict)

    @classmethod
    def from_binder(cls, binder: Binder) -> "_InvokerIndex":
        """Flatten every non-Allusions entity into a searchable candidate list.

        Args:
            binder: The Binder whose entities may be invokers.

        Returns:
            An index over the binder's candidate invokers.
        """
        return cls(
            candidates=[
                (name, entity)
                for category_name, category in binder.categories.items()
                if category_name != _ALLUSIONS_CATEGORY
                for name, entity in category.entities.items()
            ]
        )

    def find(self, invoked_by: str) -> EntityRecord | None:
        """Find the entity record matching an "Invoked by" attribution.

        Args:
            invoked_by: The name the allusion was attributed to.

        Returns:
            The unambiguously matching EntityRecord, or None when no
            entity matches or the attribution is ambiguous.
        """
        key = invoked_by.strip().lower()
        if key not in self.resolved:
            self.resolved[key] = self._match(key)
        return self.resolved[key]

    def _match(self, key: str) -> EntityRecord | None:
        """Select the single entity an attribution refers to, if unambiguous.

        Exact case-insensitive matches take precedence over the fuzzy
        matching used elsewhere in refinement; anything matching more than
        one entity is skipped rather than merged into an arbitrary record.
        """
        exact = [
            entity
            for name, entity in self.candidates
            if name.strip().lower() == key
        ]
        matches = exact or [
            entity
            for name, entity in self.candidates
            if is_similar_key(name, key)
        ]

        match matches:
            case [only]:
                return only
            case []:
                return None
            case _:
                logger.debug(
                    "Ambiguous invoker attribution %r matched %d entities; "
                    "skipping",
                    key,
                    len(matches),
                )
                return None


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
    index: _InvokerIndex, key: str, value: AppearanceValue
) -> None:
    """Resolve a single allusion appearance (nested or flat) into invoker."""
    match value:
        case ChapterAppearances(chapters=chapters):
            for chapter, appearance in chapters.items():
                invoked_by = appearance.traits.get(_INVOKED_BY_TRAIT)
                rhetorical = _rhetorical_traits(appearance.traits)
                if not isinstance(invoked_by, str) or not rhetorical:
                    continue
                invoker = index.find(invoked_by)
                if invoker is not None:
                    _attach_to_nested_appearance(
                        invoker, key, chapter, rhetorical
                    )
        case SingleAppearance(appearance=appearance):
            invoked_by = appearance.traits.get(_INVOKED_BY_TRAIT)
            rhetorical = _rhetorical_traits(appearance.traits)
            if not isinstance(invoked_by, str) or not rhetorical:
                return
            invoker = index.find(invoked_by)
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

    index = _InvokerIndex.from_binder(binder)
    for entity in allusions.entities.values():
        for key, value in entity.appearances.items():
            _resolve_appearance(index, key, value)

    del binder.categories[_ALLUSIONS_CATEGORY]
    return binder


__all__ = ["resolve_allusions"]
