"""Entity resolution logic for refinement using Binder models."""

from itertools import combinations

from lorebinders.models import (
    AppearanceValue,
    Binder,
    CategoryRecord,
    ChapterAppearances,
    EntityRecord,
    SingleAppearance,
)
from lorebinders.refinement.normalization import (
    TITLES,
    merge_values,
    remove_titles,
    to_singular,
)


def is_similar_key(key1: str, key2: str) -> bool:
    """Determine if two keys are similar.

    Args:
        key1: The first key to compare.
        key2: The second key to compare.

    Returns:
        True if the keys are similar, False otherwise.
    """
    k1 = key1.strip().lower()
    k2 = key2.strip().lower()

    if k1 == k2:
        return True

    detitled_k1 = remove_titles(k1)
    detitled_k2 = remove_titles(k2)
    singular_k1 = to_singular(k1)
    singular_k2 = to_singular(k2)

    if any(
        (
            k1 == singular_k2,
            singular_k1 == k2,
            singular_k1 == singular_k2,
        )
    ):
        return True

    k1_is_title = k1 in TITLES
    k2_is_title = k2 in TITLES
    if k1_is_title and f"{k1} " in k2 or k2_is_title and f"{k2} " in k1:
        return True

    if f"{detitled_k1} " in detitled_k2 or f"{detitled_k2} " in detitled_k1:
        return True

    destructured_match = any(
        (
            f"{detitled_k1} " in f"{k2} ",
            f"{detitled_k2} " in f"{k1} ",
            f"{k1} " in f"{detitled_k2} ",
            f"{k2} " in f"{detitled_k1} ",
        )
    )

    if detitled_k1 != k1 or detitled_k2 != k2:
        return any(
            (
                detitled_k1 == k2,
                k1 == detitled_k2,
                detitled_k1 == detitled_k2,
                detitled_k1 == singular_k2,
                singular_k1 == detitled_k2,
                destructured_match,
            )
        )

    return destructured_match


def prioritize_keys(key1: str, key2: str) -> tuple[str, str]:
    """Determine which key to keep and which to merge.

    Args:
        key1: The first key.
        key2: The second key.

    Returns:
        A tuple of (key_to_keep, key_to_merge).
    """
    l1, l2 = key1.lower(), key2.lower()
    if (l1 in l2 or l2 in l1) and l1 != l2:
        if l1 in TITLES:
            return key2, key1
        if l2 in TITLES:
            return key1, key2

    return (key2, key1) if len(key1) >= len(key2) else (key1, key2)


def _merge_appearances(
    target: dict[str, AppearanceValue],
    source: dict[str, AppearanceValue],
) -> None:
    """Merge source appearance dictionary into target in-place."""
    for key, value in source.items():
        if key in target:
            t = target[key]
            if isinstance(value, SingleAppearance) and isinstance(
                t, SingleAppearance
            ):
                t.appearance.traits = merge_values(
                    t.appearance.traits, value.appearance.traits
                )
            elif isinstance(value, ChapterAppearances) and isinstance(
                t, ChapterAppearances
            ):
                for ch_num, ch_app in value.chapters.items():
                    if ch_num in t.chapters:
                        t.chapters[ch_num].traits = merge_values(
                            t.chapters[ch_num].traits, ch_app.traits
                        )
                    else:
                        t.chapters[ch_num] = ch_app
            else:
                target[key] = value
        else:
            target[key] = value


def _merge_entities(target: EntityRecord, source: EntityRecord) -> None:
    """Merge traits and summaries from source entity into target entity."""
    _merge_appearances(target.appearances, source.appearances)

    if source.summary:
        if not target.summary:
            target.summary = source.summary
        else:
            target.summary = f"{target.summary}\n\n{source.summary}"


def _resolve_category_entities(category: CategoryRecord) -> None:
    """Resolve duplicates within a category's entities in-place."""
    names = list(category.entities.keys())
    duplicates_to_remove: set[str] = set()

    for n1, n2 in combinations(names, 2):
        if n1 in duplicates_to_remove or n2 in duplicates_to_remove:
            continue

        if is_similar_key(n1, n2):
            to_merge, to_keep = prioritize_keys(n1, n2)

            _merge_entities(
                category.entities[to_keep], category.entities[to_merge]
            )
            duplicates_to_remove.add(to_merge)

    for name in duplicates_to_remove:
        category.entities.pop(name)


def resolve_binder(binder: Binder) -> Binder:
    """Full resolution pipeline on a Binder model.

    Args:
        binder: The Binder model to resolve.

    Returns:
        The resolved Binder model.
    """
    for category in binder.categories.values():
        _resolve_category_entities(category)
    return binder
