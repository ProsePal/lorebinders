import pytest

from lorebinders.models import (
    Binder,
    CategoryRecord,
    ChapterAppearances,
    EntityAppearance,
    EntityRecord,
)
from lorebinders.refinement.deduplication import (
    _resolve_category_entities,
    prioritize_keys,
    resolve_binder,
)


@pytest.mark.parametrize(
    "key1, key2, expected_keep, expected_merge",
    [
        ("John", "John Smith", "John", "John Smith"),
        ("John Smith", "John", "John", "John Smith"),
        ("Foo", "FooBar", "Foo", "FooBar"),
        ("Apple", "Berry", "Berry", "Apple"),
        ("Sam", "Samwise", "Sam", "Samwise"),
        ("The King", "King", "The King", "King"),
        ("gandalf", "Gandalf", "Gandalf", "gandalf"),
        ("Frodo", "FRODO", "FRODO", "Frodo"),
        ("Merry", "Meriadoc", "Merry", "Meriadoc"),
        ("Aragorn", "Strider", "Strider", "Aragorn"),
    ],
)
def test_prioritize_keys(
    key1: str, key2: str, expected_keep: str, expected_merge: str
) -> None:
    keep, merge = prioritize_keys(key1, key2)
    assert keep == expected_keep
    assert merge == expected_merge


def test_resolve_category_entities_merges_duplicates() -> None:
    category = CategoryRecord(name="Characters")
    category.entities["John"] = EntityRecord(name="John", category="Characters")
    category.entities["John"].appearances["Book 1"] = ChapterAppearances(
        chapters={1: EntityAppearance(traits={"trait": "A"})}
    )
    category.entities["John Smith"] = EntityRecord(
        name="John Smith", category="Characters"
    )
    category.entities["Jane"] = EntityRecord(name="Jane", category="Characters")

    _resolve_category_entities(category)

    assert "John" not in category.entities
    assert "John Smith" in category.entities
    assert "Jane" in category.entities


def test_resolve_category_entities_preserves_distinct() -> None:
    category = CategoryRecord(name="Characters")
    category.entities["John"] = EntityRecord(name="John", category="Characters")
    category.entities["Jane"] = EntityRecord(name="Jane", category="Characters")

    _resolve_category_entities(category)

    assert len(category.entities) == 2
    assert "John" in category.entities
    assert "Jane" in category.entities


def test_resolve_binder_resolves_categories() -> None:
    binder = Binder()
    binder.add_appearance("Characters", "John", 1, "Book 1", {"A": "B"})
    binder.add_appearance("Characters", "John Smith", 1, "Book 1", {"C": "D"})

    binder.categories["Locations"] = CategoryRecord(name="Locations")

    resolve_binder(binder)

    assert "John" not in binder.categories["Characters"].entities
    assert "John Smith" in binder.categories["Characters"].entities
    assert "Locations" in binder.categories
