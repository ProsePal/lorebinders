import pytest

from lorebinders.refinement.normalization import (
    merge_values,
    remove_titles,
    to_singular,
)
from lorebinders.types import EntityTraits


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Captain John Smith", "John Smith"),
        ("Dr. Watson", "Watson"),
        ("The Kitchen", "Kitchen"),
        ("John Smith", "John Smith"),
        ("Captain", "Captain"),
        ("General Kenobi", "Kenobi"),
        ("Mrs. Robinson", "Robinson"),
        ("Saint Nick", "Saint Nick"),
        ("Sir Arthur", "Arthur"),
        ("King George", "George"),
        ("Queen Victoria", "Victoria"),
        ("", ""),
    ],
)
def test_remove_titles(name: str, expected: str) -> None:
    assert remove_titles(name) == expected


@pytest.mark.parametrize(
    "plural, expected",
    [
        ("Cats", "Cat"),
        ("Dogs", "Dog"),
        ("Trees", "Tree"),
        ("Boys", "Boy"),
        ("Cities", "City"),
        ("Parties", "Party"),
        ("Families", "Family"),
        ("Wolves", "Wolf"),
        ("Shelves", "Shelf"),
        ("Knives", "Knife"),
        ("Wives", "Wife"),
        ("Lives", "Life"),
        ("Cacti", "Cactus"),
        ("Fungi", "Fungus"),
        ("Data", "Datum"),
        ("Men", "Men"),
        ("Women", "Women"),
        ("Potatoes", "Potato"),
        ("Heroes", "Hero"),
        ("Tomatoes", "Tomato"),
        ("Glasses", "Glass"),
        ("Boxes", "Box"),
        ("Foxes", "Fox"),
        ("Watches", "Watch"),
        ("Dishes", "Dish"),
        ("John", "John"),
        ("James", "Jame"),
        ("", ""),
    ],
)
def test_to_singular(plural: str, expected: str) -> None:
    assert to_singular(plural) == expected


@pytest.mark.parametrize(
    "value1, value2, expected",
    [
        (
            {"a": "1", "b": ["2"]},
            {"b": ["3"], "c": "4"},
            {"a": "1", "b": ["2", "3"], "c": "4"},
        ),
        ({"a": ["1", "2"]}, {"a": ["2", "3"]}, {"a": ["1", "2", "3"]}),
        ({"a": "1"}, {"a": ["2"]}, {"a": ["1", "2"]}),
        ({"a": ["1"]}, {"a": "2"}, {"a": ["1", "2"]}),
        ({"a": "1"}, {"a": "1"}, {"a": "1"}),
        ({"a": "1"}, {"a": "2"}, {"a": ["1", "2"]}),
    ],
)
def test_merge_values(
    value1: EntityTraits, value2: EntityTraits, expected: EntityTraits
) -> None:
    merged_dict = merge_values(value1, value2)
    assert merged_dict == expected
