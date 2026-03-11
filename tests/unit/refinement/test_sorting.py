import pytest

from lorebinders.refinement.deduplication import is_similar_key
from lorebinders.refinement.sorting import (
    _deduplicate_entity_names,
    sort_extractions,
)


@pytest.mark.parametrize(
    "key1, key2, expected",
    [
        ("John Smith", "john smith", True),
        ("  John Smith  ", "John Smith", True),
        ("Captain John", "John", True),
        ("John", "Dr. John", True),
        ("Mr. Smith", "Smith", True),
        ("The King", "King", True),
        ("Queen Anne", "Anne", True),
        ("Soldier", "Soldiers", True),
        ("Elf", "Elves", True),
        ("City", "Cities", True),
        ("Kingdom of Rohan", "Rohan", True),
        ("The Dark Lord", "Dark Lord", True),
        ("Mr. Frodo", "Frodo", True),
        ("Master Samwise", "Samwise", True),
        ("John", "Jane", False),
        ("Rohan", "Gondor", False),
        ("Frodo", "Sam", False),
        ("The King", "The Queen", False),
    ],
)
def test_is_similar_key_basic(key1: str, key2: str, expected: bool) -> None:
    assert is_similar_key(key1, key2) == expected


@pytest.mark.parametrize(
    "key1, key2",
    [
        ("The Great Gobbo", "Gobbo"),
        ("Gobbo", "The Great Gobbo"),
        ("Mr. Bilbo", "Bilbo Baggins"),
        ("Frodo", "Frodo Baggins"),
        ("Gandalf the Grey", "Gandalf"),
        ("Lady Galadriel", "Galadriel"),
        ("The Witch King of Angmar", "Witch King"),
        ("Saruman the White", "Saruman"),
        ("King Theoden", "Theoden"),
        ("Master Elrond", "Elrond"),
    ],
)
def test_is_similar_key_complex_matches(key1: str, key2: str) -> None:
    assert is_similar_key(key1, key2) is True


@pytest.mark.parametrize(
    "names, expected_len",
    [
        (["A", "A", "A"], 1),
        (["John", "John"], 1),
        (["John Smith", "John"], 1),
        (["Captain Jack", "Jack", "Sparrow"], 2),
        (
            ["Gandalf", "Gandalf the Grey", "Mithrandir"],
            2,
        ),
        (["Bilbo", "Bilbo Baggins", "Mr. Bilbo"], 1),
        (["Thorin", "Thorin Oakenshield"], 1),
        (["Legolas", "Legolas Greenleaf"], 1),
        (["Gimli", "Gimli son of Gloin"], 1),
        (
            ["Sauron", "The Dark Lord Sauron", "Necromancer"],
            2,
        ),
    ],
)
def test_deduplicate_entity_names_reduces_list(
    names: list[str], expected_len: int
) -> None:
    assert len(_deduplicate_entity_names(names, "Characters")) == expected_len


def test_deduplicate_entity_names_distinct() -> None:
    result = set(_deduplicate_entity_names(["A", "B"], "Characters"))
    assert result == {"A", "B"}


def test_deduplicate_entity_names_merges_titles() -> None:
    result = _deduplicate_entity_names(["Dr. Dre", "Dre"], "Characters")
    assert "Dre" in result


def test_sort_extractions_merges_characters() -> None:
    raw_data = {
        1: {
            "Characters": ["John", "John Smith"],
            "Locations": ["Shire", "The Shire"],
        },
        2: {
            "Characters": ["John Smith", "Jane"],
            "Locations": ["Shire"],
        },
    }
    sorted_data = sort_extractions(raw_data)
    assert "John Smith" in sorted_data["Characters"]
    assert "John" not in sorted_data["Characters"]
    assert "Jane" in sorted_data["Characters"]
    assert len(sorted_data["Locations"]) == 1


def test_sort_extractions_handles_narrator() -> None:
    raw_data = {
        1: {
            "Characters": ["I", "Me", "John"],
        }
    }
    sorted_data = sort_extractions(raw_data, narrator_name="NarratorGuy")
    assert "NarratorGuy" in sorted_data["Characters"]
    assert "I" not in sorted_data["Characters"]
    assert "John" in sorted_data["Characters"]
