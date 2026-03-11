import pytest

from lorebinders.models import Binder, EntityTraits
from lorebinders.refinement.cleaning import (
    clean_binder,
    clean_str,
    clean_traits,
)
from lorebinders.refinement.normalization import clean_entity_name


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("Valid", "Valid"),
        ("  None Found  ", ""),
        ("none found", ""),
        ("  ", "  "),
        ("", ""),
        ("n/a", "n/a"),
        ("NONE FOUND", ""),
        ("No info", "No info"),
        (
            "  Tabs\tAnd\nNewlines  ",
            "  Tabs\tAnd\nNewlines  ",
        ),
        ("UNKNOWN", "UNKNOWN"),
    ],
)
def test_clean_str(input_str: str, expected: str) -> None:
    """Verify string cleaning logic."""
    assert clean_str(input_str) == expected


def test_clean_traits() -> None:
    """Verify trait cleaning."""
    traits: EntityTraits = {
        "Simple": "Value",
        "Empty": "  None Found  ",
        "List": ["A", "none found", "B"],
    }
    cleaned = clean_traits(traits)
    assert cleaned["Simple"] == "Value"
    assert "Empty" not in cleaned
    assert cleaned["List"] == ["A", "B"]


@pytest.mark.parametrize(
    "name, category, expected",
    [
        ("Mr. John", "Characters", "John"),
        ("Forest (Dark)", "Locations", "Forest"),
        ("Mr. John", "Other", "John"),
        ("Dr. Strange", "Characters", "Strange"),
        ("Captain Jack", "Characters", "Jack"),
        ("Lady Jane", "Characters", "Jane"),
        ("Mount Doom (Volcano)", "Locations", "Mount Doom"),
        ("Sir Lancelot", "Characters", "Lancelot"),
    ],
)
def test_clean_entity_name(name: str, category: str, expected: str) -> None:
    """Verify entity name cleaning across categories."""
    assert clean_entity_name(name, category) == expected


def test_clean_binder_replaces_narrator() -> None:
    """Verify narrator placeholders are replaced in names and traits."""
    binder = Binder()
    binder.add_appearance(
        "Characters", "I", 1, {"Description": "The narrator is tall."}
    )

    cleaned = clean_binder(binder, "Jane Doe")

    assert "Jane Doe" in cleaned.categories["Characters"].entities
    ent = cleaned.categories["Characters"].entities["Jane Doe"]
    assert ent.appearances[1].traits["Description"] == "Jane Doe is tall."


def test_clean_binder_integrates_steps() -> None:
    """Verify the full cleaning pipeline logic."""
    binder = Binder()
    binder.add_appearance("Characters", "Mr. Smith", 1, {"Trait": "A"})
    binder.add_appearance("Characters", "Smith", 1, {"Trait": "B"})
    binder.add_appearance("Characters", "I", 1, {"Trait": "C"})
    binder.add_appearance("Locations", "Cave (Deep)", 1, {"Trait": "Dark"})

    cleaned = clean_binder(binder, "Jane")

    chars = cleaned.categories["Characters"].entities
    assert "Smith" in chars
    assert "Mr. Smith" not in chars
    assert "Jane" in chars

    locs = cleaned.categories["Locations"].entities
    assert "Cave" in locs
    assert "Cave (Deep)" not in locs
