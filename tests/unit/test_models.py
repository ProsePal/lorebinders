from pathlib import Path

import pytest
from pydantic import ValidationError

from lorebinders.models import (
    AnalyzedTrait,
    Book,
    Chapter,
    EntityProfile,
    NarratorConfig,
    RunConfiguration,
)


def test_analyzed_trait_valid() -> None:
    """Test valid AnalyzedTrait creation."""
    trait = AnalyzedTrait(
        trait="Role",
        value="Hero",
        evidence="He saved the day.",
    )
    assert trait.trait == "Role"
    assert trait.value == "Hero"
    assert trait.evidence == "He saved the day."


def test_narrator_config_validation() -> None:
    """Verify NarratorConfig defaults and optional fields."""
    config = NarratorConfig(is_1st_person=False)
    assert config.is_1st_person is False
    assert config.name is None

    config = NarratorConfig(is_1st_person=True, name="Watson")
    assert config.is_1st_person is True
    assert config.name == "Watson"


def test_run_configuration_validation() -> None:
    """Verify RunConfiguration requires specific fields and handles defaults."""
    narrator = NarratorConfig(is_1st_person=False)

    config = RunConfiguration(
        book_path=Path("./book.epub"),
        author_name="Test Author",
        book_title="Test Book",
        narrator_config=narrator,
        custom_traits={"Characters": ["brave"]},
        custom_categories=["personality"],
    )
    assert config.author_name == "Test Author"
    assert config.book_path == Path("./book.epub")

    with pytest.raises(ValidationError):
        RunConfiguration(
            book_path=Path("./book.epub"),
            book_title="Test Book",
            narrator_config=narrator,
            custom_traits={},
            custom_categories=[],
        )


def test_chapter_model() -> None:
    """Verify Chapter model data integrity."""
    chapter = Chapter(
        number=1, title="The Beginning", content="Once upon a time..."
    )
    assert chapter.number == 1
    assert chapter.title == "The Beginning"
    assert chapter.content == "Once upon a time..."

    with pytest.raises(ValidationError):
        Chapter(number="one", title="Title", content="Content")


def test_book_model() -> None:
    """Verify Book model acts as a container for Chapters."""
    ch1 = Chapter(number=1, title="One", content="Content 1")
    ch2 = Chapter(number=2, title="Two", content="Content 2")

    book = Book(title="My Book", author="Me", chapters=[ch1, ch2])

    assert len(book.chapters) == 2
    assert book.chapters[0].title == "One"
    assert book.chapters[1].number == 2


def test_entity_profile_model() -> None:
    """Verify EntityProfile structure."""
    profile = EntityProfile(
        name="Sherlock",
        category="Characters",
        chapter_number=1,
        traits={"intelligence": "High", "brave": "Yes"},
        confidence_score=0.95,
    )

    assert profile.name == "Sherlock"
    assert profile.traits["intelligence"] == "High"
    assert profile.confidence_score == 0.95
