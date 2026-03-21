"""Tests for the refinement pipeline entry point."""

from lorebinders.models import Binder, ChapterAppearances
from lorebinders.refinement import refine_binder


def _make_binder_with_data() -> Binder:
    binder = Binder()
    binder.add_appearance(
        category="Characters",
        name="Alice",
        chapter=1,
        book_title="Book 1",
        traits={"Role": "Hero", "Age": "None Found"},
    )
    binder.add_appearance(
        category="Characters",
        name="Alice",
        chapter=1,
        book_title="Book 1",
        traits={"Role": "Hero", "Trait": "Brave"},
    )
    return binder


def test_refine_binder_returns_binder() -> None:
    binder = _make_binder_with_data()
    result = refine_binder(binder)
    assert isinstance(result, Binder)


def test_refine_binder_cleans_none_found_traits() -> None:
    binder = _make_binder_with_data()
    result = refine_binder(binder)
    alice = result.categories["Characters"].entities.get("Alice")
    assert alice is not None
    for book_app in alice.appearances.values():
        if isinstance(book_app, ChapterAppearances):
            for appearance in book_app.chapters.values():
                assert "Age" not in appearance.traits


def test_refine_binder_with_narrator_name() -> None:
    binder = Binder()
    binder.add_appearance(
        category="Characters",
        name="I",
        chapter=1,
        book_title="Book 1",
        traits={"Role": "Narrator"},
    )
    result = refine_binder(binder, narrator_name="Jane")
    assert "Jane" in result.categories["Characters"].entities
