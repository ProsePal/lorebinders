"""Tests for allusion resolution."""

from lorebinders.models import Binder, ChapterAppearances
from lorebinders.refinement.allusions import resolve_allusions


def test_resolve_allusions_merges_traits_into_invoker() -> None:
    binder = Binder()
    binder.add_appearance(
        category="Characters",
        name="Zeus",
        chapter=1,
        book_title="Book 1",
        traits={"Role": "God"},
    )
    binder.add_appearance(
        category="Allusions",
        name="Prometheus",
        chapter=1,
        book_title="Book 1",
        traits={
            "Invoked by": "Zeus",
            "Rhetorical significance": "Warns of punishment",
            "What it reveals about the invoker": "Authoritarian streak",
        },
    )

    result = resolve_allusions(binder)

    zeus = result.categories["Characters"].entities["Zeus"]
    appearance = zeus.appearances["Book 1"]
    assert isinstance(appearance, ChapterAppearances)
    traits = appearance.chapters[1].traits
    assert traits["Rhetorical significance"] == "Warns of punishment"
    assert traits["What it reveals about the invoker"] == "Authoritarian streak"
    assert "Allusions" not in result.categories


def test_resolve_allusions_skips_silently_when_invoker_not_found() -> None:
    binder = Binder()
    binder.add_appearance(
        category="Characters",
        name="Zeus",
        chapter=1,
        book_title="Book 1",
        traits={"Role": "God"},
    )
    binder.add_appearance(
        category="Allusions",
        name="Prometheus",
        chapter=1,
        book_title="Book 1",
        traits={
            "Invoked by": "Nobody",
            "Rhetorical significance": "Warns of punishment",
            "What it reveals about the invoker": "An unknown streak",
        },
    )

    result = resolve_allusions(binder)

    zeus = result.categories["Characters"].entities["Zeus"]
    appearance = zeus.appearances["Book 1"]
    assert isinstance(appearance, ChapterAppearances)
    assert "Rhetorical significance" not in appearance.chapters[1].traits
    assert "Allusions" not in result.categories


def test_resolve_allusions_merges_multiple_same_invoker_same_chapter() -> None:
    binder = Binder()
    binder.add_appearance(
        category="Characters",
        name="Zeus",
        chapter=1,
        book_title="Book 1",
        traits={"Role": "God"},
    )
    binder.add_appearance(
        category="Allusions",
        name="Prometheus",
        chapter=1,
        book_title="Book 1",
        traits={
            "Invoked by": "Zeus",
            "Rhetorical significance": "Warns of punishment for defiance",
            "What it reveals about the invoker": "Authoritarian streak",
        },
    )
    binder.add_appearance(
        category="Allusions",
        name="Icarus",
        chapter=1,
        book_title="Book 1",
        traits={
            "Invoked by": "Zeus",
            "Rhetorical significance": "Warns against hubris",
            "What it reveals about the invoker": "Fear of rivals",
        },
    )

    result = resolve_allusions(binder)

    zeus = result.categories["Characters"].entities["Zeus"]
    appearance = zeus.appearances["Book 1"]
    assert isinstance(appearance, ChapterAppearances)
    traits = appearance.chapters[1].traits
    assert traits["Rhetorical significance"] == [
        "Warns against hubris",
        "Warns of punishment for defiance",
    ]
    assert traits["What it reveals about the invoker"] == [
        "Authoritarian streak",
        "Fear of rivals",
    ]


def test_resolve_allusions_returns_binder_unchanged_when_no_allusions() -> None:
    binder = Binder()
    binder.add_appearance(
        category="Characters",
        name="Zeus",
        chapter=1,
        book_title="Book 1",
        traits={"Role": "God"},
    )

    result = resolve_allusions(binder)

    assert "Allusions" not in result.categories
    assert "Zeus" in result.categories["Characters"].entities
