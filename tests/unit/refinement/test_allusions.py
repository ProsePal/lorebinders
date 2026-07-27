"""Tests for allusion resolution."""

from unittest import mock

from lorebinders.models import Binder, ChapterAppearances
from lorebinders.refinement.allusions import resolve_allusions
from lorebinders.refinement.deduplication import is_similar_key


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


def test_resolve_allusions_prefers_exact_match_over_fuzzy_match() -> None:
    binder = Binder()
    binder.add_appearance(
        category="Locations",
        name="Zeus Temple",
        chapter=1,
        book_title="Book 1",
        traits={"Role": "Shrine"},
    )
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
    zeus_appearance = zeus.appearances["Book 1"]
    assert isinstance(zeus_appearance, ChapterAppearances)
    assert (
        zeus_appearance.chapters[1].traits["Rhetorical significance"]
        == "Warns of punishment"
    )

    temple = result.categories["Locations"].entities["Zeus Temple"]
    temple_appearance = temple.appearances["Book 1"]
    assert isinstance(temple_appearance, ChapterAppearances)
    assert "Rhetorical significance" not in temple_appearance.chapters[1].traits


def test_resolve_allusions_skips_ambiguous_exact_matches() -> None:
    binder = Binder()
    for category in ("Characters", "Locations"):
        binder.add_appearance(
            category=category,
            name="Zeus",
            chapter=1,
            book_title="Book 1",
            traits={"Role": "Ambiguous"},
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

    for category in ("Characters", "Locations"):
        appearance = (
            result.categories[category].entities["Zeus"].appearances["Book 1"]
        )
        assert isinstance(appearance, ChapterAppearances)
        assert "Rhetorical significance" not in appearance.chapters[1].traits
    assert "Allusions" not in result.categories


def test_resolve_allusions_skips_ambiguous_fuzzy_matches() -> None:
    binder = Binder()
    for name in ("Zeus Temple", "Zeus Altar"):
        binder.add_appearance(
            category="Locations",
            name=name,
            chapter=1,
            book_title="Book 1",
            traits={"Role": "Shrine"},
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

    for name in ("Zeus Temple", "Zeus Altar"):
        appearance = (
            result.categories["Locations"].entities[name].appearances["Book 1"]
        )
        assert isinstance(appearance, ChapterAppearances)
        assert "Rhetorical significance" not in appearance.chapters[1].traits


def test_resolve_allusions_resolves_repeated_attribution_once() -> None:
    binder = Binder()
    for chapter in (1, 2, 3):
        binder.add_appearance(
            category="Characters",
            name="Lord Zeus",
            chapter=chapter,
            book_title="Book 1",
            traits={"Role": "God"},
        )
        binder.add_appearance(
            category="Allusions",
            name="Prometheus",
            chapter=chapter,
            book_title="Book 1",
            traits={
                "Invoked by": "Zeus",
                "Rhetorical significance": f"Warning {chapter}",
                "What it reveals about the invoker": "Authoritarian streak",
            },
        )

    with mock.patch(
        "lorebinders.refinement.allusions.is_similar_key",
        wraps=is_similar_key,
    ) as spy:
        result = resolve_allusions(binder)

    assert spy.call_count == 1

    zeus = result.categories["Characters"].entities["Lord Zeus"]
    appearance = zeus.appearances["Book 1"]
    assert isinstance(appearance, ChapterAppearances)
    for chapter in (1, 2, 3):
        assert (
            appearance.chapters[chapter].traits["Rhetorical significance"]
            == f"Warning {chapter}"
        )


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
