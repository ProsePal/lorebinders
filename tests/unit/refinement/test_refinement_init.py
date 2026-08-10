"""Tests for the refinement pipeline entry point."""

import json

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from lorebinders.agent.factory import create_alias_resolution_agent
from lorebinders.models import AgentDeps, Binder, ChapterAppearances
from lorebinders.refinement import refine_binder, refine_binder_async
from lorebinders.settings import Settings


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


def _binder_with_semantic_aliases() -> Binder:
    binder = Binder()
    binder.add_appearance(
        "Characters", "Sauron", 1, "Book 1", {"Role": "Dark power"}
    )
    binder.add_appearance(
        "Characters", "Dark Lord", 2, "Book 1", {"Mood": "Wrathful"}
    )
    return binder


def _alias_model(calls: list[int]) -> FunctionModel:
    payload = {
        "groups": [{"canonical_name": "Sauron", "aliases": ["Dark Lord"]}]
    }

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    return FunctionModel(respond)


@pytest.mark.anyio
async def test_refine_binder_async_matches_sync_without_agent() -> None:
    result = await refine_binder_async(_binder_with_semantic_aliases())

    assert set(result.categories["Characters"].entities) == {
        "Sauron",
        "Dark Lord",
    }


@pytest.mark.anyio
async def test_refine_binder_async_resolves_aliases() -> None:
    calls: list[int] = []
    agent = create_alias_resolution_agent()
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda name: f"mock {name}"
    )

    with agent.override(model=_alias_model(calls)):
        result = await refine_binder_async(
            _binder_with_semantic_aliases(), None, agent, deps
        )

    assert calls
    assert set(result.categories["Characters"].entities) == {"Sauron"}


@pytest.mark.anyio
async def test_refine_binder_async_resolves_allusions_before_aliases() -> None:
    binder = _binder_with_semantic_aliases()
    binder.add_appearance(
        "Allusions",
        "Morgoth",
        2,
        "Book 1",
        {
            "Invoked by": "Dark Lord",
            "Rhetorical significance": "Claims an older tyranny",
        },
    )
    agent = create_alias_resolution_agent()
    deps = AgentDeps(
        settings=Settings(), prompt_loader=lambda name: f"mock {name}"
    )

    with agent.override(model=_alias_model([])):
        result = await refine_binder_async(binder, None, agent, deps)

    sauron = result.categories["Characters"].entities["Sauron"]
    traits = [
        appearance.traits
        for value in sauron.appearances.values()
        if isinstance(value, ChapterAppearances)
        for appearance in value.chapters.values()
    ]
    assert any(
        trait.get("Rhetorical significance") == "Claims an older tyranny"
        for trait in traits
    )


@pytest.mark.anyio
async def test_refine_binder_async_skips_alias_pass_when_disabled() -> None:
    calls: list[int] = []
    agent = create_alias_resolution_agent()
    deps = AgentDeps(
        settings=Settings(alias_resolution_enabled=False),
        prompt_loader=lambda name: f"mock {name}",
    )

    with agent.override(model=_alias_model(calls)):
        result = await refine_binder_async(
            _binder_with_semantic_aliases(), None, agent, deps
        )

    assert not calls
    assert set(result.categories["Characters"].entities) == {
        "Sauron",
        "Dark Lord",
    }
