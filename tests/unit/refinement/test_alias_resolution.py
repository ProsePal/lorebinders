import json
from collections.abc import Callable

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from lorebinders.agent.factory import (
    build_alias_resolution_user_prompt,
    create_alias_resolution_agent,
)
from lorebinders.models import (
    AgentDeps,
    AliasGroup,
    Binder,
    CategoryRecord,
    EntityAppearance,
    EntityRecord,
    SingleAppearance,
)
from lorebinders.refinement.alias_resolution import (
    apply_alias_group,
    entity_context,
    resolve_aliases,
)
from lorebinders.settings import Settings

_Responder = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


def _deps() -> AgentDeps:
    return AgentDeps(
        settings=Settings(),
        prompt_loader=lambda name: f"mock prompt {name}",
    )


def _binder_with_aliases() -> Binder:
    binder = Binder()
    binder.add_appearance(
        "Characters", "Sauron", 1, "Book 1", {"Role": "Dark power"}
    )
    binder.add_appearance(
        "Characters", "The Dark Lord", 2, "Book 1", {"Mood": "Wrathful"}
    )
    binder.add_appearance(
        "Characters", "Frodo", 1, "Book 1", {"Role": "Ring-bearer"}
    )
    return binder


def _responder(payload: dict[str, object]) -> _Responder:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload))])

    return respond


def test_entity_context_prefers_summary() -> None:
    entity = EntityRecord(
        name="Sauron", category="Characters", summary="The Dark Lord of Mordor."
    )
    assert entity_context(entity) == "The Dark Lord of Mordor."


def test_entity_context_falls_back_to_traits() -> None:
    entity = EntityRecord(name="Sauron", category="Characters")
    entity.appearances["Book 1_ch1"] = SingleAppearance(
        appearance=EntityAppearance(
            traits={"Role": "Dark power", "Aliases": ["Annatar", "Gorthaur"]}
        )
    )

    context = entity_context(entity)

    assert "Role: Dark power" in context
    assert "Aliases: Annatar, Gorthaur" in context


def test_entity_context_truncates_long_notes() -> None:
    entity = EntityRecord(
        name="Sauron", category="Characters", summary="x" * 1000
    )
    assert len(entity_context(entity)) == 400


def test_entity_context_empty_for_bare_entity() -> None:
    entity = EntityRecord(name="Sauron", category="Characters")
    assert entity_context(entity) == ""


def test_apply_alias_group_merges_into_canonical() -> None:
    binder = _binder_with_aliases()
    category = binder.categories["Characters"]

    merged = apply_alias_group(
        category,
        AliasGroup(canonical_name="Sauron", aliases=["The Dark Lord"]),
    )

    assert merged == 1
    assert "The Dark Lord" not in category.entities
    assert set(category.entities) == {"Sauron", "Frodo"}
    assert set(category.entities["Sauron"].appearances) == {"Book 1"}


def test_apply_alias_group_merges_appearances() -> None:
    binder = _binder_with_aliases()
    category = binder.categories["Characters"]

    apply_alias_group(
        category,
        AliasGroup(canonical_name="Sauron", aliases=["The Dark Lord"]),
    )

    chapters = category.entities["Sauron"].appearances["Book 1"].chapters
    assert chapters[1].traits == {"Role": "Dark power"}
    assert chapters[2].traits == {"Mood": "Wrathful"}


def test_apply_alias_group_is_case_insensitive() -> None:
    binder = _binder_with_aliases()
    category = binder.categories["Characters"]

    merged = apply_alias_group(
        category,
        AliasGroup(canonical_name="sauron", aliases=["  the dark lord "]),
    )

    assert merged == 1
    assert "The Dark Lord" not in category.entities


def test_apply_alias_group_ignores_unknown_canonical() -> None:
    binder = _binder_with_aliases()
    category = binder.categories["Characters"]

    merged = apply_alias_group(
        category, AliasGroup(canonical_name="Morgoth", aliases=["Sauron"])
    )

    assert merged == 0
    assert set(category.entities) == {"Sauron", "The Dark Lord", "Frodo"}


def test_apply_alias_group_ignores_unknown_and_self_aliases() -> None:
    binder = _binder_with_aliases()
    category = binder.categories["Characters"]

    merged = apply_alias_group(
        category,
        AliasGroup(canonical_name="Sauron", aliases=["Sauron", "Annatar"]),
    )

    assert merged == 0
    assert set(category.entities) == {"Sauron", "The Dark Lord", "Frodo"}


def test_apply_alias_group_ignores_already_merged_alias() -> None:
    binder = _binder_with_aliases()
    category = binder.categories["Characters"]
    group = AliasGroup(canonical_name="Sauron", aliases=["The Dark Lord"])

    assert apply_alias_group(category, group) == 1
    assert apply_alias_group(category, group) == 0


@pytest.mark.anyio
async def test_resolve_aliases_merges_semantic_alias() -> None:
    agent = create_alias_resolution_agent()
    binder = _binder_with_aliases()
    payload: dict[str, object] = {
        "groups": [{"canonical_name": "Sauron", "aliases": ["The Dark Lord"]}]
    }

    with agent.override(model=FunctionModel(_responder(payload))):
        result = await resolve_aliases(binder, agent, _deps())

    assert set(result.categories["Characters"].entities) == {
        "Sauron",
        "Frodo",
    }


@pytest.mark.anyio
async def test_resolve_aliases_keeps_distinct_entities() -> None:
    agent = create_alias_resolution_agent()
    binder = _binder_with_aliases()

    with agent.override(model=FunctionModel(_responder({"groups": []}))):
        result = await resolve_aliases(binder, agent, _deps())

    assert set(result.categories["Characters"].entities) == {
        "Sauron",
        "The Dark Lord",
        "Frodo",
    }


@pytest.mark.anyio
async def test_resolve_aliases_skips_single_entity_category() -> None:
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart(content='{"groups": []}')])

    agent = create_alias_resolution_agent()
    binder = Binder()
    binder.add_appearance("Characters", "Frodo", 1, "Book 1", {"Role": "Hero"})

    with agent.override(model=FunctionModel(respond)):
        await resolve_aliases(binder, agent, _deps())

    assert calls == 0


@pytest.mark.anyio
async def test_resolve_aliases_survives_agent_failure() -> None:
    def explode(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model unavailable")

    agent = create_alias_resolution_agent()
    binder = _binder_with_aliases()

    with agent.override(model=FunctionModel(explode)):
        result = await resolve_aliases(binder, agent, _deps())

    assert set(result.categories["Characters"].entities) == {
        "Sauron",
        "The Dark Lord",
        "Frodo",
    }


@pytest.mark.anyio
async def test_resolve_aliases_sends_names_and_context() -> None:
    captured: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.extend(
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
            and isinstance(part.content, str)
        )
        return ModelResponse(parts=[TextPart(content='{"groups": []}')])

    agent = create_alias_resolution_agent()
    binder = _binder_with_aliases()

    with agent.override(model=FunctionModel(respond)):
        await resolve_aliases(binder, agent, _deps())

    prompt = "\n".join(captured)
    assert "Sauron" in prompt
    assert "The Dark Lord" in prompt
    assert "Dark power" in prompt


def test_build_alias_resolution_user_prompt_lists_entries() -> None:
    prompt = build_alias_resolution_user_prompt(
        "Characters", [("Sauron", "Dark power"), ("Frodo", "")]
    )

    assert "## CATEGORY: Characters" in prompt
    assert "- Sauron: Dark power" in prompt
    assert "- Frodo" in prompt


def test_apply_alias_group_on_empty_category() -> None:
    category = CategoryRecord(name="Characters")

    merged = apply_alias_group(
        category, AliasGroup(canonical_name="Sauron", aliases=["Annatar"])
    )

    assert merged == 0
