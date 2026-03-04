import asyncio
import json

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from lorebinders import models
from lorebinders.agent.analysis import analyze_entities
from lorebinders.agent.factory import create_analysis_agent
from lorebinders.settings import Settings
from lorebinders.storage.providers.test import TestStorageProvider


@pytest.mark.anyio
async def test_analyze_entities_parallel_chapters() -> None:
    """Test that analyze_entities handles multiple chapters correctly."""

    call_count = 0

    def mock_model_call(
        messages: list[ModelMessage], info: object
    ) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        response = []
        msg_str = str(messages)
        if "Gandalf" in msg_str:
            response.append(
                {
                    "entity_name": "Gandalf",
                    "category": "Characters",
                    "traits": [
                        {
                            "trait": "Role",
                            "value": "Wizard",
                            "evidence": "Magic",
                        }
                    ],
                }
            )
        if "Frodo" in msg_str:
            response.append(
                {
                    "entity_name": "Frodo",
                    "category": "Characters",
                    "traits": [
                        {
                            "trait": "Role",
                            "value": "Hobbit",
                            "evidence": "Small",
                        }
                    ],
                }
            )

        data = json.dumps({"response": response})
        return ModelResponse(parts=[TextPart(content=data)])

    model = FunctionModel(mock_model_call)

    settings = Settings()
    agent = create_analysis_agent(settings)
    deps = models.AgentDeps(
        settings=settings, prompt_loader=lambda x: "Mock prompt"
    )

    book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=1, title="Ch 1", content="Gandalf appeared."),
            models.Chapter(number=2, title="Ch 2", content="Frodo appeared."),
        ],
    )

    entities = {"Characters": {"Gandalf": [1], "Frodo": [2]}}

    effective_traits = {"Characters": ["Role"]}
    storage = TestStorageProvider()
    storage.set_workspace("TestAuthor", "TestTitle")

    with agent.override(model=model):
        results = await analyze_entities(
            entities=entities,
            book=book,
            agent=agent,
            deps=deps,
            effective_traits=effective_traits,
            storage=storage,
        )

    assert len(results) == 2
    assert any(r.name == "Gandalf" and r.chapter_number == 1 for r in results)
    assert any(r.name == "Frodo" and r.chapter_number == 2 for r in results)
    assert call_count == 2

    assert storage.profile_exists(1, "Characters", "Gandalf")
    assert storage.profile_exists(2, "Characters", "Frodo")


@pytest.mark.anyio
async def test_analyze_entities_sequential_categories_in_chapter() -> None:
    """Test categories in a chapter are processed sequentially."""

    call_order = []

    async def mock_model_call(
        messages: list[ModelMessage], info: object
    ) -> ModelResponse:
        msg_str = str(messages)
        if "Characters" in msg_str:
            call_order.append("Characters")
        elif "Locations" in msg_str:
            call_order.append("Locations")

        await asyncio.sleep(0.01)

        return ModelResponse(
            parts=[TextPart(content=json.dumps({"response": []}))]
        )

    model = FunctionModel(mock_model_call)

    settings = Settings()
    agent = create_analysis_agent(settings)
    deps = models.AgentDeps(
        settings=settings, prompt_loader=lambda x: "Mock prompt"
    )

    book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(
                number=1, title="Ch 1", content="Gandalf in the Shire."
            ),
        ],
    )

    entities = {"Characters": {"Gandalf": [1]}, "Locations": {"Shire": [1]}}

    effective_traits = {"Characters": ["Role"], "Locations": ["Type"]}
    storage = TestStorageProvider()
    storage.set_workspace("TestAuthor", "TestTitle")

    with agent.override(model=model):
        await analyze_entities(
            entities=entities,
            book=book,
            agent=agent,
            deps=deps,
            effective_traits=effective_traits,
            storage=storage,
        )

    assert len(call_order) == 2


@pytest.mark.anyio
async def test_analyze_entities_caching() -> None:
    """Test that existing profiles are skipped."""

    call_count = 0

    def mock_model_call(
        messages: list[ModelMessage], info: object
    ) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(
            parts=[TextPart(content=json.dumps({"response": []}))]
        )

    model = FunctionModel(mock_model_call)
    settings = Settings()
    agent = create_analysis_agent(settings)
    deps = models.AgentDeps(
        settings=settings, prompt_loader=lambda x: "Mock prompt"
    )

    book = models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[models.Chapter(number=1, title="Ch 1", content="Context")],
    )

    entities = {"Characters": {"Gandalf": [1]}}
    storage = TestStorageProvider()
    storage.set_workspace("TestAuthor", "TestTitle")

    storage.save_profile(
        1,
        models.EntityProfile(
            name="Gandalf", category="Characters", chapter_number=1, traits={}
        ),
    )

    with agent.override(model=model):
        results = await analyze_entities(
            entities=entities,
            book=book,
            agent=agent,
            deps=deps,
            effective_traits={"Characters": []},
            storage=storage,
        )

    assert len(results) == 1
    assert call_count == 0
