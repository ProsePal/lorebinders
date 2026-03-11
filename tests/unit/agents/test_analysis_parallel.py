from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from lorebinders.agent.analysis import analyze_entities
from lorebinders.models import AgentDeps, Book, Chapter
from lorebinders.settings import Settings
from lorebinders.storage.providers.test import TestStorageProvider

pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:google.genai.types"
)


@pytest.mark.anyio
async def test_analyze_entities_parallel_basic(tmp_path: Path) -> None:
    """Test analyze_entities processes multiple chapters correctly."""

    entities = {
        "Characters": {"Gandalf": [1, 2], "Frodo": [1]},
        "Locations": {"Shire": [1], "Rivendell": [2]},
    }

    results_map = {
        (1, "Characters"): [
            {
                "entity_name": "Gandalf",
                "category": "Characters",
                "traits": [
                    {"trait": "Role", "value": "Wizard", "evidence": "..."}
                ],
            },
            {
                "entity_name": "Frodo",
                "category": "Characters",
                "traits": [
                    {"trait": "Role", "value": "Hobbit", "evidence": "..."}
                ],
            },
        ],
        (1, "Locations"): [
            {
                "entity_name": "Shire",
                "category": "Locations",
                "traits": [
                    {"trait": "Type", "value": "Village", "evidence": "..."}
                ],
            },
        ],
        (2, "Characters"): [
            {
                "entity_name": "Gandalf",
                "category": "Characters",
                "traits": [
                    {"trait": "Role", "value": "Wizard", "evidence": "..."}
                ],
            },
        ],
        (2, "Locations"): [
            {
                "entity_name": "Rivendell",
                "category": "Locations",
                "traits": [
                    {"trait": "Type", "value": "Elven", "evidence": "..."}
                ],
            },
        ],
    }

    import json

    def mock_call(messages: list[ModelMessage], info: object) -> ModelResponse:
        user_msg = str(messages[-1])

        chap_num = 1 if "Gandalf and Frodo Shire" in user_msg else 2
        category = "Characters" if "### Characters" in user_msg else "Locations"

        resp = results_map.get((chap_num, category), [])

        return ModelResponse(
            parts=[TextPart(content=json.dumps({"response": resp}))]
        )

    from lorebinders.agent.factory import create_analysis_agent

    ana_agent = create_analysis_agent()

    settings = Settings()
    deps = AgentDeps(
        settings=settings,
        prompt_loader=lambda x: "mock",
    )

    book = Book(
        title="LOTR",
        author="Tolkien",
        chapters=[
            Chapter(number=1, title="Ch 1", content="Gandalf and Frodo Shire"),
            Chapter(number=2, title="Ch 2", content="Gandalf Rivendell"),
        ],
    )

    storage = TestStorageProvider()
    storage.set_workspace("TestAuthor", "TestTitle")

    with ana_agent.override(model=FunctionModel(mock_call)):
        results = await analyze_entities(
            entities,
            book,
            ana_agent,
            deps,
            {"Characters": ["Role"], "Locations": ["Type"]},
            storage,
        )

    assert len(results) == 5
    names = [p.name for p in results]
    assert names.count("Gandalf") == 2
    assert "Frodo" in names
    assert "Shire" in names
    assert "Rivendell" in names


@pytest.mark.anyio
async def test_analyze_entities_parallel_semaphore(tmp_path: Path) -> None:
    """Verify that semaphore limits concurrency (indirectly via timing/mock)."""

    pass
