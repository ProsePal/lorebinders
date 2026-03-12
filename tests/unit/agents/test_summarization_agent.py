import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from lorebinders.agent.factory import (
    build_summarization_user_prompt,
    create_summarization_agent,
    run_agent_async,
)
from lorebinders.agent.summarization import summarize_binder
from lorebinders.models import AgentDeps, Binder, SummarizerResult
from lorebinders.settings import Settings
from tests.utils import TestStorageProvider


@pytest.mark.anyio
async def test_summarization_agent_run_async_and_prompt() -> None:
    captured_messages: list[ModelMessage] = []

    expected_result_dict = {
        "entity_name": "Gandalf",
        "summary": "A powerful wizard.",
    }

    expected_result_obj = SummarizerResult(
        entity_name="Gandalf", summary="A powerful wizard."
    )

    def mock_model_call(
        messages: list[ModelMessage], info: object
    ) -> ModelResponse:
        nonlocal captured_messages
        captured_messages = list(messages)
        return ModelResponse(
            parts=[TextPart(content=json.dumps(expected_result_dict))]
        )

    agent = create_summarization_agent()
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: f"Mock content for {x}",
    )

    with agent.override(model=FunctionModel(mock_model_call)):
        prompt = build_summarization_user_prompt(
            entity_name="Gandalf",
            category="Character",
            context_data="He is a wizard. He wears grey.",
        )

        result = await run_agent_async(agent, prompt, deps)

        assert result == expected_result_obj

    settings = Settings()
    deps = AgentDeps(settings=settings, prompt_loader=lambda x: "mock prompt")

    model = TestModel()
    agent = create_summarization_agent()

    prompt = build_summarization_user_prompt(
        entity_name="Frodo",
        category="Characters",
        context_data="Chapter 1: Trait=Brave\nChapter 2: Trait=Short",
    )

    with agent.override(model=model):
        run_result = await agent.run(prompt, deps=deps)

    assert isinstance(run_result.output, SummarizerResult)
    assert isinstance(run_result.output.summary, str)

    assert "Frodo" in prompt
    assert "Characters" in prompt
    assert "Brave" in prompt
    assert "Short" in prompt


@pytest.mark.anyio
async def test_summarize_binder(tmp_path: Path) -> None:
    """Test summarize_binder processes a realistic binder sequentially."""
    settings = Settings()
    deps = AgentDeps(settings=settings, prompt_loader=lambda x: "mock prompt")

    model = TestModel()
    agent = create_summarization_agent()

    binder = Binder()
    binder.add_appearance("Characters", "Frodo", 1, {"Traits": ["Brave"]})
    binder.add_appearance(
        "Characters", "Frodo", 2, {"Traits": ["Brave", "Short"]}
    )
    binder.add_appearance(
        "Locations", "Shire", 1, {"Type": "Village", "Vibe": "Peaceful"}
    )
    binder.add_appearance(
        "Locations", "Shire", 3, {"Type": "Target", "Vibe": "Threatened"}
    )

    storage = TestStorageProvider()
    storage.set_workspace("TestAuthor", "TestTitle")

    with agent.override(model=model, deps=deps):
        await summarize_binder(binder, storage=storage, agent=agent, deps=deps)

        assert "Characters" in binder.categories
        frodo = binder.categories["Characters"].entities["Frodo"]
        assert frodo.summary is not None
        assert isinstance(frodo.summary, str)

        assert "Locations" in binder.categories
        shire = binder.categories["Locations"].entities["Shire"]
        assert shire.summary is not None
        assert isinstance(shire.summary, str)


def test_summarization_agent_no_fallback() -> None:
    agent = create_summarization_agent(Settings())
    assert not isinstance(agent.model, FallbackModel)


def test_summarization_agent_with_fallback() -> None:
    settings = Settings(summarization_fallback_model="openrouter:test/fallback")
    agent = create_summarization_agent(settings)
    assert isinstance(agent.model, FallbackModel)
