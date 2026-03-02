from pydantic_ai.models.fallback import FallbackModel

from lorebinders.agent import (
    build_extraction_user_prompt,
    create_extraction_agent,
    run_agent,
)
from lorebinders.models import AgentDeps, NarratorConfig
from lorebinders.settings import Settings
from tests.utils import create_mock_model, get_system_prompt


def test_extraction_agent_run_sync_and_prompt() -> None:
    mock_model, captured_messages = create_mock_model(
        {
            "results": [
                {"category": "Characters", "entities": ["Hero", "Villain"]}
            ]
        }
    )

    agent = create_extraction_agent()
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "Mock content for extraction.txt",
    )

    with agent.override(model=mock_model):
        prompt = build_extraction_user_prompt(
            text="The Hero fought the Villain.",
            categories=["Characters"],
            narrator=NarratorConfig(is_1st_person=False),
        )

        result = run_agent(agent, prompt, deps)

        assert result.to_dict() == {"Characters": ["Hero", "Villain"]}

    system_prompt_content = get_system_prompt(captured_messages)

    assert system_prompt_content != ""
    assert "Mock content" in system_prompt_content


def test_extraction_agent_no_fallback() -> None:
    agent = create_extraction_agent(Settings())
    assert not isinstance(agent.model, FallbackModel)


def test_extraction_agent_with_fallback() -> None:
    settings = Settings(extraction_fallback_model="openrouter:test/fallback")
    agent = create_extraction_agent(settings)
    assert isinstance(agent.model, FallbackModel)
