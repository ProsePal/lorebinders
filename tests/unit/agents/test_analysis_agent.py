import pytest
from pydantic_ai.models.fallback import FallbackModel

from lorebinders.agent.factory import (
    build_analysis_user_prompt,
    create_analysis_agent,
    run_agent_async,
)
from lorebinders.models import (
    AgentDeps,
    AnalysisResult,
    AnalyzedTrait,
    CategoryTarget,
)
from lorebinders.settings import Settings
from tests.utils import create_mock_model, get_system_prompt


@pytest.mark.anyio
async def test_analysis_agent_run_async_and_prompt() -> None:
    """Test run execution and system prompt generation using PydanticAI."""
    expected_result_dict = [
        {
            "entity_name": "Gandalf",
            "category": "Character",
            "traits": [
                {"trait": "Role", "value": "Wizard", "evidence": "Uses magic"},
                {
                    "trait": "Origin",
                    "value": "Maiar",
                    "evidence": "From Valinor",
                },
            ],
        }
    ]

    expected_result_obj = AnalysisResult(
        entity_name="Gandalf",
        category="Character",
        traits=[
            AnalyzedTrait(trait="Role", value="Wizard", evidence="Uses magic"),
            AnalyzedTrait(
                trait="Origin", value="Maiar", evidence="From Valinor"
            ),
        ],
    )

    mock_model, captured_messages = create_mock_model(
        {"response": expected_result_dict}
    )

    agent = create_analysis_agent()
    deps = AgentDeps(
        settings=Settings(),
        prompt_loader=lambda x: "Mock content for analysis.txt",
    )

    with agent.override(model=mock_model):
        categories: list[CategoryTarget] = [
            CategoryTarget(
                name="Character",
                entities=["Gandalf"],
                traits=["Role", "Origin"],
            )
        ]

        prompt = build_analysis_user_prompt(
            context_text="Gandalf the Wizard came from Valinor.",
            categories=categories,
        )

        result = await run_agent_async(agent, prompt, deps)

        assert result == [expected_result_obj]

    system_prompt_content = get_system_prompt(captured_messages)

    assert system_prompt_content != ""

    assert "Mock content for analysis.txt" in system_prompt_content

    found_user_text = False
    for msg in captured_messages:
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if hasattr(part, "content") and "Gandalf" in str(part.content):
                    found_user_text = True

    assert found_user_text, "Dynamic content not found in messages"


def test_analysis_agent_no_fallback() -> None:
    agent = create_analysis_agent(Settings())
    assert not isinstance(agent.model, FallbackModel)


def test_analysis_agent_with_fallback() -> None:
    settings = Settings(analysis_fallback_model="openrouter:test/fallback")
    agent = create_analysis_agent(settings)
    assert isinstance(agent.model, FallbackModel)
