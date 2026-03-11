import json

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from lorebinders.agent.factory import (
    build_analysis_user_prompt,
    build_extraction_user_prompt,
    create_analysis_agent,
    create_extraction_agent,
    run_agent_async,
)
from lorebinders.models import AgentDeps, CategoryTarget, NarratorConfig
from lorebinders.settings import Settings


def mock_prompt_loader(filename: str) -> str:
    return f"Mock content for {filename}"


@pytest.mark.anyio
async def test_agents_flow() -> None:
    def mock_extract_call(
        messages: list[ModelMessage], info: object
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content=json.dumps(
                        {
                            "results": [
                                {
                                    "category": "Characters",
                                    "entities": [
                                        "Sherlock Holmes",
                                        "Dr. Watson",
                                    ],
                                }
                            ]
                        }
                    )
                )
            ]
        )

    def mock_analyze_call(
        messages: list[ModelMessage], info: object
    ) -> ModelResponse:
        result = [
            {
                "entity_name": "Sherlock Holmes",
                "category": "Character",
                "traits": [
                    {
                        "trait": "Role",
                        "value": "Detective",
                        "evidence": "The world's only consulting detective",
                    }
                ],
            }
        ]
        return ModelResponse(
            parts=[TextPart(content=json.dumps({"response": result}))]
        )

    settings = Settings()
    deps = AgentDeps(settings=settings, prompt_loader=mock_prompt_loader)

    extraction_agent = create_extraction_agent(settings)
    analysis_agent = create_analysis_agent(settings)

    with (
        extraction_agent.override(model=FunctionModel(mock_extract_call)),
        analysis_agent.override(model=FunctionModel(mock_analyze_call)),
    ):
        text_chunk = (
            "Sherlock Holmes sat in his chair. Dr. Watson looked at him. "
            "The world's only consulting detective was thinking."
        )

        extraction_prompt = build_extraction_user_prompt(
            text_chunk,
            categories=["Characters"],
            narrator=NarratorConfig(is_1st_person=False),
        )

        result = await run_agent_async(
            extraction_agent, extraction_prompt, deps
        )
        entities = result.to_dict()

        assert "Characters" in entities
        assert "Sherlock Holmes" in entities["Characters"]
        assert "Dr. Watson" in entities["Characters"]

        analysis_prompt = build_analysis_user_prompt(
            text_chunk,
            categories=[
                CategoryTarget(
                    name="Character",
                    entities=["Sherlock Holmes"],
                    traits=["Role"],
                )
            ],
        )

        results = await run_agent_async(analysis_agent, analysis_prompt, deps)

        assert len(results) == 1
        assert results[0].entity_name == "Sherlock Holmes"
        assert results[0].traits[0].value == "Detective"
