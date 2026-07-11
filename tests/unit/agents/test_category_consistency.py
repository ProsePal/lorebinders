from unittest.mock import AsyncMock, patch

import pytest

from lorebinders.agent.analysis import _run_analysis_batch
from lorebinders.models import (
    AgentDeps,
    AnalysisResult,
    CategoryTarget,
    Chapter,
)


@pytest.mark.anyio
async def test_run_analysis_batch_enforces_category_consistency() -> None:
    chapter = Chapter(number=1, title="Ch 1", content="Text")
    targets = [
        CategoryTarget(name="Characters", entities=["Kalia"]),
        CategoryTarget(name="Locations", entities=["Hill"]),
    ]

    # Mock result with "wrong" categories
    mock_results = [
        AnalysisResult(entity_name="Kalia", category="Character", traits=[]),
        AnalysisResult(entity_name="Hill", category="Location", traits=[]),
    ]

    agent = AsyncMock()
    deps = AgentDeps(settings=AsyncMock(), prompt_loader=AsyncMock())

    with patch(
        "lorebinders.agent.analysis.run_agent_async", return_value=mock_results
    ):
        results = await _run_analysis_batch(
            target_categories=targets,
            chapter=chapter,
            agent=agent,
            deps=deps,
            effective_traits={},
        )

    assert results[0].entity_name == "Kalia"
    assert results[0].category == "Characters"
    assert results[1].entity_name == "Hill"
    assert results[1].category == "Locations"
