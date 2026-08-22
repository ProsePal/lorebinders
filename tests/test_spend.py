from unittest.mock import patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from lorebinders.agent.factory import run_agent_async
from lorebinders.agent.spend import Spend, SpendError, estimate_cost
from lorebinders.models import AgentDeps
from lorebinders.settings import Settings


@pytest.mark.anyio
async def test_spend_limit() -> None:
    spend = Spend(limit=1.0)

    # At or below shouldn't raise
    await spend.add(0.5)
    assert spend.total == 0.5
    await spend.add(0.5)
    assert spend.total == 1.0

    # Exceeding should raise SpendError
    with pytest.raises(SpendError):
        await spend.add(0.1)


@pytest.mark.anyio
async def test_spend_no_limit() -> None:
    spend = Spend(limit=None)
    # Should never raise
    await spend.add(1000.0)
    assert spend.total == 1000.0


@pytest.mark.anyio
async def test_run_agent_async_spend_error_propagates() -> None:
    spend = Spend(limit=0.0)
    settings = Settings()
    deps = AgentDeps(
        settings=settings, prompt_loader=lambda x: "prompt", spend=spend
    )

    # Use a TestModel and mock usage so that cost > 0
    model = TestModel()
    agent = Agent(model, deps_type=AgentDeps)

    with patch("lorebinders.agent.spend.estimate_cost", return_value=1.0):
        with pytest.raises(SpendError, match="Spend ceiling exceeded"):
            # Mock estimate_cost to return 1.0 > 0.0 limit
            #
            # and estimate_cost will return 1.0, which > 0.0 limit
            await run_agent_async(agent, "test prompt", deps)


def test_estimate_cost_slug_match() -> None:
    cost = estimate_cost("deepseek-v3.2", 1_000_000, 1_000_000)
    assert cost == 0.42

    cost_flash = estimate_cost("seed-1.6-flash", 1_000_000, 1_000_000)
    assert cost_flash == 0.20

    cost_unknown = estimate_cost("something-unknown", 1_000_000, 1_000_000)
    assert cost_unknown == 4.0
