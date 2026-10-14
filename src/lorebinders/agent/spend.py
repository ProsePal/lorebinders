import asyncio
from copy import copy

from pydantic_ai.usage import RequestUsage, RunUsage


class SpendError(RuntimeError):
    """Raised when spend ceiling is exceeded."""


class SpendUsage(RunUsage):
    """Account for each response, estimating costs missing from usage."""

    def __init__(self, models: tuple[str, ...]) -> None:
        """Track usage across retries and the configured fallback models."""
        super().__init__()
        self._models = models

    def incr(self, incr_usage: RunUsage | RequestUsage) -> None:
        """Preserve known prices and estimate unpriced response tokens."""
        if incr_usage.cost is None and (
            incr_usage.input_tokens or incr_usage.output_tokens
        ):
            from decimal import Decimal

            incr_usage = copy(incr_usage)
            incr_usage.cost = Decimal(
                str(
                    max(
                        estimate_cost(
                            model,
                            incr_usage.input_tokens,
                            incr_usage.output_tokens,
                        )
                        for model in self._models
                    )
                )
            )
        super().incr(incr_usage)


class Spend:
    """Running total of billed cost, with an abort ceiling."""

    def __init__(self, limit: float | None = None) -> None:
        """Start an empty tally with the given ceiling in dollars."""
        self.limit = limit
        self.total = 0.0
        self.reserved = 0.0
        self._lock = asyncio.Lock()

    def _check_ceiling(self, additional: float = 0.0) -> None:
        """Raise SpendError if the spend ceiling is exceeded."""
        projected = self.total + self.reserved + additional
        if self.limit is not None and projected > self.limit:
            effective = self.total if self.total > self.limit else projected
            raise SpendError(
                f"Spend ceiling exceeded: ${effective:.2f} of "
                f"${self.limit:.2f}. Aborting run."
            )

    async def reserve(self, estimated_cost: float = 0.0) -> None:
        """Reserve spend before dispatch, raising if ceiling exceeded."""
        async with self._lock:
            self._check_ceiling(estimated_cost)
            self.reserved += estimated_cost

    async def release(
        self, estimated_cost: float = 0.0, actual_cost: float | None = None
    ) -> None:
        """Release reserved spend and optionally record actual spend."""
        async with self._lock:
            self.reserved = max(0.0, self.reserved - estimated_cost)
            if actual_cost is not None:
                self.total += actual_cost
                self._check_ceiling()


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for an OpenRouter model."""
    cost_per_m_in = 0.0
    cost_per_m_out = 0.0
    model_lower = model.lower()

    if "deepseek-v3" in model_lower:
        cost_per_m_in = 0.14
        cost_per_m_out = 0.28
    elif "seed-1.6" in model_lower or "flash" in model_lower:
        cost_per_m_in = 0.05
        cost_per_m_out = 0.15
    elif "claude-3-5" in model_lower:
        cost_per_m_in = 3.0
        cost_per_m_out = 15.0
    else:
        cost_per_m_in = 1.0
        cost_per_m_out = 3.0
    return (
        input_tokens * cost_per_m_in + output_tokens * cost_per_m_out
    ) / 1_000_000
