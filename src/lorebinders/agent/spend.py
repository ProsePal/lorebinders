import asyncio


class SpendError(RuntimeError):
    """Raised when spend ceiling is exceeded."""


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

    total_cents = input_tokens * cost_per_m_in + output_tokens * cost_per_m_out
    return total_cents / 1_000_000
