import asyncio


class SpendError(RuntimeError):
    """Raised when spend ceiling is exceeded."""


class Spend:
    """Running total of billed cost, with an abort ceiling."""

    def __init__(self, limit: float | None = None) -> None:
        """Start an empty tally with the given ceiling in dollars."""
        self.limit = limit
        self.total = 0.0
        self._lock = asyncio.Lock()

    def _check_ceiling(self) -> None:
        """Raise SpendError if the spend ceiling is exceeded."""
        if self.limit is not None and self.total > self.limit:
            raise SpendError(
                f"Spend ceiling exceeded: ${self.total:.2f} of "
                f"${self.limit:.2f}. Aborting run."
            )

    @property
    def exceeded(self) -> bool:
        """Whether current spend exceeds the configured limit."""
        return self.limit is not None and self.total > self.limit

    async def check(self) -> None:
        """Verify spend has not exceeded ceiling.

        Raises:
            SpendError: If the ceiling is exceeded.
        """
        async with self._lock:
            self._check_ceiling()

    async def add(self, cost: float) -> None:
        """Record spend, raising once the ceiling is crossed."""
        async with self._lock:
            self.total += cost
            self._check_ceiling()


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for an OpenRouter model."""
    cost_per_m_in = 0.0
    cost_per_m_out = 0.0
    model_lower = model.lower()

    # Minimal fallback pricing just to not be 0
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
        # Generic fallback
        cost_per_m_in = 1.0
        cost_per_m_out = 3.0

    total_cents = input_tokens * cost_per_m_in + output_tokens * cost_per_m_out
    return total_cents / 1_000_000
