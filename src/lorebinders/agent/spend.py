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


class RequestBudget:
    """Reserve and reconcile each request against a shared pipeline budget."""

    def __init__(self, spend: Spend | None, reservation: float) -> None:
        """Adopt the reservation made before entering the agent run."""
        self.spend = spend
        self.reservation = reservation
        self.active = spend is not None

    async def reserve_request(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        """Check the budget before each primary, retry, or fallback call."""
        if self.spend is None:
            return
        estimate = estimate_cost(model, input_tokens, output_tokens)
        additional = max(0.0, estimate - self.reservation)
        if not self.active:
            additional = estimate
        await self.spend.reserve(additional)
        self.reservation += additional
        self.active = True

    async def release(self, cost: float | None = None) -> None:
        """Finish accounting even if cancellation arrives while locking."""
        if self.spend is None or not self.active:
            return
        task = asyncio.create_task(
            self.spend.release(self.reservation, actual_cost=cost)
        )
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        self.active = False
        self.reservation = 0.0
        if cancelled:
            task.exception()
            raise asyncio.CancelledError
        task.result()


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
