import os
from collections.abc import Generator

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and filters."""
    config.addinivalue_line(
        "filterwarnings", "ignore::DeprecationWarning:google.genai.types"
    )


@pytest.fixture(autouse=True, scope="session")
def mock_env_vars() -> Generator[None, None, None]:
    """Mock environment variables to prevent pydantic-ai provider errors."""
    os.environ["OPENROUTER_API_KEY"] = "mock_key"
    yield
