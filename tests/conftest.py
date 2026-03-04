import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def mock_env_vars():
    """Mock environment variables to prevent pydantic-ai provider errors."""
    os.environ["OPENROUTER_API_KEY"] = "mock_key"
    yield
