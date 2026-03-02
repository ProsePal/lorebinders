from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel

from lorebinders.agent.factory import _is_moderation_error, create_agent
from lorebinders.models import AgentDeps, ExtractionResult


def test_is_moderation_error_true() -> None:
    exc = ModelHTTPError(status_code=403, model_name="test/model")
    assert _is_moderation_error(exc) is True


def test_is_moderation_error_false_wrong_status() -> None:
    exc = ModelHTTPError(status_code=500, model_name="test/model")
    assert _is_moderation_error(exc) is False


def test_is_moderation_error_false_wrong_type() -> None:
    assert _is_moderation_error(ValueError("boom")) is False


def test_create_agent_no_fallback_returns_plain_agent() -> None:
    primary = TestModel()
    agent = create_agent(
        primary,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
    )
    assert not isinstance(agent.model, FallbackModel)


def test_create_agent_with_fallback_wraps_in_fallback_model() -> None:
    primary = TestModel()
    fallback = TestModel()
    agent = create_agent(
        primary,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
        fallback_model=fallback,
    )
    assert isinstance(agent.model, FallbackModel)
