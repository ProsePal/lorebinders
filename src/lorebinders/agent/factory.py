"""Agent creation, prompt building, and run utilities."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.output import OutputDataT
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT

from lorebinders.models import (
    AgentDeps,
    AnalysisResult,
    CategoryTarget,
    ExtractionResult,
    NarratorConfig,
    ObservationEvent,
    ObservationType,
    SummarizerResult,
)
from lorebinders.settings import get_settings

if TYPE_CHECKING:
    from lorebinders.settings import Settings

logger = logging.getLogger(__name__)


def load_prompt_from_assets(filename: str) -> str:
    """Load a prompt template from the assets directory.

    Args:
        filename: The name of the prompt file.

    Returns:
        The content of the prompt file.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = Path(__file__).parent / "assets" / "prompts" / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _is_moderation_error(exc: Exception) -> bool:
    return isinstance(exc, ModelHTTPError) and exc.status_code == 403


def _emit_observation(
    on_observe: Callable[[ObservationEvent], None] | None,
    event_type: ObservationType,
    stage: str,
    message: str,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> None:
    """Helper to emit observation event if callback is provided."""
    if not on_observe:
        return
    on_observe(
        ObservationEvent(
            type=event_type,
            stage=stage,
            message=message,
            metadata=metadata or {},
        )
    )


def create_agent(
    model: Model | str,
    deps_type: type[AgentDepsT],
    output_type: type[OutputDataT],
    model_settings: ModelSettings | None = None,
    fallback: Model | str | None = None,
) -> Agent[AgentDepsT, OutputDataT]:
    """Create a PydanticAI Agent with the given model.

    Args:
        model: The primary model to use.
        deps_type: Type of agent dependencies.
        output_type: Type of structured output.
        model_settings: Optional model settings.
        fallback: Optional fallback model.

    Returns:
        A configured PydanticAI Agent instance.
    """
    logger.debug(f"Creating agent for model: {model}")
    if fallback:
        model = FallbackModel(model, fallback, fallback_on=_is_moderation_error)
    return Agent(
        model,
        deps_type=deps_type,
        output_type=output_type,
        model_settings=model_settings,
    )


async def run_agent_async(
    agent: Agent[AgentDepsT, OutputDataT],
    user_prompt: str,
    deps: AgentDepsT,
    model_settings: ModelSettings | None = None,
    on_observe: Callable[[ObservationEvent], None] | None = None,
) -> OutputDataT:
    """Run an agent asynchronously and return the output.

    Args:
        agent: The AI agent.
        user_prompt: The user prompt string.
        deps: Agent dependencies.
        model_settings: Optional model settings.
        on_observe: Optional observation callback.

    Returns:
        The output data from the agent run.
    """
    model = str(agent.model) or "unknown"
    logger.debug(f"Running agent (async) with model: {model}")
    meta: dict[str, str | int | float | bool | None] = {"model": model}
    _emit_observation(
        on_observe,
        ObservationType.AGENT_RUN_STARTED,
        "agent",
        f"Running agent with model {model}",
        meta,
    )
    try:
        res = await agent.run(
            user_prompt, deps=deps, model_settings=model_settings
        )
        logger.debug("Agent run completed successfully")
        _emit_observation(
            on_observe,
            ObservationType.AGENT_RUN_COMPLETED,
            "agent",
            f"Agent run completed with model {model}",
            meta,
        )
        return res.output
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        _emit_observation(
            on_observe,
            ObservationType.ERROR,
            "agent",
            f"Agent run failed: {e}",
            {"model": model, "error": str(e)},
        )
        raise


def create_extraction_agent(
    settings: "Settings | None" = None,
) -> Agent[AgentDeps, ExtractionResult]:
    """Create a configured extraction agent.

    Args:
        settings: Optional application settings.

    Returns:
        A PydanticAI Agent configured for entity extraction.
    """
    _settings = settings or get_settings()

    agent: Agent[AgentDeps, ExtractionResult] = create_agent(
        _settings.extraction_model,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
        model_settings=_settings.extractor_model_settings,
        fallback=_settings.extraction_fallback_model,
    )

    @agent.system_prompt
    def _extraction_system_prompt(ctx: RunContext[AgentDeps]) -> str:
        return ctx.deps.prompt_loader("extraction.txt")

    return agent


def build_extraction_user_prompt(
    text: str,
    categories: list[str],
    description: str | None = None,
    narrator: NarratorConfig | None = None,
) -> str:
    """Build the user prompt for batch extraction.

    Args:
        text: The chapter text content.
        categories: List of categories to extract.
        description: Optional category description.
        narrator: Optional narrator configuration.

    Returns:
        The formatted user prompt string.
    """
    prompt = ["## CATEGORIES TO EXTRACT"]
    prompt.extend(f"- {cat}" for cat in categories)

    if description:
        prompt.append(f"Category Description: {description}")

    if narrator and narrator.is_1st_person and narrator.name:
        prompt.append(
            f"## NARRATOR HANDLING\n"
            f"This text is in first person. "
            f"The narrator is '{narrator.name}'."
        )

    prompt.append(f"## TEXT\n{text}")
    return "\n".join(prompt)


def create_analysis_agent(
    settings: "Settings | None" = None,
) -> Agent[AgentDeps, list[AnalysisResult]]:
    """Create a configured analysis agent.

    Args:
        settings: Optional application settings.

    Returns:
        A PydanticAI Agent configured for entity analysis.
    """
    _settings = settings or get_settings()

    agent: Agent[AgentDeps, list[AnalysisResult]] = create_agent(
        _settings.analysis_model,
        deps_type=AgentDeps,
        output_type=list[AnalysisResult],
        fallback=_settings.analysis_fallback_model,
    )

    @agent.system_prompt
    def _analysis_system_prompt(ctx: RunContext[AgentDeps]) -> str:
        return ctx.deps.prompt_loader("analysis.txt")

    return agent


def _add_category_to_prompt(
    prompt: list[str], category: CategoryTarget
) -> None:
    prompt.append(f"### {category.name}\nAnalyze the following traits:\n")
    if category.traits:
        prompt.extend(f"- {t}" for t in category.traits)
    prompt.append("### Entities:\n")
    prompt.extend(f"- {entity}" for entity in category.entities)


def build_analysis_user_prompt(
    context_text: str,
    categories: list[CategoryTarget],
) -> str:
    """Build user prompt for batch analysis.

    Args:
        context_text: The chapter text content.
        categories: List of target categories and entities.

    Returns:
        The formatted user prompt string.
    """
    prompt = [f"## CONTEXT\n{context_text}\n", "## TASKS"]
    for category in categories:
        _add_category_to_prompt(prompt, category)
    return "\n".join(prompt)


def create_summarization_agent(
    settings: "Settings | None" = None,
) -> Agent[AgentDeps, SummarizerResult]:
    """Create a configured summarization agent.

    Args:
        settings: Optional application settings.

    Returns:
        A PydanticAI Agent configured for entity summarization.
    """
    _settings = settings or get_settings()

    agent: Agent[AgentDeps, SummarizerResult] = create_agent(
        _settings.summarization_model,
        deps_type=AgentDeps,
        output_type=SummarizerResult,
        fallback=_settings.summarization_fallback_model,
    )

    @agent.system_prompt
    def _summarization_system_prompt(ctx: RunContext[AgentDeps]) -> str:
        return ctx.deps.prompt_loader("summarization.txt")

    return agent


def build_summarization_user_prompt(
    entity_name: str, category: str, context_data: str
) -> str:
    """Build user prompt for summarization.

    Args:
        entity_name: The name of the entity.
        category: The category of the entity.
        context_data: Formatted trait data across chapters.

    Returns:
        The formatted user prompt string.
    """
    return (
        f"## ENTITY: {entity_name} ({category})\n\n"
        f"## CONTEXT DATA\n{context_data}\n\n"
        "## TASK\nProvide a Story Bible summary."
    )
