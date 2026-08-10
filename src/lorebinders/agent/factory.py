"""Agent creation, prompt building, and run utilities."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT

from lorebinders.agent_settings import provider_factory
from lorebinders.models import (
    AgentDeps,
    AliasResolution,
    AnalysisResult,
    CategoryTarget,
    ExtractionResult,
    NarratorConfig,
    ObservationEvent,
    ObservationType,
    SummarizerResult,
    emit_observation,
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


def create_agent(
    model: Model | str,
    deps_type: type[AgentDepsT],
    output_type: OutputSpec[OutputDataT],
    model_settings: ModelSettings | None = None,
    fallback: Model | str | None = None,
) -> Agent[AgentDepsT, OutputDataT]:
    """Create a PydanticAI Agent with the given model.

    Args:
        model: The primary model to use.
        deps_type: Type of agent dependencies.
        output_type: Structured output spec (plain type, PromptedOutput,
            NativeOutput, etc.).
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
    emit_observation(
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
        emit_observation(
            on_observe,
            ObservationType.AGENT_RUN_COMPLETED,
            "agent",
            f"Agent run completed with model {model}",
            meta,
        )
        try:
            usage = res.usage()
            emit_observation(
                on_observe,
                ObservationType.METRIC,
                "agent",
                f"Token usage for model {model}",
                {
                    "model": model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to collect token usage metrics: {e}")
        return res.output
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        emit_observation(
            on_observe,
            ObservationType.ERROR,
            "agent",
            f"Agent run failed: {e}",
            {"model": model, "error": str(e)},
        )
        raise


def init_extraction_model(settings: "Settings") -> Model:
    """Initialize the extraction model.

    Args:
        settings: Optional application settings.

    Returns:
        The initialized extraction model.
    """
    return infer_model(settings.extraction_model, provider_factory)


def create_extraction_agent(
    settings: "Settings | None" = None,
    output_type: "OutputSpec[ExtractionResult] | None" = None,
) -> Agent[AgentDeps, ExtractionResult]:
    """Create a configured extraction agent.

    Args:
        settings: Optional application settings.
        output_type: Optional output spec override. Defaults to plain
            ``ExtractionResult`` (tool-based structured output). Pass
            ``PromptedOutput(ExtractionResult)`` to use prompt-based JSON
            extraction instead, which works with models that do not support
            tool calling.

    Returns:
        A PydanticAI Agent configured for entity extraction.
    """
    _settings = settings or get_settings()
    _output: OutputSpec[ExtractionResult] = (
        output_type if output_type is not None else ExtractionResult
    )

    agent: Agent[AgentDeps, ExtractionResult] = create_agent(
        init_extraction_model(_settings),
        deps_type=AgentDeps,
        output_type=_output,
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


def init_analysis_model(settings: "Settings") -> Model:
    """Initialize the analysis model.

    Args:
        settings: Application settings.

    Returns:
        The initialized analysis model.
    """
    return infer_model(settings.analysis_model, provider_factory)


def create_analysis_agent(
    settings: "Settings | None" = None,
    output_type: "OutputSpec[list[AnalysisResult]] | None" = None,
) -> Agent[AgentDeps, list[AnalysisResult]]:
    """Create a configured analysis agent.

    Args:
        settings: Optional application settings.
        output_type: Optional output spec override. Defaults to plain
            ``list[AnalysisResult]`` (tool-based structured output). Pass
            ``PromptedOutput(list[AnalysisResult])`` to use prompt-based JSON
            extraction instead, which works with models that do not support
            tool calling.

    Returns:
        A PydanticAI Agent configured for entity analysis.
    """
    _settings = settings or get_settings()
    _output: OutputSpec[list[AnalysisResult]] = (
        output_type if output_type is not None else list[AnalysisResult]
    )

    agent: Agent[AgentDeps, list[AnalysisResult]] = create_agent(
        init_analysis_model(_settings),
        deps_type=AgentDeps,
        output_type=_output,
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


def init_alias_resolution_model(settings: "Settings") -> Model:
    """Initialize the alias resolution model.

    Args:
        settings: Application settings.

    Returns:
        The initialized alias resolution model.
    """
    return infer_model(settings.alias_resolution_model, provider_factory)


def create_alias_resolution_agent(
    settings: "Settings | None" = None,
    output_type: "OutputSpec[AliasResolution] | None" = None,
) -> Agent[AgentDeps, AliasResolution]:
    """Create a configured alias resolution agent.

    Args:
        settings: Optional application settings.
        output_type: Optional output spec override. Defaults to plain
            ``AliasResolution`` (tool-based structured output). Pass
            ``PromptedOutput(AliasResolution)`` to use prompt-based JSON
            extraction instead, which works with models that do not support
            tool calling.

    Returns:
        A PydanticAI Agent configured for entity alias resolution.
    """
    _settings = settings or get_settings()
    _output: OutputSpec[AliasResolution] = (
        output_type if output_type is not None else AliasResolution
    )

    agent: Agent[AgentDeps, AliasResolution] = create_agent(
        init_alias_resolution_model(_settings),
        deps_type=AgentDeps,
        output_type=_output,
        fallback=_settings.alias_resolution_fallback_model,
    )

    @agent.system_prompt
    def _alias_resolution_system_prompt(ctx: RunContext[AgentDeps]) -> str:
        return ctx.deps.prompt_loader("alias_resolution.txt")

    return agent


def build_alias_resolution_user_prompt(
    category: str, entries: list[tuple[str, str]]
) -> str:
    """Build the user prompt for alias resolution within one category.

    Args:
        category: The category whose entities are being resolved.
        entries: Pairs of entity name and a short excerpt of its notes.

    Returns:
        The formatted user prompt string.
    """
    prompt = [f"## CATEGORY: {category}", "", "## ENTITIES"]
    prompt.extend(
        f"- {name}: {context}" if context else f"- {name}"
        for name, context in entries
    )
    prompt.append(
        "\n## TASK\nGroup the names above that refer to the same entity."
    )
    return "\n".join(prompt)


def init_summarization_model(settings: "Settings") -> Model:
    """Initialize the summarization model.

    Args:
        settings: Application settings.

    Returns:
        The initialized summarization model.
    """
    return infer_model(settings.summarization_model, provider_factory)


def create_summarization_agent(
    settings: "Settings | None" = None,
    output_type: "OutputSpec[SummarizerResult] | None" = None,
) -> Agent[AgentDeps, SummarizerResult]:
    """Create a configured summarization agent.

    Args:
        settings: Optional application settings.
        output_type: Optional output spec override. Defaults to plain
            ``SummarizerResult`` (tool-based structured output). Pass
            ``PromptedOutput(SummarizerResult)`` to use prompt-based JSON
            extraction instead, which works with models that do not support
            tool calling.

    Returns:
        A PydanticAI Agent configured for entity summarization.
    """
    _settings = settings or get_settings()
    _output: OutputSpec[SummarizerResult] = (
        output_type if output_type is not None else SummarizerResult
    )

    agent: Agent[AgentDeps, SummarizerResult] = create_agent(
        init_summarization_model(_settings),
        deps_type=AgentDeps,
        output_type=_output,
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
