import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import RunOutputDataT
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT

from lorebinders.models import (
    AgentDeps,
    AnalysisResult,
    CategoryTarget,
    ExtractionResult,
    NarratorConfig,
    SummarizerResult,
)

if TYPE_CHECKING:
    from lorebinders.settings import Settings


def load_prompt_from_assets(filename: str) -> str:
    """Load a prompt template from the assets directory.

    Args:
        filename: The name of the prompt file to load.

    Returns:
        The content of the prompt file.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = Path(__file__).parent / "assets" / "prompts" / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8")


logger = logging.getLogger(__name__)


def _is_moderation_error(exc: Exception) -> bool:
    return isinstance(exc, ModelHTTPError) and exc.status_code == 403


def create_agent(
    model_name: Model | str,
    deps_type: type[AgentDepsT],
    output_type: type[RunOutputDataT],
    model_settings: ModelSettings | None = None,
    fallback_model: Model | str | None = None,
) -> Agent[AgentDepsT, RunOutputDataT]:
    """Create a PydanticAI Agent with the given model.

    Args:
        model_name: The model instance or name string to use.
        deps_type: The type of dependencies to inject.
        output_type: The type of the expected output.
        model_settings: Optional settings for the model.
        fallback_model: Optional fallback model instance or name string
            used when the primary model returns HTTP 403 (content
            moderation).

    Returns:
        A configured PydanticAI Agent.
    """
    logger.debug(f"Creating agent for model: {model_name}")
    if fallback_model:
        model: Model | str | FallbackModel = FallbackModel(
            model_name,
            fallback_model,
            fallback_on=(_is_moderation_error,),
        )
    else:
        model = model_name
    return Agent(
        model,
        deps_type=deps_type,
        output_type=output_type,
        model_settings=model_settings,
    )


def run_agent(
    agent: Agent[AgentDepsT, RunOutputDataT],
    user_prompt: str,
    deps: AgentDepsT,
    model_settings: ModelSettings | None = None,
) -> RunOutputDataT:
    """Run an agent synchronously and return the output.

    Args:
        agent: The agent to run.
        user_prompt: The user prompt to send to the agent.
        deps: The dependencies to inject into the agent.
        model_settings: Optional settings for the model.

    Returns:
        The output data from the agent run.
    """
    model_name = getattr(agent.model, "model_name", "unknown")
    logger.debug(f"Running agent with model: {model_name}")
    try:
        result = agent.run_sync(
            user_prompt, deps=deps, model_settings=model_settings
        )
        logger.debug("Agent run completed successfully")
        return result.output
    except Exception as e:
        logger.error(f"Agent run failed: {e}")
        raise


def create_extraction_agent(
    settings: "Settings | None" = None,
) -> Agent[AgentDeps, ExtractionResult]:
    """Create a configured extraction agent.

    Args:
        settings: Optional settings to use for the agent.

    Returns:
        A configured extraction agent.
    """
    if settings is None:
        from lorebinders.settings import get_settings

        settings = get_settings()

    agent = create_agent(
        settings.extraction_model,
        deps_type=AgentDeps,
        output_type=ExtractionResult,
        model_settings=settings.extractor_model_settings,
        fallback_model=settings.extraction_fallback_model,
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
    """Build the user prompt for batch extraction of all categories.

    Args:
        text: The text to extract entities from.
        categories: A list of categories to extract.
        description: Optional description of the categories.
        narrator: Optional narrator configuration.

    Returns:
        The constructed user prompt.
    """
    prompt = ["## CATEGORIES TO EXTRACT"]
    prompt.extend([f"- {cat}" for cat in categories])

    if description:
        prompt.append(f"Category Description: {description}")

    if narrator and narrator.is_1st_person and narrator.name:
        prompt.append(
            "## NARRATOR HANDLING\n"
            f"This text is in first person. The narrator is '{narrator.name}'."
        )

    prompt.append(f"## TEXT\n{text}")
    return "\n".join(prompt)


def create_analysis_agent(
    settings: "Settings | None" = None,
) -> Agent[AgentDeps, list[AnalysisResult]]:
    """Create a configured analysis agent.

    Args:
        settings: Optional settings to use for the agent.

    Returns:
        A configured analysis agent.
    """
    if settings is None:
        from lorebinders.settings import get_settings

        settings = get_settings()
    agent = create_agent(
        settings.analysis_model,
        deps_type=AgentDeps,
        output_type=list[AnalysisResult],
        fallback_model=settings.analysis_fallback_model,
    )

    @agent.system_prompt
    def _analysis_system_prompt(ctx: RunContext[AgentDeps]) -> str:
        return ctx.deps.prompt_loader("analysis.txt")

    return agent


def build_analysis_user_prompt(
    context_text: str,
    categories: list[CategoryTarget],
) -> str:
    """Build user prompt for batch analysis.

    Args:
        context_text: The context text for analysis.
        categories: A list of target categories for analysis.

    Returns:
        The constructed user prompt.
    """
    prompt = [f"## CONTEXT\n{context_text}\n", "## TASKS"]
    for category in categories:
        prompt.append(f"### {category.name}\nAnalyze the following traits:\n")
        if category.traits:
            prompt.append("\n- ".join(category.traits))
        prompt.append("### Entities:\n")
        prompt.extend([f"- {entity}" for entity in category.entities])

    return "\n".join(prompt)


def create_summarization_agent(
    settings: "Settings | None" = None,
) -> Agent[AgentDeps, SummarizerResult]:
    """Create a configured summarization agent.

    Args:
        settings: Optional settings to use for the agent.

    Returns:
        A configured summarization agent.
    """
    if settings is None:
        from lorebinders.settings import get_settings

        settings = get_settings()
    agent = create_agent(
        settings.summarization_model,
        deps_type=AgentDeps,
        output_type=SummarizerResult,
        fallback_model=settings.summarization_fallback_model,
    )

    @agent.system_prompt
    def _summarization_system_prompt(ctx: RunContext[AgentDeps]) -> str:
        return ctx.deps.prompt_loader("summarization.txt")

    return agent


def build_summarization_user_prompt(
    entity_name: str,
    category: str,
    context_data: str,
) -> str:
    """Build user prompt for summarization.

    Args:
        entity_name: The name of the entity to summarize.
        category: The category of the entity.
        context_data: The context data for the entity.

    Returns:
        The constructed user prompt.
    """
    return (
        f"## ENTITY: {entity_name} ({category})\n\n"
        f"## CONTEXT DATA\n{context_data}\n\n"
        f"## TASK\nProvide a Story Bible summary."
    )
