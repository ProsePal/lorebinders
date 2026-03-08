import logging
from collections.abc import Callable
from pathlib import Path

from pydantic_ai import Agent

from lorebinders import models
from lorebinders.agent import (
    create_analysis_agent,
    create_extraction_agent,
    create_summarization_agent,
    load_prompt_from_assets,
)
from lorebinders.agent.analysis import analyze_entities
from lorebinders.agent.extraction import extract_book
from lorebinders.agent.summarization import summarize_binder
from lorebinders.refinement.cleaning import clean_traits
from lorebinders.refinement.conversion import convert_to_text, ingest
from lorebinders.refinement.sorting import sort_extractions
from lorebinders.reporting.pdf import generate_pdf_report
from lorebinders.settings import Settings, get_settings
from lorebinders.storage import (
    FilesystemStorage,
    StorageProvider,
    get_storage,
    sanitize_filename,
)
from lorebinders.storage.workspace import ensure_workspace

logger = logging.getLogger(__name__)


def merge_traits(
    settings: Settings, config: models.RunConfiguration
) -> dict[str, list[str]]:
    """Merge default settings with run configuration to get effective traits.

    Args:
        settings: Application settings with defaults.
        config: Run configuration with overrides/extensions.

    Returns:
        A dictionary mapping category names to their list of traits.
    """
    effective_traits: dict[str, list[str]] = {
        "Characters": settings.character_traits.copy(),
        "Locations": settings.location_traits.copy(),
    }

    for category, traits in config.custom_traits.items():
        if category not in effective_traits:
            effective_traits[category] = []

        current_set = set(effective_traits[category])
        for trait in traits:
            if trait not in current_set:
                effective_traits[category].append(trait)
                current_set.add(trait)

    for category in config.custom_categories:
        if category not in effective_traits:
            effective_traits[category] = []

    return effective_traits


def _aggregate_to_binder(
    profiles: list[models.EntityProfile],
) -> models.Binder:
    """Aggregate profiles into the Binder model, cleaning traits.

    Args:
        profiles: The list of entity profiles to aggregate.

    Returns:
        A Binder model containing all aggregated entities.
    """
    binder = models.Binder()
    for p in profiles:
        if cleaned := clean_traits(p.traits):
            binder.add_appearance(
                category=p.category,
                name=p.name,
                chapter=p.chapter_number,
                traits=cleaned,
            )
    return binder


async def build_binder(
    config: models.RunConfiguration,
    progress: Callable[[models.ProgressUpdate], None] | None = None,
    on_observe: Callable[[models.ObservationEvent], None] | None = None,
    extraction_agent: Agent[models.AgentDeps, models.ExtractionResult]
    | None = None,
    analysis_agent: Agent[models.AgentDeps, list[models.AnalysisResult]]
    | None = None,
    summarization_agent: Agent[models.AgentDeps, models.SummarizerResult]
    | None = None,
    provider: type[StorageProvider] = FilesystemStorage,
) -> Path:
    """Execute the LoreBinders build pipeline.

    Args:
        config: The run configuration.
        progress: Optional callback for progress updates.
        on_observe: Optional callback for rich observation events.
        extraction_agent: Optional agent for extraction.
        analysis_agent: Optional agent for analysis.
        summarization_agent: Optional agent for summarization.
        provider: Optional storage provider class.

    Returns:
        Path: The path to the generated PDF.
    """
    settings = get_settings()
    deps = models.AgentDeps(
        settings=settings, prompt_loader=load_prompt_from_assets
    )

    ext_agent = extraction_agent or create_extraction_agent(settings)
    ana_agent = analysis_agent or create_analysis_agent(settings)
    sum_agent = summarization_agent or create_summarization_agent(settings)

    effective_traits = merge_traits(settings, config)
    all_categories = list(effective_traits.keys())

    storage = get_storage(provider)

    logger.info(
        f"Starting binder build for {config.book_title} by {config.author_name}"
    )
    storage.set_workspace(config.author_name, config.book_title)

    if on_observe:
        on_observe(
            models.ObservationEvent(
                type=models.ObservationType.STAGE_STARTED,
                stage="ingestion",
                message=f"Ingesting {config.book_path.name}",
            )
        )

    logger.debug("Ingesting book...")
    book_text = convert_to_text(config.book_path)
    storage.save_book(config.book_title, book_text)
    book = ingest(book_text, config.book_path.stem)

    if on_observe:
        on_observe(
            models.ObservationEvent(
                type=models.ObservationType.STAGE_STARTED,
                stage="extraction",
                message="Starting entity extraction",
                metadata={"total_chapters": len(book.chapters)},
            )
        )

    logger.debug("Starting extraction phase...")
    raw_extractions = await extract_book(
        book,
        ext_agent,
        deps,
        all_categories,
        config,
        storage,
        progress,
        on_observe,
    )

    logger.debug("Starting early refinement...")
    narrator_name = (
        config.narrator_config.name if config.narrator_config else None
    )
    sorted_extractions = sort_extractions(raw_extractions, narrator_name)

    if on_observe:
        total_batches = sum(
            len(entities) for entities in sorted_extractions.values()
        )
        on_observe(
            models.ObservationEvent(
                type=models.ObservationType.STAGE_STARTED,
                stage="analysis",
                message="Starting entity analysis",
                metadata={"total_batches": total_batches},
            )
        )

    logger.debug("Starting analysis phase...")
    profiles = await analyze_entities(
        sorted_extractions,
        book,
        ana_agent,
        deps,
        effective_traits,
        storage,
        progress=progress,
        on_observe=on_observe,
    )

    logger.debug("Aggregating profiles and cleaning traits...")
    binder = _aggregate_to_binder(profiles)

    if on_observe:
        total_entities = sum(
            len(c.entities) for c in binder.categories.values()
        )
        on_observe(
            models.ObservationEvent(
                type=models.ObservationType.STAGE_STARTED,
                stage="summarization",
                message="Starting entity summarization",
                metadata={"total_entities": total_entities},
            )
        )

    logger.debug("Starting summarization phase...")
    await summarize_binder(
        binder,
        storage,
        sum_agent,
        deps,
        progress=progress,
        on_observe=on_observe,
    )

    safe_title = sanitize_filename(config.book_title)
    output_dir = ensure_workspace(config.author_name, config.book_title)
    output_file = output_dir / f"{safe_title}_story_bible.pdf"

    if on_observe:
        on_observe(
            models.ObservationEvent(
                type=models.ObservationType.STAGE_STARTED,
                stage="reporting",
                message=f"Generating PDF report to {output_file.name}",
            )
        )

    logger.debug(f"Generating report to {output_file}...")
    generate_pdf_report(binder, output_file)
    logger.debug("Report generation complete.")

    if on_observe:
        on_observe(
            models.ObservationEvent(
                type=models.ObservationType.STAGE_COMPLETED,
                stage="workflow",
                message="Binder build complete!",
                metadata={"output_file": str(output_file)},
            )
        )

    return output_file
