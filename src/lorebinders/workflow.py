"""Core orchestration pipeline for LoreBinders."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from pydantic_ai import Agent

from lorebinders import models
from lorebinders.agent.analysis import analyze_entities
from lorebinders.agent.extraction import extract_book
from lorebinders.agent.factory import (
    create_analysis_agent,
    create_extraction_agent,
    create_summarization_agent,
    load_prompt_from_assets,
)
from lorebinders.agent.summarization import summarize_binder
from lorebinders.refinement import refine_binder
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

logger = logging.getLogger(__name__)

_ExtAgent: TypeAlias = Agent[models.AgentDeps, models.ExtractionResult]
_AnaAgent: TypeAlias = Agent[models.AgentDeps, list[models.AnalysisResult]]
_SumAgent: TypeAlias = Agent[models.AgentDeps, models.SummarizerResult]


def _add_custom_traits(
    category: str, traits: list[str], effective: dict[str, list[str]]
) -> None:
    """Helper to add custom traits to category.

    Args:
        category: The category name.
        traits: List of trait names.
        effective: Dictionary to update.
    """
    if category not in effective:
        effective[category] = []

    current_set = set(effective[category])
    for trait in traits:
        if trait in current_set:
            continue
        effective[category].append(trait)
        current_set.add(trait)


def merge_traits(
    settings: Settings, config: models.RunConfiguration
) -> dict[str, list[str]]:
    """Merge default settings with run configuration effective traits.

    Args:
        settings: Application settings.
        config: Run configuration.

    Returns:
        A dictionary mapping category names to lists of trait names.
    """
    effective: dict[str, list[str]] = {
        "Characters": settings.character_traits.copy(),
        "Locations": settings.location_traits.copy(),
    }

    for cat, traits in config.custom_traits.items():
        _add_custom_traits(cat, traits, effective)

    for cat in config.custom_categories:
        if cat not in effective:
            effective[cat] = []

    return effective


def _aggregate_to_binder(
    profiles: list[models.EntityProfile],
) -> models.Binder:
    """Aggregate profiles into the Binder model, cleaning traits.

    Args:
        profiles: List of analyzed entity profiles.

    Returns:
        A Binder model containing aggregated and cleaned entity profiles.
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
    extraction_agent: _ExtAgent | None = None,
    analysis_agent: _AnaAgent | None = None,
    summarization_agent: _SumAgent | None = None,
    provider: type[StorageProvider] = FilesystemStorage,
) -> Path:
    """Execute the LoreBinders build pipeline.

    Args:
        config: The run configuration.
        progress: Optional progress callback.
        on_observe: Optional observation callback.
        extraction_agent: Optional agent override.
        analysis_agent: Optional agent override.
        summarization_agent: Optional agent override.
        provider: Storage provider class.

    Returns:
        The Path to the generated story bible report.
    """
    settings = get_settings()
    deps = models.AgentDeps(
        settings=settings, prompt_loader=load_prompt_from_assets
    )

    ext_agent = extraction_agent or create_extraction_agent(settings)
    ana_agent = analysis_agent or create_analysis_agent(settings)
    sum_agent = summarization_agent or create_summarization_agent(settings)

    traits = merge_traits(settings, config)
    storage = get_storage(provider)
    storage.set_workspace(config.author_name, config.book_title)

    models.emit_observation(
        on_observe,
        models.ObservationType.STAGE_STARTED,
        "ingestion",
        f"Ingesting {config.book_path.name}",
    )
    text = await asyncio.to_thread(convert_to_text, config.book_path)
    await asyncio.to_thread(storage.save_book, config.book_title, text)
    book = ingest(text, config.book_path.stem)

    models.emit_observation(
        on_observe,
        models.ObservationType.STAGE_STARTED,
        "extraction",
        "Starting extraction",
        {"total_chapters": len(book.chapters)},
    )
    raw = await extract_book(
        book,
        ext_agent,
        deps,
        list(traits.keys()),
        config,
        storage,
        progress,
        on_observe,
    )

    narrator = config.narrator_config.name
    sorted_ext = sort_extractions(raw, narrator)

    models.emit_observation(
        on_observe,
        models.ObservationType.STAGE_STARTED,
        "analysis",
        "Starting analysis",
        {"total_batches": sum(len(e) for e in sorted_ext.values())}
        if on_observe
        else None,
    )
    profiles = await analyze_entities(
        sorted_ext, book, ana_agent, deps, traits, storage, progress, on_observe
    )

    models.emit_observation(
        on_observe,
        models.ObservationType.STAGE_STARTED,
        "refinement",
        "Refining binder data",
    )
    raw_binder = _aggregate_to_binder(profiles)
    binder = refine_binder(raw_binder, config.narrator_config.name)

    total_ent = sum(len(c.entities) for c in binder.categories.values())
    models.emit_observation(
        on_observe,
        models.ObservationType.STAGE_STARTED,
        "summarization",
        "Starting summarization",
        {"total_entities": total_ent},
    )
    await summarize_binder(
        binder, storage, sum_agent, deps, progress, on_observe
    )

    safe_title = sanitize_filename(config.book_title)
    output_dir = storage.path
    output_file = output_dir / f"{safe_title}_story_bible.pdf"

    models.emit_observation(
        on_observe,
        models.ObservationType.STAGE_STARTED,
        "reporting",
        f"Generating PDF to {output_file.name}",
    )
    await asyncio.to_thread(generate_pdf_report, binder, output_file)

    if on_observe:
        on_observe(
            models.ObservationEvent(
                type=models.ObservationType.STAGE_COMPLETED,
                stage="workflow",
                message="Build complete!",
                metadata={"output_file": str(output_file)},
            )
        )

    return output_file
