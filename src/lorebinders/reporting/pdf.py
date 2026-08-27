"""Module for generating PDF reports using ReportLab."""

from collections import defaultdict
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import StyleSheet1
from reportlab.platypus import (
    Flowable,
    ListFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from lorebinders.models import (
    Binder,
    ChapterAppearances,
    EntityRecord,
    SingleAppearance,
)
from lorebinders.reporting.styles import get_document_styles


def _create_occurrence_item(
    location_key: str, val: str | list[str], styles: StyleSheet1
) -> Paragraph:
    """Create a list item for a trait occurrence.

    Returns:
        A Paragraph representing the trait occurrence.
    """
    val_str = ", ".join(val) if isinstance(val, list) else str(val)
    text = f"{location_key}: {val_str}"
    return Paragraph(text, styles["Normal"])


def _add_trait_section(
    story: list[Flowable],
    trait_name: str,
    occurrences: dict[str, str | list[str]],
    styles: StyleSheet1,
) -> None:
    """Add a single trait and its occurrences to the report."""
    story.append(Paragraph(f"<b>{trait_name}</b>", styles["Normal"]))
    list_items: list[Paragraph] = [
        _create_occurrence_item(k, occurrences[k], styles)
        for k in sorted(occurrences.keys(), key=str)
    ]

    story.append(
        ListFlowable(
            list_items,
            bulletType="bullet",
            start="circle",
        )
    )
    story.append(Spacer(1, 6))


def _add_traits_to_map(
    trait_map: dict[str, dict[str, str | list[str]]],
    key: str,
    traits: dict[str, str | list[str]],
) -> None:
    """Helper to add traits from one chapter to the map."""
    for trait_name, trait_val in traits.items():
        trait_map[trait_name][key] = trait_val


def _collect_trait_map(
    entity: EntityRecord,
) -> dict[str, dict[str, str | list[str]]]:
    """Helper to group traits by name across chapters.

    Returns:
        A mapping of trait name to (location -> value).
    """
    trait_map: defaultdict[str, dict[str, str | list[str]]] = defaultdict(dict)

    if not entity.appearances:
        return dict(trait_map)

    first_val = next(iter(entity.appearances.values()))
    if isinstance(first_val, ChapterAppearances):
        for book_title, book_app in entity.appearances.items():
            if not isinstance(book_app, ChapterAppearances):
                continue
            for chap_num, appearance in book_app.chapters.items():
                key = f"{book_title}, Chapter {chap_num}"
                _add_traits_to_map(trait_map, key, appearance.traits)
    else:
        for k, appearance_val in entity.appearances.items():
            if not isinstance(appearance_val, SingleAppearance):
                continue
            _add_traits_to_map(trait_map, k, appearance_val.appearance.traits)

    return dict(trait_map)


def _process_entity(
    story: list[Flowable],
    entity: EntityRecord,
    styles: StyleSheet1,
) -> None:
    """Process a single entity and add it to the story."""
    story.append(Paragraph(entity.name, styles["Heading2"]))

    if entity.summary:
        story.append(Paragraph(entity.summary, styles["Normal"]))
    elif entity.appearances:
        story.append(
            Paragraph("<i>Error during summarization.</i>", styles["Normal"])
        )
    else:
        story.append(
            Paragraph("<i>No appearances recorded.</i>", styles["Normal"])
        )
    story.append(Spacer(1, 12))

    if not entity.appearances:
        story.append(Spacer(1, 12))
        return

    story.append(Paragraph("<b>Traits:</b>", styles["Normal"]))
    story.append(Spacer(1, 6))

    trait_map = _collect_trait_map(entity)
    for trait_name in sorted(trait_map.keys()):
        _add_trait_section(story, trait_name, trait_map[trait_name], styles)

    story.append(Spacer(1, 12))


def _process_category(
    story: list[Flowable], category: str, data: Binder, styles: StyleSheet1
) -> None:
    """Helper to process all entities in a category."""
    cat = data.categories[category]
    story.append(Paragraph(cat.name, styles["Heading1"]))
    story.append(Spacer(1, 12))

    for entity_name in sorted(cat.entities.keys()):
        _process_entity(story, cat.entities[entity_name], styles)


def generate_pdf_report(data: Binder, output_path: Path) -> None:
    """Generate a PDF report from the Binder model."""
    doc = SimpleDocTemplate(str(output_path), pagesize=LETTER)
    styles = get_document_styles()
    story: list[Flowable] = [
        Paragraph("LoreBinders Story Bible", styles["Title"])
    ]

    story.append(Spacer(1, 12))

    if getattr(data, "stage_failures", None):
        story.append(
            Paragraph("<b>Note: Partial Results</b>", styles["Heading3"])
        )
        for stage, stats in data.stage_failures.items():
            msg = (
                f"{stage.capitalize()} stage: "
                f"{stats['failed_count']}/{stats['total_count']} tasks failed."
            )
            story.append(Paragraph(f"<i>{msg}</i>", styles["Normal"]))
        story.append(Spacer(1, 12))

    for cat_name in sorted(data.categories.keys()):
        _process_category(story, cat_name, data, styles)

    doc.build(story)
