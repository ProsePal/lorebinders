from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from lorebinders.types import EntityTraits as EntityTraits

if TYPE_CHECKING:
    from lorebinders.agent.spend import Spend
    from lorebinders.settings import Settings


@dataclass
class AgentDeps:
    """Dependencies injected into agents."""

    settings: "Settings"
    prompt_loader: Callable[[str], str]
    spend: "Spend | None" = None


class NarratorConfig(BaseModel):
    """Configuration for narrator detection and handling."""

    is_1st_person: bool = False
    name: str | None = None


class BookInput(BaseModel):
    """Configuration for a single book input."""

    path: Path
    title: str


class RunConfiguration(BaseModel):
    """Configuration for a complete execution run."""

    series_title: str
    books: list[BookInput]
    author_name: str
    narrator_config: NarratorConfig
    appearance_tracking: Literal["nested", "flat"] = "nested"
    custom_traits: dict[str, list[str]] = Field(default_factory=dict)
    custom_categories: list[str] = Field(default_factory=list)
    user_id: str | None = None


class Chapter(BaseModel):
    """Represents a single chapter from the book."""

    number: int
    title: str
    content: str


class Book(BaseModel):
    """Represents the entire ingested book."""

    title: str
    author: str
    chapters: list[Chapter] = Field(default_factory=list)


class EntityProfile(BaseModel):
    """Structured output for an entity analysis."""

    name: str
    category: str
    chapter_number: int
    book_title: str
    traits: EntityTraits = Field(
        default_factory=dict, description="Map of trait keys to analysis values"
    )
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CategoryTarget(BaseModel):
    """Target category for batch analysis."""

    name: str
    traits: list[str] | None = None
    entities: list[str]


class EntityAppearance(BaseModel):
    """Traits for an entity in a specific chapter."""

    traits: EntityTraits = Field(default_factory=dict)


class AppearanceValue(BaseModel):
    """Base class for appearance values."""


class SingleAppearance(AppearanceValue):
    """Single appearance for flat tracking."""

    appearance: EntityAppearance


class ChapterAppearances(AppearanceValue):
    """Multiple appearances for nested tracking."""

    chapters: dict[int, EntityAppearance]


class EntityRecord(BaseModel):
    """Complete record for an entity across the book."""

    name: str
    category: str
    appearances: dict[str, AppearanceValue] = Field(default_factory=dict)
    summary: str | None = None


class CategoryRecord(BaseModel):
    """Record for a category containing multiple entities."""

    name: str
    entities: dict[str, EntityRecord] = Field(default_factory=dict)


class Binder(BaseModel):
    """The complete Story Bible state."""

    categories: dict[str, CategoryRecord] = Field(default_factory=dict)
    stage_failures: dict[str, dict[str, int]] = Field(default_factory=dict)

    def get_entity(self, category: str, name: str) -> EntityRecord | None:
        """Helper to safely retrieve an entity record.

        Args:
            category: The category of the entity.
            name: The name of the entity.

        Returns:
            The entity record if found, None otherwise.
        """
        cat = self.categories.get(category)
        return cat.entities.get(name) if cat else None

    def add_appearance(
        self,
        category: str,
        name: str,
        chapter: int,
        book_title: str,
        traits: EntityTraits,
        tracking: Literal["nested", "flat"] = "nested",
    ) -> None:
        """Add an entity appearance to the binder."""
        if category not in self.categories:
            self.categories[category] = CategoryRecord(name=category)

        cat = self.categories[category]
        if name not in cat.entities:
            cat.entities[name] = EntityRecord(name=name, category=category)

        ent = cat.entities[name]
        appearance = EntityAppearance(traits=traits)

        if tracking == "nested":
            if book_title not in ent.appearances:
                ent.appearances[book_title] = ChapterAppearances(chapters={})
            ch_app = ent.appearances[book_title]
            if isinstance(ch_app, ChapterAppearances):
                ch_app.chapters[chapter] = appearance
        else:
            key = f"{book_title}_ch{chapter}"
            ent.appearances[key] = SingleAppearance(appearance=appearance)


class ExtractedEntity(BaseModel):
    """An extracted entity with its presence type."""

    name: str
    presence_type: Literal[
        "literal_entity", "mentioned_entity", "allusive_figure"
    ]


class CategoryEntities(BaseModel):
    """Entities extracted for a single category."""

    category: str = Field(description="Category name (e.g. 'Characters')")
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="List of extracted entities found in this category",
    )


class ExtractionResult(BaseModel):
    """Result of entity extraction."""

    results: list[CategoryEntities] = Field(
        default_factory=list,
        description="List of categories with their extracted entities",
    )

    def to_dict(self) -> dict[str, list[ExtractedEntity]]:
        """Convert results list to category ->entities dictionary.

        Returns:
            A dictionary mapping category names to lists of extracted entities.
        """
        return {item.category: item.entities for item in self.results}


class AnalyzedTrait(BaseModel):
    """A single analyzed trait for an entity."""

    trait: str
    value: str
    evidence: str


class AnalysisResult(BaseModel):
    """Complete analysis result for an entity."""

    entity_name: str
    category: str
    traits: list[AnalyzedTrait]


class AliasGroup(BaseModel):
    """A set of entity names the model judged to be one entity."""

    canonical_name: str = Field(
        description="The supplied name to keep for the merged entity"
    )
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Other supplied names in the same category that refer to the "
            "entity named by canonical_name"
        ),
    )


class AliasResolution(BaseModel):
    """Result of an alias resolution pass over a single category."""

    groups: list[AliasGroup] = Field(
        default_factory=list,
        description=(
            "Alias groups found; entities that stand alone are omitted"
        ),
    )


class SummarizerResult(BaseModel):
    """Result of entity summarization."""

    entity_name: str
    summary: str


class ProgressUpdate(BaseModel):
    """A progress update during pipeline execution."""

    stage: str
    current: int
    total: int
    message: str


class ObservationType(StrEnum):
    """Types of observation events."""

    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    AGENT_RUN_STARTED = "agent_run_started"
    AGENT_RUN_COMPLETED = "agent_run_completed"
    ERROR = "error"
    METRIC = "metric"


class ObservationEvent(BaseModel):
    """A rich observation event for monitoring."""

    type: ObservationType
    stage: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )


def emit_observation(
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
