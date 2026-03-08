from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from lorebinders.types import EntityTraits

if TYPE_CHECKING:
    from lorebinders.settings import Settings


@dataclass
class AgentDeps:
    """Dependencies injected into agents."""

    settings: "Settings"
    prompt_loader: Callable[[str], str]


class NarratorConfig(BaseModel):
    """Configuration for narrator detection and handling."""

    is_1st_person: bool = False
    name: str | None = None


class RunConfiguration(BaseModel):
    """Configuration for a complete execution run."""

    book_path: Path
    author_name: str
    book_title: str
    narrator_config: NarratorConfig
    custom_traits: dict[str, list[str]] = Field(default_factory=dict)
    custom_categories: list[str] = Field(default_factory=list)


class Chapter(BaseModel):
    """Represents a single chapter from the book."""

    number: int
    title: str
    content: str
    profiles: list["EntityProfile"] = Field(default_factory=list)


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


class EntityRecord(BaseModel):
    """Complete record for an entity across the book."""

    name: str
    category: str
    appearances: dict[int, EntityAppearance] = Field(default_factory=dict)
    summary: str | None = None


class CategoryRecord(BaseModel):
    """Record for a category containing multiple entities."""

    name: str
    entities: dict[str, EntityRecord] = Field(default_factory=dict)


class Binder(BaseModel):
    """The complete Story Bible state."""

    categories: dict[str, CategoryRecord] = Field(default_factory=dict)

    def get_entity(self, category: str, name: str) -> EntityRecord | None:
        """Helper to safely retrieve an entity record.

        Args:
            category: The category of the entity.
            name: The name of the entity.

        Returns:
            The entity record if found, None otherwise.
        """
        cat = self.categories.get(category)
        if not cat:
            return None
        return cat.entities.get(name)

    def add_appearance(
        self,
        category: str,
        name: str,
        chapter: int,
        traits: EntityTraits,
    ) -> None:
        """Add an entity appearance to the binder."""
        if category not in self.categories:
            self.categories[category] = CategoryRecord(name=category)

        cat = self.categories[category]
        if name not in cat.entities:
            cat.entities[name] = EntityRecord(name=name, category=category)

        ent = cat.entities[name]
        ent.appearances[chapter] = EntityAppearance(traits=traits)


class ExtractionConfig(BaseModel):
    """Configuration for the entity extraction agent."""

    target_categories: list[str]
    description: str | None = None
    narrator: NarratorConfig | None = None


class CategoryEntities(BaseModel):
    """Entities extracted for a single category."""

    category: str = Field(description="Category name (e.g. 'Characters')")
    entities: list[str] = Field(
        default_factory=list,
        description="List of entity names found in this category",
    )


class ExtractionResult(BaseModel):
    """Result of entity extraction."""

    results: list[CategoryEntities] = Field(
        default_factory=list,
        description="List of categories with their extracted entity names",
    )

    def to_dict(self) -> dict[str, list[str]]:
        """Convert results list to category ->entities dictionary.

        Returns:
            A dictionary mapping category names to lists of entity names.
        """
        return {item.category: item.entities for item in self.results}


class AnalysisConfig(BaseModel):
    """Configuration for the entity analysis agent."""

    target_entity: str
    category: str
    traits: list[str]


class TraitValue(BaseModel):
    """A single analyzed trait for an entity."""

    trait: str
    value: str
    evidence: str


class AnalysisResult(BaseModel):
    """Complete analysis result for an entity."""

    entity_name: str
    category: str
    traits: list[TraitValue]


class SummarizerConfig(BaseModel):
    """Configuration for the entity summarization agent."""

    entity_name: str
    category: str
    context_data: str


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


class ObservationType(str, Enum):
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
    metadata: dict[str, Any] = Field(default_factory=dict)
