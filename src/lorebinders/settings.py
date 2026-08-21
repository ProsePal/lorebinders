"""Centralized application settings using pydantic-settings."""

from functools import cache
from pathlib import Path

from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

from lorebinders.agent_settings import get_model_settings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    ``categories`` is the canonical list of categories the extraction
    agent searches for. "Allusions" is intentionally absent: it is a
    derived category, populated during sorting from entities whose
    presence_type is "allusive_figure", never extracted directly.
    ``allusion_traits`` supplies the trait questions used once entities
    have landed in that derived category during analysis.
    ``alias_resolution_enabled`` gates the LLM alias resolution pass that
    runs after rule-based deduplication; set it false to skip the extra
    model calls on cost-sensitive runs.
    """

    model_config = SettingsConfigDict(
        env_prefix="LOREBINDERS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    extraction_model: str = "openrouter:bytedance/seed-1.6-flash"
    analysis_model: str = "openrouter:deepseek/deepseek-v3.2"
    summarization_model: str = "openrouter:bytedance/seed-1.6-flash"
    alias_resolution_model: str = "openrouter:deepseek/deepseek-v3.2"

    extraction_fallback_model: str | None = None
    analysis_fallback_model: str | None = None
    summarization_fallback_model: str | None = None
    alias_resolution_fallback_model: str | None = None

    alias_resolution_enabled: bool = True

    workspace_base_path: Path = Path(__file__).parent / "work"
    db_url: str = "sqlite:///:memory:"

    categories: list[str] = ["Characters", "Locations"]
    character_traits: list[str] = [
        "Appearance",
        "Clothing",
        "Personality",
        "Mood",
        "Relationships with other characters",
    ]
    location_traits: list[str] = [
        "Key features",
        "Relative location",
        "Character Familiarity",
    ]
    allusion_traits: list[str] = [
        "Invoked by",
        "Rhetorical significance",
        "What it reveals about the invoker",
    ]

    confidence_threshold: float = 0.8
    max_concurrency: int = 10

    def model_settings_for(self, model: str) -> ModelSettings:
        """Set reasoning level for a given model."""
        model_provider = model.split(":")[0]
        return get_model_settings(model_provider)


@cache
def get_settings() -> Settings:
    """Get application settings singleton.

    Returns:
        The application settings instance.
    """
    return Settings()
