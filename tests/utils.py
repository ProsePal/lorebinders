import json
from pathlib import Path
from typing import TypeAlias

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

import lorebinders.models as models

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class TestStorageProvider:
    """In-memory storage provider for testing purposes."""

    def set_workspace(
        self, author: str, title: str, user_id: str | None = None
    ) -> None:
        """Set the workspace directories.

        Args:
            author: The name of the author.
            title: The title of the book.
            user_id: The user ID.
        """
        self.author = author
        self.title = title
        self.extractions: dict[
            tuple[int, str], dict[str, list[models.ExtractedEntity]]
        ] = {}
        self.profiles: dict[
            tuple[int, str, str, str], models.EntityProfile
        ] = {}
        self.summaries: dict[tuple[str, str], str] = {}
        self.book_text = ""

    @property
    def path(self) -> Path:
        """The base path of the workspace.

        Returns:
            The Path to the workspace directory.
        """
        return Path("/tmp/lorebinders_test")

    def extraction_exists(self, chapter_num: int, book_title: str = "") -> bool:
        """Check if extraction exists.

        Args:
            chapter_num: The chapter number.
            book_title: The book title.

        Returns:
            True if extraction data exists for the chapter.
        """
        return (chapter_num, book_title) in self.extractions

    def save_extraction(
        self,
        chapter_num: int,
        data: dict[str, list[models.ExtractedEntity]],
        book_title: str = "",
    ) -> None:
        """Save extraction data.

        Args:
            chapter_num: The chapter number.
            data: The extraction data.
            book_title: The book title.
        """
        self.extractions[(chapter_num, book_title)] = data

    def load_extraction(
        self, chapter_num: int, book_title: str = ""
    ) -> dict[str, list[models.ExtractedEntity]]:
        """Load extraction data.

        Args:
            chapter_num (int): The chapter number of the extraction.
            book_title: The book title.

        Returns:
            The extraction data dictionary.
        """
        return self.extractions[(chapter_num, book_title)]

    def profile_exists(
        self, chapter_num: int, category: str, name: str, book_title: str = ""
    ) -> bool:
        """Check if profile exists.

        Args:
            chapter_num: The chapter number.
            category: The entity category.
            name: The entity name.
            book_title: The book title.

        Returns:
            True if the profile exists.
        """
        return (chapter_num, category, name, book_title) in self.profiles

    def filter_cached_profiles(
        self,
        chapter_num: int,
        category: str,
        names: list[str],
        book_title: str = "",
    ) -> tuple[list[str], list[str]]:
        """Split names into those that are cached and those that are not.

        Args:
            chapter_num: The chapter number.
            category: The entity category.
            names: List of entity names to check.
            book_title: The book title.

        Returns:
            A tuple of (cached_names, missing_names).
        """
        cached, missing = [], []
        for n in names:
            if self.profile_exists(chapter_num, category, n, book_title):
                cached.append(n)
            else:
                missing.append(n)
        return cached, missing

    def save_profile(
        self,
        chapter_num: int,
        profile: models.EntityProfile,
    ) -> None:
        """Save profile data.

        Args:
            chapter_num (int): The chapter number of the profile.
            profile (models.EntityProfile): The profile data.
        """
        key = (
            chapter_num,
            profile.category,
            profile.name,
            profile.book_title,
        )
        self.profiles[key] = profile

    def load_profile(
        self, chapter_num: int, category: str, name: str, book_title: str = ""
    ) -> models.EntityProfile:
        """Load profile data.

        Args:
            chapter_num (int): The chapter number of the profile.
            category (str): The category of the profile.
            name (str): The name of the profile.
            book_title: The book title.

        Returns:
            The loaded entity profile.
        """
        key = (chapter_num, category, name, book_title)
        return self.profiles[key]

    def summary_exists(self, category: str, name: str) -> bool:
        """Check if summary exists.

        Args:
            category (str): The category of the summary.
            name (str): The name of the summary.

        Returns:
            bool: True if the summary exists, False otherwise.
        """
        return (category, name) in self.summaries

    def save_summary(self, category: str, name: str, summary: str) -> None:
        """Save summary data.

        Args:
            category (str): The category of the summary.
            name (str): The name of the summary.
            summary (str): The summary data.
        """
        self.summaries[(category, name)] = summary

    def load_summary(self, category: str, name: str) -> str:
        """Load summary data.

        Args:
            category (str): The category of the summary.
            name (str): The name of the summary.

        Returns:
            The summary text.

        Raises:
            FileNotFoundError: If the summary is missing.
        """
        key = (category, name)
        if key not in self.summaries:
            raise FileNotFoundError(f"Summary {name} not found")
        return self.summaries[key]

    def save_book(self, title: str, text: str) -> None:
        """Save the book text.

        Args:
            title: The book title.
            text: The full text content.
        """
        self.book_text = text


def create_mock_model(
    response_data: JsonValue,
    model_name: str | None = None,
) -> tuple[FunctionModel, list[ModelMessage]]:
    """Create a mock pydantic-ai model that returns fixed response data.

    Args:
        response_data: JSON-serializable data to return as the model response.
        model_name: Optional model name override.

    Returns:
        A tuple of (mock FunctionModel, list that captures sent messages).
    """
    captured_messages: list[ModelMessage] = []

    def _serialize(data: JsonValue) -> JsonValue:
        if isinstance(data, list):
            return [_serialize(item) for item in data]
        if isinstance(data, dict):
            return {k: _serialize(v) for k, v in data.items()}
        return data

    def mock_call(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        nonlocal captured_messages
        captured_messages.extend(messages)
        return ModelResponse(
            parts=[TextPart(content=json.dumps(_serialize(response_data)))]
        )

    return FunctionModel(mock_call, model_name=model_name), captured_messages


def get_system_prompt(messages: list[ModelMessage]) -> str:
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    return part.content
    return ""
