"""Dummy storage provider for testing purposes."""

from pathlib import Path

import lorebinders.models as models


class TestStorageProvider:
    """In-memory storage provider for testing purposes."""

    def set_workspace(self, author: str, title: str) -> None:
        """Set the workspace directories.

        Args:
            author: The name of the author.
            title: The title of the book.
        """
        self.author = author
        self.title = title
        self.extractions: dict[int, dict[str, list[str]]] = {}
        self.profiles: dict[tuple[int, str, str], models.EntityProfile] = {}
        self.summaries: dict[tuple[str, str], str] = {}
        self.book_text = ""

    @property
    def path(self) -> Path:
        """The base path of the workspace.

        Returns:
            The Path to the workspace directory.
        """
        return Path("/tmp/lorebinders_test")

    def extraction_exists(self, chapter_num: int) -> bool:
        """Check if extraction exists.

        Args:
            chapter_num: The chapter number.

        Returns:
            True if extraction data exists for the chapter.
        """
        return chapter_num in self.extractions

    def save_extraction(
        self,
        chapter_num: int,
        data: dict[str, list[str]],
    ) -> None:
        """Save extraction data.

        Args:
            chapter_num: The chapter number.
            data: The extraction data.
        """
        self.extractions[chapter_num] = data

    def load_extraction(self, chapter_num: int) -> dict[str, list[str]]:
        """Load extraction data.

        Args:
            chapter_num (int): The chapter number of the extraction.

        Returns:
            The extraction data dictionary.
        """
        return self.extractions[chapter_num]

    def profile_exists(
        self, chapter_num: int, category: str, name: str
    ) -> bool:
        """Check if profile exists.

        Args:
            chapter_num: The chapter number.
            category: The entity category.
            name: The entity name.

        Returns:
            True if the profile exists.
        """
        return (chapter_num, category, name) in self.profiles

    def filter_cached_profiles(
        self, chapter_num: int, category: str, names: list[str]
    ) -> tuple[list[str], list[str]]:
        """Split names into those that are cached and those that are not.

        Args:
            chapter_num: The chapter number.
            category: The entity category.
            names: List of entity names to check.

        Returns:
            A tuple of (cached_names, missing_names).
        """
        cached, missing = [], []
        for n in names:
            if self.profile_exists(chapter_num, category, n):
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
        key = (chapter_num, profile.category, profile.name)
        self.profiles[key] = profile

    def load_profile(
        self, chapter_num: int, category: str, name: str
    ) -> models.EntityProfile:
        """Load profile data.

        Args:
            chapter_num (int): The chapter number of the profile.
            category (str): The category of the profile.
            name (str): The name of the profile.

        Returns:
            The loaded entity profile.
        """
        key = (chapter_num, category, name)
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
