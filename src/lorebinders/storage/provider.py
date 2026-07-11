from pathlib import Path
from typing import Protocol

from lorebinders import models


class StorageProvider(Protocol):
    """Protocol for LoreBinders storage backends."""

    def set_workspace(self, author: str, title: str) -> None:
        """Set the workspace context.

        Args:
            author: The name of the book author.
            title: The title of the book.
        """
        ...

    @property
    def path(self) -> Path:
        """The base path of the workspace.

        Returns:
            The Path to the workspace directory.
        """
        ...

    def extraction_exists(self, chapter_num: int, book_title: str = "") -> bool:
        """Check if extraction exists.

        Args:
            chapter_num: The chapter number.
            book_title: The book title.

        Returns:
            True if it exists.
        """
        ...

    def save_extraction(
        self,
        chapter_num: int,
        data: dict[str, list[models.ExtractedEntity]],
        book_title: str = "",
    ) -> None:
        """Save extraction data.

        Args:
            chapter_num: The chapter number.
            data: The extraction data dictionary.
            book_title: The book title.
        """
        ...

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
        ...

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
            True if it exists.
        """
        ...

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
        ...

    def save_profile(
        self,
        chapter_num: int,
        profile: models.EntityProfile,
    ) -> None:
        """Save profile data.

        Args:
            chapter_num: The chapter number.
            profile: The entity profile model.
        """
        ...

    def load_profile(
        self, chapter_num: int, category: str, name: str, book_title: str = ""
    ) -> models.EntityProfile:
        """Load profile data.

        Args:
            chapter_num: The chapter number.
            category: The entity category.
            name: The entity name.
            book_title: The book title.

        Returns:
            The entity profile model.
        """
        ...

    def summary_exists(self, category: str, name: str) -> bool:
        """Check if summary exists.

        Args:
            category: The entity category.
            name: The entity name.

        Returns:
            True if it exists.
        """
        ...

    def save_summary(self, category: str, name: str, summary: str) -> None:
        """Save summary data.

        Args:
            category: The entity category.
            name: The entity name.
            summary: The generated summary text.
        """
        ...

    def load_summary(self, category: str, name: str) -> str:
        """Load summary data.

        Args:
            category: The entity category.
            name: The entity name.

        Returns:
            The summary text.
        """
        ...

    def save_book(self, title: str, text: str) -> None:
        """Save the book text.

        Args:
            title: The book title.
            text: The full text content.
        """
        ...
