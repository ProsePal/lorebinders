import json
import logging
from pathlib import Path

import lorebinders.storage.workspace as workspace
from lorebinders import models

logger = logging.getLogger(__name__)


def _get_extraction_path(extractions_dir: Path, chapter_num: int) -> Path:
    """Helper to construct the path for an extraction file.

    Args:
        extractions_dir: The base directory for extractions.
        chapter_num: The chapter number.

    Returns:
        The Path to the extraction file.
    """
    return extractions_dir / f"ch{chapter_num}_extraction.json"


def _get_profile_path(
    profiles_dir: Path, chapter_num: int, category: str, entity_name: str
) -> Path:
    """Helper to construct the path for a profile file.

    Args:
        profiles_dir: The base directory for profiles.
        chapter_num: The chapter number.
        category: The entity category.
        entity_name: The entity name.

    Returns:
        The Path to the profile file.
    """
    safe_name = "".join(c if c.isalnum() else "_" for c in entity_name)
    safe_category = "".join(c if c.isalnum() else "_" for c in category)
    return profiles_dir / f"ch{chapter_num}_{safe_category}_{safe_name}.json"


def _get_summary_path(
    summaries_dir: Path, category: str, entity_name: str
) -> Path:
    """Helper to construct the path for a summary file.

    Args:
        summaries_dir: The base directory for summaries.
        category: The entity category.
        entity_name: The entity name.

    Returns:
        The Path to the summary file.
    """
    safe_category = "".join(c if c.isalnum() else "_" for c in category)
    safe_name = "".join(c if c.isalnum() else "_" for c in entity_name)
    return summaries_dir / f"{safe_category}_{safe_name}_summary.json"


class FilesystemStorage:
    """Standard filesystem-based storage implementation."""

    def __init__(self) -> None:
        """Initialize the storage instance."""
        self._path: Path | None = None
        self.extractions_dir: Path | None = None
        self.profiles_dir: Path | None = None
        self.summaries_dir: Path | None = None

    def _ensure_initialized(self) -> None:
        """Ensure that set_workspace has been called.

        Raises:
            RuntimeError: If the workspace is not set.
        """
        if self._path is None:
            raise RuntimeError("Workspace not set. Call set_workspace first.")

    def set_workspace(self, author: str, title: str) -> None:
        """Set the workspace directories.

        Args:
            author: The name of the author.
            title: The title of the book.
        """
        self._path = workspace.ensure_workspace(author, title)
        self.extractions_dir = self._path / "extractions"
        self.profiles_dir = self._path / "profiles"
        self.summaries_dir = self._path / "summaries"

    @property
    def path(self) -> Path:
        """The base path of the workspace.

        Returns:
            The Path to the workspace directory.

        Raises:
            RuntimeError: If the workspace is not set.
        """
        self._ensure_initialized()
        path = self._path
        if not isinstance(path, Path):
            raise RuntimeError("Internal error: _path is not a Path")
        return path

    def extraction_exists(self, chapter_num: int) -> bool:
        """Check if extraction exists.

        Args:
            chapter_num: The chapter number.

        Returns:
            True if extraction data exists for the chapter.
        """
        self._ensure_initialized()
        if self.extractions_dir is None:
            return False
        return _get_extraction_path(self.extractions_dir, chapter_num).exists()

    def save_extraction(
        self,
        chapter_num: int,
        data: dict[str, list[str]],
    ) -> None:
        """Save extraction data.

        Args:
            chapter_num: The chapter number.
            data: The extraction data.

        Raises:
            RuntimeError: If extractions_dir is not set.
        """
        self._ensure_initialized()
        if self.extractions_dir is None:
            raise RuntimeError("extractions_dir is not set")
        path = _get_extraction_path(self.extractions_dir, chapter_num)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved extraction for chapter {chapter_num}")

    def load_extraction(self, chapter_num: int) -> dict[str, list[str]]:
        """Load extraction data.

        Args:
            chapter_num: The chapter number of the extraction.

        Returns:
            The extraction data dictionary.

        Raises:
            RuntimeError: If extractions_dir is not set.
            TypeError: If the loaded data is not a dictionary.
        """
        self._ensure_initialized()
        if self.extractions_dir is None:
            raise RuntimeError("extractions_dir is not set")
        path = _get_extraction_path(self.extractions_dir, chapter_num)
        with path.open(encoding="utf-8") as f:
            logger.debug(f"Loaded extraction for chapter {chapter_num}")
            data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict, got {type(data)}")
            return data

    def profile_exists(
        self, chapter_num: int, category: str, name: str
    ) -> bool:
        """Check if profile exists.

        Args:
            chapter_num: The chapter number of the profile.
            category: The category of the profile.
            name: The name of the profile.

        Returns:
            bool: True if the profile exists, False otherwise.
        """
        self._ensure_initialized()
        if self.profiles_dir is None:
            return False
        return _get_profile_path(
            self.profiles_dir, chapter_num, category, name
        ).exists()

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
        profile: "models.EntityProfile",
    ) -> None:
        """Save profile data.

        Args:
            chapter_num: The chapter number of the profile.
            profile: The profile data.

        Raises:
            RuntimeError: If profiles_dir is not set.
        """
        self._ensure_initialized()
        if self.profiles_dir is None:
            raise RuntimeError("profiles_dir is not set")
        path = _get_profile_path(
            self.profiles_dir, chapter_num, profile.category, profile.name
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w", encoding="utf-8") as f:
            f.write(profile.model_dump_json(indent=2))
        logger.debug(f"Saved profile: {profile.category}/{profile.name}")

    def load_profile(
        self, chapter_num: int, category: str, name: str
    ) -> "models.EntityProfile":
        """Load profile data.

        Args:
            chapter_num: The chapter number of the profile.
            category: The category of the profile.
            name: The name of the profile.

        Returns:
            The loaded entity profile.

        Raises:
            RuntimeError: If profiles_dir is not set.
        """
        self._ensure_initialized()
        if self.profiles_dir is None:
            raise RuntimeError("profiles_dir is not set")
        path = _get_profile_path(self.profiles_dir, chapter_num, category, name)
        with path.open(encoding="utf-8") as f:
            content = f.read()
            return models.EntityProfile.model_validate_json(content)

    def summary_exists(self, category: str, name: str) -> bool:
        """Check if summary exists.

        Args:
            category: The category of the summary.
            name: The name of the summary.

        Returns:
            bool: True if the summary exists, False otherwise.
        """
        self._ensure_initialized()
        if self.summaries_dir is None:
            return False
        return _get_summary_path(self.summaries_dir, category, name).exists()

    def save_summary(self, category: str, name: str, summary: str) -> None:
        """Save summary data.

        Args:
            category: The category of the summary.
            name: The name of the summary.
            summary: The summary data.

        Raises:
            RuntimeError: If summaries_dir is not set.
        """
        self._ensure_initialized()
        if self.summaries_dir is None:
            raise RuntimeError("summaries_dir is not set")
        path = _get_summary_path(self.summaries_dir, category, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w", encoding="utf-8") as f:
            json.dump({"entity_name": name, "summary": summary}, f, indent=2)
        logger.debug(f"Saved summary: {category}/{name}")

    def load_summary(self, category: str, name: str) -> str:
        """Load summary data.

        Args:
            category: The category of the summary.
            name: The name of the summary.

        Returns:
            The summary text.

        Raises:
            RuntimeError: If summaries_dir is not set.
            TypeError: If the summary data is not a dictionary or
                summary is not a string.
        """
        self._ensure_initialized()
        if self.summaries_dir is None:
            raise RuntimeError("summaries_dir is not set")
        path = _get_summary_path(self.summaries_dir, category, name)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError(f"Expected dict, got {type(data)}")
            summary = data.get("summary")
            if not isinstance(summary, str):
                raise TypeError(f"Expected str summary, got {type(summary)}")
            return summary

    def save_book(self, title: str, text: str) -> None:
        """Save the book text.

        Args:
            title: The book title.
            text: The full text content.

        Raises:
            RuntimeError: If workspace is not set.
        """
        self._ensure_initialized()
        path = self._path
        if not isinstance(path, Path):
            raise RuntimeError("Internal error: _path is not a Path")
        safe_title = "".join(c if c.isalnum() else "_" for c in title)
        book_path = path / f"{safe_title}.txt"
        book_path.parent.mkdir(parents=True, exist_ok=True)
        with book_path.open(mode="w", encoding="utf-8") as f:
            f.write(text)
        logger.debug(f"Saved book: {title}")
