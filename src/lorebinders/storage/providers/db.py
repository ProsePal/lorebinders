"""SQLAlchemy storage backend for LoreBinders."""

from pathlib import Path
from typing import TypeVar

from sqlalchemy import JSON, String, create_engine, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.sql.selectable import Select

import lorebinders.storage.workspace as workspace
from lorebinders import models
from lorebinders.settings import get_settings
from lorebinders.types import EntityTraits


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


BaseT = TypeVar("BaseT", bound=Base)


class BookModel(Base):
    """SQLAlchemy model for books."""

    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        String(1024), index=True, unique=True
    )
    title: Mapped[str] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(String)


class ExtractionModel(Base):
    """SQLAlchemy model for extractions."""

    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(1024), index=True)
    chapter_num: Mapped[int] = mapped_column(index=True)
    data: Mapped[dict[str, list[str]]] = mapped_column(JSON)


class ProfileModel(Base):
    """SQLAlchemy model for EntityProfiles."""

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(1024), index=True)
    chapter_num: Mapped[int] = mapped_column(index=True)
    category: Mapped[str] = mapped_column(String(255), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    data: Mapped[EntityTraits] = mapped_column(JSON)


class SummaryModel(Base):
    """SQLAlchemy model for summaries."""

    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(1024), index=True)
    category: Mapped[str] = mapped_column(String(255), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    summary: Mapped[str] = mapped_column(String)


class DBStorage:
    """SQLAlchemy-backed storage provider."""

    def __init__(self, db_url: str | None = None) -> None:
        """Initialize the database storage.

        Args:
            db_url: Optional SQLAlchemy connection string.
        """
        if not db_url:
            db_url = get_settings().db_url
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self._path: Path | None = None
        self._workspace_id: str | None = None

    def _get_session(self) -> Session:
        """Create and return a new database session.

        Returns:
            A new SQLAlchemy Session instance.
        """
        return self.SessionLocal()

    def _require_workspace_id(self) -> str:
        """Return workspace_id or raise if set_workspace was not called.

        Returns:
            The current workspace identifier.

        Raises:
            RuntimeError: If the workspace is not set.
        """
        if self._workspace_id is None:
            raise RuntimeError("Workspace not set. Call set_workspace() first.")
        return self._workspace_id

    def set_workspace(self, author: str, title: str) -> None:
        """Set the workspace context.

        Args:
            author: The name of the author.
            title: The title of the book.
        """
        path = workspace.ensure_workspace(author, title)
        self._path = path
        self._workspace_id = str(path)

    @property
    def path(self) -> Path:
        """The base path of the workspace.

        Returns:
            The Path to the workspace directory.

        Raises:
            RuntimeError: If the workspace is not set.
        """
        if not self._path:
            raise RuntimeError("Workspace not set")
        return self._path

    def _get_model(
        self,
        session: Session,
        stmt: Select[tuple[BaseT]],
        model_class: type[BaseT],
    ) -> BaseT | None:
        if result := session.scalars(stmt).first():
            if not isinstance(result, model_class):
                raise TypeError(
                    f"Expected {model_class.__name__}, got {type(result)}"
                )
            return result
        return None

    def _find_extraction(
        self, session: Session, chapter_num: int
    ) -> ExtractionModel | None:
        """Query for an extraction by chapter number.

        Args:
            session: SQLAlchemy session.
            chapter_num: The chapter number.

        Returns:
            The extraction model if found, None otherwise.
        """
        stmt = select(ExtractionModel).where(
            ExtractionModel.workspace_id == self._require_workspace_id(),
            ExtractionModel.chapter_num == chapter_num,
        )
        return self._get_model(session, stmt, ExtractionModel)

    def extraction_exists(self, chapter_num: int) -> bool:
        """Check if extraction exists.

        Args:
            chapter_num: The chapter number.

        Returns:
            True if extraction data exists for the chapter.
        """
        with self._get_session() as session:
            return self._find_extraction(session, chapter_num) is not None

    def save_extraction(
        self, chapter_num: int, data: dict[str, list[str]]
    ) -> None:
        """Save extraction data.

        Args:
            chapter_num: The chapter number.
            data: Extraction results.
        """
        with self._get_session() as session:
            model = self._find_extraction(session, chapter_num)
            self._upsert_extraction(session, model, chapter_num, data)
            session.commit()

    def _upsert_extraction(
        self,
        session: Session,
        model: ExtractionModel | None,
        chapter_num: int,
        data: dict[str, list[str]],
    ) -> None:
        """Insert or update an extraction record."""
        if model:
            model.data = data
            return
        new_model = ExtractionModel(
            workspace_id=self._require_workspace_id(),
            chapter_num=chapter_num,
            data=data,
        )
        session.add(new_model)

    def load_extraction(self, chapter_num: int) -> dict[str, list[str]]:
        """Load extraction data.

        Args:
            chapter_num: The chapter number.

        Returns:
            The extraction data dictionary.

        Raises:
            FileNotFoundError: If the extraction data is missing.
        """
        with self._get_session() as session:
            if model := self._find_extraction(session, chapter_num):
                return {
                    str(k): [str(v) for v in val]
                    for k, val in model.data.items()
                }
            else:
                raise FileNotFoundError(
                    f"Extraction for chapter {chapter_num} not found"
                )

    def _find_profile(
        self,
        session: Session,
        chapter_num: int,
        category: str,
        name: str,
    ) -> ProfileModel | None:
        """Query for a profile by chapter, category, and name.

        Args:
            session: SQLAlchemy session.
            chapter_num: The chapter number.
            category: The entity category.
            name: The entity name.

        Returns:
            The profile model if found, None otherwise.
        """
        stmt = select(ProfileModel).where(
            ProfileModel.workspace_id == self._require_workspace_id(),
            ProfileModel.chapter_num == chapter_num,
            ProfileModel.category == category,
            ProfileModel.name == name,
        )
        return self._get_model(session, stmt, ProfileModel)

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
        with self._get_session() as session:
            return (
                self._find_profile(session, chapter_num, category, name)
                is not None
            )

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
        if not names:
            return [], []

        with self._get_session() as session:
            stmt = select(ProfileModel.name).where(
                ProfileModel.workspace_id == self._require_workspace_id(),
                ProfileModel.chapter_num == chapter_num,
                ProfileModel.category == category,
                ProfileModel.name.in_(names),
            )
            cached_names = set(session.scalars(stmt).all())
            cached = [n for n in names if n in cached_names]
            missing = [n for n in names if n not in cached_names]
            return cached, missing

    def save_profile(
        self, chapter_num: int, profile: "models.EntityProfile"
    ) -> None:
        """Save profile data.

        Args:
            chapter_num: The chapter number.
            profile: The entity profile model.
        """
        with self._get_session() as session:
            model = self._find_profile(
                session, chapter_num, profile.category, profile.name
            )
            self._upsert_profile(session, model, chapter_num, profile)
            session.commit()

    def _upsert_profile(
        self,
        session: Session,
        model: ProfileModel | None,
        chapter_num: int,
        profile: "models.EntityProfile",
    ) -> None:
        """Insert or update a profile record.

        Args:
            session: SQLAlchemy session.
            model: Existing profile model or None.
            chapter_num: The chapter number.
            profile: The entity profile model.

        Raises:
            TypeError: If model_dump does not return a dict.
        """
        data = profile.model_dump(mode="json")
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict from model_dump, got {type(data)}")
        if model:
            model.data = data
            return
        new_model = ProfileModel(
            workspace_id=self._require_workspace_id(),
            chapter_num=chapter_num,
            category=profile.category,
            name=profile.name,
            data=data,
        )
        session.add(new_model)

    def load_profile(
        self, chapter_num: int, category: str, name: str
    ) -> "models.EntityProfile":
        """Load profile data.

        Args:
            chapter_num: The chapter number.
            category: The entity category.
            name: The entity name.

        Returns:
            The loaded entity profile.

        Raises:
            FileNotFoundError: If the profile is missing.
        """
        with self._get_session() as session:
            if model := self._find_profile(
                session, chapter_num, category, name
            ):
                return models.EntityProfile.model_validate(model.data)
            else:
                raise FileNotFoundError(
                    f"Profile '{name}' ({category}) for chapter"
                    f" {chapter_num} not found"
                )

    def _find_summary(
        self, session: Session, category: str, name: str
    ) -> SummaryModel | None:
        """Query for a summary by category and name.

        Args:
            session: SQLAlchemy session.
            category: The entity category.
            name: The entity name.

        Returns:
            The summary model if found, None otherwise.
        """
        stmt = select(SummaryModel).where(
            SummaryModel.workspace_id == self._require_workspace_id(),
            SummaryModel.category == category,
            SummaryModel.name == name,
        )
        return self._get_model(session, stmt, SummaryModel)

    def summary_exists(self, category: str, name: str) -> bool:
        """Check if summary exists.

        Args:
            category: The entity category.
            name: The entity name.

        Returns:
            True if the summary exists.
        """
        with self._get_session() as session:
            return self._find_summary(session, category, name) is not None

    def save_summary(self, category: str, name: str, summary: str) -> None:
        """Save summary data.

        Args:
            category: The entity category.
            name: The entity name.
            summary: The generated summary text.
        """
        with self._get_session() as session:
            model = self._find_summary(session, category, name)
            self._upsert_summary(session, model, category, name, summary)
            session.commit()

    def _upsert_summary(
        self,
        session: Session,
        model: SummaryModel | None,
        category: str,
        name: str,
        summary: str,
    ) -> None:
        """Insert or update a summary record."""
        if model:
            model.summary = summary
            return
        new_model = SummaryModel(
            workspace_id=self._require_workspace_id(),
            category=category,
            name=name,
            summary=summary,
        )
        session.add(new_model)

    def load_summary(self, category: str, name: str) -> str:
        """Load summary data.

        Args:
            category: The entity category.
            name: The entity name.

        Returns:
            The generated summary text.

        Raises:
            FileNotFoundError: If the summary is missing.
            TypeError: If the summary data is not a string.
        """
        with self._get_session() as session:
            model = self._find_summary(session, category, name)
            if not model:
                raise FileNotFoundError(
                    f"Summary for '{name}' ({category}) not found"
                )
            summary = model.summary
            if not isinstance(summary, str):
                raise TypeError(f"Expected str summary, got {type(summary)}")
            return summary

    def save_book(self, title: str, text: str) -> None:
        """Save the book text.

        Args:
            title: The book title.
            text: The book text content.
        """
        with self._get_session() as session:
            stmt = select(BookModel).where(
                BookModel.workspace_id == self._require_workspace_id()
            )
            model = self._get_model(session, stmt, BookModel)
            self._upsert_book(session, model, title, text)
            session.commit()

    def _upsert_book(
        self,
        session: Session,
        model: BookModel | None,
        title: str,
        text: str,
    ) -> None:
        """Insert or update a book record."""
        if model:
            model.title = title
            model.text = text
            return
        new_model = BookModel(
            workspace_id=self._require_workspace_id(),
            title=title,
            text=text,
        )
        session.add(new_model)
