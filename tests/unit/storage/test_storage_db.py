"""Unit tests for the DB-backed storage provider."""

from collections.abc import Generator
from unittest.mock import patch

import pytest
from sqlalchemy import select

from lorebinders import models
from lorebinders.storage.providers.db import BookModel, DBStorage


@pytest.fixture
def storage() -> Generator[DBStorage, None, None]:
    """Provide a fresh in-memory database storage for each test."""
    s = DBStorage("sqlite:///:memory:")
    s.set_workspace("test_author", "test_title")
    yield s
    s.engine.dispose()


def test_init_uses_settings_db_url() -> None:
    """DBStorage without explicit url falls back to settings.db_url."""
    with patch(
        "lorebinders.storage.providers.db.get_settings"
    ) as mock_settings:
        mock_settings.return_value.db_url = "sqlite:///:memory:"
        s = DBStorage()
        assert s.engine is not None
        s.engine.dispose()


def test_get_session_provides_session() -> None:
    """_get_session provides a session and closes it on exit."""
    s = DBStorage("sqlite:///:memory:")
    session = s._get_session()
    assert session is not None
    session.close()
    s.engine.dispose()


def test_path_raises_when_workspace_not_set() -> None:
    """path property raises RuntimeError if workspace not configured."""
    s = DBStorage("sqlite:///:memory:")
    with pytest.raises(RuntimeError, match="Workspace not set"):
        _ = s.path
    s.engine.dispose()


def test_extraction_lifecycle(storage: DBStorage) -> None:
    """Test saving and loading extraction data."""
    chapter_num = 1
    data = {"Characters": ["Alice", "Bob"], "Locations": ["Paris"]}

    assert not storage.extraction_exists(chapter_num)

    storage.save_extraction(chapter_num, data)
    assert storage.extraction_exists(chapter_num)

    loaded_data = storage.load_extraction(chapter_num)
    assert loaded_data == data


def test_save_extraction_updates_existing(storage: DBStorage) -> None:
    """save_extraction overwrites when record already exists."""
    storage.save_extraction(1, {"Characters": ["Alice"]})
    updated = {"Characters": ["Bob"]}
    storage.save_extraction(1, updated)
    assert storage.load_extraction(1) == updated


def test_load_extraction_raises_when_missing(storage: DBStorage) -> None:
    """load_extraction raises FileNotFoundError for absent chapter."""
    with pytest.raises(FileNotFoundError):
        storage.load_extraction(999)


def test_profile_lifecycle(storage: DBStorage) -> None:
    """Test saving and loading entity profiles."""
    chapter_num = 1
    category = "Characters"
    name = "Alice"
    profile = models.EntityProfile(
        chapter_number=chapter_num,
        category=category,
        name=name,
        traits={"Age": "25", "Role": "Hacker"},
    )

    assert not storage.profile_exists(chapter_num, category, name)

    storage.save_profile(chapter_num, profile)
    assert storage.profile_exists(chapter_num, category, name)

    loaded_profile = storage.load_profile(chapter_num, category, name)
    assert loaded_profile == profile


def test_save_profile_updates_existing(storage: DBStorage) -> None:
    """save_profile updates the record when it already exists."""
    profile_v1 = models.EntityProfile(
        chapter_number=1,
        category="Characters",
        name="Alice",
        traits={"Role": "Hero"},
    )
    storage.save_profile(1, profile_v1)

    profile_v2 = models.EntityProfile(
        chapter_number=1,
        category="Characters",
        name="Alice",
        traits={"Role": "Villain"},
    )
    storage.save_profile(1, profile_v2)

    loaded = storage.load_profile(1, "Characters", "Alice")
    assert loaded.traits["Role"] == "Villain"


def test_load_profile_raises_when_missing(storage: DBStorage) -> None:
    """load_profile raises FileNotFoundError for absent profile."""
    with pytest.raises(FileNotFoundError):
        storage.load_profile(1, "Characters", "Ghost")


def test_summary_exists_when_absent(storage: DBStorage) -> None:
    """summary_exists returns False before any save."""
    assert not storage.summary_exists("Characters", "Alice")


def test_summary_lifecycle(storage: DBStorage) -> None:
    """Test saving and loading summaries."""
    category = "Characters"
    name = "Alice"
    storage.save_summary(category, name, "First summary.")
    assert storage.summary_exists(category, name)
    assert storage.load_summary(category, name) == "First summary."


def test_save_summary_updates_existing(storage: DBStorage) -> None:
    """save_summary overwrites when record already exists."""
    storage.save_summary("Characters", "Alice", "Old.")
    storage.save_summary("Characters", "Alice", "New.")
    assert storage.load_summary("Characters", "Alice") == "New."


def test_load_summary_raises_when_missing(storage: DBStorage) -> None:
    """load_summary raises FileNotFoundError for absent summary."""
    with pytest.raises(FileNotFoundError):
        storage.load_summary("Characters", "Ghost")


def test_save_book_creates_record(storage: DBStorage) -> None:
    """save_book inserts a new BookModel row."""
    storage.save_book("My Book", "Once upon a time...")

    with storage.SessionLocal() as session:
        row = session.scalars(
            select(BookModel).where(
                BookModel.workspace_id == storage._workspace_id
            )
        ).first()
    assert row is not None
    assert row.title == "My Book"
    assert row.text == "Once upon a time..."


def test_save_book_updates_existing(storage: DBStorage) -> None:
    """save_book updates title and text when record already exists."""
    storage.save_book("Old Title", "Old text.")
    storage.save_book("New Title", "New text.")

    with storage.SessionLocal() as session:
        rows = session.scalars(
            select(BookModel).where(
                BookModel.workspace_id == storage._workspace_id
            )
        ).all()

    assert len(rows) == 1
    assert rows[0].title == "New Title"
    assert rows[0].text == "New text."
