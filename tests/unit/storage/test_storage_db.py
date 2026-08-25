"""Unit tests for the DB-backed storage provider."""

from collections.abc import Generator
from pathlib import Path
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
    data = {
        "Characters": [
            models.ExtractedEntity(
                name="Alice", presence_type="literal_entity"
            ),
            models.ExtractedEntity(
                name="Bob", presence_type="mentioned_entity"
            ),
            models.ExtractedEntity(
                name="Carol", presence_type="allusive_figure"
            ),
        ],
        "Locations": [
            models.ExtractedEntity(name="Paris", presence_type="literal_entity")
        ],
    }

    assert not storage.extraction_exists(chapter_num, book_title="Book 1")

    storage.save_extraction(chapter_num, data, book_title="Book 1")
    assert storage.extraction_exists(chapter_num, book_title="Book 1")

    loaded_data = storage.load_extraction(chapter_num, book_title="Book 1")

    characters = loaded_data["Characters"]
    assert all(isinstance(c, models.ExtractedEntity) for c in characters)
    assert [(c.name, c.presence_type) for c in characters] == [
        ("Alice", "literal_entity"),
        ("Bob", "mentioned_entity"),
        ("Carol", "allusive_figure"),
    ]
    locations = loaded_data["Locations"]
    assert all(isinstance(loc, models.ExtractedEntity) for loc in locations)
    assert [(loc.name, loc.presence_type) for loc in locations] == [
        ("Paris", "literal_entity"),
    ]


def test_save_extraction_updates_existing(storage: DBStorage) -> None:
    """save_extraction overwrites when record already exists."""
    storage.save_extraction(
        1,
        {
            "Characters": [
                models.ExtractedEntity(
                    name="Alice", presence_type="literal_entity"
                )
            ]
        },
        book_title="Book 1",
    )
    updated = {
        "Characters": [
            models.ExtractedEntity(name="Bob", presence_type="literal_entity")
        ]
    }
    storage.save_extraction(1, updated, book_title="Book 1")
    assert storage.load_extraction(1, book_title="Book 1") == updated


def test_load_extraction_raises_when_missing(storage: DBStorage) -> None:
    """load_extraction raises FileNotFoundError for absent chapter."""
    with pytest.raises(FileNotFoundError):
        storage.load_extraction(999, book_title="Book 1")


def test_profile_lifecycle(storage: DBStorage) -> None:
    """Test saving and loading entity profiles."""
    chapter_num = 1
    category = "Characters"
    name = "Alice"
    profile = models.EntityProfile(
        chapter_number=chapter_num,
        book_title="Book 1",
        category=category,
        name=name,
        traits={"Age": "25", "Role": "Hacker"},
    )

    assert not storage.profile_exists(
        chapter_num, category, name, book_title="Book 1"
    )

    storage.save_profile(chapter_num, profile)
    assert storage.profile_exists(
        chapter_num, category, name, book_title="Book 1"
    )

    loaded_profile = storage.load_profile(
        chapter_num, category, name, book_title="Book 1"
    )
    assert loaded_profile == profile


def test_save_profile_updates_existing(storage: DBStorage) -> None:
    """save_profile updates the record when it already exists."""
    profile_v1 = models.EntityProfile(
        chapter_number=1,
        book_title="Book 1",
        category="Characters",
        name="Alice",
        traits={"Role": "Hero"},
    )
    storage.save_profile(1, profile_v1)

    profile_v2 = models.EntityProfile(
        chapter_number=1,
        book_title="Book 1",
        category="Characters",
        name="Alice",
        traits={"Role": "Villain"},
    )
    storage.save_profile(1, profile_v2)

    loaded = storage.load_profile(1, "Characters", "Alice", book_title="Book 1")
    assert loaded.traits["Role"] == "Villain"


def test_load_profile_raises_when_missing(storage: DBStorage) -> None:
    """load_profile raises FileNotFoundError for absent profile."""
    with pytest.raises(FileNotFoundError):
        storage.load_profile(1, "Characters", "Ghost", book_title="Book 1")


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
                BookModel.workspace_id == storage._workspace_id,
                BookModel.title == "My Book",
            )
        ).first()
    assert row is not None
    assert row.title == "My Book"
    assert row.text == "Once upon a time..."


def test_save_book_updates_existing(storage: DBStorage) -> None:
    """save_book updates text when record already exists."""
    storage.save_book("Title", "Old text.")
    storage.save_book("Title", "New text.")

    with storage.SessionLocal() as session:
        rows = session.scalars(
            select(BookModel).where(
                BookModel.workspace_id == storage._workspace_id,
                BookModel.title == "Title",
            )
        ).all()

    assert len(rows) == 1
    assert rows[0].text == "New text."


def test_db_storage_keying(tmp_path: Path) -> None:
    with patch("lorebinders.storage.workspace.get_settings") as mock_settings:
        mock_settings.return_value.workspace_base_path = tmp_path

        s = DBStorage("sqlite:///:memory:")

        # With user_id
        s.set_workspace("test_author", "test_title", user_id="user_123")
        assert s.path == tmp_path / "user_123" / "test_author" / "test_title"
        expected_path = tmp_path / "user_123" / "test_author" / "test_title"
        assert s._workspace_id == str(expected_path)

        # Without user_id (CLI path)
        s.set_workspace("test_author", "test_title")
        assert s.path == tmp_path / "test_author" / "test_title"
        assert s._workspace_id == str(tmp_path / "test_author" / "test_title")
