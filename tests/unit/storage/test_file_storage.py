"""Tests for FilesystemStorage covering previously uncovered branches."""

import json
from pathlib import Path

import pytest

from lorebinders import models
from lorebinders.storage.providers.file import FilesystemStorage


@pytest.fixture
def storage(tmp_path: Path) -> FilesystemStorage:
    s = FilesystemStorage()
    s.set_workspace.__func__ if False else None
    s._path = tmp_path
    s.extractions_dir = tmp_path / "extractions"
    s.profiles_dir = tmp_path / "profiles"
    s.summaries_dir = tmp_path / "summaries"
    return s


def test_path_raises_when_workspace_not_set() -> None:
    s = FilesystemStorage()
    with pytest.raises((RuntimeError, AttributeError)):
        _ = s.path


def test_save_extraction_writes_json(
    storage: FilesystemStorage, tmp_path: Path
) -> None:
    data = {
        "Characters": [
            models.ExtractedEntity(
                name="Alice", presence_type="literal_entity"
            ),
            models.ExtractedEntity(name="Bob", presence_type="literal_entity"),
        ]
    }
    storage.save_extraction(1, data, book_title="Book 1")
    path = tmp_path / "extractions" / "Book_1_ch1_extraction.json"
    assert path.exists()
    assert json.loads(path.read_text()) == {
        "Characters": [
            {"name": "Alice", "presence_type": "literal_entity"},
            {"name": "Bob", "presence_type": "literal_entity"},
        ]
    }


def test_save_profile_writes_json(
    storage: FilesystemStorage, tmp_path: Path
) -> None:
    profile = models.EntityProfile(
        chapter_number=1,
        book_title="Book 1",
        category="Characters",
        name="Alice",
        traits={"Role": "Hero"},
    )
    storage.save_profile(1, profile)
    path = tmp_path / "profiles" / "Book_1_ch1_Characters_Alice.json"
    assert path.exists()
    loaded = models.EntityProfile.model_validate_json(path.read_text())
    assert loaded == profile


def test_save_summary_writes_json(
    storage: FilesystemStorage, tmp_path: Path
) -> None:
    storage.save_summary("Characters", "Alice", "A brave hero.")
    path = tmp_path / "summaries" / "Characters_Alice_summary.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["summary"] == "A brave hero."


def test_load_extraction_reads_json_as_models(
    storage: FilesystemStorage, tmp_path: Path
) -> None:
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
        ]
    }

    storage.save_extraction(1, data, book_title="Book 1")
    loaded = storage.load_extraction(1, book_title="Book 1")

    assert isinstance(loaded, dict)
    assert "Characters" in loaded

    characters = loaded["Characters"]
    assert len(characters) == 3
    assert all(
        isinstance(character, models.ExtractedEntity)
        for character in characters
    )

    assert [(c.name, c.presence_type) for c in characters] == [
        ("Alice", "literal_entity"),
        ("Bob", "mentioned_entity"),
        ("Carol", "allusive_figure"),
    ]
