from collections.abc import Generator
from pathlib import Path

import pytest

from lorebinders import models
from lorebinders.settings import Settings
from lorebinders.storage import FilesystemStorage, StorageProvider


@pytest.fixture(params=["file", "db"])
def namespaced_storage(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[StorageProvider, None, None]:
    monkeypatch.setattr(
        "lorebinders.storage.workspace.get_settings",
        lambda: Settings(workspace_base_path=tmp_path),
    )
    if request.param == "db":
        pytest.importorskip("sqlalchemy")
        from lorebinders.storage.providers.db import DBStorage

        db = DBStorage("sqlite:///:memory:")
        yield db
        db.engine.dispose()
    else:
        yield FilesystemStorage()


def test_namespace_isolates_every_cached_stage(
    namespaced_storage: StorageProvider,
) -> None:
    storage = namespaced_storage
    storage.set_workspace(
        "Author", "Series", user_id="user_1", cache_namespace="revision-1"
    )
    original_path = storage.path
    profile = models.EntityProfile(
        name="Alice",
        category="Characters",
        chapter_number=1,
        book_title="Book",
        traits={"Role": "Hero"},
    )
    storage.save_extraction(1, {}, "Book")
    storage.save_profile(1, profile)
    storage.save_summary("Characters", "Alice", "Original summary")
    storage.set_workspace(
        "Author", "Series", user_id="user_1", cache_namespace="revision-2"
    )
    assert storage.path != original_path
    assert "user_1" in storage.path.parts
    assert not storage.extraction_exists(1, "Book")
    assert not storage.profile_exists(1, "Characters", "Alice", "Book")
    assert not storage.summary_exists("Characters", "Alice")
    storage.set_workspace(
        "Author", "Series", user_id="user_1", cache_namespace="revision-1"
    )
    assert storage.path == original_path
    assert storage.extraction_exists(1, "Book")
    assert storage.profile_exists(1, "Characters", "Alice", "Book")
    assert storage.load_summary("Characters", "Alice") == "Original summary"
