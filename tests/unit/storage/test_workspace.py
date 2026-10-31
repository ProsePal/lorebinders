from pathlib import Path

import pytest

from lorebinders.storage.workspace import (
    clean_workspace,
    ensure_workspace,
    sanitize_filename,
)


@pytest.fixture
def workspace_base(tmp_path: Path) -> Path:
    return tmp_path / "work_test"


def test_ensure_workspace_creates_directory(workspace_base: Path) -> None:
    author = "New Author"
    title = "My Book"

    path = ensure_workspace(author, title, base_path=workspace_base)

    assert path.exists()
    assert path.is_dir()
    assert (workspace_base / "New_Author" / "My_Book").exists()
    assert path == workspace_base / "New_Author" / "My_Book"


def test_clean_workspace(workspace_base: Path) -> None:
    author = "Author"
    title = "Title"
    path = ensure_workspace(author, title, base_path=workspace_base)

    (path / "dummy.txt").touch()
    assert (path / "dummy.txt").exists()

    clean_workspace(author, title, base_path=workspace_base)

    assert not path.exists()


def test_ensure_workspace_idempotent(workspace_base: Path) -> None:
    path1 = ensure_workspace("A", "B", base_path=workspace_base)
    path2 = ensure_workspace("A", "B", base_path=workspace_base)
    assert path1 == path2
    assert path1.exists()


def test_sanitize_filename() -> None:
    assert sanitize_filename("Normal") == "Normal"
    assert sanitize_filename("Space Valid") == "Space_Valid"
    assert sanitize_filename("Bad/Chars\\Here") == "Bad_Chars_Here"
    assert sanitize_filename("..") == "_"


def test_cleanup_removes_only_selected_namespace(workspace_base: Path) -> None:
    legacy = ensure_workspace("Author", "Series", workspace_base, "user")
    first = ensure_workspace(
        "Author", "Series", workspace_base, "user", cache_namespace="first"
    )
    second = ensure_workspace(
        "Author", "Series", workspace_base, "user", cache_namespace="second"
    )
    clean_workspace(
        "Author", "Series", workspace_base, "user", cache_namespace="first"
    )
    assert not first.exists()
    assert second.exists()
    assert legacy.exists()
    clean_workspace("Author", "Series", workspace_base, "user")
    assert not legacy.exists()
    assert second.exists()


@pytest.mark.parametrize("namespace", ["", "../bad", "bad/path"])
def test_cleanup_rejects_invalid_namespace(
    workspace_base: Path, namespace: str
) -> None:
    with pytest.raises(ValueError, match="cache_namespace"):
        clean_workspace(
            "Author", "Series", workspace_base, cache_namespace=namespace
        )
