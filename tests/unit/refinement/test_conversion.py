from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lorebinders.models import Book
from lorebinders.refinement.conversion import (
    _extract_chapters,
    convert_to_text,
    ingest,
)


@pytest.fixture
def mock_ebook2text() -> Generator[MagicMock, None, None]:
    with patch("lorebinders.refinement.conversion.ebook2text") as mock:
        yield mock


def test_extract_chapters_with_asterisks() -> None:
    """Test splitting text by '***' delimiter."""
    text = "Chapter 1\nSome text.\n***\nChapter 2\nMore text."
    chapters = _extract_chapters(text)
    assert len(chapters) == 2
    assert chapters[0].content.strip() == "Chapter 1\nSome text."
    assert chapters[1].content.strip() == "Chapter 2\nMore text."
    assert chapters[0].number == 1
    assert chapters[1].number == 2


def test_extract_chapters_fallback_single() -> None:
    """Test fallback to single chapter when no '***' delimiter is present."""
    text = "Just a short story without any delimiter."
    chapters = _extract_chapters(text)
    assert len(chapters) == 1
    assert chapters[0].content == text
    assert chapters[0].title == "Chapter 1"


def test_extract_chapters_numbering() -> None:
    """Test that chapter numbers are assigned sequentially."""
    text = "Part A\n***\nPart B\n***\nPart C"
    chapters = _extract_chapters(text)
    assert len(chapters) == 3
    assert chapters[0].number == 1
    assert chapters[1].number == 2
    assert chapters[2].number == 3


def test_empty_chapter_filter() -> None:
    """Test that empty chapters are discarded."""
    text = "Chapter 1\n***\n   \n***\nChapter 2"
    chapters = _extract_chapters(text)
    assert len(chapters) == 2
    assert "Chapter 1" in chapters[0].content
    assert "Chapter 2" in chapters[1].content


def test_convert_to_text(mock_ebook2text: MagicMock, tmp_path: Path) -> None:
    """Test that convert_to_text calls ebook2text with correct metadata."""
    source_file = tmp_path / "book.epub"
    source_file.touch()
    mock_ebook2text.convert_file.return_value = (
        "Chapter 1\nText.\n***\nChapter 2\nEnd."
    )

    result = convert_to_text(source_file)

    assert result == "Chapter 1\nText.\n***\nChapter 2\nEnd."
    expected_metadata = {"title": "book", "author": "Unknown"}
    mock_ebook2text.convert_file.assert_called_once_with(
        source_file, expected_metadata, save_file=False
    )


def test_convert_to_text_file_not_found(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised if file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        convert_to_text(Path("nonexistent.epub"))


def test_ingest_builds_book() -> None:
    """Test that ingest builds a Book from raw text and a title."""
    text = "Chapter 1\nText.\n***\nChapter 2\nEnd."
    book = ingest(text, "my_book")
    assert isinstance(book, Book)
    assert book.title == "my_book"
    assert len(book.chapters) == 2
