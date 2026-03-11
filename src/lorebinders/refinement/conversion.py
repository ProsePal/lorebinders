from pathlib import Path

import ebook2text
from dotenv import load_dotenv

from lorebinders.models import Book, Chapter

load_dotenv()


def convert_to_text(source: Path) -> str:
    """Convert ebook file to raw text string.

    Args:
        source: Path to the ebook file.

    Returns:
        str: The extracted raw text content of the book.

    Raises:
        FileNotFoundError: If the source file does not exist.
        TypeError: If the converted text is not a string.
    """
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")
    metadata = {"title": source.stem, "author": "Unknown"}
    text = ebook2text.convert_file(source, metadata, save_file=False)
    if not isinstance(text, str):
        raise TypeError(f"Expected str from ebook2text, got {type(text)}")
    return text


def _extract_chapters(text: str) -> list[Chapter]:
    """Extract Chapter models from text parts.

    Args:
        text: Text separated by a chapter delimiter.

    Returns:
        List of Chapter models.
    """
    chapters: list[Chapter] = []
    for part in text.split("***"):
        content = part.strip()
        if not content:
            continue
        number = len(chapters) + 1
        chapters.append(
            Chapter(
                number=number,
                title=f"Chapter {number}",
                content=content,
            )
        )
    return chapters


def ingest(text: str, title: str) -> Book:
    """Build a Book model from raw text.

    Args:
        text: The raw text content of the book.
        title: The title of the book.

    Returns:
        Book: A structured Book model containing chapters.
    """
    chapters = _extract_chapters(text)
    return Book(title=title, author="Unknown", chapters=chapters)
