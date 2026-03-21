from pathlib import Path
from typing import Literal

from lorebinders.models import BookInput, NarratorConfig, RunConfiguration


def _parse_trait(trait_str: str) -> tuple[str, str]:
    """Parse a trait string into category and trait name.

    Returns:
        A tuple of (category, trait_name).
    """
    if ":" in trait_str:
        cat, val = trait_str.split(":", 1)
        return cat.strip(), val.strip()
    return "Characters", trait_str.strip()


def _add_trait(traits_map: dict[str, list[str]], cat: str, val: str) -> None:
    """Add a trait to the category map."""
    if cat not in traits_map:
        traits_map[cat] = []
    traits_map[cat].append(val)


def _parse_book(book_str: str) -> BookInput:
    """Parse a book string into a BookInput model.

    Returns:
        A BookInput instance with path and title.
    """
    if ":" in book_str:
        path_str, title = book_str.split(":", 1)
        return BookInput(path=Path(path_str.strip()), title=title.strip())
    path = Path(book_str.strip())
    return BookInput(path=path, title=path.stem)


def build_run_configuration(
    books: list[str],
    series_title: str,
    author_name: str,
    narrator_name: str | None,
    is_1st_person: bool,
    traits: list[str] | None,
    categories: list[str] | None,
    tracking: Literal["nested", "flat"] = "nested",
) -> RunConfiguration:
    """Build a valid RunConfiguration from raw CLI arguments.

    Returns:
        A configured RunConfiguration instance.
    """
    narrator_config = NarratorConfig(
        is_1st_person=is_1st_person,
        name=narrator_name,
    )

    custom_categories = categories or []
    custom_traits: dict[str, list[str]] = {}

    if traits:
        for t in traits:
            cat, val = _parse_trait(t)
            _add_trait(custom_traits, cat, val)

    parsed_books = [_parse_book(b) for b in books]

    return RunConfiguration(
        series_title=series_title,
        books=parsed_books,
        author_name=author_name,
        narrator_config=narrator_config,
        appearance_tracking=tracking,
        custom_traits=custom_traits,
        custom_categories=custom_categories,
    )
