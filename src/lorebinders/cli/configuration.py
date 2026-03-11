from pathlib import Path

from lorebinders.models import NarratorConfig, RunConfiguration


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


def build_run_configuration(
    book_path: Path,
    author_name: str,
    book_title: str,
    narrator_name: str | None,
    is_1st_person: bool,
    traits: list[str] | None,
    categories: list[str] | None,
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

    return RunConfiguration(
        book_path=book_path,
        author_name=author_name,
        book_title=book_title,
        narrator_config=narrator_config,
        custom_traits=custom_traits,
        custom_categories=custom_categories,
    )
