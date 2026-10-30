"""Text normalization utilities shared across refinement modules."""

import re
from functools import lru_cache

from lorebinders.refinement.patterns import LOCATION_SUFFIX_PATTERN
from lorebinders.types import EntityTraits, TraitValue

MAX_ENTITY_NAME_LENGTH = 200

TITLES: frozenset[str] = frozenset(
    {
        "admiral",
        "airman",
        "ambassador",
        "aunt",
        "baron",
        "baroness",
        "brother",
        "cadet",
        "cap",
        "captain",
        "col",
        "colonel",
        "commander",
        "commodore",
        "corporal",
        "count",
        "countess",
        "cousin",
        "dad",
        "daddy",
        "doc",
        "doctor",
        "dr",
        "duchess",
        "duke",
        "earl",
        "ensign",
        "father",
        "gen",
        "general",
        "granddad",
        "grandfather",
        "grandma",
        "grandmom",
        "grandmother",
        "grandpop",
        "great aunt",
        "great grandfather",
        "great grandmother",
        "great uncle",
        "great-aunt",
        "great-grandfather",
        "great-grandmother",
        "great-uncle",
        "king",
        "lady",
        "leftenant",
        "lieutenant",
        "lord",
        "lt",
        "ma",
        "ma'am",
        "madam",
        "major",
        "marquis",
        "miss",
        "missus",
        "mister",
        "mjr",
        "mom",
        "mommy",
        "mother",
        "mr",
        "mrs",
        "ms",
        "nurse",
        "pa",
        "pfc",
        "pop",
        "prince",
        "princess",
        "private",
        "queen",
        "sarge",
        "seaman",
        "sergeant",
        "sir",
        "sister",
        "the",
        "uncle",
    }
)

_SINGULAR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(.*)(lves)$"), r"\1lf"),
    (re.compile(r"(?i)(.*)(eaves)$"), r"\1eaf"),
    (re.compile(r"(?i)(.*)(oaves)$"), r"\1oaf"),
    (re.compile(r"(?i)(.*)(ives)$"), r"\1ife"),
    (re.compile(r"(?i)(.*)(ves)$"), r"\1f"),
    (re.compile(r"(?i)(.*)(ies)$"), r"\1y"),
    (re.compile(r"(?i)(.*)(i)$"), r"\1us"),
    (re.compile(r"(?i)(.*)(a)$"), r"\1um"),
    (re.compile(r"(?i)(.*)(oes)$"), r"\1o"),
    (re.compile(r"(?i)(.*)(sses)$"), r"\1ss"),
    (re.compile(r"(?i)(.*)(ses)$"), r"\1s"),
    (re.compile(r"(?i)(.*)(xes)$"), r"\1x"),
    (re.compile(r"(?i)(.*)(zes)$"), r"\1ze"),
    (re.compile(r"(?i)(.*)(ches)$"), r"\1ch"),
    (re.compile(r"(?i)(.*)(shes)$"), r"\1sh"),
    (re.compile(r"(?i)(.*)(s)$"), r"\1"),
]


@lru_cache(maxsize=1024)
def remove_titles(name: str) -> str:
    """Remove titles from a name.

    Args:
        name: The name to remove titles from.

    Returns:
        The name with titles removed.
    """
    if not name:
        return name
    name_split = name.split(" ")
    first_word = name_split[0].lower().rstrip(".")
    if first_word in TITLES and name.lower() not in TITLES:
        return " ".join(name_split[1:])
    return name


def standardize_location(name: str) -> str:
    """Remove suffixes like (Interior) or - Night from location names.

    Args:
        name: The raw location name.

    Returns:
        The standardized location name.
    """
    return LOCATION_SUFFIX_PATTERN.sub("", name).strip()


def clean_entity_name(name: str, category: str) -> str:
    """Clean an entity name based on its category.

    Args:
        name: The raw entity name.
        category: The entity category (e.g. 'Characters', 'Locations').

    Returns:
        The cleaned entity name.

    Raises:
        ValueError: If the name exceeds MAX_ENTITY_NAME_LENGTH.
    """
    if len(name) > MAX_ENTITY_NAME_LENGTH:
        raise ValueError(f"Entity name exceeds max length: {len(name)}")

    cleaned = remove_titles(name.strip())

    if category.lower() == "locations":
        cleaned = standardize_location(cleaned)

    return cleaned


@lru_cache(maxsize=1024)
def to_singular(plural: str) -> str:
    """Convert a plural word to its singular form.

    Args:
        plural: The plural word to convert.

    Returns:
        The singular form of the word.
    """
    if not plural:
        return ""

    for pattern, replacement in _SINGULAR_PATTERNS:
        singular, n = pattern.subn(replacement, plural)
        if n > 0:
            return singular

    return plural


def _merge_trait_values(v1: TraitValue, v2: TraitValue) -> TraitValue:
    """Safely merge two trait values (strings or lists of strings).

    Args:
        v1: The first trait value.
        v2: The second trait value.

    Returns:
        The merged trait value.
    """
    if isinstance(v1, list):
        if isinstance(v2, list):
            return sorted(list(set(v1 + v2)))
        return sorted(list(set(v1 + [v2])))
    if isinstance(v2, list):
        return sorted(list(set([v1] + v2)))
    return v1 if v1 == v2 else sorted([v1, v2])


def merge_values(v1: EntityTraits, v2: EntityTraits) -> EntityTraits:
    """Merge two EntityTraits dictionaries when keys collide.

    Args:
        v1: The first trait dictionary.
        v2: The second trait dictionary.

    Returns:
        The merged trait dictionary.
    """
    for k, v in v2.items():
        v1[k] = _merge_trait_values(v1[k], v) if k in v1 else v
    return v1
