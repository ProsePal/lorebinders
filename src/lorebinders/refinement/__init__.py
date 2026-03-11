"""Refinement module for LoreBinders - data cleaning and deduplication."""

import logging

from lorebinders.models import Binder
from lorebinders.refinement.cleaning import clean_binder
from lorebinders.refinement.deduplication import resolve_binder

logger = logging.getLogger(__name__)


def refine_binder(binder: Binder, narrator_name: str | None = None) -> Binder:
    """Execute the refinement pipeline on a Binder model.

    Flow: Clean -> Resolve.

    Args:
        binder: The Binder model from extraction.
        narrator_name: Optional name of the narrator to replace placeholders.

    Returns:
        The cleaned and deduplicated Binder model.
    """
    logger.info("Starting cleaning phase")
    cleaned_binder = clean_binder(binder, narrator_name)

    logger.info("Starting resolution phase")
    return resolve_binder(cleaned_binder)


__all__ = ["refine_binder"]
