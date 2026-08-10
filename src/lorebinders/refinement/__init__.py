"""Refinement module for LoreBinders - data cleaning and deduplication."""

import logging
from collections.abc import Callable

from lorebinders.models import AgentDeps, Binder, ObservationEvent
from lorebinders.refinement.alias_resolution import AliasAgent, resolve_aliases
from lorebinders.refinement.allusions import resolve_allusions
from lorebinders.refinement.cleaning import clean_binder
from lorebinders.refinement.deduplication import resolve_binder

logger = logging.getLogger(__name__)


def _apply_rules(binder: Binder, narrator_name: str | None) -> Binder:
    """Run the deterministic cleaning and deduplication stages.

    Args:
        binder: The Binder model from extraction.
        narrator_name: Optional name of the narrator to replace placeholders.

    Returns:
        The cleaned, rule-deduplicated Binder model.
    """
    logger.info("Starting cleaning phase")
    cleaned_binder = clean_binder(binder, narrator_name)

    logger.info("Starting resolution phase")
    return resolve_binder(cleaned_binder)


def refine_binder(binder: Binder, narrator_name: str | None = None) -> Binder:
    """Execute the rule-based refinement pipeline on a Binder model.

    Flow: Clean -> Resolve -> Resolve Allusions.

    Args:
        binder: The Binder model from extraction.
        narrator_name: Optional name of the narrator to replace placeholders.

    Returns:
        The cleaned, deduplicated Binder model with allusions resolved.
    """
    logger.info("Resolving allusions")
    return resolve_allusions(_apply_rules(binder, narrator_name))


async def refine_binder_async(
    binder: Binder,
    narrator_name: str | None = None,
    alias_agent: AliasAgent | None = None,
    deps: AgentDeps | None = None,
    on_observe: Callable[[ObservationEvent], None] | None = None,
) -> Binder:
    """Execute the refinement pipeline including LLM alias resolution.

    Flow: Clean -> Resolve -> Resolve Allusions -> Resolve Aliases.

    Allusions are resolved before aliases because allusion resolution matches
    each "Invoked by" attribution against the entity names produced by the
    rule-based pass. Merging aliases first removes the names those
    attributions refer to, which would drop the alluded traits.

    The alias pass is skipped when no agent or dependencies are supplied, or
    when ``LOREBINDERS_ALIAS_RESOLUTION_ENABLED`` is false, which leaves the
    result identical to :func:`refine_binder`.

    Args:
        binder: The Binder model from extraction.
        narrator_name: Optional name of the narrator to replace placeholders.
        alias_agent: Optional alias resolution agent.
        deps: Optional agent dependencies, required to run the alias pass.
        on_observe: Optional observation callback.

    Returns:
        The cleaned, deduplicated Binder model with allusions resolved.
    """
    logger.info("Resolving allusions")
    resolved = resolve_allusions(_apply_rules(binder, narrator_name))

    if (
        alias_agent is None
        or deps is None
        or not deps.settings.alias_resolution_enabled
    ):
        logger.info("Skipping alias resolution phase")
        return resolved

    logger.info("Starting alias resolution phase")
    return await resolve_aliases(resolved, alias_agent, deps, on_observe)


__all__ = ["refine_binder", "refine_binder_async"]
