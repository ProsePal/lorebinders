"""Validated, snapshotted operator prompt overrides with bundled defaults."""

from collections.abc import Callable, Mapping
from pathlib import Path

from lorebinders.agent.factory import load_prompt_from_assets

PROMPT_STAGES = ("extraction", "analysis", "summarization", "alias_resolution")


def build_prompt_loader(
    prompt_files: Mapping[str, Path] | None = None,
) -> Callable[[str], str]:
    """Load all prompts before inference and return the AgentDeps loader.

    Operator overrides use stage names; the returned callable accepts the
    bundled filenames requested by agent factories. Contents are snapshotted,
    so file changes during a run cannot change its prompts. Hash the returned
    UTF-8 text to record the effective prompt identity for reproducibility.

    Raises:
        ValueError: If an override stage is unknown or its content is empty.
        OSError: If a configured file cannot be read.
    """
    overrides = dict(prompt_files or {})
    unknown = overrides.keys() - set(PROMPT_STAGES)
    if unknown:
        raise ValueError(
            f"Unknown prompt stage(s): {', '.join(sorted(unknown))}"
        )

    prompts: dict[str, str] = {}
    for stage in PROMPT_STAGES:
        filename = f"{stage}.txt"
        content = (
            overrides[stage].read_text(encoding="utf-8")
            if stage in overrides
            else load_prompt_from_assets(filename)
        )
        if not content.strip():
            raise ValueError(f"Prompt for {stage} is empty")
        prompts[filename] = content

    def load_prompt(filename: str) -> str:
        if filename not in prompts:
            raise ValueError(f"Unknown prompt filename: {filename}")
        return prompts[filename]

    return load_prompt
