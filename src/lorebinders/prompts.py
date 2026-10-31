"""Validated, snapshotted operator prompt overrides with bundled defaults."""

from collections.abc import Callable, Mapping
from pathlib import Path

from lorebinders.agent.factory import load_prompt_from_assets

PROMPT_STAGES = ("extraction", "analysis", "summarization", "alias_resolution")
MAX_PROMPT_BYTES = 16_384


def build_prompt_loader(
    prompt_files: Mapping[str, Path] | None = None,
) -> Callable[[str], str]:
    """Load all prompts before inference and return the AgentDeps loader.

    Operator overrides use stage names; the returned callable accepts the
    bundled filenames requested by agent factories. Contents are snapshotted,
    so file changes during a run cannot change its prompts. Hash the returned
    UTF-8 text to record the effective prompt identity for reproducibility.
    Each stage is limited to 16,384 UTF-8 bytes. This input-size budget does
    not guarantee that the full request fits a selected model's context.

    Raises:
        ValueError: If a stage is unknown or content is empty or oversized.
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
        if stage in overrides:
            with overrides[stage].open("rb") as source:
                data = source.read(MAX_PROMPT_BYTES + 1)
        else:
            data = load_prompt_from_assets(filename).encode("utf-8")
        if len(data) > MAX_PROMPT_BYTES:
            raise ValueError(
                f"Prompt for {stage} exceeds {MAX_PROMPT_BYTES} UTF-8 bytes"
            )
        content = data.decode("utf-8")
        if not content.strip():
            raise ValueError(f"Prompt for {stage} is empty")
        prompts[filename] = content

    def load_prompt(filename: str) -> str:
        if filename not in prompts:
            raise ValueError(f"Unknown prompt filename: {filename}")
        return prompts[filename]

    return load_prompt
