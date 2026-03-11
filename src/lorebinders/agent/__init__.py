"""Agent package for AI interaction logic."""

from lorebinders.agent.factory import (
    build_analysis_user_prompt,
    build_extraction_user_prompt,
    build_summarization_user_prompt,
    create_analysis_agent,
    create_extraction_agent,
    create_summarization_agent,
    load_prompt_from_assets,
)
from lorebinders.agent.summarization import summarize_binder

__all__ = [
    "build_analysis_user_prompt",
    "build_extraction_user_prompt",
    "build_summarization_user_prompt",
    "create_analysis_agent",
    "create_extraction_agent",
    "create_summarization_agent",
    "load_prompt_from_assets",
    "summarize_binder",
]
