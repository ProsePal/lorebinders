"""Agent package for AI interaction logic.

Pipeline Failure Threshold and Compounding Loss Model
=====================================================

Per-Stage Threshold Mechanics
-----------------------------
Pipeline stages (extraction, analysis, summarization) independently evaluate
a failure threshold gate:

    ratio > threshold and failed_count >= min_count

The actual tolerated loss ratio per stage is:

    max(threshold, (min_count - 1) / N)

where N denotes the number of tasks scheduled in that specific stage:
- In ``extract_book``: N is the number of chapters (``len(book.chapters)``).
- In ``analyze_entities``: N is the number of chapter-level analysis tasks
  (``len(chapter_tasks)``), with one task per chapter that has surviving
  entities from extraction.
- In ``summarize_binder``: N is the total entity count across all categories
  (``len(_collect_tasks(binder))``), not the chapter count.

Small-N Floor and Degradation
-----------------------------
For small task counts (N), the min-count floor (``failure_threshold_min_count``,
default 2) dominates the configured ratio threshold (``failure_threshold``,
default 0.2):
- At N=3: up to 1 failure (33.3%) is tolerated.
- At N=2: up to 1 failure (50.0%) is tolerated.
- At N=1: up to 1 failure (100.0%) is tolerated silently.

Across stages, content losses can compound. For example, in a 3-chapter book,
1 chapter failing extraction (33% loss) leaves 2 chapters. If 1 of the 2
subsequent chapter-analysis tasks fails (50% loss), only entities from 1
chapter reach the binder.

If exactly one entity survives into summarization (N=1 entity), that entity's
summarization task can also fail (100% loss at the summarization stage)
without aborting the pipeline. Note that reaching the 100% summarization floor
requires the specific precondition of exactly one surviving entity; for a
typical manuscript yielding multiple entities (e.g. 12 entities), the
summarization floor is 1/12 (~8.3%).

When summarization fails for an entity, it does not result in total content
loss or an empty Story Bible. As implemented in ``lorebinders.reporting.pdf``,
the PDF report still renders the entity heading, its chapter appearances,
its full trait table from analysis, and an explicit error marker:
"Error during summarization.". Compounded failures therefore produce
degraded output missing summaries, rather than total loss of book content.

Ungated Loss Channels and Model Bounds
--------------------------------------
The full pipeline comprises four LLM stages, not three:
1. Extraction (``extract_book``)
2. Analysis (``analyze_entities``)
3. Refinement (``refine_binder_async``)
4. Summarization (``summarize_binder``)

Output completeness is not strictly guaranteed up to theoretical threshold
bounds (such as 1 - (1 - threshold)^k) due to known ungated loss channels:
- Analysis within-batch entity omission: In ``_analyze_batch``, only entities
  explicitly returned by the model are processed. If the model omits entities
  from its response, they vanish without raising an exception; ``failed_count``
  remains 0 and the threshold gate is not triggered.
- Spend ceiling abort channel: When a configured spend ceiling is exceeded,
  a ``SpendError`` is raised immediately across all stages (extraction,
  analysis, summarization, and refinement). In the gated stages (extraction,
  analysis, summarization), failure metrics reflecting any ordinary task
  failures accumulated prior to the abort are emitted before the
  ``SpendError`` is re-raised, but execution aborts immediately, bypassing
  the ratio-based failure threshold gate.
- Refinement stage errors: In ``refine_binder_async`` (alias resolution),
  ordinary category-level model failures are logged and discarded without a
  threshold gate or failure metric. However, execution aborts such as
  ``SpendError`` or process interruptions propagate immediately and fail
  the pipeline.
"""

from lorebinders.agent.factory import (
    build_alias_resolution_user_prompt,
    build_analysis_user_prompt,
    build_extraction_user_prompt,
    build_summarization_user_prompt,
    create_alias_resolution_agent,
    create_analysis_agent,
    create_extraction_agent,
    create_summarization_agent,
    load_prompt_from_assets,
)
from lorebinders.agent.summarization import summarize_binder

__all__ = [
    "build_alias_resolution_user_prompt",
    "build_analysis_user_prompt",
    "build_extraction_user_prompt",
    "build_summarization_user_prompt",
    "create_alias_resolution_agent",
    "create_analysis_agent",
    "create_extraction_agent",
    "create_summarization_agent",
    "load_prompt_from_assets",
    "summarize_binder",
]
