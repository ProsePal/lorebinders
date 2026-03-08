# Implementation Plan: Observability and Monitoring

This plan introduces a standardized observation system to LoreBinders2, providing rich, event-driven monitoring across all pipeline stages.

## Approach
- **Unified Event Model**: Transition from basic `ProgressUpdate` to a more comprehensive `ObservationEvent` that includes timestamps, event types (start, progress, complete, error), and metadata.
- **Instrument Missing Stages**: Add instrumentation to the summarization, ingestion, and reporting phases which are currently "silent".
- **Decoupled Observation**: Provide a clean callback interface that UI/CLI components can implement to visualize progress, log metrics, or handle errors.

## Steps

1. **Enhance Models** (10 min)
   - Update `src/lorebinders/models.py` to include `ObservationEvent` and `EventType`.
   - Keep `ProgressUpdate` for backward compatibility or evolve it.

2. **Instrument Summarization** (15 min)
   - Modify `src/lorebinders/agent/summarization.py` to support progress reporting.
   - Update `summarize_binder` signature.

3. **Instrument Workflow** (20 min)
   - Update `src/lorebinders/workflow.py` to emit events at stage transitions.
   - Add reporting for ingestion and PDF generation.

4. **Update CLI** (15 min)
   - Update `src/lorebinders/cli/__cli__.py` to handle the new event model if necessary, or ensure it remains compatible with the basic fields.

5. **Testing** (20 min)
   - Add unit tests for the observation events.
   - Verify that all stages emit the expected events.

## Timeline
| Phase | Duration |
|-------|----------|
| Models | 10 min |
| Summarization | 15 min |
| Workflow | 20 min |
| CLI / Integration | 15 min |
| Testing | 20 min |
| **Total** | **~1.5 hours** |

## Rollback Plan
- Revert changes to `workflow.py` and `models.py`.
- The `progress` callback remains optional, so removing the instrumentation won't break core logic.

## Security Checklist
- [x] No sensitive data (API keys, etc.) should be emitted in `ObservationEvent` metadata.
- [x] Ensure error messages in events are sanitized.
