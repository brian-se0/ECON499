# Round 009B Targeted Verification Prompt

You are GPT 5.5 Pro acting as the audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

This is a targeted verification round for the only actionable finding from Round 009.

## Attachment

Use the uploaded archive:

- `econ499_round009b_b7_code_001_context_20260428.zip`

## Binding Project Standards

Treat `AGENTS.md` as binding. In particular:

- No fallback paths.
- No backwards compatibility or legacy branches.
- Hard cutover and fail-fast behavior.
- Hedging/report model comparisons must use clean, comparable forecast coverage.
- Tests are mandatory for changes affecting evaluation metrics, hedging evaluation, report artifacts, or reproducibility.

## Finding To Verify

`B7-CODE-001`

Severity: P2

Area: Hedging evaluation / clean forecast coverage / report comparability

Round 009 issue: Stage 08 could process each forecast artifact independently and produce hedging rankings even when models had unequal quote/target-date coverage. Stage 09 could then consume stale or incomparable hedging outputs.

## Codex Claims To Check

1. Stage 08 now loads the combined forecast universe through `load_forecast_frame(workflow_paths.forecast_dir, grid)` before any hedging output directories, per-model by-model files, `hedging_results.parquet`, or `hedging_summary.parquet` are written.
2. Stage 08 now partitions only the already validated combined forecast frame by model for hedging.
3. Stage 08 validates hedging results against the clean forecast model/date universe before writing combined hedging artifacts.
4. Stage 09 now rejects stale or incomparable hedging artifacts before publishing report tables.
5. The new tests would have failed against the Round 009 bug and now pass.
6. The implementation does not intersect dates, silently drop unmatched forecasts, pad missing hedges, or rank models with unequal `n_trades`.

## Required Task

Do not perform a broad new full-project closure audit in this targeted round. Verify only whether `B7-CODE-001` is fully fixed.

If the fix is complete, say so and specify that the next step should be another fresh full-project closure audit with refreshed code context before any terminal closure statement.

If the fix is incomplete, identify the remaining concrete defect and the exact required Codex action.

## Required Output Format

```text
ROUND_009B_VERIFICATION_DECISION: all_fixed | still_open | needs_more_context

B7-CODE-001:
- fixed | still-open | needs-more-context
- Rationale:
- Remaining issues, if any:
- Required Codex actions, if any:

TEST_ASSESSMENT:
- Adequate | inadequate | needs-more-context
- Rationale:

DIRECTLY_RELATED_REGRESSIONS:
- None, or list concrete regressions.

NEXT_REQUIRED_STEP:
- If all fixed: "Prepare a refreshed full-project code context and request another fresh full-project closure audit before any terminal closure statement."
- Otherwise: exact implementation/evidence checklist.
```
