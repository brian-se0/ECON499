# Round 009C Targeted Re-Verification Prompt

You are GPT 5.5 Pro acting as the audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

This is a targeted re-verification round for the remaining `B7-CODE-001` issue from Round 009B.

## Attachment

Use the uploaded archive:

- `econ499_round009c_b7_code_001_followup_context_20260428.zip`

## Binding Project Standards

Treat `AGENTS.md` as binding. In particular:

- No fallback paths.
- No backwards compatibility or legacy branches.
- Hard cutover and fail-fast behavior.
- Hedging/report model comparisons must use clean, comparable forecast coverage.
- Stage 09 must not publish stale or incomparable report artifacts.
- Tests are mandatory for changes affecting hedging evaluation, report artifacts, or reproducibility.

## Prior Verification Result

Round 009B found the main Stage 08 coverage fix present, but kept `B7-CODE-001` open because `require_hedging_summary_matches_results(...)` only checked `model_name` and `n_trades`. A stale `hedging_summary.parquet` with matching model set and counts but incorrect aggregate metrics could still be published by Stage 09.

## Codex Follow-Up Claims To Check

1. `require_hedging_summary_matches_results(...)` now recomputes the official summary from `hedging_results` using `summarize_hedging_results(...)`.
2. The validator compares the full required published summary columns, not only `n_trades`:
   - `mean_abs_revaluation_error`
   - `mean_squared_revaluation_error`
   - `mean_abs_hedged_pnl`
   - `mean_squared_hedged_pnl`
   - `hedged_pnl_variance`
   - `n_trades`
3. Stage 09, which already calls this validator before report tables and figures are built, now rejects stale same-coverage hedging summaries before `ranked_hedging_summary` or hedging figures can be published.
4. The added tests cover the exact Round 009B remaining gap.
5. The implementation still preserves the original B7 fix: unequal model/date forecast coverage fails before Stage 08 writes hedging artifacts, and stale unequal hedging results fail before Stage 09 report publication.

## Required Task

Do not perform a broad full-project closure audit in this targeted round. Verify only whether `B7-CODE-001` is now fully fixed after the Round 009C follow-up.

If the fix is complete, say so and specify that the next step should be another fresh full-project closure audit with refreshed code context before any terminal closure statement.

If the fix is incomplete, identify the remaining concrete defect and the exact required Codex action.

## Required Output Format

```text
ROUND_009C_VERIFICATION_DECISION: all_fixed | still_open | needs_more_context

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
