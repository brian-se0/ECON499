ROUND_009C_VERIFICATION_DECISION: all_fixed

B7-CODE-001:
- fixed
- Rationale:
  The Round 009B remaining gap is closed. `src/ivsurf/hedging/validation.py::require_hedging_summary_matches_results(...)` now recomputes the official summary from `hedging_results` via `summarize_hedging_results(...)`, selects the required published summary schema, sorts deterministically by `model_name`, and compares every required published column:
  `mean_abs_revaluation_error`, `mean_squared_revaluation_error`, `mean_abs_hedged_pnl`, `mean_squared_hedged_pnl`, `hedged_pnl_variance`, and `n_trades`.

  Stage 09 already calls `require_hedging_results_match_forecast_coverage(hedging_results, forecast_frame)` and then `require_hedging_summary_matches_results(hedging_summary, hedging_results)` immediately after loading artifacts and before building `ranked_hedging_summary` or hedging figures. With the updated validator, stale same-coverage `hedging_summary.parquet` artifacts are rejected before publication.

  The original Round 009 B7 coverage fix is still preserved. Stage 08 loads the combined forecast universe through `load_forecast_frame(workflow_paths.forecast_dir, grid)` before creating the hedging output directories or writing per-model/combined hedging artifacts, and then partitions only that validated combined frame by model. Stage 08 also validates combined hedging results against the clean forecast model/date universe before writing `hedging_results.parquet`, then validates the recomputed summary before writing `hedging_summary.parquet`.

  I did not perform a broad full-project closure audit in this targeted round.
- Remaining issues, if any:
  None.
- Required Codex actions, if any:
  None.

TEST_ASSESSMENT:
- Adequate
- Rationale:
  The added unit test `test_hedging_summary_validation_rejects_stale_metric_values` covers the exact Round 009B gap by mutating a required summary metric while preserving `model_name` and `n_trades`, then asserting the validator rejects it.

  The added integration test `test_stage09_rejects_stale_hedging_summary_metrics_with_equal_coverage` covers the report-publication path by creating equal-coverage forecasts and equal-coverage hedging results, writing a stale same-coverage summary, and asserting Stage 09 fails before `ranked_hedging_summary.csv` is written.

  The earlier B7 tests remain present for unequal model forecast coverage before Stage 08 outputs and for stale unequal hedging results before Stage 09 report publication. The archived targeted evidence reports ruff passing and 24 targeted tests passing across the hedging/report contract files.

DIRECTLY_RELATED_REGRESSIONS:
- None.

NEXT_REQUIRED_STEP:
- Prepare a refreshed full-project code context and request another fresh full-project closure audit before any terminal closure statement.
