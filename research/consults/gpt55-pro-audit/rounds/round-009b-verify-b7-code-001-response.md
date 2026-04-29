ROUND_009B_VERIFICATION_DECISION: still_open

B7-CODE-001:
- still-open
- Rationale:
  The main Stage 08 coverage repair is present. `scripts/08_run_hedging_eval.py` now loads the combined forecast universe with `load_forecast_frame(workflow_paths.forecast_dir, grid)` before creating `hedging_dir`, `by_model`, per-model hedging files, `hedging_results.parquet`, or `hedging_summary.parquet`. It then partitions only that validated combined frame by `model_name`. Because `load_forecast_frame` calls `assert_equal_forecast_model_coverage` and `require_forecast_surface_grid`, the original unequal quote/target coverage path is blocked before Stage 08 writes hedging outputs.

  Stage 08 also validates combined hedging results with `require_hedging_results_match_forecast_coverage(results_frame, forecast_frame)` before writing `hedging_results.parquet`, and validates the summary with `require_hedging_summary_matches_results(summary_frame, results_frame)` before writing `hedging_summary.parquet`.

  Stage 09 now reloads the current clean forecast frame and calls both hedging validators before building report tables, so it rejects hedging results whose model/date coverage no longer matches the forecast universe.

  However, the Stage 09 stale-summary protection is incomplete. `require_hedging_summary_matches_results` claims to ensure the summary is an exact model-level aggregate of the hedging results, but the implementation only checks `model_name` coverage and `n_trades`. It does not recompute or compare `mean_abs_revaluation_error`, `mean_squared_revaluation_error`, `mean_abs_hedged_pnl`, `mean_squared_hedged_pnl`, or `hedged_pnl_variance`. Therefore a stale `hedging_summary.parquet` with the same models and same `n_trades`, but old or incorrect metric values, can still pass Stage 09 validation and be published in `ranked_hedging_summary` and `hedging_ranking.svg`.
- Remaining issues, if any:
  Stage 09 can still consume a stale or corrupted hedging summary when coverage and trade counts match but summary metric values do not match `hedging_results.parquet`.
- Required Codex actions, if any:
  Patch `require_hedging_summary_matches_results` so it recomputes the expected summary from `hedging_results` using the official `summarize_hedging_results` aggregation contract, sorts deterministically, and fails if any published summary column differs from the recomputed aggregate. The comparison should cover all required summary columns, not only `n_trades`.

TEST_ASSESSMENT:
- inadequate
- Rationale:
  The new tests adequately cover the original unequal forecast-date coverage failure:
  `test_stage08_rejects_unequal_model_forecast_coverage_before_outputs` would have failed against the Round 009 bug, and the companion passing test checks equal coverage. `test_stage09_rejects_hedging_results_with_unequal_forecast_coverage` also covers stale/incomparable hedging results with missing model/date rows.

  The tests do not cover stale hedging summaries with identical model/date coverage and identical `n_trades` but incorrect aggregate metric values. Because Stage 09 publishes rankings from `hedging_summary`, this missing regression leaves B7 only partially fixed.

DIRECTLY_RELATED_REGRESSIONS:
- None.

NEXT_REQUIRED_STEP:
- Update `src/ivsurf/hedging/validation.py` so `require_hedging_summary_matches_results` verifies the full recomputed summary, not just model set and `n_trades`.
- Add a unit test proving `require_hedging_summary_matches_results` rejects a summary with correct `model_name` and `n_trades` but stale metric values.
- Add or extend the Stage 09 integration test so a stale same-coverage `hedging_summary.parquet` fails before `ranked_hedging_summary` or report figures are written.
- Re-run targeted ruff/pytest for `src/ivsurf/hedging/validation.py`, `scripts/09_make_report_artifacts.py`, `tests/integration/test_stage08_hedging_coverage.py`, the updated Stage 09/report contract tests, and `tests/unit/test_hedging.py`.
