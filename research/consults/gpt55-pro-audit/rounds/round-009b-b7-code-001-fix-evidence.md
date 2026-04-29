# Round 009B B7-CODE-001 Fix Evidence

Date: 2026-04-28

## Finding

`B7-CODE-001`: Stage 08 hedging can rank models on unequal forecast-date coverage.

Round 009 Pro decision: `ROUND_009_AUDIT_DECISION: findings_found`

Severity: P2

## Codex Fix Summary

- Added fail-fast hedging coverage validators in `src/ivsurf/hedging/validation.py`:
  - `require_hedging_results_match_forecast_coverage(...)`
  - `require_hedging_summary_matches_results(...)`
- Patched `scripts/08_run_hedging_eval.py` to load the combined forecast universe through `load_forecast_frame(workflow_paths.forecast_dir, grid)` before any hedging output directories or per-model hedging artifacts are written.
- Stage 08 now relies on the same clean cross-model forecast coverage contract used by Stage 07, then partitions the validated combined frame by model for hedging.
- Stage 08 validates combined hedging results against the clean forecast model/date universe before writing `hedging_results.parquet`, and validates the summary before writing `hedging_summary.parquet`.
- Patched `scripts/09_make_report_artifacts.py` to reject stale or incomparable hedging outputs by checking:
  - hedging result model/date coverage exactly matches the clean forecast artifacts;
  - hedging summary model set and `n_trades` exactly match the hedging results.

## Tests Added

New file: `tests/integration/test_stage08_hedging_coverage.py`

- `test_stage08_rejects_unequal_model_forecast_coverage_before_outputs`
  - Creates two valid per-model forecast parquet files that are individually grid-complete.
  - The `ridge` model intentionally lacks one quote/target pair.
  - Asserts Stage 08 raises before writing `hedging_results.parquet`, `hedging_summary.parquet`, or per-model by-model hedging outputs.
- `test_stage08_accepts_equal_model_forecast_coverage`
  - Companion passing regression with equal forecast coverage.
  - Asserts Stage 08 writes the hedging summary and each model has `n_trades == 2`.
- `test_stage09_rejects_hedging_results_with_unequal_forecast_coverage`
  - Creates valid equal-coverage forecast artifacts but stale unequal hedging outputs.
  - Asserts Stage 09 fails before publishing report tables derived from incomparable hedging summaries.

## Verification Commands

```text
uv run python -m ruff check scripts/08_run_hedging_eval.py scripts/09_make_report_artifacts.py src/ivsurf/hedging/validation.py tests/integration/test_stage08_hedging_coverage.py
```

Result:

```text
All checks passed!
```

```text
uv run python -m pytest tests/integration/test_stage08_hedging_coverage.py tests/integration/test_report_stage_contract.py tests/integration/test_stats_hedging_slice.py tests/unit/test_hedging.py
```

Result:

```text
22 passed in 1.95s
```

```text
git diff --check -- scripts/08_run_hedging_eval.py scripts/09_make_report_artifacts.py src/ivsurf/hedging/validation.py tests/integration/test_stage08_hedging_coverage.py research/consults/gpt55-pro-audit/backlog.md research/consults/gpt55-pro-audit/rounds/round-009-fresh-full-project-closure-audit-response.md
```

Result: no whitespace errors.

## Known Test Scope

The verification was targeted to B7-CODE-001 and adjacent hedging/report contracts. The full suite was not rerun in this step.
