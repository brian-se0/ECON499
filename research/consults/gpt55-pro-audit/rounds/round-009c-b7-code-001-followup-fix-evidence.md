# Round 009C B7-CODE-001 Follow-Up Fix Evidence

Date: 2026-04-28

## Verification Result Being Addressed

Round 009B decision: `ROUND_009B_VERIFICATION_DECISION: still_open`

Remaining issue: `require_hedging_summary_matches_results(...)` checked only `model_name` coverage and `n_trades`, so Stage 09 could still publish a stale `hedging_summary.parquet` with the same model set and trade counts but incorrect aggregate metric values.

## Codex Follow-Up Fix Summary

- Updated `src/ivsurf/hedging/validation.py` so `require_hedging_summary_matches_results(...)` recomputes the official summary from `hedging_results` using `summarize_hedging_results(...)`.
- The validator now requires the full published hedging summary column set:
  - `model_name`
  - `mean_abs_revaluation_error`
  - `mean_squared_revaluation_error`
  - `mean_abs_hedged_pnl`
  - `mean_squared_hedged_pnl`
  - `hedged_pnl_variance`
  - `n_trades`
- The validator compares the recomputed and published summary by model, including all metric columns.
- Counts remain exact. Floating metric comparisons use an explicit `1.0e-12` absolute/relative tolerance to avoid false failures from deterministic parquet/aggregation round-trip noise while still rejecting stale summary values.
- Stage 09 already calls this validator before report tables/figures are built, so stale same-coverage summaries now fail before `ranked_hedging_summary` or hedging figures can be published.

## Tests Added

- `tests/unit/test_hedging.py::test_hedging_summary_validation_rejects_stale_metric_values`
  - Builds valid hedging results.
  - Recomputes the official summary.
  - Mutates `mean_abs_revaluation_error` while preserving `model_name` and `n_trades`.
  - Asserts the validator rejects the stale metric value.
- `tests/integration/test_stage08_hedging_coverage.py::test_stage09_rejects_stale_hedging_summary_metrics_with_equal_coverage`
  - Creates equal-coverage forecast artifacts and equal-coverage hedging results.
  - Writes a stale `hedging_summary.parquet` with matching model/date coverage and matching `n_trades` but an incorrect metric.
  - Asserts Stage 09 fails before publishing `ranked_hedging_summary.csv`.

## Verification Commands

```text
uv run python -m ruff check scripts/08_run_hedging_eval.py scripts/09_make_report_artifacts.py src/ivsurf/hedging/validation.py tests/integration/test_stage08_hedging_coverage.py tests/unit/test_hedging.py
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
24 passed in 2.28s
```

```text
git diff --check -- src/ivsurf/hedging/validation.py tests/unit/test_hedging.py tests/integration/test_stage08_hedging_coverage.py research/consults/gpt55-pro-audit/rounds/round-009b-verify-b7-code-001-response.md
```

Result: no whitespace errors.

## Known Test Scope

The verification was targeted to the B7-CODE-001 follow-up gap and adjacent Stage 08/09 hedging/report contracts. The full suite was not rerun in this step.
