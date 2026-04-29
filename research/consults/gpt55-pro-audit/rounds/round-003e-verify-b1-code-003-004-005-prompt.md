You are GPT 5.5 Pro acting as the independent audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

We are now verifying the local implementation of Batch 1 findings B1-CODE-003, B1-CODE-004, and B1-CODE-005.

Previously Pro verified fixed:
- B1-CODE-001
- B1-CODE-002
- B1-CODE-006
- B1-CODE-007
- B1-CODE-010

Do not reopen those unless this new slice caused a regression. Focus this verification on B1-CODE-003/004/005.

Definitions from your prior audit:
- B1-CODE-003: Critical null fields lack explicit invalid reasons in option cleaning.
- B1-CODE-004: Stage 01 ingestion manifest does not log target-symbol filtering counts.
- B1-CODE-005: Stage 02 silver manifest does not summarize invalid-reason counts by date.

Local implementation summary:

1. B1-CODE-003: explicit critical-null reason codes
- Updated `src/ivsurf/cleaning/option_filters.py::apply_option_quality_flags()`.
- The cleaning reason chain now checks critical nulls before threshold/comparison logic:
  - `MISSING_OPTION_TYPE`
  - `MISSING_BID_1545`
  - `MISSING_ASK_1545`
  - `MISSING_IMPLIED_VOLATILITY_1545`
  - `MISSING_VEGA_1545`
  - `MISSING_ACTIVE_UNDERLYING_PRICE_1545`
  - `MISSING_MID_1545`
  - `MISSING_TAU_YEARS`
  - `MISSING_LOG_MONEYNESS`
- Existing reason codes remain after null checks:
  - `INVALID_OPTION_TYPE`
  - `NON_POSITIVE_BID`
  - `NON_POSITIVE_ASK`
  - `ASK_LT_BID`
  - `NON_POSITIVE_IV`
  - `NON_POSITIVE_VEGA`
  - `NON_POSITIVE_UNDERLYING_PRICE`
  - `NON_POSITIVE_MID`
  - `TAU_TOO_SHORT`
  - `TAU_TOO_LONG`
  - `OUTSIDE_MONEYNESS_RANGE`
- This prevents Polars null-comparison behavior from silently falling through to valid rows.

2. B1-CODE-004: target-symbol filtering counts in Stage 01
- Updated `src/ivsurf/io/ingest_cboe.py`.
- `IngestionResult` now includes:
  - `raw_row_count`
  - `target_symbol_row_count`
  - `non_target_symbol_row_count`
  - existing `row_count` remains the written/target-symbol row count.
- `ingest_one_zip()` now computes target-symbol counts from `underlying_symbol` before filtering to `config.target_symbol`.
- Updated `scripts/01_ingest_cboe.py` to write per-file counts into `bronze_ingestion_summary.json`.
- Stage 01 aggregate manifest now includes:
  - `rows_parsed` = raw rows seen
  - `rows_target_symbol`
  - `rows_filtered_non_target_symbol`
  - `rows_written`
- Stage 01 resume context hash now includes `artifact_schema_version: 2` to avoid reusing stale metadata.

3. B1-CODE-005: invalid-reason counts in Stage 02 silver manifest
- Updated `scripts/02_build_option_panel.py`.
- Added deterministic `_invalid_reason_counts(frame)`.
- It maps `invalid_reason is None` to `VALID`, counts every row by reason, and sorts the reason keys.
- Each daily row in `silver_build_summary.json` now records `invalid_reason_counts`.
- Stage 02 resume context hash now includes `artifact_schema_version: 2` to avoid reusing stale metadata.

Test evidence from local run:

Ruff:
`uv run python -m ruff check scripts/01_ingest_cboe.py scripts/02_build_option_panel.py src/ivsurf/io/ingest_cboe.py src/ivsurf/cleaning/option_filters.py tests/unit/test_option_filters.py tests/integration/test_stage01_ingestion_manifest_counts.py tests/integration/test_resume_stage01.py tests/integration/test_early_close_stage02.py`

Result: `All checks passed!`

Pytest:
`uv run python -m pytest tests/unit/test_option_filters.py tests/integration/test_stage01_ingestion_manifest_counts.py tests/integration/test_resume_stage01.py tests/integration/test_early_close_stage02.py`

Result: `6 passed in 1.42s`

Specific test coverage:
- `tests/unit/test_option_filters.py`
  - existing zero-IV and exclusive-threshold tests still pass;
  - new test proves null `bid_1545` and null `implied_volatility_1545` receive explicit missing-field reason codes and are invalid.
- `tests/integration/test_stage01_ingestion_manifest_counts.py`
  - creates a raw Cboe-style zip with one `^SPX` row and one non-target-symbol row;
  - proves ingestion records `raw_row_count=2`, `target_symbol_row_count=1`, `non_target_symbol_row_count=1`, `row_count=1`;
  - proves bronze output contains only `^SPX`.
- `tests/integration/test_resume_stage01.py`
  - updated fake ingestion result to include new counts;
  - proves Stage 01 summary aggregates `rows_parsed`, `rows_target_symbol`, and `rows_filtered_non_target_symbol`.
- `tests/integration/test_early_close_stage02.py`
  - now also proves Stage 02 silver summary records `invalid_reason_counts == {"VALID": 1}` for the valid early-close fixture.

Please audit this slice for:
1. Whether B1-CODE-003 is fixed, partially fixed, or still open.
2. Whether B1-CODE-004 is fixed, partially fixed, or still open.
3. Whether B1-CODE-005 is fixed, partially fixed, or still open.
4. Any required adjustments before moving to the next finding.
5. Whether the implementation introduces leakage, silent filtering, manifest ambiguity, or brittle behavior.

Respond in this exact structure:

VERIFICATION_DECISION

B1-CODE-003: fixed | partially fixed | still open
B1-CODE-004: fixed | partially fixed | still open
B1-CODE-005: fixed | partially fixed | still open

EVIDENCE
- ...

REQUIRED_ADJUSTMENTS
- ...

NEW_REGRESSION_RISKS
- ...

NEXT_FIX_SLICE
- ...
