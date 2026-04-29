You are GPT 5.5 Pro acting as the independent audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

This is a re-verification prompt for the adjustments you required in Round 003E.

Your Round 003E decision:
- B1-CODE-004: fixed.
- B1-CODE-003: partially verified pending broader null-field tests.
- B1-CODE-005: partially verified pending invalid-reason summary tests with multiple invalid reasons and reconciliation checks.

Local adjustments made after Round 003E:

1. B1-CODE-003 tests
- Updated `tests/unit/test_option_filters.py`.
- Added parametrized coverage for every critical missing-field reason:
  - `MISSING_OPTION_TYPE`
  - `MISSING_BID_1545`
  - `MISSING_ASK_1545`
  - `MISSING_IMPLIED_VOLATILITY_1545`
  - `MISSING_VEGA_1545`
  - `MISSING_ACTIVE_UNDERLYING_PRICE_1545`
  - `MISSING_MID_1545`
  - `MISSING_TAU_YEARS`
  - `MISSING_LOG_MONEYNESS`
- Each parametrized case builds a two-row frame: one row with the target critical null and one valid row.
- The test asserts:
  - exact missing-field reason code for the null row;
  - `is_valid_observation == False` for the null row;
  - the valid row remains valid;
  - `valid_option_rows(flagged).height == 1`, proving null-critical rows cannot pass into valid rows.

2. B1-CODE-005 tests
- Updated `tests/integration/test_early_close_stage02.py`.
- Added `test_stage02_summary_counts_invalid_reasons_by_date`.
- The test writes a three-row bronze artifact for one date:
  - one valid row;
  - one missing-field invalid row (`bid_1545=None`, expecting `MISSING_BID_1545`);
  - one threshold invalid row (`ask_1545=0.0`, expecting `NON_POSITIVE_ASK`).
- It runs `scripts/02_build_option_panel.py`.
- It asserts:
  - silver row reasons are `[None, "MISSING_BID_1545", "NON_POSITIVE_ASK"]`;
  - `summary[0]["rows"] == 3`;
  - `summary[0]["valid_rows"] == 1`;
  - `summary[0]["invalid_reason_counts"] == {"MISSING_BID_1545": 1, "NON_POSITIVE_ASK": 1, "VALID": 1}`;
  - the reason-count keys are sorted;
  - sum of counts equals `rows`;
  - `VALID` count equals `valid_rows`.

3. Deterministic reason counts
- `scripts/02_build_option_panel.py::_invalid_reason_counts()` now returns sorted reason-count keys:
  - `return {reason: int(count) for reason, count in sorted(Counter(reasons).items())}`

Test evidence after adjustments:

Ruff:
`uv run python -m ruff check src/ivsurf/cleaning/option_filters.py scripts/02_build_option_panel.py tests/unit/test_option_filters.py tests/integration/test_early_close_stage02.py`

Result: `All checks passed!`

Pytest:
`uv run python -m pytest tests/unit/test_option_filters.py tests/integration/test_stage01_ingestion_manifest_counts.py tests/integration/test_resume_stage01.py tests/integration/test_early_close_stage02.py`

Result: `16 passed in 1.57s`

Please verify whether your Round 003E required adjustments are now satisfied.

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
