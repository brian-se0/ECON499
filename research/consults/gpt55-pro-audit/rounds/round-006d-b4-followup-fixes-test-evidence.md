# Round 006D B4 Follow-Up Fix Evidence

Date: 2026-04-28

## Pro Finding Source

Round 006C decision:

- `ROUND_006C_VERIFICATION_DECISION: still_open`
- `B4-CODE-001: fixed`
- `B4-CODE-002: still_open`
- `B4-CODE-003: still_open`
- `B4-CODE-004: fixed`

Pro follow-up findings:

- `B4-CODE-002`: `validate_moneyness_points()` rejected bad values through `SurfaceGridConfig`, but direct `SurfaceGrid` construction still accepted bool/string-like moneyness values because the shared validator converted with `float(value)` before type rejection.
- `B4-CODE-003`: `value_instrument()` and Stage 08 prevalidation still clipped exhausted remaining maturity to one day through `max(remaining_days, 1)` and `max(maturity_days - max_target_gap_days, 1)`.

## Codex Follow-Up Fixes

### B4-CODE-002

- Updated `src/ivsurf/surfaces/grid_validation.py`.
- `validate_moneyness_points()` now checks each raw value before float conversion.
- It rejects bool-like values and non-`int`/`float` values, including numeric strings and arbitrary objects.
- Direct `SurfaceGrid(...)` construction now uses the same fail-fast behavior as `SurfaceGridConfig`.

Tests added/updated:

- `tests/unit/test_grid.py`
- `SurfaceGridConfig` rejects bool-like and string-like moneyness payloads.
- Direct `SurfaceGrid` construction rejects:
  - bool-like moneyness values,
  - string numeric moneyness values,
  - object-like values,
  - duplicate values that would have been created by bool coercion,
  - non-finite numeric values.

### B4-CODE-003

- Updated `src/ivsurf/hedging/revaluation.py`.
- `value_instrument()` now passes the actual `remaining_days` into `surface.sigma(...)`.
- The hidden `max(remaining_days, 1)` fallback was removed.
- Expired or below-grid remaining maturity now fails through `SurfaceDomainError` before interpolation.

- Updated `src/ivsurf/hedging/validation.py`.
- Stage 08 prevalidation now computes `target_remaining_days = maturity_days - max_target_gap_days`.
- The hidden `max(..., 1)` fallback was removed.
- Exhausted short or hedge maturities now fail even if the surface grid includes a one-day point.

Tests added/updated:

- `tests/unit/test_hedging.py`
- `value_instrument()` rejects expired remaining maturity before surface lookup.
- `evaluate_model_hedging()` rejects expired book maturity before producing a result row.
- Stage 08 prevalidation rejects exhausted short maturity when `max_target_gap_days >= short_maturity_days`.
- Stage 08 prevalidation rejects exhausted hedge maturity when `max_target_gap_days >= hedge_maturity_days`.

## Verification Commands

Ruff:

```text
uv run python -m ruff check src/ivsurf/surfaces/grid_validation.py src/ivsurf/hedging/revaluation.py src/ivsurf/hedging/validation.py tests/unit/test_grid.py tests/unit/test_hedging.py
```

Result:

```text
All checks passed!
```

Focused pytest:

```text
uv run python -m pytest tests/unit/test_grid.py tests/unit/test_hedging.py tests/property/test_surface_grid.py
```

Result:

```text
43 passed in 1.91s
```

Broader B4 verification slice:

```text
uv run python -m pytest tests/unit/test_split_manifests.py tests/unit/test_grid.py tests/unit/test_hedging.py tests/integration/test_stage05_stage06_clean_evaluation.py tests/integration/test_stats_hedging_slice.py tests/property/test_surface_grid.py tests/property/test_walkforward.py
```

Result:

```text
61 passed in 5.14s
```

Broader B4 ruff slice:

```text
uv run python -m ruff check src/ivsurf/config.py src/ivsurf/splits/manifests.py src/ivsurf/surfaces/grid.py src/ivsurf/surfaces/grid_validation.py src/ivsurf/hedging/revaluation.py src/ivsurf/hedging/hedge_rules.py src/ivsurf/hedging/pnl.py src/ivsurf/hedging/validation.py scripts/08_run_hedging_eval.py tests/unit/test_split_manifests.py tests/unit/test_grid.py tests/unit/test_hedging.py tests/integration/test_stats_hedging_slice.py tests/regression/test_report_artifacts.py tests/property/test_surface_grid.py
```

Result:

```text
All checks passed!
```

## Remaining Verification Need

Ask GPT 5.5 Pro to verify only the two Round 006C follow-up fixes:

- `B4-CODE-002`
- `B4-CODE-003`

Do not re-verify B4-CODE-001 or B4-CODE-004 unless the new follow-up changes reveal a concrete reopened defect.
