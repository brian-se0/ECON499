# Round 006 B4 Fix Evidence

Date: 2026-04-28

## Implemented Scope

- B4-CODE-001: split manifest schema bumped to `walkforward_split_manifest_v2`; Stage 05/06 artifact matching now also semantically validates split chronology, disjoint train/validation/test arrays, ordered date-universe membership, split IDs, and deterministic `build_walkforward_splits(...)` output for the recorded config.
- B4-CODE-002: surface grid schema bumped to `surface_grid_v2`; `SurfaceGridConfig` and `SurfaceGrid` now reject empty, duplicate, unsorted, non-finite, non-positive, and bool-like coordinates.
- B4-CODE-003: hedging surface interpolation now raises `SurfaceDomainError` for out-of-grid maturity or moneyness queries instead of boundary clipping; Stage 08 prevalidates hedging config/domain assumptions before result generation.
- B4-CODE-004: delta-vega hedge sizing now raises `InfeasibleHedgeError` when hedge straddle vega is zero, non-finite, or below explicit `hedge_vega_floor`; the delta-only fallback was removed.

## Verification Commands

- `uv run python -m ruff check src/ivsurf/config.py src/ivsurf/splits/manifests.py src/ivsurf/surfaces/grid.py src/ivsurf/surfaces/grid_validation.py src/ivsurf/hedging/revaluation.py src/ivsurf/hedging/hedge_rules.py src/ivsurf/hedging/pnl.py src/ivsurf/hedging/validation.py scripts/08_run_hedging_eval.py tests/unit/test_split_manifests.py tests/unit/test_grid.py tests/unit/test_hedging.py tests/integration/test_stats_hedging_slice.py tests/regression/test_report_artifacts.py tests/property/test_surface_grid.py`
  - Result: passed.
- `uv run python -m pytest tests/unit/test_split_manifests.py tests/unit/test_grid.py tests/unit/test_hedging.py tests/integration/test_stage05_stage06_clean_evaluation.py tests/integration/test_stats_hedging_slice.py tests/property/test_surface_grid.py tests/property/test_walkforward.py`
  - Result: 50 passed.
- `uv run python -m pytest tests/unit/test_alignment.py tests/unit/test_feature_dataset.py tests/unit/test_forecast_store.py tests/unit/test_interpolation.py tests/unit/test_stats.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage04_early_close_alignment.py tests/integration/test_stage07_negative_prediction.py tests/integration/test_report_stage_contract.py tests/regression/test_report_artifacts.py`
  - Result: 46 passed.

## Notes

- Full all-tests execution was not run because prior audit evidence identified native LightGBM small-fit tests as capable of crashing the Python process on this macOS CPU environment. The verification above covers the B4 contract surfaces and adjacent Stage 03/04/05/06/07/08/09 artifact contracts without invoking the known LightGBM crash path.
