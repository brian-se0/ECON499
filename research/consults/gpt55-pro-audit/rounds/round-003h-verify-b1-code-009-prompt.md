You are GPT 5.5 Pro acting as the independent audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

We are now verifying the local implementation of Batch 1 finding B1-CODE-009.

Previously Pro verified fixed:
- B1-CODE-001 through B1-CODE-008
- B1-CODE-010

Definition from your prior audit:
- B1-CODE-009: Forecast artifacts lack split/grid/target version metadata. Forecast artifacts must be self-describing enough to prove which split, surface version, grid, target construction, model config, and training/run context they correspond to; alignment must fail when incompatible forecast and realized-surface metadata are mixed.

Local implementation summary:

1. Target-surface versioning
- Added `COMPLETED_SURFACE_SCHEMA_VERSION = "completed_surface_v1"` in `src/ivsurf/surfaces/interpolation.py`.
- `scripts/03_build_surfaces.py` now writes `target_surface_version` into:
  - every gold surface parquet row;
  - `gold_surface_summary.json`.
- `src/ivsurf/features/tabular_dataset.py` now carries `target_surface_version` into `daily_features.parquet`.
- `src/ivsurf/features/availability.py` classifies `target_surface_version` as surface metadata.

2. Forecast-store metadata
- Updated `src/ivsurf/evaluation/forecast_store.py::write_forecasts()`.
- Forecast writing now requires:
  - `split_ids`
  - `surface_config_hash`
  - `model_config_hash`
  - `training_run_id`
- Forecast parquet rows now persist:
  - `split_id`
  - existing B1-CODE-007 fields: `surface_grid_schema_version`, `surface_grid_hash`, `maturity_coordinate`, `moneyness_coordinate`
  - `target_surface_version`
  - `surface_config_hash`
  - `model_config_hash`
  - `training_run_id`
  - existing B1-CODE-010 timestamps: `effective_decision_timestamp`, `target_effective_decision_timestamp`
- The writer validates that split IDs and the new hash/run-id fields are non-empty strings.

3. Stage 06 propagation
- Updated `scripts/06_run_walkforward.py`.
- Stage 06 now computes:
  - `surface_config_hash = sha256_file(surface_config_path)`
  - `model_config_hash` from deterministic JSON over model name, base params, tuned params, HPO profile name, and training profile name.
  - `training_run_id` from deterministic JSON over Stage 06, model name, workflow run label, resume context hash, model config hash, and surface config hash.
- During the split loop, Stage 06 now collects one `split_id` per forecast row and passes it to `write_forecasts()`.
- Stage 06 model metadata also records `model_config_hash`, `training_run_id`, and `surface_config_hash`.

4. Evaluation alignment safeguards
- Updated `src/ivsurf/evaluation/alignment.py`.
- `load_actual_surface_frame()` now loads `target_surface_version` from gold artifacts.
- `build_forecast_realization_panel()` now requires non-null:
  - `split_id`
  - `surface_config_hash`
  - `model_config_hash`
  - `training_run_id`
  - `target_surface_version`
  - actual/origin target-surface versions from gold joins.
- It fails if forecast `target_surface_version` differs from target-day or origin-day gold `target_surface_version`.
- It continues to fail on grid hash/schema/coordinate mismatches, so forecast rows cannot align against incompatible realized surfaces.

Test evidence:

Ruff:
`uv run python -m ruff check src/ivsurf/evaluation/forecast_store.py src/ivsurf/evaluation/alignment.py src/ivsurf/features/tabular_dataset.py src/ivsurf/features/availability.py src/ivsurf/surfaces/interpolation.py scripts/03_build_surfaces.py scripts/06_run_walkforward.py tests/unit/test_forecast_store.py tests/unit/test_alignment.py tests/unit/test_stats.py tests/integration/test_stage03_stage04_target_gap_alignment.py`

Result: `All checks passed!`

Pytest:
`uv run python -m pytest tests/unit/test_forecast_store.py tests/unit/test_alignment.py tests/unit/test_stats.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage05_stage06_clean_evaluation.py`

Result: `18 passed in 3.86s`

Specific test coverage:
- `tests/unit/test_forecast_store.py`
  - verifies forecast parquet persists split IDs, surface config hash, model config hash, training run ID, target decision timestamps, and grid metadata.
- `tests/unit/test_alignment.py`
  - verifies forecast-realization alignment rejects grid metadata mismatch.
  - existing alignment tests now include split/model/run metadata and target-surface version.
- `tests/unit/test_stats.py`
  - daily loss tests still pass with the enriched alignment metadata.
- `tests/integration/test_stage03_stage04_target_gap_alignment.py`
  - verifies Stage 03 gold parquet and summary include `target_surface_version`.
- `tests/integration/test_stage05_stage06_clean_evaluation.py`
  - verifies Stage 06 still emits clean walk-forward forecasts after adding split IDs and model/run hashes.

Please audit this slice for:
1. Whether B1-CODE-009 is fixed, partially fixed, or still open.
2. Any required adjustments before moving to broader audit rounds.
3. Whether the implementation introduces leakage, reproducibility ambiguity, excessive brittleness, or false assurance.

Respond in this exact structure:

VERIFICATION_DECISION

B1-CODE-009: fixed | partially fixed | still open

EVIDENCE
- ...

REQUIRED_ADJUSTMENTS
- ...

NEW_REGRESSION_RISKS
- ...

NEXT_FIX_SLICE
- ...
