You are GPT 5.5 Pro acting as the independent audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

We are now verifying the local implementation of Batch 1 findings B1-CODE-007 and B1-CODE-010.

Previously Pro verified:
- B1-CODE-001 fixed.
- B1-CODE-002 fixed.
- B1-CODE-006 fixed, including propagation to evaluation/loss artifacts.

Do not reopen those unless this new slice caused a regression. Focus this verification on B1-CODE-007 and B1-CODE-010.

Definitions from your prior audit:
- B1-CODE-007: Surface/forecast artifacts lack explicit coordinate metadata. Required: declare that moneyness is log spot moneyness, maturity is ACT/365 years from effective decision timestamp to last tradable close, and persist grid/config identity or coordinate metadata in gold and forecast artifacts.
- B1-CODE-010: Effective 15:45 decision timestamp metadata is not persisted downstream. Required: persist the effective decision timestamp through silver/gold/features/forecasts, including early-close close-minus-buffer sessions; add field-availability metadata for feature columns.

Local implementation summary:

1. Coordinate metadata contract
- Added constants in `src/ivsurf/surfaces/grid.py`:
  - `MONEYNESS_COORDINATE = "log_spot_moneyness"`
  - `MATURITY_COORDINATE = "ACT/365_years_from_effective_decision_to_last_tradable_close"`
  - `SURFACE_GRID_SCHEMA_VERSION = "surface_grid_v1"`
- `SurfaceGrid.metadata` now includes schema version, coordinate labels, maturity days, and moneyness points.
- `SurfaceGrid.grid_hash` is a SHA256 of the deterministic JSON metadata payload.
- `require_surface_grid_metadata(frame, grid, dataset_name=...)` now fails if persisted artifacts are missing metadata columns or if their values do not match the configured grid.

2. Effective decision timestamp construction
- Added `DECISION_TIMESTAMP_COLUMN = "effective_decision_timestamp"` in `src/ivsurf/cleaning/derived_fields.py`.
- Added `add_effective_decision_timestamp(frame, calendar_config)`:
  - requires a single quote date in the daily artifact;
  - computes `MarketCalendar.effective_decision_datetime(quote_date).isoformat()`;
  - writes the ISO-8601 timestamp with timezone offset to `effective_decision_timestamp`.
- `scripts/02_build_option_panel.py` now applies this after derived-field construction and before quality flags, persists the column to silver parquet, and records it in `silver_build_summary.json`.

3. Gold surface artifact propagation
- `scripts/03_build_surfaces.py` now requires `effective_decision_timestamp` in silver input.
- Stage 03 requires a single non-empty timestamp per silver artifact.
- Built gold surfaces now persist row-level:
  - `effective_decision_timestamp`
  - `surface_grid_schema_version`
  - `surface_grid_hash`
  - `maturity_coordinate`
  - `moneyness_coordinate`
  - existing B1-CODE-006 `completion_status`
- `gold_surface_summary.json` records the same timestamp/grid/coordinate metadata.

4. Daily feature artifact propagation and availability manifest
- `src/ivsurf/features/lagged_surface.py` now requires surface grid metadata on input gold surfaces and carries one decision timestamp per quote date in `DailySurfaceArrays`.
- `src/ivsurf/features/tabular_dataset.py` now writes daily-feature rows with:
  - `effective_decision_timestamp`
  - `target_effective_decision_timestamp`
  - `surface_grid_schema_version`
  - `surface_grid_hash`
  - `maturity_coordinate`
  - `moneyness_coordinate`
- Added `src/ivsurf/features/availability.py`.
- `build_feature_availability_manifest(feature_frame)` declares every daily-feature artifact column as one of:
  - alignment key
  - decision timestamp
  - model feature
  - target
  - target alignment
  - target weight
  - surface metadata
- It fails if any column lacks declared availability metadata.
- `scripts/04_build_features.py` writes `feature_availability_manifest.json`, includes it in resume output requirements and run-manifest output artifacts, and validates gold surface grid metadata before building features.

5. Forecast artifact propagation
- `src/ivsurf/models/base.py` now carries `decision_timestamps` and `target_decision_timestamps` in `DatasetMatrices`.
- `src/ivsurf/evaluation/forecast_store.py::write_forecasts()` now requires:
  - `decision_timestamps`
  - `target_decision_timestamps`
- Forecast parquet rows now persist:
  - `effective_decision_timestamp`
  - `target_effective_decision_timestamp`
  - `surface_grid_schema_version`
  - `surface_grid_hash`
  - `maturity_coordinate`
  - `moneyness_coordinate`
- `scripts/06_run_walkforward.py` now collects the timestamp arrays from the feature dataset for every test split and passes them into `write_forecasts()`.

6. Evaluation alignment checks
- `src/ivsurf/evaluation/alignment.py::load_actual_surface_frame()` now loads the decision timestamp and surface-grid metadata from gold artifacts.
- `build_forecast_realization_panel()` now:
  - renames joined target/origin gold timestamps and metadata;
  - requires non-null forecast, actual target, and origin timestamp/metadata columns;
  - fails if forecast target timestamp differs from actual target gold timestamp;
  - fails if forecast origin timestamp differs from origin gold timestamp;
  - fails if forecast grid hash/schema/coordinate labels differ from actual target or origin gold metadata.

Test evidence from local run:

Ruff:
`uv run python -m ruff check src/ivsurf/surfaces/grid.py src/ivsurf/cleaning/derived_fields.py src/ivsurf/features/lagged_surface.py src/ivsurf/features/availability.py src/ivsurf/features/tabular_dataset.py src/ivsurf/models/base.py src/ivsurf/evaluation/forecast_store.py src/ivsurf/evaluation/alignment.py scripts/02_build_option_panel.py scripts/03_build_surfaces.py scripts/04_build_features.py scripts/06_run_walkforward.py tests/unit/test_grid.py tests/unit/test_forecast_store.py tests/unit/test_feature_dataset.py tests/unit/test_alignment.py tests/unit/test_stats.py tests/integration/test_early_close_stage02.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage04_early_close_alignment.py tests/integration/test_stage05_stage06_clean_evaluation.py tests/integration/test_neural_imputed_cell_supervision.py tests/integration/test_smoke_pipeline.py`

Result: `All checks passed!`

Pytest:
`uv run python -m pytest tests/unit/test_grid.py tests/unit/test_forecast_store.py tests/unit/test_feature_dataset.py tests/unit/test_alignment.py tests/unit/test_stats.py tests/integration/test_early_close_stage02.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage04_early_close_alignment.py tests/integration/test_stage05_stage06_clean_evaluation.py tests/integration/test_neural_imputed_cell_supervision.py tests/integration/test_smoke_pipeline.py`

Result: `28 passed in 6.40s`

Specific test coverage added/updated:
- `tests/unit/test_grid.py`: validates explicit grid metadata and grid hash.
- `tests/unit/test_forecast_store.py`: verifies forecast artifacts persist target decision timestamps and grid/coordinate metadata.
- `tests/unit/test_feature_dataset.py`: verifies daily feature rows carry origin and target decision timestamps; availability manifest rejects undeclared columns.
- `tests/unit/test_alignment.py`: verifies alignment preserves provenance and rejects bad forecasts before IV conversion under the metadata contract.
- `tests/integration/test_early_close_stage02.py`: verifies 2019-11-29 early-close silver timestamp is `2019-11-29T12:45:00-05:00`.
- `tests/integration/test_stage03_stage04_target_gap_alignment.py`: verifies Stage 03 gold summary/gold parquet metadata and Stage 04 feature/target timestamps plus `feature_availability_manifest.json`.
- `tests/integration/test_stage04_early_close_alignment.py`: verifies a pre-Thanksgiving quote row targets the 2019-11-29 early-close surface and carries target timestamp `2019-11-29T12:45:00-05:00`.
- `tests/integration/test_stage05_stage06_clean_evaluation.py`: verifies Stage 06 still emits clean walk-forward forecasts after timestamp propagation.
- `tests/integration/test_neural_imputed_cell_supervision.py` and `tests/integration/test_smoke_pipeline.py`: verify the model-ready dataset path still works under the stricter metadata/grid-domain contract.

Please audit this slice for:
1. Whether B1-CODE-007 is fixed, partially fixed, or still open.
2. Whether B1-CODE-010 is fixed, partially fixed, or still open.
3. Any required adjustments before moving to the next findings.
4. Whether the new metadata/timestamp contract introduces leakage, false assurance, or brittle behavior.

Respond in this exact structure:

VERIFICATION_DECISION

B1-CODE-007: fixed | partially fixed | still open
B1-CODE-010: fixed | partially fixed | still open

EVIDENCE
- ...

REQUIRED_ADJUSTMENTS
- ...

NEW_REGRESSION_RISKS
- ...

NEXT_FIX_SLICE
- ...
