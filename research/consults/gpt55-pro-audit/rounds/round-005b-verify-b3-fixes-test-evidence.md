# Round 005B B3 Fix Verification Evidence

Date: 2026-04-28

Scope implemented from Round 005:
- B3-CODE-001: tuning manifests now require `schema_version="tuning_result_v2"`; stale manifests without the field are rejected; persisted params reject unexpected keys for naive, ridge, elastic net, HAR, LightGBM, random forest, and neural surface.
- B3-CODE-002: `weighted_surface_mse` now validates tensor shape, finiteness, binary masks, nonnegative training weights, finite nonnegative loss weights, positive supervised weight, and finite scalar loss output without clamping.
- B3-CODE-003: observed metric paths now use shared observed-mask/weight contracts, reject nonbinary masks and negative/nonfinite weights, remove the legacy pooled diagnostic from stage 05, and fail fast instead of falling back to full-grid scoring.
- B3-CODE-004: a shared complete-and-unique fixed-grid contract now protects daily surface pivoting, dense reshaping, inferred interpolator construction, and actual/forecast artifact loading boundaries.

Commands run:
- `uv run python -m ruff check scripts/05_tune_models.py src/ivsurf/training/tuning.py src/ivsurf/training/model_factory.py src/ivsurf/models/losses.py src/ivsurf/evaluation/metric_contracts.py src/ivsurf/evaluation/loss_panels.py src/ivsurf/evaluation/alignment.py src/ivsurf/evaluation/slice_reports.py src/ivsurf/features/tabular_dataset.py src/ivsurf/surfaces/grid.py src/ivsurf/features/lagged_surface.py src/ivsurf/surfaces/masks.py src/ivsurf/hedging/revaluation.py tests/unit/test_losses.py tests/unit/test_training_behaviour.py tests/unit/test_tuning_workflow.py tests/integration/test_stage07_negative_prediction.py tests/unit/test_stats.py tests/unit/test_slice_reports.py tests/unit/test_feature_dataset.py tests/unit/test_grid.py tests/unit/test_hedging.py`
  - Result: all checks passed.
- `uv run python -m pytest tests/unit/test_losses.py tests/unit/test_tuning_workflow.py tests/unit/test_stats.py tests/unit/test_slice_reports.py tests/unit/test_feature_dataset.py tests/unit/test_grid.py tests/unit/test_hedging.py tests/unit/test_training_behaviour.py::test_model_factory_rejects_string_numeric_hyperparameters tests/unit/test_training_behaviour.py::test_model_factory_rejects_float_integer_hyperparameters tests/unit/test_training_behaviour.py::test_model_factory_rejects_string_lightgbm_integer_params tests/unit/test_training_behaviour.py::test_model_factory_rejects_unexpected_persisted_params`
  - Result: 55 passed.
- `uv run python -m pytest tests/integration/test_stage05_stage06_clean_evaluation.py tests/integration/test_neural_imputed_cell_supervision.py tests/integration/test_stats_hedging_slice.py tests/integration/test_report_stage_contract.py tests/regression/test_report_artifacts.py tests/regression/test_observed_mask_preservation.py`
  - Result: 9 passed.
- `uv run python -m pytest tests/integration/test_stage07_negative_prediction.py tests/integration/test_smoke_pipeline.py tests/unit/test_alignment.py tests/unit/test_reporting_helpers.py`
  - Result: 13 passed.
- `uv run python -m pytest -k "not lightgbm"`
  - Result: 185 passed, 2 skipped, 11 deselected.

Full-suite limitation:
- `uv run python -m pytest tests/unit/test_losses.py tests/unit/test_training_behaviour.py ...` hit a native LightGBM segmentation fault inside `lightgbm/basic.py::__init_from_np2d` during an existing small-fit LightGBM unit. The crash occurred before Python assertions in the rest of that file could run. The B3 model-factory LightGBM parameter-validation test was run separately and passed.

Pattern search after fixes:
- `rg "legacy_pooled|legacy_validation_metric|np\\.maximum\\(vega|clamp_min\\(training_weights|observed_mask > 0\\.5|observed_mask_array > 0\\.5" -n src scripts tests`
  - Result: no matches.

## Round 005B Follow-up Fixes After Pro Verification

Date: 2026-04-28

Pro verification response:
- `research/consults/gpt55-pro-audit/rounds/round-005b-verify-b3-fixes-response.md`

Additional fixes implemented:
- B3-CODE-003A: `build_forecast_realization_panel` now validates raw `actual_vega_sum` before deriving `observed_weight`. The contract rejects null, nonfinite, and negative vega sums on every aligned actual cell, including unobserved cells, and requires strictly positive `actual_vega_sum` where `actual_observed_mask == True`.
- B3-CODE-004A: saved-artifact and hedging reshape boundaries no longer use artifact-inferred grids as the final expected grid. Stage 07, Stage 08, and Stage 09 load `configs/data/surface.yaml`, build a configured `SurfaceGrid`, require artifact metadata to match `grid.grid_hash`, and validate complete unique grid coverage against that configured grid. `surface_interpolator_from_frame` now requires an explicit `SurfaceGrid`.

New/updated tests:
- `tests/unit/test_alignment.py`
  - invalid raw `actual_vega_sum` on unobserved cells: negative, null, NaN, Inf;
  - nonpositive raw `actual_vega_sum` on observed cells;
  - explicit configured-grid forecast validation rejects full maturity-band omission;
  - explicit configured-grid forecast validation rejects full moneyness-band omission;
  - explicit configured-grid forecast validation rejects coordinate mismatch.
- `tests/unit/test_hedging.py`
  - `surface_interpolator_from_frame` rejects incomplete explicit grids;
  - rejects shrunken inferred-grid cases when a full expected grid is provided;
  - rejects configured coordinate mismatches.
- `tests/integration/test_stage07_negative_prediction.py`
  - supplies the test surface config explicitly to the Stage 07 hard-cutover API.

Commands run:
- `uv run python -m ruff check src/ivsurf/evaluation/alignment.py src/ivsurf/hedging/revaluation.py scripts/07_run_stats.py scripts/08_run_hedging_eval.py scripts/09_make_report_artifacts.py tests/unit/test_alignment.py tests/unit/test_hedging.py tests/integration/test_stats_hedging_slice.py tests/regression/test_report_artifacts.py tests/integration/test_report_stage_contract.py`
  - Result: all checks passed.
- `uv run python -m pytest tests/unit/test_alignment.py tests/unit/test_hedging.py`
  - Result: 25 passed.
- `uv run python -m pytest tests/integration/test_stats_hedging_slice.py tests/integration/test_report_stage_contract.py tests/regression/test_report_artifacts.py`
  - Result: 3 passed.
- `uv run python -m pytest tests/unit/test_alignment.py tests/unit/test_hedging.py tests/unit/test_grid.py tests/unit/test_stats.py tests/unit/test_slice_reports.py tests/unit/test_feature_dataset.py tests/unit/test_losses.py tests/unit/test_tuning_workflow.py tests/unit/test_training_behaviour.py -k 'not lightgbm_training and not lightgbm_sklearn and not lightgbm_model_predict' tests/integration/test_stage07_negative_prediction.py tests/integration/test_neural_imputed_cell_supervision.py tests/integration/test_stats_hedging_slice.py tests/integration/test_report_stage_contract.py tests/regression/test_report_artifacts.py`
  - Result: 89 passed, 3 deselected.
- `uv run python -m ruff check src/ivsurf/evaluation/alignment.py src/ivsurf/hedging/revaluation.py scripts/07_run_stats.py scripts/08_run_hedging_eval.py scripts/09_make_report_artifacts.py tests/unit/test_alignment.py tests/unit/test_hedging.py tests/unit/test_stats.py tests/integration/test_stage07_negative_prediction.py tests/integration/test_stats_hedging_slice.py tests/regression/test_report_artifacts.py tests/integration/test_report_stage_contract.py`
  - Result: all checks passed.
