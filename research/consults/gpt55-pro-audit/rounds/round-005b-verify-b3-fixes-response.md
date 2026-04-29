# Round 005B GPT 5.5 Pro Verification Response

Source conversation: `https://chatgpt.com/c/69f0058e-37cc-83ea-a7a6-6ccfe52f2b28`

Note: Pro's response stalled for an extended period on `Inspecting configuration for potential B3-002 issue`. Codex stopped the stream after repeated identical DOM hashes and saved the visible response below. The visible response contained actionable verification findings and a complete `NEXT_CODEX_ACTIONS` list.

```text
ROUND_005B_VERIFICATION_DECISION: still_open

B3-CODE-001: fixed
B3-CODE-002: fixed
B3-CODE-003: still_open
B3-CODE-004: still_open

EVIDENCE_REVIEWED:
- AGENTS.md
- research/consults/gpt55-pro-audit/rounds/round-005b-verify-b3-fixes-test-evidence.md
- research/consults/gpt55-pro-audit/backlog.md
- src/ivsurf/training/tuning.py
- src/ivsurf/training/model_factory.py
- scripts/05_tune_models.py
- scripts/06_run_walkforward.py
- tests/unit/test_tuning_workflow.py
- tests/unit/test_training_behaviour.py
- src/ivsurf/models/losses.py
- src/ivsurf/models/neural_surface.py
- tests/unit/test_losses.py
- tests/integration/test_neural_imputed_cell_supervision.py
- src/ivsurf/evaluation/metric_contracts.py
- src/ivsurf/evaluation/loss_panels.py
- src/ivsurf/evaluation/alignment.py
- src/ivsurf/evaluation/slice_reports.py
- scripts/07_run_stats.py
- tests/unit/test_stats.py
- tests/unit/test_slice_reports.py
- tests/unit/test_alignment.py
- src/ivsurf/surfaces/grid.py
- src/ivsurf/features/lagged_surface.py
- src/ivsurf/surfaces/masks.py
- src/ivsurf/hedging/revaluation.py
- scripts/08_run_hedging_eval.py
- tests/unit/test_grid.py
- tests/unit/test_hedging.py

OPEN_FINDINGS:

Finding ID: B3-CODE-003A
Severity: P1
Evidence: src/ivsurf/evaluation/alignment.py::build_forecast_realization_panel derives observed_weight with:
pl.when(pl.col("actual_observed_mask")).then(pl.col("actual_vega_sum")).otherwise(0.0).alias("observed_weight")
then validates only the derived observed_weight through require_observed_weight_contract. The raw actual_vega_sum column is not included in _require_non_null_columns, is not validated as finite/nonnegative before masking, and can therefore be hidden when actual_observed_mask == False. A negative, null, or nonfinite actual_vega_sum on an unobserved target cell is converted to observed_weight = 0.0 before the shared observed-weight contract sees it. This leaves an invalid target-day vega artifact able to pass Stage 07 alignment.

Exact Codex action required: Add a strict validation step for actual_vega_sum before deriving observed_weight. The contract should require no nulls, finite numeric values, and nonnegative vega sums for every aligned actual cell, plus strictly positive actual_vega_sum where actual_observed_mask == True. Do not require zero actual_vega_sum on unobserved cells, because low-count cells may have positive vega but be excluded from observed-cell scoring. Add tests proving build_forecast_realization_panel rejects negative, null, and NaN/Inf actual_vega_sum even when the affected cell is unobserved.

Finding ID: B3-CODE-004A
Severity: P2
Evidence: The new require_complete_unique_surface_grid contract is sound when it receives the expected SurfaceGrid, but several artifact-reader/interpolator paths infer the expected grid from the same artifact being validated. src/ivsurf/evaluation/alignment.py::load_actual_surface_frame calls infer_surface_grid_from_frame(frame, ...) and then validates frame against that inferred grid. src/ivsurf/evaluation/alignment.py::load_forecast_frame does the same for forecast artifacts. src/ivsurf/hedging/revaluation.py::surface_interpolator_from_frame also infers the grid from the input frame before reshaping. This cannot detect a surface artifact that is missing an entire terminal maturity band or moneyness band, because the inferred grid simply shrinks. It also cannot detect coordinate mismatches relative to the configured/persisted project grid when the corrupted coordinates are internally consistent. scripts/08_run_hedging_eval.py reads per-model forecast parquet files directly and then passes per-date groups to surface_interpolator_from_frame, so a forecast group with a complete-but-shrunk inferred grid can still be reshaped and valued.

Exact Codex action required: Do not use an artifact-inferred grid as the final expected-grid contract at saved-artifact reader or hedging reshape boundaries. Stage 07 and Stage 08 should load or receive the configured SurfaceGrid from configs/data/surface.yaml, require surface_grid_hash == grid.grid_hash, and call require_complete_unique_surface_grid(..., grid, group_columns=...) before alignment, stats, or hedging valuation. surface_interpolator_from_frame should accept an explicit expected SurfaceGrid or be called only after an explicit expected-grid validation; avoid a fallback path that silently infers a smaller valid grid. Add tests for missing an entire maturity row, missing an entire moneyness column, coordinate values mismatching the configured grid while row count remains complete, and Stage 08 rejecting a per-model forecast group before valuation.

NEXT_CODEX_ACTIONS:
- Fix B3-CODE-003A by validating raw actual_vega_sum before observed_weight derivation in alignment, with targeted unit tests for observed and unobserved corrupted vega cells.
- Fix B3-CODE-004A by replacing final artifact-boundary inferred-grid validation with explicit configured-grid validation in Stage 07/Stage 08/loaders/interpolator calls, with targeted tests for full-row/full-column omissions and coordinate mismatches.
- Rerun the targeted Round 005B test set plus the new regression tests, then refresh the Round 005B test-evidence file.
```
