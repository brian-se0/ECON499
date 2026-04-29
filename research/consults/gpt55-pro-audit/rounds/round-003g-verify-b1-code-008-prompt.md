You are GPT 5.5 Pro acting as the independent audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

We are now verifying the local implementation of Batch 1 finding B1-CODE-008.

Previously Pro verified fixed:
- B1-CODE-001
- B1-CODE-002
- B1-CODE-003
- B1-CODE-004
- B1-CODE-005
- B1-CODE-006
- B1-CODE-007
- B1-CODE-010

Definition from your prior audit:
- B1-CODE-008: Empty observed-weight metrics can fall back to full-grid/unweighted means. Observed-cell and completed-grid evaluation must remain separate; empty observed slices must be explicitly skipped, errored, or labeled rather than silently scored as full-grid metrics.

Local implementation summary:

1. Low-level metric helpers no longer fall back
- Updated `src/ivsurf/evaluation/metrics.py`.
- `_normalize_weights()` now rejects:
  - empty weight arrays;
  - non-finite weights;
  - negative weights;
  - non-positive total weight.
- `weighted_rmse()`, `weighted_mae()`, and `weighted_mse()` now all use `_normalize_weights()`.
- `weighted_mse()` no longer returns an unweighted mean when total weight is non-positive.

2. Stage 05 validation-loss semantics now fail fast for empty observed weights
- Updated `src/ivsurf/evaluation/loss_panels.py::daily_loss_metric_values()`.
- For metrics whose name starts with `observed_`, it calls `_require_observed_metric_weights()`.
- If a validation surface has no observed cells with positive observed weight, it raises:
  - `<metric> requires at least one observed cell with positive observed weight; refusing to fall back to full-grid scoring.`
- Full-grid metrics still use explicit all-ones full-grid weights.

3. Stage 07 daily loss artifacts label empty observed-weight rows as NaN
- Updated `src/ivsurf/evaluation/loss_panels.py::build_daily_loss_frame()`.
- It computes `observed_has_positive_weight`.
- All observed metrics are written as `NaN` if observed cells are absent or their observed weight sum is non-positive.
- Full-grid metrics remain finite and separately computed with full-grid weights.
- Observed cell counts remain recorded.

4. Slice metrics label empty observed slices as NaN
- Updated `src/ivsurf/evaluation/slice_reports.py`.
- `_build_metric_row()` now returns a NaN metric row when the scoped slice is empty or its scoped weights have non-positive total weight.
- Empty observed slices retain `evaluation_scope="observed"` and `cell_count=0`, while the corresponding `evaluation_scope="full"` slice remains separately scored.

Test evidence:

Ruff:
`uv run python -m ruff check src/ivsurf/evaluation/metrics.py src/ivsurf/evaluation/loss_panels.py src/ivsurf/evaluation/slice_reports.py tests/unit/test_metrics.py tests/unit/test_stats.py tests/unit/test_slice_reports.py`

Result: `All checks passed!`

Pytest:
`uv run python -m pytest tests/unit/test_metrics.py tests/unit/test_stats.py tests/unit/test_slice_reports.py tests/unit/test_tuning_workflow.py`

Result: `17 passed in 1.88s`

Specific test coverage:
- `tests/unit/test_metrics.py`
  - `test_weighted_mse_rejects_empty_weight_population()` proves zero-total weights raise instead of falling back.
- `tests/unit/test_stats.py`
  - `test_observed_metrics_do_not_fall_back_when_observed_weight_is_empty()` proves daily observed metrics become NaN when observed weights sum to zero, while full-grid metrics remain finite.
  - `test_validation_observed_metric_rejects_empty_observed_weight()` proves Stage 05 validation observed metrics raise instead of falling back.
- `tests/unit/test_slice_reports.py`
  - `test_observed_slice_metrics_remain_nan_when_slice_has_no_observed_cells()` proves an empty observed maturity slice is labeled with NaN observed metrics and `cell_count=0`, while the full slice is finite.
- Existing `tests/unit/test_tuning_workflow.py` still passes under the stricter metric behavior.

Please audit this slice for:
1. Whether B1-CODE-008 is fixed, partially fixed, or still open.
2. Any required adjustments before moving to the next finding.
3. Whether the implementation introduces leakage, metric ambiguity, HPO instability, or brittle behavior.

Respond in this exact structure:

VERIFICATION_DECISION

B1-CODE-008: fixed | partially fixed | still open

EVIDENCE
- ...

REQUIRED_ADJUSTMENTS
- ...

NEW_REGRESSION_RISKS
- ...

NEXT_FIX_SLICE
- ...
