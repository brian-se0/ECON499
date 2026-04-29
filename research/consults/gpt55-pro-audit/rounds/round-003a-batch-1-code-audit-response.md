# Round 003A Batch 1 Code Audit Response

Captured: 2026-04-27

FORMAL_FINDINGS

Line numbers are relative to the pasted snippets.

B1-CODE-001

Severity: P1

File and line number(s):

src/ivsurf/surfaces/aggregation.py, lines 12-31

scripts/03_build_surfaces.py, lines 79-96

src/ivsurf/surfaces/interpolation.py, lines 15-37 and 40-78

Evidence from code:

aggregate_daily_surface() computes observed_total_variance for every populated cell, then separately sets observed_mask = observation_count >= observed_cell_min_count.

scripts/03_build_surfaces.py builds observed_matrix directly from observed["observed_total_variance"], regardless of observed_mask.

_fill_axis() in interpolation.py uses finite_mask = np.isfinite(values) and never uses observed_mask to decide interpolation anchors.

Why it violates the Batch 1 standard:

Cells below observed_cell_min_count are marked unobserved but can still be used as interpolation anchors if their observed_total_variance is finite. This breaks the observed-mask semantics required by Batch 1: non-observed cells must not silently drive completed-grid construction.

Minimal fix direction:

Before interpolation, force all ~observed_mask cells to NaN, or make complete_surface() use observed_mask as the anchor mask.

Consider nulling observed_total_variance / observed_iv for cells that fail observed_cell_min_count, or add separate columns for raw aggregated value versus accepted observed cell.

Required tests:

Add a regression test where a low-count, mask-false cell has an extreme value and prove it does not influence completion.

Extend test_thresholded_observed_mask_survives_surface_completion to assert mask-false cells are not interpolation anchors, not only that the mask survives.

B1-CODE-002

Severity: P1

File and line number(s):

src/ivsurf/config.py, CleaningConfig, approximately lines 45-63

configs/data/surface.yaml, moneyness and maturity grids

src/ivsurf/surfaces/grid.py, lines 30-52

Evidence from code:

Default cleaning allows max_abs_log_moneyness = 0.5 and max_tau_years = 2.5.

The configured grid only spans log-moneyness [-0.30, 0.30] and maturities up to 730 days.

assign_grid_indices() uses -inf and inf boundary edges, so valid rows outside the explicit grid support are silently assigned to the nearest boundary cell.

Why it violates the Batch 1 standard:

Batch 1 requires domain truncation, interpolation, and extrapolation to be explicit and flagged. Rows with log-moneyness 0.40 or maturity beyond the grid maximum can become “observed” boundary-cell inputs rather than being flagged as outside-grid support.

Minimal fix direction:

Align cleaning bounds with the grid domain, or add explicit outside_grid_domain / boundary_bucketed flags.

Replace infinite grid assignment edges with explicit domain checks unless boundary bins are intentionally documented.

Persist counts of outside-domain or boundary-assigned rows by date and axis.

Required tests:

Add tests showing rows with log-moneyness outside [-0.30, 0.30] are rejected or flagged, not silently assigned to edge cells.

Add tests showing maturities beyond configured grid support are rejected or flagged.

Add a stage-03 integration test that asserts outside-domain counts appear in the surface manifest.

B1-CODE-003

Severity: P1

File and line number(s):

src/ivsurf/io/ingest_cboe.py, lines 76-84

scripts/02_build_option_panel.py, lines 62-80

src/ivsurf/cleaning/option_filters.py, lines 8-39

Evidence from code:

Ingestion only asserts non-null values for quote_date, expiration, root, strike, and option_type.

Stage 02 checks required columns but does not assert non-null 15:45 bid, ask, IV, vega, underlying price, or derived fields.

apply_option_quality_flags() compares values to thresholds but has no explicit is_null() invalid-reason branches.

Why it violates the Batch 1 standard:

Rows with null bid_1545, ask_1545, implied_volatility_1545, vega_1545, active_underlying_price_1545, mid_1545, tau_years, or log_moneyness can avoid explicit invalid-reason labeling. Batch 1 requires no silent coercions/drops and explicit filter reasons.

Minimal fix direction:

Add explicit null checks and reason codes before numeric threshold checks, for example NULL_BID_1545, NULL_ASK_1545, NULL_IV_1545, NULL_VEGA_1545, NULL_UNDERLYING_1545, NULL_TAU, NULL_LOG_MONEYNESS.

Alternatively fail fast on nulls for fields that should never be null after ingestion.

Required tests:

Unit tests for each critical null field proving the row is either rejected or flagged with the correct reason code.

Integration test proving no null-valued row can reach valid_option_rows().

B1-CODE-004

Severity: P2

File and line number(s):

src/ivsurf/io/ingest_cboe.py, lines 54-66 and 102-108

scripts/01_ingest_cboe.py, lines 67-86

Evidence from code:

ingest_one_zip() filters to underlying_symbol == config.target_symbol before producing row_count.

IngestionResult.row_count is the post-filter SPX row count.

Stage 01 manifest fields rows_parsed and rows_written are computed from that same post-filter count.

Why it violates the Batch 1 standard:

The target-symbol filter is a row drop, but the manifest does not record total raw rows, rows dropped by symbol, or whether the raw file contained non-target underlyings. Batch 1 requires every filter/drop rule to be logged and testable.

Minimal fix direction:

Count raw CSV rows before the target-symbol filter.

Persist raw_rows, target_symbol_rows, and dropped_non_target_symbol_rows per file.

If raw files are expected to contain only ^SPX, assert that rather than silently filtering.

Required tests:

Ingestion test with a mixed-symbol CSV proving non-target rows are counted in the manifest.

Ingestion test for an all-non-target file proving the failure message includes raw and target row counts.

B1-CODE-005

Severity: P2

File and line number(s):

scripts/02_build_option_panel.py, lines 90-104

src/ivsurf/cleaning/option_filters.py, lines 8-44

Evidence from code:

Stage 02 persists invalid_reason in the silver parquet, but the stage summary only records rows and valid_rows.

No per-date/per-reason invalid count is written to silver_build_summary.json.

Why it violates the Batch 1 standard:

Batch 1 requires filter/drop counts by date and reason. Persisting row-level reasons is useful, but the manifest does not provide the required audit trail of how many rows each rule removed on each date.

Minimal fix direction:

Add invalid_reason_counts to each stage-02 summary row.

Include counts for valid rows and each invalid reason code, including null-field reason codes after B1-CODE-003 is fixed.

Required tests:

Stage-02 integration test with multiple invalid reasons proving the manifest contains exact per-reason counts.

Test that manifest counts reconcile to total rows.

B1-CODE-006

Severity: P2

File and line number(s):

src/ivsurf/surfaces/interpolation.py, lines 9-13 and 15-37

scripts/03_build_surfaces.py, lines 103-118

src/ivsurf/evaluation/interpolation_sensitivity.py, lines 27-68

Evidence from code:

CompletedSurface stores only completed_total_variance and observed_mask.

_fill_axis() performs interpolation and boundary constant-fill extrapolation but returns no per-cell provenance.

Stage 03 writes completed_total_variance, completed_iv, and observed_mask, but no cell_provenance, interpolated_mask, extrapolated_mask, or completion-status field.

Why it violates the Batch 1 standard:

Completed-grid cells must be distinguishable as observed, interpolated, extrapolated, generated-only, or missing. The current artifacts can separate observed from non-observed, but cannot distinguish interpolation from extrapolation or other completion mechanisms.

Minimal fix direction:

Extend completion to return per-cell provenance/status.

Persist provenance in gold surfaces and propagate it to feature/evaluation artifacts.

At minimum, add completion_status with values like observed, interpolated, extrapolated_boundary_fill.

Required tests:

Unit tests proving interior missing cells are marked interpolated and outside-support cells are marked extrapolated.

Stage-03 integration test proving provenance columns are present and reconcile to total grid cell count.

Evaluation test proving interpolation and extrapolation metrics can be separated.

B1-CODE-007

Severity: P2

File and line number(s):

src/ivsurf/cleaning/derived_fields.py, lines 44-57

src/ivsurf/surfaces/grid.py, lines 10-23 and 30-52

scripts/03_build_surfaces.py, lines 103-118

src/ivsurf/evaluation/forecast_store.py, lines 68-93

Evidence from code:

log_moneyness is computed as log(strike) - log(active_underlying_price_1545), i.e. log spot moneyness.

SurfaceGrid and artifacts call the coordinate moneyness_points, not log_spot_moneyness_points.

Gold and forecast artifacts persist numeric moneyness_point values but no coordinate-system metadata.

Why it violates the Batch 1 standard:

Batch 1 requires surface coordinates to be declared in artifact metadata. Without explicit metadata, downstream users cannot tell whether moneyness_point=-0.10 means log spot moneyness, forward log-moneyness, normalized log-moneyness, delta, or another coordinate.

Minimal fix direction:

Add coordinate metadata such as moneyness_coordinate="log_spot_moneyness" and maturity_coordinate="ACT/365_years_from_effective_decision_to_last_tradable_close".

Rename artifact columns or add aliases to make log-moneyness explicit.

Persist grid/config hashes or grid version IDs with gold and forecast artifacts.

Required tests:

Schema test requiring coordinate metadata in gold surface artifacts.

Forecast-store test requiring coordinate metadata or grid-version metadata.

Regression test proving documentation/config and artifact coordinate labels agree.

B1-CODE-008

Severity: P2

File and line number(s):

src/ivsurf/evaluation/metrics.py, lines 8-11 and 19-24

src/ivsurf/evaluation/alignment.py, lines 146-155

Evidence from code:

_normalize_weights() returns uniform weights when total weight is non-positive.

weighted_mse() returns an unweighted full-vector mean when total weight is non-positive.

build_forecast_realization_panel() defines observed_weight as actual_vega_sum only where actual_observed_mask is true, otherwise zero.

Why it violates the Batch 1 standard:

If an observed-cell metric or slice has zero observed weight, the metric can silently degrade into a full-grid/unweighted completed-grid metric. Batch 1 requires observed-cell and completed-grid evaluation to remain separate, with empty observed slices explicitly skipped, errored, or labeled.

Minimal fix direction:

Add an explicit empty-weight policy.

For observed-cell metrics, raise or return null with an explicit reason when observed weight is zero.

Allow uniform fallback only for metrics explicitly named full-grid/uniform metrics.

Required tests:

Unit test proving observed-weight zero raises or returns null, not a full-grid mean.

Slice-metric test proving empty moneyness/maturity slices are labeled as empty rather than scored.

Regression test proving full-grid metrics and observed metrics use different weight policies.

B1-CODE-009

Severity: P2

File and line number(s):

src/ivsurf/evaluation/forecast_store.py, lines 30-95

src/ivsurf/evaluation/alignment.py, lines 115-137

Evidence from code:

write_forecasts() writes model_name, quote_date, target_date, grid indices, coordinates, and predicted_total_variance.

It does not write split_id, grid_version, surface_version, target_version, model_config_hash, training_run_id, or provenance/mask metadata.

build_forecast_realization_panel() joins forecasts to realized surfaces only by dates and grid indices.

Why it violates the Batch 1 standard:

Forecast artifacts are not self-describing enough to prove which split, surface version, grid, or completed-target construction they correspond to. Batch 1 requires completed-grid forecast claims to be reproducible and tied to the surface construction/version used.

Minimal fix direction:

Extend forecast artifacts with split_id, grid_version or grid hash, target_surface_version, surface_config_hash, model_config_hash, and training_run_id.

Carry target observed mask/provenance through evaluation or join it deterministically from gold artifacts with version checks.

Required tests:

Forecast-store schema test requiring split/grid/target metadata.

Alignment test that fails when forecast grid metadata does not match realized surface metadata.

Integration test proving forecast artifacts from two different grid configs cannot be accidentally combined.

B1-CODE-010

Severity: P2

File and line number(s):

src/ivsurf/schemas.py, lines 11-35

src/ivsurf/io/ingest_cboe.py, lines 54-66

src/ivsurf/calendar.py, lines 63-83

scripts/03_build_surfaces.py, lines 103-118

Evidence from code:

Raw schema uses quote_date and _1545 field names, but no row-level decision_datetime or availability_timestamp.

MarketCalendar.effective_decision_datetime() computes an effective snapshot time, including early-close adjustment, but this timestamp is not persisted into bronze, silver, gold, features, or forecasts in the pasted code.

Why it violates the Batch 1 standard:

Batch 1 requires 15:45 availability to be auditable at field/artifact level. The current code uses 15:45-named fields, but downstream artifacts lose the effective prediction timestamp needed to prove normal-session and early-close availability.

Minimal fix direction:

Add decision_datetime or effective_decision_timestamp to silver/gold/features, or persist it in mandatory sidecar metadata keyed by date.

Add a field-availability manifest listing every feature/input column and its maximum allowed availability time.

Ensure early-close sessions persist the effective pre-close decision timestamp, not only the date.

Required tests:

Stage-02 or stage-03 test proving decision_datetime is present and equals 15:45 on normal sessions.

Early-close integration test proving decision_datetime is the configured close-minus-buffer time.

Schema test rejecting feature columns without declared availability metadata.

CONTEXT_GAPS

src/ivsurf/evaluation/loss_panels.py is needed to confirm exactly how observed, full-grid, slice, interpolation, and extrapolation metrics are computed.

src/ivsurf/features/liquidity.py is needed to audit whether liquidity features are strictly 15:45-safe and mask-aware.

src/ivsurf/training/model_factory.py, src/ivsurf/training/fit_sklearn.py, src/ivsurf/training/fit_torch.py, and src/ivsurf/training/tuning.py are needed to confirm split-safe preprocessing, HPO discipline, feature-column selection, and whether HAR/factor models receive the intended columns.

src/ivsurf/models/ridge.py, src/ivsurf/models/elasticnet.py, and src/ivsurf/models/positive_target.py are needed to audit target transformations and benchmark behavior.

configs/data/raw.yaml, configs/data/cleaning.yaml, and configs/eval/stats_tests.yaml are needed to confirm the actual configured sample window, filter thresholds, and official metric set used in runs.

Stage 05/06 training and forecasting scripts are needed to confirm forecast artifacts are tied to split IDs and model-config hashes somewhere outside forecast_store.py.

Report/table-generation code is needed to confirm whether documentation ever says “arbitrage-free” instead of “arbitrage-aware.”

Existing run manifests and sample generated gold/forecast artifacts are needed to verify whether missing metadata is supplied through sidecars rather than columns.

FIX_ORDER

Fix B1-CODE-001 first because it can make the completed-grid target depend on cells that are explicitly marked unobserved.

Fix B1-CODE-002 and B1-CODE-006 together because grid-domain handling and cell provenance should be designed as one surface-construction contract.

Fix B1-CODE-007 and B1-CODE-010 so all downstream artifacts carry coordinate and timestamp metadata.

Fix B1-CODE-003, B1-CODE-004, and B1-CODE-005 to close raw/silver filtering and logging gaps.

Fix B1-CODE-008 before relying on observed-cell headline metrics.

Fix B1-CODE-009 before producing final forecast comparisons or reusable forecast artifacts.

TEST_COMMANDS_TO_RUN
Bash
pytest tests/unit/test_option_filters.py \
       tests/unit/test_schema.py \
       tests/unit/test_calendar.py \
       tests/unit/test_interpolation.py \
       tests/regression/test_observed_mask_preservation.py
Bash
pytest tests/unit/test_metrics.py \
       tests/unit/test_alignment.py \
       tests/unit/test_forecast_store.py \
       tests/unit/test_feature_dataset.py
Bash
pytest tests/integration/test_stage03_stage04_target_gap_alignment.py \
       tests/integration/test_stage04_early_close_alignment.py

After adding the required new tests, run focused new test files such as:

Bash
pytest tests/unit/test_grid_domain_assignment.py \
       tests/unit/test_surface_provenance.py \
       tests/unit/test_surface_metadata.py \
       tests/integration/test_stage01_ingestion_manifest_counts.py \
       tests/integration/test_stage02_filter_reason_manifest.py \
       tests/integration/test_stage03_completion_uses_observed_mask.py

Then run the full relevant suite:

Bash
pytest tests/unit tests/property tests/regression tests/integration

Extended Pro
ChatGPT is AI and can make mistakes. Check important info.
