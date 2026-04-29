# Round 003B Verify First Fix Slice Response

Captured: 2026-04-27

VERIFICATION_DECISION

B1-CODE-001: fixed.
Evidence: complete_surface() now forces completed[~normalized_observed_mask] = np.nan before interpolation, so mask-false finite values cannot become interpolation anchors. The new unit test test_surface_completion_ignores_finite_values_without_observed_mask() verifies a finite 999.0 in a mask-false cell is ignored and replaced by the interpolated value.

B1-CODE-002: fixed.
Evidence: assign_grid_indices() now computes inside_maturity_grid_domain, inside_moneyness_grid_domain, inside_grid_domain, and grid_domain_reason; out-of-domain rows receive null grid indices instead of edge buckets. Stage 03 filters to inside_grid_domain and records rows_inside_grid_domain, rows_outside_grid_domain, and grid_domain_reason_counts. The new unit/integration tests cover out-of-grid maturity/moneyness handling and manifest counts.

B1-CODE-006: partially fixed.
Evidence: CompletedSurface now carries completion_status, complete_surface() distinguishes observed, interpolated, and extrapolated_boundary_fill, and Stage 03 persists completion_status plus completion_status_counts. However, the pasted changes do not propagate this provenance into evaluation/alignment/loss artifacts, and the Stage 03 integration test only verifies a one-cell observed surface, not persisted interpolated/extrapolated statuses in a gold artifact.

REQUIRED_ADJUSTMENTS

Finish B1-CODE-006 propagation:

Add completion_status to the actual-surface load path used by evaluation, especially load_actual_surface_frame() and the forecast-realization panel.

Ensure daily loss panels can separate at least observed, interpolated, and extrapolated cells, or explicitly expose those masks for reporting.

Add a Stage 03 integration test with a multi-cell grid where the persisted gold parquet contains all three statuses: observed, interpolated, and extrapolated_boundary_fill.

Add a reconciliation assertion that completion_status_counts sums to total grid cells and agrees with the gold artifact.

Add an evaluation/loss-panel test proving interpolation and extrapolation metrics or masks remain separable downstream.

NEW_REGRESSION_RISKS

The stricter grid-domain filter can reduce coverage or skip entire dates if the configured grid domain is narrower than the cleaned option universe. This is intended, but manifests and reports should monitor rows_outside_grid_domain and skipped_no_rows_inside_grid_domain.

completion_status currently reflects the sequential fill path. Later interpolation cycles can use values filled in earlier axes as anchors, so interpolated may mean “filled by the completion algorithm,” not necessarily “directly interpolated between originally observed cells.” That semantic should be documented or refined if downstream metrics require direct-observed interpolation status.

Downstream code that explicitly selects gold-surface columns may silently drop completion_status until B1-CODE-006 propagation is completed.

NEXT_FIX_SLICE

Finish the residual B1-CODE-006 downstream propagation first, because provenance is now created at Stage 03 but may be lost before evaluation.

Then implement B1-CODE-007 and B1-CODE-010 together. They are the next highest-leverage metadata fixes: coordinate-system metadata and effective 15:45 decision timestamp metadata should travel through the same surface/feature/forecast artifact contract.

Extended Pro
ChatGPT is AI and can make mistakes. Check important info.
