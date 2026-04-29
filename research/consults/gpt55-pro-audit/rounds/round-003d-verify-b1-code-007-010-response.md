# Round 003D: Pro Verification of B1-CODE-007 and B1-CODE-010

Captured: 2026-04-27

VERIFICATION_DECISION

B1-CODE-007: fixed
B1-CODE-010: fixed

EVIDENCE

B1-CODE-007 is fixed because the implementation now defines an explicit coordinate contract: MONEYNESS_COORDINATE = "log_spot_moneyness", MATURITY_COORDINATE = "ACT/365_years_from_effective_decision_to_last_tradable_close", SURFACE_GRID_SCHEMA_VERSION = "surface_grid_v1", deterministic SurfaceGrid.metadata, and SurfaceGrid.grid_hash.

B1-CODE-007 is also fixed downstream: Stage 03 gold surfaces, Stage 04 feature rows, Stage 06 forecast rows, and evaluation alignment now carry or validate surface_grid_schema_version, surface_grid_hash, maturity_coordinate, and moneyness_coordinate.

require_surface_grid_metadata(frame, grid, dataset_name=...) addresses the original false-assumption risk by failing if persisted artifact metadata is missing or mismatched against the configured grid.

Forecast alignment now rejects grid hash, schema, or coordinate-label mismatches between forecast rows, target gold surfaces, and origin gold surfaces. That directly addresses the risk that forecasts and realized surfaces could be joined under incompatible coordinate definitions.

B1-CODE-010 is fixed because effective_decision_timestamp is now created in Stage 02 silver artifacts from MarketCalendar.effective_decision_datetime(), persisted to gold, features, and forecasts, and checked during evaluation alignment.

Early-close behavior is specifically covered: tests/integration/test_early_close_stage02.py verifies the 2019-11-29 early-close silver timestamp is 2019-11-29T12:45:00-05:00, and tests/integration/test_stage04_early_close_alignment.py verifies pre-Thanksgiving features target the early-close surface with that target timestamp.

The feature availability manifest closes the remaining B1-CODE-010 requirement: build_feature_availability_manifest(feature_frame) declares daily-feature columns and fails if any column lacks declared availability metadata.

Local verification is credible for this slice: ruff passed on the touched modules and scripts, and the focused pytest set passed with 28 tests, including grid metadata, forecast store metadata, feature dataset metadata, alignment rejection tests, early-close timestamp tests, clean walk-forward forecast emission, neural imputed-cell supervision, and smoke pipeline coverage.

REQUIRED_ADJUSTMENTS

none

NEW_REGRESSION_RISKS

Low-priority hardening risk: ISO-8601 timestamps are persisted as strings. That is acceptable for auditability, but downstream code should continue treating them as metadata identifiers unless typed timestamp arithmetic is explicitly needed.

Low-priority brittleness risk: surface_grid_hash will intentionally change whenever coordinate labels, schema version, maturity days, or moneyness points change. This is desirable for safety, but migration/reuse workflows should surface a clear error message rather than silently mixing old and new artifacts.

Low-priority false-assurance risk: the availability manifest classifies columns, but later audit should confirm the manifest records enough detail to distinguish origin-time features from target columns and that training code never includes target/timestamp/metadata columns as model features. This belongs with the training/model-matrix audit, not this fix slice.

No evidence in the provided summary suggests this slice reintroduced B1-CODE-001, B1-CODE-002, or B1-CODE-006.

NEXT_FIX_SLICE

Move to B1-CODE-003, B1-CODE-004, and B1-CODE-005 together.

Reason: after coordinate and timestamp metadata are fixed, the next Batch 1 risk cluster is raw/silver data quality and auditability: null handling for critical 15:45 fields, target-symbol filtering counts, and per-date/per-reason cleaning summaries. These three findings share the same ingestion-to-silver manifest contract and should be implemented as one coherent filter/drop-reason slice.

Extended Pro
ChatGPT is AI and can make mistakes. Check important info.
