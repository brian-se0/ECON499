# Round 003I Pro Response - B1-CODE-009 Surface Config Hash Reverification

Captured: 2026-04-27

VERIFICATION_DECISION: B1-CODE-009: fixed

REASONING:

Stage 03 now persists surface_config_hash into both gold parquet rows and gold_surface_summary.json, closing the prior gap where realized surfaces lacked the construction-config identity.

Stage 04 now carries surface_config_hash from gold into daily_features.parquet, and rejects multiple hashes in one daily feature frame, preventing mixed surface-construction regimes from silently entering one feature artifact.

Stage 06 now reads the feature-carried surface_config_hash, compares it to the current surface.yaml hash, and writes that validated hash into forecast artifacts. This prevents forecasts from being generated from stale feature artifacts under a different surface config.

Forecast artifacts now contain the required self-description fields: split_id, grid metadata, target_surface_version, surface_config_hash, model_config_hash, training_run_id, and timestamp metadata.

Evaluation alignment now loads gold surface_config_hash and rejects mismatches between forecast, target-day gold, and origin-day gold hashes. It also fails clearly for old forecast artifacts missing surface_config_hash.

The added tests directly cover the Round 003H residual gap: Stage 03/04 propagation, alignment rejection on same-grid/different-surface-config mismatch, and missing old-artifact metadata failure.

The focused verification suite passed: 34 tests passed, and ruff passed on the touched modules and tests.

REMAINING_REQUIRED_FIXES:

None.

Extended Pro
ChatGPT is AI and can make mistakes. Check important info.
