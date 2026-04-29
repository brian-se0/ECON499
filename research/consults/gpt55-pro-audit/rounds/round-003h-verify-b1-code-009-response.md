# Round 003H: Pro Verification of B1-CODE-009

Captured: 2026-04-27

VERIFICATION_DECISION

B1-CODE-009: partially fixed

EVIDENCE

The fix addresses most of the original artifact self-description gap:

forecast rows now persist split_id, grid metadata, target_surface_version, surface_config_hash, model_config_hash, and training_run_id;

Stage 06 now propagates one split ID per forecast row;

forecast writing validates non-empty split/hash/run identifiers;

Stage 03 gold artifacts now carry target_surface_version;

alignment now requires non-null split/model/run/surface-version metadata and rejects target-surface-version mismatches;

tests verify forecast metadata persistence, grid metadata mismatch rejection, Stage 03 target_surface_version, and Stage 06 clean forecast emission.

The remaining gap is that surface_config_hash appears to be written only to forecast artifacts and Stage 06 metadata, not to gold surface artifacts. Alignment is reported to compare grid metadata and target_surface_version, but not forecast surface_config_hash against the realized gold surface’s construction-config hash.

Because target_surface_version = "completed_surface_v1" is a schema/version label, not a deterministic hash of the actual surface-construction configuration, two gold surfaces built with the same grid and same schema version but different interpolation order, interpolation cycles, total-variance floor, or observed-cell threshold could still share the same target_surface_version and surface_grid_hash.

Therefore, the implementation materially improves reproducibility metadata, but it does not yet fully prove that forecast rows and realized surfaces were built from the same surface-construction config.

REQUIRED_ADJUSTMENTS

Persist surface_config_hash into Stage 03 gold parquet rows and gold_surface_summary.json.

Carry surface_config_hash into daily_features.parquet from the gold artifact, rather than relying only on Stage 06 recomputation from the current config file.

Update load_actual_surface_frame() to load gold surface_config_hash.

Update build_forecast_realization_panel() to reject mismatches between forecast surface_config_hash, target-day gold surface_config_hash, and origin-day gold surface_config_hash.

Add tests for the remaining mismatch class:

a unit alignment test where forecast and gold share grid hash/schema/coordinate labels and target_surface_version, but have different surface_config_hash, and alignment must fail;

a Stage 03/04 integration assertion that gold and feature artifacts carry surface_config_hash;

a forecast-store/alignment regression test proving old forecast artifacts missing surface_config_hash fail clearly.

NEW_REGRESSION_RISKS

Low brittleness risk: deterministic hashes over config files and JSON payloads are good for safety, but small irrelevant formatting/config-order changes can invalidate reuse unless hashing is canonicalized. The summary suggests model config hashing uses deterministic JSON; sha256_file(surface_config_path) is stricter and may be intentionally brittle.

Low false-assurance risk: training_run_id is deterministic rather than a unique runtime UUID. That is acceptable if it is intended as a reproducible identity hash, but reports should not imply it uniquely identifies a wall-clock execution unless run manifests also provide timestamps and artifact hashes.

No leakage risk is evident from this slice.

No regression to B1-CODE-001 through B1-CODE-008 or B1-CODE-010 is evident from the provided summary.

NEXT_FIX_SLICE

Complete the B1-CODE-009 adjustment above before broader audit rounds.

After that, Batch 1 code findings can be treated as closed under the reviewed scope, and the next audit round should move to Batch 2: static-arbitrage diagnostics, total-variance modeling, and whether neural outputs may be described only as arbitrage-aware unless hard no-arbitrage constraints are proven.

Extended Pro
ChatGPT is AI and can make mistakes. Check important info.
