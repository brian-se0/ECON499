ROUND_007B_VERIFICATION_DECISION: all_fixed

B5-CODE-001: fixed. scripts/official_smoke.py now passes surface_config_path=smoke_surface_config_path into both stage07.main(...) and stage08.main(...), while Stages 07 and 08 still retain their standalone production defaults. This fixes the official-smoke mixed-grid defect without adding fallback behavior.

Test assessment: sufficient for the targeted finding. The new test_official_smoke_passes_smoke_surface_config_to_grid_dependent_stages is deterministic and does not skip behind the Windows/CUDA runtime preflight because it monkeypatches run_runtime_preflight and stage loading. It asserts that Stages 03, 04, 05, 06, 07, 08, and 09 receive the official-smoke surface config, and that Stages 01 and 02 do not. Against the original bug, it would have failed when Stage 07/08 calls lacked surface_config_path. Codex's executed ruff, official-smoke e2e contract, workflow, stats, hedging, report, and regression tests are relevant and sufficient for this targeted verification. The real Windows/CUDA official smoke was still not executed locally, but the non-skipping contract test covers the B5 defect directly.

Directly related regressions: none found.

NEXT_REQUIRED_STEP: Codex should prepare a refreshed full-project closure-audit package, with B5-CODE-001 recorded as Pro-verified fixed, for another fresh full-project closure audit before any terminal closure statement.
