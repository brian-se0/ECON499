ROUND_006C_VERIFICATION_DECISION: still_open

ATTACHMENT_CHECK:

econ499_round006b_b4_fix_context_20260428.zip: accessible

B4-CODE-001: fixed
B4-CODE-002: still_open
B4-CODE-003: still_open
B4-CODE-004: fixed

METHOD:

Files/modules inspected:

AGENTS.md

pyproject.toml

research/consults/gpt55-pro-audit/backlog.md

research/consults/gpt55-pro-audit/rounds/round-006-fresh-full-project-closure-audit-response.md

research/consults/gpt55-pro-audit/rounds/round-006-b4-fixes-test-evidence.md

src/ivsurf/splits/manifests.py

src/ivsurf/splits/walkforward.py

scripts/05_tune_models.py

scripts/06_run_walkforward.py

src/ivsurf/config.py

src/ivsurf/surfaces/grid_validation.py

src/ivsurf/surfaces/grid.py

src/ivsurf/hedging/revaluation.py

src/ivsurf/hedging/validation.py

src/ivsurf/hedging/hedge_rules.py

src/ivsurf/hedging/pnl.py

scripts/08_run_hedging_eval.py

Tests/contracts reviewed:

tests/unit/test_split_manifests.py

tests/unit/test_grid.py

tests/property/test_surface_grid.py

tests/unit/test_hedging.py

tests/integration/test_stage05_stage06_clean_evaluation.py

tests/integration/test_stats_hedging_slice.py

AGENTS.md fail-fast, no-fallback, no-legacy-branch, Python 3.13.5, and mandatory-test requirements.

Evidence reviewed:

Archived Round 006 B4 fix evidence reports targeted ruff passed, B4/adjacent pytest command with 50 passed, and adjacent artifact-contract pytest command with 46 passed.

Targeted local AST parse passed for the B4-relevant source, script, and test files inspected.

Targeted local pytest for tests/unit/test_split_manifests.py passed: 12 passed.

A small local custom check confirmed SurfaceGridConfig rejects the invalid coordinate payloads covered by tests/unit/test_grid.py.

Static inspection confirms B4-CODE-001 is wired into Stage 05 and Stage 06: require_split_manifest_matches_artifacts() now calls semantic validation, and semantic validation checks ordered date-universe membership, duplicate-free/disjoint/chronological arrays, split IDs, validated walk-forward config, and exact deterministic equality to build_walkforward_splits(...).

Static inspection confirms B4-CODE-004 removed the delta-only fallback: compute_delta_vega_hedge() now raises InfeasibleHedgeError for non-finite, non-positive-floor, zero, or below-floor hedge straddle vega, and Stage 08 passes hedge_vega_floor through the evaluation path.

Supplemental command limitations:

Broad compileall over src scripts tests was intentionally not rerun, per Round 006C guardrails and the prior stall.

The ChatGPT tool environment lacks polars and hypothesis, so I did not rerun the grid, hedging, property, or integration pytest files locally. Those were reviewed statically and against archived ruff/pytest evidence.

OPEN_FINDINGS:

Finding ID: B4-CODE-002
Severity: P2
Evidence: SurfaceGridConfig now rejects bool-like and non-numeric moneyness inputs in its pre-validator, but the shared runtime validator used by SurfaceGrid.__post_init__() still silently accepts and normalizes bool/string-like moneyness values. In src/ivsurf/surfaces/grid_validation.py, validate_moneyness_points() immediately does tuple(float(value) for value in values) before type rejection. Because src/ivsurf/surfaces/grid.py calls only validate_moneyness_points(self.moneyness_points) in SurfaceGrid.__post_init__(), programmatic SurfaceGrid construction can still bypass the stricter config validator. A targeted local check confirmed validate_moneyness_points((False, True)) -> (0.0, 1.0) and validate_moneyness_points(("0.0", "0.1")) -> (0.0, 0.1).
Required fix: Make validate_moneyness_points() fail fast before float conversion unless each value is an int or float and not bool. Ideally return normalized floats and ensure SurfaceGrid cannot retain string/bool-like coordinate payloads, either by rejecting them or by controlled explicit normalization after validation.
Required tests: Add tests proving both SurfaceGridConfig and direct SurfaceGrid construction reject bool-like moneyness values, string moneyness values, object-like numeric strings, duplicate values created by bool coercion, and non-finite numeric values. Keep positive tests for valid integer/float numeric moneyness coordinates.
Hard-cutover/fail-fast note: Do not silently coerce string or bool-like moneyness coordinates. Reject them with a clear error.

Finding ID: B4-CODE-003
Severity: P2
Evidence: SurfaceInterpolator.sigma() itself now validates domain and uses RegularGridInterpolator(bounds_error=True), but value_instrument() in src/ivsurf/hedging/revaluation.py still calls surface.sigma(max(remaining_days, 1), log_moneyness, ...). This silently clips nonpositive remaining maturity to one day before the domain check. Stage 08 prevalidation in src/ivsurf/hedging/validation.py mirrors the same pattern with target_remaining_days = float(max(maturity_days - max_target_gap_days, 1)), so a target gap that exhausts or exceeds a book/hedge maturity can pass prevalidation if the grid contains a one-day point. The direct surface.sigma(0, ...) unit test does not cover this bypass through value_instrument().
Required fix: Remove the max(remaining_days, 1) clipping path. Pass the actual remaining maturity into domain validation, or fail before surface lookup when remaining maturity is not strictly inside the configured surface domain. Stage 08 prevalidation should use actual maturity_days - max_target_gap_days and fail if any generated book or hedge instrument can reach nonpositive or below-grid remaining maturity.
Required tests: Add unit tests where value_instrument() and evaluate_model_hedging() receive an expired or below-grid remaining maturity and raise SurfaceDomainError or a typed hedging-domain error before valuation. Add Stage 08 prevalidation tests where max_target_gap_days >= short_maturity_days and max_target_gap_days >= hedge_maturity_days fail even when the grid includes one-day maturity. Add a positive official-config test that remains in-domain.
Hard-cutover/fail-fast note: Do not keep the one-day clipping behavior as a compatibility mode. If expired-option valuation is desired, define it as a separate explicit evaluation design with its own schema/config contract, not as a hidden surface-interpolation fallback.

NEXT_CODEX_ACTIONS:

Fix validate_moneyness_points() so direct SurfaceGrid construction rejects bool-like and string-like moneyness coordinates fail-fast, then add direct SurfaceGrid tests for those cases.

Remove remaining-maturity clipping from value_instrument() and Stage 08 hedging prevalidation, then add tests proving expired/below-grid maturities fail before hedging result artifacts are written.

Rerun the targeted B4 ruff/pytest commands and update the B4 fix evidence.