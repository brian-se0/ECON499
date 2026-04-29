# ECON499 SPX IV Surface Audit: Round 005C Re-Verification of B3 Follow-up Fixes

You are GPT 5.5 Pro acting as the independent audit brain. Codex is the implementation agent.

Attached is a compact current-code verification package for the ECON499 SPX implied-volatility-surface forecasting project after Codex implemented the Round 005B follow-up fixes.

Hard project rules remain in force:
- Python 3.13.5 only.
- No fallback paths.
- No backwards compatibility or legacy branches.
- Use hard cutovers and fail-fast contracts.
- No silent type/date coercion, no silent row drops, no hidden clipping, no inferred fallback grids, and no future-aware feature generation.
- Maintain leak-free temporal integrity and research-grade reproducibility.

Round 005B Pro decision:
- `B3-CODE-001: fixed`
- `B3-CODE-002: fixed`
- `B3-CODE-003: still_open`
- `B3-CODE-004: still_open`

Open follow-up findings to re-verify:

1. `B3-CODE-003A`
   - `build_forecast_realization_panel` must validate raw `actual_vega_sum` before deriving `observed_weight`.
   - Null, nonfinite, and negative `actual_vega_sum` must fail fast on every aligned actual cell, including cells where `actual_observed_mask == False`.
   - Observed cells must have strictly positive `actual_vega_sum`.
   - Do not require zero `actual_vega_sum` on unobserved cells.

2. `B3-CODE-004A`
   - Saved-artifact and hedging reshape boundaries must not infer the expected grid from the artifact being validated.
   - Stage 07, Stage 08, and Stage 09 should load the configured `SurfaceGrid` from `configs/data/surface.yaml`.
   - Artifact metadata must match `surface_grid_hash == grid.grid_hash`.
   - Complete unique grid validation must use the configured grid at actual/forecast loaders and Stage 08 hedging boundaries.
   - `surface_interpolator_from_frame` should require an explicit expected `SurfaceGrid` or be called only after explicit expected-grid validation. No inferred fallback grid is allowed.

Codex reports the follow-up evidence in:
- `research/consults/gpt55-pro-audit/rounds/round-005b-verify-b3-fixes-response.md`
- `research/consults/gpt55-pro-audit/rounds/round-005b-verify-b3-fixes-test-evidence.md`
- `research/consults/gpt55-pro-audit/backlog.md`

Your task:

1. Verify only whether Codex fully fixed `B3-CODE-003A` and `B3-CODE-004A`.
2. Use the attached code and tests. Do not rely on Codex summaries without checking the relevant files.
3. If either fix is incomplete, return precise actionable findings with file/function references and exact expected behavior.
4. If both follow-up fixes are complete, say so explicitly.
5. Do not perform a new full-project closure audit in this response. If both follow-up fixes are verified, the next step will be a fresh full-project audit round.

Required response format:

```text
ROUND_005C_REVERIFICATION_DECISION: all_fixed | still_open

B3-CODE-003A: fixed | still_open
B3-CODE-004A: fixed | still_open

EVIDENCE_REVIEWED:
- ...

OPEN_FINDINGS:
- Finding ID, severity, evidence, and exact Codex action required.

NEXT_CODEX_ACTIONS:
- ...
```

Only use `ROUND_005C_REVERIFICATION_DECISION: all_fixed` if there are no known unresolved actionable issues in `B3-CODE-003A` or `B3-CODE-004A` under this targeted re-verification scope.
