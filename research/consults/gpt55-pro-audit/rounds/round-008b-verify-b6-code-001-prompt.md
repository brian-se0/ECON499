# Round 008B Targeted Verification Prompt

You are GPT 5.5 Pro acting as the audit brain for the ECON499 SPX implied-volatility-surface forecasting project. Codex in the app is the implementation agent.

This is a targeted verification round for the only actionable finding from Round 008.

## Non-Negotiable Project Rules

- Fail fast. Do not approve fallback paths.
- No backwards compatibility or legacy branches.
- Preserve temporal integrity and leak-free evaluation.
- Do not accept silent penalty disabling, silent zero substitution, or silent non-finite losses.
- Official smoke must exercise a mathematically valid configuration for enabled neural penalties.

## Finding Under Verification

`B6-CODE-001`

Severity: `P1`

Area: neural loss/penalty definitions; official smoke workflow; fail-fast runtime validation

Round 008 issue: the official-smoke 2x2 grid reaches the enabled neural roughness penalty path. A second-difference roughness term on a 2-point axis creates empty curvature tensors, and means over those tensors can produce non-finite loss components instead of a clear fail-fast configuration error.

## Files To Inspect

- `src/ivsurf/models/penalties.py`
- `src/ivsurf/models/neural_surface.py`
- `configs/official_smoke/data/surface.yaml`
- `scripts/official_smoke.py`
- `tests/unit/test_penalties.py`
- `tests/unit/test_training_behaviour.py`
- `tests/unit/test_repository_contract.py`
- `tests/e2e/test_official_smoke_runtime.py`
- `tests/integration/test_synthetic_cpu_stage01_to_stage09_pipeline.py`
- `research/consults/gpt55-pro-audit/rounds/round-008-fresh-full-project-closure-audit-response.md`
- `research/consults/gpt55-pro-audit/rounds/round-008b-b6-code-001-fix-evidence.md`
- `research/consults/gpt55-pro-audit/backlog.md`

## Your Task

1. Verify whether `B6-CODE-001` is fully fixed.
2. Verify that enabled neural penalties now fail fast when the configured grid is mathematically invalid.
3. Verify that the official-smoke grid is now valid for the enabled neural penalties.
4. Verify that the tests would have failed against the original 2x2 roughness-penalty bug.
5. Identify any directly related regression introduced by this B6 patch, if one exists.
6. State the next required workflow step after this targeted verification.

Do not perform a broad new full-project audit in this targeted round. If `B6-CODE-001` is fixed, say so, then specify whether the next step should be another fresh full-project closure audit with refreshed code context before any terminal closure statement.

## Required Output Format

Use this exact decision line:

```text
ROUND_008B_VERIFICATION_DECISION: all_fixed | still_open | needs_more_context
```

Then include:

- `B6-CODE-001:` fixed/open/needs-more-context with concise rationale.
- `Test assessment:` concise assessment.
- `Directly related regressions:` none or list.
- `NEXT_REQUIRED_STEP:` one explicit next action for Codex.
