# Round 007B Targeted Verification Prompt

You are GPT 5.5 Pro acting as the audit brain for the ECON499 SPX implied-volatility-surface forecasting project. Codex in the app is the implementation agent.

This is a targeted verification round for the only actionable finding from Round 007.

## Non-Negotiable Project Rules

- Fail fast. Do not approve fallback paths.
- No backwards compatibility or legacy branches.
- Preserve temporal integrity and leak-free evaluation.
- Do not accept silent coercion, silent row drops, or silent default drift.
- Official smoke must exercise the intended smoke configuration, not default production grid settings by accident.

## Finding Under Verification

`B5-CODE-001`

Severity: `P2`

Area: official smoke / runtime preflight / configured-grid evaluation boundary

Round 007 issue: `scripts/official_smoke.py` passed `surface_config_path=smoke_surface_config_path` into Stages 03, 04, 05, 06, and 09, but not into Stage 07 or Stage 08. Because `scripts/07_run_stats.py` and `scripts/08_run_hedging_eval.py` have defaults of `configs/data/surface.yaml`, the official smoke could validate stats/hedging/report behavior against the wrong grid.

## Files To Inspect

- `scripts/official_smoke.py`
- `tests/e2e/test_official_smoke_runtime.py`
- `scripts/07_run_stats.py`
- `scripts/08_run_hedging_eval.py`
- `configs/official_smoke/data/surface.yaml`
- `configs/data/surface.yaml`
- `research/consults/gpt55-pro-audit/rounds/round-007-fresh-full-project-closure-audit-response.md`
- `research/consults/gpt55-pro-audit/rounds/round-007b-b5-code-001-fix-evidence.md`
- `research/consults/gpt55-pro-audit/backlog.md`

## Your Task

1. Verify whether `B5-CODE-001` is fully fixed.
2. Verify whether the new test is deterministic, non-skipping for the B5 contract, and would have failed against the original bug.
3. Verify whether the tests executed by Codex are relevant and sufficient for this targeted finding.
4. Identify any directly related regression introduced by this B5 patch, if one exists.
5. State the next required workflow step after this targeted verification.

Do not perform a broad new full-project audit in this targeted round. If `B5-CODE-001` is fixed, say so, then specify whether the next step should be another fresh full-project closure audit with refreshed code context before any terminal closure statement.

## Required Output Format

Use this exact decision line:

```text
ROUND_007B_VERIFICATION_DECISION: all_fixed | still_open | needs_more_context
```

Then include:

- `B5-CODE-001:` fixed/open/needs-more-context with concise rationale.
- `Test assessment:` concise assessment.
- `Directly related regressions:` none or list.
- `NEXT_REQUIRED_STEP:` one explicit next action for Codex.
