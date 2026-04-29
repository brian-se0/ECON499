# Round 006C Targeted B4 Re-Verification After Stalled Round 006B

You are GPT 5.5 Pro acting as the audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

The previous Round 006B targeted verification attempt stalled during a broad `python -S -m compileall -q src scripts tests` command and was stopped after an extended wait. A follow-up constrained to no tools correctly returned `blocked` because it could not reconstruct the interrupted inspection notes. Treat that as a procedural block, not as evidence that any B4 fix is still open.

Use the already uploaded archive in this conversation:

- `econ499_round006b_b4_fix_context_20260428.zip`

The archive contains the current B4 fix context: `AGENTS.md`, `pyproject.toml`, configs, scripts, `src/ivsurf/`, tests, backlog, Round 006 Pro response, Round 006 B4 fix evidence, and the original Round 006B verification prompt.

## Scope

Re-run the targeted verification for only these four B4 findings:

1. `B4-CODE-001`: split manifests must be semantically revalidated before Stage 05/06 training/evaluation.
2. `B4-CODE-002`: `SurfaceGridConfig` and `SurfaceGrid` coordinate validation must fail fast.
3. `B4-CODE-003`: hedging surface interpolation must fail out-of-domain instead of clipping, with Stage 08 prevalidation.
4. `B4-CODE-004`: delta-vega hedge sizing must fail on zero, non-finite, or below-floor hedge vega instead of falling back to delta-only.

Do not perform a fresh full-project closure audit in this response.

## Verification Guardrails

- Do inspect the uploaded archive contents, changed code, targeted tests, and included test evidence.
- Do not run broad compileall commands over the entire archive if they risk stalling.
- Do not block solely because a broad supplemental compile command was skipped, timed out, or failed in the ChatGPT tool environment.
- If the ChatGPT environment lacks dependencies or a command times out, record that as a supplemental check limitation and continue from static inspection plus the archived ruff/pytest evidence.
- If you run commands, prefer small targeted reads/searches/tests that directly verify the four B4 fixes.
- Return `blocked` only if the archive itself is inaccessible or the evidence needed to decide a B4 item is genuinely missing after targeted inspection.
- Return `still_open` only for a concrete unresolved actionable defect, with code/test evidence and a required fix.
- Return `all_fixed` only if no known unresolved actionable B4 issues remain under this targeted verification scope.

Project owner constraints remain binding:

- No fallback paths.
- Fail fast.
- No backwards compatibility or legacy branches.
- Python 3.13.5 only.
- No hidden leakage.
- Tests are mandatory for timing, features, surfaces, loss definitions, and evaluation changes.

## Required Response Format

```text
ROUND_006C_VERIFICATION_DECISION: all_fixed | still_open | blocked

ATTACHMENT_CHECK:
- econ499_round006b_b4_fix_context_20260428.zip: accessible | inaccessible | missing

B4-CODE-001: fixed | still_open | blocked
B4-CODE-002: fixed | still_open | blocked
B4-CODE-003: fixed | still_open | blocked
B4-CODE-004: fixed | still_open | blocked

METHOD:
- Files/modules inspected:
- Tests/contracts reviewed:
- Evidence reviewed:
- Supplemental command limitations:

OPEN_FINDINGS:
- None, or for each still-open issue:
  Finding ID:
  Severity:
  Evidence:
  Required fix:
  Required tests:
  Hard-cutover/fail-fast note:

NEXT_CODEX_ACTIONS:
- If all fixed: "Prepare next fresh closure audit package."
- If still open or blocked: ordered implementation or evidence-collection checklist.
```
