# GPT 5.5 Pro Audit Protocol

Date initialized: 2026-04-27
Workspace: `/Volumes/T9/ECON499`
Remote: `https://github.com/brian-se0/ECON499.git`
Baseline commit: `96e0da6fe741b63d2d4743aaab1ea6b5dde4c468`
Frozen archive noted by user: `/Volumes/T9/ECON499_audit_hpo_30_trials__train_30_epochs__mac_cpu_20260426_under512MB.zip`
Frozen archive SHA256: `0f3612fa75a35c3ca7ac0487537187e0409918cd4dd3a7b37a2fbe6d8f7eb096`

## Roles

- GPT 5.5 Pro is the audit brain. It reviews literature, code summaries, evidence packets, and patch-review packets, then issues written findings and accept/revise/withdraw judgments.
- Codex App is the implementation and verification agent. It gathers repo evidence, sends bounded prompts to GPT 5.5 Pro through the user-authenticated Codex app browser session, validates each finding locally, implements accepted fixes, runs tests, and records decisions.
- The repository is the source of truth. Browser chat content is advisory until it is saved here and validated against local code, tests, data contracts, and `AGENTS.md`.

## Scope

The audit covers research correctness, leakage control, data semantics, schemas, sample-window enforcement, feature availability timestamps, surface construction, observed-cell masks, completed surfaces, split manifests, model training, loss definitions, evaluation metrics, statistical tests, hedging evaluation, reproducibility, memory safety, and performance.

The local literature set in `/Volumes/T9/ECON499/lit_review` must be consulted before GPT 5.5 Pro recommends additional literature. Additional peer-reviewed literature searches are allowed only when the local dossier leaves a concrete gap.

## Anti-Loop Rules

This workflow intentionally avoids arbitrary limits on audit rounds. Instead, it prevents loops by requiring every Pro response to be stateful, evidence-based, and classifiable.

Every Pro item must be classified as one of:

- `NEW_FINDING`
- `REVISION_TO_EXISTING_FINDING`
- `DUPLICATE`
- `NON_ACTIONABLE`
- `RESIDUAL_RISK`
- `ACCEPTED_FIXED`
- `WITHDRAWN`

Codex rejects or closes Pro items that are vague, duplicate, stylistic-only, not tied to code/data/tests/literature, inconsistent with `AGENTS.md`, or not testable.

Patch-review prompts are constrained to the finding under review. They must ask Pro to choose exactly one of:

- `ACCEPT`
- `REVISION_NEEDED`
- `WITHDRAW`

Patch-review responses may not introduce unrelated findings. New concerns found during patch review must be restated in a later full-audit prompt as separate findings with evidence.

The audit may continue until Pro and Codex can state the academically defensible conclusion:

> No known unresolved actionable issues remain under the documented audit scope, given the reviewed code, tests, artifacts, literature, and reproduction evidence.

The audit must not claim certainty that no issues exist.

## Finding Lifecycle

Allowed statuses:

- `proposed`
- `validated`
- `rejected`
- `fixing`
- `fixed`
- `pro-review`
- `revision-needed`
- `residual-risk`
- `closed`

Each finding is stored in `findings/PRO-###.md` and referenced from `backlog.md`.

## Required Finding Format

Each Pro finding must include:

- Finding ID
- Classification
- Severity: `critical`, `high`, `medium`, or `low`
- Area
- Files or artifacts likely involved
- Evidence
- Why this matters
- Required fix
- Required tests
- Acceptance criteria

Codex must add:

- Local validation
- Implementation summary
- Test results
- Final resolution
