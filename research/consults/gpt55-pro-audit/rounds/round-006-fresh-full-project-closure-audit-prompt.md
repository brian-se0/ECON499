# ECON499 SPX IV Surface Audit: Round 006 Fresh Full-Project Closure Audit

You are GPT 5.5 Pro acting as the independent audit brain. Codex is the implementation agent.

This is a fresh full-project closure audit after Pro verified all prior B1, B2, and B3 findings fixed through Round 005C.

Attached files should include:
- `econ499_code_audit_context_20260428_round006.zip`
- `econ499_extracted_text_manifests_20260427.zip`
- `econ499_lit_review_pdfs_20260427.zip`

Read `AGENTS.md` first and treat it as binding.

Non-negotiable constraints include:
- Python 3.13.5 only.
- No hidden leakage.
- No same-day EOD fields in the 15:45 forecasting problem.
- Every feature must have an availability timestamp less than or equal to the prediction timestamp.
- All preprocessing must be fit inside the training window only.
- Split manifests must be explicit, serialized, and versioned.
- Observed-cell masks must be preserved and evaluated separately from completed-grid values.
- Raw data is immutable.
- No silent type coercions, date coercions, row drops, filters, joins, or fallbacks.
- Every filter/drop rule must be explicit, logged, and testable.
- Every join must state and assert expected cardinality.
- Use Polars lazy queries and Arrow datasets in core ETL.
- Do not use pandas in ETL or core modeling.
- Reproducibility artifacts must record git/config/data/split/package/seed/hardware metadata.
- Tests are mandatory for any issue affecting timing, features, surfaces, loss definitions, or evaluation metrics.

Additional hard-cutover rule from the project owner:
- Always avoid fallback paths.
- Always choose fail-fast behavior.
- Always avoid backwards compatibility and legacy branches.
- If an old artifact/schema/API would be unsafe, require a hard cutover with schema/version bump and clear failure.
- Do not propose compatibility shims, silent legacy readers, or fallback behavior.

## Literature Instructions

Use the extracted literature text/manifests first. Consult PDFs when extracted text is incomplete, ambiguous, or when exact paper context matters.

If you rely on a paper for an audit judgment, read the relevant paper sections carefully rather than only skimming titles/abstracts.

Search for additional up-to-date peer-reviewed literature only if the uploaded literature is insufficient for an academically defensible decision. If you search, cite what changed your judgment.

## Audit Scope

Perform a full project closure audit, not merely a verification of Round 005C.

Prior B1, B2, and B3 findings are recorded in the uploaded backlog and have been marked `pro-verified-fixed`. You may reopen a prior finding only if the current uploaded code still has a concrete defect.

Audit at minimum:
- ingestion and schemas;
- option cleaning and invalid reason accounting;
- timing and effective decision timestamps;
- early-close and calendar handling;
- surface grid assignment and domain filtering;
- aggregation, observed masks, completion, interpolation/extrapolation provenance;
- feature construction and availability manifests;
- walk-forward split integrity and HPO-clean evaluation boundary;
- model training, hyperparameter tuning, neural loss/penalty definitions;
- forecast artifact metadata and alignment;
- evaluation metrics, slice reports, statistical tests, report artifacts, and hedging evaluation;
- reproducibility/provenance/resume behavior;
- tests and property-based coverage;
- memory/performance risks in raw ETL and core modeling;
- whether the currently reviewed evidence is sufficient to make a defensible terminal audit statement.

## Decision Standard

Your goal is not to continue audit loops indefinitely. Your goal is to identify concrete actionable issues if they exist.

Return `no_known_unresolved_actionable_issues` only if, after inspecting the uploaded code, tests, artifacts, prior audit logs/backlog, literature where relevant, and reproduction evidence, you know of no unresolved actionable issue within the documented audit scope.

Do not claim mathematical certainty or omniscience. The required terminal statement is:

```text
No known unresolved actionable issues remain under the documented audit scope, given the reviewed code, tests, artifacts, literature, and reproduction evidence.
```

## Required Response Format

```text
ROUND_006_AUDIT_DECISION: findings_found | no_known_unresolved_actionable_issues | blocked

ATTACHMENT_CHECK:
- econ499_code_audit_context_20260428_round006.zip: accessible | inaccessible | missing
- econ499_extracted_text_manifests_20260427.zip: accessible | inaccessible | missing
- econ499_lit_review_pdfs_20260427.zip: accessible | inaccessible | missing

METHOD:
- Files/modules inspected:
- Literature consulted:
- Tests/contracts reviewed:
- Prior audit logs/backlog reviewed:
- Reproduction/provenance evidence reviewed:

FINDINGS:
- If no findings: "None."
- If findings exist, use this exact schema for each:

ID: B4-CODE-001
Severity: P0 | P1 | P2 | P3
Area:
Title:
Evidence:
Why it matters:
Required fix:
Required tests:
Hard-cutover/fail-fast note:

REOPENED_PRIOR_FINDINGS:
- None, or list prior IDs with concrete reason.

TERMINAL_AUDIT_STATEMENT:
- If no known unresolved actionable issues remain, write exactly:
  "No known unresolved actionable issues remain under the documented audit scope, given the reviewed code, tests, artifacts, literature, and reproduction evidence."
- Otherwise:
  "Not reached."

NEXT_CODEX_ACTIONS:
- If no findings: "None."
- If findings exist or blocked: ordered implementation or evidence-collection checklist.
```
