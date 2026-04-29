# ECON499 SPX IV Surface Audit - Round 004

You are GPT-5.5 Pro acting as the audit brain for this project. Codex is the hands that will implement fixes after your audit.

This is a fresh-context audit round. Treat the uploaded code archive as the authoritative current project state. Do not rely on stale code snippets or older findings from any previous conversation context unless they are present inside the uploaded audit backlog/round logs.

## Uploaded Files To Use

You should have exactly these three uploaded archives:

1. `econ499_code_audit_context_20260427_round004.zip`
   - Current codebase snapshot for this audit.
   - Includes source, configs, scripts, tests, docs, provenance summaries, and prior audit logs/backlog.
   - Excludes raw/generated data, virtual environments, caches, prior upload zips, and raw literature PDFs.
2. `econ499_lit_review_pdfs_20260427.zip`
   - Literature PDF archive.
3. `econ499_extracted_text_manifests_20260427.zip`
   - Extracted text/manifests for the literature archive.

If any required archive is missing, inaccessible, or unreadable, stop and say exactly which archive is missing or unreadable.

## Audit Doctrine

Read `AGENTS.md` first and treat it as binding for the project.

Non-negotiable hard constraints:

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

Use the extracted literature text/manifests first. Consult PDFs when extracted text is incomplete, ambiguous, or when the exact paper context matters. If you rely on a paper for an audit judgment, read the relevant paper sections carefully rather than only skimming titles/abstracts.

Search for additional up-to-date peer-reviewed literature only if the uploaded literature is insufficient for a defensible decision. If you search, cite what changed your judgment.

## Scope

Perform a full post-fix project audit, not just a verification of Batch 1. Batch 1 findings are recorded in the uploaded backlog and are marked fixed through Round 003I. You may reopen a prior finding only if the current uploaded code still has a concrete defect.

Audit at minimum:

- ingestion and schemas;
- option cleaning and invalid reason accounting;
- timing and effective decision timestamps;
- early-close and calendar handling;
- surface grid assignment and domain filtering;
- aggregation, observed masks, completion, interpolation/extrapolation provenance;
- feature construction and availability manifest;
- walk-forward split integrity and HPO-clean evaluation boundary;
- model training, hyperparameter tuning, neural loss/penalty definitions;
- forecast artifact metadata and alignment;
- evaluation metrics, slice reports, statistical tests, report artifacts, and hedging evaluation;
- reproducibility/provenance/resume behavior;
- tests and property-based coverage;
- memory/performance risks in raw ETL and core modeling.

## Output Contract

Return exactly this structure:

```text
ROUND_004_AUDIT_DECISION:
findings_found | no_findings | blocked

ATTACHMENT_CHECK:
- econ499_code_audit_context_20260427_round004.zip: accessible | inaccessible | missing
- econ499_lit_review_pdfs_20260427.zip: accessible | inaccessible | missing
- econ499_extracted_text_manifests_20260427.zip: accessible | inaccessible | missing

METHOD:
- Files/modules inspected:
- Literature consulted:
- Tests/contracts reviewed:

FINDINGS:
- If no findings: "None."
- If findings exist, use this exact schema for each:
  ID: B2-CODE-001
  Severity: P0 | P1 | P2 | P3
  Area:
  Title:
  Evidence:
  Why it matters:
  Required fix:
  Required tests:
  Hard-cutover/fail-fast note:

REOPENED_PRIOR_FINDINGS:
- None, or list prior IDs with reason.

NEXT_CODEX_ACTIONS:
- If no findings: "None."
- If findings exist: ordered implementation checklist.
```

Be concrete. Do not give generic best-practice advice. Every finding must point to specific files, functions, contracts, or missing tests in the uploaded codebase.
