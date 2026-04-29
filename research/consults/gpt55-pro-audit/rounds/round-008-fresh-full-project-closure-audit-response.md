ROUND_008_AUDIT_DECISION: findings_found

ATTACHMENT_CHECK:

econ499_code_audit_context_20260428_round008.zip: accessible

econ499_extracted_text_manifests_20260427.zip: accessible

econ499_lit_review_pdfs_20260427.zip: accessible

METHOD:

Files/modules inspected: AGENTS.md project standards; current Round 008 code/config audit context; neural model loss/penalty code path; official smoke workflow/configuration; surface-grid assumptions relevant to penalty reachability.

Literature consulted: Uploaded extracted literature text/manifests were available for scope alignment; no additional literature was needed for this concrete code defect.

Tests/contracts reviewed: Project fail-fast/no-fallback standards; mandatory loss-definition testing expectations; official smoke workflow reachability; neural penalty behavior under small-grid configurations.

Prior audit logs/backlog reviewed: Prior B1-B5 closure context was considered only as background. No prior finding was reopened solely because of backlog status.

Reproduction/provenance evidence reviewed: Current uploaded code context and audit notes showing the official smoke workflow reaches the neural-penalty path with a 2x2 grid.

Supplemental command limitations: No further inspection, broad archive extraction, broad Python startup, pytest, or ruff runs were performed in this continuation, per user instruction. The decision below is based on the concrete defect already isolated in the Round 008 continuation notes.

FINDINGS:

ID: B6-CODE-001
Severity: P1
Area: Neural loss/penalty definitions; official smoke workflow; fail-fast runtime validation
Title: Official-smoke 2x2 surface grid can produce non-finite neural roughness penalty instead of failing fast
Evidence: The current official-smoke workflow is reachable with a 2x2 surface grid while the neural roughness/curvature penalty path is enabled. A second-difference/curvature roughness term on a grid dimension of length 2 yields empty curvature tensors. Taking means over those empty tensors produces non-finite loss components rather than an explicit domain validation failure. This violates the project's hard-cutover, no-fallback, fail-fast standard for invalid configurations and leaves a reachable training/loss path capable of silently generating NaN/non-finite objective values.
Why it matters: Neural loss definitions are part of the mandatory audit scope. A smoke workflow is supposed to validate that the official pipeline is runnable under a representative, supported configuration. If the official smoke grid is too small for the enabled roughness penalty, the project can report a passing or partially exercised smoke configuration while an official loss path is mathematically invalid. Non-finite losses can contaminate model artifacts, HPO decisions, forecast metadata, and reproducibility evidence if not rejected at the boundary.
Required fix: Add an explicit invariant at the neural loss/penalty boundary: any enabled second-difference/curvature roughness penalty must require every penalized grid axis to have sufficient length, normally at least 3 cells along the relevant axis. Invalid grids must raise a clear configuration/runtime error before loss computation. Then make the official smoke configuration use a semantically valid grid for the enabled penalty, for example a grid with at least 3 points along each penalized dimension, or otherwise use only a penalty definition whose mathematical domain is valid for the configured grid. The fix should not introduce legacy branches, silent penalty disabling, implicit fallbacks, or automatic behavior changes hidden from provenance.
Required tests: Add unit tests for the neural roughness penalty showing that a 2-point penalized axis fails fast with a clear error before any mean over empty tensors occurs; add positive tests showing finite loss for the minimum valid grid, such as 3x3 when both axes are penalized; add an official-smoke contract test asserting the shipped smoke workflow uses a grid/penalty combination that is mathematically valid and produces finite loss components; add a regression test that rejects non-finite neural loss terms before artifact emission.
Hard-cutover/fail-fast note: The correct behavior is not to silently skip the penalty, downweight it, substitute zero, or preserve a backwards-compatible 2x2 exception. The project should hard-cut to a single validated configuration contract and fail immediately when the enabled penalty is undefined for the configured surface grid.

REOPENED_PRIOR_FINDINGS:

None.

TERMINAL_AUDIT_STATEMENT:

Not reached.

NEXT_CODEX_ACTIONS:

Implement explicit neural roughness-penalty grid-domain validation before loss computation.

Update the official smoke configuration so its grid size is valid for the enabled neural penalty.

Add the required failing and passing neural-loss tests, including finite-loss assertions.

Add an official-smoke regression/contract test that exercises the neural penalty path under the shipped smoke config.

Re-run the targeted neural-loss tests and official smoke workflow evidence after the fix, then update the audit trail with the new provenance.
