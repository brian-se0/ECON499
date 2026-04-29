# Round 003G: Pro Verification of B1-CODE-008

Captured: 2026-04-27

VERIFICATION_DECISION

B1-CODE-008: fixed

EVIDENCE

Low-level weighted metric helpers no longer silently fall back: _normalize_weights() now rejects empty, non-finite, negative, or non-positive-total weights, and weighted_mse() no longer substitutes an unweighted mean when weights sum to zero.

Stage 05 validation semantics now fail fast for observed metrics with no positive observed weight, with an explicit error refusing full-grid fallback.

Stage 07 daily loss construction now labels empty or zero-weight observed-cell metrics as NaN, while full-grid metrics remain separately and explicitly computed.

Slice reports now preserve the distinction between evaluation_scope="observed" and evaluation_scope="full"; empty observed slices receive NaN metrics and cell_count=0 rather than being scored as full-grid slices.

The tests directly cover the original failure mode:

zero-total weights raise in metric helpers,

observed daily metrics become NaN rather than falling back,

validation observed metrics raise,

empty observed slice metrics remain NaN while full-slice metrics remain finite.

Ruff and focused pytest passed: 17 tests covering metrics, daily losses, slice reports, and tuning workflow.

REQUIRED_ADJUSTMENTS

none

NEW_REGRESSION_RISKS

Low downstream-stats risk: if observed daily loss rows contain NaN, later DM/SPA/MCS code must either exclude those dates under a documented rule or fail clearly. This does not reopen B1-CODE-008 because the fallback problem is fixed, but the next statistical-evaluation audit should check NaN handling.

HPO behavior is now stricter: an observed-metric validation split with no positive observed weights will fail fast. That is preferable to false scoring, but split construction and surface coverage diagnostics should make such cases easy to diagnose.

No leakage or metric ambiguity is introduced by this slice; it reduces ambiguity by making observed/full-grid scopes explicit.

NEXT_FIX_SLICE

Proceed to B1-CODE-009.

Reason: the remaining Batch 1 artifact-level risk is forecast-store reproducibility metadata. Forecast artifacts still need to be audited for split IDs, model/config hashes, grid/target versions, and alignment safeguards so completed-grid forecast comparisons cannot mix incompatible runs or surfaces.

Extended Pro
ChatGPT is AI and can make mistakes. Check important info.
