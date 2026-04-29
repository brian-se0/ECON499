# Round 002C Prompt: Batch 1 Paper 3 Reading

Continue the literature-first audit in one-paper units.

You are GPT 5.5 Pro acting as the literature-grounded audit brain for the ECON499 SPX implied-volatility-surface forecasting project. Codex is the implementation agent. Use the already uploaded archives:

- `econ499_extracted_text_manifests_20260427.zip`
- `econ499_lit_review_pdfs_20260427.zip`

This round is only for one Batch 1 paper:

`s11147-023-09195-5` — “Implied volatility surfaces: a comprehensive analysis using half a billion option prices”

Task:

Read the relevant parts of this paper carefully enough to establish what it implies for the ECON499 audit. Use the extracted `paper.md` first. Consult the PDF only if extraction is ambiguous, if equations/tables/figures matter, or if page layout matters.

Read these sections line-by-line if present:

- abstract and introduction
- data/sample description
- OptionMetrics or other source data definitions
- filtering and liquidity/moneyness/maturity handling
- surface construction methods compared
- kernel/spline/interpolation/smoothing method descriptions
- accuracy metrics and observed/fitted/completed surface evaluation
- liquidity, maturity, moneyness, and sparse-region breakdowns
- conclusions on preferred construction methods and limitations

Return exactly these sections, with concise bullets:

## READING_COVERAGE

## PAPER_SPECIFIC_EVIDENCE

For each evidence item, cite paper ID plus section/table/figure/equation as precisely as available, say whether it is direct S&P 500/SPX or broader US-option evidence, and give one ECON499 implication.

## AUDIT_STANDARD_CONTRIBUTION

State what this paper contributes under:

- option filtering and liquidity
- moneyness/maturity coordinate choices
- observed surface construction
- completed-grid interpolation or smoothing
- observed-vs-completed evaluation
- interpolation sensitivity and robustness
- sparse-region diagnostics
- limitations for 15:45 ECON499 forecasting

Use `not addressed` where the paper does not support a standard.

## REPO_CHECKS_DERIVED_FROM_THIS_PAPER

List testable code/artifact checks Codex should perform later. Do not claim any finding against the repo yet.

## FORMAL_FINDINGS_IF_ANY

none
