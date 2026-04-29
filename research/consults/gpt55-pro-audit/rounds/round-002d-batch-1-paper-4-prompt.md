# Round 002D Prompt: Batch 1 Paper 4 Reading

Continue the literature-first audit in one-paper units.

You are GPT 5.5 Pro acting as the literature-grounded audit brain for the ECON499 SPX implied-volatility-surface forecasting project. Codex is the implementation agent. Use the already uploaded archives:

- `econ499_extracted_text_manifests_20260427.zip`
- `econ499_lit_review_pdfs_20260427.zip`

This round is only for one Batch 1 paper:

`s11147-020-09166-0` — “Option-implied information: What’s the vol surface got to do with it?”

Task:

Read the relevant parts of this paper carefully enough to establish what it implies for the ECON499 audit. Use the extracted `paper.md` first. Consult the PDF only if extraction is ambiguous, if equations/tables/figures matter, or if page layout matters.

Read these sections line-by-line if present:

- abstract and introduction
- option-implied information measures and why surface construction matters
- data/sample and S&P 500/index-option relevance
- surface construction method comparison
- one-dimensional kernel surface method
- bias/noise/sensitivity results
- risk-neutral moments, variance risk premium, or other downstream measures affected by construction
- implications and limitations for using completed surfaces as research targets

Return exactly these sections, with concise bullets:

## READING_COVERAGE

## PAPER_SPECIFIC_EVIDENCE

For each evidence item, cite paper ID plus section/table/figure/equation as precisely as available, say whether it is direct SPX/S&P 500 or broader option-methodology evidence, and give one ECON499 implication.

## AUDIT_STANDARD_CONTRIBUTION

State what this paper contributes under:

- surface-construction sensitivity
- completed-grid target reliability
- observed-vs-completed evaluation
- risk-neutral moment or downstream-measure robustness
- interpolation/extrapolation caution
- documentation standards
- limitations for 15:45 ECON499 forecasting

Use `not addressed` where the paper does not support a standard.

## REPO_CHECKS_DERIVED_FROM_THIS_PAPER

List testable code/artifact checks Codex should perform later. Do not claim any finding against the repo yet.

## FORMAL_FINDINGS_IF_ANY

none
