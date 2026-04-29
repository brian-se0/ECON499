# Round 002B Prompt: Batch 1 Paper 2 Reading

Continue the literature-first audit in one-paper units.

You are GPT 5.5 Pro acting as the literature-grounded audit brain for the ECON499 SPX implied-volatility-surface forecasting project. Codex is the implementation agent. Use the already uploaded archives:

- `econ499_extracted_text_manifests_20260427.zip`
- `econ499_lit_review_pdfs_20260427.zip`

This round is only for one Batch 1 paper:

`dynamics-of-implied-volatility-surfaces` — “Dynamics of implied volatility surfaces”

Task:

Read the relevant parts of this paper carefully enough to establish what it implies for the ECON499 audit. Use the extracted `paper.md` first. Consult the PDF if extraction is ambiguous, if equations/tables/figures matter, or if page layout matters.

Read these sections line-by-line if present:

- abstract and introduction
- data/sample description
- SP500 or index-option surface construction
- strike/moneyness and maturity coordinate definitions
- PCA/Karhunen-Loeve/factor construction
- sticky-moneyness or benchmark comparisons
- surface dynamics and forecasting implications
- figures/tables defining factors, grids, and empirical findings
- limitations or assumptions that affect whether ECON499 can claim academic defensibility

Return exactly these sections, with concise bullets:

## READING_COVERAGE

## PAPER_SPECIFIC_EVIDENCE

For each evidence item, cite paper ID plus section/table/figure/equation as precisely as available, say whether it is direct SPX/SP500 evidence, and give one ECON499 implication.

## AUDIT_STANDARD_CONTRIBUTION

State what this paper contributes under:

- surface coordinate system
- observed surface construction
- factor/PCA/HAR benchmark design
- forecast target and dynamics
- observed-vs-completed evaluation
- leakage risks
- limitations for 15:45 ECON499 forecasting

Use `not addressed` where the paper does not support a standard.

## REPO_CHECKS_DERIVED_FROM_THIS_PAPER

List testable code/artifact checks Codex should perform later. Do not claim any finding against the repo yet.

## FORMAL_FINDINGS_IF_ANY

none
