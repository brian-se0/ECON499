# Round 002 Prompt: Batch 1 Literature Reading

You are GPT 5.5 Pro acting as the literature-grounded audit brain for the ECON499 SPX implied-volatility-surface forecasting project. Codex is the implementation agent. The two uploaded archives remain the evidence base:

- `econ499_extracted_text_manifests_20260427.zip`
- `econ499_lit_review_pdfs_20260427.zip`

This round is not a code audit yet. This round establishes the academically defensible audit standard for surface construction, grid design, timestamp integrity, observed-cell masks, completed-grid interpolation, and observed-vs-completed evaluation.

Read Batch 1 papers in depth before drawing conclusions. Use extracted `paper.md` files first, and use original PDFs whenever equations, tables, figures, page layout, or extraction quality matter.

Mandatory Batch 1 papers from your triage:

1. `ebsco-fulltext-04-08-2026` — “Predictable Dynamics in the S&P 500 Index Options Implied Volatility Surface”
2. `dynamics-of-implied-volatility-surfaces` — “Dynamics of implied volatility surfaces”
3. `s11147-023-09195-5` — “Implied volatility surfaces: a comprehensive analysis using half a billion option prices”
4. `s11147-020-09166-0` — “Option-implied information: What’s the vol surface got to do with it?”
5. `2406-11520v3` — “Operator Deep Smoothing for Implied Volatility”

Reading requirements:

- Do not summarize lazily. Read the relevant sections line-by-line: abstract/intro, data description, timestamp or data-availability statements, option filters, moneyness/strike coordinate definitions, maturity conventions, cross-sectional surface construction, interpolation/smoothing method, observed-versus-fitted/completed evaluation, out-of-sample setup, and any benchmark or horizon tables that affect the ECON499 project.
- If a required claim depends on an equation, table, figure, or ambiguous extracted text, verify it against the PDF before using it.
- If you did not read a section that might matter, say so explicitly and explain why it is deferred.
- Do not invent page numbers or line numbers. If exact pages/lines are not available from the uploaded files, cite paper ID plus section/table/figure/equation names as precisely as possible.
- Do not audit repository code in this round. Only produce literature-derived standards and the minimal repository files you need next.

Return exactly these sections:

## READING_COVERAGE

For each Batch 1 paper:

- paper ID and title
- sections/tables/figures/equations actually read
- whether the extracted text was sufficient or the PDF was consulted
- any extraction caveat that limits confidence
- one-sentence reason this paper is or is not mandatory for the code audit

## LITERATURE_STANDARD_FOR_ECON499_BATCH_1

Give a concise but rigorous standard for the ECON499 project, organized under:

- timestamp and data availability at 15:45
- option universe and filtering
- moneyness or strike coordinate system
- maturity/time-to-expiry convention
- observed surface aggregation
- completed-grid interpolation or smoothing
- observed-cell masks
- observed-cell versus completed-grid evaluation
- interpolation sensitivity and robustness
- minimum benchmark implications from Batch 1

For each bullet, identify which paper(s) support the standard and whether the support is direct SPX evidence or general IVS methodology.

## BATCH_1_REPO_AUDIT_CHECKLIST

List the exact checks Codex should perform once repository files are provided. Each checklist item must be testable against code or artifacts. Include leak checks, mask checks, date/timestamp checks, interpolation checks, and evaluation checks.

## REPOSITORY_FILES_NEEDED_NOW

From your earlier file list, give the smallest file set needed for the Batch 1 code audit. Group files by purpose. Do not request the entire repo yet unless truly necessary.

## OPEN_LITERATURE_GAPS

State whether the uploaded papers are sufficient for Batch 1. If not, list exact peer-reviewed search targets and why they are needed. Do not browse unless necessary.

## FORMAL_FINDINGS_IF_ANY

Because this round has not audited code, this should usually be `none`. Only include a finding if it is a literature-scope problem with the project specification itself, not with unseen code.
