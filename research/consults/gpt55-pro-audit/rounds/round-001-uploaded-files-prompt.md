# Round 001 Uploaded-Archive Prompt To GPT 5.5 Pro

You are GPT 5.5 Pro acting as the audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

The Codex App agent is the implementation and verification agent. The repository is the source of truth. Your job is to guide the audit through evidence-backed findings only.

## Attached Files

The following two files are attached in this ChatGPT conversation:

1. `econ499_extracted_text_manifests_20260427.zip`
   - 73 extracted academic-paper `paper.md` files.
   - 73 extraction `manifest.json` files.
   - extraction inventory and quality reports.
   - Size: 2.4 MB.
   - SHA256 recorded locally: `dcd639179a9770189851e57eae7495f22189ce3ddd3c3eed6a0ae22397ab3461`.
   - This is the primary literature-reading package.

2. `econ499_lit_review_pdfs_20260427.zip`
   - 73 original PDFs from the local `lit_review` folder.
   - Size: 156 MB.
   - SHA256 recorded locally: `c3e8938519a7dc8b4f5c7cc2c4c173b6550ae9e0f488f7e7e1d3b0a2b7140ff6`.
   - This is backup source evidence for layout-sensitive equations, figures, and tables.

The extraction verification from Codex:

- Real PDFs, excluding macOS `._*` sidecar files: 73.
- Extracted `paper.md` files: 73.
- Extracted `manifest.json` files: 73.
- Missing extracted papers: none.
- Extra extracted papers: none.
- Missing output files: none.
- Zero-length `paper.md` files: none.
- Canonical inventory success: 73.
- Canonical inventory partial: 0.
- Canonical inventory failed: 0.
- Extraction audit issues reported by `quality_eval_all.json`: 0.

## Project Context

Research-grade SPX implied-volatility-surface forecasting infrastructure from raw Cboe 15:45 option data.

Non-negotiable constraints:

- Official sample window: 2004-01-02 through 2021-04-09.
- Prediction timestamp: 15:45 America/New_York.
- Same-day EOD fields are forbidden in the 15:45 forecasting problem.
- Every feature must have an availability timestamp less than or equal to the prediction timestamp.
- All preprocessing objects must be fit inside the training window only.
- Split manifests must be explicit, serialized, and versioned.
- Observed-cell masks must be preserved and evaluated separately from completed-grid values.
- Raw data is immutable.
- No silent type coercions, date coercions, or row drops.
- Every filter/drop rule must be explicit, logged, and testable.
- Every join must state and assert expected key cardinality.
- Core ETL must use Polars lazy queries and Arrow datasets, not pandas.
- HPO must use blocked time-series validation only, never random cross-validation.
- Required models include no-change surface benchmark, ridge, elastic net, HAR/factor benchmark, and an arbitrage-aware neural model.
- The neural model predicts total variance, not raw implied volatility.
- The neural model should be described as arbitrage-aware, not arbitrage-free, unless hard constraints guarantee arbitrage-freeness by construction.

Current repo baseline:

- Workspace: `/Volumes/T9/ECON499`.
- GitHub remote: `https://github.com/brian-se0/ECON499.git`.
- Baseline commit: `96e0da6fe741b63d2d4743aaab1ea6b5dde4c468`.
- Project source lives under `src/ivsurf`.
- Pipeline orchestration lives under `scripts`.
- Tests cover unit, integration, regression, property, and e2e smoke paths.

Key modules to audit later:

- `io/`: ingestion, Parquet/Arrow IO, atomic writes, paths.
- `qc/`: raw checks, schema checks, timing checks, sample-window enforcement.
- `cleaning/`: option filters and derived fields.
- `surfaces/`: grid construction, aggregation, interpolation, masks, arbitrage diagnostics.
- `features/`: lagged surfaces, liquidity, factor features, tabular datasets.
- `splits/`: walk-forward split generation and manifests.
- `models/`: no-change/naive, ridge, elastic net, HAR factor, random forest, LightGBM, neural surface, losses, penalties.
- `training/`: sklearn, LightGBM, Torch training, tuning, model factory.
- `evaluation/`: alignment, metrics, loss panels, forecast store, diagnostics, slice reports.
- `stats/`: Diebold-Mariano, SPA, MCS, bootstrap.
- `hedging/`: book, hedge rules, PnL, revaluation.
- `reports/`: tables and figures.
- `reproducibility.py`: run metadata and provenance capture.

## Round 001 Task

First, verify whether you can access and inspect both attached ZIP files.

Then inspect the extracted text package and build a literature reading matrix before issuing any architecture or code findings.

You must not be lazy about the literature. For every paper you decide is relevant to this project, you must:

1. cite the paper by source file or extracted folder ID,
2. state why it is relevant,
3. identify the sections/pages or extracted-text regions you relied on,
4. state whether you read the relevant sections line by line,
5. summarize the concrete implication for this project,
6. distinguish direct claims from the paper versus your inference.

Do not claim you read a paper line by line unless you actually inspected the relevant extracted text or original PDF content.

For papers you exclude from detailed reading, group them by exclusion reason. Acceptable reasons include:

- irrelevant to SPX/option/implied-volatility forecasting,
- hedging-only and not needed for the current audit phase,
- duplicate or superseded by another local paper,
- too remote from the project’s methodology,
- extraction quality problem requiring original-PDF follow-up.

After the reading matrix, answer:

1. Which papers are mandatory grounding for auditing:
   - surface construction and interpolation,
   - static-arbitrage diagnostics/penalties,
   - total variance modeling,
   - benchmark selection,
   - walk-forward validation and HPO,
   - forecast evaluation/statistical tests,
   - hedging evaluation?

2. What additional up-to-date peer-reviewed literature, if any, is necessary before making academically defensible audit decisions? Only propose additional searches for concrete gaps not covered by the local corpus.

3. Based only on the literature and repo context above, what are the highest-risk audit areas for the next round?

4. If you need specific source files from the repository before issuing formal findings, list the exact files and the reason for each.

## Required Response Structure

```text
ATTACHMENT_CHECK
- Can you access the extracted text package?
- Can you access the original PDF package?
- Any extraction/archive issue?

READING_MATRIX
| Paper/File ID | Relevance | Sections/pages/read regions | Line-by-line relevant-section read? | Direct claims | Project implications |

EXCLUDED_OR_DEFERRED_PAPERS
| Group | Papers | Reason |

MANDATORY_GROUNDING_BY_AUDIT_AREA
...

ADDITIONAL_LITERATURE_GAPS
...

NEXT_ROUND_AUDIT_PRIORITIES
...

REPOSITORY_FILES_NEEDED_NEXT
...

FORMAL_FINDINGS_IF_ANY
Use the required finding format below. If none, write: none.
```

Required finding format:

```text
Finding ID:
Classification: NEW_FINDING / REVISION_TO_EXISTING_FINDING / DUPLICATE / NON_ACTIONABLE / RESIDUAL_RISK / ACCEPTED_FIXED / WITHDRAWN
Severity: critical / high / medium / low
Area:
Files or artifacts likely involved:
Evidence:
Why this matters:
Required fix:
Required tests:
Acceptance criteria:
```

Do not claim certainty that no issues exist. The valid endpoint of the full audit is:

> No known unresolved actionable issues remain under the documented audit scope, given the reviewed code, tests, artifacts, literature, and reproduction evidence.
