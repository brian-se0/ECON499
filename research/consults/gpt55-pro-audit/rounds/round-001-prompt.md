# Round 001 Prompt To GPT 5.5 Pro

You are GPT 5.5 Pro acting as the audit brain for the ECON499 SPX implied-volatility-surface forecasting project.

The Codex App agent is the implementation and verification agent. The repository is the source of truth. Your role is to guide the audit through evidence-backed findings only.

Attached files:

1. `econ499_extracted_text_manifests_20260427.zip`
   - 73 extracted academic-paper `paper.md` files
   - 73 extraction `manifest.json` files
   - extraction inventory and quality reports
   - primary literature-reading input

2. `econ499_lit_review_pdfs_20260427.zip`
   - 73 original PDFs
   - backup source evidence for equations, figures, tables, or layout-sensitive checks

Relevant local audit files already created in the repo:

- `research/consults/gpt55-pro-audit/audit-protocol.md`
- `research/consults/gpt55-pro-audit/pro-system-brief.md`
- `research/consults/gpt55-pro-audit/literature-dossier.md`
- `research/consults/gpt55-pro-audit/literature-gaps.md`
- `research/consults/gpt55-pro-audit/repo-dossier.md`
- `research/consults/gpt55-pro-audit/upload-manifest.md`

Project summary:

- Research-grade SPX implied-volatility-surface forecasting infrastructure from raw Cboe 15:45 option data.
- Official sample window: 2004-01-02 through 2021-04-09.
- Forecasting decision time: 15:45 America/New_York.
- Same-day EOD fields are forbidden for the 15:45 forecasting problem.
- Every feature must have availability timestamp <= prediction timestamp.
- Observed-cell masks must be preserved and evaluated separately from completed-grid values.
- Models include no-change surface, ridge, elastic net, HAR/factor, and an arbitrage-aware neural model that predicts total variance.
- Hyperparameter tuning must use blocked time-series validation only.
- The codebase must use Polars lazy queries and Arrow datasets in core ETL, no pandas ETL, no silent coercions, no silent row drops.

Round 001 task:

Before issuing any code or architecture findings, inspect the attached local literature package and build a literature reading matrix.

You must not be lazy about the literature. For every paper you decide is relevant to this project, you must:

1. Cite the paper by source file or extracted folder ID.
2. State why it is relevant.
3. Identify the sections/pages or extracted-text regions you relied on.
4. State whether you read the relevant sections line by line.
5. Summarize the concrete implication for this project.
6. Distinguish direct claims from the paper versus your inference.

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

3. Based only on the literature and repo dossier, what are the highest-risk audit areas for the next round? Do not issue formal findings yet unless the evidence is already concrete enough to satisfy the required finding format.

Required response structure:

```text
ATTACHMENT_CHECK
- Can you access the extracted text package?
- Can you access the original PDF package?
- Any extraction or archive issue?

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

FORMAL_FINDINGS_IF_ANY
Use the required finding format from the standing brief. If none, write: none.
```

Do not claim certainty that no issues exist. The target endpoint of the full audit is: no known unresolved actionable issues remain under the documented audit scope, given the reviewed code, tests, artifacts, literature, and reproduction evidence.
