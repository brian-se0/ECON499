# Round 001B Archive Triage Prompt To GPT 5.5 Pro

The previous broad prompt was stopped because it was too large for one useful response. Continue using the same two uploaded archives already attached in this conversation:

- `econ499_extracted_text_manifests_20260427.zip`
- `econ499_lit_review_pdfs_20260427.zip`

Do not redo the full line-by-line reading matrix in this response.

Your task now is narrower:

1. Confirm archive access and extraction inventory.
2. Summarize any archive/extraction limitations.
3. Select the papers that must be read in detail for the audit.
4. Group those selected papers into small reading batches.
5. State exactly what each batch should decide.
6. List repository source files you want Codex to provide after the literature batches.

Important:

- Do not issue formal code findings in this response.
- Do not claim full line-by-line reading yet.
- Use the extracted `paper.md` files as primary evidence.
- Use original PDFs only for equations, figures, tables, or extraction uncertainty.
- Prefer a focused mandatory reading set over trying to read all 73 papers in detail.
- Exclude/defer papers that are hedging-only, option-return-only, generic realized-volatility-only, duplicate, or remote from SPX implied-volatility-surface forecasting.

Required output:

```text
ATTACHMENT_CHECK
- Extracted text package accessible:
- Original PDF package accessible:
- Inventory count:
- Extraction concerns:

MANDATORY_READING_SET
| Batch | Paper/File ID | Why mandatory | Audit area | What to read line-by-line later |

DEFERRED_OR_EXCLUDED_GROUPS
| Group | Papers or count | Reason |

BATCH_PLAN
Batch 1:
Batch 2:
Batch 3:
...

REPOSITORY_FILES_NEEDED_AFTER_LITERATURE_TRIAGE
| File path | Why needed |

FORMAL_FINDINGS_IF_ANY
none
```

The project constraints remain:

- 15:45 SPX implied-volatility-surface forecasting.
- No same-day EOD leakage.
- Feature availability timestamps must be <= prediction timestamp.
- Observed-cell masks and completed-grid values must be evaluated separately.
- Blocked time-series validation only.
- Neural model predicts total variance and is arbitrage-aware unless hard constraints guarantee arbitrage-freeness.
- Audit endpoint is no known unresolved actionable issues under the documented scope, not certainty that no issues exist.
