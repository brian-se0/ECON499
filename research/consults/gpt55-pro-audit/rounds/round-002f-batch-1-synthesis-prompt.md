# Round 002F Prompt: Batch 1 Literature Synthesis

You are GPT 5.5 Pro acting as the literature-grounded audit brain for the ECON499 SPX implied-volatility-surface forecasting project. Codex is the implementation agent.

Use the five Batch 1 paper-specific responses already produced in this same chat:

1. `ebsco-fulltext-04-08-2026` — “Predictable Dynamics in the S&P 500 Index Options Implied Volatility Surface”
2. `dynamics-of-implied-volatility-surfaces` — “Dynamics of implied volatility surfaces”
3. `s11147-023-09195-5` — “Implied volatility surfaces: a comprehensive analysis using half a billion option prices”
4. `s11147-020-09166-0` — “Option-implied information: What’s the vol surface got to do with it?”
5. `2406-11520v3` — “Operator Deep Smoothing for Implied Volatility”

Do not inspect the archives, PDFs, figures, images, or tables again for this synthesis unless a contradiction among your own five Batch 1 responses makes that unavoidable. If you need a source-specific check later, list it as an open item instead of starting a long file-read loop.

Task:
Synthesize Batch 1 into an audit standard for ECON499's surface construction, observed/completed grid handling, 15:45 availability, smoothing/interpolation, benchmark framing, and arbitrage-aware modeling language. Then identify the smallest set of repository files Codex should provide for the first code-audit pass under this Batch 1 standard.

Return exactly these sections:

## BATCH_1_LITERATURE_STANDARD

Concise numbered standards. Each standard must cite one or more of the five paper IDs and state whether the support is direct SPX/S&P 500 evidence, broader IVS methodology, or only indirectly relevant.

## BATCH_1_CODE_AUDIT_CHECKLIST

Concrete checks Codex should perform against the repo. Make each item testable against code, config, artifacts, docs, or tests.

## MINIMAL_REPO_FILES_FOR_BATCH_1_CODE_AUDIT

List the smallest file set Codex should paste or summarize next. Include why each file is needed. Prioritize files under `src/ivsurf`, `scripts`, `configs`, `tests`, `docs`, or `research/consults/gpt55-pro-audit` if relevant.

## OPEN_LITERATURE_GAPS

List what Batch 1 does not settle and what later literature batches must cover.

## FORMAL_FINDINGS_IF_ANY

Do not claim a project finding unless it follows from the project specification or prior uploaded repo context already in this chat. Otherwise write `none`.
