ROUND_009_AUDIT_DECISION: findings_found

ATTACHMENT_CHECK:
- econ499_code_audit_context_20260428_round009.zip: accessible
- econ499_extracted_text_manifests_20260427.zip: accessible
- econ499_lit_review_pdfs_20260427.zip: accessible

METHOD:
- Files/modules inspected:
  AGENTS.md; README.md; pyproject.toml; Makefile; configs/data/*.yaml; configs/data/surface.yaml; configs/data/features.yaml; configs/eval/*.yaml; configs/workflow/*.yaml; configs/official_smoke/**/*.yaml; scripts/01_ingest_cboe.py through scripts/09_make_report_artifacts.py; scripts/check_runtime.py; scripts/official_smoke.py; scripts/write_provenance_supplement.py; src/ivsurf/calendar.py; src/ivsurf/config.py; src/ivsurf/reproducibility.py; src/ivsurf/resume.py; src/ivsurf/runtime_preflight.py; src/ivsurf/schemas.py; src/ivsurf/qc/*.py; src/ivsurf/io/*.py; src/ivsurf/cleaning/*.py; src/ivsurf/surfaces/*.py; src/ivsurf/features/*.py; src/ivsurf/splits/*.py; src/ivsurf/models/*.py; src/ivsurf/training/*.py; src/ivsurf/evaluation/*.py; src/ivsurf/stats/*.py; src/ivsurf/hedging/*.py; src/ivsurf/reports/*.py.
- Literature consulted:
  Uploaded extracted text/manifests were available and sufficient for the audit decision. Consulted the project literature dossier and prior literature synthesis around implied-volatility-surface forecasting, surface construction/completion, arbitrage-aware neural modeling, loss/statistical comparison methods, and hedging/reporting context. The original PDFs were accessible as backup evidence; no additional original equation/table/layout inspection was required for the concrete code finding below.
- Tests/contracts reviewed:
  tests/unit/test_schema.py; test_calendar.py; test_raw_checks.py; test_option_filters.py; test_aggregation.py; test_grid.py; test_interpolation.py; test_feature_dataset.py; test_feature_ordering.py; test_split_manifests.py; test_walkforward_clean_evaluation.py; test_alignment.py; test_forecast_store.py; test_metrics.py; test_losses.py; test_penalties.py; test_training_behaviour.py; test_tuning_workflow.py; test_stats.py; test_hedging.py; test_reporting_helpers.py; test_report_artifacts.py; test_repository_contract.py; tests/property/test_surface_grid.py; test_walkforward.py; test_arbitrage_penalties.py; relevant integration/e2e tests including stage01 ingestion manifest counts, early-close alignment, stage03-stage04 target-gap alignment, stage05-stage06 clean evaluation, stage07 negative prediction, stats/hedging slice, report-stage contract, synthetic CPU stage01-to-stage09 pipeline, official-smoke runtime.
- Prior audit logs/backlog reviewed:
  research/consults/gpt55-pro-audit/backlog.md; Round 003 through Round 008B responses and targeted fix evidence, especially B2-CODE-005, B4-CODE-003, B4-CODE-004, B5-CODE-001, and B6-CODE-001 closure evidence.
- Reproduction/provenance evidence reviewed:
  upload-manifest.md; repo-dossier.md; Round 008B B6 fix evidence; run/profile provenance descriptions in README; write_run_manifest and provenance supplement code; stage resume context hashing; forecast metadata persistence; stats/hedging/report run-manifest inputs/outputs.
- Supplemental command limitations:
  The uploaded code and extracted literature archives were usable. A broad combined unzip/hash attempt over the large PDF bundle timed out/stalled earlier, so the PDF archive was treated as accessible backup evidence by archive listing/availability rather than fully re-extracted for this response. No broad pytest/ruff run was restarted. The final decision is based on static inspection plus archived targeted ruff/pytest evidence and narrow follow-up inspection of forecast/evaluation/reporting/provenance paths.

FINDINGS:
- ID: B7-CODE-001
  Severity: P2
  Area: Hedging evaluation / clean forecast coverage / report comparability
  Title: Stage 08 hedging can rank models on unequal forecast-date coverage
  Evidence:
  Stage 07 correctly loads combined forecasts through the evaluation alignment path and asserts identical forecast coverage across models before statistical loss matrices. Stage 08 does not apply the same combined coverage contract before hedging. In scripts/08_run_hedging_eval.py, forecast artifacts are enumerated with sorted_artifact_files, then each forecast file is processed independently. The stage validates each individual forecast artifact against the fixed grid and HPO boundary, but it does not concatenate or load all forecasts through load_forecast_frame, and it does not call assert_equal_forecast_model_coverage on the combined forecast universe. The final hedging summary is produced by summarize_hedging_results in src/ivsurf/hedging/pnl.py, which groups by model and reports means and n_trades, but it does not fail if different models have different quote_date/target_date coverage. Therefore a missing, stale, truncated, or partial model forecast artifact can still produce hedging_results.parquet and hedging_summary.parquet, and Stage 09 can rank/report those hedging summaries as if they were comparable.
  Why it matters:
  Hedging evaluation is part of the documented audit scope and is a model-comparison artifact, not a standalone per-model diagnostic. Ranking mean absolute revaluation error or hedged PnL across models is not clean if models are evaluated on different trading days. This is the hedging analogue of the already-fixed stats coverage issue: unequal coverage can change rankings because the market regimes/dates differ, even when every individual model file is internally grid-complete and after the HPO boundary.
  Required fix:
  Add a Stage 08 fail-fast combined forecast coverage gate before any per-model hedging output is written. The simplest hard-cut fix is to load all forecast artifacts through load_forecast_frame(workflow_paths.forecast_dir, grid), which already enforces complete fixed-grid metadata and identical model/date/cell coverage, then partition that validated combined frame by model for the existing hedging loop. Alternatively, explicitly concatenate all forecast files and call assert_equal_forecast_model_coverage on the combined frame before per-model processing. Also require the expected official model set, unless an explicitly documented non-official diagnostic mode is used and clearly barred from official report artifacts.
  Required tests:
  Add a Stage 08 unit/integration regression that creates two valid per-model forecast parquet files with complete grid coverage individually but one missing quote_date/target_date pair, then asserts scripts/08_run_hedging_eval.py fails before writing hedging_results.parquet or hedging_summary.parquet. Add a companion passing test for equal coverage. Add a report-stage or end-to-end contract test proving official report generation cannot consume hedging summaries created from unequal model coverage.
  Hard-cutover/fail-fast note:
  Do not intersect forecast dates, silently drop unmatched dates, pad missing hedges, or rank with unequal n_trades. Official hedging must fail before writing artifacts unless all compared models share the identical clean forecast-date universe.

REOPENED_PRIOR_FINDINGS:
- None.

TERMINAL_AUDIT_STATEMENT:
- Not reached.

NEXT_CODEX_ACTIONS:
- Patch Stage 08 to validate combined cross-model forecast coverage before per-model hedging evaluation.
- Add the unequal-coverage failing regression and equal-coverage passing regression for Stage 08.
- Add or update report/e2e contract coverage so Stage 09 cannot publish hedging rankings derived from unequal model coverage.
- Run targeted ruff/pytest for the modified hedging/evaluation/report tests and record evidence in the audit trail.
