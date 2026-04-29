ROUND_007_AUDIT_DECISION: findings_found

ATTACHMENT_CHECK:
- econ499_code_audit_context_20260428_round007.zip: accessible
- econ499_extracted_text_manifests_20260427.zip: accessible
- econ499_lit_review_pdfs_20260427.zip: accessible

METHOD:
- Files/modules inspected:
  AGENTS.md; pyproject.toml; README.md; Makefile; configs/data/*.yaml; configs/eval/*.yaml; configs/workflow/*.yaml; configs/models/*.yaml; configs/official_smoke/**/*.yaml; scripts/01_ingest_cboe.py through scripts/09_make_report_artifacts.py; scripts/official_smoke.py; scripts/check_runtime.py; scripts/clean_pipeline_artifacts.py; scripts/write_provenance_supplement.py; src/ivsurf/config.py; src/ivsurf/schemas.py; src/ivsurf/calendar.py; src/ivsurf/runtime_preflight.py; src/ivsurf/reproducibility.py; src/ivsurf/resume.py; src/ivsurf/workflow.py; src/ivsurf/io/*; src/ivsurf/cleaning/*; src/ivsurf/surfaces/*; src/ivsurf/features/*; src/ivsurf/splits/*; src/ivsurf/models/*; src/ivsurf/training/*; src/ivsurf/evaluation/*; src/ivsurf/stats/*; src/ivsurf/hedging/*; src/ivsurf/reporting/*.
- Literature consulted:
  Extracted text/manifests inventory, canonical extraction reports, upload manifest, literature-dossier.md, literature-gaps.md, and selected extracted paper manifests/text covering IV-surface dynamics/forecasting, surface construction, observed/imputed surface evaluation, neural/deep IV-surface forecasting, hedging/trading evaluation, SVI/eSSVI/arbitrage-aware surface methods, Patton-style volatility forecast loss considerations, and Hansen MCS/statistical comparison material. The original-PDF archive was accessible, but no original equation, figure, table, or layout was decisive for the concrete code finding below, so PDF screenshot/layout inspection was not needed.
- Tests/contracts reviewed:
  tests/unit/test_schema.py; test_calendar.py; test_raw_checks.py; test_option_filters.py; test_aggregation.py; test_interpolation.py; test_grid.py; test_feature_dataset.py; test_feature_ordering.py; test_split_manifests.py; test_walkforward_clean_evaluation.py; test_training_behaviour.py; test_tuning_workflow.py; test_forecast_store.py; test_alignment.py; test_metrics.py; test_losses.py; test_stats.py; test_slice_reports.py; test_hedging.py; test_runtime_preflight.py; test_runtime_defaults.py; test_resume.py; test_cleanup.py; integration tests for stage01 ingestion counts, early-close stage02 handling, stage03/stage04 target-gap alignment, stage04 early-close alignment, stage05/stage06 clean evaluation, stage07 negative prediction, stats/hedging slices, report-stage contracts, neural imputed-cell supervision, synthetic CPU stage01-to-stage09 pipeline, sample-window enforcement, and resume stage01; regression tests for observed-mask preservation and report artifacts; property tests for surface grid, walk-forward splits, and arbitrage penalties; e2e official smoke runtime test.
- Prior audit logs/backlog reviewed:
  research/consults/gpt55-pro-audit/backlog.md; Round 001B literature triage; Round 002A-002F literature responses/synthesis; Round 003A-003I B1 findings and verifications; Round 004/004A/004B B2 audit and verifications; Round 005/005B/005C B3 audit and verifications; Round 006/006B/006C/006D B4 audit and verifications; Round 007 prompt.
- Reproduction/provenance evidence reviewed:
  Round 005B/005C B3 evidence; Round 006 and Round 006D B4 fix evidence; archived ruff/pytest evidence including focused B4 follow-up pytest, broader B4 pytest slice, and ruff slices; run-manifest/provenance requirements and inspected run-manifest write paths; workflow profile/path labeling; split/tuning/forecast manifest contracts.
- Supplemental command limitations:
  The uploaded archives were accessible and statically inspectable. Local `python --version` reports Python 3.13.5, but ordinary Python/pytest invocations in this ChatGPT container hang due environment startup hooks/import behavior, so broad fresh pytest/ruff execution was not used as primary evidence. Static inspection plus archived ruff/pytest evidence was sufficient to identify the concrete actionable defect below. The PDF zip listing was accessible; a full compressed-data test of the large PDF zip timed out, but file listing confirmed the 73-PDF archive is readable.

FINDINGS:
ID: B5-CODE-001
Severity: P2
Area: official smoke / runtime preflight / configured-grid evaluation boundary
Title: Official smoke does not pass the smoke surface-grid config into Stages 07 and 08.
Evidence:
  scripts/official_smoke.py defines the smoke surface config as repo_root/configs/official_smoke/data/surface.yaml and passes it into Stages 03, 04, 05, 06, and 09. However, the calls at scripts/official_smoke.py lines 266-279 invoke stage07.main(...) and stage08.main(...) without surface_config_path=smoke_surface_config_path. Both downstream entrypoints default to Path("configs/data/surface.yaml"): scripts/07_run_stats.py lines 141-162 and scripts/08_run_hedging_eval.py lines 50-73. Stage 07 then loads realized and forecast surfaces with that configured grid at scripts/07_run_stats.py lines 235-236. Stage 08 similarly loads actual surfaces with the configured grid at scripts/08_run_hedging_eval.py line 134 and validates each forecast artifact against that grid at lines 165-169. The validation path is fail-fast: src/ivsurf/evaluation/alignment.py lines 233-290 calls require_surface_grid_metadata(...) and require_complete_unique_surface_grid(...). The smoke grid is 2x2, with maturity_days [7, 30] and moneyness_points [-0.10, 0.00], while the default official grid is 9x9, with 81 cells. Their surface-grid hashes differ deterministically. Therefore, in the supported official-smoke runtime, Stages 07 and 08 will either read the wrong default full-grid config from the repository root and fail grid metadata/cell-count validation against smoke artifacts, or fail from a missing relative default config if the script is invoked outside the repository root.
Why it matters:
  The official workflow documents make official-smoke as the required pre-pipeline runtime smoke bundle. After the B3/B4 hard-cutover grid-contract fixes, the smoke stages must all consume the same explicit smoke grid used to build the gold surfaces, features, HPO/training matrices, and forecast artifacts. The current mixed-grid invocation prevents the official smoke from being reliable closure/reproduction evidence and can be masked in unsupported environments because the e2e official-smoke test skips on runtime-preflight RuntimeError before reaching Stages 07/08. The synthetic CPU stage01-to-stage09 integration test does not catch this because it changes cwd into a temp directory where the default configs/data/surface.yaml is already the synthetic 2x2 grid.
Required fix:
  In scripts/official_smoke.py, pass surface_config_path=smoke_surface_config_path into both stage07.main(...) and stage08.main(...). Do not rely on the default configs/data/surface.yaml for any custom-grid smoke stage.
Required tests:
  Add a deterministic official-smoke argument-propagation contract test that does not skip behind the Windows/GPU runtime preflight. For example, monkeypatch run_runtime_preflight to a successful report and monkeypatch/load stage spies so the test asserts that Stages 03, 04, 05, 06, 07, 08, and 09 receive the same smoke_surface_config_path where their signatures accept a surface config. Also rerun the supported-runtime official smoke after the patch and retain evidence that Stage 07 stats, Stage 08 hedging, and Stage 09 report artifacts complete under the 2x2 smoke grid.
Hard-cutover/fail-fast note:
  The fix should preserve the hard cutover: no fallback from the smoke grid to the default full grid, no inferred grid compatibility, and no bypass of surface_grid_hash/cell-count validation. The official smoke should fail fast only if a stage artifact does not match the explicitly supplied smoke grid.

REOPENED_PRIOR_FINDINGS:
- None.

TERMINAL_AUDIT_STATEMENT:
- Not reached.

NEXT_CODEX_ACTIONS:
- Patch scripts/official_smoke.py so stage07.main(...) and stage08.main(...) receive surface_config_path=smoke_surface_config_path.
- Add a non-skipping official-smoke call-contract test that verifies the smoke surface config is propagated consistently into all stages that accept it, especially Stages 07 and 08.
- Rerun the new contract test plus the relevant existing smoke/report/stat/hedging tests.
- In a supported Windows/CUDA runtime, rerun make official-smoke and archive the resulting Stage 07, Stage 08, Stage 09, and summary evidence.