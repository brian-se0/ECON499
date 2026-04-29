# Round 004 Pro Response - Full Post-Fix Code Audit

Captured: 2026-04-27
Conversation: https://chatgpt.com/c/69efec14-7dfc-83ea-b661-43f97287aade

ROUND_004_AUDIT_DECISION: findings_found

ATTACHMENT_CHECK:

econ499_code_audit_context_20260427_round004.zip: accessible

econ499_lit_review_pdfs_20260427.zip: accessible

econ499_extracted_text_manifests_20260427.zip: accessible

METHOD:

Files/modules inspected: AGENTS.md; pyproject.toml; configs/data/{raw,cleaning,surface,features}.yaml; configs/eval/{metrics,walkforward,stats_tests,report_artifacts,hedging}.yaml; configs/models/*.yaml; configs/workflow/*.yaml; src/ivsurf/schemas.py; src/ivsurf/io/ingest_cboe.py; scripts/01_ingest_cboe.py; scripts/02_build_option_panel.py; src/ivsurf/cleaning/{derived_fields.py,option_filters.py}; src/ivsurf/calendar.py; src/ivsurf/qc/{raw_checks.py,schema_checks.py,timing_checks.py}; src/ivsurf/surfaces/{grid.py,aggregation.py,interpolation.py,arbitrage_diagnostics.py}; scripts/03_build_surfaces.py; src/ivsurf/features/{availability.py,lagged_surface.py,liquidity.py,tabular_dataset.py}; scripts/04_build_features.py; src/ivsurf/splits/{walkforward.py,manifests.py}; src/ivsurf/models/{base.py,naive.py,ridge.py,elasticnet.py,har_factor.py,lightgbm_model.py,random_forest.py,positive_target.py,losses.py,neural_surface.py,penalties.py}; src/ivsurf/training/{model_factory.py,tuning.py,fit_sklearn.py,fit_lightgbm.py,fit_torch.py}; scripts/05_tune_models.py; scripts/06_run_walkforward.py; src/ivsurf/evaluation/{alignment.py,forecast_store.py,loss_panels.py,metrics.py,slice_reports.py,interpolation_sensitivity.py}; scripts/07_run_stats.py; src/ivsurf/stats/{bootstrap.py,diebold_mariano.py,spa.py,mcs.py}; scripts/08_run_hedging_eval.py; scripts/09_make_report_artifacts.py; src/ivsurf/reproducibility.py; src/ivsurf/resume.py; src/ivsurf/runtime_preflight.py; scripts/official_smoke.py; scripts/check_runtime.py; prior audit backlog and Round 003 logs under research/consults/gpt55-pro-audit/.

Literature consulted: extracted literature manifests/text were used first, including the extracted text for Cont and da Fonseca on implied-volatility surface dynamics, Hansen/Lunde/Nason on the model confidence set, Fengler on arbitrage-free implied-volatility surface smoothing, and the SPX implied-volatility-surface deep-forecasting paper in the uploaded literature text archive. The PDF archive was checked as accessible, but the concrete findings below are code/contract defects that did not require overriding the extracted text with PDF-only context. No additional web literature search was needed.

Tests/contracts reviewed: AGENTS.md; research/consults/gpt55-pro-audit/backlog.md; unit tests for option filters, grids, interpolation, metrics, feature ordering, tuning, forecast storage, training behavior, and runtime preflight; integration tests for early-close handling, Stage03/Stage04 target-gap alignment, HPO-clean Stage05/Stage06 evaluation, Stage07 negative predictions, stats/hedging slices, report artifacts, observed-mask preservation, and the synthetic CPU Stage01-to-Stage09 pipeline.

FINDINGS:

ID: B2-CODE-001
Severity: P1
Area: option cleaning and invalid reason accounting
Title: Non-finite numeric values can pass option cleaning as valid observations
Evidence: src/ivsurf/cleaning/option_filters.py builds invalid_reason using .is_null() checks and threshold comparisons for bid_1545, ask_1545, implied_volatility_1545, vega_1545, active_underlying_price_1545, mid_1545, tau_years, and log_moneyness, then sets is_valid_observation = invalid_reason.is_null(). It does not explicitly reject NaN, +inf, or -inf. src/ivsurf/cleaning/derived_fields.py computes mid_1545, log_moneyness, and total_variance without finite checks. src/ivsurf/io/ingest_cboe.py applies float schema overrides but only asserts non-null quote_date, expiration, root, strike, and option_type. scripts/02_build_option_panel.py counts valid rows directly from is_valid_observation. src/ivsurf/surfaces/grid.py later has non-finite domain checks for tau_years and log_moneyness, but that is after Stage02 has already classified rows as valid option observations. tests/unit/test_option_filters.py covers nulls and threshold violations, but not NaN, infinities, non-positive strikes that produce non-finite logs, or non-finite total variance.
Why it matters: The project doctrine forbids silent type/date/data coercions and requires every filter/drop rule to be explicit, logged, and testable. Polars null checks do not catch NaN, and threshold comparisons are not a complete finite-value validator. A row with non-finite bid/ask/IV/vega/underlying/derived fields can be counted as a valid silver option row, then either contaminate aggregation or be reclassified later as a grid-domain exclusion rather than as a cleaning failure. That breaks invalid-reason accounting and can distort observed-cell values.
Required fix: Add explicit finite-value validation in the option-cleaning stage before threshold rules. Critical numeric raw and derived fields should receive explicit reason codes, for example NONFINITE_BID_1545, NONFINITE_ASK_1545, NONFINITE_IV_1545, NONFINITE_VEGA_1545, NONFINITE_UNDERLYING_1545, NONFINITE_MID_1545, NONFINITE_TAU_YEARS, NONFINITE_LOG_MONEYNESS, NONFINITE_TOTAL_VARIANCE, and a positive-strike rule before log_moneyness is trusted. Align bronze/schema QC with the cleaning contract so non-finite numeric inputs are never silently treated as valid.
Required tests: Add unit tests that inject NaN, +inf, and -inf into each critical numeric field and assert is_valid_observation == False with the expected reason code. Add a Stage02 integration test with raw CSV values such as NaN, inf, and non-positive strike values and assert they are accounted for as cleaning invalids, not valid rows. Add a Stage03 guard test proving no non-finite valid option row can enter surface aggregation.
Hard-cutover/fail-fast note: Treat the expanded invalid-reason taxonomy as a schema cutover. Do not add compatibility readers or fallback mappings that silently reinterpret old silver invalid-reason outputs.

ID: B2-CODE-002
Severity: P1
Area: walk-forward split integrity
Title: Split manifests are serialized as an unversioned bare JSON list
Evidence: src/ivsurf/splits/manifests.py defines WalkforwardSplit with only split_id, train_dates, validation_dates, and test_dates. serialize_splits writes payload = [asdict(split) for split in splits] directly to JSON. load_splits accepts that bare list and reconstructs WalkforwardSplit(**item) with no schema_version, generator identity, date-universe hash, feature dataset hash, sample-window metadata, or version rejection. Later run manifests can record a split manifest hash, for example in scripts/04_build_features.py, but the split artifact itself is not versioned. AGENTS.md explicitly requires split manifests to be explicit, serialized, and versioned.
Why it matters: Walk-forward split artifacts define the clean evaluation boundary. An unversioned split list is not a hard-cutover artifact: stale or schema-incompatible split files can be consumed without a clear failure, and downstream tuning/training cannot prove which generator/config/date universe produced the train/validation/test partitions.
Required fix: Replace the bare list with a versioned envelope, for example schema_version, split_manifest_id or content hash, generator, walkforward_config, sample_window, date_universe_hash, optional feature_dataset_hash, created_at_utc, and splits. Update load_splits to reject any missing or unknown schema_version, including the current bare-list schema.
Required tests: Add unit tests asserting serialized split manifests include the versioned envelope and required metadata. Add a legacy bare-list fixture and assert loading fails with a clear schema-cutover error. Add an integration test covering Stage04 to Stage05/Stage06 split loading and confirming downstream manifests reference the new split hash.
Hard-cutover/fail-fast note: Do not support a compatibility shim for current bare-list split files. Require Stage04 regeneration under the new split-manifest schema.

ID: B2-CODE-003
Severity: P2
Area: model training and feature semantics
Title: HAR factor model assumes positional 1/5/22 lag blocks without validating the feature layout
Evidence: src/ivsurf/config.py allows arbitrary unique lag_windows as long as lag 1 is present; it does not require 5 or 22. src/ivsurf/models/base.py orders feature_surface_mean_{window} columns by numeric window. src/ivsurf/models/har_factor.py then slices the first three target_dim-wide blocks as lag 1, lag 5, and lag 22 features. With lag_windows = (1, 2, 5), the model would label lag 2 as lag_5 and lag 5 as lag_22; with lag_windows = (1,), the second and third blocks can slice into non-HAR features. scripts/06_run_walkforward.py validates the naive model’s lag-1 layout, but there is no corresponding HAR layout validator. The reviewed feature-ordering and training tests cover the naive layout but do not test HAR rejection of incompatible lag configurations.
Why it matters: HAR coefficients are only meaningful if the feature blocks correspond to the intended daily/weekly/monthly lag surfaces. Positional slicing without a schema check can silently train/evaluate a model with mislabeled or wrong features, undermining model comparisons and loss interpretations.
Required fix: Add an explicit HAR feature-layout contract before tuning/training. Either require exact contiguous feature_surface_mean_01, feature_surface_mean_05, and feature_surface_mean_22 target-sized blocks in that order, or refactor HarFactorSurfaceModel to receive column-derived block indices and fail if any required lag block is absent. Apply the check in the model factory or Stage05/Stage06 before fitting.
Required tests: Add unit tests showing HAR accepts the default 1/5/22 feature layout and rejects lag_windows = (1, 2, 5) and lag_windows = (1,) with a clear error before training. Add an integration test proving Stage05/Stage06 fails fast under an incompatible HAR feature config.
Hard-cutover/fail-fast note: Do not infer missing HAR windows from nearby windows, do not pad blocks, and do not fall back to positional slicing when the named schema is absent.

ID: B2-CODE-004
Severity: P2
Area: model artifact and hyperparameter validation
Title: Model factory silently coerces persisted hyperparameter types
Evidence: src/ivsurf/training/model_factory.py defines _float_param and _int_param that accept int | float | str; _float_param calls float(value) and _int_param calls int(value). That means strings such as "5" are accepted and floats such as 5.7 are truncated to 5. make_model_from_params uses these helpers for ridge, elastic net, and HAR parameters. src/ivsurf/models/lightgbm_model.py similarly accepts int | float | str for n_factors and casts it with int(raw_n_factors). src/ivsurf/training/tuning.py permits HyperparameterValue = bool | float | int | str, so a corrupt or stale tuning artifact can be loaded and silently coerced.
Why it matters: The project doctrine explicitly forbids silent type coercions and fallback behavior. Hyperparameter artifacts are part of the evaluation boundary; silently converting "5" or truncating 5.7 changes model definition without a hard failure or schema bump.
Required fix: Replace the permissive helpers with strict model-specific parameter validators. Integer parameters must be true integers, not booleans and not non-integral floats. Float parameters must be numeric, finite, and not strings or booleans. LightGBM n_factors must be a strict positive integer. Persisted tuning results should carry a schema version and model-specific parameter schema.
Required tests: Add unit tests asserting "1.0", "5", 5.7, True, NaN, and infinities are rejected for relevant parameter types, while valid integers and finite floats are accepted. Add a stale/corrupt tuning-manifest fixture and assert Stage06 fails before model construction.
Hard-cutover/fail-fast note: Do not keep string-to-number conversion or integer truncation as a legacy reader. Require regenerated tuning artifacts if old artifacts contain non-strict parameter types.

ID: B2-CODE-005
Severity: P2
Area: forecast alignment, evaluation metrics, and statistical tests
Title: Evaluation does not prove equal forecast coverage across models before loss matrices and stats
Evidence: src/ivsurf/evaluation/alignment.py scans all forecast parquet files and joins forecasts to actuals with validate="m:1", which protects the actual side but does not assert uniqueness or completeness of forecast keys by model. It checks required joined columns and finite predicted total variance, but it does not assert that each evaluated model has exactly the same (quote_date, target_date, split_id, maturity_index, moneyness_index) key set or that there is exactly one forecast row per model/key. scripts/07_run_stats.py builds the daily loss matrix by pivoting model_name over target_date; if one model is missing a date, the pivot can create nulls/NaNs, and the downstream src/ivsurf/stats/{diebold_mariano.py,spa.py,mcs.py} shape checks do not reject non-finite losses. The reviewed stats/hedging tests do not include a missing-forecast-coverage or null-pivot failure case.
Why it matters: Diebold-Mariano, SPA, MCS, and model-ranking comparisons require aligned loss vectors over the same evaluation observations. A partial forecast artifact can silently change the compared sample or inject NaNs into statistical routines, making reported significance and rankings unreliable.
Required fix: Add a forecast-panel contract before metrics/statistics: exactly one row per (model_name, quote_date, target_date, split_id, maturity_index, moneyness_index); all evaluated models must share the identical nonempty key set; duplicate forecast artifacts for the same model/key must fail. _daily_loss_matrix must reject any null or non-finite value after pivoting. DM, SPA, and MCS entry points should also assert finite input arrays.
Required tests: Add fixtures with one model missing a target date or grid cell and assert Stage07 fails before stats. Add a duplicate forecast-row fixture and assert failure. Add direct unit tests proving DM/SPA/MCS reject NaN and infinity inputs. Keep a valid aligned multi-model fixture proving the existing happy path still works.
Hard-cutover/fail-fast note: Do not impute, drop, inner-join, or otherwise shrink the evaluation sample to reconcile mismatched model coverage. Fail and require forecast regeneration.

ID: B2-CODE-006
Severity: P2
Area: reproducibility and provenance
Title: Run manifests can record missing git provenance instead of failing
Evidence: src/ivsurf/reproducibility.py implements git_commit_hash so that subprocess failures, missing git, or missing repository metadata return None. write_run_manifest writes this value directly under git_commit_hash without requiring a nonempty source identifier. In an archive-based or misconfigured run, manifests can therefore contain null git provenance even though AGENTS.md requires reproducibility artifacts to record git/config/data/split/package/seed/hardware metadata.
Why it matters: A null source revision breaks reproducibility and makes audit trails ambiguous. Under the project owner’s hard-cutover rule, “unknown git commit” is not an acceptable fallback value for official artifacts.
Required fix: Make source provenance fail fast. write_run_manifest should require a nonempty immutable source identifier. For normal runs, require a git commit hash and raise a clear error if unavailable. If the project supports official source-archive runs, add an explicit, versioned source_archive_hash contract rather than silently storing git_commit_hash: null.
Required tests: Add tests that monkeypatch git subprocess failure and assert manifest writing fails. Add tests that successful manifest creation includes nonempty source provenance. If source-archive provenance is implemented, add tests requiring the explicit archive hash and rejecting manifests with neither git nor source archive identity.
Hard-cutover/fail-fast note: Do not preserve None, "unknown", or best-effort git fields. Old manifests lacking source provenance should be invalid under the new manifest schema.

ID: B2-CODE-007
Severity: P3
Area: resume behavior and run metadata
Title: Stage06 resumed models are omitted from final model-run metadata
Evidence: In scripts/06_run_walkforward.py, model_run_metadata is initialized for the current run. When resumer.item_complete(model_name) is true, the script appends the existing forecast path and immediately continues. It only adds model_run_metadata[model_name] = item_metadata for models trained in the current invocation. The final Stage06 run manifest includes model_run_metadata, so a resumed run can list forecast artifacts for all models while omitting metadata for the skipped/resumed models.
Why it matters: Resume behavior should not degrade provenance. A final walk-forward manifest must be complete regardless of whether artifacts were produced in the current invocation or resumed from prior successful items. Missing per-model metadata makes it harder to audit config hashes, forecast hashes, and training context for resumed models.
Required fix: When skipping a completed model, load its saved resume metadata, validate that it matches the current config/split/data context and forecast artifact hash, and include it in the final model_run_metadata. Fail if the metadata is missing, stale, or inconsistent.
Required tests: Add a Stage06 resume test where one model is completed before rerun and another model is trained during rerun; assert the final manifest contains validated metadata for both. Add a negative test where resumed metadata is missing or mismatched and assert Stage06 fails.
Hard-cutover/fail-fast note: Do not allow a resumed artifact to be accepted with partial or missing metadata. Require regeneration if resume metadata cannot be validated.

REOPENED_PRIOR_FINDINGS:

None.

NEXT_CODEX_ACTIONS:

Patch option cleaning to explicitly reject non-finite and non-positive critical numeric/derived fields, update invalid-reason schema, and add the required unit/integration tests.

Replace bare-list split manifests with a versioned envelope, reject legacy split files, update Stage04/Stage05/Stage06 consumers, and add split-schema cutover tests.

Add HAR feature-layout validation or column-derived HAR block indexing, then add HAR-compatible and HAR-incompatible config tests.

Replace permissive hyperparameter coercion with strict model-specific validators and schema-versioned tuning artifacts; add corrupt-artifact tests.

Add forecast coverage/cardinality contracts before evaluation/statistics, finite loss-matrix checks, and missing/duplicate forecast tests.

Make run manifest source provenance fail fast when git/source identity is unavailable; add manifest provenance tests.

Fix Stage06 resume metadata inclusion and validation for skipped models; add positive and negative resume tests.

Run the targeted tests above, then the full synthetic Stage01-to-Stage09 pipeline and the existing early-close, target-gap, HPO-clean, stats/hedging, observed-mask, and report-artifact regression suites.

Extended Pro
ChatGPT is AI and can make mistakes. Check important info.
