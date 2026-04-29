ROUND_005_AUDIT_DECISION: findings_found

ATTACHMENT_CHECK:

* econ499_code_audit_context_20260428_round005(1).zip: accessible
* econ499_extracted_text_manifests_20260427(3).zip: accessible
* econ499_lit_review_pdfs_20260427(3).zip: accessible

METHOD:

* Files/modules inspected: `AGENTS.md`; `pyproject.toml`; `configs/data/*`; `configs/eval/*`; `configs/models/*`; `configs/workflow/*`; `scripts/01_ingest_cboe.py` through `scripts/09_make_report_artifacts.py`; `src/ivsurf/schemas.py`; `src/ivsurf/calendar.py`; `src/ivsurf/io/ingest_cboe.py`; `src/ivsurf/cleaning/option_filters.py`; `src/ivsurf/surfaces/grid.py`; `src/ivsurf/surfaces/aggregation.py`; `src/ivsurf/surfaces/interpolation.py`; `src/ivsurf/surfaces/masks.py`; `src/ivsurf/features/tabular_dataset.py`; `src/ivsurf/features/lagged_surface.py`; `src/ivsurf/features/availability.py`; `src/ivsurf/splits/*`; `src/ivsurf/training/tuning.py`; `src/ivsurf/training/model_factory.py`; `src/ivsurf/models/losses.py`; `src/ivsurf/models/neural_surface.py`; `src/ivsurf/models/naive.py`; `src/ivsurf/models/har_factor.py`; `src/ivsurf/models/penalties.py`; `src/ivsurf/evaluation/alignment.py`; `src/ivsurf/evaluation/forecast_store.py`; `src/ivsurf/evaluation/loss_panels.py`; `src/ivsurf/evaluation/slice_reports.py`; `src/ivsurf/hedging/revaluation.py`; report and provenance files included in the uploaded code archive.
* Literature consulted: uploaded extracted literature inventory/manifests; `research/consults/gpt55-pro-audit/literature-dossier.md`; `research/consults/gpt55-pro-audit/upload-manifest.md`; prior local audit literature notes. The findings below are code-contract defects and did not require reopening original PDFs for exact equations or figures.
* Tests/contracts reviewed: `AGENTS.md` project doctrine; Python version contract in `pyproject.toml`; unit, integration, property, regression, and e2e tests under `tests/`, including training behavior, tuning workflow, losses, split manifest, feature timing, surface/grid, evaluation/loss panel, hedging, report, and resume-related tests. I reviewed the tests and reproduction evidence in the archive; I did not run the full suite in this container.
* Prior audit logs/backlog reviewed: `research/consults/gpt55-pro-audit/backlog.md`; Round 003, Round 004, Round 004A, and Round 004B audit/verification logs, including the Round 004B statement that B2-CODE-001 through B2-CODE-007 were pro-verified fixed.
* Reproduction/provenance evidence reviewed: provenance summaries and run metadata in `provenance/`, including `hpo_30_trials__train_30_epochs__mac_cpu.json`, tuning diagnostics exports, workflow manifests, repo dossier, and audit upload manifests.

FINDINGS:
ID: B3-CODE-001
Severity: P2
Area: model artifact and hyperparameter validation
Title: Tuning manifests are not schema-versioned, and several persisted model-parameter paths still ignore unexpected keys.
Evidence: `src/ivsurf/training/tuning.py::TuningResult` has strict Pydantic field validation but no required `schema_version` or tuning-result schema constant. `write_tuning_result` writes the current model dump, and `load_tuning_result` validates the current Pydantic shape but cannot reject stale or legacy tuning manifests by explicit artifact version. In `src/ivsurf/training/model_factory.py`, `_reject_extra_params` exists and is used for LightGBM and random forest, while the `ridge`, `elasticnet`, and `har_factor` branches in `make_model_from_params` read required keys without rejecting unexpected persisted keys. Existing tests verify some numeric type strictness and profile loading behavior, but they do not assert hard-cutover rejection of missing/wrong tuning schema versions or unexpected keys for Ridge, ElasticNet, and HAR.
Why it matters: The project owner’s hard-cutover rule forbids fallback paths, silent legacy readers, and ignored unsafe artifact fields. A stale or tampered tuning artifact can carry irrelevant hyperparameters that are silently ignored for some model families, making the recorded tuning provenance and model configuration hash less trustworthy.
Required fix: Add a required tuning artifact schema version, for example `TUNING_RESULT_SCHEMA_VERSION = "tuning_result_v2"`, to `TuningResult`; write it in every stage 05 tuning result; make `load_tuning_result` reject missing or wrong versions with an explicit “rerun stage 05” failure. Apply the same unexpected-key rejection contract to every `make_model_from_params` branch, including `ridge`, `elasticnet`, `har_factor`, and any model type expected to have empty parameters.
Required tests: Add unit tests proving `load_tuning_result` rejects manifests without `schema_version` and with a wrong version. Add unit tests proving `make_model_from_params` rejects extra keys for Ridge, ElasticNet, HAR, random forest, LightGBM, and neural configs. Add an integration test showing stage 06 refuses a stale tuning manifest with an irrelevant added parameter before fitting or forecasting.
Hard-cutover/fail-fast note: Do not default a missing tuning schema version and do not add compatibility shims for old tuning artifacts. Require hard artifact regeneration.

ID: B3-CODE-002
Severity: P1
Area: neural loss definitions and training input validation
Title: `weighted_surface_mse` silently clamps invalid training weights and can propagate non-finite losses instead of failing fast.
Evidence: `src/ivsurf/models/losses.py::weighted_surface_mse` computes observation weights and then applies `torch.clamp_min(training_weights, 0.0)`, silently converting negative training weights into zeros. The function does not explicitly validate same-shape tensors, finite predictions, finite targets, finite masks, finite weights, nonnegative weights, binary observed masks, or finite observed/imputed loss-weight scalars. If `training_weights` contains NaN, the `weight_sum` check can also become NaN and fail to trigger the current nonpositive-weight guard. `src/ivsurf/models/neural_surface.py` passes batch tensors directly into this loss during training; the existing total-variance validation covers predictions, not target/mask/weight corruption. `tests/unit/test_losses.py` currently verifies only that changing the imputed-loss weight changes the loss; it does not test invalid tensors.
Why it matters: Neural loss definitions and training artifacts are in the closure audit scope. Silent clipping of negative training weights violates the no-silent-coercion and fail-fast doctrine. Nonfinite or corrupted weights can affect optimizer behavior, HPO pruning, and validation diagnostics without a deterministic contract failure.
Required fix: Remove the `torch.clamp_min` behavior. Add explicit validation at the loss boundary, or immediately before loss invocation, requiring equal tensor shapes, finite predictions and targets, binary/boolean observed masks, finite and nonnegative training weights, finite observed/imputed loss weights, and strictly positive supervised total weight. Invalid artifacts should raise a clear exception before backpropagation.
Required tests: Add unit tests rejecting negative training weights, NaN/Inf training weights, NaN/Inf targets, NaN/Inf predictions, non-binary observed masks, tensor shape mismatches, negative or NaN loss-weight scalars, and zero total supervised weight. Add an integration test proving a corrupted `target_training_weight_*` feature artifact causes the neural stage 05 or stage 06 path to fail before any optimizer step.
Hard-cutover/fail-fast note: Do not clip, sanitize, or reinterpret invalid tensor values. Abort on invalid weights, masks, targets, or predictions.

ID: B3-CODE-003
Severity: P1
Area: evaluation metrics and observed-cell weight contracts
Title: Stage 05 and Stage 07 observed-weight scoring still coerces or masks invalid vega weights instead of enforcing the observed-cell weight contract.
Evidence: `src/ivsurf/evaluation/loss_panels.py::daily_loss_metric_values` casts observed masks via thresholding and computes observed weights with `np.maximum(vega_weights[row_index], 0.0)`, silently clamping negative target-day vega weights to zero. This shared path is used for daily observed loss metrics and stage 05 validation scoring through `mean_daily_loss_metric`; it is also used by neural validation diagnostics. `scripts/05_tune_models.py::_legacy_pooled_validation_score` likewise multiplies observed masks by `np.maximum(vega_weights, 0.0)` and still persists the diagnostic as `legacy_pooled_observed_wrmse`. `loss_panels._has_positive_total_weight` and `slice_reports._has_positive_total_weight` check finite positive sums but not elementwise nonnegativity. `src/ivsurf/evaluation/alignment.py::build_forecast_realization_panel` derives `observed_weight` from `actual_vega_sum` without a strict post-load assertion that observed-cell weights are finite and nonnegative. Existing tests check positive-weight equivalence between stage 05 and stage 07 but do not inject negative, nonfinite, or non-binary weight/mask corruption.
Why it matters: Observed-cell metrics are central to model selection, evaluation, slice reports, and statistical tests. If target-day observed weights are silently clipped or converted into NaN report rows, HPO decisions and final metrics can be inconsistent with the project’s observed-mask and fail-fast contracts.
Required fix: Introduce a shared strict observed-metric validator for NumPy/Polars evaluation paths. It should require shape equality across truth, prediction, observed mask, and vega-weight arrays; finite numeric values; observed masks that are boolean or exactly 0/1; finite nonnegative vega weights; and strictly positive observed total weight where an observed metric is requested. Remove `np.maximum(..., 0.0)` from official and persisted diagnostic scoring paths, or delete the legacy pooled diagnostic if it is no longer needed.
Required tests: Add unit tests proving `daily_loss_metric_values` and `mean_daily_loss_metric` reject negative observed-cell vega weights, NaN/Inf vega weights, non-binary masks, shape mismatches, and zero observed total weight for observed metrics. Add integration tests showing stage 05 fails on corrupted `target_vega_weight_*` and stage 07 fails on corrupted realized surface `vega_sum` before writing official loss artifacts. Preserve the existing positive-weight equivalence test between stage 05 and stage 07.
Hard-cutover/fail-fast note: Do not clip negative observed weights, silently zero them, or downgrade invalid observed-weight artifacts into NaN report rows. Invalid observed-weight artifacts should abort.

ID: B3-CODE-004
Severity: P2
Area: surface grid integrity, feature construction, hedging, and reports
Title: Several dense-surface reshape paths do not assert per-date grid-key completeness and uniqueness before reshaping.
Evidence: `src/ivsurf/features/lagged_surface.py::pivot_surface_arrays` sorts by `quote_date`, `maturity_index`, and `moneyness_index`, then reshapes surface columns into dense arrays, but it does not assert exactly one row for every `(quote_date, maturity_index, moneyness_index)` or that index/coordinate pairs match the configured grid. `scripts/04_build_features.py` calls `require_surface_grid_metadata` before feature construction, but that metadata check does not prove row-level grid completeness or uniqueness. `src/ivsurf/surfaces/masks.py::_ordered_surface_frame` checks row count before reshape but not duplicate/missing grid keys. `src/ivsurf/hedging/revaluation.py::surface_interpolator_from_frame` derives coordinate counts and reshapes long-form surface data without duplicate or missing key validation.
Why it matters: Fixed grid cell identity and observed masks are first-class artifacts in this project. A corrupted gold or forecast artifact with one duplicated grid cell and one missing grid cell can preserve the expected row count while permuting feature vectors, target vectors, masks, or hedging valuations. That is an undetected artifact-integrity failure at saved-artifact reader boundaries.
Required fix: Add a shared contract such as `require_complete_unique_surface_grid(frame, grid, key_date_columns=..., context=...)`. It should check required columns, no null grid keys, uniqueness of date/grid keys, exact row count per date or forecast group, index ranges, and consistency between `maturity_index`/`moneyness_index` and configured maturity/moneyness coordinate values. Call it before dense reshapes in feature construction, mask reshaping, hedging interpolation, and any report/evaluation artifact-reader path that assumes a complete dense grid.
Required tests: Add unit tests showing each reshape/interpolator path rejects duplicate grid keys, missing grid keys, out-of-range indexes, and mismatched coordinate values even when total row count still equals grid size. Add an integration test showing stage 04 fails on a corrupted gold surface file with a duplicate/missing cell pair. Add a hedging test showing `surface_interpolator_from_frame` rejects duplicate or missing forecast cells before valuation.
Hard-cutover/fail-fast note: Do not reindex, deduplicate, infer, or fill missing grid cells at these artifact-reader boundaries. Corrupted artifacts should fail and require rebuild from the appropriate upstream stage.

REOPENED_PRIOR_FINDINGS:

* None.

TERMINAL_AUDIT_STATEMENT:

* Not reached.

NEXT_CODEX_ACTIONS:

* Implement B3-CODE-001: add tuning-result schema versioning, reject stale/missing versions, and reject unexpected model parameters for every model family.
* Implement B3-CODE-002: replace neural loss clipping with strict tensor/value validation and add corrupted-feature integration coverage.
* Implement B3-CODE-003: add one shared strict observed-weight/mask validation contract for stage 05, stage 07, slice reports, and persisted diagnostics; remove observed-weight clipping.
* Implement B3-CODE-004: add a complete-and-unique surface-grid contract and apply it to all dense reshape/interpolator artifact-reader boundaries.
* Rerun the targeted unit, integration, property, regression, and smoke tests under Python 3.13.5, then refresh the relevant provenance/run evidence and backlog statuses.
