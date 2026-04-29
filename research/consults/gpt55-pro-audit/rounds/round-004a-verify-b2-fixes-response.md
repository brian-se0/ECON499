ROUND_004A_VERIFICATION_DECISION: still_has_issues

Finding-by-finding status:

* B2-CODE-001: fixed
  `src/ivsurf/cleaning/option_filters.py` now rejects null, non-finite, and non-positive critical inputs with explicit reason codes, including `NONFINITE_*`, `NON_POSITIVE_STRIKE`, and `NON_POSITIVE_TOTAL_VARIANCE`. `scripts/03_build_surfaces.py::_require_valid_surface_inputs` adds a Stage03 fail-fast guard against valid silver rows containing null/non-finite/non-positive surface inputs. The added unit and integration tests directly cover non-finite cleaning and Stage03 rejection.

* B2-CODE-002: fixed
  `src/ivsurf/splits/manifests.py` now writes a versioned manifest envelope with `schema_version`, `generator`, `walkforward_config`, `sample_window`, `date_universe_hash`, `feature_dataset_hash`, and `splits`. `load_split_manifest` rejects legacy bare-list manifests. `scripts/04_build_features.py` ties the split manifest to the actual `daily_features.parquet` hash, and `scripts/05_tune_models.py` / `scripts/06_run_walkforward.py` validate the manifest against the current feature artifact before use. The new `tests/unit/test_split_manifests.py` covers versioned serialization, legacy rejection, date-universe mismatch, and feature-hash mismatch.

* B2-CODE-003: fixed
  `src/ivsurf/models/har_factor.py::validate_har_feature_layout` now requires exact leading lag-1, lag-5, and lag-22 `feature_surface_mean_*` blocks aligned to the target grid suffixes. `scripts/05_tune_models.py` and `scripts/06_run_walkforward.py` call the guard before HAR tuning/training. The new tests cover both accepted and reordered/missing HAR lag layouts.

* B2-CODE-004: fixed for the scoped coercion defect
  `src/ivsurf/training/model_factory.py` now rejects string numerics, booleans, non-finite floats, and floats for integer parameters. `src/ivsurf/models/lightgbm_model.py` now requires integer `n_factors` without string/float coercion. The added tests verify rejection of `"1.0"` for ridge alpha, float `max_iter`, and string LightGBM integer parameters. A later full audit should still review whether tuning manifests themselves need an explicit schema-version bump and model-specific extra-key rejection for every model class, but the Round 004A coercion issue is fixed.

* B2-CODE-005: fixed
  `src/ivsurf/evaluation/alignment.py` now checks duplicate forecast keys and equal forecast coverage across models before realization alignment. `scripts/07_run_stats.py::_daily_loss_matrix` now rejects duplicate model/date rows, missing pivot coverage, and non-finite losses. `src/ivsurf/stats/{diebold_mariano.py,spa.py,mcs.py}` now reject empty and non-finite statistical inputs before computing tests. The added tests cover duplicate forecasts, unequal model coverage, and non-finite statistical inputs.

* B2-CODE-006: fixed
  `src/ivsurf/reproducibility.py::write_run_manifest` now calls `git_commit_hash(repo_root)` once and raises `RuntimeError` if no commit hash is available, instead of writing `git_commit_hash: null`. `tests/unit/test_reporting_helpers.py::test_write_run_manifest_requires_git_commit_hash` verifies the fail-fast behavior.

* B2-CODE-007: partially fixed
  The direct omission is fixed: `scripts/06_run_walkforward.py` now adds `model_run_metadata[model_name] = resumer.metadata_for(model_name)` when a model is skipped through resume, and the integration test checks that resumed/skipped models appear in the final Stage06 manifest.
  The remaining issue is that the resumed metadata is not validated before inclusion. `StageResumer.item_complete` only checks that the output path exists, and `metadata_for` can return an empty or stale metadata dict. The patch does not compute and compare a stored forecast artifact hash for the skipped model, nor does it require the resumed metadata to contain the expected Stage06 fields before writing the final run manifest.

Remaining issue requiring a minimum fix:

* File/function: `scripts/06_run_walkforward.py`, resumed-model branch inside the model loop; supporting behavior in `src/ivsurf/resume.py::metadata_for` / `item_complete`.
* Minimum required fix: when marking a Stage06 model complete, store a `forecast_artifact_hash` for the model output and enough required metadata to validate the model record. When skipping a completed model, compute `sha256_file(output_path)`, compare it to the stored hash, and require non-empty metadata containing at least `model_name`, `n_splits`, `n_forecast_rows`, `workflow_run_label`, `max_hpo_validation_date`, `first_clean_test_split_id`, `model_config_hash`, `training_run_id`, and `surface_config_hash`. Raise immediately if any field is missing, stale, or hash-mismatched.
* Minimum required tests: add a positive Stage06 resume test asserting the final manifest includes validated metadata for both newly run and skipped models, and negative tests where the resumed metadata is empty/missing and where the resumed forecast parquet is modified after completion; both should fail before final manifest writing.
