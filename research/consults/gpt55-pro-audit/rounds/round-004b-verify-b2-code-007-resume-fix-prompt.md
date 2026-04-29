You are GPT-5.5 Pro acting as the audit brain for the ECON499 SPX implied-volatility-surface forecasting project. Codex is the implementation agent.

Context: You just completed Round 004A verification and returned `ROUND_004A_VERIFICATION_DECISION: still_has_issues`. You found B2-CODE-001 through B2-CODE-006 fixed, B2-CODE-004 fixed for the scoped coercion defect, and only B2-CODE-007 partially fixed.

Task: Verify the new Round 004B fix for the remaining B2-CODE-007 resume-integrity issue. Focus on the exact remaining issue you identified: Stage06 skipped/resumed model metadata must be validated before final manifest inclusion, including stored forecast artifact hash, required metadata fields, current-context values, and stale/tampered artifact detection. Do not re-audit the whole project in this response unless the new patch directly introduces a concrete regression.

Project rules that matter here:
- No fallback paths.
- No backwards compatibility or legacy branches; hard cutover only.
- Fail fast on stale, malformed, or missing resume artifacts/metadata.
- No silent partial metadata in final run manifests.

Codex implemented:
- `scripts/06_run_walkforward.py`
  - Stores `forecast_artifact_hash` in Stage06 per-model resume metadata after writing forecasts.
  - Computes model config hash and training run id before the resume branch, so resumed records can be checked against the current run context.
  - Adds strict resume metadata validation requiring: `model_name`, `n_splits`, `n_forecast_rows`, `workflow_run_label`, `max_hpo_validation_date`, `first_clean_test_split_id`, `model_config_hash`, `training_run_id`, `surface_config_hash`, and `forecast_artifact_hash`.
  - On a resume skip, recomputes `sha256_file(output_path)`, compares it to stored `forecast_artifact_hash`, and validates required metadata values against the current clean-evaluation policy and model/run context.
  - Uses Polars lazy scan of the forecast parquet to confirm the artifact is non-empty, has the expected unique model/config/run/surface values, and that its unique quote-date count matches `n_forecast_rows`.
  - Raises `ValueError` before final run-manifest writing if resumed metadata is missing/stale or the forecast artifact was modified after completion.
- `tests/integration/test_stage05_stage06_clean_evaluation.py`
  - Positive resume test now asserts final Stage06 manifest includes validated metadata and matching forecast hashes for both newly run and skipped models.
  - Negative checks verify empty resumed metadata fails before final manifest writing.
  - Negative checks verify a modified completed forecast parquet fails via `forecast_artifact_hash` before final manifest writing.

Verification Codex ran locally:
- `uv run python -m ruff check src/ivsurf/cleaning/option_filters.py src/ivsurf/splits/manifests.py src/ivsurf/models/har_factor.py src/ivsurf/training/model_factory.py src/ivsurf/models/lightgbm_model.py src/ivsurf/evaluation/alignment.py src/ivsurf/evaluation/loss_panels.py src/ivsurf/stats/diebold_mariano.py src/ivsurf/stats/spa.py src/ivsurf/stats/mcs.py src/ivsurf/reproducibility.py scripts/02_build_option_panel.py scripts/03_build_surfaces.py scripts/04_build_features.py scripts/05_tune_models.py scripts/06_run_walkforward.py scripts/07_run_stats.py tests/unit/test_option_filters.py tests/unit/test_split_manifests.py tests/unit/test_training_behaviour.py tests/unit/test_stats.py tests/unit/test_alignment.py tests/unit/test_reporting_helpers.py tests/integration/test_early_close_stage02.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage05_stage06_clean_evaluation.py`
  - Result: All checks passed.
- `uv run python -m pytest tests/unit/test_option_filters.py tests/unit/test_split_manifests.py tests/unit/test_training_behaviour.py tests/unit/test_stats.py tests/unit/test_alignment.py tests/unit/test_reporting_helpers.py tests/integration/test_early_close_stage02.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage05_stage06_clean_evaluation.py`
  - Result: 74 passed in 4.99s.

Your prior Round 004A response:

```text
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

```

Current patch for the affected files:

```diff
diff --git a/scripts/06_run_walkforward.py b/scripts/06_run_walkforward.py
index 5053862..6e12e94 100644
--- a/scripts/06_run_walkforward.py
+++ b/scripts/06_run_walkforward.py
@@ -3,9 +3,10 @@ from __future__ import annotations
 from datetime import UTC, datetime
 from pathlib import Path

-import numpy as np
-import polars as pl
-import typer
+import numpy as np
+import orjson
+import polars as pl
+import typer

 from ivsurf.config import (
     EvaluationMetricsConfig,
@@ -20,11 +21,16 @@ from ivsurf.evaluation.forecast_store import write_forecasts
 from ivsurf.exceptions import ModelConvergenceError
 from ivsurf.io.paths import sorted_artifact_files
 from ivsurf.models.base import dataset_to_matrices
+from ivsurf.models.har_factor import validate_har_feature_layout
 from ivsurf.models.naive import validate_naive_feature_layout
 from ivsurf.progress import create_progress
-from ivsurf.reproducibility import write_run_manifest
+from ivsurf.reproducibility import sha256_bytes, sha256_file, write_run_manifest
 from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
-from ivsurf.splits.manifests import WalkforwardSplit, load_splits
+from ivsurf.splits.manifests import (
+    WalkforwardSplit,
+    load_split_manifest,
+    require_split_manifest_matches_artifacts,
+)
 from ivsurf.splits.walkforward import clean_evaluation_splits
 from ivsurf.surfaces.grid import SurfaceGrid
 from ivsurf.training.fit_lightgbm import fit_and_predict_lightgbm
@@ -48,7 +54,7 @@ def _indices_for_dates(all_dates: np.ndarray, subset: tuple[str, ...]) -> np.nda
     return np.asarray([lookup[item] for item in subset], dtype=np.int64)


-def _merged_params(
+def _merged_params(
     base_params: dict[str, object],
     tuning_result: TuningResult,
     *,
@@ -63,14 +69,51 @@ def _merged_params(
         raise ValueError(message)
     merged = dict(base_params)
     merged.update(tuning_result.best_params)
-    return merged
-
-
-def _require_split_boundary_match(
-    policy: CleanEvaluationPolicy,
-    *,
-    split_manifest_splits: list[WalkforwardSplit],
-) -> list[WalkforwardSplit]:
+    return merged
+
+
+def _stable_payload_hash(payload: dict[str, object]) -> str:
+    return sha256_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))
+
+
+def _model_config_hash(
+    *,
+    model_name: str,
+    base_params: dict[str, object],
+    tuned_params: dict[str, object],
+    hpo_profile_name: str,
+    training_profile_name: str,
+) -> str:
+    return _stable_payload_hash(
+        {
+            "model_name": model_name,
+            "base_params": base_params,
+            "tuned_params": tuned_params,
+            "hpo_profile_name": hpo_profile_name,
+            "training_profile_name": training_profile_name,
+        }
+    )
+
+
+def _single_feature_metadata_value(feature_frame: pl.DataFrame, column_name: str) -> str:
+    if column_name not in feature_frame.columns:
+        message = f"daily_features.parquet is missing required metadata column {column_name}."
+        raise ValueError(message)
+    values = feature_frame[column_name].unique().to_list()
+    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
+        message = (
+            "daily_features.parquet must contain exactly one non-empty "
+            f"{column_name}; found {values!r}."
+        )
+        raise ValueError(message)
+    return values[0]
+
+
+def _require_split_boundary_match(
+    policy: CleanEvaluationPolicy,
+    *,
+    split_manifest_splits: list[WalkforwardSplit],
+) -> list[WalkforwardSplit]:
     boundary, clean_splits = clean_evaluation_splits(
         split_manifest_splits,
         tuning_splits_count=policy.tuning_splits_count,
@@ -86,9 +129,184 @@ def _require_split_boundary_match(
         message = (
             "Split manifest clean evaluation start does not match the tuning manifests: "
             f"{boundary.first_clean_test_split_id!r} != {policy.first_clean_test_split_id!r}."
-        )
-        raise ValueError(message)
-    return clean_splits
+        )
+        raise ValueError(message)
+    return clean_splits
+
+
+def _require_metadata_string(
+    metadata: dict[str, object],
+    field_name: str,
+    *,
+    expected_value: str | None = None,
+) -> str:
+    value = metadata.get(field_name)
+    if not isinstance(value, str) or not value:
+        message = f"Stage 06 resume metadata field {field_name!r} must be a non-empty string."
+        raise ValueError(message)
+    if expected_value is not None and value != expected_value:
+        message = (
+            f"Stage 06 resume metadata field {field_name!r} is stale: "
+            f"{value!r} != {expected_value!r}."
+        )
+        raise ValueError(message)
+    return value
+
+
+def _require_metadata_int(
+    metadata: dict[str, object],
+    field_name: str,
+    *,
+    expected_value: int | None = None,
+) -> int:
+    value = metadata.get(field_name)
+    if not isinstance(value, int) or isinstance(value, bool):
+        message = f"Stage 06 resume metadata field {field_name!r} must be an integer."
+        raise ValueError(message)
+    if expected_value is not None and value != expected_value:
+        message = (
+            f"Stage 06 resume metadata field {field_name!r} is stale: "
+            f"{value!r} != {expected_value!r}."
+        )
+        raise ValueError(message)
+    return value
+
+
+def _forecast_artifact_summary(output_path: Path) -> dict[str, object]:
+    summary = (
+        pl.scan_parquet(output_path)
+        .select(
+            pl.len().alias("row_count"),
+            pl.col("quote_date").n_unique().alias("n_forecast_rows"),
+            pl.col("model_name").n_unique().alias("model_name_unique_count"),
+            pl.col("model_name").first().alias("model_name"),
+            pl.col("model_config_hash").n_unique().alias("model_config_hash_unique_count"),
+            pl.col("model_config_hash").first().alias("model_config_hash"),
+            pl.col("training_run_id").n_unique().alias("training_run_id_unique_count"),
+            pl.col("training_run_id").first().alias("training_run_id"),
+            pl.col("surface_config_hash").n_unique().alias("surface_config_hash_unique_count"),
+            pl.col("surface_config_hash").first().alias("surface_config_hash"),
+        )
+        .collect()
+        .row(0, named=True)
+    )
+    return dict(summary)
+
+
+def _require_single_forecast_artifact_value(
+    summary: dict[str, object],
+    *,
+    count_field_name: str,
+    value_field_name: str,
+    expected_value: str,
+    output_path: Path,
+) -> None:
+    unique_count = summary[count_field_name]
+    value = summary[value_field_name]
+    if unique_count != 1 or value != expected_value:
+        message = (
+            f"Stage 06 resumed forecast artifact {output_path} has stale "
+            f"{value_field_name}: count={unique_count!r}, value={value!r}, "
+            f"expected={expected_value!r}."
+        )
+        raise ValueError(message)
+
+
+def _validated_stage06_resume_metadata(
+    metadata: dict[str, object],
+    *,
+    model_name: str,
+    output_path: Path,
+    n_splits: int,
+    workflow_run_label: str,
+    max_hpo_validation_date: str,
+    first_clean_test_split_id: str,
+    model_config_hash: str,
+    training_run_id: str,
+    surface_config_hash: str,
+) -> dict[str, object]:
+    forecast_artifact_hash = sha256_file(output_path)
+    _require_metadata_string(metadata, "model_name", expected_value=model_name)
+    _require_metadata_int(metadata, "n_splits", expected_value=n_splits)
+    n_forecast_rows = _require_metadata_int(metadata, "n_forecast_rows")
+    if n_forecast_rows <= 0:
+        message = "Stage 06 resume metadata field 'n_forecast_rows' must be positive."
+        raise ValueError(message)
+    _require_metadata_string(
+        metadata,
+        "workflow_run_label",
+        expected_value=workflow_run_label,
+    )
+    _require_metadata_string(
+        metadata,
+        "max_hpo_validation_date",
+        expected_value=max_hpo_validation_date,
+    )
+    _require_metadata_string(
+        metadata,
+        "first_clean_test_split_id",
+        expected_value=first_clean_test_split_id,
+    )
+    _require_metadata_string(
+        metadata,
+        "model_config_hash",
+        expected_value=model_config_hash,
+    )
+    _require_metadata_string(
+        metadata,
+        "training_run_id",
+        expected_value=training_run_id,
+    )
+    _require_metadata_string(
+        metadata,
+        "surface_config_hash",
+        expected_value=surface_config_hash,
+    )
+    _require_metadata_string(
+        metadata,
+        "forecast_artifact_hash",
+        expected_value=forecast_artifact_hash,
+    )
+
+    summary = _forecast_artifact_summary(output_path)
+    if summary["row_count"] == 0:
+        message = f"Stage 06 resumed forecast artifact {output_path} is empty."
+        raise ValueError(message)
+    if summary["n_forecast_rows"] != n_forecast_rows:
+        message = (
+            "Stage 06 resume metadata field 'n_forecast_rows' does not match "
+            f"{output_path}: {n_forecast_rows!r} != {summary['n_forecast_rows']!r}."
+        )
+        raise ValueError(message)
+    _require_single_forecast_artifact_value(
+        summary,
+        count_field_name="model_name_unique_count",
+        value_field_name="model_name",
+        expected_value=model_name,
+        output_path=output_path,
+    )
+    _require_single_forecast_artifact_value(
+        summary,
+        count_field_name="model_config_hash_unique_count",
+        value_field_name="model_config_hash",
+        expected_value=model_config_hash,
+        output_path=output_path,
+    )
+    _require_single_forecast_artifact_value(
+        summary,
+        count_field_name="training_run_id_unique_count",
+        value_field_name="training_run_id",
+        expected_value=training_run_id,
+        output_path=output_path,
+    )
+    _require_single_forecast_artifact_value(
+        summary,
+        count_field_name="surface_config_hash_unique_count",
+        value_field_name="surface_config_hash",
+        expected_value=surface_config_hash,
+        output_path=output_path,
+    )
+    return dict(metadata)


 @app.command()
@@ -116,9 +334,10 @@ def main(
     hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
     training_profile = TrainingProfileConfig.model_validate(
         load_yaml_config(training_profile_config_path)
-    )
-    grid = SurfaceGrid.from_config(surface_config)
-    workflow_paths = resolve_workflow_run_paths(
+    )
+    grid = SurfaceGrid.from_config(surface_config)
+    current_surface_config_hash = sha256_file(surface_config_path)
+    workflow_paths = resolve_workflow_run_paths(
         raw_config,
         hpo_profile_name=hpo_profile.profile_name,
         training_profile_name=training_profile.profile_name,
@@ -146,27 +365,42 @@ def main(
                 hpo_profile_config_path,
                 training_profile_config_path,
             ],
-            input_artifact_paths=[
-                raw_config.gold_dir / "daily_features.parquet",
-                split_manifest_path,
-                *tuning_manifest_paths,
+            input_artifact_paths=[
+                raw_config.gold_dir / "daily_features.parquet",
+                split_manifest_path,
+                *tuning_manifest_paths,
             ],
             extra_tokens={
+                "artifact_schema_version": 4,
                 "run_profile_name": run_profile_name,
                 "workflow_run_label": workflow_paths.run_label,
             },
         ),
-    )
-
-    feature_frame = pl.read_parquet(raw_config.gold_dir / "daily_features.parquet").sort(
-        "quote_date"
-    )
-    matrices = dataset_to_matrices(feature_frame)
+    )
+
+    feature_frame = pl.read_parquet(raw_config.gold_dir / "daily_features.parquet").sort(
+        "quote_date"
+    )
+    feature_dataset_hash = sha256_file(raw_config.gold_dir / "daily_features.parquet")
+    split_manifest = load_split_manifest(split_manifest_path)
+    require_split_manifest_matches_artifacts(
+        split_manifest,
+        date_universe=feature_frame["quote_date"].to_list(),
+        feature_dataset_hash=feature_dataset_hash,
+    )
+    surface_config_hash = _single_feature_metadata_value(feature_frame, "surface_config_hash")
+    if surface_config_hash != current_surface_config_hash:
+        message = (
+            "daily_features.parquet surface_config_hash does not match the current surface "
+            f"config file hash: {surface_config_hash!r} != {current_surface_config_hash!r}."
+        )
+        raise ValueError(message)
+    matrices = dataset_to_matrices(feature_frame)
     validate_naive_feature_layout(
         feature_columns=matrices.feature_columns,
         target_columns=matrices.target_columns,
     )
-    splits = load_splits(split_manifest_path)
+    splits = split_manifest.splits

     ridge_params = load_yaml_config(ridge_config_path)
     elasticnet_params = load_yaml_config(elasticnet_config_path)
@@ -221,24 +455,69 @@ def main(
             message = f"Unknown model requested via only_model: {only_model!r}."
             raise ValueError(message)
         selected_model_names = (only_model,)
+    if "har_factor" in selected_model_names:
+        validate_har_feature_layout(
+            feature_columns=matrices.feature_columns,
+            target_columns=matrices.target_columns,
+        )
     model_run_metadata: dict[str, dict[str, object]] = {}
     total_steps = len(selected_model_names) * len(clean_splits)
     with create_progress() as progress:
         task_id = progress.add_task("Stage 06 walk-forward forecasting", total=total_steps)
         for model_name in selected_model_names:
-            output_path = workflow_paths.forecast_dir / f"{model_name}.parquet"
-            if resumer.item_complete(model_name, required_output_paths=[output_path]):
-                progress.update(
-                    task_id,
-                    description=f"Stage 06 resume: skipping completed model {model_name}",
-                )
-                progress.advance(task_id, advance=len(clean_splits))
-                continue
-            resumer.clear_item(model_name, output_paths=[output_path])
-            prediction_blocks: list[np.ndarray] = []
-            quote_date_blocks: list[np.ndarray] = []
-            target_date_blocks: list[np.ndarray] = []
-            model_metadata: dict[str, object] = {}
+            output_path = workflow_paths.forecast_dir / f"{model_name}.parquet"
+            model_config_hash = _model_config_hash(
+                model_name=model_name,
+                base_params={} if model_name == "naive" else base_param_map[model_name],
+                tuned_params={} if model_name == "naive" else tuned_param_map[model_name],
+                hpo_profile_name=hpo_profile.profile_name,
+                training_profile_name=training_profile.profile_name,
+            )
+            training_run_id = _stable_payload_hash(
+                {
+                    "stage": "06_run_walkforward",
+                    "model_name": model_name,
+                    "workflow_run_label": workflow_paths.run_label,
+                    "resume_context_hash": resumer.context_hash,
+                    "model_config_hash": model_config_hash,
+                    "surface_config_hash": surface_config_hash,
+                }
+            )
+            if resumer.item_complete(model_name, required_output_paths=[output_path]):
+                progress.update(
+                    task_id,
+                    description=f"Stage 06 resume: skipping completed model {model_name}",
+                )
+                model_run_metadata[model_name] = _validated_stage06_resume_metadata(
+                    resumer.metadata_for(model_name),
+                    model_name=model_name,
+                    output_path=output_path,
+                    n_splits=len(clean_splits),
+                    workflow_run_label=workflow_paths.run_label,
+                    max_hpo_validation_date=(
+                        clean_evaluation_policy.max_hpo_validation_date.isoformat()
+                    ),
+                    first_clean_test_split_id=(
+                        clean_evaluation_policy.first_clean_test_split_id
+                    ),
+                    model_config_hash=model_config_hash,
+                    training_run_id=training_run_id,
+                    surface_config_hash=surface_config_hash,
+                )
+                progress.advance(task_id, advance=len(clean_splits))
+                continue
+            resumer.clear_item(model_name, output_paths=[output_path])
+            prediction_blocks: list[np.ndarray] = []
+            quote_date_blocks: list[np.ndarray] = []
+            target_date_blocks: list[np.ndarray] = []
+            split_id_blocks: list[np.ndarray] = []
+            decision_timestamp_blocks: list[np.ndarray] = []
+            target_decision_timestamp_blocks: list[np.ndarray] = []
+            model_metadata: dict[str, object] = {
+                "model_config_hash": model_config_hash,
+                "training_run_id": training_run_id,
+                "surface_config_hash": surface_config_hash,
+            }
             for split in clean_splits:
                 split_id = split.split_id
                 progress.update(
@@ -325,35 +604,64 @@ def main(
                         )
                         raise RuntimeError(message) from exc
                     raise
-                prediction_blocks.append(predictions)
-                quote_date_blocks.append(matrices.quote_dates[test_index])
-                target_date_blocks.append(matrices.target_dates[test_index])
-                progress.advance(task_id)
+                prediction_blocks.append(predictions)
+                quote_date_blocks.append(matrices.quote_dates[test_index])
+                target_date_blocks.append(matrices.target_dates[test_index])
+                split_id_blocks.append(
+                    np.full(test_index.shape[0], split_id, dtype=object)
+                )
+                decision_timestamp_blocks.append(matrices.decision_timestamps[test_index])
+                target_decision_timestamp_blocks.append(
+                    matrices.target_decision_timestamps[test_index]
+                )
+                progress.advance(task_id)

             write_forecasts(
                 output_path=workflow_paths.forecast_dir / f"{model_name}.parquet",
-                model_name=model_name,
-                quote_dates=np.concatenate(quote_date_blocks),
-                target_dates=np.concatenate(target_date_blocks),
-                predictions=np.vstack(prediction_blocks),
-                grid=grid,
-            )
-            item_metadata = {
-                "model_name": model_name,
-                "n_splits": len(clean_splits),
-                "n_forecast_rows": int(np.concatenate(quote_date_blocks).shape[0]),
-                "workflow_run_label": workflow_paths.run_label,
+                model_name=model_name,
+                quote_dates=np.concatenate(quote_date_blocks),
+                target_dates=np.concatenate(target_date_blocks),
+                split_ids=np.concatenate(split_id_blocks),
+                decision_timestamps=np.concatenate(decision_timestamp_blocks),
+                target_decision_timestamps=np.concatenate(target_decision_timestamp_blocks),
+                predictions=np.vstack(prediction_blocks),
+                grid=grid,
+                surface_config_hash=surface_config_hash,
+                model_config_hash=model_config_hash,
+                training_run_id=training_run_id,
+            )
+            forecast_artifact_hash = sha256_file(output_path)
+            item_metadata = {
+                "model_name": model_name,
+                "n_splits": len(clean_splits),
+                "n_forecast_rows": int(np.concatenate(quote_date_blocks).shape[0]),
+                "workflow_run_label": workflow_paths.run_label,
                 "max_hpo_validation_date": (
                     clean_evaluation_policy.max_hpo_validation_date.isoformat()
-                ),
-                "first_clean_test_split_id": clean_evaluation_policy.first_clean_test_split_id,
-                **model_metadata,
-            }
-            resumer.mark_complete(
-                model_name,
-                output_paths=[output_path],
-                metadata=item_metadata,
-            )
+                ),
+                "first_clean_test_split_id": clean_evaluation_policy.first_clean_test_split_id,
+                "forecast_artifact_hash": forecast_artifact_hash,
+                **model_metadata,
+            }
+            item_metadata = _validated_stage06_resume_metadata(
+                item_metadata,
+                model_name=model_name,
+                output_path=output_path,
+                n_splits=len(clean_splits),
+                workflow_run_label=workflow_paths.run_label,
+                max_hpo_validation_date=(
+                    clean_evaluation_policy.max_hpo_validation_date.isoformat()
+                ),
+                first_clean_test_split_id=clean_evaluation_policy.first_clean_test_split_id,
+                model_config_hash=model_config_hash,
+                training_run_id=training_run_id,
+                surface_config_hash=surface_config_hash,
+            )
+            resumer.mark_complete(
+                model_name,
+                output_paths=[output_path],
+                metadata=item_metadata,
+            )
             model_run_metadata[model_name] = item_metadata
     forecast_paths = sorted_artifact_files(workflow_paths.forecast_dir, "*.parquet")
     run_manifest_path = write_run_manifest(
diff --git a/tests/integration/test_stage05_stage06_clean_evaluation.py b/tests/integration/test_stage05_stage06_clean_evaluation.py
index c4c4940..4ad45c6 100644
--- a/tests/integration/test_stage05_stage06_clean_evaluation.py
+++ b/tests/integration/test_stage05_stage06_clean_evaluation.py
@@ -13,8 +13,16 @@ from pytest import MonkeyPatch

 from ivsurf.calendar import MarketCalendar
 from ivsurf.config import WalkforwardConfig
-from ivsurf.splits.manifests import serialize_splits
+from ivsurf.reproducibility import sha256_file
+from ivsurf.splits.manifests import WalkforwardSplit, serialize_splits
 from ivsurf.splits.walkforward import build_walkforward_splits, clean_evaluation_splits
+from ivsurf.surfaces.grid import (
+    MATURITY_COORDINATE,
+    MONEYNESS_COORDINATE,
+    SURFACE_GRID_SCHEMA_VERSION,
+    SurfaceGrid,
+)
+from ivsurf.surfaces.interpolation import COMPLETED_SURFACE_SCHEMA_VERSION
 from ivsurf.training.tuning import load_tuning_result


@@ -40,6 +48,7 @@ def _calendar_dates(count: int) -> list[date]:

 def _feature_frame(row_count: int) -> pl.DataFrame:
     dates = _calendar_dates(row_count + 1)
+    grid = SurfaceGrid(maturity_days=(30,), moneyness_points=(0.0, 0.1))
     rows = []
     for index in range(row_count):
         level = 0.0100 + (0.0001 * index)
@@ -47,6 +56,14 @@ def _feature_frame(row_count: int) -> pl.DataFrame:
             {
                 "quote_date": dates[index],
                 "target_date": dates[index + 1],
+                "effective_decision_timestamp": f"{dates[index].isoformat()}T15:45:00-05:00",
+                "target_effective_decision_timestamp": (
+                    f"{dates[index + 1].isoformat()}T15:45:00-05:00"
+                ),
+                "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
+                "surface_grid_hash": grid.grid_hash,
+                "maturity_coordinate": MATURITY_COORDINATE,
+                "moneyness_coordinate": MONEYNESS_COORDINATE,
                 "feature_surface_mean_01_0000": level,
                 "feature_surface_mean_01_0001": level + 0.0010,
                 "target_total_variance_0000": level + 0.0002,
@@ -62,6 +79,36 @@ def _feature_frame(row_count: int) -> pl.DataFrame:
     return pl.DataFrame(rows)


+def _with_surface_metadata(feature_frame: pl.DataFrame, surface_config_path: Path) -> pl.DataFrame:
+    return feature_frame.with_columns(
+        pl.lit(sha256_file(surface_config_path)).alias("surface_config_hash"),
+        pl.lit(COMPLETED_SURFACE_SCHEMA_VERSION).alias("target_surface_version"),
+    )
+
+
+def _serialize_test_splits(
+    *,
+    splits: list[WalkforwardSplit],
+    split_manifest_path: Path,
+    walkforward_config: WalkforwardConfig,
+    feature_frame: pl.DataFrame,
+    feature_path: Path,
+    sample_start_date: str,
+    sample_end_date: str,
+) -> str:
+    return serialize_splits(
+        splits,
+        split_manifest_path,
+        walkforward_config=walkforward_config.model_dump(mode="json"),
+        sample_window={
+            "sample_start_date": sample_start_date,
+            "sample_end_date": sample_end_date,
+        },
+        date_universe=feature_frame["quote_date"].to_list(),
+        feature_dataset_hash=sha256_file(feature_path),
+    )
+
+
 def _observed_quote_dates(count: int, *, skipped_session_indices: set[int]) -> list[date]:
     calendar = MarketCalendar()
     observed_dates: list[date] = []
@@ -76,6 +123,7 @@ def _observed_quote_dates(count: int, *, skipped_session_indices: set[int]) -> l


 def _feature_frame_from_observed_dates(observed_dates: list[date]) -> pl.DataFrame:
+    grid = SurfaceGrid(maturity_days=(30,), moneyness_points=(0.0, 0.1))
     rows = []
     for index, (quote_date, target_date) in enumerate(pairwise(observed_dates)):
         level = 0.0100 + (0.0001 * index)
@@ -83,6 +131,14 @@ def _feature_frame_from_observed_dates(observed_dates: list[date]) -> pl.DataFra
             {
                 "quote_date": quote_date,
                 "target_date": target_date,
+                "effective_decision_timestamp": f"{quote_date.isoformat()}T15:45:00-05:00",
+                "target_effective_decision_timestamp": (
+                    f"{target_date.isoformat()}T15:45:00-05:00"
+                ),
+                "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
+                "surface_grid_hash": grid.grid_hash,
+                "maturity_coordinate": MATURITY_COORDINATE,
+                "moneyness_coordinate": MONEYNESS_COORDINATE,
                 "target_gap_sessions": 0,
                 "feature_surface_mean_01_0000": level,
                 "feature_surface_mean_01_0001": level + 0.0010,
@@ -109,7 +165,6 @@ def test_stage05_and_stage06_emit_only_clean_evaluation_forecasts(
     feature_frame = _feature_frame(row_count=714)
     gold_dir.mkdir(parents=True)
     manifests_dir.mkdir(parents=True)
-    feature_frame.write_parquet(gold_dir / "daily_features.parquet")

     walkforward_config = WalkforwardConfig(
         train_size=504,
@@ -123,7 +178,6 @@ def test_stage05_and_stage06_emit_only_clean_evaluation_forecasts(
         config=walkforward_config,
     )
     split_manifest_path = manifests_dir / "walkforward_splits.json"
-    serialize_splits(splits, split_manifest_path)
     boundary, clean_splits = clean_evaluation_splits(splits, tuning_splits_count=3)
     expected_clean_quote_dates = {
         date.fromisoformat(day)
@@ -154,6 +208,18 @@ def test_stage05_and_stage06_emit_only_clean_evaluation_forecasts(
             "observed_cell_min_count: 1\n"
         ),
     )
+    feature_frame = _with_surface_metadata(feature_frame, surface_config_path)
+    feature_path = gold_dir / "daily_features.parquet"
+    feature_frame.write_parquet(feature_path)
+    _serialize_test_splits(
+        splits=splits,
+        split_manifest_path=split_manifest_path,
+        walkforward_config=walkforward_config,
+        feature_frame=feature_frame,
+        feature_path=feature_path,
+        sample_start_date="2021-01-01",
+        sample_end_date="2023-12-31",
+    )
     metrics_config_path = _write_text(
         tmp_path / "configs" / "eval" / "metrics.yaml",
         (
@@ -239,6 +305,82 @@ def test_stage05_and_stage06_emit_only_clean_evaluation_forecasts(
             min(forecast_frame["quote_date"].to_list()) > boundary.max_hpo_validation_date
         )

+    stage06.main(
+        raw_config_path=raw_config_path,
+        surface_config_path=surface_config_path,
+        metrics_config_path=metrics_config_path,
+        ridge_config_path=repo_root / "configs" / "models" / "ridge.yaml",
+        elasticnet_config_path=repo_root / "configs" / "models" / "elasticnet.yaml",
+        har_config_path=repo_root / "configs" / "models" / "har_factor.yaml",
+        lightgbm_config_path=repo_root / "configs" / "models" / "lightgbm.yaml",
+        random_forest_config_path=repo_root / "configs" / "models" / "random_forest.yaml",
+        neural_config_path=repo_root / "configs" / "models" / "neural_surface.yaml",
+        hpo_profile_config_path=hpo_profile_config_path,
+        training_profile_config_path=training_profile_config_path,
+    )
+    latest_stage06_manifest = sorted(
+        (manifests_dir / "runs" / "06_run_walkforward").glob("*_06_run_walkforward.json")
+    )[-1]
+    latest_payload = orjson.loads(latest_stage06_manifest.read_bytes())
+    model_run_metadata = latest_payload["extra_metadata"]["model_run_metadata"]
+    assert set(model_run_metadata) == {"naive", "ridge"}
+    for model_name in ("naive", "ridge"):
+        model_metadata = model_run_metadata[model_name]
+        assert model_metadata["forecast_artifact_hash"] == sha256_file(
+            forecast_dir / f"{model_name}.parquet"
+        )
+        assert model_metadata["n_forecast_rows"] == len(expected_clean_quote_dates)
+
+    run_manifest_dir = manifests_dir / "runs" / "06_run_walkforward"
+    run_manifest_count = len(list(run_manifest_dir.glob("*_06_run_walkforward.json")))
+    resume_state_path = manifests_dir / "resume" / "06_run_walkforward" / "state.json"
+    original_resume_payload = orjson.loads(resume_state_path.read_bytes())
+    empty_metadata_payload = orjson.loads(orjson.dumps(original_resume_payload))
+    empty_metadata_payload["items"]["ridge"]["metadata"] = {}
+    resume_state_path.write_bytes(
+        orjson.dumps(empty_metadata_payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
+    )
+
+    with pytest.raises(ValueError, match="model_name"):
+        stage06.main(
+            raw_config_path=raw_config_path,
+            surface_config_path=surface_config_path,
+            metrics_config_path=metrics_config_path,
+            ridge_config_path=repo_root / "configs" / "models" / "ridge.yaml",
+            elasticnet_config_path=repo_root / "configs" / "models" / "elasticnet.yaml",
+            har_config_path=repo_root / "configs" / "models" / "har_factor.yaml",
+            lightgbm_config_path=repo_root / "configs" / "models" / "lightgbm.yaml",
+            random_forest_config_path=repo_root / "configs" / "models" / "random_forest.yaml",
+            neural_config_path=repo_root / "configs" / "models" / "neural_surface.yaml",
+            hpo_profile_config_path=hpo_profile_config_path,
+            training_profile_config_path=training_profile_config_path,
+        )
+    assert len(list(run_manifest_dir.glob("*_06_run_walkforward.json"))) == run_manifest_count
+
+    resume_state_path.write_bytes(
+        orjson.dumps(original_resume_payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
+    )
+    tampered_ridge_forecast = ridge_forecast.with_columns(
+        (pl.col("predicted_total_variance") + 1.0e-6).alias("predicted_total_variance")
+    )
+    tampered_ridge_forecast.write_parquet(forecast_dir / "ridge.parquet")
+
+    with pytest.raises(ValueError, match="forecast_artifact_hash"):
+        stage06.main(
+            raw_config_path=raw_config_path,
+            surface_config_path=surface_config_path,
+            metrics_config_path=metrics_config_path,
+            ridge_config_path=repo_root / "configs" / "models" / "ridge.yaml",
+            elasticnet_config_path=repo_root / "configs" / "models" / "elasticnet.yaml",
+            har_config_path=repo_root / "configs" / "models" / "har_factor.yaml",
+            lightgbm_config_path=repo_root / "configs" / "models" / "lightgbm.yaml",
+            random_forest_config_path=repo_root / "configs" / "models" / "random_forest.yaml",
+            neural_config_path=repo_root / "configs" / "models" / "neural_surface.yaml",
+            hpo_profile_config_path=hpo_profile_config_path,
+            training_profile_config_path=training_profile_config_path,
+        )
+    assert len(list(run_manifest_dir.glob("*_06_run_walkforward.json"))) == run_manifest_count
+

 def test_stage05_and_stage06_preserve_target_dates_when_feature_rows_have_gaps(
     tmp_path: Path,
@@ -251,7 +393,6 @@ def test_stage05_and_stage06_preserve_target_dates_when_feature_rows_have_gaps(
     feature_frame = _feature_frame_from_observed_dates(observed_dates)
     gold_dir.mkdir(parents=True)
     manifests_dir.mkdir(parents=True)
-    feature_frame.write_parquet(gold_dir / "daily_features.parquet")

     walkforward_config = WalkforwardConfig(
         train_size=6,
@@ -265,7 +406,6 @@ def test_stage05_and_stage06_preserve_target_dates_when_feature_rows_have_gaps(
         config=walkforward_config,
     )
     split_manifest_path = manifests_dir / "walkforward_splits.json"
-    serialize_splits(splits, split_manifest_path)
     boundary, clean_splits = clean_evaluation_splits(splits, tuning_splits_count=2)
     expected_clean_quote_dates = {
         date.fromisoformat(day)
@@ -301,6 +441,18 @@ def test_stage05_and_stage06_preserve_target_dates_when_feature_rows_have_gaps(
             "observed_cell_min_count: 1\n"
         ),
     )
+    feature_frame = _with_surface_metadata(feature_frame, surface_config_path)
+    feature_path = gold_dir / "daily_features.parquet"
+    feature_frame.write_parquet(feature_path)
+    _serialize_test_splits(
+        splits=splits,
+        split_manifest_path=split_manifest_path,
+        walkforward_config=walkforward_config,
+        feature_frame=feature_frame,
+        feature_path=feature_path,
+        sample_start_date="2021-01-04",
+        sample_end_date="2021-03-31",
+    )
     metrics_config_path = _write_text(
         tmp_path / "configs" / "eval" / "metrics.yaml",
         (
@@ -401,7 +553,6 @@ def test_stage06_rejects_stale_tuning_manifests_with_different_primary_loss_metr
     feature_frame = _feature_frame(row_count=714)
     gold_dir.mkdir(parents=True)
     manifests_dir.mkdir(parents=True)
-    feature_frame.write_parquet(gold_dir / "daily_features.parquet")

     walkforward_config = WalkforwardConfig(
         train_size=504,
@@ -414,7 +565,6 @@ def test_stage06_rejects_stale_tuning_manifests_with_different_primary_loss_metr
         dates=feature_frame["quote_date"].to_list(),
         config=walkforward_config,
     )
-    serialize_splits(splits, manifests_dir / "walkforward_splits.json")

     raw_config_path = _write_text(
         tmp_path / "configs" / "data" / "raw.yaml",
@@ -439,6 +589,18 @@ def test_stage06_rejects_stale_tuning_manifests_with_different_primary_loss_metr
             "observed_cell_min_count: 1\n"
         ),
     )
+    feature_frame = _with_surface_metadata(feature_frame, surface_config_path)
+    feature_path = gold_dir / "daily_features.parquet"
+    feature_frame.write_parquet(feature_path)
+    _serialize_test_splits(
+        splits=splits,
+        split_manifest_path=manifests_dir / "walkforward_splits.json",
+        walkforward_config=walkforward_config,
+        feature_frame=feature_frame,
+        feature_path=feature_path,
+        sample_start_date="2021-01-01",
+        sample_end_date="2023-12-31",
+    )
     metrics_config_path = _write_text(
         tmp_path / "configs" / "eval" / "metrics.yaml",
         (

```

Respond exactly with this structure:

```text
ROUND_004B_VERIFICATION_DECISION: all_fixed | still_has_issues
Finding status:
- B2-CODE-007: fixed | not fixed | partially fixed
B2 backlog implication: mark B2-CODE-001 through B2-CODE-007 pro-verified-fixed? yes | no
If any issue remains, give exact file/function/test references and the minimum required fix.
If fixed, state any residual risks that belong to a later full-project audit, but do not invent new blockers unless you see a concrete defect in the patch.
```
