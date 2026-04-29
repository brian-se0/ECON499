# ECON499 SPX IV Surface Audit - Round 004A Verification Request

You are GPT 5.5 Pro acting as the independent architecture/research-integrity auditor. Think deeply and do not stop at a superficial summary.

Context:
- You previously completed Round 004 and returned `ROUND_004_AUDIT_DECISION: findings_found`.
- Codex implemented fixes for all seven Round 004 findings.
- The patch below is relative to the exact Round 004 code ZIP that you reviewed in this conversation.
- Project rules remain strict: fail fast, no fallback paths, no backwards compatibility/legacy branches, no leakage, no silent coercion, no silent data drops, explicit reproducibility.

Verification scope:
1. B2-CODE-001 P1: option cleaning must reject null/non-finite/non-positive critical numeric values with explicit reason codes, and Stage03 must refuse invalid valid-row surface inputs.
2. B2-CODE-002 P1: split manifests must be a versioned envelope tied to the date universe and daily feature artifact hash; legacy bare-list manifests must be rejected; Stage05/06 must validate feature artifact matching before use.
3. B2-CODE-003 P2: HAR must validate the exact lag-1/lag-5/lag-22 feature layout before use.
4. B2-CODE-004 P2: model construction must not silently coerce persisted hyperparameter types, especially strings to floats and floats to ints.
5. B2-CODE-005 P2: evaluation/statistical comparison must prove duplicate-free, equal forecast coverage across models and finite loss/stat inputs before computing tests.
6. B2-CODE-006 P2: run manifests must fail when a git commit hash cannot be obtained.
7. B2-CODE-007 P3: Stage06 resumed/skipped models must still appear in final model-run metadata.

Local verification already run by Codex:

```text
uv run python -m ruff check src/ivsurf/cleaning/option_filters.py src/ivsurf/splits/manifests.py src/ivsurf/models/har_factor.py src/ivsurf/training/model_factory.py src/ivsurf/models/lightgbm_model.py src/ivsurf/evaluation/alignment.py src/ivsurf/evaluation/loss_panels.py src/ivsurf/stats/diebold_mariano.py src/ivsurf/stats/spa.py src/ivsurf/stats/mcs.py src/ivsurf/reproducibility.py scripts/02_build_option_panel.py scripts/03_build_surfaces.py scripts/04_build_features.py scripts/05_tune_models.py scripts/06_run_walkforward.py scripts/07_run_stats.py tests/unit/test_option_filters.py tests/unit/test_split_manifests.py tests/unit/test_training_behaviour.py tests/unit/test_stats.py tests/unit/test_alignment.py tests/unit/test_reporting_helpers.py tests/integration/test_early_close_stage02.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage05_stage06_clean_evaluation.py
# All checks passed

uv run python -m pytest tests/unit/test_option_filters.py tests/unit/test_split_manifests.py tests/unit/test_training_behaviour.py tests/unit/test_stats.py tests/unit/test_alignment.py tests/unit/test_reporting_helpers.py tests/integration/test_early_close_stage02.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage05_stage06_clean_evaluation.py
# 74 passed in 4.98s
```

Required output format:

```text
ROUND_004A_VERIFICATION_DECISION: all_fixed | still_has_issues

Finding-by-finding status:
- B2-CODE-001: fixed | not fixed | partially fixed
...
- B2-CODE-007: fixed | not fixed | partially fixed

If any issue remains, give exact file/function/test references and the minimum required fix. If all are fixed, say what residual risks still need a later full-project audit but do not invent new blockers unless you see a concrete defect in the patch.
```

Patch relative to Round 004 reviewed code:

```diff
--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/cleaning/option_filters.py	2026-04-27 18:25:46
+++ /Volumes/T9/ECON499/src/ivsurf/cleaning/option_filters.py	2026-04-27 19:49:51
@@ -23,12 +23,34 @@
         .then(pl.lit("MISSING_VEGA_1545"))
         .when(pl.col("active_underlying_price_1545").is_null())
         .then(pl.lit("MISSING_ACTIVE_UNDERLYING_PRICE_1545"))
+        .when(pl.col("strike").is_null())
+        .then(pl.lit("MISSING_STRIKE"))
         .when(pl.col("mid_1545").is_null())
         .then(pl.lit("MISSING_MID_1545"))
         .when(pl.col("tau_years").is_null())
         .then(pl.lit("MISSING_TAU_YEARS"))
-        .when(pl.col("log_moneyness").is_null())
-        .then(pl.lit("MISSING_LOG_MONEYNESS"))
+        .when(pl.col("total_variance").is_null())
+        .then(pl.lit("MISSING_TOTAL_VARIANCE"))
+        .when(~pl.col("bid_1545").is_finite())
+        .then(pl.lit("NONFINITE_BID_1545"))
+        .when(~pl.col("ask_1545").is_finite())
+        .then(pl.lit("NONFINITE_ASK_1545"))
+        .when(~pl.col("implied_volatility_1545").is_finite())
+        .then(pl.lit("NONFINITE_IV_1545"))
+        .when(~pl.col("vega_1545").is_finite())
+        .then(pl.lit("NONFINITE_VEGA_1545"))
+        .when(~pl.col("active_underlying_price_1545").is_finite())
+        .then(pl.lit("NONFINITE_UNDERLYING_1545"))
+        .when(~pl.col("strike").is_finite())
+        .then(pl.lit("NONFINITE_STRIKE"))
+        .when(pl.col("strike") <= 0.0)
+        .then(pl.lit("NON_POSITIVE_STRIKE"))
+        .when(~pl.col("mid_1545").is_finite())
+        .then(pl.lit("NONFINITE_MID_1545"))
+        .when(~pl.col("tau_years").is_finite())
+        .then(pl.lit("NONFINITE_TAU_YEARS"))
+        .when(~pl.col("total_variance").is_finite())
+        .then(pl.lit("NONFINITE_TOTAL_VARIANCE"))
         .when(~pl.col("option_type").is_in(config.allowed_option_types))
         .then(pl.lit("INVALID_OPTION_TYPE"))
         .when(pl.col("bid_1545") <= config.min_valid_bid_exclusive)
@@ -52,6 +74,12 @@
         .then(pl.lit("TAU_TOO_SHORT"))
         .when(pl.col("tau_years") > config.max_tau_years)
         .then(pl.lit("TAU_TOO_LONG"))
+        .when(pl.col("total_variance") <= 0.0)
+        .then(pl.lit("NON_POSITIVE_TOTAL_VARIANCE"))
+        .when(pl.col("log_moneyness").is_null())
+        .then(pl.lit("MISSING_LOG_MONEYNESS"))
+        .when(~pl.col("log_moneyness").is_finite())
+        .then(pl.lit("NONFINITE_LOG_MONEYNESS"))
         .when(pl.col("log_moneyness").abs() > config.max_abs_log_moneyness)
         .then(pl.lit("OUTSIDE_MONEYNESS_RANGE"))
         .otherwise(pl.lit(None, dtype=pl.String))
--- /tmp/econ499_round004_extract.ZdatuB/scripts/02_build_option_panel.py	2026-04-27 18:27:18
+++ /Volumes/T9/ECON499/scripts/02_build_option_panel.py	2026-04-27 19:50:28
@@ -67,7 +67,7 @@
         context_hash=build_resume_context_hash(
             config_paths=[raw_config_path, cleaning_config_path],
             input_artifact_paths=bronze_files,
-            extra_tokens={"artifact_schema_version": 2},
+            extra_tokens={"artifact_schema_version": 3},
         ),
     )

--- /tmp/econ499_round004_extract.ZdatuB/scripts/03_build_surfaces.py	2026-04-27 18:45:16
+++ /Volumes/T9/ECON499/scripts/03_build_surfaces.py	2026-04-27 19:51:19
@@ -48,6 +48,43 @@
         message = f"{context} requires exactly one non-empty {column_name}, found {values!r}."
         raise ValueError(message)
     return values[0]
+
+
+def _require_valid_surface_inputs(frame: pl.DataFrame, *, silver_path: Path) -> None:
+    finite_columns = (
+        "tau_years",
+        "log_moneyness",
+        "total_variance",
+        "implied_volatility_1545",
+        "vega_1545",
+        "spread_1545",
+    )
+    nonfinite = frame.filter(
+        pl.any_horizontal(
+            *[(pl.col(column).is_null() | ~pl.col(column).is_finite()) for column in finite_columns]
+        )
+    )
+    if not nonfinite.is_empty():
+        message = (
+            f"Stage 03 valid rows in {silver_path} contain null or non-finite surface inputs "
+            f"for columns {finite_columns}."
+        )
+        raise ValueError(message)
+    positive_columns = (
+        "tau_years",
+        "total_variance",
+        "implied_volatility_1545",
+        "vega_1545",
+    )
+    nonpositive = frame.filter(
+        pl.any_horizontal(*[pl.col(column) <= 0.0 for column in positive_columns])
+    )
+    if not nonpositive.is_empty():
+        message = (
+            f"Stage 03 valid rows in {silver_path} contain non-positive surface inputs "
+            f"for columns {positive_columns}."
+        )
+        raise ValueError(message)


 def _gold_path(silver_path: Path, raw_config: RawDataConfig) -> Path:
@@ -80,7 +117,7 @@
         context_hash=build_resume_context_hash(
             config_paths=[raw_config_path, surface_config_path],
             input_artifact_paths=silver_files,
-            extra_tokens={"artifact_schema_version": 4},
+            extra_tokens={"artifact_schema_version": 5},
         ),
     )

@@ -124,6 +161,7 @@
                 context=f"Stage 03 silver artifact {silver_path}",
             )
             valid_frame = valid_option_rows(silver_frame)
+            _require_valid_surface_inputs(valid_frame, silver_path=silver_path)
             if valid_frame.is_empty():
                 decision_timestamp = _single_string_value(
                     silver_frame,
--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/splits/manifests.py	2026-04-08 19:32:36
+++ /Volumes/T9/ECON499/src/ivsurf/splits/manifests.py	2026-04-27 19:55:47
@@ -2,6 +2,7 @@

 from __future__ import annotations

+from collections.abc import Mapping, Sequence
 from dataclasses import asdict, dataclass
 from hashlib import sha256
 from pathlib import Path
@@ -10,7 +11,10 @@

 from ivsurf.io.atomic import write_bytes_atomic

+SPLIT_MANIFEST_SCHEMA_VERSION = "walkforward_split_manifest_v1"
+SPLIT_MANIFEST_GENERATOR = "ivsurf.splits.walkforward.build_walkforward_splits"

+
 @dataclass(frozen=True, slots=True)
 class WalkforwardSplit:
     """Explicit walk-forward split."""
@@ -21,18 +25,220 @@
     test_dates: tuple[str, ...]


-def serialize_splits(splits: list[WalkforwardSplit], output_path: Path) -> str:
+@dataclass(frozen=True, slots=True)
+class WalkforwardSplitManifest:
+    """Versioned walk-forward split manifest."""
+
+    schema_version: str
+    generator: str
+    walkforward_config: dict[str, object]
+    sample_window: dict[str, str]
+    date_universe_hash: str
+    feature_dataset_hash: str
+    splits: list[WalkforwardSplit]
+
+
+def date_universe_hash(dates: Sequence[object]) -> str:
+    """Hash the ordered quote-date universe used to generate walk-forward splits."""
+
+    normalized_dates = [str(value) for value in dates]
+    if not normalized_dates:
+        message = "Split manifest date universe cannot be empty."
+        raise ValueError(message)
+    raw = orjson.dumps(normalized_dates)
+    return sha256(raw).hexdigest()
+
+
+def _require_non_empty_string(value: object, *, field_name: str) -> str:
+    if not isinstance(value, str) or not value:
+        message = f"Split manifest field {field_name} must be a non-empty string."
+        raise ValueError(message)
+    return value
+
+
+def _normalize_metadata_mapping(
+    values: Mapping[str, object],
+    *,
+    field_name: str,
+) -> dict[str, object]:
+    if not values:
+        message = f"Split manifest field {field_name} cannot be empty."
+        raise ValueError(message)
+    return dict(values)
+
+
+def _manifest_payload(
+    *,
+    splits: list[WalkforwardSplit],
+    walkforward_config: Mapping[str, object],
+    sample_window: Mapping[str, str],
+    date_universe: Sequence[object],
+    feature_dataset_hash: str,
+) -> dict[str, object]:
+    _require_non_empty_string(feature_dataset_hash, field_name="feature_dataset_hash")
+    if not splits:
+        message = "Split manifest must contain at least one split."
+        raise ValueError(message)
+    return {
+        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
+        "generator": SPLIT_MANIFEST_GENERATOR,
+        "walkforward_config": _normalize_metadata_mapping(
+            walkforward_config,
+            field_name="walkforward_config",
+        ),
+        "sample_window": _normalize_metadata_mapping(sample_window, field_name="sample_window"),
+        "date_universe_hash": date_universe_hash(date_universe),
+        "feature_dataset_hash": feature_dataset_hash,
+        "splits": [asdict(split) for split in splits],
+    }
+
+
+def serialize_splits(
+    splits: list[WalkforwardSplit],
+    output_path: Path,
+    *,
+    walkforward_config: Mapping[str, object],
+    sample_window: Mapping[str, str],
+    date_universe: Sequence[object],
+    feature_dataset_hash: str,
+) -> str:
     """Write split manifest and return its SHA256 hash."""

-    payload = [asdict(split) for split in splits]
+    payload = _manifest_payload(
+        splits=splits,
+        walkforward_config=walkforward_config,
+        sample_window=sample_window,
+        date_universe=date_universe,
+        feature_dataset_hash=feature_dataset_hash,
+    )
     raw = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
     output_path.parent.mkdir(parents=True, exist_ok=True)
     write_bytes_atomic(output_path, raw)
     return sha256(raw).hexdigest()


-def load_splits(path: Path) -> list[WalkforwardSplit]:
-    """Load a split manifest from disk."""
+def load_split_manifest(path: Path) -> WalkforwardSplitManifest:
+    """Load and validate a versioned split manifest from disk."""

     payload = orjson.loads(path.read_bytes())
-    return [WalkforwardSplit(**item) for item in payload]
+    if isinstance(payload, list):
+        message = (
+            "Legacy bare-list split manifests are not supported. "
+            "Regenerate splits with Stage 04."
+        )
+        raise ValueError(message)
+    if not isinstance(payload, dict):
+        message = "Split manifest must be a versioned JSON object."
+        raise ValueError(message)
+    schema_version = _require_non_empty_string(
+        payload.get("schema_version"),
+        field_name="schema_version",
+    )
+    if schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
+        message = (
+            "Unsupported split manifest schema_version: "
+            f"{schema_version!r}; expected {SPLIT_MANIFEST_SCHEMA_VERSION!r}."
+        )
+        raise ValueError(message)
+    generator = _require_non_empty_string(payload.get("generator"), field_name="generator")
+    if generator != SPLIT_MANIFEST_GENERATOR:
+        message = (
+            "Unsupported split manifest generator: "
+            f"{generator!r}; expected {SPLIT_MANIFEST_GENERATOR!r}."
+        )
+        raise ValueError(message)
+    splits_payload = payload.get("splits")
+    if not isinstance(splits_payload, list) or not splits_payload:
+        message = "Split manifest must contain a non-empty splits list."
+        raise ValueError(message)
+    splits = [_split_from_payload(item, index=index) for index, item in enumerate(splits_payload)]
+    return WalkforwardSplitManifest(
+        schema_version=schema_version,
+        generator=generator,
+        walkforward_config=_dict_payload(payload.get("walkforward_config"), "walkforward_config"),
+        sample_window=_string_dict_payload(payload.get("sample_window"), "sample_window"),
+        date_universe_hash=_require_non_empty_string(
+            payload.get("date_universe_hash"),
+            field_name="date_universe_hash",
+        ),
+        feature_dataset_hash=_require_non_empty_string(
+            payload.get("feature_dataset_hash"),
+            field_name="feature_dataset_hash",
+        ),
+        splits=splits,
+    )
+
+
+def _string_tuple_payload(value: object, field_name: str) -> tuple[str, ...]:
+    if not isinstance(value, list | tuple) or not value:
+        message = f"Split manifest field {field_name} must be a non-empty string array."
+        raise ValueError(message)
+    normalized: list[str] = []
+    for item in value:
+        if not isinstance(item, str) or not item:
+            message = f"Split manifest field {field_name} must contain non-empty strings."
+            raise ValueError(message)
+        normalized.append(item)
+    return tuple(normalized)
+
+
+def _split_from_payload(value: object, *, index: int) -> WalkforwardSplit:
+    if not isinstance(value, dict):
+        message = f"Split manifest split at index {index} must be an object."
+        raise ValueError(message)
+    return WalkforwardSplit(
+        split_id=_require_non_empty_string(value.get("split_id"), field_name="split_id"),
+        train_dates=_string_tuple_payload(value.get("train_dates"), "train_dates"),
+        validation_dates=_string_tuple_payload(
+            value.get("validation_dates"),
+            "validation_dates",
+        ),
+        test_dates=_string_tuple_payload(value.get("test_dates"), "test_dates"),
+    )
+
+
+def _dict_payload(value: object, field_name: str) -> dict[str, object]:
+    if not isinstance(value, dict) or not value:
+        message = f"Split manifest field {field_name} must be a non-empty object."
+        raise ValueError(message)
+    return dict(value)
+
+
+def _string_dict_payload(value: object, field_name: str) -> dict[str, str]:
+    payload = _dict_payload(value, field_name)
+    normalized: dict[str, str] = {}
+    for key, item in payload.items():
+        if not isinstance(key, str) or not isinstance(item, str) or not item:
+            message = f"Split manifest field {field_name} must contain string keys and values."
+            raise ValueError(message)
+        normalized[key] = item
+    return normalized
+
+
+def require_split_manifest_matches_artifacts(
+    manifest: WalkforwardSplitManifest,
+    *,
+    date_universe: Sequence[object],
+    feature_dataset_hash: str,
+) -> None:
+    """Fail fast if a split manifest was built for a different feature artifact."""
+
+    actual_date_hash = date_universe_hash(date_universe)
+    if manifest.date_universe_hash != actual_date_hash:
+        message = (
+            "Split manifest date_universe_hash does not match daily_features quote dates: "
+            f"{manifest.date_universe_hash!r} != {actual_date_hash!r}."
+        )
+        raise ValueError(message)
+    if manifest.feature_dataset_hash != feature_dataset_hash:
+        message = (
+            "Split manifest feature_dataset_hash does not match daily_features.parquet: "
+            f"{manifest.feature_dataset_hash!r} != {feature_dataset_hash!r}."
+        )
+        raise ValueError(message)
+
+
+def load_splits(path: Path) -> list[WalkforwardSplit]:
+    """Load a versioned split manifest from disk."""
+
+    return load_split_manifest(path).splits
--- /tmp/econ499_round004_extract.ZdatuB/scripts/04_build_features.py	2026-04-27 18:45:16
+++ /Volumes/T9/ECON499/scripts/04_build_features.py	2026-04-27 19:54:28
@@ -26,7 +26,7 @@
     sample_window_expr,
     sample_window_label,
 )
-from ivsurf.reproducibility import write_run_manifest
+from ivsurf.reproducibility import sha256_file, write_run_manifest
 from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
 from ivsurf.splits.manifests import serialize_splits
 from ivsurf.splits.walkforward import build_walkforward_splits
@@ -71,7 +71,7 @@
                 walkforward_config_path,
             ],
             input_artifact_paths=gold_files,
-            extra_tokens={"artifact_schema_version": 4},
+            extra_tokens={"artifact_schema_version": 5},
         ),
     )
     resume_item_id = "daily_feature_dataset"
@@ -150,6 +150,7 @@
         availability_manifest_path,
         orjson.dumps(availability_manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS),
     )
+    feature_dataset_hash = sha256_file(output_path)

     dates = daily_dataset["quote_date"].to_list()
     if any(not isinstance(value, date) for value in dates):
@@ -158,6 +159,13 @@
     split_hash = serialize_splits(
         splits=build_walkforward_splits(dates=dates, config=walkforward_config),
         output_path=split_manifest_path,
+        walkforward_config=walkforward_config.model_dump(mode="json"),
+        sample_window={
+            "sample_start_date": raw_config.sample_start_date.isoformat(),
+            "sample_end_date": raw_config.sample_end_date.isoformat(),
+        },
+        date_universe=dates,
+        feature_dataset_hash=feature_dataset_hash,
     )
     resumer.mark_complete(
         resume_item_id,
@@ -165,6 +173,7 @@
         metadata={
             "feature_rows": daily_dataset.height,
             "feature_availability_manifest_path": str(availability_manifest_path),
+            "feature_dataset_hash": feature_dataset_hash,
             "split_hash": split_hash,
         },
     )
@@ -189,6 +198,7 @@
         split_manifest_path=split_manifest_path,
         extra_metadata={
             "feature_rows": daily_dataset.height,
+            "feature_dataset_hash": feature_dataset_hash,
             "feature_availability_columns": len(availability_manifest),
             "split_hash": split_hash,
             "gold_input_file_count": len(gold_files),
--- /tmp/econ499_round004_extract.ZdatuB/scripts/05_tune_models.py	2026-04-26 02:24:08
+++ /Volumes/T9/ECON499/scripts/05_tune_models.py	2026-04-27 19:56:14
@@ -25,12 +25,17 @@
 from ivsurf.io.parquet import write_parquet_frame
 from ivsurf.models.base import DatasetMatrices, dataset_to_matrices
 from ivsurf.models.elasticnet import ElasticNetSurfaceModel
+from ivsurf.models.har_factor import validate_har_feature_layout
 from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
 from ivsurf.models.neural_surface import NeuralSurfaceRegressor
 from ivsurf.progress import create_progress
-from ivsurf.reproducibility import write_run_manifest
+from ivsurf.reproducibility import sha256_file, write_run_manifest
 from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
-from ivsurf.splits.manifests import WalkforwardSplit, load_splits
+from ivsurf.splits.manifests import (
+    WalkforwardSplit,
+    load_split_manifest,
+    require_split_manifest_matches_artifacts,
+)
 from ivsurf.splits.walkforward import clean_evaluation_boundary
 from ivsurf.surfaces.grid import SurfaceGrid
 from ivsurf.training.fit_lightgbm import fit_and_predict_lightgbm
@@ -355,7 +360,7 @@
                 raw_config.gold_dir / "daily_features.parquet",
                 split_manifest_path,
             ],
-            extra_tokens={"model_name": model_name},
+            extra_tokens={"artifact_schema_version": 2, "model_name": model_name},
         ),
     )
     resume_item_id = model_name
@@ -370,11 +375,23 @@
         return
     resumer.clear_item(resume_item_id, output_paths=[output_path, diagnostics_path])

-    feature_frame = pl.read_parquet(raw_config.gold_dir / "daily_features.parquet").sort(
-        "quote_date"
-    )
-    matrices = dataset_to_matrices(feature_frame)
-    splits = load_splits(split_manifest_path)
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
+    matrices = dataset_to_matrices(feature_frame)
+    if model_name == "har_factor":
+        validate_har_feature_layout(
+            feature_columns=matrices.feature_columns,
+            target_columns=matrices.target_columns,
+        )
+    splits = split_manifest.splits
     tuning_splits = splits[: hpo_profile.tuning_splits_count]
     if not tuning_splits:
         message = "No walk-forward splits available for tuning."
--- /tmp/econ499_round004_extract.ZdatuB/scripts/06_run_walkforward.py	2026-04-27 18:45:24
+++ /Volumes/T9/ECON499/scripts/06_run_walkforward.py	2026-04-27 20:01:14
@@ -21,11 +21,16 @@
 from ivsurf.exceptions import ModelConvergenceError
 from ivsurf.io.paths import sorted_artifact_files
 from ivsurf.models.base import dataset_to_matrices
+from ivsurf.models.har_factor import validate_har_feature_layout
 from ivsurf.models.naive import validate_naive_feature_layout
 from ivsurf.progress import create_progress
 from ivsurf.reproducibility import sha256_bytes, sha256_file, write_run_manifest
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
@@ -191,7 +196,7 @@
                 *tuning_manifest_paths,
             ],
             extra_tokens={
-                "artifact_schema_version": 3,
+                "artifact_schema_version": 4,
                 "run_profile_name": run_profile_name,
                 "workflow_run_label": workflow_paths.run_label,
             },
@@ -201,6 +206,13 @@
     feature_frame = pl.read_parquet(raw_config.gold_dir / "daily_features.parquet").sort(
         "quote_date"
     )
+    feature_dataset_hash = sha256_file(raw_config.gold_dir / "daily_features.parquet")
+    split_manifest = load_split_manifest(split_manifest_path)
+    require_split_manifest_matches_artifacts(
+        split_manifest,
+        date_universe=feature_frame["quote_date"].to_list(),
+        feature_dataset_hash=feature_dataset_hash,
+    )
     surface_config_hash = _single_feature_metadata_value(feature_frame, "surface_config_hash")
     if surface_config_hash != current_surface_config_hash:
         message = (
@@ -213,7 +225,7 @@
         feature_columns=matrices.feature_columns,
         target_columns=matrices.target_columns,
     )
-    splits = load_splits(split_manifest_path)
+    splits = split_manifest.splits

     ridge_params = load_yaml_config(ridge_config_path)
     elasticnet_params = load_yaml_config(elasticnet_config_path)
@@ -268,19 +280,25 @@
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
             output_path = workflow_paths.forecast_dir / f"{model_name}.parquet"
-            if resumer.item_complete(model_name, required_output_paths=[output_path]):
-                progress.update(
-                    task_id,
-                    description=f"Stage 06 resume: skipping completed model {model_name}",
-                )
-                progress.advance(task_id, advance=len(clean_splits))
-                continue
+            if resumer.item_complete(model_name, required_output_paths=[output_path]):
+                progress.update(
+                    task_id,
+                    description=f"Stage 06 resume: skipping completed model {model_name}",
+                )
+                model_run_metadata[model_name] = resumer.metadata_for(model_name)
+                progress.advance(task_id, advance=len(clean_splits))
+                continue
             resumer.clear_item(model_name, output_paths=[output_path])
             prediction_blocks: list[np.ndarray] = []
             quote_date_blocks: list[np.ndarray] = []
--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/models/har_factor.py	2026-04-09 17:27:08
+++ /Volumes/T9/ECON499/src/ivsurf/models/har_factor.py	2026-04-27 19:56:10
@@ -2,6 +2,8 @@

 from __future__ import annotations

+from collections.abc import Sequence
+
 import numpy as np
 from sklearn.linear_model import Ridge
 from sklearn.preprocessing import StandardScaler
@@ -10,7 +12,44 @@
 from ivsurf.models.base import SurfaceForecastModel
 from ivsurf.models.positive_target import LogPositiveTargetAdapter

+HAR_LAG_WINDOWS = (1, 5, 22)

+
+def validate_har_feature_layout(
+    feature_columns: Sequence[str],
+    target_columns: Sequence[str],
+    *,
+    target_prefix: str = "target_total_variance_",
+) -> None:
+    """Fail fast unless HAR lag blocks are present in the exact grid order."""
+
+    target_suffixes = tuple(
+        column.removeprefix(target_prefix)
+        for column in target_columns
+        if column.startswith(target_prefix)
+    )
+    if len(target_suffixes) != len(target_columns):
+        message = (
+            "HAR factor benchmark requires target_total_variance columns with the expected "
+            f"prefix {target_prefix!r}."
+        )
+        raise ValueError(message)
+    expected_feature_columns = tuple(
+        f"feature_surface_mean_{lag_window:02d}_{suffix}"
+        for lag_window in HAR_LAG_WINDOWS
+        for suffix in target_suffixes
+    )
+    leading_feature_columns = tuple(feature_columns[: len(expected_feature_columns)])
+    if leading_feature_columns != expected_feature_columns:
+        message = (
+            "HAR factor benchmark requires lag-1, lag-5, and lag-22 surface-mean feature "
+            "blocks to be present first and aligned one-to-one with target surface columns. "
+            f"Expected leading columns {expected_feature_columns!r}, "
+            f"found {leading_feature_columns!r}."
+        )
+        raise ValueError(message)
+
+
 class HarFactorSurfaceModel(SurfaceForecastModel):
     """Project surfaces to factors, run HAR-style regression, then reconstruct."""

--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/training/model_factory.py	2026-04-26 02:26:48
+++ /Volumes/T9/ECON499/src/ivsurf/training/model_factory.py	2026-04-27 19:57:22
@@ -1,10 +1,11 @@
 """Shared model construction helpers for tuning and walk-forward training."""

-from __future__ import annotations
+from __future__ import annotations
+
+from collections.abc import Mapping
+from math import isfinite
+from typing import Any

-from collections.abc import Mapping
-from typing import Any
-
 import optuna

 from ivsurf.config import NeuralModelConfig
@@ -26,22 +27,101 @@
 )


-def _float_param(params: Mapping[str, object], key: str) -> float:
-    value = params[key]
-    if not isinstance(value, int | float | str):
-        message = f"Expected {key} to be numeric-like, found {type(value).__name__}."
-        raise TypeError(message)
-    return float(value)
-
-
+def _float_param(params: Mapping[str, object], key: str) -> float:
+    value = _required_param(params, key)
+    if isinstance(value, bool) or not isinstance(value, int | float):
+        message = f"Expected {key} to be a real numeric value, found {type(value).__name__}."
+        raise TypeError(message)
+    normalized = float(value)
+    if not isfinite(normalized):
+        message = f"Expected {key} to be finite, found {value!r}."
+        raise ValueError(message)
+    return normalized
+
+
 def _int_param(params: Mapping[str, object], key: str) -> int:
-    value = params[key]
-    if not isinstance(value, int | float | str):
-        message = f"Expected {key} to be integer-like, found {type(value).__name__}."
+    value = _required_param(params, key)
+    if isinstance(value, bool) or not isinstance(value, int):
+        message = f"Expected {key} to be an integer, found {type(value).__name__}."
         raise TypeError(message)
-    return int(value)
+    return value


+def _str_param(params: Mapping[str, object], key: str) -> str:
+    value = _required_param(params, key)
+    if not isinstance(value, str) or not value:
+        message = f"Expected {key} to be a non-empty string, found {type(value).__name__}."
+        raise TypeError(message)
+    return value
+
+
+def _required_param(params: Mapping[str, object], key: str) -> object:
+    if key not in params:
+        message = f"Missing required model parameter {key!r}."
+        raise KeyError(message)
+    return params[key]
+
+
+def _reject_extra_params(
+    params: Mapping[str, object],
+    *,
+    allowed_keys: set[str],
+    model_name: str,
+) -> None:
+    extra_keys = sorted(set(params) - allowed_keys)
+    if extra_keys:
+        message = f"Unexpected {model_name} parameters: {extra_keys!r}."
+        raise ValueError(message)
+
+
+def _lightgbm_params(params: Mapping[str, object]) -> dict[str, object]:
+    allowed_keys = {
+        "n_factors",
+        "device_type",
+        "n_estimators",
+        "learning_rate",
+        "num_leaves",
+        "max_depth",
+        "min_child_samples",
+        "feature_fraction",
+        "lambda_l2",
+        "objective",
+        "metric",
+        "verbosity",
+        "n_jobs",
+        "random_state",
+    }
+    _reject_extra_params(params, allowed_keys=allowed_keys, model_name="LightGBM")
+    return {
+        "n_factors": _int_param(params, "n_factors"),
+        "device_type": _str_param(params, "device_type"),
+        "n_estimators": _int_param(params, "n_estimators"),
+        "learning_rate": _float_param(params, "learning_rate"),
+        "num_leaves": _int_param(params, "num_leaves"),
+        "max_depth": _int_param(params, "max_depth"),
+        "min_child_samples": _int_param(params, "min_child_samples"),
+        "feature_fraction": _float_param(params, "feature_fraction"),
+        "lambda_l2": _float_param(params, "lambda_l2"),
+        "objective": _str_param(params, "objective"),
+        "metric": _str_param(params, "metric"),
+        "verbosity": _int_param(params, "verbosity"),
+        "n_jobs": _int_param(params, "n_jobs"),
+        "random_state": _int_param(params, "random_state"),
+    }
+
+
+def _random_forest_params(params: Mapping[str, object]) -> dict[str, object]:
+    allowed_keys = {"n_estimators", "max_depth", "min_samples_leaf", "random_state", "n_jobs"}
+    _reject_extra_params(params, allowed_keys=allowed_keys, model_name="random_forest")
+    return {
+        "n_estimators": _int_param(params, "n_estimators"),
+        "max_depth": _int_param(params, "max_depth"),
+        "min_samples_leaf": _int_param(params, "min_samples_leaf"),
+        "random_state": _int_param(params, "random_state"),
+        "n_jobs": _int_param(params, "n_jobs"),
+    }
+
+
 def suggest_model_from_trial(
     *,
     model_name: str,
@@ -69,12 +149,13 @@
             alpha=trial.suggest_float("alpha", 1.0e-4, 100.0, log=True),
             target_dim=target_dim,
         )
-    if model_name == "lightgbm":
-        params = {
-            key: value for key, value in (base_lightgbm_params or {}).items() if key != "model_name"
-        }
-        params.update(
-            {
+    if model_name == "lightgbm":
+        if base_lightgbm_params is None:
+            message = "LightGBM tuning requires explicit base_lightgbm_params."
+            raise ValueError(message)
+        params = {key: value for key, value in base_lightgbm_params.items() if key != "model_name"}
+        params.update(
+            {
                 "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                 "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 "num_leaves": trial.suggest_int("num_leaves", 15, 63),
@@ -82,15 +163,10 @@
                 "min_child_samples": trial.suggest_int("min_child_samples", 10, 60, step=5),
                 "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                 "lambda_l2": trial.suggest_float("lambda_l2", 1.0e-4, 10.0, log=True),
-                "n_factors": trial.suggest_int("n_factors", 2, min(12, target_dim)),
-            }
-        )
-        params.setdefault("device_type", "gpu")
-        params.setdefault("random_state", 7)
-        params.setdefault("objective", "regression")
-        params.setdefault("metric", "l2")
-        params.setdefault("verbosity", -1)
-        return LightGBMSurfaceModel(**params)
+                "n_factors": trial.suggest_int("n_factors", 2, min(12, target_dim)),
+            }
+        )
+        return LightGBMSurfaceModel(**_lightgbm_params(params))
     if model_name == "random_forest":
         return RandomForestSurfaceModel(
             n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
@@ -156,12 +232,18 @@
             alpha=_float_param(params, "alpha"),
             target_dim=target_dim,
         )
-    if model_name == "lightgbm":
-        return LightGBMSurfaceModel(**dict(params))
-    if model_name == "random_forest":
-        return RandomForestSurfaceModel(**dict(params))
+    if model_name == "lightgbm":
+        return LightGBMSurfaceModel(**_lightgbm_params(params))
+    if model_name == "random_forest":
+        return RandomForestSurfaceModel(**_random_forest_params(params))
     if model_name == "neural_surface":
-        config = base_neural_config.model_copy(update=dict(params))
+        config = NeuralModelConfig.model_validate(
+            {
+                **base_neural_config.model_dump(mode="python"),
+                **dict(params),
+            },
+            strict=True,
+        )
         return NeuralSurfaceRegressor(
             config=config,
             grid_shape=grid_shape,
--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/models/lightgbm_model.py	2026-04-26 20:36:14
+++ /Volumes/T9/ECON499/src/ivsurf/models/lightgbm_model.py	2026-04-27 19:57:46
@@ -19,14 +19,17 @@

     def __init__(self, **params: Any) -> None:
         self.params = dict(params)
-        raw_n_factors = self.params.pop("n_factors", 4)
-        if not isinstance(raw_n_factors, int | float | str):
+        if "n_factors" not in self.params:
+            message = "LightGBM params must include integer n_factors."
+            raise KeyError(message)
+        raw_n_factors = self.params.pop("n_factors")
+        if isinstance(raw_n_factors, bool) or not isinstance(raw_n_factors, int):
             message = (
-                "LightGBM n_factors must be integer-like, found "
-                f"{type(raw_n_factors).__name__}."
+                "LightGBM n_factors must be an integer, "
+                f"found {type(raw_n_factors).__name__}."
             )
             raise TypeError(message)
-        self.n_factors = int(raw_n_factors)
+        self.n_factors = raw_n_factors
         if self.n_factors <= 0:
             message = f"LightGBM n_factors must be positive, found {self.n_factors}."
             raise ValueError(message)
--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/evaluation/alignment.py	2026-04-27 18:45:16
+++ /Volumes/T9/ECON499/src/ivsurf/evaluation/alignment.py	2026-04-27 20:00:04
@@ -18,9 +18,24 @@
     COMPLETION_STATUS_EXTRAPOLATED,
     COMPLETION_STATUS_INTERPOLATED,
 )
-
-
-def _require_files(paths: list[Path], description: str) -> None:
+
+FORECAST_DUPLICATE_KEY_COLUMNS = (
+    "model_name",
+    "quote_date",
+    "target_date",
+    "maturity_index",
+    "moneyness_index",
+)
+FORECAST_COVERAGE_KEY_COLUMNS = (
+    "quote_date",
+    "target_date",
+    "split_id",
+    "maturity_index",
+    "moneyness_index",
+)
+
+
+def _require_files(paths: list[Path], description: str) -> None:
     if not paths:
         message = f"No {description} files found."
         raise FileNotFoundError(message)
@@ -40,6 +55,84 @@
             raise ValueError(message)


+def _require_unique_keys(
+    frame: pl.DataFrame,
+    *,
+    key_columns: tuple[str, ...],
+    context: str,
+) -> None:
+    missing_columns = [column for column in key_columns if column not in frame.columns]
+    if missing_columns:
+        message = f"{context} is missing key columns: {', '.join(missing_columns)}."
+        raise ValueError(message)
+    duplicate_keys = (
+        frame.group_by(key_columns)
+        .agg(pl.len().alias("row_count"))
+        .filter(pl.col("row_count") > 1)
+        .sort(key_columns)
+    )
+    if duplicate_keys.is_empty():
+        return
+    preview = _format_key_preview(duplicate_keys, key_columns=key_columns)
+    message = (
+        f"{context} contains duplicate rows for key columns {key_columns!r}; "
+        f"duplicate key count={duplicate_keys.height}. First duplicates: {preview}."
+    )
+    raise ValueError(message)
+
+
+def _format_key_preview(frame: pl.DataFrame, *, key_columns: tuple[str, ...]) -> str:
+    rows = []
+    for row in frame.select(key_columns).head(5).iter_rows(named=True):
+        rows.append(", ".join(f"{column}={row[column]!r}" for column in key_columns))
+    return "; ".join(rows)
+
+
+def assert_equal_forecast_model_coverage(forecast_frame: pl.DataFrame) -> None:
+    """Fail fast unless every model forecasts the identical date-cell universe."""
+
+    _require_unique_keys(
+        forecast_frame,
+        key_columns=FORECAST_DUPLICATE_KEY_COLUMNS,
+        context="Forecast artifacts",
+    )
+    _require_non_null_columns(
+        forecast_frame,
+        columns=("model_name", *FORECAST_COVERAGE_KEY_COLUMNS),
+    )
+    model_names = tuple(sorted(str(value) for value in forecast_frame["model_name"].unique()))
+    if not model_names:
+        message = "Forecast artifacts must contain at least one model."
+        raise ValueError(message)
+    reference_model = model_names[0]
+    reference_keys = (
+        forecast_frame.filter(pl.col("model_name") == reference_model)
+        .select(FORECAST_COVERAGE_KEY_COLUMNS)
+        .unique()
+        .sort(FORECAST_COVERAGE_KEY_COLUMNS)
+    )
+    for model_name in model_names[1:]:
+        model_keys = (
+            forecast_frame.filter(pl.col("model_name") == model_name)
+            .select(FORECAST_COVERAGE_KEY_COLUMNS)
+            .unique()
+            .sort(FORECAST_COVERAGE_KEY_COLUMNS)
+        )
+        missing = reference_keys.join(model_keys, on=FORECAST_COVERAGE_KEY_COLUMNS, how="anti")
+        extra = model_keys.join(reference_keys, on=FORECAST_COVERAGE_KEY_COLUMNS, how="anti")
+        if missing.is_empty() and extra.is_empty():
+            continue
+        message = (
+            "Forecast artifacts must have identical coverage across models. "
+            f"Model {model_name!r} differs from reference model {reference_model!r}: "
+            f"missing={missing.height}, extra={extra.height}. "
+            "First missing: "
+            f"{_format_key_preview(missing, key_columns=FORECAST_COVERAGE_KEY_COLUMNS)}. "
+            f"First extra: {_format_key_preview(extra, key_columns=FORECAST_COVERAGE_KEY_COLUMNS)}."
+        )
+        raise ValueError(message)
+
+
 def _require_equal_columns(
     frame: pl.DataFrame,
     *,
@@ -88,13 +181,13 @@
     return "; ".join(violations)


-def load_actual_surface_frame(gold_dir: Path) -> pl.DataFrame:
+def load_actual_surface_frame(gold_dir: Path) -> pl.DataFrame:
     """Load persisted daily surface artifacts."""

     gold_files = sorted_artifact_files(gold_dir, "year=*/*.parquet")
     _require_files(gold_files, "gold surface")
-    return (
-        scan_parquet_files(gold_files)
+    frame = (
+        scan_parquet_files(gold_files)
         .select(
             "quote_date",
             "maturity_index",
@@ -117,8 +210,14 @@
             "vega_sum",
         )
         .collect(engine="streaming")
-        .sort(["quote_date", "maturity_index", "moneyness_index"])
-    )
+        .sort(["quote_date", "maturity_index", "moneyness_index"])
+    )
+    _require_unique_keys(
+        frame,
+        key_columns=("quote_date", "maturity_index", "moneyness_index"),
+        context="Gold surface artifacts",
+    )
+    return frame


 def load_forecast_frame(forecast_dir: Path) -> pl.DataFrame:
@@ -126,11 +225,13 @@

     forecast_files = sorted_artifact_files(forecast_dir, "*.parquet")
     _require_files(forecast_files, "forecast")
-    return (
-        scan_parquet_files(forecast_files)
-        .collect(engine="streaming")
-        .sort(["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"])
-    )
+    frame = (
+        scan_parquet_files(forecast_files)
+        .collect(engine="streaming")
+        .sort(["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"])
+    )
+    assert_equal_forecast_model_coverage(frame)
+    return frame


 def load_daily_spot_frame(silver_dir: Path) -> pl.DataFrame:
@@ -200,15 +301,21 @@
     raise ValueError(message)


-def build_forecast_realization_panel(
+def build_forecast_realization_panel(
     actual_surface_frame: pl.DataFrame,
     forecast_frame: pl.DataFrame,
     *,
     total_variance_floor: float,
 ) -> pl.DataFrame:
-    """Align forecast artifacts with realized target-day surfaces and origin-day references."""
-
-    actual_target = actual_surface_frame.rename(
+    """Align forecast artifacts with realized target-day surfaces and origin-day references."""
+
+    _require_unique_keys(
+        actual_surface_frame,
+        key_columns=("quote_date", "maturity_index", "moneyness_index"),
+        context="Actual surface frame",
+    )
+    assert_equal_forecast_model_coverage(forecast_frame)
+    actual_target = actual_surface_frame.rename(
         {
             "quote_date": "target_date",
             "observed_total_variance": "actual_observed_total_variance",
--- /tmp/econ499_round004_extract.ZdatuB/scripts/07_run_stats.py	2026-04-27 18:45:18
+++ /Volumes/T9/ECON499/scripts/07_run_stats.py	2026-04-27 19:59:35
@@ -44,18 +44,76 @@
 app = typer.Typer(add_completion=False)


-def _daily_loss_matrix(
-    loss_frame: pl.DataFrame,
-    metric_column: str,
-) -> tuple[np.ndarray, tuple[str, ...], tuple[object, ...]]:
-    pivoted = (
-        loss_frame.select("model_name", "target_date", metric_column)
-        .pivot(on="model_name", index="target_date", values=metric_column)
-        .sort("target_date")
-    )
-    model_columns = tuple(column for column in pivoted.columns if column != "target_date")
-    matrix = pivoted.select(model_columns).to_numpy().astype(np.float64)
-    return matrix, model_columns, tuple(pivoted["target_date"].to_list())
+def _daily_loss_matrix(
+    loss_frame: pl.DataFrame,
+    metric_column: str,
+) -> tuple[np.ndarray, tuple[str, ...], tuple[object, ...]]:
+    if metric_column not in loss_frame.columns:
+        message = f"Daily loss frame is missing configured metric column {metric_column!r}."
+        raise ValueError(message)
+    duplicate_keys = (
+        loss_frame.group_by(["model_name", "target_date"])
+        .agg(pl.len().alias("row_count"))
+        .filter(pl.col("row_count") > 1)
+    )
+    if not duplicate_keys.is_empty():
+        message = (
+            "Daily loss frame contains duplicate model_name/target_date rows; "
+            f"duplicate key count={duplicate_keys.height}."
+        )
+        raise ValueError(message)
+    pivoted = (
+        loss_frame.select("model_name", "target_date", metric_column)
+        .pivot(on="model_name", index="target_date", values=metric_column)
+        .sort("target_date")
+    )
+    model_columns = tuple(column for column in pivoted.columns if column != "target_date")
+    if not model_columns:
+        message = "Daily loss matrix must contain at least one model column."
+        raise ValueError(message)
+    if any(pivoted[column].null_count() > 0 for column in model_columns):
+        message = (
+            "Daily loss matrix contains missing model/date coverage after pivoting "
+            f"metric {metric_column!r}."
+        )
+        raise ValueError(message)
+    matrix = pivoted.select(model_columns).to_numpy().astype(np.float64)
+    if not np.isfinite(matrix).all():
+        message = f"Daily loss matrix for metric {metric_column!r} contains non-finite values."
+        raise ValueError(message)
+    return matrix, model_columns, tuple(pivoted["target_date"].to_list())
+
+
+def _require_finite_loss_metrics(
+    daily_loss_frame: pl.DataFrame,
+    *,
+    loss_metrics: tuple[str, ...],
+) -> None:
+    for metric_column in loss_metrics:
+        _daily_loss_matrix(daily_loss_frame, metric_column)
+
+
+def _require_matching_loss_matrix_contract(
+    *,
+    metric_column: str,
+    metric_model_columns: tuple[str, ...],
+    metric_target_dates: tuple[object, ...],
+    reference_model_columns: tuple[str, ...],
+    reference_target_dates: tuple[object, ...],
+) -> None:
+    if metric_model_columns != reference_model_columns:
+        message = (
+            "All loss metrics must produce the same model ordering in the daily loss frame, "
+            f"found {metric_model_columns!r} != {reference_model_columns!r} "
+            f"for {metric_column}."
+        )
+        raise ValueError(message)
+    if metric_target_dates != reference_target_dates:
+        message = (
+            "All loss metrics must produce the same target-date coverage in the daily loss "
+            f"frame for {metric_column}."
+        )
+        raise ValueError(message)


 def _loss_summary_frame(
@@ -144,9 +202,9 @@
             extra_tokens={
                 "run_profile_name": run_profile_name,
                 "workflow_run_label": workflow_paths.run_label,
-                "artifact_schema_version": 3,
+                "artifact_schema_version": 4,
             },
-        ),
+        ),
     )
     tuning_results = load_required_tuning_results(
         raw_config.manifests_dir,
@@ -197,18 +255,22 @@
                 positive_floor=metrics_config.positive_floor,
                 full_grid_weighting=stats_config.full_grid_weighting,
             )
-            write_parquet_frame(daily_loss_frame, daily_loss_path)
-            resumer.mark_complete(
-                "daily_loss_frame",
-                output_paths=[daily_loss_path],
-                metadata={"rows": daily_loss_frame.height},
-            )
-
-        primary_metric_column = stats_config.loss_metrics[0]
-        model_columns = _daily_loss_matrix(
-            loss_frame=daily_loss_frame,
-            metric_column=primary_metric_column,
-        )[1]
+            write_parquet_frame(daily_loss_frame, daily_loss_path)
+            resumer.mark_complete(
+                "daily_loss_frame",
+                output_paths=[daily_loss_path],
+                metadata={"rows": daily_loss_frame.height},
+            )
+
+        _require_finite_loss_metrics(
+            daily_loss_frame,
+            loss_metrics=stats_config.loss_metrics,
+        )
+        primary_metric_column = stats_config.loss_metrics[0]
+        _, model_columns, target_dates = _daily_loss_matrix(
+            loss_frame=daily_loss_frame,
+            metric_column=primary_metric_column,
+        )
         benchmark_model = stats_config.benchmark_model
         if benchmark_model not in model_columns:
             message = f"Benchmark model {benchmark_model} not found in loss frame."
@@ -223,19 +285,19 @@
         )
         if not resumer.item_complete("dm_results", required_output_paths=[dm_results_path]):
             resumer.clear_item("dm_results", output_paths=[dm_results_path])
-            dm_results: list[dict[str, object]] = []
-            for metric_column in stats_config.loss_metrics:
-                loss_matrix, metric_model_columns, _ = _daily_loss_matrix(
-                    loss_frame=daily_loss_frame,
-                    metric_column=metric_column,
-                )
-                if metric_model_columns != model_columns:
-                    message = (
-                        "All loss metrics must produce the same model ordering in the daily "
-                        f"loss frame, found {metric_model_columns!r} != {model_columns!r} "
-                        f"for {metric_column}."
-                    )
-                    raise ValueError(message)
+            dm_results: list[dict[str, object]] = []
+            for metric_column in stats_config.loss_metrics:
+                loss_matrix, metric_model_columns, metric_target_dates = _daily_loss_matrix(
+                    loss_frame=daily_loss_frame,
+                    metric_column=metric_column,
+                )
+                _require_matching_loss_matrix_contract(
+                    metric_column=metric_column,
+                    metric_model_columns=metric_model_columns,
+                    metric_target_dates=metric_target_dates,
+                    reference_model_columns=model_columns,
+                    reference_target_dates=target_dates,
+                )
                 benchmark_index = metric_model_columns.index(benchmark_model)
                 benchmark_losses = loss_matrix[:, benchmark_index]
                 for model_name in metric_model_columns:
@@ -273,12 +335,19 @@
         progress.update(task_id, description="Stage 07 running SPA bootstrap")
         if not resumer.item_complete("spa_result", required_output_paths=[spa_result_path]):
             resumer.clear_item("spa_result", output_paths=[spa_result_path])
-            spa_results: list[dict[str, object]] = []
-            for metric_column in stats_config.loss_metrics:
-                loss_matrix, metric_model_columns, _ = _daily_loss_matrix(
-                    loss_frame=daily_loss_frame,
-                    metric_column=metric_column,
-                )
+            spa_results: list[dict[str, object]] = []
+            for metric_column in stats_config.loss_metrics:
+                loss_matrix, metric_model_columns, metric_target_dates = _daily_loss_matrix(
+                    loss_frame=daily_loss_frame,
+                    metric_column=metric_column,
+                )
+                _require_matching_loss_matrix_contract(
+                    metric_column=metric_column,
+                    metric_model_columns=metric_model_columns,
+                    metric_target_dates=metric_target_dates,
+                    reference_model_columns=model_columns,
+                    reference_target_dates=target_dates,
+                )
                 candidate_models = tuple(
                     model for model in metric_model_columns if model != benchmark_model
                 )
@@ -318,12 +387,19 @@
         progress.update(task_id, description="Stage 07 running simplified Tmax bootstrap")
         if not resumer.item_complete("mcs_result", required_output_paths=[mcs_result_path]):
             resumer.clear_item("mcs_result", output_paths=[mcs_result_path])
-            mcs_results: list[dict[str, object]] = []
-            for metric_column in stats_config.loss_metrics:
-                loss_matrix, metric_model_columns, _ = _daily_loss_matrix(
-                    loss_frame=daily_loss_frame,
-                    metric_column=metric_column,
-                )
+            mcs_results: list[dict[str, object]] = []
+            for metric_column in stats_config.loss_metrics:
+                loss_matrix, metric_model_columns, metric_target_dates = _daily_loss_matrix(
+                    loss_frame=daily_loss_frame,
+                    metric_column=metric_column,
+                )
+                _require_matching_loss_matrix_contract(
+                    metric_column=metric_column,
+                    metric_model_columns=metric_model_columns,
+                    metric_target_dates=metric_target_dates,
+                    reference_model_columns=model_columns,
+                    reference_target_dates=target_dates,
+                )
                 mcs_result = asdict(
                     model_confidence_set(
                         losses=loss_matrix,
--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/stats/diebold_mariano.py	2026-04-08 14:03:06
+++ /Volumes/T9/ECON499/src/ivsurf/stats/diebold_mariano.py	2026-04-27 19:58:55
@@ -59,11 +59,25 @@
     if loss_a.ndim != 1:
         message = "DM test expects one-dimensional loss series."
         raise ValueError(message)
+    if loss_a.size == 0:
+        message = "DM test requires at least one aligned loss observation."
+        raise ValueError(message)
+    if not np.isfinite(loss_a).all() or not np.isfinite(loss_b).all():
+        message = "DM test loss series must contain only finite values."
+        raise ValueError(message)
+    if max_lag < 0:
+        message = "DM test max_lag must be non-negative."
+        raise ValueError(message)
+    if max_lag >= loss_a.size:
+        message = f"DM test max_lag must be smaller than n_obs ({max_lag} >= {loss_a.size})."
+        raise ValueError(message)
     if alternative not in {"two-sided", "greater", "less"}:
         message = f"Unsupported DM alternative: {alternative}"
         raise ValueError(message)

-    differential = loss_a.astype(np.float64) - loss_b.astype(np.float64)
+    loss_a_float = loss_a.astype(np.float64)
+    loss_b_float = loss_b.astype(np.float64)
+    differential = loss_a_float - loss_b_float
     mean_diff = float(differential.mean())
     lrv = long_run_variance(differential, max_lag=max_lag)
     n_obs = differential.shape[0]
@@ -88,12 +102,11 @@
         model_a=model_a,
         model_b=model_b,
         n_obs=n_obs,
-        mean_loss_a=float(loss_a.mean()),
-        mean_loss_b=float(loss_b.mean()),
+        mean_loss_a=float(loss_a_float.mean()),
+        mean_loss_b=float(loss_b_float.mean()),
         mean_differential=mean_diff,
         statistic=float(statistic),
         p_value=float(np.clip(p_value, 0.0, 1.0)),
         alternative=alternative,
         max_lag=max_lag,
     )
-
--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/stats/spa.py	2026-04-09 13:59:40
+++ /Volumes/T9/ECON499/src/ivsurf/stats/spa.py	2026-04-27 19:59:01
@@ -45,10 +45,21 @@
     if benchmark_losses.shape[0] != candidate_losses.shape[0]:
         message = "benchmark and candidate losses must align in time."
         raise ValueError(message)
+    if benchmark_losses.size == 0:
+        message = "SPA test requires at least one aligned loss observation."
+        raise ValueError(message)
+    if candidate_losses.shape[1] == 0:
+        message = "SPA test requires at least one candidate model."
+        raise ValueError(message)
     if candidate_losses.shape[1] != len(candidate_models):
         message = "candidate_models length must match candidate_losses columns."
         raise ValueError(message)
+    if not np.isfinite(benchmark_losses).all() or not np.isfinite(candidate_losses).all():
+        message = "SPA test losses must contain only finite values."
+        raise ValueError(message)

+    benchmark_losses = benchmark_losses.astype(np.float64)
+    candidate_losses = candidate_losses.astype(np.float64)
     differentials = benchmark_losses[:, None] - candidate_losses
     means = differentials.mean(axis=0)
     scales = differentials.std(axis=0, ddof=1)
--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/stats/mcs.py	2026-04-09 15:10:42
+++ /Volumes/T9/ECON499/src/ivsurf/stats/mcs.py	2026-04-27 19:59:10
@@ -68,9 +68,18 @@
     if losses.ndim != 2:
         message = "losses must be a two-dimensional array."
         raise ValueError(message)
+    if losses.shape[0] == 0:
+        message = "MCS requires at least one aligned loss observation."
+        raise ValueError(message)
+    if losses.shape[1] == 0:
+        message = "MCS requires at least one model."
+        raise ValueError(message)
     if losses.shape[1] != len(model_names):
         message = "model_names length must match losses columns."
         raise ValueError(message)
+    if not np.isfinite(losses).all():
+        message = "MCS losses must contain only finite values."
+        raise ValueError(message)

     remaining_losses = losses.astype(np.float64)
     remaining_models = list(model_names)
--- /tmp/econ499_round004_extract.ZdatuB/src/ivsurf/reproducibility.py	2026-04-08 19:32:30
+++ /Volumes/T9/ECON499/src/ivsurf/reproducibility.py	2026-04-27 20:00:37
@@ -230,13 +230,20 @@
     split_manifest_hash = (
         sha256_file(split_manifest_path.resolve()) if split_manifest_path is not None else None
     )
+    commit_hash = git_commit_hash(repo_root)
+    if commit_hash is None:
+        message = (
+            "Cannot write run manifest without a git commit hash. "
+            f"repo_root={repo_root.resolve()} is not a usable Git checkout."
+        )
+        raise RuntimeError(message)
     manifest: dict[str, Any] = {
         "schema_version": 1,
         "script_name": script_name,
         "started_at_utc": started_at.astimezone(UTC).isoformat(),
         "completed_at_utc": finished_at.isoformat(),
         "duration_seconds": (finished_at - started_at.astimezone(UTC)).total_seconds(),
-        "git_commit_hash": git_commit_hash(repo_root),
+        "git_commit_hash": commit_hash,
         "random_seed": random_seed,
         "config_snapshots": [asdict(snapshot) for snapshot in config_snapshots],
         "package_versions": collect_package_versions(),
--- /tmp/econ499_round004_extract.ZdatuB/tests/unit/test_option_filters.py	2026-04-27 18:29:42
+++ /Volumes/T9/ECON499/tests/unit/test_option_filters.py	2026-04-27 19:50:09
@@ -18,9 +18,11 @@
         "implied_volatility_1545": 0.2,
         "vega_1545": 1.0,
         "active_underlying_price_1545": 100.0,
+        "strike": 100.0,
         "mid_1545": 1.1,
         "tau_years": 0.1,
         "log_moneyness": 0.0,
+        "total_variance": 0.004,
     }
     rows = [dict(row), dict(row)]
     if missing_column is not None:
@@ -38,9 +40,11 @@
             "implied_volatility_1545": [0.0],
             "vega_1545": [1.0],
             "active_underlying_price_1545": [100.0],
+            "strike": [100.0],
             "mid_1545": [1.1],
             "tau_years": [0.1],
             "log_moneyness": [0.0],
+            "total_variance": [0.0],
         }
     )
     flagged = apply_option_quality_flags(frame=frame, config=CleaningConfig())
@@ -58,9 +62,11 @@
             "implied_volatility_1545": [0.2, 0.2],
             "vega_1545": [1.0, 1.0],
             "active_underlying_price_1545": [100.0, 100.0],
+            "strike": [100.0, 100.0],
             "mid_1545": [0.0, 0.15],
             "tau_years": [0.1, 0.1],
             "log_moneyness": [0.0, 0.0],
+            "total_variance": [0.004, 0.004],
         }
     )
     flagged = apply_option_quality_flags(
@@ -86,9 +92,11 @@
             "implied_volatility_1545": [0.2, None],
             "vega_1545": [1.0, 1.0],
             "active_underlying_price_1545": [100.0, 100.0],
+            "strike": [100.0, 100.0],
             "mid_1545": [1.1, 1.1],
             "tau_years": [0.1, 0.1],
             "log_moneyness": [0.0, 0.0],
+            "total_variance": [0.004, 0.004],
         }
     )

@@ -110,8 +118,10 @@
         ("implied_volatility_1545", "MISSING_IMPLIED_VOLATILITY_1545"),
         ("vega_1545", "MISSING_VEGA_1545"),
         ("active_underlying_price_1545", "MISSING_ACTIVE_UNDERLYING_PRICE_1545"),
+        ("strike", "MISSING_STRIKE"),
         ("mid_1545", "MISSING_MID_1545"),
         ("tau_years", "MISSING_TAU_YEARS"),
+        ("total_variance", "MISSING_TOTAL_VARIANCE"),
         ("log_moneyness", "MISSING_LOG_MONEYNESS"),
     ),
 )
@@ -127,3 +137,56 @@
     assert flagged["invalid_reason"].to_list() == [expected_reason, None]
     assert flagged["is_valid_observation"].to_list() == [False, True]
     assert valid_option_rows(flagged).height == 1
+
+
+@pytest.mark.parametrize(
+    ("column_name", "bad_value", "expected_reason"),
+    (
+        ("bid_1545", float("nan"), "NONFINITE_BID_1545"),
+        ("bid_1545", float("inf"), "NONFINITE_BID_1545"),
+        ("bid_1545", float("-inf"), "NONFINITE_BID_1545"),
+        ("ask_1545", float("nan"), "NONFINITE_ASK_1545"),
+        ("implied_volatility_1545", float("inf"), "NONFINITE_IV_1545"),
+        ("vega_1545", float("-inf"), "NONFINITE_VEGA_1545"),
+        ("active_underlying_price_1545", float("nan"), "NONFINITE_UNDERLYING_1545"),
+        ("strike", float("inf"), "NONFINITE_STRIKE"),
+        ("mid_1545", float("nan"), "NONFINITE_MID_1545"),
+        ("tau_years", float("inf"), "NONFINITE_TAU_YEARS"),
+        ("total_variance", float("nan"), "NONFINITE_TOTAL_VARIANCE"),
+        ("log_moneyness", float("-inf"), "NONFINITE_LOG_MONEYNESS"),
+    ),
+)
+def test_every_critical_nonfinite_field_is_invalid_with_explicit_reason(
+    column_name: str,
+    bad_value: float,
+    expected_reason: str,
+) -> None:
+    frame = _base_cleaning_frame()
+    frame = frame.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(bad_value).otherwise(
+        pl.col(column_name)
+    ).alias(column_name))
+
+    flagged = apply_option_quality_flags(frame=frame, config=CleaningConfig())
+
+    assert flagged["invalid_reason"].to_list() == [expected_reason, None]
+    assert flagged["is_valid_observation"].to_list() == [False, True]
+    assert valid_option_rows(flagged).height == 1
+
+
+@pytest.mark.parametrize("bad_strike", (0.0, -1.0))
+def test_non_positive_strike_is_invalid_before_log_moneyness_reason(bad_strike: float) -> None:
+    frame = _base_cleaning_frame().with_columns(
+        pl.when(pl.int_range(pl.len()) == 0)
+        .then(pl.lit(bad_strike))
+        .otherwise(pl.col("strike"))
+        .alias("strike"),
+        pl.when(pl.int_range(pl.len()) == 0)
+        .then(pl.lit(float("-inf")))
+        .otherwise(pl.col("log_moneyness"))
+        .alias("log_moneyness"),
+    )
+
+    flagged = apply_option_quality_flags(frame=frame, config=CleaningConfig())
+
+    assert flagged["invalid_reason"].to_list() == ["NON_POSITIVE_STRIKE", None]
+    assert flagged["is_valid_observation"].to_list() == [False, True]
--- /tmp/econ499_round004_extract.ZdatuB/tests/integration/test_early_close_stage02.py	2026-04-27 18:30:08
+++ /Volumes/T9/ECON499/tests/integration/test_early_close_stage02.py	2026-04-27 19:50:51
@@ -23,18 +23,24 @@
     return path


-def _bronze_row(*, bid: float | None = 10.0, ask: float | None = 10.5) -> dict[str, object]:
+def _bronze_row(
+    *,
+    bid: float | None = 10.0,
+    ask: float | None = 10.5,
+    strike: float | None = 3100.0,
+    implied_volatility: float | None = 0.2,
+) -> dict[str, object]:
     return {
         "quote_date": date(2019, 11, 29),
         "expiration": date(2019, 12, 20),
         "root": "SPXW",
-        "strike": 3100.0,
+        "strike": strike,
         "option_type": "C",
         "bid_1545": bid,
         "ask_1545": ask,
         "vega_1545": 1.0,
         "active_underlying_price_1545": 3140.0,
-        "implied_volatility_1545": 0.2,
+        "implied_volatility_1545": implied_volatility,
     }


@@ -192,3 +198,87 @@
     assert list(summary[0]["invalid_reason_counts"]) == sorted(expected_counts)
     assert sum(summary[0]["invalid_reason_counts"].values()) == summary[0]["rows"]
     assert summary[0]["invalid_reason_counts"]["VALID"] == summary[0]["valid_rows"]
+
+
+def test_stage02_accounts_for_nonfinite_and_nonpositive_inputs_as_cleaning_invalids(
+    tmp_path: Path,
+) -> None:
+    repo_root = Path(__file__).resolve().parents[2]
+    bronze_dir = tmp_path / "data" / "bronze" / "year=2019"
+    bronze_dir.mkdir(parents=True)
+    silver_dir = tmp_path / "data" / "silver"
+    gold_dir = tmp_path / "data" / "gold"
+    manifests_dir = tmp_path / "data" / "manifests"
+    manifests_dir.mkdir(parents=True)
+    (manifests_dir / "bronze_ingestion_summary.json").write_bytes(orjson.dumps([]))
+
+    bronze_frame = pl.DataFrame(
+        [
+            _bronze_row(),
+            _bronze_row(bid=float("nan")),
+            _bronze_row(ask=float("inf")),
+            _bronze_row(strike=0.0),
+            _bronze_row(implied_volatility=float("-inf")),
+        ]
+    )
+    bronze_path = bronze_dir / "2019-11-29.parquet"
+    bronze_frame.write_parquet(bronze_path)
+
+    raw_config_path = _write_yaml(
+        tmp_path / "raw.yaml",
+        (
+            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
+            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
+            f"silver_dir: '{silver_dir.as_posix()}'\n"
+            f"gold_dir: '{gold_dir.as_posix()}'\n"
+            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
+            'sample_start_date: "2019-11-29"\n'
+            'sample_end_date: "2019-11-29"\n'
+        ),
+    )
+    cleaning_config_path = _write_yaml(
+        tmp_path / "cleaning.yaml",
+        (
+            'target_symbol: "^SPX"\n'
+            'allowed_option_types: ["C", "P"]\n'
+            "min_valid_bid_exclusive: 0.0\n"
+            "min_valid_ask_exclusive: 0.0\n"
+            "require_ask_ge_bid: true\n"
+            "require_positive_iv: true\n"
+            "require_positive_vega: true\n"
+            "require_positive_underlying_price: true\n"
+            "min_valid_mid_price_exclusive: 0.0\n"
+            "max_abs_log_moneyness: 0.5\n"
+            "min_tau_years: 0.0001\n"
+            "max_tau_years: 2.5\n"
+        ),
+    )
+
+    script_module = _load_script_module(
+        repo_root / "scripts" / "02_build_option_panel.py",
+        "stage02_nonfinite_reason_summary_script",
+    )
+    script_module.main(
+        raw_config_path=raw_config_path,
+        cleaning_config_path=cleaning_config_path,
+    )
+
+    silver_output = pl.read_parquet(silver_dir / "year=2019" / "2019-11-29.parquet")
+    summary = orjson.loads((manifests_dir / "silver_build_summary.json").read_bytes())
+
+    assert silver_output["invalid_reason"].to_list() == [
+        None,
+        "NONFINITE_BID_1545",
+        "NONFINITE_ASK_1545",
+        "NON_POSITIVE_STRIKE",
+        "NONFINITE_IV_1545",
+    ]
+    assert silver_output["is_valid_observation"].to_list() == [True, False, False, False, False]
+    assert summary[0]["valid_rows"] == 1
+    assert summary[0]["invalid_reason_counts"] == {
+        "NONFINITE_ASK_1545": 1,
+        "NONFINITE_BID_1545": 1,
+        "NONFINITE_IV_1545": 1,
+        "NON_POSITIVE_STRIKE": 1,
+        "VALID": 1,
+    }
--- /tmp/econ499_round004_extract.ZdatuB/tests/integration/test_stage03_stage04_target_gap_alignment.py	2026-04-27 18:48:00
+++ /Volumes/T9/ECON499/tests/integration/test_stage03_stage04_target_gap_alignment.py	2026-04-27 19:51:08
@@ -7,6 +7,7 @@

 import orjson
 import polars as pl
+import pytest

 from ivsurf.reproducibility import sha256_file
 from ivsurf.surfaces.grid import (
@@ -263,3 +264,62 @@
     assert str((manifests_dir / "gold_surface_skipped_dates.json").resolve()) in stage04_input_paths
     for gold_file in gold_files:
         assert str(gold_file.resolve()) in stage04_input_paths
+
+
+def test_stage03_rejects_nonfinite_valid_silver_surface_inputs(tmp_path: Path) -> None:
+    repo_root = Path(__file__).resolve().parents[2]
+    silver_year = tmp_path / "data" / "silver" / "year=2021"
+    silver_year.mkdir(parents=True)
+    manifests_dir = tmp_path / "data" / "manifests"
+    manifests_dir.mkdir(parents=True)
+    silver_path = silver_year / "2021-01-04.parquet"
+    pl.DataFrame(
+        [
+            {
+                "quote_date": date(2021, 1, 4),
+                "effective_decision_timestamp": "2021-01-04T15:45:00-05:00",
+                "tau_years": 30.0 / 365.0,
+                "log_moneyness": 0.0,
+                "total_variance": float("inf"),
+                "implied_volatility_1545": 0.2,
+                "vega_1545": 1.0,
+                "spread_1545": 0.01,
+                "is_valid_observation": True,
+            }
+        ]
+    ).write_parquet(silver_path)
+    (manifests_dir / "silver_build_summary.json").write_bytes(orjson.dumps([]))
+
+    raw_config_path = _write_yaml(
+        tmp_path / "raw.yaml",
+        (
+            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
+            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
+            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
+            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
+            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
+            'sample_start_date: "2021-01-04"\n'
+            'sample_end_date: "2021-01-04"\n'
+        ),
+    )
+    surface_config_path = _write_yaml(
+        tmp_path / "surface.yaml",
+        (
+            "moneyness_points: [0.0]\n"
+            "maturity_days: [30]\n"
+            'interpolation_order: ["maturity", "moneyness"]\n'
+            "interpolation_cycles: 2\n"
+            "total_variance_floor: 1.0e-8\n"
+            "observed_cell_min_count: 1\n"
+        ),
+    )
+
+    stage03 = _load_script_module(
+        repo_root / "scripts" / "03_build_surfaces.py",
+        "stage03_nonfinite_valid_silver_script",
+    )
+    with pytest.raises(ValueError, match="null or non-finite surface inputs"):
+        stage03.main(
+            raw_config_path=raw_config_path,
+            surface_config_path=surface_config_path,
+        )
--- /dev/null	2026-04-27 20:03:11
+++ /Volumes/T9/ECON499/tests/unit/test_split_manifests.py	2026-04-27 19:55:20
@@ -0,0 +1,111 @@
+from __future__ import annotations
+
+from dataclasses import asdict
+from datetime import date
+from pathlib import Path
+
+import orjson
+import pytest
+
+from ivsurf.config import WalkforwardConfig
+from ivsurf.splits.manifests import (
+    SPLIT_MANIFEST_GENERATOR,
+    SPLIT_MANIFEST_SCHEMA_VERSION,
+    WalkforwardSplit,
+    date_universe_hash,
+    load_split_manifest,
+    require_split_manifest_matches_artifacts,
+    serialize_splits,
+)
+
+
+def _split() -> WalkforwardSplit:
+    return WalkforwardSplit(
+        split_id="split_0000",
+        train_dates=("2021-01-04", "2021-01-05"),
+        validation_dates=("2021-01-06",),
+        test_dates=("2021-01-07",),
+    )
+
+
+def test_serialize_splits_writes_versioned_artifact_bound_manifest(tmp_path: Path) -> None:
+    dates = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6), date(2021, 1, 7)]
+    walkforward_config = WalkforwardConfig(
+        train_size=2,
+        validation_size=1,
+        test_size=1,
+        step_size=1,
+        expanding_train=True,
+    )
+    manifest_path = tmp_path / "walkforward_splits.json"
+
+    split_hash = serialize_splits(
+        [_split()],
+        manifest_path,
+        walkforward_config=walkforward_config.model_dump(mode="json"),
+        sample_window={
+            "sample_start_date": "2021-01-04",
+            "sample_end_date": "2021-01-07",
+        },
+        date_universe=dates,
+        feature_dataset_hash="daily-feature-sha",
+    )
+
+    payload = orjson.loads(manifest_path.read_bytes())
+    manifest = load_split_manifest(manifest_path)
+
+    assert split_hash
+    assert payload["schema_version"] == SPLIT_MANIFEST_SCHEMA_VERSION
+    assert payload["generator"] == SPLIT_MANIFEST_GENERATOR
+    assert payload["date_universe_hash"] == date_universe_hash(dates)
+    assert payload["feature_dataset_hash"] == "daily-feature-sha"
+    assert manifest.splits == [_split()]
+
+
+def test_load_split_manifest_rejects_legacy_bare_list(tmp_path: Path) -> None:
+    manifest_path = tmp_path / "walkforward_splits.json"
+    manifest_path.write_bytes(orjson.dumps([asdict(_split())]))
+
+    with pytest.raises(ValueError, match="Legacy bare-list split manifests"):
+        load_split_manifest(manifest_path)
+
+
+def test_require_split_manifest_matches_artifacts_rejects_stale_inputs(tmp_path: Path) -> None:
+    dates = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6), date(2021, 1, 7)]
+    manifest_path = tmp_path / "walkforward_splits.json"
+    serialize_splits(
+        [_split()],
+        manifest_path,
+        walkforward_config={
+            "train_size": 2,
+            "validation_size": 1,
+            "test_size": 1,
+            "step_size": 1,
+            "expanding_train": True,
+        },
+        sample_window={
+            "sample_start_date": "2021-01-04",
+            "sample_end_date": "2021-01-07",
+        },
+        date_universe=dates,
+        feature_dataset_hash="daily-feature-sha",
+    )
+    manifest = load_split_manifest(manifest_path)
+
+    require_split_manifest_matches_artifacts(
+        manifest,
+        date_universe=dates,
+        feature_dataset_hash="daily-feature-sha",
+    )
+    with pytest.raises(ValueError, match="date_universe_hash"):
+        require_split_manifest_matches_artifacts(
+            manifest,
+            date_universe=dates[:-1],
+            feature_dataset_hash="daily-feature-sha",
+        )
+    with pytest.raises(ValueError, match="feature_dataset_hash"):
+        require_split_manifest_matches_artifacts(
+            manifest,
+            date_universe=dates,
+            feature_dataset_hash="different-daily-feature-sha",
+        )
--- /tmp/econ499_round004_extract.ZdatuB/tests/integration/test_stage05_stage06_clean_evaluation.py	2026-04-27 18:47:42
+++ /Volumes/T9/ECON499/tests/integration/test_stage05_stage06_clean_evaluation.py	2026-04-27 20:01:21
@@ -14,7 +14,7 @@
 from ivsurf.calendar import MarketCalendar
 from ivsurf.config import WalkforwardConfig
 from ivsurf.reproducibility import sha256_file
-from ivsurf.splits.manifests import serialize_splits
+from ivsurf.splits.manifests import WalkforwardSplit, serialize_splits
 from ivsurf.splits.walkforward import build_walkforward_splits, clean_evaluation_splits
 from ivsurf.surfaces.grid import (
     MATURITY_COORDINATE,
@@ -86,6 +86,29 @@
     )


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
@@ -155,7 +178,6 @@
         config=walkforward_config,
     )
     split_manifest_path = manifests_dir / "walkforward_splits.json"
-    serialize_splits(splits, split_manifest_path)
     boundary, clean_splits = clean_evaluation_splits(splits, tuning_splits_count=3)
     expected_clean_quote_dates = {
         date.fromisoformat(day)
@@ -187,7 +209,17 @@
         ),
     )
     feature_frame = _with_surface_metadata(feature_frame, surface_config_path)
-    feature_frame.write_parquet(gold_dir / "daily_features.parquet")
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
@@ -273,7 +305,26 @@
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
+    assert set(latest_payload["extra_metadata"]["model_run_metadata"]) == {"naive", "ridge"}

+
 def test_stage05_and_stage06_preserve_target_dates_when_feature_rows_have_gaps(
     tmp_path: Path,
     monkeypatch: MonkeyPatch,
@@ -298,7 +349,6 @@
         config=walkforward_config,
     )
     split_manifest_path = manifests_dir / "walkforward_splits.json"
-    serialize_splits(splits, split_manifest_path)
     boundary, clean_splits = clean_evaluation_splits(splits, tuning_splits_count=2)
     expected_clean_quote_dates = {
         date.fromisoformat(day)
@@ -335,7 +385,17 @@
         ),
     )
     feature_frame = _with_surface_metadata(feature_frame, surface_config_path)
-    feature_frame.write_parquet(gold_dir / "daily_features.parquet")
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
@@ -448,7 +508,6 @@
         dates=feature_frame["quote_date"].to_list(),
         config=walkforward_config,
     )
-    serialize_splits(splits, manifests_dir / "walkforward_splits.json")

     raw_config_path = _write_text(
         tmp_path / "configs" / "data" / "raw.yaml",
@@ -474,7 +533,17 @@
         ),
     )
     feature_frame = _with_surface_metadata(feature_frame, surface_config_path)
-    feature_frame.write_parquet(gold_dir / "daily_features.parquet")
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
--- /tmp/econ499_round004_extract.ZdatuB/tests/unit/test_training_behaviour.py	2026-04-26 02:26:50
+++ /Volumes/T9/ECON499/tests/unit/test_training_behaviour.py	2026-04-27 19:57:38
@@ -11,11 +11,12 @@
 from ivsurf.config import NeuralModelConfig, TrainingProfileConfig
 from ivsurf.exceptions import ModelConvergenceError
 from ivsurf.models.elasticnet import ElasticNetSurfaceModel
+from ivsurf.models.har_factor import validate_har_feature_layout
 from ivsurf.models.lightgbm_model import LightGBMSurfaceModel
 from ivsurf.models.naive import validate_naive_feature_layout
 from ivsurf.models.neural_surface import NeuralSurfaceRegressor
 from ivsurf.models.ridge import RidgeSurfaceModel
-from ivsurf.training.model_factory import suggest_model_from_trial
+from ivsurf.training.model_factory import make_model_from_params, suggest_model_from_trial


 def _fit_small_lightgbm_model() -> tuple[LightGBMSurfaceModel, np.ndarray]:
@@ -348,6 +349,95 @@

     with pytest.raises(ValueError, match="lag-1 surface features"):
         validate_naive_feature_layout(
+            feature_columns=feature_columns,
+            target_columns=target_columns,
+        )
+
+
+def test_har_feature_layout_guard_accepts_aligned_har_lag_blocks() -> None:
+    target_columns = ("target_total_variance_0000", "target_total_variance_0001")
+    feature_columns = (
+        "feature_surface_mean_01_0000",
+        "feature_surface_mean_01_0001",
+        "feature_surface_mean_05_0000",
+        "feature_surface_mean_05_0001",
+        "feature_surface_mean_22_0000",
+        "feature_surface_mean_22_0001",
+        "feature_surface_change_01_0000",
+    )
+
+    validate_har_feature_layout(
+        feature_columns=feature_columns,
+        target_columns=target_columns,
+    )
+
+
+def test_har_feature_layout_guard_rejects_missing_or_reordered_lag_blocks() -> None:
+    target_columns = ("target_total_variance_0000", "target_total_variance_0001")
+    feature_columns = (
+        "feature_surface_mean_01_0000",
+        "feature_surface_mean_01_0001",
+        "feature_surface_mean_22_0000",
+        "feature_surface_mean_22_0001",
+        "feature_surface_mean_05_0000",
+        "feature_surface_mean_05_0001",
+    )
+
+    with pytest.raises(ValueError, match="lag-1, lag-5, and lag-22"):
+        validate_har_feature_layout(
             feature_columns=feature_columns,
             target_columns=target_columns,
+        )
+
+
+def test_model_factory_rejects_string_numeric_hyperparameters() -> None:
+    with pytest.raises(TypeError, match="alpha"):
+        make_model_from_params(
+            model_name="ridge",
+            params={"alpha": "1.0"},
+            target_dim=2,
+            grid_shape=(1, 2),
+            moneyness_points=(-0.1, 0.0),
+            base_neural_config=NeuralModelConfig(device="cpu"),
         )
+
+
+def test_model_factory_rejects_float_integer_hyperparameters() -> None:
+    with pytest.raises(TypeError, match="max_iter"):
+        make_model_from_params(
+            model_name="elasticnet",
+            params={"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 50_000.5, "tol": 1.0e-4},
+            target_dim=2,
+            grid_shape=(1, 2),
+            moneyness_points=(-0.1, 0.0),
+            base_neural_config=NeuralModelConfig(device="cpu"),
+        )
+
+
+def test_model_factory_rejects_string_lightgbm_integer_params() -> None:
+    params: dict[str, object] = {
+        "n_factors": 2,
+        "device_type": "cpu",
+        "n_estimators": "100",
+        "learning_rate": 0.05,
+        "num_leaves": 31,
+        "max_depth": 6,
+        "min_child_samples": 20,
+        "feature_fraction": 0.9,
+        "lambda_l2": 1.0,
+        "objective": "regression",
+        "metric": "l2",
+        "verbosity": -1,
+        "n_jobs": -1,
+        "random_state": 7,
+    }
+
+    with pytest.raises(TypeError, match="n_estimators"):
+        make_model_from_params(
+            model_name="lightgbm",
+            params=params,
+            target_dim=2,
+            grid_shape=(1, 2),
+            moneyness_points=(-0.1, 0.0),
+            base_neural_config=NeuralModelConfig(device="cpu"),
+        )
--- /tmp/econ499_round004_extract.ZdatuB/tests/unit/test_stats.py	2026-04-27 18:42:14
+++ /Volumes/T9/ECON499/tests/unit/test_stats.py	2026-04-27 19:59:50
@@ -123,6 +123,34 @@
     assert ranked["model_name"].to_list() == ["good", "bad"]


+def test_alignment_rejects_duplicate_forecast_keys() -> None:
+    duplicate_forecasts = pl.concat([_forecast_frame(), _forecast_frame().head(1)])
+
+    with pytest.raises(ValueError, match="duplicate rows"):
+        build_forecast_realization_panel(
+            actual_surface_frame=_actual_surface_frame(),
+            forecast_frame=duplicate_forecasts,
+            total_variance_floor=1.0e-8,
+        )
+
+
+def test_alignment_rejects_unequal_model_forecast_coverage() -> None:
+    incomplete_forecasts = _forecast_frame().filter(
+        ~(
+            (pl.col("model_name") == "bad")
+            & (pl.col("maturity_index") == 1)
+            & (pl.col("moneyness_index") == 1)
+        )
+    )
+
+    with pytest.raises(ValueError, match="identical coverage"):
+        build_forecast_realization_panel(
+            actual_surface_frame=_actual_surface_frame(),
+            forecast_frame=incomplete_forecasts,
+            total_variance_floor=1.0e-8,
+        )
+
+
 def test_observed_metrics_do_not_fall_back_when_observed_weight_is_empty() -> None:
     actual = _actual_surface_frame().with_columns(
         pl.when(pl.col("quote_date") == date(2021, 1, 5))
@@ -205,6 +233,36 @@
         seed=7,
     )
     assert "bad" not in mcs.superior_models
+
+
+def test_statistical_tests_reject_nonfinite_losses() -> None:
+    with pytest.raises(ValueError, match="finite"):
+        diebold_mariano_test(
+            loss_a=np.asarray([1.0, np.nan]),
+            loss_b=np.asarray([1.0, 1.1]),
+            model_a="benchmark",
+            model_b="candidate",
+        )
+    with pytest.raises(ValueError, match="finite"):
+        superior_predictive_ability_test(
+            benchmark_losses=np.asarray([1.0, 1.1]),
+            candidate_losses=np.asarray([[0.9], [np.inf]]),
+            benchmark_model="benchmark",
+            candidate_models=("candidate",),
+            alpha=0.10,
+            block_size=1,
+            bootstrap_reps=10,
+            seed=7,
+        )
+    with pytest.raises(ValueError, match="finite"):
+        model_confidence_set(
+            losses=np.asarray([[1.0, 0.9], [1.1, np.nan]]),
+            model_names=("benchmark", "candidate"),
+            alpha=0.10,
+            block_size=1,
+            bootstrap_reps=10,
+            seed=7,
+        )


 def test_forecast_origin_guard_rejects_hpo_contaminated_rows() -> None:
--- /tmp/econ499_round004_extract.ZdatuB/tests/unit/test_reporting_helpers.py	2026-04-26 13:03:38
+++ /Volumes/T9/ECON499/tests/unit/test_reporting_helpers.py	2026-04-27 20:00:43
@@ -6,6 +6,7 @@
 import numpy as np
 import orjson
 import polars as pl
+import pytest

 from ivsurf.config import SurfaceGridConfig
 from ivsurf.evaluation.diagnostics import (
@@ -307,3 +308,23 @@
     assert payload["config_snapshots"][0]["content"] == "foo: 1\nbar: true\n"
     assert "package_versions" in payload
     assert "hardware_metadata" in payload
+
+
+def test_write_run_manifest_requires_git_commit_hash(tmp_path: Path) -> None:
+    config_path = tmp_path / "config.yaml"
+    config_path.write_text("foo: 1\n", encoding="utf-8")
+    input_artifact = tmp_path / "input.txt"
+    input_artifact.write_text("input\n", encoding="utf-8")
+    output_artifact = tmp_path / "output.txt"
+    output_artifact.write_text("output\n", encoding="utf-8")
+
+    with pytest.raises(RuntimeError, match="git commit hash"):
+        write_run_manifest(
+            manifests_dir=tmp_path / "manifests",
+            repo_root=tmp_path,
+            script_name="unit_test_stage",
+            started_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
+            config_paths=[config_path],
+            input_artifact_paths=[input_artifact],
+            output_artifact_paths=[output_artifact],
+        )

```
