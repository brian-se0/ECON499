# ECON499 SPX IV Surface Audit - Round 003I

You are GPT-5.5 Pro acting as the audit brain for this project. Codex is the hands. This is a focused re-verification round for B1-CODE-009 only.

## Prior Pro Decision

In Round 003H you marked B1-CODE-009 as partially fixed. You identified the remaining gap:

- Forecast artifacts carried `surface_config_hash`, but gold surfaces and daily features did not.
- Alignment could verify grid/schema/coordinate/target-surface version, but could not prove forecasts were evaluated against realized gold surfaces built from the same surface-construction config.

You requested these adjustments:

1. Persist `surface_config_hash` into Stage 03 gold parquet rows and `gold_surface_summary.json`.
2. Carry `surface_config_hash` into `daily_features.parquet` from gold.
3. Update `load_actual_surface_frame()` to load gold `surface_config_hash`.
4. Update `build_forecast_realization_panel()` to reject mismatches between forecast, target-day gold, and origin-day gold `surface_config_hash`.
5. Add tests for mismatch rejection, Stage 03/04 propagation, and old forecast artifacts missing `surface_config_hash`.

## Implemented Changes

### Stage 03 gold artifact propagation

`scripts/03_build_surfaces.py`

```python
surface_config_hash = sha256_file(surface_config_path)
```

Gold parquet rows now include:

```python
pl.lit(COMPLETED_SURFACE_SCHEMA_VERSION).alias("target_surface_version"),
pl.lit(surface_config_hash).alias("surface_config_hash"),
```

`gold_surface_summary.json` rows now include:

```python
"target_surface_version": COMPLETED_SURFACE_SCHEMA_VERSION,
"surface_config_hash": surface_config_hash,
```

The Stage 03 resume artifact schema version was bumped from 3 to 4.

### Stage 04 feature propagation from gold

`src/ivsurf/features/lagged_surface.py`

```python
@dataclass(frozen=True, slots=True)
class DailySurfaceArrays:
    quote_dates: list[date]
    completed_surfaces: np.ndarray
    observed_masks: np.ndarray
    observed_surfaces: np.ndarray
    vega_weights: np.ndarray
    decision_timestamps: list[str]
    surface_config_hashes: list[str]
```

```python
surface_config_hashes = _per_date_string_values(
    sorted_frame,
    dates=dates,
    column_name="surface_config_hash",
)
if len(set(surface_config_hashes)) != 1:
    message = "Daily surface frame contains multiple surface_config_hash values."
    raise ValueError(message)
```

`src/ivsurf/features/tabular_dataset.py`

```python
row: dict[str, object] = {
    "quote_date": quote_date,
    "target_date": target_date,
    DECISION_TIMESTAMP_COLUMN: surface_arrays.decision_timestamps[position],
    TARGET_DECISION_TIMESTAMP_COLUMN: surface_arrays.decision_timestamps[position + 1],
    "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
    "surface_grid_hash": grid.grid_hash,
    "surface_config_hash": surface_arrays.surface_config_hashes[position],
    "maturity_coordinate": MATURITY_COORDINATE,
    "moneyness_coordinate": MONEYNESS_COORDINATE,
    "target_surface_version": COMPLETED_SURFACE_SCHEMA_VERSION,
    "target_gap_sessions": _count_intervening_trading_sessions(...),
}
```

`src/ivsurf/features/availability.py` now classifies `surface_config_hash` as `surface_metadata` with availability reference `effective_decision_timestamp`.

The Stage 04 resume artifact schema version was bumped from 3 to 4.

### Stage 06 forecast artifacts now use the feature/gold hash

`scripts/06_run_walkforward.py`

```python
def _single_feature_metadata_value(feature_frame: pl.DataFrame, column_name: str) -> str:
    if column_name not in feature_frame.columns:
        message = f"daily_features.parquet is missing required metadata column {column_name}."
        raise ValueError(message)
    values = feature_frame[column_name].unique().to_list()
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        message = (
            "daily_features.parquet must contain exactly one non-empty "
            f"{column_name}; found {values!r}."
        )
        raise ValueError(message)
    return values[0]
```

Stage 06 now computes the current config-file hash, reads the feature artifact, extracts the feature-carried hash, and rejects mismatches:

```python
current_surface_config_hash = sha256_file(surface_config_path)
...
feature_frame = pl.read_parquet(raw_config.gold_dir / "daily_features.parquet").sort(
    "quote_date"
)
surface_config_hash = _single_feature_metadata_value(feature_frame, "surface_config_hash")
if surface_config_hash != current_surface_config_hash:
    message = (
        "daily_features.parquet surface_config_hash does not match the current surface "
        f"config file hash: {surface_config_hash!r} != {current_surface_config_hash!r}."
    )
    raise ValueError(message)
```

This `surface_config_hash` value is then written into forecast artifacts through `write_forecasts()`, and is included in `training_run_id`.

The Stage 06 resume artifact schema version was bumped from 2 to 3.

### Evaluation alignment checks gold target and origin surface hashes

`src/ivsurf/evaluation/alignment.py`

`load_actual_surface_frame()` now selects:

```python
"target_surface_version",
"surface_config_hash",
```

`build_forecast_realization_panel()` renames target-day gold:

```python
"target_surface_version": "actual_target_surface_version",
"surface_config_hash": "actual_surface_config_hash",
```

and origin-day gold:

```python
"target_surface_version": "origin_target_surface_version",
"surface_config_hash": "origin_surface_config_hash",
```

Required non-null columns now include:

```python
"surface_config_hash",
"actual_surface_config_hash",
"origin_surface_config_hash",
```

Missing required columns now fail clearly:

```python
def _require_non_null_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        message = (
            "Aligned evaluation panel is missing required columns: "
            f"{', '.join(missing_columns)}."
        )
        raise ValueError(message)
```

Equality checks now include:

```python
("surface_config_hash", "actual_surface_config_hash", "target surface config"),
("surface_config_hash", "origin_surface_config_hash", "origin surface config"),
```

The Stage 07 resume artifact schema version was bumped from 2 to 3.

## Tests Added Or Updated

Focused tests now cover your Round 003H requirements:

- `tests/integration/test_stage03_stage04_target_gap_alignment.py`
  - asserts `gold_surface_summary.json` contains the exact `sha256_file(surface_config_path)`;
  - asserts Stage 03 gold parquet rows contain the same `surface_config_hash`;
  - asserts Stage 04 `daily_features.parquet` carries that same hash.

- `tests/unit/test_alignment.py`
  - `test_build_forecast_realization_panel_rejects_surface_config_mismatch`
    - same grid/schema/coordinates/version, different `surface_config_hash`, must fail with target surface config mismatch.
  - `test_build_forecast_realization_panel_rejects_missing_surface_config_hash`
    - old forecast artifacts missing `surface_config_hash` fail with `missing required columns: surface_config_hash`.

- `tests/integration/test_stage05_stage06_clean_evaluation.py`
  - feature fixtures now carry the exact hash of the surface config file;
  - Stage 06 emits forecasts only after verifying feature/gold hash equals current config hash.

- Existing forecast store tests continue to assert persisted `surface_config_hash`, `model_config_hash`, `training_run_id`, split IDs, timestamps, grid metadata, and target surface version.

## Verification Run

Command:

```bash
uv run python -m pytest tests/unit/test_forecast_store.py tests/unit/test_alignment.py tests/unit/test_feature_dataset.py tests/unit/test_stats.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage04_early_close_alignment.py tests/integration/test_neural_imputed_cell_supervision.py tests/integration/test_smoke_pipeline.py tests/integration/test_stage05_stage06_clean_evaluation.py tests/integration/test_stats_hedging_slice.py tests/integration/test_stage07_negative_prediction.py tests/integration/test_report_stage_contract.py tests/regression/test_report_artifacts.py
```

Result:

```text
34 passed in 6.98s
```

Ruff:

```bash
uv run python -m ruff check scripts/03_build_surfaces.py scripts/04_build_features.py scripts/06_run_walkforward.py scripts/07_run_stats.py src/ivsurf/evaluation/alignment.py src/ivsurf/features/availability.py src/ivsurf/features/lagged_surface.py src/ivsurf/features/tabular_dataset.py tests/unit/test_alignment.py tests/unit/test_feature_dataset.py tests/unit/test_stats.py tests/integration/test_stage03_stage04_target_gap_alignment.py tests/integration/test_stage04_early_close_alignment.py tests/integration/test_neural_imputed_cell_supervision.py tests/integration/test_smoke_pipeline.py tests/integration/test_stage05_stage06_clean_evaluation.py tests/integration/test_stats_hedging_slice.py tests/integration/test_stage07_negative_prediction.py tests/integration/test_report_stage_contract.py tests/regression/test_report_artifacts.py
```

Result:

```text
All checks passed!
```

## Pro Task

Please re-audit B1-CODE-009 only. Decide whether the Round 003H remaining gap is now fully closed.

Return exactly this structure:

```text
VERIFICATION_DECISION:
B1-CODE-009: fixed | still_partial | not_fixed

REASONING:
- ...

REMAINING_REQUIRED_FIXES:
- If fixed: "None."
- If still_partial or not_fixed: list exact code/test changes Codex must make.
```
