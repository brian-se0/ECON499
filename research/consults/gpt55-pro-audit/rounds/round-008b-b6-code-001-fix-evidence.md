# Round 008B B6-CODE-001 Fix Evidence

Date: 2026-04-28

## Finding

`B6-CODE-001`: Official-smoke 2x2 surface grid can produce non-finite neural roughness penalty instead of failing fast.

Round 008 Pro decision: `ROUND_008_AUDIT_DECISION: findings_found`

## Patch

- `src/ivsurf/models/penalties.py`
  - Added explicit grid-shape and finite-input validation before penalty computation.
  - `calendar_monotonicity_penalty` now requires at least 2 maturity points.
  - `convexity_penalty` now requires at least 3 moneyness points instead of returning a silent zero penalty.
  - `roughness_penalty` now requires at least 3 maturity points and 3 moneyness points before computing second differences.
  - Penalty outputs must be finite scalar tensors.
- `src/ivsurf/models/neural_surface.py`
  - Added neural penalty grid-domain validation at `NeuralSurfaceRegressor` construction.
  - Enabled penalties are computed only when their configured weight is positive.
  - Disabled penalties are not evaluated, so explicitly disabled small-grid test fixtures remain valid.
  - The aggregate neural training loss is rejected if it becomes non-finite after penalty terms.
- `configs/official_smoke/data/surface.yaml`
  - Hard-cut official smoke surface grid from 2x2 to 3x3:
    - moneyness: `[-0.10, 0.00, 0.10]`
    - maturities: `[7, 30, 60]`
- `scripts/official_smoke.py`
  - Synthetic raw rows now include the 3x3 smoke grid domain.
- `tests/integration/test_synthetic_cpu_stage01_to_stage09_pipeline.py`
  - Synthetic end-to-end test fixture now mirrors the 3x3 valid grid.
- `tests/unit/test_penalties.py`
  - Added failing tests for invalid calendar, convexity, and roughness penalty domains.
  - Added finite positive test for the minimum 3x3 roughness grid.
- `tests/unit/test_training_behaviour.py`
  - Added fail-fast neural regressor test for enabled roughness penalty on a 2x2 grid.
  - Added non-finite aggregate neural loss rejection test.
  - Explicitly disabled penalties in small-grid neural training fixtures.
- `tests/unit/test_repository_contract.py`
  - Updated official-smoke grid-size contract to 3x3.
  - Added contract that the shipped official-smoke grid is valid for enabled neural penalties and produces finite penalty components.

## Verification

Command:

```text
uv run python -m ruff check src/ivsurf/models/penalties.py src/ivsurf/models/neural_surface.py scripts/official_smoke.py tests/unit/test_penalties.py tests/unit/test_training_behaviour.py tests/unit/test_repository_contract.py tests/e2e/test_official_smoke_runtime.py tests/integration/test_synthetic_cpu_stage01_to_stage09_pipeline.py
```

Result:

```text
All checks passed!
```

Command:

```text
uv run python -m pytest tests/unit/test_penalties.py tests/unit/test_training_behaviour.py::test_neural_training_uses_validation_early_stopping tests/unit/test_training_behaviour.py::test_neural_prediction_reuses_train_window_standardization_and_stays_positive tests/unit/test_training_behaviour.py::test_neural_regressor_rejects_enabled_roughness_penalty_on_too_small_grid tests/unit/test_training_behaviour.py::test_neural_training_rejects_nonfinite_penalty_loss tests/unit/test_repository_contract.py tests/e2e/test_official_smoke_runtime.py
```

Result:

```text
14 passed, 1 skipped in 4.71s
```

Expected skip:

```text
Official runtime smoke is unavailable in this environment: The Windows/CUDA runtime profile requires Windows. Detected platform.system()='Darwin'.
```

Command:

```text
uv run python -m pytest tests/property/test_arbitrage_penalties.py tests/integration/test_neural_imputed_cell_supervision.py tests/integration/test_synthetic_cpu_stage01_to_stage09_pipeline.py
```

Result:

```text
3 passed, 1 skipped in 3.11s
```

Expected skip:

```text
Official GPU runtime is unavailable in this environment: The Windows/CUDA runtime profile requires Windows. Detected platform.system()='Darwin'.
```

Command:

```text
git diff --check -- src/ivsurf/models/penalties.py src/ivsurf/models/neural_surface.py configs/official_smoke/data/surface.yaml scripts/official_smoke.py tests/unit/test_penalties.py tests/unit/test_training_behaviour.py tests/unit/test_repository_contract.py tests/e2e/test_official_smoke_runtime.py tests/integration/test_synthetic_cpu_stage01_to_stage09_pipeline.py research/consults/gpt55-pro-audit/backlog.md research/consults/gpt55-pro-audit/rounds/round-008-fresh-full-project-closure-audit-response.md
```

Result: no output.

## Runtime Limitation

The full `tests/unit/test_training_behaviour.py` file was not used as final evidence on this macOS runtime because its LightGBM tests triggered an unrelated native LightGBM segmentation fault. The targeted neural tests in that file passed when run explicitly. The real Windows/CUDA official smoke run was not executed locally because this machine is macOS; the official smoke and synthetic pipeline tests skipped at the runtime preflight boundary as designed.
