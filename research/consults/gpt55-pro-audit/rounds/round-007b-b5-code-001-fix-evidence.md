# Round 007B B5-CODE-001 Fix Evidence

Date: 2026-04-28

## Finding

`B5-CODE-001`: Official smoke did not pass the smoke surface-grid config into Stages 07 and 08, allowing those stages to use their committed default `configs/data/surface.yaml` rather than the official smoke 2x2 grid.

Round 007 Pro decision: `ROUND_007_AUDIT_DECISION: findings_found`

## Patch

- `scripts/official_smoke.py`
  - `stage07.main(...)` now receives `surface_config_path=smoke_surface_config_path`.
  - `stage08.main(...)` now receives `surface_config_path=smoke_surface_config_path`.
- `tests/e2e/test_official_smoke_runtime.py`
  - Added a deterministic non-skipping contract test that monkeypatches the runtime preflight and stage modules.
  - The test records every official-smoke stage invocation and asserts Stages 03, 04, 05, 06, 07, 08, and 09 all receive the official smoke surface config.
  - The test also asserts Stages 01 and 02 do not receive a surface config.
  - The Stage 09 spy writes the minimal required report artifacts so the official smoke entrypoint completes without relying on Windows/CUDA availability.

## Verification

Command:

```text
uv run python -m ruff check scripts/official_smoke.py tests/e2e/test_official_smoke_runtime.py
```

Result:

```text
All checks passed!
```

Command:

```text
uv run python -m pytest tests/e2e/test_official_smoke_runtime.py tests/unit/test_make_workflow.py
```

Result:

```text
4 passed, 1 skipped in 9.93s
```

Expected skip:

```text
Official runtime smoke is unavailable in this environment: The Windows/CUDA runtime profile requires Windows. Detected platform.system()='Darwin'.
```

Command:

```text
uv run python -m pytest tests/integration/test_stats_hedging_slice.py tests/integration/test_report_stage_contract.py tests/integration/test_stage07_negative_prediction.py tests/unit/test_hedging.py tests/regression/test_report_artifacts.py
```

Result:

```text
21 passed in 2.43s
```

Command:

```text
git diff --check -- scripts/official_smoke.py tests/e2e/test_official_smoke_runtime.py
```

Result: no output.

## Runtime Boundary

The real official Windows/CUDA smoke run was not executed in this local macOS runtime. The new argument-propagation test is intentionally independent of that runtime preflight so the exact B5 contract is enforced deterministically here.
