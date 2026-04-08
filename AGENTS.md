# AGENTS.md

## Mission

Build a research-grade, leak-free, memory-safe SPX implied-volatility-surface forecasting pipeline from raw Cboe 15:45 option data.

The codebase must prioritize:
- correctness,
- temporal integrity,
- reproducibility,
- maintainability,
- explicit performance awareness.

This project is not a demo. It is research infrastructure.

## Non-negotiable rules

1. Python
- Target Python 3.14.3 only.
- Do not write compatibility code for older Python versions.
- Do not add legacy branches.
- Do not add fallback paths.

2. Research integrity
- No hidden leakage. Ever.
- Every feature must have an availability timestamp less than or equal to the prediction timestamp.
- Same-day EOD fields are forbidden in the 15:45 forecasting problem.
- All preprocessing objects are fit inside the training window only.
- Split manifests must be explicit, serialized, and versioned.
- Observed-cell masks must be preserved and evaluated separately from completed-grid values.
- Do not cherry-pick horizons, slices, or subperiods after seeing results.

3. Data handling
- Raw data is immutable.
- No silent type coercions.
- No silent date coercions.
- No silent row drops.
- Every filter and drop rule must be explicit, logged, and testable.
- Invalid rows must be rejected with reason codes or flagged explicitly.
- Every join must state and assert expected key cardinality.
- Never load full raw history into memory unless the code proves it is safe and necessary.

4. Code quality
- No hand-wavy TODOs in core paths.
- No `pass` in core implementations.
- No placeholder functions that silently return partial results.
- No broad `except Exception` without re-raising a typed error.
- No dead abstractions.
- No giant inheritance hierarchies.
- Prefer small, composable functions and clear data contracts.

5. Performance
- Be explicit about algorithmic cost and memory cost.
- Use Polars lazy queries and Arrow datasets in the core pipeline.
- Do not use pandas in ETL or core modeling.
- Do not use row-wise Python loops over option rows.
- Use GPU only where it is justified.
- Do not attempt GPU ETL, joins, interpolation, or statistical tests.

## Required architecture

- Reusable library code lives in `src/ivsurf`.
- Pipeline orchestration lives in `scripts`.
- Ad hoc exploration lives in `research/`.
- Notebooks are never a dependency of the core pipeline.
- Research-only experiments must not pollute reusable library modules.
- Reusable library modules must not contain one-off thesis hacks.

## Required modeling stance

- The project must always include:
  - a no-change surface benchmark,
  - ridge,
  - elastic net,
  - at least one classical factor/HAR-style benchmark,
  - the flagship arbitrage-aware neural model.
- The neural model predicts total variance, not raw IV.
- The neural model is described as arbitrage-aware, not arbitrage-free, unless hard constraints truly guarantee arbitrage-freeness by construction.
- Hyperparameter tuning must use blocked time-series validation only.
- Never use random cross-validation.

## Required data rules

- Use explicit schemas at ingestion.
- Process raw files incrementally by day or partition.
- Project only needed columns as early as possible.
- Filter to `^SPX` as early as possible.
- Keep float64 for cleaning, interpolation, and statistical evaluation.
- Cast to float32 only when materializing final tensors for model training.
- Persist:
  - cleaned option-level data,
  - daily observed surfaces,
  - daily completed surfaces,
  - daily masks,
  - daily feature sets,
  - split manifests,
  - model forecasts.

## Required testing

Tests are mandatory.

At minimum, include:
- schema tests,
- calendar alignment tests,
- early-close tests,
- zero-IV handling tests,
- next-trading-day alignment tests,
- walk-forward split integrity tests,
- aggregation correctness tests,
- interpolation sanity tests,
- arbitrage-penalty direction tests,
- end-to-end smoke tests.

Property-based tests are required for critical invariants.

Any change that affects:
- timing,
- feature construction,
- surface construction,
- loss definitions,
- evaluation metrics,
must come with tests.

## Required reproducibility

Every run must record:
- git commit hash,
- configuration snapshot,
- data manifest hash,
- split manifest hash,
- package versions,
- random seed,
- hardware metadata.

No hidden randomness.
No mutable global configuration.
No results that cannot be regenerated from saved artifacts.

## Required logging and failure behavior

- Fail fast on schema drift.
- Fail fast on impossible joins.
- Fail fast on unexpected missing critical columns.
- Fail fast on malformed timestamps.
- Do not silently clip, coerce, or infer core values.
- If a numerical floor is required, such as for QLIKE positivity, it must be:
  - explicit in config,
  - logged,
  - tested.

## Required style

- Use modern type hints on all public functions.
- Keep mypy clean enough to be useful.
- Keep ruff clean.
- Use Pydantic for validated config where appropriate.
- Prefer explicit dataclasses or typed configs over unstructured dicts.
- Prefer descriptive names over short cryptic names.
- Keep functions short and single-purpose.
- Keep modules cohesive.

## Forbidden patterns

- fallback code paths
- backwards compatibility code
- legacy APIs
- pandas ETL
- row-wise Python `apply`
- hidden in-place mutation of shared state
- notebook-only business logic
- silent data coercion
- future-aware feature generation
- ad hoc data cleaning inside model classes
- evaluation code that retrains models implicitly
- core-path TODO comments without implementation

## Performance rules

- Polars lazy API first.
- Arrow datasets for storage and scanning.
- Materialize only when necessary.
- Prefer vectorized expressions.
- Pre-aggregate before joining whenever possible.
- Avoid duplicate dense arrays in memory.
- For GPU training:
  - keep data transfer minimal,
  - use pinned memory,
  - use non-blocking transfer,
  - use mixed precision where appropriate,
  - avoid CPU-GPU ping-pong inside the training step.
- Do not optimize the wrong stage:
  - raw ETL is the main memory bottleneck,
  - training is not.

## Behavior when unsure

When unsure:
1. Do not guess silently.
2. Inspect the schema, upstream assumptions, and timing constraints.
3. State the uncertainty explicitly in code comments or docs.
4. Choose the simplest correct leak-free implementation.
5. Add a test that locks the assumption down.
6. If a shortcut would weaken research integrity, reject it.

If a requested change conflicts with:
- no leakage,
- reproducibility,
- explicit data semantics,
- memory safety,
the correct response is to refuse the shortcut in code and implement the safer path.

## Definition of done

Code is done only when:
- it is typed,
- it is tested,
- it is reproducible,
- it is memory-safe,
- it respects the causal timeline,
- it does not rely on silent coercions,
- it does not hide uncertainty,
- it produces auditable artifacts.