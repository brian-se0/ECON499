# IMPLEMENTATION_PLAN

Target environment
- Python 3.14.3 only
- Windows-first
- CUDA available for model training
- Clean modern code only
- No backwards compatibility work
- No fallback paths
- No legacy APIs
- No “optional old path” branches

Core engineering principles
1. The only genuinely large stage is raw option ingestion and daily aggregation.
2. After data is reduced to one daily surface tensor per date, the modeling dataset is small by modern standards.
3. Use CPU for ETL, cleaning, joins, interpolation, statistics, and report generation.
4. Use CUDA only where it pays off: PyTorch training and possibly LightGBM on a sufficiently large dense feature matrix.
5. Strong baselines are first-class pipeline citizens, not afterthoughts.
6. Surface construction is part of the research contribution and must be explicit, tested, and reproducible.

Recommended Python stack

Data / IO / columnar
- polars
- pyarrow
- numpy
- scipy
- exchange_calendars

Modeling
- scikit-learn
- lightgbm
- torch
- optuna

Statistics / evaluation
- scipy
- statsmodels
- arch

Config / CLI / validation
- pydantic
- pydantic-settings
- typer
- orjson

Experiment tracking / reproducibility
- mlflow
- rich

Dev tooling
- uv
- ruff
- mypy
- pytest
- pytest-xdist
- hypothesis

Why this stack
- Polars + Arrow cover fast, memory-safe ETL.
- SciPy provides interpolation and numerical routines without bespoke math code.
- scikit-learn covers ridge, elastic net, PCA/factor models, and random forest.
- LightGBM covers strong tabular baselines efficiently.
- PyTorch gives full control over the arbitrage-aware neural model, mixed precision, and CUDA.
- Optuna is enough for disciplined HPO without framework bloat.
- MLflow gives local experiment tracking without cloud lock-in.

CUDA guidance

Where CUDA helps
- training the flagship PyTorch joint surface model
- possibly LightGBM GPU if the feature matrix is large and dense enough

Where CUDA does NOT help
- zip extraction
- CSV parsing
- Polars scans
- joins and group-bys
- interpolation
- bootstrap tests and forecast-comparison tests
- parquet writing
- most report generation

Do not try to GPU-accelerate ETL. That is wasted effort.

Repo structure

repo/
  pyproject.toml
  uv.lock
  README.md
  AGENTS.md

  configs/
    data/
      raw.yaml
      cleaning.yaml
      surface.yaml
    models/
      ridge.yaml
      elasticnet.yaml
      har_factor.yaml
      lightgbm.yaml
      random_forest.yaml
      neural_surface.yaml
    eval/
      walkforward.yaml
      metrics.yaml
      stats_tests.yaml
      hedging.yaml

  data/
    raw/          # immutable daily zip files
    bronze/       # selected SPX columns converted to parquet
    silver/       # cleaned option-level data
    gold/         # daily grid surfaces, features, targets
    manifests/    # data manifests, split manifests, run manifests

  src/ivsurf/
    __init__.py
    config.py
    schemas.py
    calendar.py

    io/
      ingest_cboe.py
      parquet.py

    qc/
      raw_checks.py
      schema_checks.py
      timing_checks.py

    cleaning/
      option_filters.py
      derived_fields.py

    surfaces/
      grid.py
      aggregation.py
      interpolation.py
      masks.py
      arbitrage_diagnostics.py

    features/
      lagged_surface.py
      liquidity.py
      factors.py
      tabular_dataset.py

    splits/
      walkforward.py
      manifests.py

    models/
      base.py
      no_change.py
      ridge.py
      elasticnet.py
      har_factor.py
      lightgbm_model.py
      random_forest.py
      neural_surface.py
      losses.py
      penalties.py

    training/
      fit_sklearn.py
      fit_lightgbm.py
      fit_torch.py

    evaluation/
      metrics.py
      slice_reports.py
      forecast_store.py

    stats/
      diebold_mariano.py
      spa.py
      mcs.py
      bootstrap.py

    hedging/
      book.py
      revaluation.py
      hedge_rules.py
      pnl.py

    reports/
      tables.py
      figures.py

  scripts/
    01_ingest_cboe.py
    02_build_option_panel.py
    03_build_surfaces.py
    04_build_features.py
    05_tune_models.py
    06_run_walkforward.py
    07_run_stats.py
    08_run_hedging_eval.py
    09_make_report_artifacts.py

  research/
    notebooks/
    notes/

  tests/
    unit/
    integration/
    regression/
    property/

Data pipeline stages

Stage 0: raw data handling
- Treat raw daily zip files as immutable.
- For each zip file, extract the single CSV member temporarily, read only needed columns, filter immediately to underlying_symbol == "^SPX", and write a compact parquet file.
- Do not keep giant extracted CSVs on disk after conversion unless explicitly needed for audit.

Stage 1: bronze conversion
- Select only columns required for the thesis.
- Explicitly set types at read time.
- Normalize dates and categorical/string columns.
- Partition bronze parquet by year/month or year only.
- Use Parquet with ZSTD compression.

Stage 2: silver option-level panel
- Compute:
  - quote mid at 15:45
  - maturity tau
  - log-moneyness
  - option-level total variance
  - quote quality flags
  - observed/invalid flags
- Apply cleaning rules.
- Persist reason codes for dropped rows.
- Keep float64 here.

Stage 3: gold daily surfaces
- Aggregate by date x moneyness_bin x maturity_bin using vega-weighted averages.
- Persist:
  - observed-cell values
  - completed full-grid values
  - observed-cell masks
  - daily coverage statistics
  - arbitrage diagnostics
- Interpolate on CPU with SciPy.
- Keep interpolation deterministic and fully parameterized.

Stage 4: daily feature/target dataset
- Build one daily row/tensor per date.
- Store:
  - model features
  - target tensor
  - observed-cell target mask
  - metadata for slices and diagnostics
- At this stage the data is small enough to load into memory for many experiments.

Memory-safe ETL rules
- Never load the full raw history into memory.
- Use Polars lazy scanning and projection/predicate pushdown.
- No pandas in ETL.
- No row-wise Python UDFs over option rows.
- Reduce dimensionality before any expensive joins.
- Join only on reduced tables and assert join cardinality.
- Use daily or yearly processing loops; they are fine here.
- Keep float64 through cleaning, interpolation, and statistics.
- Cast to float32 only when materializing final model tensors.

Modeling pipeline stages

1. Baseline family implementation
- NoChangeSurface
- RidgeSurface
- ElasticNetSurface
- HARFactorSurface
- LightGBMSurface
- RandomForestSurface

2. Flagship model
- NeuralSurfaceMLP
- input: daily feature vector or lagged surface tensor flattened consistently
- output: full next-day total variance grid
- loss:
  - observed-cell weighted forecast loss
  - optional lower weight on imputed cells
  - calendar monotonicity penalty
  - convexity penalty
  - optional roughness penalty
- training on CUDA with mixed precision

3. Forecast materialization
- Persist every model’s out-of-sample predictions to parquet or Arrow.
- Never recompute predictions inside evaluation code.
- Evaluation reads forecast artifacts only.

Feature design guidance
- Do not feed raw option rows into the models.
- Model on daily aggregated surfaces and daily summary features.
- Keep the feature set intentionally small and interpretable.
- For trees, use compact tabular features or factor summaries, not huge flattened tensors.
- For the neural model, flattened lagged surfaces are acceptable because the daily sample is small.

Hyperparameter optimization
- Use Optuna with blocked time-series inner validation.
- Do not use random CV.
- Tune on a designated pre-OOS tuning segment or on a pre-registered periodic retuning schedule.
- Do not fully nest expensive HPO every single OOS date.
- Search spaces should be small, justified, and logged.

Suggested search spaces
- Ridge: alpha
- Elastic net: alpha, l1_ratio
- HAR/factor: number of factors, lag structure
- LightGBM: learning_rate, num_leaves, max_depth, min_data_in_leaf, feature_fraction, lambda_l2
- Random forest: n_estimators, max_depth, min_samples_leaf
- Neural model: hidden width, depth, dropout, learning rate, weight decay, batch size, penalty weights

GPU training guidance
- Use torch.cuda.amp.autocast
- Prefer bf16 where supported; otherwise fp16 with GradScaler
- Use pin_memory=True and non_blocking=True only for GPU dataloaders
- Use persistent_workers=True and a modest prefetch_factor
- Keep tensors contiguous
- Avoid CPU-GPU ping-pong inside the training step
- Compute penalties vectorized on the GPU
- Do not move statistics or interpolation to the GPU

OOM avoidance
- The raw data risk is in ETL, not in training.
- Write intermediate parquet and daily surface artifacts early.
- For training, store daily tensors as:
  - contiguous torch tensors saved to disk, or
  - numpy arrays / memmaps loaded by date range
- Random forest should use compact factor/tabular features only.
- LightGBM should not receive unnecessarily wide sparse designs.
- Do not duplicate dense arrays in memory for train/validation/test when a view or index split is enough.

Evaluation pipeline stages

Stage A: statistical metrics
- vega-weighted RMSE/MAE on total variance
- vega-weighted RMSE/MAE on IV
- MSE on IV change
- observed-cell and full-grid reporting
- maturity and moneyness slice reports

Stage B: forecast-comparison tests
- DM for pairwise loss differentials
- SPA using block bootstrap
- MCS using block bootstrap
- Implement formulas explicitly and test them.
- Use arch/bootstrap helpers where convenient, but keep core logic visible and auditable.

Stage C: arbitrage diagnostics
- daily count and magnitude of calendar violations
- daily count and magnitude of convexity violations
- comparison of violation profiles across models

Stage D: economic evaluation
- standardized book definition
- next-day revaluation error
- next-day delta-vega hedge error / P&L variance
- baseline vs model comparisons

Testing strategy

Unit tests
- schema parsing
- date handling
- early-close handling
- zero-IV handling
- maturity calculation
- grid assignment
- vega-weighted aggregation correctness
- penalty sign correctness

Property tests
- synthetic arbitrage-free surfaces should not trigger penalties
- known arbitrage violations should trigger the correct penalty direction
- walk-forward splits never overlap improperly
- next-trading-day alignment respects weekends and holidays

Integration tests
- one-week or one-month end-to-end pipeline
- ingest -> clean -> surface -> features -> model -> evaluation

Regression tests
- golden-sample outputs for a fixed small date range
- persisted metrics for smoke models
- no-leakage checks on split manifests and preprocessing artifacts

Experiment tracking / reproducibility
- Use MLflow with local file backend.
- Log:
  - git commit hash
  - config files
  - data manifest hash
  - split manifest hash
  - package versions
  - random seed
  - hardware info
  - metrics
  - model artifacts
  - forecast artifacts
- Save every run’s exact config snapshot.
- Make every figure and table reproducible from saved artifacts only.

Coding style guidance
- Polars:
  - use lazy API in core pipeline
  - prefer expressions over Python control flow
  - never use row-wise apply for heavy logic
- PyArrow:
  - use explicit schemas
  - use partitioned datasets
  - use ZSTD parquet
- PyTorch:
  - vectorize losses and penalties
  - keep tensors on device throughout the step
  - avoid per-cell Python loops
  - separate model definition, loss construction, and trainer
- scikit-learn / LightGBM:
  - keep tabular baselines simple
  - do not bury data preparation inside model classes

What not to do
- no fallback paths
- no pandas-heavy ETL
- no backward compatibility code
- no hidden global state
- no notebooks in the core pipeline
- no silent coercions
- no feature creation that uses future data
- no evaluation code that recomputes training outputs on the fly

Practical execution order
1. Build leak-free ingestion and cleaning
2. Build and test daily surface construction
3. Build no-change, ridge, and elastic net baselines
4. Run first full walk-forward backtest
5. Add LightGBM and HAR/factor model
6. Add neural model
7. Run formal statistical tests
8. Run revaluation / hedging evaluation
9. Freeze report artifacts