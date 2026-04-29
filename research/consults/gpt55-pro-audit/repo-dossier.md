# Repository Dossier For GPT 5.5 Pro Audit

Generated: 2026-04-27
Workspace: `/Volumes/T9/ECON499`
Remote: `https://github.com/brian-se0/ECON499.git`
Commit: `96e0da6fe741b63d2d4743aaab1ea6b5dde4c468`
Working tree note: this audit scaffold is newly added and not part of the baseline commit.

## Project Purpose

Research-grade SPX implied-volatility-surface forecasting infrastructure built from raw Cboe 15:45 option data.

Official sample window: `2004-01-02` through `2021-04-09`.
Target universe: rows with `underlying_symbol == "^SPX"`, including both SPX and SPXW roots when the underlying remains `^SPX`.
Decision time: `15:45:00` America/New_York, 15 minutes before regular close.

## Baseline And Artifacts

- Live repo is the audit source of truth.
- GitHub remote is reported by the user as up to date.
- Frozen archive retained as baseline artifact: `/Volumes/T9/ECON499_audit_hpo_30_trials__train_30_epochs__mac_cpu_20260426_under512MB.zip`
- Frozen archive SHA256: `0f3612fa75a35c3ca7ac0487537187e0409918cd4dd3a7b37a2fbe6d8f7eb096`
- Local directory sizes observed at initialization:
  - `lit_review`: 248 MB
  - `data`: 3.0 GB
  - `provenance`: 5.6 MB

## Runtime And Tooling

- Python requirement: exactly `3.13.5`
- Package manager: `uv`
- Official command surface: `make`
- `pyproject.toml` requires Python `==3.13.5`
- Dev checks: ruff, pytest, mypy
- Core dependencies include Polars, PyArrow, Pydantic, Optuna, scikit-learn, LightGBM, Torch, scipy/statsmodels, Hypothesis.

## Official Workflow

Make targets define the project workflow:

- `make check`: ruff, pytest, mypy
- `make check-runtime`: runtime contract check
- `make official-smoke`: stage01-stage09 smoke bundle
- `make ingest`: raw Cboe zip ingestion
- `make silver`: option-level panel
- `make surfaces`: observed/completed surfaces and masks
- `make features`: feature/target dataset
- `make hpo-all`: required Optuna tuning
- `make train`: walk-forward training and forecasts
- `make stats`: statistical tests
- `make hedging`: hedging evaluation
- `make report`: report artifact generation
- `make pipeline`, `make pipeline-30`, `make pipeline-100`: full official runs

The default Makefile shell is PowerShell-oriented. The README documents a refreshed `mac_cpu` profile that invokes scripts directly with Mac-specific configs for stages 06 through 09.

## Key Configs

Raw Mac config:

```yaml
raw_options_dir: "/Volumes/T9/Options Data"
bronze_dir: "data/bronze"
silver_dir: "data/silver"
gold_dir: "data/gold"
manifests_dir: "data/manifests"
raw_file_glob: "UnderlyingOptionsEODCalcs_*.zip"
target_symbol: "^SPX"
calendar_name: "XNYS"
timezone: "America/New_York"
decision_time: "15:45:00"
decision_snapshot_minutes_before_close: 15
sample_start_date: "2004-01-02"
sample_end_date: "2021-04-09"
am_settled_roots:
  - "SPX"
```

Surface grid config:

```yaml
moneyness_points: [-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30]
maturity_days: [1, 7, 14, 30, 60, 90, 180, 365, 730]
interpolation_order: ["maturity", "moneyness"]
interpolation_cycles: 2
total_variance_floor: 1.0e-8
observed_cell_min_count: 1
```

Feature config:

```yaml
lag_windows: [1, 5, 22]
include_daily_change: true
include_mask: true
include_liquidity: true
```

Walk-forward config:

```yaml
train_size: 504
validation_size: 126
test_size: 21
step_size: 21
expanding_train: true
```

Mac CPU neural config:

```yaml
model_name: "neural_surface"
hidden_width: 256
depth: 3
dropout: 0.10
learning_rate: 0.001
weight_decay: 0.0001
epochs: 80
batch_size: 64
seed: 7
observed_loss_weight: 1.0
imputed_loss_weight: 0.25
calendar_penalty_weight: 0.05
convexity_penalty_weight: 0.05
roughness_penalty_weight: 0.005
output_total_variance_floor: 1.0e-8
device: "cpu"
```

HPO and training profiles:

```yaml
hpo_30_trials:
  n_trials: 30
  tuning_splits_count: 3
  seed: 7
  sampler: "tpe"
  pruner: median

train_30_epochs:
  epochs: 30
  neural_early_stopping_patience: 5
  neural_min_epochs_before_early_stop: 10
  lightgbm_early_stopping_rounds: 25
```

## Source Layout

Reusable library code lives in `src/ivsurf`.

Important modules:

- `io/`: ingestion, Parquet/Arrow IO, atomic writes, paths
- `qc/`: raw checks, schema checks, timing checks, sample-window enforcement
- `cleaning/`: option filters and derived fields
- `surfaces/`: grid construction, aggregation, interpolation, masks, arbitrage diagnostics
- `features/`: lagged surfaces, liquidity, factor features, tabular datasets
- `splits/`: walk-forward split generation and manifests
- `models/`: no-change/naive, ridge, elastic net, HAR factor, random forest, LightGBM, neural surface, losses, penalties
- `training/`: sklearn, LightGBM, Torch training, tuning, model factory
- `evaluation/`: alignment, metrics, loss panels, forecast store, diagnostics, slice reports
- `stats/`: Diebold-Mariano, SPA, MCS, bootstrap
- `hedging/`: book, hedge rules, PnL, revaluation
- `reports/`: tables and figures
- `reproducibility.py`: run metadata and provenance capture
- `workflow.py`: workflow helpers

Pipeline orchestration lives in `scripts`:

- `01_ingest_cboe.py`
- `02_build_option_panel.py`
- `03_build_surfaces.py`
- `04_build_features.py`
- `05_tune_models.py`
- `06_run_walkforward.py`
- `07_run_stats.py`
- `08_run_hedging_eval.py`
- `09_make_report_artifacts.py`
- `official_smoke.py`
- `check_runtime.py`
- `clean_pipeline_artifacts.py`

## Test Inventory

Observed test files:

- Unit tests: 29 files
- Integration tests: 12 files
- Property tests: 2 files
- Regression tests: 2 files
- E2E tests: 1 file

Explicitly relevant tests include:

- `tests/integration/test_sample_window_enforcement.py`
- `tests/integration/test_stage04_early_close_alignment.py`
- `tests/integration/test_stage03_stage04_target_gap_alignment.py`
- `tests/integration/test_stage05_stage06_clean_evaluation.py`
- `tests/integration/test_neural_imputed_cell_supervision.py`
- `tests/regression/test_observed_mask_preservation.py`
- `tests/property/test_walkforward.py`
- `tests/property/test_arbitrage_penalties.py`
- `tests/unit/test_schema.py`
- `tests/unit/test_calendar.py`
- `tests/unit/test_option_filters.py`
- `tests/unit/test_interpolation.py`
- `tests/unit/test_aggregation.py`
- `tests/unit/test_losses.py`
- `tests/unit/test_metrics.py`
- `tests/unit/test_arbitrage_diagnostics.py`

## Audit Areas For Pro

Pro should prioritize concrete findings in:

- hidden leakage and feature availability timestamps
- same-day EOD field exclusion for 15:45 forecasting
- early-close and next-trading-day alignment
- raw schema validation and explicit row rejection
- join cardinality assertions
- memory safety in raw ETL and Arrow/Polars usage
- observed-cell masks versus completed-grid values
- interpolation sanity and total-variance handling
- HPO split integrity and blocked validation
- neural model target/loss/penalty correctness
- evaluation metrics and statistical test assumptions
- reproducibility metadata completeness
- consistency between README/Makefile/profile behavior and executable code
