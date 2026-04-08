# ivsurf

Research-grade, leak-free SPX implied-volatility-surface forecasting infrastructure built from raw Cboe 15:45 option data.

## Scope

- Raw daily zip ingestion from `D:\Options Data`
- Explicit schema validation and early SPX filtering
- Leak-free 15:45 option cleaning and derived-field construction
- Daily observed and completed total-variance surfaces
- Daily feature/target dataset with next-decision-session alignment
- Walk-forward split manifest generation
- Mandatory profile-backed Optuna HPO before model training
- Baseline models plus an arbitrage-aware neural surface model
- Statistical forecast comparison, hedging evaluation, and saved-artifact report generation

## Official Workflow

`make` is the official interface for running this repository.

- Use `make <target>` from the repo root.
- The `scripts/*.py` files are internal stage entrypoints invoked by the `Makefile`.
- Direct script invocation is not the documented workflow for this project.
- HPO is a required stage before walk-forward training. The official `pipeline` targets always run stage `05` before stage `06`.

## Requirements

- Python `3.14.3`
- `uv`
- GNU Make

Windows note:
- Install GNU Make through a Windows-friendly package manager such as Scoop, Chocolatey, or Git for Windows.
- The provided `Makefile` is PowerShell-oriented and is intended for this Windows-first repo.
- On Windows with a compatible NVIDIA GPU, the default runtime now uses CUDA-enabled PyTorch for the neural model and LightGBM `device_type="gpu"` for the LightGBM baseline.

## Setup

Install runtime dependencies:

```powershell
make sync
```

Install runtime plus development tooling:

```powershell
make sync-dev
```

Install runtime, development tooling, and MLflow support:

```powershell
make sync-tracking
```

## Profile-Backed HPO And Training

The official workflow is now locked to explicit serialized profiles:

- HPO profiles:
  - `hpo_30_trials`
  - `hpo_100_trials`
- Training profiles:
  - `train_30_epochs`
  - `train_100_epochs`

The standard high-level commands are:

```powershell
make hpo-30
make hpo-100
make train-30
make train-100
make pipeline-30
make pipeline-100
```

`make train-30` and `make train-100` require the matching HPO manifests to already exist. If they do not, stage `06` fails fast instead of silently falling back to static YAML defaults.

## Core Commands

Show the official target list:

```powershell
make help
```

Run the full pipeline with the lighter official profile:

```powershell
make pipeline
```

Or explicitly choose the pipeline profile:

```powershell
make pipeline-30
make pipeline-100
```

Run individual stages:

```powershell
make ingest
make silver
make surfaces
make features
make hpo MODEL=ridge
make hpo-all
make train
make stats
make hedging
make report
```

Run code-quality checks:

```powershell
make check
```

Or individually:

```powershell
make lint
make test
make typecheck
```

## Optional Variables

You can pass Make variables to adapt the workflow without bypassing the official interface.

- `LIMIT=<n>` limits file-based stages `ingest`, `silver`, and `surfaces` for smoke runs.
- `MODEL=<name>` selects the model for `make hpo`.
- `HPO_PROFILE=<name>` selects the HPO profile for `make hpo`, `make hpo-all`, `make train`, `make stats`, `make hedging`, `make report`, and `make pipeline`.
- `TRAIN_PROFILE=<name>` selects the training profile for the same targets.

Examples:

```powershell
make ingest LIMIT=5
make hpo MODEL=neural_surface HPO_PROFILE=hpo_100_trials TRAIN_PROFILE=train_100_epochs
make train HPO_PROFILE=hpo_30_trials TRAIN_PROFILE=train_30_epochs
```

## Progress Bars

Long-running stages emit Rich progress bars so you can see forward progress while the pipeline runs.

This currently covers the stages most likely to take meaningful time:

- `make ingest`
- `make silver`
- `make surfaces`
- `make features`
- `make hpo`
- `make train`
- `make stats`
- `make hedging`
- `make report`

## Timing Conventions

The pipeline treats `15:45 ET` as the decision timestamp and rejects sessions that close before then. Time to maturity is computed to the contract's last tradable session close using explicit root-based settlement conventions:

- `SPX` is treated as AM-settled by default.
- `SPXW` and other roots are treated as PM-settled unless explicitly configured otherwise.

Feature/target pairs are aligned to the next session that actually contains the configured decision time. This means early-close sessions are not silently treated as valid `t+1` targets for the 15:45 forecasting problem.

This is serialized in configuration so the contract-timing assumption is visible, versioned, and testable.
