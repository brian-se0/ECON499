# ivsurf

Research-grade, leak-free SPX implied-volatility-surface forecasting infrastructure built from raw Cboe 15:45 option data.

## Scope

- Raw daily zip ingestion from `D:\Options Data`
- Input contract is the calcs-included Cboe Option EOD Summary daily zip format
- Explicit schema validation and early SPX-underlying filtering
- Leak-free 15:45 option cleaning and derived-field construction
- Daily observed and completed total-variance surfaces
- Daily feature/target dataset with next observed gold-surface alignment
- Walk-forward split manifest generation
- Mandatory profile-backed Optuna HPO before model training
- Baseline models plus an arbitrage-aware neural surface model
- Statistical forecast comparison, hedging evaluation, and saved-artifact report generation

The official thesis sample window is `2004-01-02` through `2021-04-09`. This window is enforced in executable config and is not inferred from whatever files happen to be present in `data/`.

The thesis universe is the SPX underlying universe defined by `underlying_symbol == "^SPX"`. That includes both `SPX` and `SPXW` roots whenever they are written on the SPX underlying.

## Hardware Run Profiles

The project has exactly two supported full-run hardware profiles:

- `mac_cpu`: macOS on a MacBook-class machine with CPU-only neural training and a
  no-OpenMP LightGBM build.
- `windows_cuda`: Windows with CUDA-enabled PyTorch for the neural model and
  LightGBM GPU mode.

Running either profile end to end is a valid reproduction route. Running both is
not required.

Stages `06` through `09` accept an optional `--run-profile-name` argument. When supplied, it becomes part of the forecast, stats, hedging, and report artifact label:

- base label: `hpo_30_trials__train_30_epochs`
- profiled label: `hpo_30_trials__train_30_epochs__mac_cpu`

The repository includes explicit model hardware profiles:

- Windows/CUDA:
  - `configs/data/raw.windows.yaml`
  - `configs/models/neural_surface.windows_cuda.yaml`
  - `configs/models/lightgbm.windows_cuda.yaml`
- Mac/non-CUDA:
  - `configs/data/raw.mac.yaml`
  - `configs/models/neural_surface.mac_cpu.yaml`
  - `configs/models/lightgbm.mac_cpu.yaml`

The default Windows-first configs remain CUDA-oriented. The Mac CPU refresh run used:

```bash
uv sync --extra dev
uv run python scripts/install_mac_lightgbm_no_openmp.py
uv run python scripts/check_runtime.py \
  --raw-config-path configs/data/raw.mac.yaml \
  --run-profile-name mac_cpu \
  --neural-config-path configs/models/neural_surface.mac_cpu.yaml \
  --lightgbm-config-path configs/models/lightgbm.mac_cpu.yaml

uv run python scripts/06_run_walkforward.py \
  --raw-config-path configs/data/raw.mac.yaml \
  --run-profile-name mac_cpu \
  --neural-config-path configs/models/neural_surface.mac_cpu.yaml \
  --lightgbm-config-path configs/models/lightgbm.mac_cpu.yaml
uv run python scripts/07_run_stats.py --raw-config-path configs/data/raw.mac.yaml --run-profile-name mac_cpu
uv run python scripts/08_run_hedging_eval.py --raw-config-path configs/data/raw.mac.yaml --run-profile-name mac_cpu
uv run python scripts/09_make_report_artifacts.py --raw-config-path configs/data/raw.mac.yaml --run-profile-name mac_cpu
```

The Mac CPU profile records any carried-forward forecast artifacts in
`data/manifests/forecast_profile_reuse/mac_cpu.json`. In the refreshed
2026-04-26 canonical Mac run, all seven model forecasts were regenerated
locally and `reused_models` is empty.

The refreshed Mac run also writes a checksum supplement at
`provenance/hpo_30_trials__train_30_epochs__mac_cpu.json`, covering raw zips,
bronze/silver/gold daily artifacts, forecasts, stats, hedging outputs, report
artifacts, tuning manifests, and run manifests.
The reviewer-readable neural tuning diagnostics behind the paper medians are
exported under `provenance/tuning_diagnostics/`.

The Mac CPU profile intentionally requires LightGBM to be rebuilt from the
official source package with `USE_OPENMP=OFF`. The stock macOS wheel can load a
different OpenMP runtime than scikit-learn in the same Python process, which is
not acceptable for the canonical Mac profile. `scripts/check_runtime.py` rejects
that unsafe install before a run begins.

## Official Workflow

`make` is the official interface for running this repository.

- Use `make <target>` from the repo root.
- The `scripts/*.py` files are internal stage entrypoints invoked by the `Makefile`.
- Direct script invocation is not the documented workflow for this project.
- HPO is a required stage before walk-forward training. The official `pipeline` targets always run stage `05` before stage `06`.
- Use `make sync-dev` for the official `pipeline` targets because `make pipeline`, `make pipeline-30`, and `make pipeline-100` all run `make check`, which requires the dev toolchain.
- Before the first expensive raw-data run, the required operator flow is `make check-runtime`, then `make official-smoke`, then `make pipeline`.
- The official `pipeline` target runs `make check` and then `make check-runtime` before any raw-data work starts, so code-quality and Windows/GPU/CUDA contract failures surface before the raw-data stages.

## Requirements

- Python `3.13.5`
- `uv`
- GNU Make

Windows note:
- Install GNU Make through a Windows-friendly package manager such as Scoop, Chocolatey, or Git for Windows.
- The provided `Makefile` is PowerShell-oriented and is intended for this Windows-first repo.
- On Windows with a compatible NVIDIA GPU, the default runtime now uses CUDA-enabled PyTorch for the neural model and LightGBM `device_type="gpu"` for the LightGBM baseline.
- The official LightGBM HPO/training path assumes GPU mode rather than a CPU fallback.

## Setup

Install runtime dependencies:

```powershell
make sync
```

Install runtime plus development tooling:

```powershell
make sync-dev
```

Use `make sync-dev` for the official end-to-end workflow. The `pipeline` targets invoke `make check`, which runs `ruff`, `pytest`, and `mypy`.

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
make check-runtime
make official-smoke
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

Validate the official runtime before the first raw-data run:

```powershell
make check-runtime
make official-smoke
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

Clean derived artifacts safely:

```powershell
make clean
make clean-train HPO_PROFILE=hpo_30_trials TRAIN_PROFILE=train_30_epochs
```

Or individually:

```powershell
make lint
make test
make typecheck
```

The cleanup workflow never deletes or mutates the protected raw options directory at `D:\Options Data`. Cleanup targets only operate on configured derived-artifact roots such as `data/bronze`, `data/silver`, `data/gold`, and `data/manifests`.

## Optional Variables

You can pass Make variables to adapt the workflow without bypassing the official interface.

- `LIMIT=<n>` limits file-based stages `ingest`, `silver`, and `surfaces` for smoke runs.
- `MODEL=<name>` selects the model for `make hpo`.
- `HPO_PROFILE=<name>` selects the HPO profile for `make hpo`, `make hpo-all`, `make train`, `make stats`, `make hedging`, `make report`, and `make pipeline`.
- `TRAIN_PROFILE=<name>` selects the training profile for the same targets.
- The same `HPO_PROFILE` and `TRAIN_PROFILE` variables also scope the profile-aware cleanup targets for stages `05` to `09`.

Examples:

```powershell
make ingest LIMIT=5
make hpo MODEL=neural_surface HPO_PROFILE=hpo_100_trials TRAIN_PROFILE=train_100_epochs
make train HPO_PROFILE=hpo_30_trials TRAIN_PROFILE=train_30_epochs
make clean-stats HPO_PROFILE=hpo_30_trials TRAIN_PROFILE=train_30_epochs
```

## Cleanup Behavior

Each `make clean-<stage>` target removes that stage's derived artifacts plus every downstream stage. This avoids stale downstream outputs surviving after an upstream rebuild.

Available cleanup targets:

- `make clean-ingest`
- `make clean-silver`
- `make clean-surfaces`
- `make clean-features`
- `make clean-hpo`
- `make clean-train`
- `make clean-stats`
- `make clean-hedging`
- `make clean-report`
- `make clean`

`make clean` is the aggregate cleanup command and is equivalent in effect to cleaning every stage's derived artifacts. It is intended for a full rerun from derived-artifact scratch, while still preserving the immutable raw dataset at `D:\Options Data`.

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

## Resume Behavior

The official pipeline now supports context-aware crash recovery.

- Each stage records a resume ledger under `data/manifests/resume/`.
- Resume is only honored when the stage is rerun with the same executable context:
  config snapshots plus the exact upstream artifact set for that stage.
- Completed items are skipped.
- Missing or interrupted items are discarded and rerun.
- Writes are atomic, so a crash should not leave a half-written artifact masquerading as complete.

Item granularity is stage-specific:

- stages `01` to `03`: one daily file
- stage `04`: the feature/split bundle
- stage `05`: one tuned model
- stage `06`: one forecasted model
- stage `07`: one stats artifact
- stage `08`: one hedging model
- stage `09`: one report bundle

## Timing Conventions

The pipeline treats the vendor `"1545"` columns as the decision snapshot on every trading session. On regular sessions the effective decision time is `15:45 ET`. On early-close sessions the effective decision time is the vendor snapshot taken `15` minutes before the scheduled close, which is `12:45 ET` on the documented `13:00 ET` early-close schedule. Time to maturity is computed from that effective snapshot time to the contract's last tradable session close using explicit root-based settlement conventions:

- `SPX` is treated as AM-settled by default.
- `SPXW` and other roots are treated as PM-settled unless explicitly configured otherwise.
- Stage `08` hedging uses `active_underlying_price_1545` as the single official daily spot source. The vendor `underlying_bid_1545` and `underlying_ask_1545` fields are not used for SPX/index hedging because they may legitimately be zero.

Feature/target pairs are aligned to the next observed trading session. This means pre-early-close features target the early-close session itself rather than skipping to the next regular close.

When stage `03` skips a trading session because no valid rows survive cleaning, stage `04` targets the next observed gold-surface date and records `target_gap_sessions` so skipped sessions remain explicit in downstream diagnostics.

This is serialized in configuration and persisted artifacts so the contract-timing assumption is visible, versioned, and testable.
