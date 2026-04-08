# ivsurf

Research-grade, leak-free SPX implied-volatility-surface forecasting infrastructure built from raw Cboe 15:45 option data.

## Scope

- Raw daily zip ingestion from `D:\Options Data`
- Explicit schema validation and early SPX filtering
- Leak-free 15:45 option cleaning and derived-field construction
- Daily observed and completed total-variance surfaces
- Daily feature/target dataset with next-trading-day alignment
- Walk-forward split manifest generation
- Baseline models plus an arbitrage-aware neural surface model

## Quick start

1. Install dependencies with `uv sync`.
2. Ingest raw zips with `uv run python scripts/01_ingest_cboe.py`.
3. Build the silver option panel with `uv run python scripts/02_build_option_panel.py`.
4. Build gold surfaces with `uv run python scripts/03_build_surfaces.py`.
5. Build model-ready daily features with `uv run python scripts/04_build_features.py`.
6. Run walk-forward forecasting with `uv run python scripts/06_run_walkforward.py`.

## Timing conventions

The pipeline treats 15:45 ET as the decision timestamp and rejects sessions that close before then. Time to maturity is computed to the contract's last tradable session close using explicit root-based settlement conventions:

- `SPX` is treated as AM-settled by default.
- `SPXW` and other roots are treated as PM-settled unless explicitly configured otherwise.

This is serialized in configuration so the contract-timing assumption is visible, versioned, and testable.

