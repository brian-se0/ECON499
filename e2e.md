IMPLEMENTATION_PLAN_FOR_CODEX_TO_REACH_END_TO_END_READINESS

Goal:
Make the committed repo runnable end to end through the official default path (`make pipeline`) with no hidden manual config creation, no invalid forecast-to-IV conversions, and evaluation outputs that match the final thesis spec.

Rules:
- Do not add fallback discovery for missing configs.
- Keep a single official path.
- Do not add backcompat or legacy branches.
- Preserve SPX-only, 15:45, 2004-01-02 to 2021-04-09 scope.

1) Restore the missing committed data-config contract
Targets:
- Create:
  - configs/data/raw.yaml
  - configs/data/cleaning.yaml
  - configs/data/surface.yaml
  - configs/data/features.yaml
- Update tests that currently expect these files to parse and validate them, not just read them.

Required contents:
- raw.yaml:
  - raw_options_dir
  - bronze_dir
  - silver_dir
  - gold_dir
  - manifests_dir
  - target_symbol: "^SPX"
  - calendar_name: "XNYS"
  - timezone: "America/New_York"
  - decision_time: "15:45"
  - sample_start_date: "2004-01-02"
  - sample_end_date: "2021-04-09"
  - am_settled_roots: ["SPX"]
- cleaning.yaml:
  - explicit values for all cleaning thresholds already modeled in CleaningConfig
- surface.yaml:
  - committed fixed moneyness grid
  - committed fixed maturity grid
  - interpolation_order
  - interpolation_cycles
  - total_variance_floor
  - observed_cell_min_count
- features.yaml:
  - lag_windows
  - include_daily_change
  - include_mask
  - include_liquidity

Files to touch:
- create configs/data/*.yaml
- tests/unit/test_runtime_defaults.py
- tests/integration/test_report_stage_contract.py
- README.md only if it currently claims behavior not reflected by committed configs

Acceptance criteria:
- Fresh checkout contains all default config paths referenced by scripts.
- `python -c "from ivsurf.config import RawDataConfig, CleaningConfig, SurfaceGridConfig, FeatureConfig, load_yaml_config; ..."` can parse all four files.
- No test references a non-existent config path.

2) Add a canonical surface-level MSE metric to the official evaluation stack
Targets:
- src/ivsurf/evaluation/metrics.py
- src/ivsurf/evaluation/loss_panels.py
- src/ivsurf/evaluation/slice_reports.py
- configs/eval/stats_tests.yaml
- configs/eval/report_artifacts.yaml
- scripts/07_run_stats.py
- scripts/09_make_report_artifacts.py
- src/ivsurf/reports/tables.py
- regression fixtures if report outputs change

Implementation:
- Add explicit surface-level MSE metrics for the forecast target representation:
  - observed_mse_total_variance
  - full_mse_total_variance
- Keep the existing IV-change MSE metrics, but do not treat them as the only MSE.
- Ensure daily loss frame writes these fields.
- Ensure slice metric frame writes these fields.
- Ensure report artifacts expose QLIKE and the new surface-level MSE in tables/outputs.
- Change the official statistical loss metric in configs/eval/stats_tests.yaml to a thesis-approved committed metric after MSE exists. The simplest compliant choice is `observed_mse_total_variance`.
- Keep the code path single-metric for DM/SPA/MCS unless there is already explicit need for multi-metric statistical outputs.

Tests to add/update:
- unit test for the new MSE helpers
- unit test for daily loss frame column contract
- unit test for slice metric frame column contract
- integration/regression tests for stage 07 and stage 09 outputs after the new metric is added

Acceptance criteria:
- Daily loss artifacts contain QLIKE and surface-level MSE.
- Slice artifacts contain QLIKE and surface-level MSE.
- Stage 07 can run DM/SPA/MCS using the committed official metric without code edits.
- Report artifacts surface the committed official MSE metric.

3) Make forecast-to-IV conversion safe and explicit
Targets:
- src/ivsurf/evaluation/forecast_store.py
- src/ivsurf/evaluation/alignment.py
- possibly src/ivsurf/evaluation/metrics.py if a reusable helper is needed
- any place in report generation that reconstructs IV from predicted total variance

Implementation:
- Pick one authoritative place to floor predicted total variance before IV conversion.
  Preferred path:
  - keep raw predictions in parquet unchanged
  - floor/clamp only at evaluation-time conversion using a shared helper
- Replace the direct sqrt expression in `build_forecast_realization_panel` with the existing safe conversion helper or an equivalent shared helper.
- Add explicit null/NaN checks for:
  - predicted_total_variance
  - predicted_iv
  - predicted_iv_change
- Decide whether the floor comes from:
  - the committed surface config (`total_variance_floor`), or
  - the committed metrics config (`positive_floor`)
  and use one source consistently.

Tests to add:
- unit test with negative predicted_total_variance values proving predicted_iv stays finite
- integration test for stage 07 with at least one negative prediction from a synthetic forecast artifact
- regression test proving no NaN enters report tables because of negative TV

Acceptance criteria:
- Stage 07 and stage 09 never emit NaN/inf `predicted_iv` caused by negative TV.
- All IV reconstruction uses one shared safe conversion path.

4) Remove dead hedging-config ambiguity
Targets:
- configs/eval/hedging.yaml
- scripts/08_run_hedging_eval.py
- src/ivsurf/hedging/hedge_rules.py
- src/ivsurf/hedging/pnl.py
- tests/unit/test_hedging.py
- tests/integration/test_stats_hedging_slice.py

Implementation:
- Choose one of these and do it cleanly:
  A. Wire `hedge_spot_assumption` all the way through and support the documented value(s), or
  B. Delete `hedge_spot_assumption` from config and rename the behavior in code/comments to an explicit hardcoded `no_change` spot assumption.
- Do not leave a config key that has no effect.

Acceptance criteria:
- Every field in configs/eval/hedging.yaml materially affects runtime behavior or is removed.
- A test fails if the hedging spot assumption is changed incorrectly.

5) Either honor `drop_early_close_days` directly or delete it
Targets:
- src/ivsurf/config.py
- scripts/02_build_option_panel.py
- scripts/03_build_surfaces.py
- tests/unit/test_config_models.py
- tests/integration/test_sample_window_enforcement.py or a new early-close integration test

Implementation:
- Preferred single-path cleanup:
  - if early-close days are always excluded from the 15:45 problem, remove the unused config switch and make exclusion unconditional and explicit
  - record excluded dates in the stage summary
- Alternative:
  - actually branch on `drop_early_close_days` and make the behavior explicit and tested

Acceptance criteria:
- The code and config agree on what happens to early-close sessions.
- No dead config remains.

6) Add one real official-path integration test
Targets:
- add tests/integration/test_official_default_path_pipeline.py
- optionally expand existing smoke tests

Implementation:
- Build a tiny synthetic dataset in temp directories.
- Use the committed default config filenames, not ad hoc inline config names.
- Execute the stage `main()` functions in the same order the Makefile uses:
  - 02_build_option_panel
  - 03_build_surfaces
  - 04_build_features
  - 05_tune_models (at least one light synthetic profile)
  - 06_run_walkforward
  - 07_run_stats
  - 08_run_hedging_eval
  - 09_make_report_artifacts
- The test does not need to exercise stage 01 on real zips if synthetic parquet-based stage setup is simpler, but the committed default config paths must be used.

Acceptance criteria:
- From a clean temp workspace and committed default config names, the official script path succeeds through stage 09 on synthetic data.
- The test asserts that the expected artifact files are created.
- The test asserts that the core report tables contain QLIKE and surface-level MSE.

7) Pre-run gate before touching the full Cboe dataset
Do not start the expensive real-data run until all of this is true:
- committed `configs/data/*.yaml` files exist and parse
- `uv run pytest` passes in the declared Python 3.14 environment
- the new official-path integration test passes
- stage 07 outputs include a canonical surface-level MSE metric
- no NaN/inf forecast-derived IVs appear in stats/report outputs
- hedging config has no dead keys
- early-close handling is explicit and tested