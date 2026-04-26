# Results Dossier: Profiled Runs

## Current canonical profile: `hpo_30_trials__train_30_epochs__mac_cpu`

This addendum records the refreshed Mac CPU profile created on 2026-04-26 after
repairing the nonuniform-grid butterfly-convexity diagnostics and neural penalty.
The proprietary raw zip source was read from `/Volumes/T9/Options Data`; raw files
were not modified.

Canonical refreshed artifacts:
- Report overview: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs__mac_cpu/index.md`
- Report tables: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs__mac_cpu/tables/`
- Report detailed frames: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs__mac_cpu/details/`
- Forecast files: `data/gold/forecasts/hpo_30_trials__train_30_epochs__mac_cpu/`
- Forecast reuse manifest: `data/manifests/forecast_profile_reuse/mac_cpu.json`

Latest refreshed run manifests:
- Stage 01: `data/manifests/runs/01_ingest_cboe/20260426T065107Z_01_ingest_cboe.json`
- Stage 02: `data/manifests/runs/02_build_option_panel/20260426T072825Z_02_build_option_panel.json`
- Stage 03: `data/manifests/runs/03_build_surfaces/20260426T073017Z_03_build_surfaces.json`
- Stage 04: `data/manifests/runs/04_build_features/20260426T073437Z_04_build_features.json`
- Stage 06: `data/manifests/runs/06_run_walkforward/20260426T165740Z_06_run_walkforward.json`
- Stage 07: `data/manifests/runs/07_run_stats/20260426T165815Z_07_run_stats.json`
- Stage 08: `data/manifests/runs/08_run_hedging_eval/20260426T170000Z_08_run_hedging_eval.json`
- Stage 09: `data/manifests/runs/09_make_report_artifacts/20260426T170024Z_09_make_report_artifacts.json`

Key refreshed empirical outputs:
- Primary observed-cell MSE ranking remains led by `naive` at `0.000025`, followed by `har_factor` at `0.000041` and `random_forest` at `0.000068`.
- Observed-cell QLIKE remains led by `har_factor` at `0.024208`; refreshed `neural_surface` QLIKE is `2598062.037716`.
- Conditional surface-revaluation ranking remains led by `naive` at `5.729051`; refreshed `neural_surface` mean absolute revaluation error is `96.935700`.
- The Mac CPU profile now regenerates `lightgbm` locally with a no-OpenMP LightGBM build. Its refreshed observed-cell MSE is `0.000273`, observed-cell QLIKE is `0.057431`, and mean absolute conditional revaluation error is `10.075435`.
- Corrected price-convexity diagnostics: `neural_surface` averages `6.809524` calendar violations and `0.074869` butterfly-convexity violations per forecast surface, with magnitudes `0.000157` and `0.024715`.

The full historical Windows/CUDA dossier for `hpo_30_trials__train_30_epochs` is retained below for comparison.

# Historical Results Dossier: `hpo_30_trials__train_30_epochs`

This file is a self-contained export of the saved outputs for the official end-to-end run.
It is meant to give another model enough information to draft the full paper without reopening the local artifacts.

Important naming note:
- In the codebase and saved artifacts, the no-change / persistence benchmark is named `naive`.
- When discussing results in prose, `naive` and "no-change surface benchmark" refer to the same model.

## Canonical artifact locations

- Report overview: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/index.md`
- Report tables: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/`
- Report detailed frames: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/details/`
- Forecast files: `data/gold/forecasts/hpo_30_trials__train_30_epochs/`
- Tuning manifests: `data/manifests/tuning/hpo_30_trials/`
- Run manifests: `data/manifests/runs/`

## Run identity and reproducibility

### Workflow identity

| item | value |
| --- | --- |
| workflow run label | `hpo_30_trials__train_30_epochs` |
| official benchmark model | `naive` |
| official loss metrics | `observed_mse_total_variance`, `observed_qlike_total_variance` |
| primary loss metric | `observed_mse_total_variance` |
| target symbol | `^SPX` |
| decision timestamp | `15:45 America/New_York` |
| sample window | `2004-01-02` through `2021-04-09` |
| random seed | `7` |

### Data and split identity

| item | value |
| --- | --- |
| raw files processed | `4347` |
| gold surface files built | `4347` |
| feature rows | `4325` |
| feature columns | `898` |
| total walk-forward splits | `175` |
| tuning splits used in HPO | `3` |
| first clean evaluation split | `split_0002` |
| max HPO validation date | `2006-10-02` |
| clean evaluation splits | `173` |
| forecast rows per model | `3633` |
| evaluation quote-date range | `2006-10-03` through `2021-03-10` |
| evaluation target-date range | `2006-10-04` through `2021-03-11` |
| split manifest hash | `51c7742f77248fe89b477da29c2432fb843328fd0bbdd7c846a2bf2e15c8c92f` |
| stage-06 data manifest hash | `5ea7056f07d7733b17ebadf2a99379d01c4fce441ff68308d6bc691820785169` |

### Software and hardware

These values are from `data/manifests/runs/06_run_walkforward/20260413T211819Z_06_run_walkforward.json` and `data/manifests/runs/09_make_report_artifacts/20260413T212227Z_09_make_report_artifacts.json`.

| item | value |
| --- | --- |
| Python | `3.13.5` |
| OS | `Windows-11-10.0.26200-SP0` |
| CPU | `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel` |
| CPU count | `28` |
| CUDA available | `true` |
| GPU | `NVIDIA GeForce RTX 4070 SUPER` |
| GPU memory | `12878086144` bytes |
| key packages | `polars 1.39.3`, `pyarrow 23.0.1`, `numpy 2.4.4`, `scikit-learn 1.8.0`, `optuna 4.8.0`, `lightgbm 4.6.0`, `torch 2.11.0+cu128`, `arch 8.0.0` |

### Git commits actually used by the saved artifacts

| stage group | commit hash |
| --- | --- |
| `01_ingest_cboe`, `02_build_option_panel`, `03_build_surfaces`, saved stage-05 tuning manifests | `67f2d85431091f7622eb954c1b22eaed666762a9` |
| `04_build_features` | `69c11b4578bbd67eea55c41b7a236e0a8f815e79` |
| `06_run_walkforward`, `07_run_stats`, `08_run_hedging_eval`, `09_make_report_artifacts` | `a0c9b967752c102ee6c6ea988c4b067c9edbb891` |

## End-to-end pipeline execution summary

### Stage runtimes

| stage | latest manifest | duration_seconds | key metadata |
| --- | --- | --- | --- |
| 01 ingest | `data/manifests/runs/01_ingest_cboe/20260412T203814Z_01_ingest_cboe.json` | `126.821796` | `4347` files processed, `4347` files written |
| 02 option panel | `data/manifests/runs/02_build_option_panel/20260412T204240Z_02_build_option_panel.json` | `6.780630` | `4347` bronze files processed |
| 03 surfaces | `data/manifests/runs/03_build_surfaces/20260412T204300Z_03_build_surfaces.json` | `3.576103` | `4347` silver files processed, `4347` gold files written |
| 04 features | `data/manifests/runs/04_build_features/20260412T053330Z_04_build_features.json` | `932.209832` | `4325` feature rows |
| 05 tuning | per-model manifests under `data/manifests/runs/05_tune_models/` | per model | see tuning table below |
| 06 walk-forward | `data/manifests/runs/06_run_walkforward/20260413T211819Z_06_run_walkforward.json` | `11043.070643` | `173` clean splits, `3633` forecast rows per model |
| 07 stats | `data/manifests/runs/07_run_stats/20260413T211849Z_07_run_stats.json` | `26.867939` | DM, SPA, and simplified Tmax MCS on `2` official loss metrics |
| 08 hedging | `data/manifests/runs/08_run_hedging_eval/20260413T212141Z_08_run_hedging_eval.json` | `169.204809` | `25431` hedging results rows |
| 09 report | `data/manifests/runs/09_make_report_artifacts/20260413T212227Z_09_make_report_artifacts.json` | `43.149464` | report bundle written to `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs` |

## Data scope and geometry

### Surface grid and feature construction

| item | value |
| --- | --- |
| moneyness grid | `[-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30]` |
| maturity grid (days) | `[1, 7, 14, 30, 60, 90, 180, 365, 730]` |
| grid size | `9 x 9 = 81 cells` |
| interpolation order | `["maturity", "moneyness"]` |
| interpolation cycles | `2` |
| total variance floor | `1.0e-8` |
| observed cell minimum count | `1` |
| lag windows | `[1, 5, 22]` |
| extra feature families | daily change, observed mask, liquidity |

### Data volume summary

These values are derived from `silver_build_summary.json`, `gold_surface_summary.json`, `daily_features.parquet`, and the walk-forward manifest.

| item | value |
| --- | --- |
| total silver-panel rows processed | `23827107` |
| total valid silver rows | `18169131` |
| silver rows per day | min `335`, median `2726`, max `21104` |
| valid silver rows per day | min `228`, median `1894`, max `18152` |
| gold observed cells total | `277737` |
| observed cells per day | min `37`, median `65`, max `81` |
| completed cells per day | always `81` |
| feature quote-date range | `2004-02-03` through `2021-04-08` |
| feature target-date range | `2004-02-04` through `2021-04-09` |

## Walk-forward protocol and evaluation configuration

### Walk-forward geometry

| item | value |
| --- | --- |
| train size | `504` trading days |
| validation size | `126` trading days |
| test size | `21` trading days |
| step size | `21` trading days |
| expanding train | `true` |

Additional split facts from `walkforward_splits.json`:

- Total splits: `175`
- First split id: `split_0000`
- Last split id: `split_0174`
- First split validation range: `2006-02-02` through `2006-08-02`
- First clean split (`split_0002`) test range: `2006-10-03` through `2006-10-31`
- Last split test range: `2021-02-09` through `2021-03-10`

### HPO and training profiles

| item | `30/30` run value |
| --- | --- |
| HPO profile | `hpo_30_trials` |
| HPO trials requested | `30` |
| tuning splits count | `3` |
| Optuna sampler | `tpe` |
| Optuna pruner | `median` |
| training profile | `train_30_epochs` |
| neural epochs | `30` |
| neural early stopping patience | `5` |
| neural min epochs before early stop | `10` |
| LightGBM early stopping rounds | `25` |

### Statistical evaluation config

| item | value |
| --- | --- |
| official loss metrics | `observed_mse_total_variance`, `observed_qlike_total_variance` |
| benchmark model | `naive` |
| DM alternative | `greater` |
| DM max lag | `0` |
| SPA block size | `5` |
| SPA bootstrap reps | `500` |
| SPA alpha | `0.10` |
| MCS block size | `5` |
| MCS bootstrap reps | `500` |
| MCS alpha | `0.10` |
| full-grid weighting | `uniform` |
| positive floor | `1.0e-8` |

### Hedging evaluation config

| item | value |
| --- | --- |
| risk-free rate | `0.0` |
| level notional | `1.0` |
| skew notional | `1.0` |
| calendar notional | `0.5` |
| skew moneyness abs | `0.10` |
| short maturity days | `30` |
| long maturity days | `90` |
| hedge maturity days | `30` |
| hedge straddle moneyness | `0.0` |
| spot source | `median_valid_active_underlying_price_1545` |

## Model universe and tuned hyperparameters

### Models evaluated

- `naive` (no-change / persistence benchmark)
- `ridge`
- `elasticnet`
- `har_factor`
- `lightgbm`
- `random_forest`
- `neural_surface`

### Saved tuning manifests

| model | stage-05 best value | best params | trials completed | trials pruned |
| --- | --- | --- | --- | --- |
| `ridge` | `0.00011775166321408725` | `alpha=97.2956052631225` | `13` | `17` |
| `elasticnet` | `0.00011115214657925672` | `alpha=0.06673358872570778`, `l1_ratio=0.8537754503645` | `11` | `19` |
| `har_factor` | `0.00011388628600664281` | `n_factors=9`, `alpha=6.644070263467316` | `14` | `16` |
| `lightgbm` | `0.00012129157327578915` | `n_estimators=400`, `learning_rate=0.11962323112686886`, `num_leaves=21`, `max_depth=3`, `min_child_samples=20`, `feature_fraction=0.8655344460953254`, `lambda_l2=0.1024333825675971`, `n_factors=9` | `22` | `8` |
| `random_forest` | `0.00012688219090806608` | `n_estimators=400`, `max_depth=10`, `min_samples_leaf=1` | `19` | `11` |
| `neural_surface` | `0.0009115835487545474` | `hidden_width=384`, `depth=2`, `dropout=0.10712118850586368`, `learning_rate=0.00422336970502679`, `weight_decay=5.138298522573402e-05`, `batch_size=128`, `calendar_penalty_weight=0.006503132708010469`, `convexity_penalty_weight=1.0627295606594362e-05`, `roughness_penalty_weight=0.00035852333027431416` | `9` | `21` |

### Tuning-diagnostics notes that matter for interpretation

These are derived from the saved stage-05 diagnostics parquet files.

- `neural_surface` tuning diagnostics:
  - completed split rows: `27`
  - pruned split rows: `21`
  - median selected metric value: `0.0009240862003406462`
  - median running mean selected metric: `0.0008948484630641971`
  - median prediction/target ratio: `0.031044738622147343`
  - median share of predictions below `1e-6`: `0.7056633352929649`
  - median best epoch: `1`
- `ridge` tuning diagnostics:
  - completed split rows: `39`
  - pruned split rows: `17`
  - median selected metric value: `0.00013989372532167017`
  - maximum selected metric value observed during tuning: `1.6255346165926275e+123`
  - maximum running mean selected metric observed during tuning: `5.418448721975425e+122`

## Official report overview

The following bullets are a direct export of the saved report overview:

- Benchmark model: `naive`
- Official loss metrics: `observed_mse_total_variance`, `observed_qlike_total_variance`
- Primary loss metric: `observed_mse_total_variance`
- Best full-sample loss model: `naive` (`0.000025`)
- Best primary tail-risk model by 95th percentile: `naive` (`0.000055`)
- Best hedging revaluation model: `naive` (`5.729051`)
- Simplified Tmax-set included models on the primary metric: `naive`
- Interpolation sensitivity summary: mean RMSE diff `0.003225`, max abs diff `1.407986`
- Best model by `observed_mse_total_variance`: `naive` (`0.000025`)
- Best model by `observed_qlike_total_variance`: `har_factor` (`0.024208`)

## Official saved tables

### Ranked loss summary: primary metric

| rank | model_name | mean_observed_mse_total_variance | std_observed_mse_total_variance | n_target_dates | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- |
| 1 | naive | 0.000025 | 0.000301 | 3633 | 0.000000 |
| 2 | har_factor | 0.000041 | 0.000238 | 3633 | -63.095180 |
| 3 | random_forest | 0.000068 | 0.000369 | 3633 | -170.126566 |
| 4 | lightgbm | 0.000271 | 0.001167 | 3633 | -970.394083 |
| 5 | elasticnet | 0.000631 | 0.011485 | 3633 | -2396.377315 |
| 6 | neural_surface | 0.002981 | 0.003455 | 3633 | -11686.766850 |
| 7 | ridge | 2.014039 | 121.321040 | 3633 | -7963691.906928 |

### Ranked loss summary: secondary metric

| rank | model_name | mean_observed_qlike_total_variance | std_observed_qlike_total_variance | n_target_dates | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- |
| 1 | har_factor | 0.024208 | 0.050316 | 3633 | 99.743762 |
| 2 | random_forest | 0.029224 | 0.062592 | 3633 | 99.690676 |
| 3 | elasticnet | 0.034404 | 0.079165 | 3633 | 99.635842 |
| 4 | lightgbm | 0.057419 | 0.161566 | 3633 | 99.392234 |
| 5 | ridge | 1.345556 | 79.073370 | 3633 | 85.757665 |
| 6 | naive | 9.447580 | 319.957890 | 3633 | 0.000000 |
| 7 | neural_surface | 2603051.233548 | 1539348.126971 | 3633 | -27552471.974195 |

### Diebold-Mariano results: primary metric

| model_a | model_b | n_obs | mean_loss_a | mean_loss_b | mean_differential | statistic | p_value | alternative | max_lag | loss_metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | ridge | 3633 | 0.000025 | 2.014039 | -2.014014 | -1.000734 | 0.841522 | greater | 0 | observed_mse_total_variance |
| naive | elasticnet | 3633 | 0.000025 | 0.000631 | -0.000606 | -3.184386 | 0.999275 | greater | 0 | observed_mse_total_variance |
| naive | har_factor | 3633 | 0.000025 | 0.000041 | -0.000016 | -4.574213 | 0.999998 | greater | 0 | observed_mse_total_variance |
| naive | random_forest | 3633 | 0.000025 | 0.000068 | -0.000043 | -7.878656 | 1.000000 | greater | 0 | observed_mse_total_variance |
| naive | lightgbm | 3633 | 0.000025 | 0.000271 | -0.000245 | -13.041010 | 1.000000 | greater | 0 | observed_mse_total_variance |
| naive | neural_surface | 3633 | 0.000025 | 0.002981 | -0.002956 | -52.021752 | 1.000000 | greater | 0 | observed_mse_total_variance |

### Diebold-Mariano results: secondary metric

| model_a | model_b | n_obs | mean_loss_a | mean_loss_b | mean_differential | statistic | p_value | alternative | max_lag | loss_metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | har_factor | 3633 | 9.447580 | 0.024208 | 9.423372 | 1.775452 | 0.037912 | greater | 0 | observed_qlike_total_variance |
| naive | random_forest | 3633 | 9.447580 | 0.029224 | 9.418356 | 1.774504 | 0.037990 | greater | 0 | observed_qlike_total_variance |
| naive | elasticnet | 3633 | 9.447580 | 0.034404 | 9.413176 | 1.773527 | 0.038071 | greater | 0 | observed_qlike_total_variance |
| naive | lightgbm | 3633 | 9.447580 | 0.057419 | 9.390161 | 1.769188 | 0.038431 | greater | 0 | observed_qlike_total_variance |
| naive | ridge | 3633 | 9.447580 | 1.345556 | 8.102024 | 1.481742 | 0.069204 | greater | 0 | observed_qlike_total_variance |
| naive | neural_surface | 3633 | 9.447580 | 2603051.233548 | -2603041.785968 | -101.938280 | 1.000000 | greater | 0 | observed_qlike_total_variance |

### SPA results

#### Primary metric

| benchmark_model | candidate_models | observed_statistic | p_value | mean_differentials | superior_models_by_mean | alpha | block_size | bootstrap_reps | loss_metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | ["elasticnet","har_factor","lightgbm","neural_surface","random_forest","ridge"] | -1.000597 | 0.586000 | [-0.0006060426479436651,-0.000015956740013184216,-0.00024541218774537727,-0.0029555776058440924,-0.000043024925069164886,-2.0140137783342555] | [] | 0.100000 | 5 | 500 | observed_mse_total_variance |

#### Secondary metric

| benchmark_model | candidate_models | observed_statistic | p_value | mean_differentials | superior_models_by_mean | alpha | block_size | bootstrap_reps | loss_metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive | ["elasticnet","har_factor","lightgbm","neural_surface","random_forest","ridge"] | 1.775208 | 0.102000 | [9.413175706005022,9.42337157863746,9.390160697520644,-2603041.7859679605,9.41835624078733,8.102023855303717] | ["elasticnet","har_factor","lightgbm","random_forest","ridge"] | 0.100000 | 5 | 500 | observed_qlike_total_variance |

### Simplified Tmax model confidence set

#### Primary metric

| model_name | included_in_simplified_tmax_set | simplified_tmax_alpha |
| --- | --- | --- |
| naive | true | 0.100000 |
| elasticnet | false | 0.100000 |
| har_factor | false | 0.100000 |
| lightgbm | false | 0.100000 |
| neural_surface | false | 0.100000 |
| random_forest | false | 0.100000 |
| ridge | false | 0.100000 |

#### Secondary metric

| model_name | included_in_simplified_tmax_set | simplified_tmax_alpha |
| --- | --- | --- |
| har_factor | true | 0.100000 |
| elasticnet | false | 0.100000 |
| lightgbm | false | 0.100000 |
| naive | false | 0.100000 |
| neural_surface | false | 0.100000 |
| random_forest | false | 0.100000 |
| ridge | false | 0.100000 |

### Ranked hedging summary

| rank | model_name | mean_abs_revaluation_error | mean_squared_revaluation_error | mean_abs_hedged_pnl | mean_squared_hedged_pnl | hedged_pnl_variance | n_trades | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | naive | 5.729051 | 102.158799 | 2.877114 | 20.091176 | 19.649361 | 3633 | 0.000000 |
| 2 | har_factor | 6.353638 | 115.587334 | 2.837992 | 19.927431 | 19.484645 | 3633 | -10.902102 |
| 3 | random_forest | 7.901773 | 172.314897 | 2.857332 | 19.769932 | 19.312888 | 3633 | -37.924641 |
| 4 | lightgbm | 10.107414 | 328.011463 | 2.929702 | 21.897986 | 21.464956 | 3633 | -76.423868 |
| 5 | ridge | 13.087057 | 2369.326573 | 2.989144 | 24.336585 | 23.875317 | 3633 | -128.433226 |
| 6 | elasticnet | 13.671021 | 1449.113075 | 2.933073 | 22.074411 | 21.628252 | 3633 | -138.626267 |
| 7 | neural_surface | 96.937066 | 12126.918827 | 3.242177 | 50.687661 | 50.161790 | 3633 | -1592.026489 |

### Slice leaders: primary metric

| slice_family | slice_label | evaluation_scope | best_model_name | best_metric_value | benchmark_model | benchmark_metric_value | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| maturity | 1d | full | har_factor | 0.000027 | naive | 0.000050 | 46.605807 |
| maturity | 7d | full | har_factor | 0.000009 | naive | 0.000015 | 38.749967 |
| maturity | 14d | full | har_factor | 0.000006 | naive | 0.000007 | 21.648583 |
| maturity | 180d | full | naive | 0.000022 | naive | 0.000022 | 0.000000 |
| maturity | 30d | full | naive | 0.000006 | naive | 0.000006 | 0.000000 |
| maturity | 365d | full | naive | 0.000035 | naive | 0.000035 | 0.000000 |
| maturity | 60d | full | naive | 0.000007 | naive | 0.000007 | 0.000000 |
| maturity | 730d | full | naive | 0.000071 | naive | 0.000071 | 0.000000 |
| maturity | 90d | full | naive | 0.000008 | naive | 0.000008 | 0.000000 |
| maturity | 1d | observed | random_forest | 0.000001 | naive | 0.000001 | 12.187596 |
| maturity | 14d | observed | naive | 0.000002 | naive | 0.000002 | 0.000000 |
| maturity | 180d | observed | naive | 0.000016 | naive | 0.000016 | 0.000000 |
| maturity | 30d | observed | naive | 0.000003 | naive | 0.000003 | 0.000000 |
| maturity | 365d | observed | naive | 0.000020 | naive | 0.000020 | 0.000000 |
| maturity | 60d | observed | naive | 0.000006 | naive | 0.000006 | 0.000000 |
| maturity | 730d | observed | naive | 0.000029 | naive | 0.000029 | 0.000000 |
| maturity | 7d | observed | naive | 0.000001 | naive | 0.000001 | 0.000000 |
| maturity | 90d | observed | naive | 0.000009 | naive | 0.000009 | 0.000000 |
| moneyness | +0.05 | full | har_factor | 0.000020 | naive | 0.000020 | 0.888646 |
| moneyness | -0.30 | full | naive | 0.000070 | naive | 0.000070 | 0.000000 |
| moneyness | -0.10 | full | naive | 0.000020 | naive | 0.000020 | 0.000000 |
| moneyness | -0.20 | full | naive | 0.000032 | naive | 0.000032 | 0.000000 |
| moneyness | +0.10 | full | naive | 0.000006 | naive | 0.000006 | 0.000000 |
| moneyness | +0.30 | full | naive | 0.000018 | naive | 0.000018 | 0.000000 |
| moneyness | +0.20 | full | naive | 0.000012 | naive | 0.000012 | 0.000000 |
| moneyness | +0.00 | full | naive | 0.000020 | naive | 0.000020 | 0.000000 |
| moneyness | -0.05 | full | naive | 0.000021 | naive | 0.000021 | 0.000000 |
| moneyness | +0.30 | observed | naive | 0.000041 | naive | 0.000041 | 0.000000 |
| moneyness | +0.05 | observed | naive | 0.000010 | naive | 0.000010 | 0.000000 |
| moneyness | -0.10 | observed | naive | 0.000013 | naive | 0.000013 | 0.000000 |
| moneyness | -0.05 | observed | naive | 0.000007 | naive | 0.000007 | 0.000000 |
| moneyness | +0.20 | observed | naive | 0.000027 | naive | 0.000027 | 0.000000 |
| moneyness | +0.10 | observed | naive | 0.000015 | naive | 0.000015 | 0.000000 |
| moneyness | -0.20 | observed | naive | 0.000018 | naive | 0.000018 | 0.000000 |
| moneyness | +0.00 | observed | naive | 0.000006 | naive | 0.000006 | 0.000000 |
| moneyness | -0.30 | observed | naive | 0.000034 | naive | 0.000034 | 0.000000 |
| stress_window | covid_2020 | full | naive | 0.000087 | naive | 0.000087 | 0.000000 |
| stress_window | gfc_2008_2009 | full | naive | 0.000127 | naive | 0.000127 | 0.000000 |
| stress_window | volmageddon_2018 | full | naive | 0.000010 | naive | 0.000010 | 0.000000 |
| stress_window | covid_2020 | observed | naive | 0.000102 | naive | 0.000102 | 0.000000 |
| stress_window | gfc_2008_2009 | observed | naive | 0.000117 | naive | 0.000117 | 0.000000 |
| stress_window | volmageddon_2018 | observed | naive | 0.000010 | naive | 0.000010 | 0.000000 |

### Slice leaders: secondary metric

| slice_family | slice_label | evaluation_scope | best_model_name | best_metric_value | benchmark_model | benchmark_metric_value | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| maturity | 1d | full | har_factor | 0.113667 | naive | 84.607536 | 99.865654 |
| maturity | 7d | full | har_factor | 0.053100 | naive | 0.078433 | 32.299672 |
| maturity | 14d | full | har_factor | 0.035877 | naive | 0.046254 | 22.434266 |
| maturity | 30d | full | har_factor | 0.027319 | naive | 0.029783 | 8.273427 |
| maturity | 730d | full | naive | 0.002037 | naive | 0.002037 | 0.000000 |
| maturity | 365d | full | naive | 0.002869 | naive | 0.002869 | 0.000000 |
| maturity | 60d | full | naive | 0.019983 | naive | 0.019983 | 0.000000 |
| maturity | 90d | full | naive | 0.012704 | naive | 0.012704 | 0.000000 |
| maturity | 180d | full | naive | 0.007887 | naive | 0.007887 | 0.000000 |
| maturity | 1d | observed | har_factor | 0.117016 | naive | 170.512604 | 99.931374 |
| maturity | 7d | observed | har_factor | 0.047565 | naive | 0.066673 | 28.659115 |
| maturity | 14d | observed | har_factor | 0.032372 | naive | 0.038734 | 16.423604 |
| maturity | 180d | observed | naive | 0.006081 | naive | 0.006081 | 0.000000 |
| maturity | 30d | observed | naive | 0.021745 | naive | 0.021745 | 0.000000 |
| maturity | 365d | observed | naive | 0.002645 | naive | 0.002645 | 0.000000 |
| maturity | 60d | observed | naive | 0.013460 | naive | 0.013460 | 0.000000 |
| maturity | 730d | observed | naive | 0.001848 | naive | 0.001848 | 0.000000 |
| maturity | 90d | observed | naive | 0.007942 | naive | 0.007942 | 0.000000 |
| moneyness | +0.30 | full | elasticnet | 0.033781 | naive | 56.633440 | 99.940352 |
| moneyness | +0.20 | full | har_factor | 0.043322 | naive | 19.453860 | 99.777307 |
| moneyness | +0.10 | full | har_factor | 0.031432 | naive | 7.765433 | 99.595234 |
| moneyness | -0.30 | full | har_factor | 0.026861 | naive | 0.635990 | 95.776511 |
| moneyness | -0.20 | full | har_factor | 0.018397 | naive | 0.101022 | 81.788905 |
| moneyness | +0.00 | full | har_factor | 0.050668 | naive | 0.095266 | 46.814110 |
| moneyness | -0.05 | full | har_factor | 0.029017 | naive | 0.044725 | 35.121057 |
| moneyness | +0.05 | full | har_factor | 0.036105 | naive | 0.052875 | 31.716758 |
| moneyness | -0.10 | full | har_factor | 0.020066 | naive | 0.024874 | 19.332249 |
| moneyness | +0.30 | observed | elasticnet | 0.028367 | naive | 124.364575 | 99.977190 |
| moneyness | +0.20 | observed | har_factor | 0.030149 | naive | 28.118970 | 99.892781 |
| moneyness | +0.10 | observed | har_factor | 0.028038 | naive | 9.036358 | 99.689717 |
| moneyness | -0.30 | observed | har_factor | 0.023535 | naive | 0.740376 | 96.821161 |
| moneyness | -0.20 | observed | har_factor | 0.015736 | naive | 0.104101 | 84.884379 |
| moneyness | +0.05 | observed | har_factor | 0.029213 | naive | 0.032693 | 10.645375 |
| moneyness | +0.00 | observed | naive | 0.028618 | naive | 0.028618 | 0.000000 |
| moneyness | -0.05 | observed | naive | 0.017873 | naive | 0.017873 | 0.000000 |
| moneyness | -0.10 | observed | naive | 0.015310 | naive | 0.015310 | 0.000000 |
| stress_window | gfc_2008_2009 | full | har_factor | 0.048224 | naive | 0.074850 | 35.572317 |
| stress_window | covid_2020 | full | naive | 0.037595 | naive | 0.037595 | 0.000000 |
| stress_window | volmageddon_2018 | full | naive | 0.047540 | naive | 0.047540 | 0.000000 |
| stress_window | covid_2020 | observed | naive | 0.034535 | naive | 0.034535 | 0.000000 |
| stress_window | gfc_2008_2009 | observed | naive | 0.021217 | naive | 0.021217 | 0.000000 |
| stress_window | volmageddon_2018 | observed | naive | 0.046825 | naive | 0.046825 | 0.000000 |

### Tail-risk summary: primary metric

| loss_metric | model_name | mean_loss | p90_loss | p95_loss | p99_loss | max_loss | p95_improvement_vs_benchmark_pct | max_improvement_vs_benchmark_pct | n_target_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_mse_total_variance | naive | 0.000025 | 0.000021 | 0.000055 | 0.000281 | 0.009466 | 0.000000 | 0.000000 | 3633 |
| observed_mse_total_variance | har_factor | 0.000041 | 0.000077 | 0.000145 | 0.000410 | 0.008825 | -161.428035 | 6.774194 | 3633 |
| observed_mse_total_variance | ridge | 2.014039 | 0.000074 | 0.000170 | 0.008059 | 7312.550531 | -207.055006 | -77248961.735406 | 3633 |
| observed_mse_total_variance | elasticnet | 0.000631 | 0.000084 | 0.000183 | 0.002774 | 0.460190 | -229.819093 | -4761.396537 | 3633 |
| observed_mse_total_variance | random_forest | 0.000068 | 0.000097 | 0.000216 | 0.000977 | 0.008812 | -290.377918 | 6.912751 | 3633 |
| observed_mse_total_variance | lightgbm | 0.000271 | 0.000333 | 0.000790 | 0.006345 | 0.018709 | -1326.261439 | -97.644623 | 3633 |
| observed_mse_total_variance | neural_surface | 0.002981 | 0.007236 | 0.009927 | 0.016812 | 0.034380 | -17816.402114 | -263.187484 | 3633 |

### Tail-risk summary: secondary metric

| loss_metric | model_name | mean_loss | p90_loss | p95_loss | p99_loss | max_loss | p95_improvement_vs_benchmark_pct | max_improvement_vs_benchmark_pct | n_target_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_qlike_total_variance | har_factor | 0.024208 | 0.037121 | 0.048626 | 0.115967 | 1.607816 | 20.401484 | 99.988199 | 3633 |
| observed_qlike_total_variance | naive | 9.447580 | 0.046226 | 0.061089 | 0.124176 | 13624.020964 | 0.000000 | 0.000000 | 3633 |
| observed_qlike_total_variance | random_forest | 0.029224 | 0.044280 | 0.063050 | 0.194203 | 1.981348 | -3.209780 | 99.985457 | 3633 |
| observed_qlike_total_variance | elasticnet | 0.034404 | 0.049186 | 0.071502 | 0.255378 | 1.865935 | -17.044747 | 99.986304 | 3633 |
| observed_qlike_total_variance | ridge | 1.345556 | 0.047553 | 0.073173 | 0.337673 | 4766.128618 | -19.780904 | 65.016726 | 3633 |
| observed_qlike_total_variance | lightgbm | 0.057419 | 0.065886 | 0.167780 | 1.005894 | 2.282860 | -174.646878 | 99.983244 | 3633 |
| observed_qlike_total_variance | neural_surface | 2603051.233548 | 4271378.494819 | 5390357.436230 | 8966994.619643 | 14173345.198720 | -8823725586.897585 | -103932.027225 | 3633 |

### Worst-day drilldown: primary metric

| loss_metric | model_name | rank_within_model | quote_date | target_date | loss_value | benchmark_model | benchmark_loss_value | excess_loss_vs_benchmark | loss_ratio_vs_benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_mse_total_variance | elasticnet | 1 | 2008-10-24 | 2008-10-27 | 0.460190 | naive | 0.000025 | 0.460165 | 18596.979288 |
| observed_mse_total_variance | elasticnet | 2 | 2008-10-27 | 2008-10-28 | 0.362175 | naive | 0.000187 | 0.361988 | 1932.721573 |
| observed_mse_total_variance | elasticnet | 3 | 2020-03-18 | 2020-03-19 | 0.201666 | naive | 0.001387 | 0.200279 | 145.436503 |
| observed_mse_total_variance | elasticnet | 4 | 2008-10-10 | 2008-10-13 | 0.143929 | naive | 0.000460 | 0.143469 | 313.151832 |
| observed_mse_total_variance | elasticnet | 5 | 2008-10-31 | 2008-11-03 | 0.135363 | naive | 0.000095 | 0.135268 | 1423.414281 |
| observed_mse_total_variance | har_factor | 1 | 2006-10-04 | 2006-10-05 | 0.008825 | naive | 0.008821 | 0.000004 | 1.000473 |
| observed_mse_total_variance | har_factor | 2 | 2006-10-25 | 2006-10-26 | 0.007936 | naive | 0.007958 | -0.000022 | 0.997186 |
| observed_mse_total_variance | har_factor | 3 | 2009-06-29 | 2009-06-30 | 0.005518 | naive | 0.004675 | 0.000843 | 1.180284 |
| observed_mse_total_variance | har_factor | 4 | 2008-12-23 | 2008-12-24 | 0.002557 | naive | 0.002325 | 0.000232 | 1.099631 |
| observed_mse_total_variance | har_factor | 5 | 2020-03-17 | 2020-03-18 | 0.002062 | naive | 0.001149 | 0.000912 | 1.793752 |
| observed_mse_total_variance | lightgbm | 1 | 2008-11-19 | 2008-11-20 | 0.018709 | naive | 0.000362 | 0.018347 | 51.625448 |
| observed_mse_total_variance | lightgbm | 2 | 2008-11-18 | 2008-11-19 | 0.015124 | naive | 0.000293 | 0.014831 | 51.574472 |
| observed_mse_total_variance | lightgbm | 3 | 2008-11-20 | 2008-11-21 | 0.013846 | naive | 0.000693 | 0.013153 | 19.977737 |
| observed_mse_total_variance | lightgbm | 4 | 2008-12-03 | 2008-12-04 | 0.012736 | naive | 0.000434 | 0.012302 | 29.359034 |
| observed_mse_total_variance | lightgbm | 5 | 2008-11-28 | 2008-12-01 | 0.012643 | naive | 0.000437 | 0.012206 | 28.933572 |
| observed_mse_total_variance | naive | 1 | 2006-10-05 | 2006-10-06 | 0.009466 | naive | 0.009466 | 0.000000 | 1.000000 |
| observed_mse_total_variance | naive | 2 | 2006-10-04 | 2006-10-05 | 0.008821 | naive | 0.008821 | 0.000000 | 1.000000 |
| observed_mse_total_variance | naive | 3 | 2006-10-25 | 2006-10-26 | 0.007958 | naive | 0.007958 | 0.000000 | 1.000000 |
| observed_mse_total_variance | naive | 4 | 2006-10-26 | 2006-10-27 | 0.006593 | naive | 0.006593 | 0.000000 | 1.000000 |
| observed_mse_total_variance | naive | 5 | 2009-06-29 | 2009-06-30 | 0.004675 | naive | 0.004675 | 0.000000 | 1.000000 |
| observed_mse_total_variance | neural_surface | 1 | 2008-11-19 | 2008-11-20 | 0.034380 | naive | 0.000362 | 0.034018 | 94.865806 |
| observed_mse_total_variance | neural_surface | 2 | 2008-11-18 | 2008-11-19 | 0.030881 | naive | 0.000293 | 0.030587 | 105.304786 |
| observed_mse_total_variance | neural_surface | 3 | 2008-11-20 | 2008-11-21 | 0.028460 | naive | 0.000693 | 0.027767 | 41.063425 |
| observed_mse_total_variance | neural_surface | 4 | 2008-12-03 | 2008-12-04 | 0.028375 | naive | 0.000434 | 0.027941 | 65.410949 |
| observed_mse_total_variance | neural_surface | 5 | 2008-11-28 | 2008-12-01 | 0.026949 | naive | 0.000437 | 0.026513 | 61.675055 |
| observed_mse_total_variance | random_forest | 1 | 2006-10-04 | 2006-10-05 | 0.008812 | naive | 0.008821 | -0.000009 | 0.998986 |
| observed_mse_total_variance | random_forest | 2 | 2009-06-29 | 2009-06-30 | 0.008375 | naive | 0.004675 | 0.003700 | 1.791537 |
| observed_mse_total_variance | random_forest | 3 | 2006-10-25 | 2006-10-26 | 0.007914 | naive | 0.007958 | -0.000044 | 0.994424 |
| observed_mse_total_variance | random_forest | 4 | 2008-11-19 | 2008-11-20 | 0.005654 | naive | 0.000362 | 0.005291 | 15.600624 |
| observed_mse_total_variance | random_forest | 5 | 2008-10-23 | 2008-10-24 | 0.005229 | naive | 0.000283 | 0.004946 | 18.506752 |
| observed_mse_total_variance | ridge | 1 | 2006-10-27 | 2006-10-30 | 7312.550531 | naive | 0.000000 | 7312.550531 | 486579451448.702576 |
| observed_mse_total_variance | ridge | 2 | 2008-10-27 | 2008-10-28 | 1.756430 | naive | 0.000187 | 1.756243 | 9373.064770 |
| observed_mse_total_variance | ridge | 3 | 2020-03-18 | 2020-03-19 | 0.631727 | naive | 0.001387 | 0.630340 | 455.585633 |
| observed_mse_total_variance | ridge | 4 | 2008-10-24 | 2008-10-27 | 0.275540 | naive | 0.000025 | 0.275516 | 11135.021752 |
| observed_mse_total_variance | ridge | 5 | 2008-10-13 | 2008-10-14 | 0.268851 | naive | 0.000064 | 0.268787 | 4185.331888 |

### Worst-day drilldown: secondary metric

| loss_metric | model_name | rank_within_model | quote_date | target_date | loss_value | benchmark_model | benchmark_loss_value | excess_loss_vs_benchmark | loss_ratio_vs_benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_qlike_total_variance | elasticnet | 1 | 2010-12-29 | 2010-12-30 | 1.865935 | naive | 1.863990 | 0.001944 | 1.001043 |
| observed_qlike_total_variance | elasticnet | 2 | 2010-09-28 | 2010-09-29 | 1.829316 | naive | 1.500018 | 0.329299 | 1.219530 |
| observed_qlike_total_variance | elasticnet | 3 | 2006-10-25 | 2006-10-26 | 1.179251 | naive | 1.338893 | -0.159642 | 0.880766 |
| observed_qlike_total_variance | elasticnet | 4 | 2008-10-24 | 2008-10-27 | 1.168344 | naive | 0.016554 | 1.151790 | 70.578431 |
| observed_qlike_total_variance | elasticnet | 5 | 2008-10-27 | 2008-10-28 | 1.163367 | naive | 0.039438 | 1.123930 | 29.498832 |
| observed_qlike_total_variance | har_factor | 1 | 2010-09-28 | 2010-09-29 | 1.607816 | naive | 1.500018 | 0.107798 | 1.071865 |
| observed_qlike_total_variance | har_factor | 2 | 2010-12-29 | 2010-12-30 | 1.559676 | naive | 1.863990 | -0.304315 | 0.836740 |
| observed_qlike_total_variance | har_factor | 3 | 2006-10-25 | 2006-10-26 | 1.219626 | naive | 1.338893 | -0.119267 | 0.910921 |
| observed_qlike_total_variance | har_factor | 4 | 2011-03-29 | 2011-03-30 | 1.095057 | naive | 1.068373 | 0.026684 | 1.024976 |
| observed_qlike_total_variance | har_factor | 5 | 2018-02-02 | 2018-02-05 | 0.440289 | naive | 0.320403 | 0.119886 | 1.374174 |
| observed_qlike_total_variance | lightgbm | 1 | 2008-11-19 | 2008-11-20 | 2.282860 | naive | 0.015084 | 2.267777 | 151.347032 |
| observed_qlike_total_variance | lightgbm | 2 | 2010-09-28 | 2010-09-29 | 2.148034 | naive | 1.500018 | 0.648017 | 1.432006 |
| observed_qlike_total_variance | lightgbm | 3 | 2008-11-20 | 2008-11-21 | 2.044928 | naive | 0.026132 | 2.018795 | 78.252393 |
| observed_qlike_total_variance | lightgbm | 4 | 2008-11-18 | 2008-11-19 | 1.865587 | naive | 0.009836 | 1.855751 | 189.673502 |
| observed_qlike_total_variance | lightgbm | 5 | 2010-12-29 | 2010-12-30 | 1.694831 | naive | 1.863990 | -0.169160 | 0.909249 |
| observed_qlike_total_variance | naive | 1 | 2021-03-01 | 2021-03-02 | 13624.020964 | naive | 13624.020964 | 0.000000 | 1.000000 |
| observed_qlike_total_variance | naive | 2 | 2021-02-24 | 2021-02-25 | 12482.612406 | naive | 12482.612406 | 0.000000 | 1.000000 |
| observed_qlike_total_variance | naive | 3 | 2020-10-29 | 2020-10-30 | 4002.924509 | naive | 4002.924509 | 0.000000 | 1.000000 |
| observed_qlike_total_variance | naive | 4 | 2021-02-23 | 2021-02-24 | 3827.293326 | naive | 3827.293326 | 0.000000 | 1.000000 |
| observed_qlike_total_variance | naive | 5 | 2020-09-10 | 2020-09-11 | 170.426694 | naive | 170.426694 | 0.000000 | 1.000000 |
| observed_qlike_total_variance | neural_surface | 1 | 2008-11-19 | 2008-11-20 | 14173345.198720 | naive | 0.015084 | 14173345.183636 | 939651781.308485 |
| observed_qlike_total_variance | neural_surface | 2 | 2008-11-18 | 2008-11-19 | 12521025.423850 | naive | 0.009836 | 12521025.414015 | 1273007915.820570 |
| observed_qlike_total_variance | neural_surface | 3 | 2008-11-20 | 2008-11-21 | 12497834.804367 | naive | 0.026132 | 12497834.778235 | 478249436.706899 |
| observed_qlike_total_variance | neural_surface | 4 | 2008-11-17 | 2008-11-18 | 11733040.377796 | naive | 0.007085 | 11733040.370711 | 1655961424.928307 |
| observed_qlike_total_variance | neural_surface | 5 | 2008-12-03 | 2008-12-04 | 11489658.082490 | naive | 0.020272 | 11489658.062218 | 566774567.913390 |
| observed_qlike_total_variance | random_forest | 1 | 2010-09-28 | 2010-09-29 | 1.981348 | naive | 1.500018 | 0.481331 | 1.320883 |
| observed_qlike_total_variance | random_forest | 2 | 2010-12-29 | 2010-12-30 | 1.749102 | naive | 1.863990 | -0.114888 | 0.938364 |
| observed_qlike_total_variance | random_forest | 3 | 2011-03-29 | 2011-03-30 | 1.175585 | naive | 1.068373 | 0.107212 | 1.100350 |
| observed_qlike_total_variance | random_forest | 4 | 2006-10-25 | 2006-10-26 | 1.135654 | naive | 1.338893 | -0.203240 | 0.848203 |
| observed_qlike_total_variance | random_forest | 5 | 2008-10-23 | 2008-10-24 | 0.713732 | naive | 0.012170 | 0.701562 | 58.644579 |
| observed_qlike_total_variance | ridge | 1 | 2006-10-26 | 2006-10-27 | 4766.128618 | naive | 0.114215 | 4766.014404 | 41729.613624 |
| observed_qlike_total_variance | ridge | 2 | 2010-09-28 | 2010-09-29 | 1.952939 | naive | 1.500018 | 0.452921 | 1.301944 |
| observed_qlike_total_variance | ridge | 3 | 2010-12-29 | 2010-12-30 | 1.917840 | naive | 1.863990 | 0.053849 | 1.028889 |
| observed_qlike_total_variance | ridge | 4 | 2006-10-27 | 2006-10-30 | 1.871908 | naive | 0.004027 | 1.867881 | 464.807863 |
| observed_qlike_total_variance | ridge | 5 | 2006-10-25 | 2006-10-26 | 1.357526 | naive | 1.338893 | 0.018633 | 1.013917 |

### Arbitrage diagnostic summary

| model_name | mean_calendar_violation_count | mean_calendar_violation_magnitude | mean_convexity_violation_count | mean_convexity_violation_magnitude | n_surfaces |
| --- | --- | --- | --- | --- | --- |
| neural_surface | 36.573631 | 0.000163 | 18.862373 | 0.000324 | 3633 |
| random_forest | 2.646023 | 0.000338 | 4.093862 | 0.008158 | 3633 |
| lightgbm | 4.125241 | 0.000517 | 4.346821 | 0.005847 | 3633 |
| har_factor | 4.128269 | 0.000620 | 5.098816 | 0.011677 | 3633 |
| naive | 3.646023 | 0.005406 | 5.905312 | 0.017751 | 3633 |
| actual_surface | 3.513228 | 0.006529 | 6.008282 | 0.022032 | 4347 |
| elasticnet | 4.268373 | 0.015412 | 4.904211 | 0.051516 | 3633 |
| ridge | 6.306083 | 0.275690 | 5.992843 | 0.686816 | 3633 |

### Interpolation-sensitivity summary

| mean_mean_abs_diff | mean_rmse_diff | max_max_abs_diff | n_quote_dates |
| --- | --- | --- | --- |
| 0.001076 | 0.003225 | 1.407986 | 4347 |

## Derived summaries from the saved daily loss frame

This section is not copied from an existing prebuilt markdown artifact.
It is derived from `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/details/daily_loss_frame.parquet`.

### Day-level comparison counts versus `naive` on the primary metric

| model | days better than `naive` | days worse than `naive` | mean diff vs `naive` | median diff vs `naive` |
| --- | --- | --- | --- | --- |
| `elasticnet` | `284` | `3349` | `0.0006060426479436668` | `8.174689479602482e-06` |
| `har_factor` | `238` | `3395` | `1.5956740013184243e-05` | `6.866630333989989e-06` |
| `lightgbm` | `127` | `3506` | `0.00024541218774537716` | `1.5245410724682316e-05` |
| `neural_surface` | `2` | `3631` | `0.0029555776058440967` | `0.0017849179940299259` |
| `random_forest` | `298` | `3335` | `4.302492506916481e-05` | `5.630249017865299e-06` |
| `ridge` | `390` | `3243` | `2.014013778334248` | `4.452528407435164e-06` |

### Day-level comparison counts versus `naive` on the secondary metric

| model | days better than `naive` | days worse than `naive` | mean diff vs `naive` | median diff vs `naive` |
| --- | --- | --- | --- | --- |
| `elasticnet` | `1302` | `2331` | `-9.413175706005019` | `0.004019318952973125` |
| `har_factor` | `1692` | `1941` | `-9.42337157863746` | `0.0008148437998558612` |
| `lightgbm` | `1216` | `2417` | `-9.390160697520644` | `0.005699609980109386` |
| `neural_surface` | `0` | `3633` | `2603041.7859679605` | `2090120.2998799044` |
| `random_forest` | `1507` | `2126` | `-9.418356240787332` | `0.002588912763950551` |
| `ridge` | `1550` | `2083` | `-8.10202385530371` | `0.0023509754713122947` |

Interpretation of the day-level counts:

- On the primary metric, `naive` wins on the overwhelming majority of target dates against every alternative.
- On the secondary metric, several models improve the mean relative to `naive`, but they still lose to `naive` on more than half of days except `ridge`, whose mean is still dominated by catastrophic outliers.
- This is why the QLIKE story should be written carefully: the mean favors `har_factor`, but the improvement is not broad daily dominance over `naive`.

## Interpolation-sensitivity worst days

Top 10 quote dates by `max_abs_diff` from `details/interpolation_sensitivity.parquet`:

| rank | quote_date | observed_cell_count | mean_abs_diff | rmse_diff | max_abs_diff |
| --- | --- | --- | --- | --- | --- |
| 1 | `2005-11-04` | `52` | `0.2774897893348251` | `0.6238123499335929` | `1.4079862867692325` |
| 2 | `2010-12-30` | `63` | `0.007374270027037277` | `0.028920332236640218` | `0.16719860804095327` |
| 3 | `2008-12-26` | `61` | `0.002740553020043779` | `0.019493504106529076` | `0.16640497877313237` |
| 4 | `2009-06-30` | `58` | `0.0023010535077609266` | `0.016134094950413874` | `0.14419698021012844` |
| 5 | `2008-11-20` | `61` | `0.002652696009850654` | `0.016212697862533994` | `0.13864246626534565` |
| 6 | `2008-11-21` | `58` | `0.006383745418681964` | `0.022396854362059668` | `0.13697899955506987` |
| 7 | `2008-11-19` | `62` | `0.0016421258144953365` | `0.014779132330458028` | `0.13301219097412226` |
| 8 | `2009-01-20` | `59` | `0.0033833873842425764` | `0.015817558388301983` | `0.129463956921721` |
| 9 | `2009-03-03` | `68` | `0.002391972353832844` | `0.01473547752053168` | `0.12935032473673955` |
| 10 | `2008-12-29` | `67` | `0.0021763627867007105` | `0.014452631987910814` | `0.12816860961962362` |

## Writing-safe conclusions supported directly by the artifacts

These are the strongest claims that can be made from the saved results without adding new experiments.

1. Under this leak-safe `15:45 -> next observed session` SPX forecasting protocol, the persistence benchmark (`naive`) is best on the primary official loss metric, best on primary-metric tail risk, and best on hedging revaluation.
2. On the primary metric, `naive` is not merely first on average; it also wins on most target dates against every alternative and is the only model retained by the simplified Tmax model confidence set.
3. `har_factor` is the only learned model with a materially positive secondary story:
   - best mean `observed_qlike_total_variance`
   - strongest short-maturity gains, especially `1d`, `7d`, and `14d`
   - second-best hedging ranking
4. The QLIKE result should not be written as universal dominance over `naive`.
   - `har_factor` improves mean QLIKE
   - `har_factor` still loses to `naive` on `1941` of `3633` target dates
   - `naive` has a few extreme QLIKE blowups that strongly affect the average
5. `ridge` is catastrophically unstable out of sample despite a reasonable tuned stage-05 score.
6. `neural_surface` under the current configuration is materially underperforming and appears degenerate in tuning diagnostics:
   - heavy pruning
   - very small prediction/target ratio
   - large share of near-zero predictions
   - extremely poor out-of-sample MSE, QLIKE, and hedging results
7. `lightgbm` and `random_forest` do not overturn the persistence result.
   - `random_forest` is relatively competitive but still clearly worse than `naive` on the primary metric and hedging
   - `lightgbm` is weaker than `random_forest` and `har_factor` on the primary metric

## Minimal paper-ready phrasing supported by this file

If a paper drafter needs a concise statement that stays faithful to the saved outputs, this is the safest version:

> In a leak-safe SPX implied-volatility-surface forecasting pipeline using `15:45` Cboe option data, a no-change persistence benchmark (`naive`) delivered the best out-of-sample performance on the primary observed-cell MSE loss, the strongest primary-metric tail-risk profile, and the best hedging revaluation performance. A HAR-style factor model (`har_factor`) provided the strongest learned-model result, winning the secondary QLIKE metric and several short-maturity slices, but it did not overturn the primary headline finding.

## Exact source files behind this dossier

- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/index.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/ranked_loss_summary__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/ranked_loss_summary__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/dm_results__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/dm_results__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/spa_result__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/spa_result__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/mcs_result__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/mcs_result__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/ranked_hedging_summary.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/slice_leaders__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/slice_leaders__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/tail_risk_summary__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/tail_risk_summary__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/worst_day_drilldown__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/worst_day_drilldown__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/arbitrage_diagnostic_summary.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/interpolation_sensitivity_summary.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/details/daily_loss_frame.parquet`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/details/interpolation_sensitivity.parquet`
- `data/manifests/tuning/hpo_30_trials/*.json`
- `data/manifests/tuning/hpo_30_trials/*__diagnostics.parquet`
- `data/manifests/runs/01_ingest_cboe/20260412T203814Z_01_ingest_cboe.json`
- `data/manifests/runs/02_build_option_panel/20260412T204240Z_02_build_option_panel.json`
- `data/manifests/runs/03_build_surfaces/20260412T204300Z_03_build_surfaces.json`
- `data/manifests/runs/04_build_features/20260412T053330Z_04_build_features.json`
- `data/manifests/runs/06_run_walkforward/20260413T211819Z_06_run_walkforward.json`
- `data/manifests/runs/07_run_stats/20260413T211849Z_07_run_stats.json`
- `data/manifests/runs/08_run_hedging_eval/20260413T212141Z_08_run_hedging_eval.json`
- `data/manifests/runs/09_make_report_artifacts/20260413T212227Z_09_make_report_artifacts.json`
