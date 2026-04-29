# Results Dossier: `hpo_30_trials__train_30_epochs`

This file records the current canonical end-to-end run completed on 2026-04-29
on Windows/CUDA after commit `b208f9b8aad15367831c67860566a817a06cca8c`
(`Remove audit consultation artifacts`). The proprietary raw zip source was read
from `D:\Options Data`; raw files were not modified.

Important naming note:
- In code and saved artifacts, the no-change / persistence benchmark is named
  `naive`.
- In prose, `naive` and "no-change surface benchmark" refer to the same model.

## Canonical artifact locations

- Report overview: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/index.md`
- Report tables: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/`
- Report detailed frames: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/details/`
- Forecast files: `data/gold/forecasts/hpo_30_trials__train_30_epochs/`
- Tuning manifests: `data/manifests/tuning/hpo_30_trials/`
- Stats outputs: `data/manifests/stats/hpo_30_trials__train_30_epochs/`
- Hedging outputs: `data/manifests/hedging/hpo_30_trials__train_30_epochs/`
- Run manifests: `data/manifests/runs/`

## Run identity and validation

| item | value |
| --- | --- |
| workflow run label | `hpo_30_trials__train_30_epochs` |
| git commit hash | `b208f9b8aad15367831c67860566a817a06cca8c` |
| commit status in generated run manifests | all 20260429 stage manifests record this hash |
| stale generated-output commit hashes found under `data/manifests` | none |
| official benchmark model | `naive` |
| official loss metrics | `observed_mse_total_variance`, `observed_qlike_total_variance` |
| primary loss metric | `observed_mse_total_variance` |
| target symbol | `^SPX` |
| decision timestamp | `15:45 America/New_York` |
| sample window | `2004-01-02` through `2021-04-09` |
| random seed | `7` |

Validation from the terminal run:
- `ruff check .`: passed.
- `pytest`: `262 passed in 49.43s`.
- `mypy src tests`: passed with no issues.
- Runtime preflight: `windows_cuda`, `torch.cuda.is_available(): True`,
  LightGBM GPU mode `True`.
- Environment tools: `uv 0.11.6`, GNU Make `4.4.1`, Python `3.13.5`.
- GPU from the run: NVIDIA GeForce RTX 4070 SUPER, driver `595.97`,
  CUDA runtime reported by `nvidia-smi` as `13.2`.

## Provenance manifests

Every latest 20260429 run manifest below records
`git_commit_hash = b208f9b8aad15367831c67860566a817a06cca8c`.

| stage | latest manifest | duration_seconds | key output |
| --- | --- | --- | --- |
| 01 ingest | `20260429T171441Z_01_ingest_cboe.json` | `2140.216455` | `4,347` raw zips processed |
| 02 option panel | `20260429T192424Z_02_build_option_panel.json` | `7570.951996` | `18,169,131` valid silver rows |
| 03 surfaces | `20260429T192805Z_03_build_surfaces.json` | `207.648632` | `4,347` gold surfaces |
| 04 features | `20260429T194344Z_04_build_features.json` | `926.397094` | `4,325` feature rows, split hash `ae04611a7db447a08e4f52ece8acdf2680f58b64fbdd245f63003d7d447de83a` |
| 05 tune ridge | `20260429T194353Z_05_tune_models.json` | `0.949510` | tuning manifest `ridge.json` |
| 05 tune elasticnet | `20260429T194621Z_05_tune_models.json` | `145.028282` | tuning manifest `elasticnet.json` |
| 05 tune har_factor | `20260429T194625Z_05_tune_models.json` | `1.106004` | tuning manifest `har_factor.json` |
| 05 tune lightgbm | `20260429T201744Z_05_tune_models.json` | `1876.229410` | tuning manifest `lightgbm.json` |
| 05 tune random_forest | `20260429T202218Z_05_tune_models.json` | `271.209178` | tuning manifest `random_forest.json` |
| 05 tune neural_surface | `20260429T202322Z_05_tune_models.json` | `61.668101` | tuning manifest `neural_surface.json` |
| 06 walk-forward | `20260429T225000Z_06_run_walkforward.json` | `8794.483038` | `173` clean splits, `3,633` target dates per model |
| 07 stats | `20260429T225057Z_07_run_stats.json` | `54.173952` | DM, SPA, and simplified Tmax MCS outputs |
| 08 hedging | `20260429T225436Z_08_run_hedging_eval.json` | `215.781369` | `25,431` hedging result rows |
| 09 report | `20260429T225541Z_09_make_report_artifacts.json` | `63.274548` | report bundle refreshed |

Stage 06 provenance also records:

| item | value |
| --- | --- |
| data manifest hash | `a6c0350098266843a5e786d321566200ea5c8aa1f6d68e51902e7edc94e33ba4` |
| split manifest hash | `ae04611a7db447a08e4f52ece8acdf2680f58b64fbdd245f63003d7d447de83a` |
| hardware platform | `Windows-11-10.0.26200-SP0` |
| CPU | `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel` |
| CPU count | `28` |
| CUDA available | `true` |
| GPU | `NVIDIA GeForce RTX 4070 SUPER` |
| GPU memory | `12,878,086,144` bytes |
| key packages | `polars 1.39.3`, `pyarrow 23.0.1`, `numpy 2.4.4`, `scikit-learn 1.8.0`, `optuna 4.8.0`, `lightgbm 4.6.0`, `torch 2.11.0+cu128`, `arch 8.0.0` |

## Data and split identity

| item | value |
| --- | --- |
| raw files processed | `4,347` |
| raw rows parsed | `2,450,845,548` |
| non-`^SPX` rows filtered at ingest | `2,427,018,441` |
| target-symbol rows written to bronze | `23,827,107` |
| total valid silver rows | `18,169,131` |
| silver rows per day | min `335`, median `2,726`, max `21,104` |
| valid silver rows per day | min `228`, median `1,894`, max `18,152` |
| gold observed cells total | `274,896` |
| observed cells per day | min `36`, median `64`, max `81` |
| completed cells per day | `81` |
| feature rows | `4,325` |
| feature columns | `906` |
| feature quote-date range | `2004-02-03` through `2021-04-08` |
| feature target-date range | `2004-02-04` through `2021-04-09` |
| total walk-forward splits | `175` |
| clean evaluation splits | `173` |
| first clean split | `split_0002` |
| max HPO validation date | `2006-10-02` |
| forecast rows per model | `294,273` cell rows, representing `3,633` target dates |
| evaluation quote-date range | `2006-10-03` through `2021-03-10` |
| evaluation target-date range | `2006-10-04` through `2021-03-11` |

Walk-forward geometry:

| item | value |
| --- | --- |
| train size | `504` trading days |
| validation size | `126` trading days |
| test size | `21` trading days |
| step size | `21` trading days |
| expanding train | `true` |
| first split test range | `2006-08-03` through `2006-08-31` |
| first clean split test range | `2006-10-03` through `2006-10-31` |
| last split test range | `2021-02-09` through `2021-03-10` |
| date universe hash | `831f1598ca7cba46c1e7e9cbda7d648cb7755eeba6300ff57ecebc60cdd17788` |
| feature dataset hash | `10c6e918a473fffa7bbb4bb0c1770fbc5e53263e342485f226f388e8dbcd5206` |

## HPO and training profiles

| item | value |
| --- | --- |
| HPO profile | `hpo_30_trials` |
| HPO trials requested | `30` per tuned model |
| tuning splits count | `3` |
| Optuna sampler | `tpe` |
| Optuna pruner | `median` |
| training profile | `train_30_epochs` |
| neural epochs | `30` |
| neural early stopping patience | `5` |
| neural min epochs before early stop | `10` |
| LightGBM early stopping rounds | `25` |

Saved tuning manifests:

| model | stage-05 best value | best params | trials completed | trials pruned |
| --- | --- | --- | --- | --- |
| `ridge` | `0.00011373380478431107` | `alpha=97.2956052631225` | `13` | `17` |
| `elasticnet` | `0.00010517596071241086` | `alpha=0.07318748070568248`, `l1_ratio=0.8650179160564369` | `12` | `18` |
| `har_factor` | `0.00010790596335050441` | `n_factors=9`, `alpha=75.38868435013475` | `14` | `16` |
| `lightgbm` | `0.00010957357286069775` | `n_estimators=400`, `learning_rate=0.011800277461226189`, `num_leaves=60`, `max_depth=3`, `min_child_samples=25`, `feature_fraction=0.996023148831143`, `lambda_l2=0.00010749481010319476`, `n_factors=9` | `15` | `15` |
| `random_forest` | `0.00011098236755063946` | `n_estimators=500`, `max_depth=11`, `min_samples_leaf=2` | `21` | `9` |
| `neural_surface` | `0.0006605830765860249` | `hidden_width=384`, `depth=2`, `dropout=0.10712118850586368`, `learning_rate=0.00422336970502679`, `weight_decay=5.138298522573402e-05`, `batch_size=128`, `calendar_penalty_weight=0.006503132708010469`, `convexity_penalty_weight=1.0627295606594362e-05`, `roughness_penalty_weight=0.00035852333027431416` | `7` | `23` |

Tuning-diagnostics notes:
- `ridge` still shows numerical instability during HPO: maximum selected metric
  `4.9862380461949144e+141`.
- `neural_surface` remains degenerate in tuning diagnostics: median best epoch
  `1.0`, median prediction/target ratio `0.026815021367272304`, and median
  share of predictions below `1e-6` equal to `0.8669410150891632`.
- `lightgbm` tuning took materially longer than the other non-neural models
  because its HPO profile trains factor-wise boosted models.

## Official report overview

Directly supported by the refreshed report artifacts:
- Best full-sample primary-loss model: `naive` (`0.000028`).
- Best primary tail-risk model by 95th percentile: `naive` (`0.000042`).
- Best hedging revaluation model: `naive` (`5.728994` mean absolute
  revaluation error).
- Best QLIKE model: `har_factor` (`0.023590`).
- Primary simplified Tmax set includes `har_factor` and `naive`.
- Secondary simplified Tmax set includes only `har_factor`.
- Interpolation sensitivity: mean RMSE difference `0.003354`, maximum absolute
  difference `2.567543`.

## Ranked performance

Primary observed-cell MSE:

| rank | model_name | mean_observed_mse_total_variance | std_observed_mse_total_variance | n_target_dates | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- |
| 1 | naive | 0.000028 | 0.000376 | 3633 | 0.000000 |
| 2 | har_factor | 0.000033 | 0.000291 | 3633 | -18.004611 |
| 3 | random_forest | 0.000056 | 0.000375 | 3633 | -101.681414 |
| 4 | lightgbm | 0.000223 | 0.001012 | 3633 | -698.150215 |
| 5 | elasticnet | 0.000669 | 0.012365 | 3633 | -2294.071104 |
| 6 | neural_surface | 0.001860 | 0.002538 | 3633 | -6558.243237 |
| 7 | ridge | 0.021571 | 1.200646 | 3633 | -77136.182330 |

Secondary observed-cell QLIKE:

| rank | model_name | mean_observed_qlike_total_variance | std_observed_qlike_total_variance | n_target_dates | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- |
| 1 | har_factor | 0.023590 | 0.051403 | 3633 | 99.480705 |
| 2 | random_forest | 0.028306 | 0.064004 | 3633 | 99.376877 |
| 3 | elasticnet | 0.033896 | 0.080998 | 3633 | 99.253828 |
| 4 | lightgbm | 0.060413 | 0.176669 | 3633 | 98.670091 |
| 5 | ridge | 0.646641 | 36.946620 | 3633 | 85.765155 |
| 6 | naive | 4.542662 | 173.909336 | 3633 | 0.000000 |
| 7 | neural_surface | 2410355.657234 | 1451771.766403 | 3633 | -53060322.490153 |

Conditional surface-revaluation hedging:

| rank | model_name | mean_abs_revaluation_error | mean_squared_revaluation_error | mean_abs_hedged_pnl | mean_squared_hedged_pnl | hedged_pnl_variance | n_trades | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | naive | 5.728994 | 102.158676 | 2.877204 | 20.091356 | 19.649644 | 3633 | 0.000000 |
| 2 | har_factor | 6.752145 | 130.711190 | 2.835992 | 19.876234 | 19.431263 | 3633 | -17.859160 |
| 3 | random_forest | 7.812179 | 170.700021 | 2.861378 | 19.815163 | 19.362006 | 3633 | -36.362137 |
| 4 | lightgbm | 10.335272 | 338.833157 | 2.933908 | 21.910841 | 21.480562 | 3633 | -80.402906 |
| 5 | ridge | 13.764123 | 2967.527885 | 2.998331 | 24.863429 | 24.400164 | 3633 | -140.253743 |
| 6 | elasticnet | 13.931918 | 1561.475057 | 2.937375 | 22.237059 | 21.793887 | 3633 | -143.182611 |
| 7 | neural_surface | 96.921383 | 12124.931261 | 3.242347 | 50.685226 | 50.159077 | 3633 | -1591.769586 |

## Statistical tests

Primary metric:
- Diebold-Mariano tests do not support any candidate beating `naive` on the
  primary metric; all candidate comparisons have one-sided p-values at or above
  `0.860298`.
- SPA versus `naive` has p-value `0.808000` and no superior models by mean.
- Simplified Tmax MCS at alpha `0.10` retains `har_factor` and `naive`.

Secondary metric:
- Diebold-Mariano p-values versus `naive` are `0.058619` for `har_factor`,
  `0.058811` for `random_forest`, `0.059038` for `elasticnet`, `0.060128`
  for `lightgbm`, `0.093267` for `ridge`, and `1.000000` for
  `neural_surface`.
- SPA versus `naive` has p-value `0.098000`; superior models by mean are
  `elasticnet`, `har_factor`, `lightgbm`, `random_forest`, and `ridge`.
- Simplified Tmax MCS at alpha `0.10` retains only `har_factor`.

## Slice and tail behavior

Primary slice leaders:
- `har_factor` is the best full-grid model at the `1d`, `7d`, and `14d`
  maturity slices, improving over `naive` by `44.870188%`, `35.310966%`, and
  `16.329552%`, respectively.
- `random_forest` is the only observed-cell primary maturity slice leader other
  than `naive`, with an `8.422732%` improvement at `1d`.
- `naive` leads most longer-maturity, observed-cell, moneyness, and stress-window
  primary slices.

QLIKE slice leaders:
- `har_factor` leads the full-grid QLIKE maturity slices from `1d` through
  `60d`, with the largest gain at `1d` (`99.729711%`).
- `har_factor` also leads observed-cell QLIKE at `1d`, `7d`, and `14d`.
- `elasticnet` is the QLIKE leader at the `+0.30` and `-0.30` moneyness slices,
  while `har_factor` leads most other QLIKE moneyness slices.

Primary tail risk:

| model_name | mean_loss | p95_loss | max_loss |
| --- | --- | --- | --- |
| naive | 0.000028 | 0.000042 | 0.011255 |
| har_factor | 0.000033 | 0.000101 | 0.010600 |
| random_forest | 0.000056 | 0.000154 | 0.010518 |
| lightgbm | 0.000223 | 0.000510 | 0.014632 |
| elasticnet | 0.000669 | 0.000130 | 0.473019 |
| neural_surface | 0.001860 | 0.005533 | 0.025196 |
| ridge | 0.021571 | 0.000164 | 72.336926 |

QLIKE tail risk:

| model_name | mean_loss | p95_loss | max_loss |
| --- | --- | --- | --- |
| har_factor | 0.023590 | 0.048978 | 1.671184 |
| naive | 4.542662 | 0.058161 | 9235.519517 |
| random_forest | 0.028306 | 0.061114 | 1.846141 |
| elasticnet | 0.033896 | 0.072634 | 1.931891 |
| ridge | 0.646641 | 0.074560 | 2226.961412 |
| lightgbm | 0.060413 | 0.183435 | 2.587808 |
| neural_surface | 2410355.657234 | 4989056.113311 | 12669558.704161 |

Day-level comparison counts versus `naive` on the primary metric:

| model | days better than `naive` | days worse than `naive` | mean diff vs `naive` | median diff vs `naive` |
| --- | --- | --- | --- | --- |
| `elasticnet` | `339` | `3294` | `0.000640709829941675` | `5.245837334822338e-06` |
| `har_factor` | `331` | `3302` | `5.028497817404498e-06` | `4.023889985390616e-06` |
| `lightgbm` | `136` | `3497` | `0.00019498598123148084` | `8.558200437727519e-06` |
| `neural_surface` | `2` | `3631` | `0.001831648069376993` | `0.0008983828720370775` |
| `random_forest` | `327` | `3306` | `2.839854490993681e-05` | `4.005071910459933e-06` |
| `ridge` | `397` | `3236` | `0.021543321031158386` | `3.6759008673584564e-06` |

Day-level comparison counts versus `naive` on QLIKE:

| model | days better than `naive` | days worse than `naive` | mean diff vs `naive` | median diff vs `naive` |
| --- | --- | --- | --- | --- |
| `elasticnet` | `1267` | `2366` | `-4.50876596067787` | `0.0041996031813356545` |
| `har_factor` | `1654` | `1979` | `-4.519072212418158` | `0.0009277323627895455` |
| `lightgbm` | `1103` | `2530` | `-4.482248731565233` | `0.0059567255308529965` |
| `neural_surface` | `0` | `3633` | `2410351.1145716277` | `1880039.258078793` |
| `random_forest` | `1483` | `2150` | `-4.514355627905028` | `0.0023560051607952118` |
| `ridge` | `1415` | `2218` | `-3.896021111638333` | `0.003076472434817497` |

## Arbitrage and interpolation diagnostics

Arbitrage diagnostic summary:

| model_name | mean_calendar_violation_count | mean_calendar_violation_magnitude | mean_convexity_violation_count | mean_convexity_violation_magnitude | n_surfaces |
| --- | --- | --- | --- | --- | --- |
| random_forest | 1.960363 | 0.000111 | 2.372419 | 0.444777 | 3633 |
| neural_surface | 6.150840 | 0.000187 | 0.069639 | 0.022733 | 3633 |
| lightgbm | 4.036334 | 0.000415 | 2.283237 | 0.101566 | 3633 |
| har_factor | 4.276356 | 0.000630 | 2.865400 | 0.431172 | 3633 |
| naive | 3.126342 | 0.003123 | 3.592898 | 1.546821 | 3633 |
| actual_surface | 3.003911 | 0.004233 | 3.692432 | 1.944201 | 4347 |
| elasticnet | 3.473438 | 0.019336 | 2.813928 | 4.347004 | 3633 |
| ridge | 5.652353 | 0.135481 | 3.705753 | 8.492121 | 3633 |

Interpolation sensitivity summary:

| mean_mean_abs_diff | mean_rmse_diff | max_max_abs_diff | n_quote_dates |
| --- | --- | --- | --- |
| 0.001070 | 0.003354 | 2.567543 | 4347 |

## Interpretation

The primary empirical headline remains the persistence result: under the
leak-safe `15:45 -> next observed session` SPX forecasting protocol, `naive`
has the best observed-cell MSE, the best 95th-percentile primary tail risk, and
the best conditional revaluation result.

The strongest learned model is `har_factor`. It is second on primary MSE, is
retained alongside `naive` in the primary simplified Tmax set, wins mean QLIKE,
wins the secondary simplified Tmax set, and ranks second in hedging. This should
not be written as broad daily dominance over `naive`: on QLIKE, `har_factor`
beats `naive` on `1,654` days and is worse on `1,979` days. Its mean QLIKE edge
is driven by materially better behavior on extreme `naive` QLIKE blowups,
especially short-maturity and high-moneyness slices.

`random_forest` is competitive among learned nonparametric models but does not
overturn the persistence benchmark on the primary metric or hedging. `lightgbm`
is weaker than `random_forest` and `har_factor` on the main ranking despite a
better HPO score than previous historical runs.

`ridge` remains unstable out of sample. It has low median daily differences but
catastrophic outliers: the refreshed primary max loss is `72.336926`, and HPO
diagnostics still show a maximum selected metric of `4.9862380461949144e+141`.

`neural_surface` remains materially underperforming as a forecasting model. The
arbitrage-aware penalties reduce convexity violations, but the model produces a
poor surface forecast: primary MSE `0.001860`, QLIKE `2410355.657234`, hedging
MAE `96.921383`, and tuning diagnostics show near-zero predictions on most
cells. It should be described as an underperforming arbitrage-aware neural model,
not as evidence of arbitrage-free forecasting.

## Paper-ready summary

In a leak-safe SPX implied-volatility-surface forecasting pipeline using `15:45`
Cboe option data, a no-change persistence benchmark (`naive`) delivered the best
out-of-sample performance on the primary observed-cell MSE loss, the strongest
primary-metric tail-risk profile, and the best hedging revaluation performance.
A HAR-style factor model (`har_factor`) provided the strongest learned-model
result, winning the secondary QLIKE metric and short-maturity slices while being
retained with `naive` in the primary simplified Tmax model confidence set.

## Exact source files behind this dossier

- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/index.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/ranked_loss_summary__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/ranked_loss_summary__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/ranked_hedging_summary.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/dm_results__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/dm_results__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/spa_result__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/spa_result__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/mcs_result__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/mcs_result__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/slice_leaders__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/slice_leaders__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/tail_risk_summary__observed_mse_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/tail_risk_summary__observed_qlike_total_variance.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/arbitrage_diagnostic_summary.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/interpolation_sensitivity_summary.md`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/details/daily_loss_frame.parquet`
- `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/details/interpolation_sensitivity.parquet`
- `data/manifests/tuning/hpo_30_trials/*.json`
- `data/manifests/tuning/hpo_30_trials/*__diagnostics.parquet`
- `data/manifests/runs/01_ingest_cboe/20260429T171441Z_01_ingest_cboe.json`
- `data/manifests/runs/02_build_option_panel/20260429T192424Z_02_build_option_panel.json`
- `data/manifests/runs/03_build_surfaces/20260429T192805Z_03_build_surfaces.json`
- `data/manifests/runs/04_build_features/20260429T194344Z_04_build_features.json`
- `data/manifests/runs/05_tune_models/20260429T194353Z_05_tune_models.json`
- `data/manifests/runs/05_tune_models/20260429T194621Z_05_tune_models.json`
- `data/manifests/runs/05_tune_models/20260429T194625Z_05_tune_models.json`
- `data/manifests/runs/05_tune_models/20260429T201744Z_05_tune_models.json`
- `data/manifests/runs/05_tune_models/20260429T202218Z_05_tune_models.json`
- `data/manifests/runs/05_tune_models/20260429T202322Z_05_tune_models.json`
- `data/manifests/runs/06_run_walkforward/20260429T225000Z_06_run_walkforward.json`
- `data/manifests/runs/07_run_stats/20260429T225057Z_07_run_stats.json`
- `data/manifests/runs/08_run_hedging_eval/20260429T225436Z_08_run_hedging_eval.json`
- `data/manifests/runs/09_make_report_artifacts/20260429T225541Z_09_make_report_artifacts.json`
