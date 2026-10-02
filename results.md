# Results Dossier: `hpo_30_trials__train_30_epochs`

This file records the end-to-end run completed on 2026-10-01 on Windows/CUDA.
Every stage manifest records the same git commit. The raw zip
archive was read from `D:\Options Data`; raw files were not modified.

Naming note: in code and artifacts the no-change benchmark is `naive`.

## Canonical artifact locations

- Report overview: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/index.md`
- Report tables: `data/manifests/report_artifacts/hpo_30_trials__train_30_epochs/tables/`
- Forecasts: `data/gold/forecasts/hpo_30_trials__train_30_epochs/`
- Tuning manifests: `data/manifests/tuning/hpo_30_trials/`
- Stats: `data/manifests/stats/hpo_30_trials__train_30_epochs/`
- Revaluation ("hedging"): `data/manifests/hedging/hpo_30_trials__train_30_epochs/`
- Post-hoc: `data/manifests/stats_sensitivity/…`, `data/manifests/postmortem/…`
- Run manifests: `data/manifests/runs/`; stage logs: `data/run_logs/`

## Run identity and validation

| item | value |
| --- | --- |
| quality gate before the run | `ruff` passed; `pytest` 358 passed; `mypy src tests scripts` passed |
| runtime preflight | `windows_cuda`, CUDA available, LightGBM GPU mode |
| hardware | Windows 11 (26200), Intel i7-14700KF (28 threads), NVIDIA RTX 4070 SUPER |
| key packages | Python 3.13.5, polars 1.39.3, numpy 2.4.4, scikit-learn 1.8.0, lightgbm 4.6.0, torch 2.11.0+cu128 |
| driver | `scripts/run_pipeline_windows.ps1` (same stages and arguments as `make pipeline`; GNU Make is not installed on this PC) |

| stage | duration (min) | key output |
| --- | --- | --- |
| 01 ingest | 46.7 | 4,347 zips; 2,450,845,548 raw rows; 23,827,107 `^SPX` rows |
| 02 option panel | 214.3 | 18,181,369 valid silver rows |
| 03 surfaces | 4.4 | 4,347 gold surfaces; 274,180 observed cells |
| 04 features | 26.8 | 4,325 × 906 feature table; split hash `ec06f1ba…` |
| 05 tuning (6 models) | 11.2 | tuning manifests |
| 06 walk-forward | 178.7 | 173 clean splits, 3,633 target dates per model |
| 07 stats | 1.2 | DM, SPA, MCS |
| 08 revaluation | 7.0 | 3,633 trades per model |
| 09 report | 1.9 | report bundle |
| 10 stats sensitivity | 6.5 | post-hoc inference sensitivity |
| 11 neural post-mortem | 99.2 | residual-variant tuning and 6 seed runs |

## Data and split identity

| item | value |
| --- | --- |
| silver rows / valid rows | 23,827,107 / 18,181,369 |
| valid rows per day | min 228, median 1,901, max 18,152 |
| rows rejected as vendor placeholders | 2,058,233 IV sentinels (0.02 / 0.001); 1,360 bid/ask 998/999 |
| observed cells | 274,180 (per day min 36, median 64, max 81) |
| feature rows / columns | 4,325 / 906 |
| splits | 175 total, 173 clean, first clean `split_0002` |
| evaluation quote dates | 2006-10-03 through 2021-03-10 (3,633 target dates per model) |
| date universe hash | `831f1598…` |
| feature dataset hash | `b613b4cd…` |

## HPO

| model | best value | selected parameters | completed / pruned |
| --- | --- | --- | --- |
| `ridge` | 6.256e-06 | `alpha=17.04` | 15 / 15 |
| `elasticnet` | 3.734e-06 | `alpha=0.1170`, `l1_ratio=0.9446` | 18 / 12 |
| `har_factor` | 3.659e-06 | `n_factors=12`, `alpha=0.1702` | 20 / 10 |
| `lightgbm` | 5.016e-06 | 300 trees, lr 0.0326, 45 leaves, depth 5, `min_child_samples=30`, `feature_fraction=0.852`, `lambda_l2=0.269`, 11 factors | 13 / 17 |
| `random_forest` | 7.286e-06 | 300 trees, depth 10, `min_samples_leaf=1` | 22 / 8 |
| `neural_surface` | 4.495e-06 | width 192, depth 5, dropout 0.201, lr 0.00155, wd 9.0e-6, batch 32, penalties (cal 0.0343, convex 1.0e-5, rough 6.6e-4) | 11 / 19 |

Neural tuning diagnostics (medians over completed tuning splits): best epoch 6,
prediction/target ratio 1.054, share of predictions below `1e-6` = 0.

## Ranked performance

Primary observed-cell MSE (target-day vega weights):

| rank | model | mean | median | p95 | max | days better than naive |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | naive | 1.8081e-05 | 7.9e-07 | 3.185e-05 | 0.010687 | — |
| 2 | har_factor | 1.8598e-05 | 3.19e-06 | 5.436e-05 | 0.008722 | 491 |
| 3 | random_forest | 4.4855e-05 | 4.15e-06 | 1.361e-04 | 0.009612 | 467 |
| 4 | neural_surface | 5.5722e-05 | 7.63e-06 | 2.248e-04 | 0.009267 | 188 |
| 5 | lightgbm | 2.4004e-04 | 7.10e-06 | 5.878e-04 | 0.018047 | 224 |
| 6 | elasticnet | 1.0587e-03 | 8.39e-06 | 1.555e-04 | 0.773014 | 345 |
| 7 | ridge | 2.0189e-03 | 3.78e-06 | 1.663e-04 | 4.802986 | 572 |

Secondary observed-cell QLIKE (unweighted):

| rank | model | mean | median | p95 | max | days better than naive |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | har_factor | 0.091229 | 0.042240 | 0.27588 | 17.90 | 1,714 |
| 2 | random_forest | 0.093885 | 0.042629 | 0.28293 | 22.00 | 1,696 |
| 3 | elasticnet | 0.094943 | 0.045401 | 0.26909 | 22.71 | 1,622 |
| 4 | lightgbm | 0.146282 | 0.054427 | 0.48184 | 28.99 | 1,494 |
| 5 | naive | 0.152937 | 0.035722 | 0.56062 | 53.95 | — |
| 6 | ridge | 8.661 | 0.041846 | 0.29916 | 30,440.6 | 1,742 |
| 7 | neural_surface | 805.83 | 0.14343 | 121.56 | 138,235 | 779 |

Stylized surface revaluation (persisted as "hedging"):

| rank | model | mean abs revaluation error | mean sq revaluation error | mean abs hedged PnL | mean sq hedged PnL |
| --- | --- | --- | --- | --- | --- |
| 1 | naive | 5.0834 | 89.09 | 1.7840 | 10.401 |
| 2 | har_factor | 5.5333 | 106.74 | 1.7530 | 10.004 |
| 3 | random_forest | 7.3596 | 150.66 | 1.7681 | 10.101 |
| 4 | lightgbm | 10.3081 | 341.84 | 1.8856 | 14.066 |
| 5 | ridge | 10.6104 | 1,068.68 | 1.8906 | 13.277 |
| 6 | elasticnet | 13.1061 | 1,337.84 | 1.8277 | 11.263 |
| 7 | neural_surface | 15.8340 | 596.56 | 2.0364 | 20.695 |

## Statistical tests

Primary metric (benchmark `naive`, one-sided "challenger better"):

- Diebold–Mariano p-values: `har_factor` 0.568 (statistic −0.17), `ridge` 0.927,
  `elasticnet` 0.998, `random_forest`, `lightgbm`, `neural_surface` ≈ 1.
- SPA (consistent, bootstrap long-run scale): p = 1.0. Every challenger loses on mean.
- MCS (`T_R`, alpha 0.10): {`elasticnet`, `har_factor`, `naive`, `ridge`}; final step
  p = 0.262. Ridge and elastic net survive because their catastrophic outlier days make
  their loss differentials too noisy to reject, not because they forecast well.

Secondary metric:

- Diebold–Mariano p-values versus `naive`: `har_factor`, `random_forest`,
  `elasticnet` all < 1e-7 (statistics 5.59–5.67); `lightgbm` 0.245; `ridge` 0.845;
  `neural_surface` 1.0.
- SPA: p = 0.000; superior by mean: `elasticnet`, `har_factor`, `lightgbm`, `random_forest`.
- MCS (`T_R`): {`elasticnet`, `har_factor`, `random_forest`, `ridge`}; `naive` eliminated.

Post-hoc sensitivity (stage 10, 10,000 bootstrap repetitions):

- SPA p-values: primary 1.0 at block lengths 5, 10, 20; secondary 0.0000, 0.0001, 0.0000.
- MCS sets: primary {`elasticnet`, `har_factor`, `naive`, `ridge`} at all three block
  lengths; secondary {`elasticnet`, `har_factor`, `random_forest`, `ridge`} at 5 and 10,
  plus `neural_surface` at 20.
- DM versus `naive` at Newey–West lags 0/5/10: `har_factor` primary 0.568/0.571/0.569;
  secondary QLIKE p < 1e-4 for `har_factor`, `random_forest`, `elasticnet` at every lag.
- Weighting robustness (observed MSE): `naive` is first under the official target-day
  vega weights, but `har_factor` is first and `naive` second under both maturity-balanced
  (1.7533e-05 vs 1.8317e-05) and equal-cell (2.3266e-05 vs 2.8148e-05) weighting.

## Slice, cell and tail behavior

- Completed-grid maturity slices (uniform weights): `random_forest` leads at 1d
  (−48.4% loss versus naive), `har_factor` at 7d (−36.7%) and 14d (−21.1%); `naive`
  leads from 30d onward.
- Observed-cell maturity slices (day-normalized vega weights): `naive` leads every
  maturity except 730d (`har_factor`, −6.8%).
- Observed-cell moneyness: `har_factor` leads the downside wings (−0.30: −53.5%;
  −0.20: −5.7%); `naive` leads −0.10 through +0.30.
- Stress windows (observed): `har_factor` leads 2008–2009 (−21.7%); `naive` leads
  2018 and 2020.
- Cell winners under the primary criterion: `naive` 74 cells, `har_factor` 6, `random_forest` 1.
- QLIKE slices: `har_factor` leads the 7d–60d observed and 7d–90d completed-grid maturity
  slices, `elasticnet` the 1d
  slice and several wing moneyness slices; `naive` leads 90d+ observed and the
  near-the-money observed moneyness slices.
- Primary tail risk: `naive` has the lowest p95 (3.185e-05); `har_factor` has a lower
  maximum (0.00872 vs 0.01069) and p99 (2.40e-04 vs 2.46e-04).

## Arbitrage and interpolation diagnostics

Evaluated on the 3,633 evaluation target dates for all rows:

| model | calendar count | calendar magnitude | convexity count | convexity magnitude |
| --- | --- | --- | --- | --- |
| neural_surface | 0.020 | 1.2e-06 | 1.066 | 0.932 |
| random_forest | 0.381 | 4.45e-05 | 0.594 | 0.102 |
| lightgbm | 0.225 | 5.76e-05 | 0.643 | 0.0034 |
| har_factor | 0.419 | 9.60e-05 | 0.969 | 0.119 |
| naive | 3.005 | 0.00410 | 2.311 | 0.416 |
| actual_surface | 3.006 | 0.00410 | 2.312 | 0.416 |
| elasticnet | 0.487 | 0.00542 | 0.674 | 2.003 |
| ridge | 1.577 | 0.0599 | 2.024 | 3.171 |

Interpolation-order sensitivity: mean RMSE difference 0.00161, maximum
absolute difference 2.628, over 4,347 dates.

## Neural post-mortem (stage 11)

Post-hoc and outside the official universe. The persistence-anchored residual variant was tuned
under the official protocol (best value 1.585e-06; width 448, depth 2, dropout 0.215, lr 1.68e-4,
batch 64) and both neural variants were run through the 173 clean splits at seeds 7/17/27.

| run | mean observed MSE | mean observed QLIKE | pred/target | share < 1e-6 | median best epoch |
| --- | --- | --- | --- | --- | --- |
| naive (reference) | 1.8081e-05 | 0.152937 | 0.9998 | 0 | n/a |
| neural_surface seed 7 | 5.5722e-05 | 805.83 | 0.974 | 0.0108 | 5 |
| neural_surface seed 17 | 5.9366e-05 | 957.96 | 0.969 | 0.0102 | 5 |
| neural_surface seed 27 | 5.1422e-05 | 1,268.60 | 0.974 | 0.0123 | 5 |
| residual seed 7 | 1.7452e-05 | 0.154268 | 0.991 | 0 | 1 |
| residual seed 17 | 1.8006e-05 | 0.154388 | 0.990 | 0 | 1 |
| residual seed 27 | 1.8265e-05 | 0.154298 | 0.988 | 0 | 1 |

The level model is stable across seeds apart from its crisis-period floor cells. The
residual variant has no floor predictions and essentially matches persistence on both losses
(seed range brackets naive; three-seed mean MSE 1% lower, within seed dispersion; no test run).

## Interpretation

The persistence benchmark has the lowest mean primary loss, the lowest p95 primary
loss, and the best revaluation error, but its lead is narrow:

- `har_factor` is statistically indistinguishable from `naive` on the primary metric
  (2.9% higher mean loss, DM p = 0.57, retained in the MCS) and ranks first under both
  alternative observed-cell weightings: persistence and a HAR factor model are tied on
  the primary loss.
- On the secondary QLIKE metric the evidence reverses decisively: `har_factor`,
  `random_forest` and `elasticnet` beat `naive` with DM p < 1e-7, SPA p ≈ 0, and
  `naive` is eliminated from the MCS. The result is broad (HAR better on 1,714 of
  3,633 days) and robust to lag, block and repetition choices.
- The neural model ranks fourth on primary MSE (5.57e-05) and has the fewest
  calendar violations, but under the
  out-of-distribution inputs of late 2008 and March 2020 it forecasts the total-variance
  floor in some short-maturity cells (8.3% of 2008 cells below 1e-6), which drives its
  QLIKE mean.
- Ridge and elastic net remain unstable: median daily losses are competitive, but
  October 2008 produces single-day primary losses of 4.8 (ridge) and 0.77 (elastic net).
