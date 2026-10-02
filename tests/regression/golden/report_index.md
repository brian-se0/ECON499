# Report Artifacts

- Benchmark model: `naive`
- Official loss metrics: `observed_mse_total_variance`, `observed_qlike_total_variance`
- Primary loss metric: `observed_mse_total_variance`
- Best full-sample loss model: `ridge` (7.7046e-06)
- Best primary tail-risk model by 95th percentile: `ridge` (9.05914e-06)
- Best hedging revaluation model: `naive` (0.128137)
- Model confidence set (T_R) included models: neural_surface, ridge
- Interpolation sensitivity summary: mean RMSE diff 0.00384082, max abs diff 0.00931233
- Best model by `observed_mse_total_variance`: `ridge` (7.7046e-06)
- Best model by `observed_qlike_total_variance`: `ridge` (0.0532853)

## Strongest Slice Gains

| slice_family | slice_label | evaluation_scope | best_model_name | best_metric_value | benchmark_model | benchmark_metric_value | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| maturity | 30d | observed | neural_surface | 1.81645e-06 | naive | 5.30564e-06 | 65.7639 |
| maturity | 30d | full | neural_surface | 6.84482e-06 | naive | 9.61931e-06 | 28.8429 |
| moneyness | -0.10 | observed | ridge | 8.28688e-06 | naive | 1.03534e-05 | 19.9602 |
| moneyness | +0.10 | full | neural_surface | 6.84482e-06 | naive | 8.04841e-06 | 14.9544 |
| moneyness | +0.10 | observed | neural_surface | 6.84482e-06 | naive | 8.04841e-06 | 14.9544 |

## Tail Risk

| loss_metric | model_name | mean_loss | p90_loss | p95_loss | p99_loss | max_loss | p95_improvement_vs_benchmark_pct | max_improvement_vs_benchmark_pct | n_target_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_mse_total_variance | ridge | 7.7046e-06 | 9.05914e-06 | 9.05914e-06 | 9.05914e-06 | 9.05914e-06 | 20.6068 | 20.6068 | 3 |
| observed_mse_total_variance | neural_surface | 7.76252e-06 | 9.12723e-06 | 9.12723e-06 | 9.12723e-06 | 9.12723e-06 | 20.01 | 20.01 | 3 |
| observed_mse_total_variance | naive | 8.30435e-06 | 1.14105e-05 | 1.14105e-05 | 1.14105e-05 | 1.14105e-05 | 0 | 0 | 3 |

## Worst Primary-Loss Days

| loss_metric | model_name | rank_within_model | quote_date | target_date | loss_value | benchmark_model | benchmark_loss_value | excess_loss_vs_benchmark | loss_ratio_vs_benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_mse_total_variance | naive | 1 | 2021-01-06 | 2021-01-07 | 1.14105e-05 | naive | 1.14105e-05 | 0 | 1 |
| observed_mse_total_variance | naive | 2 | 2021-01-05 | 2021-01-06 | 7.46087e-06 | naive | 7.46087e-06 | 0 | 1 |
| observed_mse_total_variance | naive | 3 | 2021-01-04 | 2021-01-05 | 6.0417e-06 | naive | 6.0417e-06 | 0 | 1 |
| observed_mse_total_variance | neural_surface | 1 | 2021-01-06 | 2021-01-07 | 9.12723e-06 | naive | 1.14105e-05 | -2.28324e-06 | 0.7999 |
| observed_mse_total_variance | neural_surface | 2 | 2021-01-05 | 2021-01-06 | 7.82407e-06 | naive | 7.46087e-06 | 3.632e-07 | 1.04868 |
| observed_mse_total_variance | neural_surface | 3 | 2021-01-04 | 2021-01-05 | 6.33625e-06 | naive | 6.0417e-06 | 2.94547e-07 | 1.04875 |
| observed_mse_total_variance | ridge | 1 | 2021-01-06 | 2021-01-07 | 9.05914e-06 | naive | 1.14105e-05 | -2.35133e-06 | 0.793932 |
| observed_mse_total_variance | ridge | 2 | 2021-01-05 | 2021-01-06 | 7.76569e-06 | naive | 7.46087e-06 | 3.04824e-07 | 1.04086 |
| observed_mse_total_variance | ridge | 3 | 2021-01-04 | 2021-01-05 | 6.28897e-06 | naive | 6.0417e-06 | 2.47265e-07 | 1.04093 |
