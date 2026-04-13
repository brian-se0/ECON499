# Report Artifacts

- Benchmark model: `naive`
- Official loss metrics: `observed_mse_total_variance`, `observed_qlike_total_variance`
- Primary loss metric: `observed_mse_total_variance`
- Best full-sample loss model: `ridge` (0.000018)
- Best primary tail-risk model by 95th percentile: `ridge` (0.000021)
- Best hedging revaluation model: `naive` (0.811883)
- Simplified Tmax-set included models: neural_surface, ridge
- Interpolation sensitivity summary: mean RMSE diff 0.004684, max abs diff 0.009312
- Best model by `observed_mse_total_variance`: `ridge` (0.000018)
- Best model by `observed_qlike_total_variance`: `neural_surface` (0.153613)

## Strongest Slice Gains

| slice_family | slice_label | evaluation_scope | best_model_name | best_metric_value | benchmark_model | benchmark_metric_value | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| maturity | 30d | full | neural_surface | 0.000030 | naive | 0.000034 | 10.302660 |
| maturity | 60d | full | neural_surface | 0.000011 | naive | 0.000011 | 0.905714 |
| maturity | 90d | full | naive | 0.000024 | naive | 0.000024 | 0.000000 |
| maturity | 30d | observed | neural_surface | 0.000037 | naive | 0.000042 | 10.943469 |
| maturity | 60d | observed | neural_surface | 0.000011 | naive | 0.000011 | 0.905714 |

## Tail Risk

| loss_metric | model_name | mean_loss | p90_loss | p95_loss | p99_loss | max_loss | p95_improvement_vs_benchmark_pct | max_improvement_vs_benchmark_pct | n_target_dates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_mse_total_variance | ridge | 0.000018 | 0.000021 | 0.000021 | 0.000021 | 0.000021 | 11.326325 | 11.326325 | 3 |
| observed_mse_total_variance | neural_surface | 0.000018 | 0.000021 | 0.000021 | 0.000021 | 0.000021 | 11.228318 | 11.228318 | 3 |
| observed_mse_total_variance | naive | 0.000019 | 0.000024 | 0.000024 | 0.000024 | 0.000024 | 0.000000 | 0.000000 | 3 |

## Worst Primary-Loss Days

| loss_metric | model_name | rank_within_model | quote_date | target_date | loss_value | benchmark_model | benchmark_loss_value | excess_loss_vs_benchmark | loss_ratio_vs_benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_mse_total_variance | naive | 1 | 2021-01-06 | 2021-01-07 | 0.000024 | naive | 0.000024 | 0.000000 | 1.000000 |
| observed_mse_total_variance | naive | 2 | 2021-01-05 | 2021-01-06 | 0.000018 | naive | 0.000018 | 0.000000 | 1.000000 |
| observed_mse_total_variance | naive | 3 | 2021-01-04 | 2021-01-05 | 0.000015 | naive | 0.000015 | 0.000000 | 1.000000 |
| observed_mse_total_variance | neural_surface | 1 | 2021-01-06 | 2021-01-07 | 0.000021 | naive | 0.000024 | -0.000003 | 0.887717 |
| observed_mse_total_variance | neural_surface | 2 | 2021-01-05 | 2021-01-06 | 0.000018 | naive | 0.000018 | 0.000000 | 1.002949 |
| observed_mse_total_variance | neural_surface | 3 | 2021-01-04 | 2021-01-05 | 0.000015 | naive | 0.000015 | 0.000000 | 1.002828 |
| observed_mse_total_variance | ridge | 1 | 2021-01-06 | 2021-01-07 | 0.000021 | naive | 0.000024 | -0.000003 | 0.886737 |
| observed_mse_total_variance | ridge | 2 | 2021-01-05 | 2021-01-06 | 0.000018 | naive | 0.000018 | 0.000000 | 1.001857 |
| observed_mse_total_variance | ridge | 3 | 2021-01-04 | 2021-01-05 | 0.000015 | naive | 0.000015 | 0.000000 | 1.001760 |
