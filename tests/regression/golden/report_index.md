# Report Artifacts

- Benchmark model: `no_change`
- Official loss metrics: `observed_mse_total_variance`, `observed_qlike_total_variance`
- Primary loss metric: `observed_mse_total_variance`
- Best full-sample loss model: `ridge` (0.000018)
- Best hedging revaluation model: `no_change` (0.810879)
- MCS included models: neural_surface, ridge
- Interpolation sensitivity summary: mean RMSE diff 0.004684, max abs diff 0.009312
- Best model by `observed_mse_total_variance`: `ridge` (0.000018)
- Best model by `observed_qlike_total_variance`: `neural_surface` (0.153613)

## Strongest Slice Gains

| slice_family | slice_label | evaluation_scope | best_model_name | best_metric_value | benchmark_model | benchmark_metric_value | improvement_vs_benchmark_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| maturity | 30d | full | neural_surface | 0.000030 | no_change | 0.000034 | 10.561000 |
| maturity | 60d | full | neural_surface | 0.000011 | no_change | 0.000011 | 0.905714 |
| maturity | 90d | full | no_change | 0.000024 | no_change | 0.000024 | 0.000000 |
| maturity | 30d | observed | neural_surface | 0.000037 | no_change | 0.000042 | 11.255844 |
| maturity | 60d | observed | neural_surface | 0.000011 | no_change | 0.000011 | 0.905714 |
